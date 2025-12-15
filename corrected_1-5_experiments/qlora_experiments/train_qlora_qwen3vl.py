#!/usr/bin/env python3
"""
Universal QLoRA Training Script for Qwen3-VL-8B-Instruct
- 4-bit quantization
- r=4, alpha=8
- Attention modules only (q_proj, k_proj, v_proj, o_proj)
- 5 epochs
- Supports all 5 experiments + checkpoint loading for curriculum
"""
import os
import sys
import json
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from PIL import Image

# ============================================================================
# Dataset
# ============================================================================
class VQADataset(Dataset):
    def __init__(self, jsonl_path, image_root, processor, max_length=3072):
        self.data = []
        self.image_root = Path(image_root)
        self.processor = processor
        self.max_length = max_length
        
        with open(jsonl_path) as f:
            for line in f:
                self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data):,} samples from {jsonl_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        img_path = self.image_root / item['image_filename']
        image = Image.open(img_path).convert('RGB')
        
        # Build conversation
        # Use the full instruction that includes the question
        user_message = item['instruction']
        assistant_message = item['answer']
        
        # Qwen3-VL chat template format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_message}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": assistant_message}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Process (NO padding here - will pad dynamically per batch)
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=False,  # Don't pad to max_length
            max_length=self.max_length,
            truncation=True
        )
        
        # Extract tensors (remove batch dimension)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        pixel_values = inputs['pixel_values'].squeeze(0)
        image_grid_thw = inputs['image_grid_thw'].squeeze(0)
        
        # Labels for causal LM (mask prompt, keep answer)
        labels = input_ids.clone()
        
        # Find assistant response start (Qwen chat template format)
        # Template format: <|im_start|>assistant\n{answer}<|im_end|>
        # This masks the prompt so loss is only computed on the answer
        # Method: Search for assistant start token sequence directly in tokenized sequence
        # Token IDs for "<|im_start|>assistant\n" = [151644, 77091, 198]
        assistant_start_token_ids = [151644, 77091, 198]  # <|im_start|>assistant\n
        assistant_token_start = -1
        
        # Search for the assistant start sequence in input_ids
        input_ids_list = input_ids.tolist()
        for i in range(len(input_ids_list) - len(assistant_start_token_ids) + 1):
            if input_ids_list[i:i+len(assistant_start_token_ids)] == assistant_start_token_ids:
                # Answer starts right after the assistant start tokens
                assistant_token_start = i + len(assistant_start_token_ids)
                break
        
        # Fallback: Try alternative pattern "<|im_start|>assistant" (without newline)
        if assistant_token_start == -1:
            alt_pattern = [151644, 77091]  # <|im_start|>assistant
            for i in range(len(input_ids_list) - len(alt_pattern) + 1):
                if input_ids_list[i:i+len(alt_pattern)] == alt_pattern:
                    # Look for the next token after "assistant"
                    if i + len(alt_pattern) < len(input_ids_list):
                        assistant_token_start = i + len(alt_pattern) + 1  # Skip potential newline token
                    break
        
        if assistant_token_start > 0 and assistant_token_start < len(labels):
            labels[:assistant_token_start] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'labels': labels
        }

# ============================================================================
# Collate Function - DYNAMIC PADDING (batch-wise)
# ============================================================================
def collate_fn(batch):
    """
    Collate batch of samples with DYNAMIC PADDING.
    Each batch is padded only to the longest sequence in THAT batch,
    not to the global max_length. This dramatically speeds up training!
    """
    # Find max length in this batch
    max_len_in_batch = max(x['input_ids'].shape[0] for x in batch)
    
    # Pad each sample to batch max (not global max)
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for x in batch:
        seq_len = x['input_ids'].shape[0]
        pad_len = max_len_in_batch - seq_len
        
        if pad_len > 0:
            # Pad input_ids
            input_ids_padded = torch.cat([
                x['input_ids'],
                torch.full((pad_len,), 0, dtype=x['input_ids'].dtype)  # Pad token ID
            ])
            
            # Pad attention_mask
            attention_mask_padded = torch.cat([
                x['attention_mask'],
                torch.zeros(pad_len, dtype=x['attention_mask'].dtype)
            ])
            
            # Pad labels
            labels_padded = torch.cat([
                x['labels'],
                torch.full((pad_len,), -100, dtype=x['labels'].dtype)  # Ignore index
            ])
        else:
            input_ids_padded = x['input_ids']
            attention_mask_padded = x['attention_mask']
            labels_padded = x['labels']
        
        input_ids_list.append(input_ids_padded)
        attention_mask_list.append(attention_mask_padded)
        labels_list.append(labels_padded)
    
    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'image_grid_thw': torch.stack([x['image_grid_thw'] for x in batch]),
        'labels': torch.stack(labels_list),
    }

# ============================================================================
# Main Training
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python train_qlora_qwen3vl.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print(f"QLoRA Training: {config.get('experiment_name', 'Unknown')}")
    print("="*80)
    print(f"Config: {config_path}")
    print(f"Model: {config['model_name']}")
    print(f"Epochs: {config['train']['epochs']}")
    print(f"QLoRA rank: {config['lora']['r']}")
    print(f"Target modules: {config['lora']['target_modules']}")
    print("="*80)
    
    # Setup
    model_name = config['model_name']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with 4-bit quantization
    print("\nLoading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Check if loading from checkpoint (for curriculum learning)
    prev_checkpoint = config.get('prev_checkpoint', None)
    
    # Load base model with quantization (always the same)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    
    if prev_checkpoint:
        print(f"\n*** CURRICULUM LEARNING MODE ***")
        print(f"Continuing training from previous checkpoint: {prev_checkpoint}")
        print("Loading existing LoRA adapter (will continue training the same adapter)...")
        from peft import PeftModel
        # Load the previous adapter and continue training it (don't merge!)
        # is_trainable=True ensures the adapter remains trainable
        model = PeftModel.from_pretrained(model, prev_checkpoint, is_trainable=True)
        print("✓ Loaded previous adapter - continuing training on the same adapter")
        print("✓ Adapter is trainable and ready for continued training")
    else:
        print("\n*** NEW TRAINING (No previous checkpoint) ***")
        print("Creating new LoRA adapter...")
        # LoRA configuration
        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['alpha'],
            target_modules=config['lora']['target_modules'],
            lora_dropout=config['lora']['dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("✓ Created new LoRA adapter")
    
    model.print_trainable_parameters()
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = VQADataset(
        config['data']['train_jsonl'],
        config['data']['image_root'],
        processor,
        max_length=config['train']['max_seq_len']
    )
    
    val_dataset = VQADataset(
        config['data']['val_jsonl'],
        config['data']['image_root'],
        processor,
        max_length=config['train']['max_seq_len']
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['train']['epochs'],
        per_device_train_batch_size=config['train']['train_bs'],
        per_device_eval_batch_size=config['train']['eval_bs'],
        gradient_accumulation_steps=config['train']['grad_accum'],
        learning_rate=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
        warmup_ratio=config['train']['warmup_ratio'],
        bf16=config['train']['bf16'],
        logging_steps=config['train']['logging_steps'],
        save_steps=config['train']['save_steps'],
        eval_steps=config['train']['eval_steps'],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=config['train']['gradient_checkpointing'],
        dataloader_num_workers=config['train'].get('dataloader_num_workers', 16),  # OPTIMIZED: Use config value, default 16
        remove_unused_columns=False,
        report_to="none",
        seed=config['train']['seed'],
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Train
    print("\nStarting training...")
    print("="*80)
    
    # Check for resume from checkpoint
    resume_from_checkpoint = config.get('resume_from_checkpoint', None)
    if resume_from_checkpoint:
        if os.path.exists(resume_from_checkpoint):
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        else:
            print(f"Warning: Checkpoint path {resume_from_checkpoint} does not exist. Starting fresh training.")
            resume_from_checkpoint = None
    else:
        # Auto-detect latest checkpoint in output_dir
        if os.path.isdir(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                # Sort by checkpoint number and get the latest
                checkpoints.sort(key=lambda x: int(x.split('-')[1]))
                latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
                print(f"Auto-detected latest checkpoint: {latest_checkpoint}")
                resume_from_checkpoint = latest_checkpoint
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    print("="*80)
    print("✓ Training complete!")
    print(f"✓ Model saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()

