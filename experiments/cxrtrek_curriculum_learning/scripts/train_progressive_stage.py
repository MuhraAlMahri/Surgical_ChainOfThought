#!/usr/bin/env python3
"""
Progressive Stage-by-Stage Training (Curriculum Learning)

Trains a VQA model on one stage at a time, building knowledge progressively:
  Stage 1 → Stage 2 → Stage 3

Each stage loads the previous stage's checkpoint and continues training.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from tqdm import tqdm
import numpy as np


class StageDataset(Dataset):
    """Dataset for a single stage of clinical flow."""
    
    def __init__(self, data_path: str, image_dir: str, stage_num: int, processor, split: str = 'train'):
        """
        Args:
            data_path: Path to PROPER PRE-SPLIT JSON file (e.g., train_stage1.json or test_stage1.json)
            image_dir: Directory containing images
            stage_num: Which stage (1, 2, or 3) - used for logging only
            processor: Qwen2VL processor
            split: 'train' or 'val' - used for logging only
        """
        self.image_dir = image_dir
        self.stage_num = stage_num
        self.processor = processor
        self.split = split
        
        # Load PROPER PRE-SPLIT data (already split by image ID with zero overlap!)
        with open(data_path, 'r') as f:
            self.samples = json.load(f)
        
        # Update image paths to be absolute
        for sample in self.samples:
            image_path = sample['image_path']
            if not os.path.isabs(image_path):
                if image_path.startswith('images/'):
                    image_path = image_path[7:]
                sample['image_path'] = os.path.join(image_dir, image_path)
        
        print(f"  {split.capitalize()}: {len(self.samples)} samples (from proper image-based split)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'image_path': sample['image_path'],
            'question': sample['question'],
            'answer': sample['answer']
        }


def collate_fn(batch, processor):
    """Custom collate function that processes images in batch."""
    images = []
    questions = []
    answers = []
    
    for item in batch:
        image = Image.open(item['image_path']).convert('RGB')
        images.append(image)
        questions.append(item['question'])
        answers.append(item['answer'])
    
    # Prepare messages
    messages_batch = []
    for q, a in zip(questions, answers):
        messages_batch.append([{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": q}
            ]
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": a}]
        }])
    
    # Process
    texts = [processor.apply_chat_template(msg, tokenize=False) 
             for msg in messages_batch]
    
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )
    
    inputs['labels'] = inputs['input_ids'].clone()
    
    return inputs


def train_stage(
    stage_num: int,
    train_data_path: str,
    val_data_path: str,
    image_dir: str,
    output_dir: str,
    prev_checkpoint: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-6,
    device: str = "cuda"
):
    """
    Train a single stage with PROPER IMAGE-BASED SPLIT.
    
    Args:
        stage_num: Stage number (1, 2, or 3)
        train_data_path: Path to train split JSON (pre-split by image ID)
        val_data_path: Path to validation/test split JSON (pre-split by image ID)
        image_dir: Directory with images
        output_dir: Where to save checkpoint
        prev_checkpoint: Previous stage checkpoint (None for Stage 1)
        epochs: Number of epochs
        batch_size: Batch size
        gradient_accumulation_steps: Gradient accumulation
        learning_rate: Learning rate
        device: Device to use
    """
    print(f"\n{'='*60}")
    print(f"TRAINING STAGE {stage_num} (PROPER IMAGE-BASED SPLIT)")
    print(f"{'='*60}\n")
    
    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True
    )
    
    # Create datasets from PROPER PRE-SPLIT files
    print(f"\nCreating Stage {stage_num} datasets from proper split...")
    train_dataset = StageDataset(train_data_path, image_dir, stage_num, processor, split='train')
    val_dataset = StageDataset(val_data_path, image_dir, stage_num, processor, split='val')
    
    if len(train_dataset) == 0:
        print(f"ERROR: No training samples for Stage {stage_num}!")
        return
    
    # Create dataloaders
    from functools import partial
    collate_fn_with_processor = partial(collate_fn, processor=processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_processor,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_processor,
        num_workers=0
    )
    
    # Load model
    print(f"\nLoading model...")
    if prev_checkpoint:
        print(f"  Loading from previous stage: {prev_checkpoint}")
        # Load base model
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        # Load LoRA weights from previous stage
        model = PeftModel.from_pretrained(base_model, prev_checkpoint, is_trainable=True)
        print(f"  ✓ Loaded previous stage checkpoint")
        print(f"  ✓ Model set to trainable mode")
    else:
        print(f"  Loading base model: Qwen2-VL-2B-Instruct")
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        
        # Apply LoRA
        print("  Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=256,
            lora_alpha=512,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)
    
    model.train()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Setup scheduler
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        train_losses = []
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_losses.append(loss.item() * gradient_accumulation_steps)
            pbar.set_postfix({'loss': f'{train_losses[-1]:.4f}'})
        
        avg_train_loss = np.mean(train_losses)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                outputs = model(**batch)
                val_losses.append(outputs.loss.item())
        
        avg_val_loss = np.mean(val_losses)
        print(f"Average validation loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_dir = os.path.join(output_dir, f"stage{stage_num}_best")
            os.makedirs(best_checkpoint_dir, exist_ok=True)
            model.save_pretrained(best_checkpoint_dir)
            print(f"✓ Saved best model to: {best_checkpoint_dir}")
    
    # Save final checkpoint
    final_checkpoint_dir = os.path.join(output_dir, f"stage{stage_num}_final")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    model.save_pretrained(final_checkpoint_dir)
    print(f"✓ Saved final model to: {final_checkpoint_dir}")
    
    print(f"\n{'='*60}")
    print(f"STAGE {stage_num} TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_checkpoint_dir}")
    print(f"Final checkpoint: {final_checkpoint_dir}")
    print(f"{'='*60}\n")
    
    return best_checkpoint_dir


def main():
    parser = argparse.ArgumentParser(description='Progressive Stage Training (PROPER IMAGE-BASED SPLIT)')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                       help='Stage number to train (1, 2, or 3)')
    parser.add_argument('--train_data_path', required=True, help='Path to train split JSON (pre-split by image ID)')
    parser.add_argument('--val_data_path', required=True, help='Path to validation/test split JSON (pre-split by image ID)')
    parser.add_argument('--image_dir', required=True, help='Directory with images')
    parser.add_argument('--output_dir', required=True, help='Output directory for checkpoints')
    parser.add_argument('--prev_checkpoint', help='Previous stage checkpoint (for Stage 2 and 3)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # INFO: prev_checkpoint is now OPTIONAL
    # - For CURRICULUM LEARNING: pass prev_checkpoint to continue from previous stage
    # - For CXRTREK SEQUENTIAL: omit prev_checkpoint to train independently from base model
    if args.prev_checkpoint:
        print(f"INFO: Stage {args.stage} will continue from checkpoint: {args.prev_checkpoint}")
    else:
        print(f"INFO: Stage {args.stage} will train independently from base model (Qwen2-VL-2B-Instruct)")
    
    # Train stage
    train_stage(
        stage_num=args.stage,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        prev_checkpoint=args.prev_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        device=args.device
    )


if __name__ == "__main__":
    main()
