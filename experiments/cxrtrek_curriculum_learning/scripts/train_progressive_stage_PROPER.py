#!/usr/bin/env python3
"""
Progressive Stage Training with PROPER Image-Based Split (Zero Leakage)

This is the CORRECTED version that uses pre-split data files instead of 
randomly splitting QA pairs at runtime. This ensures zero image overlap
between train and test sets.

Key differences from original:
- Loads pre-split train/test JSON files
- No runtime splitting (already done properly by image ID)
- Zero data leakage

Author: Fixed version for scientific validity
Date: October 2025
"""

import os
import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StageDataset(Dataset):
    """Dataset for a single stage using PRE-SPLIT data files."""
    
    def __init__(self, data_path: str, image_dir: str, processor, split: str = 'train'):
        """
        Args:
            data_path: Path to pre-split JSON file (e.g., train_stage1.json)
            image_dir: Directory containing images
            processor: Qwen2VL processor
            split: 'train' or 'test' (for logging only)
        """
        self.image_dir = image_dir
        self.processor = processor
        self.split = split
        
        # Load pre-split data
        with open(data_path, 'r') as f:
            self.samples = json.load(f)
        
        logger.info(f"  {split.capitalize()}: {len(self.samples)} samples (from {data_path})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get image path
        image_path = sample['image_path']
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.image_dir, image_path)
        
        return {
            'image_path': image_path,
            'question': sample['question'],
            'answer': sample['answer'],
            'image_id': sample.get('image_id', '')
        }


def collate_fn(batch, processor, device):
    """Collate function for batching."""
    images = []
    questions = []
    answers = []
    
    for item in batch:
        try:
            img = Image.open(item['image_path']).convert('RGB')
            images.append(img)
            questions.append(item['question'])
            answers.append(item['answer'])
        except Exception as e:
            logger.warning(f"Error loading image {item['image_path']}: {e}")
            continue
    
    if not images:
        return None
    
    # Prepare inputs
    messages = []
    for q in questions:
        messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q}
                ]
            }
        ])
    
    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text,
        images=images,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Process answers
    answer_inputs = processor.tokenizer(
        answers,
        padding=True,
        return_tensors="pt"
    )
    labels = answer_inputs.input_ids.to(device)
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    inputs['labels'] = labels
    
    return inputs


def train_stage(stage_num: int,
               train_data_path: str,
               val_data_path: str,
               image_dir: str,
               output_dir: str,
               prev_checkpoint: str = None,
               epochs: int = 3,
               batch_size: int = 2,
               learning_rate: float = 1e-4,
               gradient_accumulation_steps: int = 4,
               device: str = "cuda"):
    """Train a single stage with proper data split."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting Stage {stage_num} Training (PROPER SPLIT)")
    logger.info(f"{'='*80}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    
    # Load or initialize model
    if prev_checkpoint and os.path.exists(prev_checkpoint):
        logger.info(f"Loading from previous checkpoint: {prev_checkpoint}")
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, prev_checkpoint)
    else:
        logger.info("Initializing from base model")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        
        # Add LoRA adapters
        lora_config = LoraConfig(
            r=256,
            lora_alpha=512,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    
    # Create datasets with pre-split files
    logger.info(f"\nLoading datasets:")
    train_dataset = StageDataset(train_data_path, image_dir, processor, split='train')
    val_dataset = StageDataset(val_data_path, image_dir, processor, split='test')
    
    # Create dataloaders (num_workers=0 to avoid CUDA multiprocessing issues)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, processor, device)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, processor, device)
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for i, batch in enumerate(pbar):
            if batch is None:
                continue
            
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * gradient_accumulation_steps
            train_steps += 1
            pbar.set_postfix({'loss': train_loss / train_steps})
        
        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if batch is None:
                    continue
                
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        
        logger.info(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint = os.path.join(output_dir, f"stage{stage_num}_best")
            model.save_pretrained(best_checkpoint)
            logger.info(f"Saved best checkpoint to {best_checkpoint}")
    
    # Save final checkpoint
    final_checkpoint = os.path.join(output_dir, f"stage{stage_num}_final")
    model.save_pretrained(final_checkpoint)
    logger.info(f"Saved final checkpoint to {final_checkpoint}")
    
    return best_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train progressive stages with proper image-based split")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3], help="Stage number")
    parser.add_argument("--train_data", type=str, required=True, help="Path to train JSON (pre-split)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation JSON (pre-split)")
    parser.add_argument("--images", type=str, required=True, help="Image directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--prev_checkpoint", type=str, default=None, help="Previous stage checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    train_stage(
        stage_num=args.stage,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        image_dir=args.images,
        output_dir=args.output,
        prev_checkpoint=args.prev_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        device=args.device
    )


if __name__ == "__main__":
    main()

