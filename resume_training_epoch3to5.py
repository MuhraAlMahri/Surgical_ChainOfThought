#!/usr/bin/env python3
"""
Resume training from epoch 3 checkpoint to complete 5 epochs total.

Loads:
- Epoch 3 checkpoint (model + optimizer state)
- Continues training for epochs 4 and 5
- Saves checkpoints for epoch 4 and 5
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor
import os
import sys
import traceback
import json
from pathlib import Path
import logging

# Set HuggingFace cache to workspace directory
workspace_dir = Path(__file__).parent.absolute()
hf_cache_dir = workspace_dir / ".hf_cache"
hf_cache_dir.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(hf_cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "transformers")
os.environ["HF_HUB_CACHE"] = str(hf_cache_dir / "hub")

# Import from existing training script
from train_multihead_cot import (
    load_question_categories,
    train_kvasir_epoch,
    train_endovis_epoch,
    create_multihead_model
)
from data.vqa_data_loader import create_data_loader
from data.temporal_linker import TemporalLinker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Using HuggingFace cache directory: {hf_cache_dir}")


def resume_training(args):
    """Resume training from checkpoint."""
    
    logger.info("="*80)
    logger.info("RESUMING TRAINING FROM EPOCH 3 → EPOCH 5")
    logger.info("="*80)
    
    # Load checkpoint
    logger.info(f"\n[1/5] Loading epoch 3 checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    start_epoch = checkpoint['epoch'] + 1  # Should be 4
    logger.info(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"  Resuming from epoch {start_epoch}")
    
    # Load question categories
    categories = {}
    if args.question_categories and Path(args.question_categories).exists():
        all_categories = load_question_categories(args.question_categories)
        categories = all_categories.get(args.dataset, {})
    else:
        logger.warning(f"Question categories file not found: {args.question_categories}, continuing without categories")
    
    # Load base model and create multi-head architecture
    logger.info("\n[2/5] Loading base model and creating multi-head architecture...")
    model = create_multihead_model(
        base_checkpoint=args.base_checkpoint,
        model_type=args.model_type,
        freeze_base=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Load trained weights from checkpoint
    # Handle DataParallel wrapper if present
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        # Checkpoint was saved with DataParallel
        if not any(k.startswith('module.') for k in model.state_dict().keys()):
            # Current model is not wrapped, need to remove 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # If current model is wrapped, keep 'module.' prefix
    else:
        # Checkpoint was saved without DataParallel
        if any(k.startswith('module.') for k in model.state_dict().keys()):
            # Current model is wrapped, need to add 'module.' prefix
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    logger.info("✓ Model restored from epoch 3")
    
    # Get processor
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if args.model_type == "qwen3vl":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True, token=hf_token)
    elif args.model_type == "medgemma":
        processor = AutoProcessor.from_pretrained("google/medgemma-4b-it", trust_remote_code=True, token=hf_token)
    else:  # llava_med
        processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", trust_remote_code=True, token=hf_token)
    
    # Setup optimizer
    logger.info("\n[3/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Restore optimizer state
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("✓ Optimizer state restored")
    else:
        logger.warning("⚠️  No optimizer state found in checkpoint, starting fresh")
    
    # Load dataset
    logger.info("\n[4/5] Loading training data...")
    train_loader = create_data_loader(
        data_file=args.data_path,
        image_base_path=args.image_base_path,
        batch_size=args.batch_size,
        shuffle=True,
        is_temporal=(args.dataset == "endovis")
    )
    logger.info(f"Dataset loaded: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    
    # Temporal linker for EndoVis
    temporal_linker = None
    if args.dataset == "endovis":
        temporal_linker = TemporalLinker(args.image_base_path)
    
    # Criterion
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training loop for epochs 4 and 5
    logger.info("\n[5/5] Continuing training...")
    logger.info("="*80)
    logger.info(f"Training epochs {start_epoch} to {args.total_epochs}")
    logger.info("="*80)
    
    for epoch in range(start_epoch, args.total_epochs + 1):  # Epochs 4 and 5
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.total_epochs}")
        logger.info(f"{'='*60}")
        
        # Train one epoch
        if args.dataset == "kvasir":
            train_kvasir_epoch(
                model, train_loader, optimizer, categories,
                processor, device, criterion,
                grad_accum=args.grad_accum,
                use_bf16=args.bf16
            )
        else:  # endovis
            train_endovis_epoch(
                model, train_loader, optimizer, categories,
                processor, device, criterion, temporal_linker,
                grad_accum=args.grad_accum,
                use_bf16=args.bf16
            )
        
        # Save checkpoint
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_path / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        logger.info(f"✓ Saved: {checkpoint_path}")
    
    # Save final model
    final_path = output_path / "multihead_cot_trained_5epochs.pt"
    torch.save(model.state_dict(), final_path)
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE - 5 EPOCHS DONE!")
    logger.info("="*80)
    logger.info(f"Final model: {final_path}")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Epoch 3 checkpoint to resume from")
    parser.add_argument("--base_checkpoint", required=True, help="Base model checkpoint")
    parser.add_argument("--model_type", required=True, choices=["qwen3vl", "medgemma", "llava_med"])
    parser.add_argument("--dataset", required=True, choices=["kvasir", "endovis"])
    parser.add_argument("--data_path", required=True, help="Path to training data JSON")
    parser.add_argument("--image_base_path", required=True, help="Base path for images")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--question_categories", default="question_categories.json")
    parser.add_argument("--total_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    try:
        resume_training(args)
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("ERROR OCCURRED")
        logger.error("="*80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("\nFull traceback:")
        traceback.print_exc(file=sys.stderr)
        logger.error("="*80)
        sys.exit(1)


