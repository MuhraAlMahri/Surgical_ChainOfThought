#!/usr/bin/env python3
"""
Multi-head CoT training script - Version 2 (Debuggable)

Key improvements:
1. Better error handling and logging
2. LoRA checkpoint support
3. Model path validation
4. Dry-run mode for testing
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
import os
import sys
import traceback
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL LOADING (with LoRA support)
# ============================================================================

def load_model_with_lora(checkpoint_path, model_name, model_type):
    """
    Load model that might have LoRA adapters.
    
    Handles:
    1. Pure fine-tuned checkpoints
    2. LoRA adapter checkpoints
    3. Base models (HuggingFace names)
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"LOADING MODEL: {model_name} ({model_type})")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"{'='*80}")
    
    # Get HuggingFace token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Check if checkpoint exists (if it's a path)
    if not checkpoint_path.startswith(("http://", "https://")) and not os.path.exists(checkpoint_path):
        # Might be a HuggingFace model name
        if "/" in checkpoint_path:
            logger.info(f"Treating as HuggingFace model name: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Check if it's a LoRA checkpoint (directory with adapter_config.json)
    is_lora_checkpoint = False
    if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            is_lora_checkpoint = True
    
    if is_lora_checkpoint:
        logger.info("✓ LoRA checkpoint detected")
        
        # Get base model name from adapter config
        try:
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")
            
            if not base_model_name:
                # Fallback to model-specific defaults
                if model_type == "llava_med":
                    base_model_name = "microsoft/llava-med-v1.5-mistral-7b"
                elif model_type == "qwen3vl":
                    base_model_name = "Qwen/Qwen3-VL-8B-Instruct"
                else:
                    raise ValueError(f"Cannot determine base model for {model_type}")
            
            logger.info(f"  Base model: {base_model_name}")
        except Exception as e:
            logger.warning(f"Could not read adapter config: {e}")
            # Use defaults
            if model_type == "llava_med":
                base_model_name = "microsoft/llava-med-v1.5-mistral-7b"
            elif model_type == "qwen3vl":
                base_model_name = "Qwen/Qwen3-VL-8B-Instruct"
            else:
                raise ValueError(f"Cannot determine base model for {model_type}")
        
        # Load base model
        logger.info("  Loading base model...")
        if model_type == "qwen3vl":
            base_model = AutoModelForImageTextToText.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
        elif model_type == "llava_med":
            from transformers import AutoModelForVision2Seq
            base_model = AutoModelForVision2Seq.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
        
        # Load LoRA adapters
        logger.info("  Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        # Merge LoRA weights into base model
        logger.info("  Merging LoRA weights...")
        model = model.merge_and_unload()
        
        logger.info("✓ Model loaded and LoRA weights merged")
    else:
        logger.info("✓ Regular checkpoint or HuggingFace model")
        
        # Load directly
        if model_type == "qwen3vl":
            model = AutoModelForImageTextToText.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
        elif model_type == "llava_med":
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
        
        logger.info("✓ Model loaded")
    
    return model


# ============================================================================
# MODEL PATH VALIDATION
# ============================================================================

def validate_model_paths(model_name, dataset, base_checkpoint):
    """Check if required paths exist before starting training."""
    
    logger.info(f"\nValidating paths for {model_name} + {dataset}...")
    
    # Check if base_checkpoint exists (if it's a local path)
    if not base_checkpoint.startswith(("http://", "https://")) and "/" in base_checkpoint:
        if os.path.exists(base_checkpoint):
            logger.info(f"✓ Checkpoint exists: {base_checkpoint}")
            return True
        else:
            logger.warning(f"⚠️  Checkpoint path not found: {base_checkpoint}")
            logger.info("  (Will try to load as HuggingFace model name)")
            return True  # Still proceed, might be HF model name
    
    logger.info(f"✓ Using HuggingFace model: {base_checkpoint}")
    return True


# ============================================================================
# MULTI-HEAD MODEL (Simplified)
# ============================================================================

class MultiHeadCoT_Model(nn.Module):
    """Simplified multi-head model for debugging."""
    
    def __init__(self, base_model, freeze_base=True):
        super().__init__()
        
        self.base_model = base_model
        
        # Freeze base
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            logger.info("✓ Base model frozen")
        
        # Get dimensions
        if hasattr(base_model.config, 'hidden_size'):
            hidden_dim = base_model.config.hidden_size
        elif hasattr(base_model.config, 'd_model'):
            hidden_dim = base_model.config.d_model
        else:
            hidden_dim = 4096  # Default fallback
            logger.warning(f"Could not determine hidden_dim, using default: {hidden_dim}")
        
        # Get vocab size from tokenizer or config
        vocab_size = getattr(base_model.config, 'vocab_size', None)
        if vocab_size is None:
            # Try to get from model
            if hasattr(base_model, 'lm_head'):
                vocab_size = base_model.lm_head.out_features
            else:
                vocab_size = 50000  # Default fallback
                logger.warning(f"Could not determine vocab_size, using default: {vocab_size}")
        
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Vocab size: {vocab_size}")
        
        # Three heads
        self.head_abnormality = nn.Linear(hidden_dim, vocab_size)
        self.head_characteristics = nn.Linear(hidden_dim, vocab_size)
        self.head_treatment = nn.Linear(hidden_dim, vocab_size)
        
        logger.info("✓ Multi-head architecture created")
        
        # Count trainable params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def forward(self, hidden_state, category):
        """Simplified forward pass."""
        
        if category == "abnormality_detection":
            return self.head_abnormality(hidden_state)
        elif category == "characteristics":
            return self.head_characteristics(hidden_state)
        else:
            return self.head_treatment(hidden_state)


# ============================================================================
# TRAINING FUNCTION (Simplified for Debugging)
# ============================================================================

def train_one_epoch_simple(model, device):
    """
    Simplified training loop for debugging.
    Just verifies the pipeline works.
    """
    
    model.train()
    
    logger.info("\nTraining epoch...")
    logger.info("  Processing 10 samples (debug mode)...")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5
    )
    criterion = nn.CrossEntropyLoss()
    
    for i in range(10):  # Just 10 samples for testing
        # Dummy forward pass
        dummy_hidden = torch.randn(1, model.head_abnormality.in_features).to(device)
        
        # Test each head
        for category in ["abnormality_detection", "characteristics", "treatment"]:
            output = model(dummy_hidden, category)
            
            # Dummy loss
            dummy_target = torch.randint(0, output.size(-1), (1,)).long().to(device)
            loss = criterion(output, dummy_target)
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 5 == 0:
            logger.info(f"    Batch {i+1}/10 - OK")
    
    logger.info("✓ Epoch complete")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["qwen3vl", "medgemma", "llava_med"])
    parser.add_argument("--dataset", required=True, choices=["kvasir", "endovis"])
    parser.add_argument("--base_checkpoint", required=True, help="Path to checkpoint or HF model name")
    parser.add_argument("--dry_run", action="store_true", help="Just test model loading")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("MULTI-HEAD COT TRAINING - DEBUG VERSION")
    logger.info("="*80)
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Base checkpoint: {args.base_checkpoint}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("="*80)
    
    try:
        # Step 1: Validate paths
        logger.info("\n[1/5] Validating paths...")
        if not validate_model_paths(args.model_type, args.dataset, args.base_checkpoint):
            logger.error("❌ Validation failed")
            return 1
        
        # Step 2: Load checkpoint
        logger.info("\n[2/5] Loading model...")
        base_model = load_model_with_lora(args.base_checkpoint, args.model_type, args.model_type)
        
        if args.dry_run:
            logger.info("\n✓ DRY RUN COMPLETE - Model loaded successfully")
            return 0
        
        # Step 3: Create multi-head
        logger.info("\n[3/5] Creating multi-head architecture...")
        model = MultiHeadCoT_Model(base_model, freeze_base=True)
        model = model.to(args.device)
        
        # Step 4: Setup optimizer
        logger.info("\n[4/5] Setting up optimizer...")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=2e-5
        )
        logger.info("✓ Optimizer ready")
        
        # Step 5: Test training
        logger.info("\n[5/5] Testing training loop...")
        train_one_epoch_simple(model, args.device)
        
        logger.info("\n" + "="*80)
        logger.info("SUCCESS - All components working!")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("ERROR OCCURRED")
        logger.error("="*80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("\nFull traceback:")
        traceback.print_exc(file=sys.stderr)
        logger.error("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())





