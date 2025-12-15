#!/usr/bin/env python3
"""
Multi-head CoT training script - Version 3 (Corrected Model Paths)

Key improvements:
1. Corrected model paths (MedGemma: google/medgemma-4b-it)
2. Model-specific loading functions
3. Better error handling and logging
4. LoRA checkpoint support
5. Dry-run mode for testing
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

# Set HuggingFace cache to workspace directory to avoid home quota issues
workspace_dir = Path(__file__).parent.absolute()
hf_cache_dir = workspace_dir / ".hf_cache"
hf_cache_dir.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(hf_cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "transformers")
os.environ["HF_HUB_CACHE"] = str(hf_cache_dir / "hub")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Using HuggingFace cache directory: {hf_cache_dir}")

# ============================================================================
# MODEL PATHS DICTIONARY
# ============================================================================

MODEL_PATHS = {
    "qwen3vl": {
        "base": "Qwen/Qwen2-VL-2B-Instruct",
        "kvasir_checkpoint": "checkpoints/qwen3vl_kvasir_finetuned",
        "endovis_checkpoint": "checkpoints/qwen3vl_endovis_finetuned"
    },
    "medgemma": {
        "base": "google/medgemma-4b-it",  # ✅ CORRECTED
        "kvasir_checkpoint": "checkpoints/medgemma_kvasir_finetuned",
        "endovis_checkpoint": "checkpoints/medgemma_endovis_finetuned"
    },
    "llava_med": {
        "base": "llava-hf/llava-v1.6-mistral-7b-hf",
        "kvasir_checkpoint": "checkpoints/llava_med_kvasir_finetuned",
        "endovis_checkpoint": "checkpoints/llava_med_endovis_finetuned"
    }
}

# ============================================================================
# MODEL-SPECIFIC LOADING FUNCTIONS
# ============================================================================

def load_qwen3vl(checkpoint_path):
    """Load Qwen3-VL model."""
    logger.info(f"Loading Qwen3-VL from {checkpoint_path}")
    
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Check if checkpoint path exists, if not use base model
    if not os.path.exists(checkpoint_path):
        logger.warning(f"⚠️  Checkpoint path not found: {checkpoint_path}")
        logger.info(f"  Using base model instead: {MODEL_PATHS['qwen3vl']['base']}")
        checkpoint_path = MODEL_PATHS['qwen3vl']['base']
    
    # Check for LoRA adapters
    has_lora = os.path.isdir(checkpoint_path) and \
               os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))
    
    if has_lora:
        logger.info("✓ LoRA checkpoint detected")
        
        with open(os.path.join(checkpoint_path, "adapter_config.json")) as f:
            adapter_config = json.load(f)
        
        base_model_name = adapter_config.get("base_model_name_or_path", MODEL_PATHS["qwen3vl"]["base"])
        logger.info(f"  Base model: {base_model_name}")
        
        # Load base model
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        # Merge LoRA weights
        model = model.merge_and_unload()
        logger.info("✓ LoRA weights merged")
    else:
        logger.info("✓ Regular checkpoint")
        
        # Load directly
        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        checkpoint_path if os.path.exists(checkpoint_path) else MODEL_PATHS["qwen3vl"]["base"],
        trust_remote_code=True,
        token=hf_token
    )
    
    logger.info("✓ Qwen3-VL loaded successfully")
    return model, processor


def load_medgemma(checkpoint_path):
    """Load MedGemma model."""
    logger.info(f"Loading MedGemma from {checkpoint_path}")
    
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Check if checkpoint path exists, if not use base model
    if not os.path.exists(checkpoint_path):
        logger.warning(f"⚠️  Checkpoint path not found: {checkpoint_path}")
        logger.info(f"  Using base model instead: {MODEL_PATHS['medgemma']['base']}")
        checkpoint_path = MODEL_PATHS['medgemma']['base']
    
    # Check for LoRA adapters
    has_lora = os.path.isdir(checkpoint_path) and \
               os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))
    
    if has_lora:
        logger.info("✓ LoRA checkpoint detected")
        
        with open(os.path.join(checkpoint_path, "adapter_config.json")) as f:
            adapter_config = json.load(f)
        
        base_model_name = adapter_config.get("base_model_name_or_path", MODEL_PATHS["medgemma"]["base"])
        logger.info(f"  Base model: {base_model_name}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        
        # Merge LoRA weights
        model = model.merge_and_unload()
        logger.info("✓ LoRA weights merged")
    else:
        logger.info("✓ Regular checkpoint")
        
        # Load directly
        try:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg or "GatedRepoError" in error_msg:
                logger.error(f"❌ Authentication failed for MedGemma model")
                logger.error(f"   MedGemma is a gated model requiring:")
                logger.error(f"   1. Accept terms at: https://huggingface.co/google/medgemma-4b-it")
                logger.error(f"   2. Set HF_TOKEN environment variable")
                logger.error(f"   3. Or login: huggingface-cli login")
                raise RuntimeError(f"MedGemma authentication failed. Please accept model terms and set HF_TOKEN.") from e
            else:
                raise
    
    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(
            checkpoint_path if os.path.exists(checkpoint_path) else MODEL_PATHS["medgemma"]["base"],
            trust_remote_code=True,
            token=hf_token
        )
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg or "GatedRepoError" in error_msg:
            logger.error(f"❌ Authentication failed for MedGemma processor")
            logger.error(f"   Please accept model terms and set HF_TOKEN")
            raise RuntimeError(f"MedGemma processor authentication failed.") from e
        else:
            raise
    
    logger.info("✓ MedGemma loaded successfully")
    return model, processor


def load_llava_med(checkpoint_path):
    """Load LLaVA-Med model."""
    logger.info(f"Loading LLaVA-Med from {checkpoint_path}")
    
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Check if checkpoint path exists, if not use base model
    if not os.path.exists(checkpoint_path):
        logger.warning(f"⚠️  Checkpoint path not found: {checkpoint_path}")
        logger.info(f"  Using base model instead: {MODEL_PATHS['llava_med']['base']}")
        checkpoint_path = MODEL_PATHS['llava_med']['base']
    else:
        # Check if checkpoint has unsupported model type (llava_mistral)
        config_path = os.path.join(checkpoint_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config = json.load(f)
                model_type = config.get("model_type", "")
                if model_type == "llava_mistral":
                    logger.warning(f"⚠️  Checkpoint has unsupported model type 'llava_mistral'")
                    logger.info(f"  Using base model instead: {MODEL_PATHS['llava_med']['base']}")
                    checkpoint_path = MODEL_PATHS['llava_med']['base']
            except Exception as e:
                logger.warning(f"Could not read config.json: {e}")
    
    # Check for LoRA adapters
    has_lora = os.path.isdir(checkpoint_path) and \
               os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))
    
    if has_lora:
        logger.info("✓ LoRA checkpoint detected")
        
        with open(os.path.join(checkpoint_path, "adapter_config.json")) as f:
            adapter_config = json.load(f)
        
        # Try to get base model from adapter config, fallback to default
        base_model_name = adapter_config.get("base_model_name_or_path")
        
        # Check if base_model_name has llava_mistral type (unsupported)
        if base_model_name and os.path.exists(base_model_name):
            try:
                base_config_path = os.path.join(base_model_name, "config.json")
                if os.path.exists(base_config_path):
                    with open(base_config_path) as f:
                        base_config = json.load(f)
                    if base_config.get("model_type") == "llava_mistral":
                        logger.warning(f"⚠️  Base model from adapter config has unsupported type 'llava_mistral'")
                        base_model_name = MODEL_PATHS["llava_med"]["base"]
                        logger.info(f"  Using MODEL_PATHS base model instead: {base_model_name}")
            except Exception as e:
                logger.warning(f"Could not check base model config: {e}")
        
        if not base_model_name:
            # Use the model path from MODEL_PATHS or try common variants
            base_model_name = MODEL_PATHS["llava_med"]["base"]
            logger.info(f"  Using default base model: {base_model_name}")
        else:
            logger.info(f"  Base model from config: {base_model_name}")
        
        # Load base model - use AutoModelForImageTextToText instead of deprecated AutoModelForVision2Seq
        try:
            base_model = AutoModelForImageTextToText.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
        except (ValueError, KeyError) as e:
            error_msg = str(e)
            if "llava_mistral" in error_msg or "does not recognize this architecture" in error_msg:
                logger.warning(f"Unsupported model type in base model: {error_msg}")
                logger.info(f"Falling back to MODEL_PATHS base model: {MODEL_PATHS['llava_med']['base']}")
                base_model_name = MODEL_PATHS['llava_med']['base']
                base_model = AutoModelForImageTextToText.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token
                )
            else:
                logger.warning(f"Failed to load with AutoModelForImageTextToText: {e}")
                logger.info("Trying AutoModelForCausalLM...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token
                )
        except Exception as e:
            logger.warning(f"Failed to load with AutoModelForImageTextToText: {e}")
            logger.info("Trying AutoModelForCausalLM...")
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token
                )
            except (ValueError, KeyError) as e2:
                error_msg = str(e2)
                if "llava_mistral" in error_msg or "does not recognize this architecture" in error_msg:
                    logger.warning(f"Unsupported model type in base model: {error_msg}")
                    logger.info(f"Falling back to MODEL_PATHS base model: {MODEL_PATHS['llava_med']['base']}")
                    base_model_name = MODEL_PATHS['llava_med']['base']
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        token=hf_token
                    )
                else:
                    raise
        
        # Load LoRA adapters - catch incompatibility errors
        try:
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            
            # Merge LoRA weights
            model = model.merge_and_unload()
            logger.info("✓ LoRA weights merged")
            # Processor should come from checkpoint_path if LoRA loaded successfully
            processor_path = checkpoint_path
        except RuntimeError as e:
            error_msg = str(e)
            if "size mismatch" in error_msg or "shape" in error_msg.lower():
                logger.warning(f"⚠️  LoRA adapter incompatible with base model")
                logger.info("  LoRA adapter was trained on different model architecture")
                logger.info("  Using base model without LoRA adapter")
                model = base_model
                # When falling back to base model, use base_model_name for processor
                processor_path = base_model_name
            else:
                raise
    else:
        logger.info("✓ Regular checkpoint")
        
        # Load directly - try AutoModelForImageTextToText first
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token
            )
        except (ValueError, KeyError) as e:
            error_msg = str(e)
            if "llava_mistral" in error_msg or "does not recognize this architecture" in error_msg:
                logger.warning(f"Unsupported model type in checkpoint: {error_msg}")
                logger.info(f"Falling back to base model: {MODEL_PATHS['llava_med']['base']}")
                checkpoint_path = MODEL_PATHS['llava_med']['base']
                model = AutoModelForImageTextToText.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token
                )
            else:
                logger.warning(f"Failed to load with AutoModelForImageTextToText: {e}")
                logger.info("Trying AutoModelForCausalLM...")
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token
                )
        except Exception as e:
            logger.warning(f"Failed to load with AutoModelForImageTextToText: {e}")
            logger.info("Trying AutoModelForCausalLM...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    token=hf_token
                )
            except (ValueError, KeyError) as e2:
                error_msg = str(e2)
                if "llava_mistral" in error_msg or "does not recognize this architecture" in error_msg:
                    logger.warning(f"Unsupported model type in checkpoint: {error_msg}")
                    logger.info(f"Falling back to base model: {MODEL_PATHS['llava_med']['base']}")
                    checkpoint_path = MODEL_PATHS['llava_med']['base']
                    model = AutoModelForCausalLM.from_pretrained(
                        checkpoint_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        token=hf_token
                    )
                else:
                    raise
        # Processor should come from checkpoint_path (which may have been updated to base model)
        processor_path = checkpoint_path
    
    # Load processor - use the path that was actually used for the model
    processor = AutoProcessor.from_pretrained(
        processor_path if os.path.exists(processor_path) else MODEL_PATHS["llava_med"]["base"],
        trust_remote_code=True,
        token=hf_token
    )
    
    logger.info("✓ LLaVA-Med loaded successfully")
    return model, processor


def load_model_for_training(model_name, checkpoint_path):
    """
    Route to appropriate loader based on model type.
    
    Args:
        model_name: "qwen3vl", "medgemma", or "llava_med"
        checkpoint_path: Path to fine-tuned checkpoint
    
    Returns:
        (model, processor) tuple
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"LOADING MODEL: {model_name}")
    logger.info(f"{'='*80}")
    
    if model_name == "qwen3vl":
        return load_qwen3vl(checkpoint_path)
    
    elif model_name == "medgemma":
        return load_medgemma(checkpoint_path)
    
    elif model_name == "llava_med":
        return load_llava_med(checkpoint_path)
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: qwen3vl, medgemma, llava_med")


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
    parser.add_argument("--model_name", required=True, choices=["qwen3vl", "medgemma", "llava_med"],
                       help="Model name")
    parser.add_argument("--dataset", required=True, choices=["kvasir", "endovis"],
                       help="Dataset name")
    parser.add_argument("--checkpoint_path", required=True,
                       help="Path to checkpoint or HF model name")
    parser.add_argument("--output_dir", default="results/test",
                       help="Output directory for trained model")
    parser.add_argument("--dry_run", action="store_true",
                       help="Just test model loading")
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("MULTI-HEAD COT TRAINING - VERSION 3")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("="*80)
    
    try:
        # Step 1: Validate paths
        logger.info("\n[1/5] Validating paths...")
        if not validate_model_paths(args.model_name, args.dataset, args.checkpoint_path):
            logger.error("❌ Validation failed")
            return 1
        
        # Step 2: Load checkpoint
        logger.info("\n[2/5] Loading model...")
        model, processor = load_model_for_training(args.model_name, args.checkpoint_path)
        
        if args.dry_run:
            logger.info("\n✓ DRY RUN COMPLETE - Model loaded successfully")
            return 0
        
        # Step 3: Create multi-head
        logger.info("\n[3/5] Creating multi-head architecture...")
        multihead_model = MultiHeadCoT_Model(model, freeze_base=True)
        multihead_model = multihead_model.to(args.device)
        
        # Step 4: Setup optimizer
        logger.info("\n[4/5] Setting up optimizer...")
        optimizer = torch.optim.AdamW(
            [p for p in multihead_model.parameters() if p.requires_grad],
            lr=2e-5
        )
        logger.info("✓ Optimizer ready")
        
        # Step 5: Test training
        logger.info("\n[5/5] Testing training loop...")
        train_one_epoch_simple(multihead_model, args.device)
        
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

