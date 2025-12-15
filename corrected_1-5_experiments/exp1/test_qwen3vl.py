#!/usr/bin/env python3
"""
Quick test to verify Qwen3-VL loads correctly after upgrade.
"""

import sys
from pathlib import Path

print("="*80)
print("QWEN3-VL UPGRADE VERIFICATION TEST")
print("="*80)
print()

print("1. Checking transformers version...")
try:
    import transformers
    version = transformers.__version__
    print(f"   ✅ transformers version: {version}")
    
    major, minor = map(int, version.split('.')[:2])
    if major < 4 or (major == 4 and minor < 57):
        print(f"   ⚠️  WARNING: transformers >= 4.57.0 recommended, you have {version}")
        print(f"   Run: pip install 'transformers>=4.57.0' --upgrade")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

print()
print("2. Testing model import...")
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print("   ✅ AutoModelForImageTextToText imported successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

print()
print("3. Loading Qwen3-VL model (this may take a minute)...")
model_name = "Qwen/Qwen3-VL-8B-Instruct"

try:
    print(f"   Model: {model_name}")
    print(f"   Loading on CPU for testing...")
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cpu"  # CPU for testing
    )
    
    print(f"   ✅ Model loaded successfully!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params / 1e9:.2f}B")
    print(f"   Trainable parameters: {trainable_params / 1e9:.2f}B")
    
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("4. Loading processor...")
try:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print(f"   ✅ Processor loaded successfully")
    
    # Check processor components
    if hasattr(processor, 'tokenizer'):
        print(f"   ✅ Tokenizer available")
    if hasattr(processor, 'image_processor'):
        print(f"   ✅ Image processor available")
        
except Exception as e:
    print(f"   ❌ Processor loading failed: {e}")
    sys.exit(1)

print()
print("5. Checking LoRA compatibility...")
try:
    from peft import LoraConfig, get_peft_model, TaskType
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM
    )
    
    lora_model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_model.parameters())
    
    print(f"   ✅ LoRA applied successfully")
    print(f"   Trainable params: {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.2f}% of total)")
    
except Exception as e:
    print(f"   ❌ LoRA application failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print()
print("Qwen3-VL is ready to use. You can now:")
print("  1. Run resolution tests: sbatch exp1/slurm/test_resolutions.slurm")
print("  2. Start full training: sbatch exp1/slurm/train_exp1_category_based.slurm")
print()
print("Note: First training run will download the model (~16 GB) if not cached.")
print("="*80)







