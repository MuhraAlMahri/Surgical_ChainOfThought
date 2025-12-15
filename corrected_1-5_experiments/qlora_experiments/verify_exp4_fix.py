#!/usr/bin/env python3
"""
Verification script for Exp4 curriculum learning fix.
Checks that adapters are loaded correctly and no merge/re-quantization happens.
"""

import os
import sys
import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    prepare_model_for_kbit_training
)


def verify_adapter_state(model_path: str, base_model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
    """
    Verify that an adapter checkpoint has the correct state:
    - Only one adapter
    - Adapter is trainable
    - Only LoRA params are trainable (not base model)
    """
    print("=" * 80)
    print(f"VERIFYING ADAPTER: {model_path}")
    print("=" * 80)
    
    # Load base model with quantization (once)
    print("\n1. Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    base = AutoModelForImageTextToText.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    print("✓ Base model loaded and quantized")
    
    # Load adapter
    print(f"\n2. Loading adapter from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Adapter path does not exist: {model_path}")
        return False
    
    # Check if it's a checkpoint or final model
    adapter_path = model_path
    if os.path.isdir(model_path):
        if os.path.exists(os.path.join(model_path, "adapter_model.safetensors")):
            print("  Using final adapter_model.safetensors")
        else:
            # Find best checkpoint
            checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoint_numbers = [int(c.split("-")[1]) for c in checkpoints]
                best_checkpoint = f"checkpoint-{max(checkpoint_numbers)}"
                adapter_path = os.path.join(model_path, best_checkpoint)
                print(f"  Using checkpoint: {best_checkpoint}")
    
    try:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        print("✓ Adapter loaded with is_trainable=True")
    except Exception as e:
        print(f"❌ ERROR loading adapter: {e}")
        return False
    
    # Check adapter configuration
    print("\n3. Checking adapter configuration...")
    adapters = list(model.peft_config.keys())
    print(f"   Adapters found: {adapters}")
    
    if len(adapters) != 1:
        print(f"❌ ERROR: Expected 1 adapter, found {len(adapters)}")
        return False
    else:
        print(f"✓ Only one adapter: {adapters[0]}")
    
    active_adapter = model.active_adapter
    print(f"   Active adapter: {active_adapter}")
    if active_adapter != adapters[0]:
        print(f"❌ WARNING: Active adapter ({active_adapter}) != found adapter ({adapters[0]})")
    else:
        print(f"✓ Active adapter matches")
    
    # Check trainable parameters
    print("\n4. Checking trainable parameters...")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = (trainable_params / total_params) * 100
    
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable %: {trainable_pct:.4f}%")
    
    # Expected: LoRA params should be a small fraction (typically 0.01-0.1% for r=4)
    if trainable_pct > 1.0:
        print(f"❌ WARNING: Trainable params > 1% - might have base model params trainable!")
        return False
    elif trainable_pct < 0.001:
        print(f"❌ WARNING: Trainable params < 0.001% - might be too small!")
        return False
    else:
        print(f"✓ Trainable params in expected range (LoRA only)")
    
    # Check that base model params are frozen
    print("\n5. Verifying base model is frozen...")
    base_trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    if base_trainable > 0:
        print(f"❌ ERROR: Base model has {base_trainable:,} trainable params (should be 0)")
        return False
    else:
        print(f"✓ Base model is frozen (0 trainable params)")
    
    # Check adapter config
    print("\n6. Checking adapter configuration details...")
    adapter_config = model.peft_config[active_adapter]
    print(f"   LoRA rank (r): {adapter_config.r}")
    print(f"   LoRA alpha: {adapter_config.lora_alpha}")
    print(f"   LoRA dropout: {adapter_config.lora_dropout}")
    print(f"   Target modules: {adapter_config.target_modules}")
    print(f"   Task type: {adapter_config.task_type}")
    
    print("\n" + "=" * 80)
    print("✓ VERIFICATION PASSED")
    print("=" * 80)
    return True


def verify_training_script():
    """Check the training script for common issues."""
    print("\n" + "=" * 80)
    print("VERIFYING TRAINING SCRIPT")
    print("=" * 80)
    
    script_path = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments/train_qlora_qwen3vl.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check 1: No merge_and_unload during training
    if "merge_and_unload()" in content:
        # Check if it's in a comment or actually used
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if "merge_and_unload()" in line and not line.strip().startswith('#'):
                # Check if it's in the prev_checkpoint block (shouldn't be)
                if "prev_checkpoint" in '\n'.join(lines[max(0, i-10):i]):
                    issues.append(f"Line {i}: merge_and_unload() found in prev_checkpoint block (should be removed)")
    
    # Check 2: No second prepare_model_for_kbit_training in prev_checkpoint block
    prev_checkpoint_block = False
    prepare_count = 0
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if "prev_checkpoint" in line:
            prev_checkpoint_block = True
        if prev_checkpoint_block and "prepare_model_for_kbit_training" in line:
            prepare_count += 1
        if prev_checkpoint_block and "else:" in line:
            prev_checkpoint_block = False
    
    if prepare_count > 0:
        issues.append(f"prepare_model_for_kbit_training() called {prepare_count} times in prev_checkpoint block (should be 0)")
    
    # Check 3: is_trainable=True is used
    if "is_trainable=True" not in content:
        issues.append("is_trainable=True not found in PeftModel.from_pretrained() call")
    
    # Check 4: No get_peft_model in prev_checkpoint block
    prev_checkpoint_block = False
    for i, line in enumerate(lines, 1):
        if "prev_checkpoint" in line:
            prev_checkpoint_block = True
        if prev_checkpoint_block and "get_peft_model" in line and not line.strip().startswith('#'):
            issues.append(f"Line {i}: get_peft_model() found in prev_checkpoint block (should not create new adapter)")
        if prev_checkpoint_block and "else:" in line:
            prev_checkpoint_block = False
    
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✓ No issues found in training script")
        print("  - No merge_and_unload() in prev_checkpoint block")
        print("  - No prepare_model_for_kbit_training() in prev_checkpoint block")
        print("  - is_trainable=True is used")
        print("  - No get_peft_model() in prev_checkpoint block")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_exp4_fix.py <adapter_path> [base_model]")
        print("\nExample:")
        print("  python verify_exp4_fix.py models/exp4_curriculum/stage1")
        print("  python verify_exp4_fix.py models/exp4_curriculum/stage2 Qwen/Qwen3-VL-8B-Instruct")
        sys.exit(1)
    
    adapter_path = sys.argv[1]
    base_model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-VL-8B-Instruct"
    
    # Verify training script first
    script_ok = verify_training_script()
    
    # Verify adapter state
    adapter_ok = verify_adapter_state(adapter_path, base_model)
    
    print("\n" + "=" * 80)
    if script_ok and adapter_ok:
        print("✓ ALL VERIFICATIONS PASSED")
        print("=" * 80)
        sys.exit(0)
    else:
        print("❌ VERIFICATION FAILED")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()

