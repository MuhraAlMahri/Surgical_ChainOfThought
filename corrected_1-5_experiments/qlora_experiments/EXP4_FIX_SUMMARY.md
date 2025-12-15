# Experiment 4 Curriculum Learning Fix

## Problem Identified

The original Exp4 implementation had a critical bug in curriculum learning:

**Original (WRONG) Approach:**
1. Stage 1: Train LoRA adapter A → Save
2. Stage 2: 
   - Load adapter A
   - **Merge A into base model** ❌
   - **Re-quantize the merged model** ❌
   - **Create NEW LoRA adapter B** ❌
3. Stage 3: Same mistake - creates adapter C

**Result:** Each stage creates a fresh adapter, losing the learned knowledge from previous stages. This caused Exp4 to achieve only **62.88%** accuracy vs **92%+** for Exp1-3.

## Root Cause

The bug was in `train_qlora_qwen3vl.py` lines 240-265:
- Called `model.merge_and_unload()` which merges adapter into base
- Called `prepare_model_for_kbit_training()` again (re-quantization)
- Called `get_peft_model()` which creates a **new** adapter

This defeats the purpose of curriculum learning!

## Fixed Implementation

**New (CORRECT) Approach:**
1. Stage 1: Train LoRA adapter A → Save
2. Stage 2:
   - Load base model (quantized once)
   - **Load adapter A with `is_trainable=True`** ✅
   - **Continue training the SAME adapter A** ✅
   - Save updated adapter A
3. Stage 3: Continue training adapter A further

**Key Changes:**
- ✅ Load adapter with `PeftModel.from_pretrained(model, prev_checkpoint, is_trainable=True)`
- ✅ **Never merge** during training
- ✅ **Never re-quantize** during training
- ✅ **Never create new adapter** - continue the existing one
- ✅ Merge only at inference time (in evaluation script)

## Code Changes

**File:** `train_qlora_qwen3vl.py`

**Before:**
```python
if prev_checkpoint:
    model = PeftModel.from_pretrained(model, prev_checkpoint)
    model = model.merge_and_unload()  # ❌ WRONG
    model = prepare_model_for_kbit_training(model)  # ❌ WRONG
    # ... then create new adapter ❌
```

**After:**
```python
if prev_checkpoint:
    model = PeftModel.from_pretrained(model, prev_checkpoint, is_trainable=True)  # ✅ CORRECT
    # Continue training the same adapter - no merge, no re-quantize, no new adapter
```

## Expected Results After Fix

With the corrected implementation, Exp4 should:
- **Retain knowledge** from Stage 1 when training Stage 2
- **Retain knowledge** from Stages 1-2 when training Stage 3
- Achieve **similar or better accuracy** than Exp1-3 (~92%+)
- Demonstrate the benefits of curriculum learning

## Training Workflow

1. **Stage 1:** Train on Initial Assessment questions
   ```bash
   python train_qlora_qwen3vl.py configs/exp4_stage1.yaml
   # Output: models/exp4_curriculum/stage1/
   ```

2. **Stage 2:** Continue training same adapter on Findings questions
   ```bash
   python train_qlora_qwen3vl.py configs/exp4_stage2.yaml
   # Loads: models/exp4_curriculum/stage1/
   # Output: models/exp4_curriculum/stage2/
   ```

3. **Stage 3:** Continue training same adapter on Clinical Context questions
   ```bash
   python train_qlora_qwen3vl.py configs/exp4_stage3.yaml
   # Loads: models/exp4_curriculum/stage2/
   # Output: models/exp4_curriculum/stage3/
   ```

4. **Evaluation:** Uses final adapter (merged for inference)
   ```bash
   python scripts/evaluation/evaluate_exp4.py \
     --model_path models/exp4_curriculum/stage3/ \
     --test_data datasets/qlora_experiments/exp1_random/test.jsonl \
     --output results/exp4_evaluation.json
   ```

## Key Principles

1. **Quantize once** - Base model is quantized at the start, never re-quantized
2. **One adapter** - Create adapter once, continue training it across all stages
3. **Never merge during training** - Merging is only for final inference
4. **Preserve adapter state** - Each stage builds on the previous adapter's weights

## Testing

To verify the fix works:
1. Train Stage 1 and check adapter is saved
2. Train Stage 2 and verify it loads Stage 1 adapter (check logs)
3. Compare Stage 2 adapter weights with Stage 1 - should be similar but updated
4. Evaluate final model - should achieve ~92%+ accuracy

---

**Date Fixed:** 2025-01-XX  
**Issue:** Exp4 curriculum learning creating new adapters instead of continuing  
**Solution:** Load adapter with `is_trainable=True` and continue training without merging

