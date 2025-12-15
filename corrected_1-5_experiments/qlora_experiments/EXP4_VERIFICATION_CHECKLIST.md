# Exp4 Curriculum Learning Fix - Verification Checklist

## ✅ Quick Verification (No Training Yet)

### 1. Verify Adapter State

Run the verification script:
```bash
python verify_exp4_fix.py models/exp4_curriculum/stage1
```

**Expected Output:**
- ✓ Only one adapter (e.g., 'default')
- ✓ Active adapter matches
- ✓ Trainable params: ~2.5M (small fraction, LoRA only)
- ✓ Base model is frozen (0 trainable params)
- ✓ LoRA config: r=4, alpha=8, target_modules=[q_proj, k_proj, v_proj, o_proj]

### 2. Check Training Script

**Manual Check:**
```bash
grep -n "merge_and_unload\|prepare_model_for_kbit_training\|get_peft_model" train_qlora_qwen3vl.py
```

**Expected:**
- ❌ NO `merge_and_unload()` in `prev_checkpoint` block
- ❌ NO `prepare_model_for_kbit_training()` in `prev_checkpoint` block  
- ✅ `is_trainable=True` in `PeftModel.from_pretrained()`
- ❌ NO `get_peft_model()` in `prev_checkpoint` block

**Current Status:** ✅ All checks pass (verified in code)

---

## 🔍 Code Review Checklist

### ✅ Base Model Loading
- [x] Base model loaded once with quantization
- [x] `prepare_model_for_kbit_training()` called once (outside prev_checkpoint block)
- [x] Quantization config consistent: `nf4`, `bfloat16`, `double_quant=True`

### ✅ Adapter Loading (Stage 2/3)
- [x] `PeftModel.from_pretrained(model, prev_checkpoint, is_trainable=True)` ✅
- [x] NO `merge_and_unload()` ✅
- [x] NO second `prepare_model_for_kbit_training()` ✅
- [x] NO `get_peft_model()` (doesn't create new adapter) ✅

### ✅ Adapter Configuration
- [x] Same LoRA config across all stages (r=4, alpha=8, same target_modules)
- [x] Single adapter name (default: 'default')
- [x] Adapter remains trainable across stages

---

## 🚂 Training Verification

### Stage 1 (Baseline)
```bash
python train_qlora_qwen3vl.py configs/exp4_stage1.yaml
```

**Check:**
- [ ] Training completes successfully
- [ ] Final checkpoint saved: `models/exp4_curriculum/stage1/`
- [ ] Verify adapter: `python verify_exp4_fix.py models/exp4_curriculum/stage1`
- [ ] Note final validation loss: `L1_final`

### Stage 2 (Continue from Stage 1)
```bash
python train_qlora_qwen3vl.py configs/exp4_stage2.yaml
```

**Critical Checks:**
- [ ] Script loads: `prev_checkpoint: models/exp4_curriculum/stage1`
- [ ] Log shows: "Continuing training from previous checkpoint"
- [ ] Log shows: "Loading existing LoRA adapter (will continue training the same adapter)"
- [ ] Log shows: "✓ Loaded previous adapter - continuing training on the same adapter"
- [ ] **First validation loss should be close to `L1_final`** (not a big jump)
- [ ] Loss curve continues smoothly (no reset spike)
- [ ] Only LoRA params trainable (check `print_trainable_parameters()`)

**Expected Behavior:**
- Loss starts near Stage 1's final loss
- Smooth continuation, not a restart
- Validation accuracy on Stage 1 questions should remain high

### Stage 3 (Continue from Stage 2)
```bash
python train_qlora_qwen3vl.py configs/exp4_stage3.yaml
```

**Same checks as Stage 2:**
- [ ] Smooth loss continuation from Stage 2
- [ ] No big jumps or resets
- [ ] Validation on all stages should remain good

---

## 🔧 Optimizer State Handling

### Current Implementation: Option B (Fresh Optimizer)

The current script uses **Option B** - fresh optimizer each stage. This is fine, but consider:

**Option A - Resume Optimizer (More Seamless):**
```python
# In TrainingArguments, add:
resume_from_checkpoint=prev_checkpoint if prev_checkpoint else None
```

**Option B - Fresh Optimizer (Current):**
- ✅ Currently implemented
- ⚠️ Consider reducing LR for later stages:
  - Stage 1: `lr: 5.0e-5`
  - Stage 2: `lr: 3.5e-5` (30% reduction)
  - Stage 3: `lr: 2.5e-5` (50% reduction from Stage 1)

**Recommendation:** Keep Option B but add LR decay in configs.

---

## 📊 Sanity Checks During Training

### Loss Curve Monitoring
- [ ] **Stage 1 → Stage 2 transition:** Loss should continue smoothly
  - If loss jumps up significantly, adapter wasn't loaded correctly
  - Expected: Loss starts within 5-10% of Stage 1's final loss

- [ ] **Stage 2 → Stage 3 transition:** Same smooth continuation

### Validation Monitoring
- [ ] Run same small dev set at end of each stage
- [ ] Stage 2's first eval should be close to Stage 1's last eval
- [ ] Stage 3's first eval should be close to Stage 2's last eval

### Parameter Check
```python
# Add this check in training script (optional):
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
# Expected: ~0.01-0.1% (LoRA params only)
```

---

## 🚨 Common Footguns to Avoid

### ✅ Adapter Naming
- [x] Single adapter name ('default') - PEFT handles this automatically
- [x] No adapter switching between stages

### ✅ LoRA Configuration
- [x] Identical config across stages:
  - `r: 4` ✅
  - `alpha: 8` ✅
  - `target_modules: [q_proj, k_proj, v_proj, o_proj]` ✅
  - `dropout: 0.05` ✅

### ✅ Precision
- [x] `bf16` compute dtype consistent
- [x] No re-quantization between stages
- [x] Base model quantized once at start

### ✅ Save Format
- [x] Using `trainer.save_model()` which saves adapter only
- [x] No base model saving between stages

---

## 📈 Post-Fix Expectations

### Expected Results After Retraining

**With Instructions:**
- Exp4 accuracy: **~92% ± 0.5%** (should match Exp1-3)
- Exp4 precision: **~89-91%**
- Exp4 recall: **~90-92%**
- Exp4 F1: **~90-91%**

**By Stage:**
- Stage 1: ~92% accuracy
- Stage 2: ~92-93% accuracy  
- Stage 3: ~83-92% accuracy (small sample size)

**By Question Type:**
- Single choice: ~92% accuracy
- Numeric: ~94% accuracy
- Multi-label: ~91% accuracy

### If Exp4 is Still Low

**Check these:**

1. **Learning Rate Too High**
   - Reduce LR by 30-50% for Stage 2/3
   - Current: `5.0e-5` → Try: `3.5e-5` (Stage 2), `2.5e-5` (Stage 3)

2. **Stage Data Order**
   - Harder data first can destabilize
   - Current order (Stage 1 → 2 → 3) should be fine

3. **Adapter Mixing**
   - Verify only one adapter path is used
   - Check logs for adapter loading messages

4. **Validation Loss Jumps**
   - If Stage 2 starts with much higher loss, adapter wasn't loaded correctly
   - Re-run verification script

---

## 🔬 Verification Commands

### Quick Adapter Check
```bash
# Check Stage 1 adapter
python verify_exp4_fix.py models/exp4_curriculum/stage1

# Check Stage 2 adapter (after training)
python verify_exp4_fix.py models/exp4_curriculum/stage2

# Check Stage 3 adapter (after training)
python verify_exp4_fix.py models/exp4_curriculum/stage3
```

### Training Script Check
```bash
# Verify no merge/re-quantize in prev_checkpoint block
grep -A 20 "if prev_checkpoint:" train_qlora_qwen3vl.py | grep -E "merge|prepare_model_for_kbit|get_peft_model"
# Should return nothing (or only in comments)
```

### Adapter Count Check
```python
# Quick Python check
from peft import PeftModel
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", 
                         bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", 
                                                    quantization_config=bnb)
m = PeftModel.from_pretrained(base, "models/exp4_curriculum/stage1", is_trainable=True)
print("Adapters:", list(m.peft_config.keys()))  # Should be ['default']
print("Active:", m.active_adapter)  # Should be 'default'
```

---

## ✅ Pre-Training Checklist

Before starting Stage 2/3 training, verify:

- [ ] Stage 1 training completed successfully
- [ ] Stage 1 adapter saved: `models/exp4_curriculum/stage1/adapter_model.safetensors`
- [ ] Verification script passes for Stage 1
- [ ] Training script has no merge/re-quantize in prev_checkpoint block
- [ ] Config files have correct `prev_checkpoint` paths
- [ ] Learning rates are set appropriately (consider reducing for later stages)

---

## 📝 Training Log Monitoring

**Watch for these in training logs:**

✅ **Good Signs:**
```
*** CURRICULUM LEARNING MODE ***
Continuing training from previous checkpoint: models/exp4_curriculum/stage1
Loading existing LoRA adapter (will continue training the same adapter)...
✓ Loaded previous adapter - continuing training on the same adapter
✓ Adapter is trainable and ready for continued training
trainable params: 2,621,440 || all params: 8,000,000,000 || trainable%: 0.0328
```

❌ **Bad Signs:**
```
Merging adapter into base model...  # Should NOT appear
Re-preparing model for quantization...  # Should NOT appear
Creating new LoRA adapter...  # Should NOT appear in Stage 2/3
trainable params: 8,000,000,000  # Should be small, not all params
```

---

**Last Updated:** 2025-01-XX  
**Status:** ✅ Code verified, ready for retraining

