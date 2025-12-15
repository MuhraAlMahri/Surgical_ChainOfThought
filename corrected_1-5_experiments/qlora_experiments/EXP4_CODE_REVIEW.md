# Exp4 Curriculum Learning - Code Review & Verification

## ✅ Code Review Results

### Training Script Analysis (`train_qlora_qwen3vl.py`)

**Lines 222-260: Adapter Loading Logic**

```python
# ✅ CORRECT: Base model loaded once, quantized once
model = AutoModelForImageTextToText.from_pretrained(...)
model = prepare_model_for_kbit_training(model)  # Called ONCE, outside prev_checkpoint block

if prev_checkpoint:
    # ✅ CORRECT: Load adapter with is_trainable=True
    model = PeftModel.from_pretrained(model, prev_checkpoint, is_trainable=True)
    # ✅ CORRECT: No merge_and_unload()
    # ✅ CORRECT: No prepare_model_for_kbit_training()
    # ✅ CORRECT: No get_peft_model() (doesn't create new adapter)
else:
    # ✅ CORRECT: Only creates new adapter when no prev_checkpoint
    model = get_peft_model(model, lora_config)
```

**Verification:**
- ✅ `prepare_model_for_kbit_training()`: Called ONCE (line 233), outside prev_checkpoint block
- ✅ `get_peft_model()`: Only in `else` block (line 257), NOT in prev_checkpoint block
- ✅ `merge_and_unload()`: NOT FOUND in code (correct!)
- ✅ `is_trainable=True`: Used in `PeftModel.from_pretrained()` (line 242)

**Status:** ✅ **ALL CHECKS PASS**

---

## 🔍 Detailed Verification

### 1. Adapter State Check

**Run this to verify any stage:**
```bash
python verify_exp4_fix.py models/exp4_curriculum/stage1
```

**Expected Output:**
```
✓ Only one adapter: default
✓ Active adapter matches
✓ Trainable params: ~2,621,440 / ~8,000,000,000 (0.0328%)
✓ Base model is frozen (0 trainable params)
```

### 2. Training Script Check

**Manual verification:**
```bash
# Check for merge_and_unload (should return nothing)
grep -n "merge_and_unload" train_qlora_qwen3vl.py

# Check prepare_model_for_kbit_training (should only appear once, line 233)
grep -n "prepare_model_for_kbit_training" train_qlora_qwen3vl.py

# Check get_peft_model in prev_checkpoint block (should return nothing)
sed -n '/if prev_checkpoint:/,/else:/p' train_qlora_qwen3vl.py | grep "get_peft_model"
```

**Result:** ✅ All checks pass

---

## 📋 Pre-Training Checklist

### Before Stage 1
- [x] Config file: `exp4_stage1.yaml` has no `prev_checkpoint`
- [x] Base model path correct: `Qwen/Qwen3-VL-8B-Instruct`
- [x] LoRA config: r=4, alpha=8, target_modules=[q_proj, k_proj, v_proj, o_proj]

### Before Stage 2
- [ ] Stage 1 completed successfully
- [ ] Stage 1 adapter exists: `models/exp4_curriculum/stage1/adapter_model.safetensors`
- [ ] Run verification: `python verify_exp4_fix.py models/exp4_curriculum/stage1`
- [ ] Config file: `exp4_stage2.yaml` has `prev_checkpoint: models/exp4_curriculum/stage1`
- [ ] **Consider:** Reduce LR from `5.0e-5` to `3.5e-5` (30% reduction)

### Before Stage 3
- [ ] Stage 2 completed successfully
- [ ] Stage 2 adapter exists: `models/exp4_curriculum/stage2/adapter_model.safetensors`
- [ ] Run verification: `python verify_exp4_fix.py models/exp4_curriculum/stage2`
- [ ] Config file: `exp4_stage3.yaml` has `prev_checkpoint: models/exp4_curriculum/stage2`
- [ ] **Consider:** Reduce LR from `5.0e-5` to `2.5e-5` (50% reduction from Stage 1)

---

## 🎯 Learning Rate Recommendation

**Current:** All stages use `lr: 5.0e-5`

**Recommended (Option B - Fresh Optimizer):**
- Stage 1: `lr: 5.0e-5` (baseline)
- Stage 2: `lr: 3.5e-5` (30% reduction - adapter already trained)
- Stage 3: `lr: 2.5e-5` (50% reduction - fine-tuning phase)

**Rationale:** Since we're using fresh optimizer each stage, slightly lower LR prevents overshooting on already-trained adapter weights.

**Alternative (Option A - Resume Optimizer):**
If you want to resume optimizer state, modify training script:
```python
training_args = TrainingArguments(
    ...
    resume_from_checkpoint=prev_checkpoint if prev_checkpoint else None,
    ...
)
```

**Recommendation:** Keep Option B (current) but add LR decay in configs.

---

## 📊 Expected Training Behavior

### Stage 1 (Baseline)
- Starts from random LoRA initialization
- Loss decreases from high initial value
- Final validation loss: `L1_final` (note this value)

### Stage 2 (Continue from Stage 1)
- **First validation loss should be ~L1_final** (±5-10%)
- Loss continues decreasing smoothly
- **No big jump** (if jump > 50%, adapter wasn't loaded correctly)
- Validation accuracy on Stage 1 questions should remain high

### Stage 3 (Continue from Stage 2)
- **First validation loss should be ~L2_final** (±5-10%)
- Smooth continuation
- Final model should perform well on all stages

---

## 🚨 Red Flags During Training

**If you see these, STOP and investigate:**

1. **Loss Jump at Stage Transition**
   ```
   Stage 1 final loss: 0.5
   Stage 2 first loss: 2.5  # ❌ BIG JUMP - adapter not loaded!
   ```
   **Fix:** Check adapter path, verify adapter exists, re-run verification script

2. **All Parameters Trainable**
   ```
   trainable params: 8,000,000,000  # ❌ Should be ~2.5M
   ```
   **Fix:** Base model is not frozen - check quantization/prepare_model_for_kbit_training

3. **Multiple Adapters**
   ```
   Adapters found: ['default', 'stage2']  # ❌ Should be ['default']
   ```
   **Fix:** Check if get_peft_model was called in prev_checkpoint block

4. **Validation Accuracy Drops**
   ```
   Stage 1 final: 92% accuracy
   Stage 2 first eval: 65% accuracy  # ❌ Should be ~92%
   ```
   **Fix:** Adapter not loaded correctly, knowledge lost

---

## ✅ Success Criteria

**After retraining Exp4, you should see:**

1. **Overall Accuracy:** ~92% ± 0.5% (matches Exp1-3)
2. **Smooth Loss Curves:** No big jumps between stages
3. **Validation Continuity:** Stage N+1 first eval ≈ Stage N final eval
4. **Consistent Performance:** All question types perform similarly to Exp1-3

**If Exp4 achieves ~92% accuracy, the fix is successful!** 🎉

---

**Last Verified:** 2025-01-XX  
**Code Status:** ✅ Verified and ready for retraining

