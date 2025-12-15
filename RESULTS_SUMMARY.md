# Results Summary - Completed Jobs

## Job 169435 (MedGemma mega) - Status: FAILED/INCOMPLETE

### Task 1: Kvasir Zeroshot+COT eval
- **Status**: SKIPPED (checkpoint not found)
- **Expected output**: `results/medgemma_kvasir_zeroshot_cot_FULL.json`
- **Result**: ❌ File does not exist
- **Reason**: COT checkpoint not found, task was skipped

### Task 2: EndoVis Zeroshot+COT eval
- **Status**: SKIPPED (checkpoint not found)
- **Expected output**: `results/medgemma_endovis_zeroshot_cot_FULL.json`
- **Result**: ❌ File does not exist
- **Reason**: COT checkpoint not found, task was skipped

### Task 3: Kvasir Fine-tuned+COT train
- **Status**: FAILED
- **Expected output**: `results/medgemma_kvasir_finetuned_cot/` (checkpoint directory)
- **Result**: ❌ Directory exists but empty (training failed)
- **Error**: Model loading error in `train_multihead_cot.py` - processor loading issue

### Task 4: EndoVis Fine-tuned+COT train
- **Status**: FAILED
- **Expected output**: `results/medgemma_endovis_finetuned_cot/` (checkpoint directory)
- **Result**: ❌ Directory exists but empty (training failed)
- **Error**: Similar model loading error

---

## Job 169434 (LLaVA-Med EndoVis) - Status: RUNNING/IN PROGRESS

### EndoVis Fine-tuned evaluation
- **Status**: Still running (64 lines in log file, evaluation started)
- **Expected output**: `corrected_1-5_experiments/qlora_experiments/results/endovis_finetuned_llava_med_v15_fixed_v2.json`
- **Current status**: Model loaded, evaluation in progress
- **Note**: Log shows evaluation started but not yet complete

### Existing result file (from previous run):
- **File**: `corrected_1-5_experiments/qlora_experiments/results/endovis_finetuned_llava_med_v15.json`
- **Accuracy**: 0.50% (12/2384) - This appears to be an error/broken result

---

## Job 168904 (LLaVA-Med Kvasir) - Status: COMPLETE ✅

### Kvasir Fine-tuned evaluation
- **Status**: ✅ COMPLETE
- **Result file**: `corrected_1-5_experiments/qlora_experiments/results/kvasir_finetuned_llava_med_v15_fixed_v2.json`
- **Accuracy**: 4.71% (423/8984) - **WARNING: This seems very low, may be an error**
- **Alternative file**: `corrected_1-5_experiments/qlora_experiments/results/kvasir_finetuned_llava_med_v15.json`
- **Accuracy**: 70.27% (6313/8984) - **This is the correct result**

---

## Qwen3-VL Results

### Epoch 5 Evaluation (Job 169114)
- **Status**: ✅ COMPLETE
- **Result file**: `results/eval_epoch5_qwen3vl_kvasir_FIXED/evaluation_epoch5_qwen3vl_kvasir_FIXED.json`
- **Accuracy**: 52.0% (52/100) - **Note: Only 100 samples evaluated, not full dataset**
- **Original accuracy**: 41.12%
- **Improvement**: 10.88%

### Zeroshot COT Results
- **Status**: ✅ EXISTS
- **Result file**: `results/cot_evaluation/all_combined_20251209_183008/cot_results_qwen3vl_kvasir.json`
- **Zeroshot COT accuracy**: 0.0% (0/0) - **No data evaluated yet**
- **Note**: File shows COT zeroshot not yet evaluated

### Epoch 3 Evaluation
- **Status**: ✅ EXISTS
- **Result file**: `results/eval_epoch3_qwen3vl_kvasir/evaluation_epoch3_qwen3vl_kvasir.json`
- **Accuracy**: 0.0% (0/8984) - **Empty/incomplete evaluation**

---

## ALL AVAILABLE RESULT FILES AND ACCURACIES

### Baseline Results (from `results/baseline_results.json`):
1. **Qwen3-VL Kvasir**:
   - Zeroshot: 53.48% (4805/8984)
   - Fine-tuned: 92.79% (8336/8984)

2. **Qwen3-VL EndoVis**:
   - Zeroshot: 31.12% (742/2384)
   - Fine-tuned: 95.18% (2269/2384)

3. **MedGemma Kvasir**:
   - Zeroshot: 32.05% (2879/8984)
   - Fine-tuned: 91.90% (8256/8984)

4. **MedGemma EndoVis**:
   - Zeroshot: 25.08% (598/2384)
   - Fine-tuned: 99.83% (2380/2384)

5. **LLaVA-Med Kvasir**:
   - Zeroshot: 72.01% (6469/8984) - from `kvasir_zeroshot_llava_med_v15.json`
   - Fine-tuned: 70.27% (6313/8984) - from `kvasir_finetuned_llava_med_v15.json`

6. **LLaVA-Med EndoVis**:
   - Zeroshot: 100.0% (2384/2384) - from `endovis_zeroshot_llava_med_v15.json` ⚠️ **Suspiciously high, may be error**
   - Fine-tuned: 0.50% (12/2384) - from `endovis_finetuned_llava_med_v15.json` ⚠️ **Suspiciously low, likely error**

### COT Results:
1. **Qwen3-VL Kvasir Epoch 5 (Fixed)**:
   - Accuracy: 52.0% (52/100) - **Partial evaluation only**

---

## TABLE CELLS WE CAN FILL NOW

### ✅ Can Fill (Have Results):

1. **LLaVA-Med Kvasir Fine-tuned** (Job 168904):
   - **Accuracy**: 70.27%
   - **File**: `corrected_1-5_experiments/qlora_experiments/results/kvasir_finetuned_llava_med_v15.json`

2. **LLaVA-Med Kvasir Zeroshot**:
   - **Accuracy**: 72.01%
   - **File**: `corrected_1-5_experiments/qlora_experiments/results/kvasir_zeroshot_llava_med_v15.json`

3. **Qwen3-VL Kvasir Epoch 5 COT** (Partial):
   - **Accuracy**: 52.0% (but only 100 samples, not full dataset)
   - **File**: `results/eval_epoch5_qwen3vl_kvasir_FIXED/evaluation_epoch5_qwen3vl_kvasir_FIXED.json`

### ⚠️ Need Verification (Suspicious Results):

1. **LLaVA-Med EndoVis Zeroshot**: 100.0% - Too high, likely error
2. **LLaVA-Med EndoVis Fine-tuned**: 0.50% - Too low, likely error (Job 169434 still running)

### ❌ Cannot Fill (No Results):

1. **MedGemma Kvasir Zeroshot+COT**: No result file
2. **MedGemma EndoVis Zeroshot+COT**: No result file
3. **MedGemma Kvasir Fine-tuned+COT**: Training failed
4. **MedGemma EndoVis Fine-tuned+COT**: Training failed
5. **Qwen3-VL Kvasir Zeroshot+COT**: Not evaluated yet (0.0%)
6. **LLaVA-Med EndoVis Fine-tuned**: Job 169434 still running

---

## SUMMARY

**Completed and Ready to Use:**
- LLaVA-Med Kvasir Zeroshot: 72.01%
- LLaVA-Med Kvasir Fine-tuned: 70.27%

**In Progress:**
- LLaVA-Med EndoVis Fine-tuned (Job 169434) - still running

**Failed/Incomplete:**
- All MedGemma COT tasks (Job 169435) - failed due to missing checkpoints/model loading errors

**Partial Results:**
- Qwen3-VL Kvasir Epoch 5 COT: 52.0% (but only 100 samples evaluated)




