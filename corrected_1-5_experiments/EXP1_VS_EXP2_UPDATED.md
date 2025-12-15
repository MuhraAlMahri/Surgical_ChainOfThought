# Experiment 1 vs Experiment 2 Comparison (UPDATED)

## Quick Reference

| Feature | Exp1: Random Baseline | Exp2: Qwen Reordered |
|---------|----------------------|----------------------|
| **Model** | Qwen2-VL-7B (7B params) | **Qwen3-VL-8B (8B params)** |
| **Data Order** | Random | **Qwen clinical stages (1→2→3)** |
| **Instructions** | Standard | **ULTRA_CONDENSED (363 chars)** |
| **Training** | 1 session | 1 session |
| **Training Time** | ~14 hours | ~14 hours |
| **Resolution** | 768×768 letterbox | 768×768 letterbox |
| **GPUs** | 2 GPUs | 2 GPUs |
| **Expected Accuracy** | ~22-23% | **~24-26%** (+2-4%) |
| **Data Splits** | Image-level (no leakage) | Image-level (no leakage) |
| **Complexity** | Simple baseline | Intelligently ordered |

---

## 🎯 Key Difference

### Exp1: Random Order
Questions presented in **random shuffle**:
```
Q54: What abnormalities? → Q12: Is there text? → Q3: What procedure? → ...
```
No logical flow, just random order.

### Exp2: Qwen Reordered
Questions presented in **clinical stages** (as determined by Qwen):
```
Stage 1 (35%): Quality/Procedure questions first
Stage 2 (64%): Findings/Instruments questions middle  
Stage 3 (0.1%): Clinical/Diagnosis questions last
```
Logical clinical workflow ordering.

---

## 📊 Data Comparison

### Exp1 (Random)
- **Train**: 41,079 QA pairs (random order)
- **Val**: 8,786 QA pairs (random order)
- **Test**: 8,984 QA pairs (random order)

### Exp2 (Qwen Reordered)
- **Train**: 41,079 QA pairs (Qwen stage order)
  - Stage 1: 14,679 (35.7%)
  - Stage 2: 26,357 (64.2%)
  - Stage 3: 43 (0.1%)
- **Val**: 8,786 QA pairs (Qwen stage order)
- **Test**: 8,984 QA pairs (Qwen stage order)

**Same data, different ordering!**

---

## ⏱️ Time Investment

### Both Experiments
- **Training time**: ~14 hours each on 2 GPUs
- **Wall-clock time**: 14 hours
- **GPU-hours**: 28 GPU-hours each (2 GPUs × 14h)

**Time efficient**: No additional time cost for exp2 vs exp1!

---

## 💡 Research Hypothesis

**Question**: Does intelligent ordering (by Qwen) improve learning vs random order?

**If Exp2 > Exp1 significantly**:
- ✅ Ordering matters
- ✅ Clinical workflow helps model learn
- ✅ Qwen's stage classification is meaningful

**If Exp2 ≈ Exp1**:
- Model learns regardless of order
- Or: Both Qwen3-8B and ordering cancel out differences

---

## 🚀 How to Run Both

### Option 1: Run Sequentially (Recommended)
```bash
# Start Exp1 first (baseline)
cd exp1
sbatch slurm/train_exp1_768_letterbox_2gpu.slurm

# Wait ~14 hours, then start Exp2
cd ../exp2
sbatch slurm/train_exp2_qwen_reordered_2gpu.slurm

# Total time: ~28 hours
```

### Option 2: Run in Parallel (If you have 4 GPUs)
```bash
# Terminal 1: Exp1 on GPUs 0-1
cd exp1
sbatch slurm/train_exp1_768_letterbox_2gpu.slurm

# Terminal 2: Exp2 on GPUs 2-3
cd exp2
# (Modify SLURM script to use different GPUs)
sbatch slurm/train_exp2_qwen_reordered_2gpu.slurm

# Both complete in ~14 hours
```

---

## 📈 Expected Results

### Baseline (Exp1)
- **Overall Accuracy**: ~22-23%
- **Model**: Qwen2-VL-7B
- **Order**: Random

### Qwen Reordered (Exp2)
- **Overall Accuracy**: ~24-26%
- **Model**: Qwen3-VL-8B (larger + newer)
- **Order**: Qwen clinical stages

**Expected improvement sources**:
1. **Better ordering** (+1-2%)
2. **Larger model** (+1-2%)
3. **ULTRA_CONDENSED instructions** (+0.5-1%)

**Total**: +2-4% improvement

---

## 🔬 What Each Experiment Tests

### Exp1 Tests
- ❓ Baseline performance with random ordering
- ❓ Qwen2-VL-7B capability on endoscopic VQA
- ❓ 768×768 letterbox effectiveness

### Exp2 Tests
- ❓ Does Qwen's clinical ordering help?
- ❓ Qwen3-VL-8B vs Qwen2-VL-7B performance
- ❓ ULTRA_CONDENSED instructions effectiveness
- ❓ Stage-based presentation benefits

---

## 📂 Directory Structure

```
corrected_1-5_experiments/
├── exp1/                                    # Random Baseline
│   ├── train_exp1.py
│   ├── config_exp1_768_letterbox_2gpu.yaml
│   ├── slurm/train_exp1_768_letterbox_2gpu.slurm
│   └── outputs/                             # Checkpoint
│
├── exp2/                                    # Qwen Reordered
│   ├── prepare_qwen_reordered_data.py       # Data prep script
│   ├── train_exp2_qwen_reordered.py
│   ├── config_exp2_qwen_reordered_2gpu.yaml
│   ├── slurm/train_exp2_qwen_reordered_2gpu.slurm
│   └── outputs/                             # Checkpoint
│
└── datasets/
    ├── kvasir_raw_6500_image_level_70_15_15/  # Exp1 data (random)
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    └── kvasir_qwen_reordered_ultra_condensed/ # Exp2 data (ordered)
        ├── train.json
        ├── val.json
        └── test.json
```

---

## ✨ Recommendation

### For Quick Baseline
**Run Exp1** to:
- Get baseline results quickly
- Establish performance floor
- Test infrastructure

### For Best Results
**Then run Exp2** to:
- Test intelligent ordering hypothesis
- Use larger/newer model (Qwen3-VL-8B)
- Achieve potentially higher accuracy

### Timeline
```
Hour 0: Submit Exp1
Hour 14: Exp1 completes → Submit Exp2
Hour 28: Exp2 completes → Compare results
Total: ~28 hours (~1.2 days) for both
```

---

## 🎓 Scientific Value

This is a **controlled experiment** testing ordering effect:
- ✅ Same images
- ✅ Same questions
- ✅ Same training time
- ✅ Same hardware
- ✅ Same resolution/preprocessing
- ⚠️ Different: Order + Model + Instructions

**Result will show**: Does intelligent ordering + better model help?

---

*Comparison Date: November 11, 2025*  
*Both experiments use 768×768 letterbox, 2 GPUs, image-level splits*  
*Key difference: Random order (exp1) vs Qwen clinical stages (exp2)*






