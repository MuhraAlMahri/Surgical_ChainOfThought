# Experiment 1 vs Experiment 2 Comparison

## Quick Reference

| Feature | Exp1: Random Baseline | Exp2: Curriculum Learning |
|---------|----------------------|---------------------------|
| **Model** | Qwen2-VL-7B (7B params) | **Qwen3-VL-8B** (8B params) |
| **Strategy** | Random order | **3-stage curriculum** (easy→hard) |
| **Training Stages** | 1 stage | **3 stages** (progressive) |
| **Resolution** | 768×768 letterbox | 768×768 letterbox |
| **GPUs** | 2 GPUs | 2 GPUs |
| **Training Time** | ~14 hours | **~42 hours** (14h × 3 stages) |
| **Expected Accuracy** | ~22-23% | **~24-27%** (+3-5%) |
| **Instructions** | Standard | ULTRA_CONDENSED |
| **Data Splits** | Image-level (no leakage) | Image-level (no leakage) |

---

## 🎯 When to Use Each

### Use Exp1 (Random Baseline) When:
- ✅ Need quick results (~14 hours)
- ✅ Want simple baseline comparison
- ✅ Testing infrastructure
- ✅ Limited time/resources

### Use Exp2 (Curriculum Learning) When:
- ✅ Want best possible accuracy
- ✅ Can afford 42 hours training
- ✅ Need production model
- ✅ Research on curriculum learning

---

## 📊 Training Comparison

### Exp1: Single Stage
```
Base Model (Qwen2-VL-7B)
    ↓
Random Training (all questions mixed)
    ↓
Final Model (~14 hours)
```

### Exp2: Three Stages
```
Base Model (Qwen3-VL-8B)
    ↓
Stage 1: Initial Assessment (~14h)
    ↓
Stage 2: Findings Identification (~14h)
    ↓
Stage 3: Clinical Context (~14h)
    ↓
Final Model (~42 hours total)
```

---

## ⏱️ Time Investment

### Exp1: Random Baseline
| Phase | Time |
|-------|------|
| Training | 14 hours |
| **Total** | **14 hours** |

### Exp2: Curriculum Learning
| Stage | Time | Cumulative |
|-------|------|------------|
| Stage 1 | 14 hours | 14 hours |
| Stage 2 | 14 hours | 28 hours |
| Stage 3 | 14 hours | **42 hours** |

**Additional time**: +28 hours (3× longer)  
**Expected improvement**: +3-5% accuracy

---

## 💰 Cost-Benefit Analysis

### Exp1
- **GPU-hours**: 2 GPUs × 14h = **28 GPU-hours**
- **Wall-clock time**: 14 hours
- **Accuracy**: ~22-23%
- **Cost per accuracy point**: ~1.2 GPU-hours per %

### Exp2
- **GPU-hours**: 2 GPUs × 42h = **84 GPU-hours**
- **Wall-clock time**: 42 hours (sequential)
- **Accuracy**: ~24-27%
- **Cost per accuracy point**: ~3.4 GPU-hours per %

**Verdict**: Exp1 is more cost-efficient per accuracy point, but Exp2 achieves higher absolute accuracy.

---

## 🚀 How to Run Both

### Run Exp1 First (Quick Baseline)
```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1
sbatch slurm/train_exp1_768_letterbox_2gpu.slurm
```

**Complete in**: ~14 hours  
**Use for**: Baseline comparison

---

### Then Run Exp2 (Best Performance)
```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp2
./submit_all_stages.sh
```

**Complete in**: ~42 hours  
**Use for**: Production model

---

### Or Run Both in Parallel

If you have **4 GPUs** available:
```bash
# Terminal 1: Start Exp1 on GPUs 0-1
cd exp1
sbatch slurm/train_exp1_768_letterbox_2gpu.slurm

# Terminal 2: Start Exp2 Stage 1 on GPUs 2-3
cd exp2
# Modify SLURM scripts to use --gres=gpu:2 --constraint=gpu_2_3
sbatch slurm/train_stage1_2gpu.slurm
```

**Complete both in**: ~42 hours (Exp2 time, since it's longer)

---

## 📈 Expected Results Summary

### Baseline (Exp1)
- **Stage 1 Accuracy**: ~30%
- **Stage 2 Accuracy**: ~18%
- **Stage 3 Accuracy**: ~8%
- **Overall Accuracy**: ~22-23%

### Curriculum (Exp2)
- **Stage 1 Accuracy**: ~33%
- **Stage 2 Accuracy**: ~20%
- **Stage 3 Accuracy**: ~10%
- **Overall Accuracy**: ~24-27%

**Improvement**: Curriculum learning helps especially on harder stages (2 & 3)

---

## 🎓 Research Questions

### What Exp1 Answers
- ❓ How well does Qwen2-VL perform on random endoscopic VQA?
- ❓ Does 768×768 letterbox work better than 448×448?
- ❓ Baseline performance for comparison

### What Exp2 Answers
- ❓ Does curriculum learning improve over random training?
- ❓ How much benefit from progressive difficulty?
- ❓ Does Qwen3-VL-8B perform better than Qwen2-VL-7B?
- ❓ What's the performance ceiling with best practices?

---

## 📂 Directory Structure

```
corrected_1-5_experiments/
├── exp1/                          # Random Baseline
│   ├── train_exp1.py
│   ├── dataset.py
│   ├── templates.py
│   ├── config_exp1_768_letterbox_2gpu.yaml
│   ├── slurm/
│   │   └── train_exp1_768_letterbox_2gpu.slurm
│   └── outputs/                   # Final checkpoint here
│
└── exp2/                          # Curriculum Learning
    ├── train_exp2_curriculum.py
    ├── dataset.py
    ├── templates.py
    ├── config_exp2_curriculum_2gpu.yaml
    ├── slurm/
    │   ├── train_stage1_2gpu.slurm
    │   ├── train_stage2_2gpu.slurm
    │   └── train_stage3_2gpu.slurm
    ├── submit_all_stages.sh
    └── outputs/
        ├── stage1/                # Stage 1 checkpoint
        ├── stage2/                # Stage 2 checkpoint
        └── stage3/                # Final checkpoint
```

---

## ✨ Recommendation

### For Quick Testing
**Run Exp1 first** to:
- Verify infrastructure works
- Get baseline results quickly
- Test evaluation pipeline

### For Best Results
**Then run Exp2** to:
- Achieve best accuracy
- Publish/present results
- Deploy in production

### Timeline
```
Day 0: Submit Exp1
Day 1 (14h later): Exp1 completes → Submit Exp2
Day 3 (42h later): Exp2 completes
Total: ~56 hours (~2.3 days) for both experiments
```

---

*Comparison Date: November 11, 2025*  
*Both experiments use 768×768 letterbox, 2 GPUs, image-level splits*






