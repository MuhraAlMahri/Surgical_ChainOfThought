# Experiment 2: Qwen Reordered Training

## ✅ Complete Setup Ready!

Train on QA pairs that have been intelligently reordered by Qwen into 3 clinical stages.

---

## 🎯 Overview

**Strategy**: Train on Qwen-reordered data (not random order like exp1)
- Qwen model analyzed all questions and organized them into 3 clinical stages
- Stage 1 → Initial Assessment (quality, procedure, artifacts)  
- Stage 2 → Findings Identification (abnormalities, instruments, landmarks)
- Stage 3 → Clinical Context (diagnosis, treatment)

**Model**: Qwen3-VL-8B-Instruct (larger and newer than exp1's Qwen2-VL-7B)  
**Instructions**: ULTRA_CONDENSED (363 chars, concise)  
**Resolution**: 768×768 with letterbox padding (no warping)  
**Hardware**: 2 GPUs with DistributedDataParallel  
**Data**: Image-level splits (no leakage between train/val/test)

---

## 📊 Data Distribution

### Stage Distribution (Qwen's Classification)

| Stage | Train | Val | Test | Total | Purpose |
|-------|-------|-----|------|-------|---------|
| **Stage 1** | 14,679 (35.7%) | 3,086 (35.1%) | 3,275 (36.5%) | 21,040 | Initial Assessment |
| **Stage 2** | 26,357 (64.2%) | 5,689 (64.8%) | 5,703 (63.5%) | 37,749 | Findings |
| **Stage 3** | 43 (0.1%) | 11 (0.1%) | 6 (0.1%) | 60 | Clinical Context |
| **Total** | 41,079 | 8,786 | 8,984 | 58,849 | All QA pairs |

**Note**: Stage 3 has very few samples - this is expected as clinical diagnosis questions are rare in the dataset.

### Stage Examples

**Stage 1 (Initial Assessment):**
- "What type of procedure is shown?"
- "Is there text visible?"
- "Are there artifacts present?"

**Stage 2 (Findings Identification):**
- "What abnormalities are present?"
- "What instruments are visible?"
- "Where is the abnormality located?"

**Stage 3 (Clinical Context):**
- "What is the diagnosis?"
- "What treatment is recommended?"

---

## ⏱️ Time Estimate

**Single Training Session (2 GPUs)**: ~14 hours

Unlike the curriculum learning approach, this is **just one training session** where all stages are mixed together in Qwen's suggested order.

---

## 🚀 Quick Start

```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp2
sbatch slurm/train_exp2_qwen_reordered_2gpu.slurm
```

That's it! One job submission, ~14 hours to complete.

---

## 📁 Files Created

### Data Preparation
- ✅ `prepare_qwen_reordered_data.py` - Applies ULTRA_CONDENSED instructions
- ✅ `datasets/kvasir_qwen_reordered_ultra_condensed/train.json`
- ✅ `datasets/kvasir_qwen_reordered_ultra_condensed/val.json`
- ✅ `datasets/kvasir_qwen_reordered_ultra_condensed/test.json`

### Training
- ✅ `train_exp2_qwen_reordered.py` - Training script
- ✅ `dataset.py` - Dataset loader with letterbox support
- ✅ `templates.py` - Prompt templates
- ✅ `config_exp2_qwen_reordered_2gpu.yaml` - Qwen3-VL-8B config

### Job Script
- ✅ `slurm/train_exp2_qwen_reordered_2gpu.slurm` - 2-GPU job script

### Output
```
exp2/
└── outputs/           # Final checkpoint here
```

---

## 💡 Qwen Reordering vs Random (Exp1)

### Hypothesis
If Qwen's clinical ordering is meaningful, exp2 should outperform exp1's random ordering.

### Exp1 (Random Order)
- Questions presented in **random order**
- Model sees easy and hard questions mixed
- No logical flow

### Exp2 (Qwen Reordered)
- Questions presented in **clinical order** (as determined by Qwen)
- Stage 1 → Stage 2 → Stage 3 flow (but all in one training session)
- Mimics clinical workflow

**Test**: Does intelligent ordering help model learn better?

---

## 🔧 Technical Details

### Model Configuration
```yaml
model_name: Qwen/Qwen3-VL-8B-Instruct  # 8B parameters
vision_frozen: true
lora:
  r: 16
  alpha: 32
  dropout: 0.05
```

### ULTRA_CONDENSED Instructions
```
You are a surgical image analysis assistant analysing an endoscopic image.
- Select your answer(s) ONLY from the provided candidate list
- For multi-label questions: Select ALL applicable items, separated by semicolons (;)
- For single-choice questions: Select EXACTLY one option
- Output format: item1; item2; item3 (for multi-label) or item1 (for single-choice)
```

**Character count**: 363 chars (very concise!)

### Batch Configuration (2 GPUs)
```yaml
train_bs: 4           # Per GPU
grad_accum: 8         # Gradient accumulation
# Effective batch = 4 × 2 GPUs × 8 = 64
```

### Hardware Requirements
| Resource | Amount |
|----------|--------|
| GPUs | 2 × A100 (or similar) |
| Memory per GPU | ~40-50 GB |
| System RAM | 128 GB |
| Storage | ~30 GB |

---

## 📈 Expected Results

### Comparison to Exp1

| Experiment | Model | Strategy | Expected Accuracy |
|------------|-------|----------|-------------------|
| Exp1 | Qwen2-VL-7B | Random order | ~22-23% |
| **Exp2** | **Qwen3-VL-8B** | **Qwen reordered** | **~24-26%** |

**Expected improvement**: +2-4% if Qwen ordering helps

**Two factors**:
1. Larger model (8B vs 7B params)
2. Better ordering (Qwen reordered vs random)

---

## 🆚 Exp1 vs Exp2 Key Differences

| Feature | Exp1 (Random) | Exp2 (Qwen Reordered) |
|---------|---------------|------------------------|
| **Model** | Qwen2-VL-7B (7B) | **Qwen3-VL-8B (8B)** |
| **Data Order** | Random | **Qwen clinical stages** |
| **Instructions** | Standard | **ULTRA_CONDENSED** |
| **Training** | 1 session (~14h) | 1 session (~14h) |
| **Complexity** | Simple baseline | Intelligently ordered |

---

## 🐛 Troubleshooting

### Job Fails to Start
**Check data exists:**
```bash
ls datasets/kvasir_qwen_reordered_ultra_condensed/
# Should see: train.json, val.json, test.json
```

**If missing, regenerate:**
```bash
python3 exp2/prepare_qwen_reordered_data.py
```

### Out of Memory
**Solution**: Reduce batch size in config:
```yaml
train_bs: 2   # Was 4
grad_accum: 16  # Was 8
# Keeps effective batch = 64
```

### Training Slower Than Expected
**Check**:
- Both GPUs utilized: `nvidia-smi -l 1`
- TF32 enabled: Check logs for "TF32 enabled"
- NCCL backend active: Check logs for "nccl"

---

## 📊 Monitoring Training

### Check Status
```bash
# Job status
squeue -j <JOB_ID>

# Live output
tail -f slurm/logs/train_qwen_reordered_2gpu_*.out

# Check errors
tail -f slurm/logs/train_qwen_reordered_2gpu_*.err
```

### Verify Training Started
Look for in logs:
```
✓ TF32 enabled
Running in distributed mode with 2 GPUs
✓ Letterbox mode enabled: 768×768
EXP2: QWEN REORDERED TRAINING
```

### Monitor GPU Usage
```bash
ssh <node> nvidia-smi -l 1
# Should show both GPUs at ~80-90% utilization
```

---

## 🎯 Research Questions

### What Exp2 Tests
1. **Does Qwen's clinical ordering help?**
   - Compare exp2 vs exp1 accuracy
   - If exp2 > exp1 significantly, ordering matters

2. **Does Qwen3-VL-8B outperform Qwen2-VL-7B?**
   - Larger model (8B vs 7B params)
   - Newer architecture

3. **Do ULTRA_CONDENSED instructions work?**
   - Very concise (363 chars)
   - Compare to longer instructions

---

## ✨ Summary

**Experiment 2 is fully configured and ready!**

### Key Features
✅ **Qwen reordered data** (intelligent clinical ordering)  
✅ **Qwen3-VL-8B** (8B parameters, newer model)  
✅ **ULTRA_CONDENSED** (very concise instructions)  
✅ **768×768 letterbox** (no image warping)  
✅ **2 GPU training** (~1.7× speedup)  
✅ **Image-level splits** (no data leakage)

### To Start
```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp2
sbatch slurm/train_exp2_qwen_reordered_2gpu.slurm
```

### Timeline
- **Training**: ~14 hours
- **Expected completion**: Tomorrow this time

**Much simpler than curriculum learning - just one training session!** 🚀

---

*Setup Date: November 11, 2025*  
*Model: Qwen3-VL-8B-Instruct*  
*Resolution: 768×768 letterbox (aspect-ratio preserving)*  
*Strategy: Qwen reordered clinical stages (1→2→3)*






