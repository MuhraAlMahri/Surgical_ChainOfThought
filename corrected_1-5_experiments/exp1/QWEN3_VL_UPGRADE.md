# Qwen2-VL → Qwen3-VL Upgrade Guide

## Summary of Changes

**Date:** November 11, 2025  
**Change:** Upgraded from Qwen2-VL-7B-Instruct to Qwen3-VL-8B-Instruct

---

## What Changed

### 1. Model Name
- **Before:** `Qwen/Qwen2-VL-7B-Instruct`
- **After:** `Qwen/Qwen3-VL-8B-Instruct`

### 2. Model Class
- **Before:** `AutoModelForVision2Seq`
- **After:** `AutoModelForImageTextToText` (recommended by Qwen3-VL)

### 3. Model Loading
- Added `device_map="auto"` for better GPU memory management

---

## Files Updated

### Configuration Files:
- ✅ `config_exp1_category_based.yaml` (primary)
- ✅ `config_exp1.yaml`
- ✅ `config_exp1_actual.yaml`
- ✅ `config_exp1_4gpu.yaml`

### Python Files:
- ✅ `train_exp1.py` (main training script)
- ✅ `test_resolutions.py` (resolution testing)

---

## Qwen3-VL Improvements Over Qwen2-VL

### Performance Enhancements:
1. **Better Visual Understanding**
   - Enhanced perception of fine-grained details
   - Improved spatial reasoning
   - Better multi-object tracking

2. **Improved Instruction Following**
   - More accurate responses to complex queries
   - Better handling of multi-step instructions

3. **Enhanced Reasoning**
   - Stronger logical reasoning capabilities
   - Better at complex visual question answering

4. **Optimized Architecture**
   - More efficient parameter usage
   - Better memory management
   - Faster inference in some cases

### Medical Image Analysis Benefits:
- Better at detecting subtle abnormalities
- Improved understanding of anatomical structures
- More accurate classification of medical conditions
- Enhanced spatial relationship understanding

---

## Requirements

### Minimum Versions:
```bash
transformers >= 4.57.0
torch >= 2.0.0
peft >= 0.7.0
```

### Installation:
```bash
pip install "transformers>=4.57.0" --upgrade
```

---

## Testing Checklist

Before running full training, test the upgrade:

### 1. Quick Model Load Test
```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments
python3 -c "
from transformers import AutoModelForImageTextToText, AutoProcessor

model_name = 'Qwen/Qwen3-VL-8B-Instruct'
print('Loading model...')
model = AutoModelForImageTextToText.from_pretrained(
    model_name, 
    trust_remote_code=True,
    device_map='cpu'  # Test on CPU first
)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
print('✅ Model and processor loaded successfully!')
print(f'Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters')
"
```

### 2. Sanity Check Training (Optional)
```bash
# Run a quick sanity check with 10 steps
sbatch exp1/slurm/01_sanity_overfit.slurm
```

### 3. Resolution Test (Recommended)
```bash
# Test different resolutions with new model
sbatch exp1/slurm/test_resolutions.slurm
```

---

## Expected Behavior Changes

### Positive Changes:
- ✅ May see improved accuracy on validation set
- ✅ Better handling of complex medical terminology
- ✅ More consistent responses

### Neutral Changes:
- ⚠️ Training speed should be similar (maybe slightly different)
- ⚠️ Memory usage similar or slightly higher (8B vs 7B params)
- ⚠️ First run will download new model (~16 GB)

### Things to Monitor:
- 📊 Compare validation loss with previous Qwen2-VL runs
- 📊 Check if eval metrics improve
- 📊 Monitor training stability

---

## Rollback Plan

If issues arise with Qwen3-VL, you can rollback:

### Quick Rollback:
```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/exp1

# Revert config files
sed -i 's/Qwen3-VL-8B-Instruct/Qwen2-VL-7B-Instruct/g' config_exp1*.yaml

# Revert train_exp1.py
sed -i 's/AutoModelForImageTextToText/AutoModelForVision2Seq/g' train_exp1.py
sed -i '/device_map="auto"/d' train_exp1.py
```

---

## Training Command

To start training with Qwen3-VL:

```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments
sbatch exp1/slurm/train_exp1_category_based.slurm
```

**Note:** First run will download the Qwen3-VL-8B model (~16 GB).  
Model will be cached for future runs.

---

## Performance Comparison

After training completes, compare:

| Metric | Qwen2-VL (Before) | Qwen3-VL (After) | Change |
|--------|-------------------|------------------|--------|
| Val Loss | TBD | TBD | TBD |
| Accuracy | TBD | TBD | TBD |
| Training Time | ~50h (full res) | TBD | TBD |

Fill in results after training!

---

## References

- [Qwen3-VL Official Docs](https://qwen-3.com/)
- [Hugging Face Model Card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Qwen3-VL Blog Post](https://ollama.com/blog/qwen3-vl)

---

## Support

If you encounter issues:
1. Check transformers version: `pip show transformers`
2. Verify model download: Check `~/.cache/huggingface/hub/`
3. Review error logs in `exp1/slurm/logs/`
4. Consider rollback if persistent issues

---

**Status:** ✅ Upgrade Complete - Ready for Testing







