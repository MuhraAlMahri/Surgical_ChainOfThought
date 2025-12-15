# Download Instructions for Colab Package

## 📦 Package Ready!

**File**: `colab_package.zip`  
**Location**: `/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/colab_package.zip`  
**Size**: 896MB

---

## 📥 How to Download

### Option 1: SCP (Recommended)

From your local machine:

```bash
scp muhra.almahri@<cluster-address>:/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/colab_package.zip .
```

### Option 2: SFTP

Use an SFTP client (FileZilla, WinSCP, etc.) to download:
- Host: Your cluster address
- Path: `/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/colab_package.zip`

### Option 3: rsync

```bash
rsync -avz --progress muhra.almahri@<cluster-address>:/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/colab_package.zip .
```

---

## 🚀 Quick Start in Colab

### 1. Upload to Colab

Open Google Colab: https://colab.research.google.com/

```python
# Upload the zip file
from google.colab import files
uploaded = files.upload()  # Select colab_package.zip

# Unzip
!unzip colab_package.zip
%cd colab_package
```

### 2. Install Dependencies

```python
!pip install -q transformers accelerate peft pillow torch
```

### 3. Enable GPU

**IMPORTANT**: Enable GPU for faster inference!
- Click: Runtime → Change runtime type
- Select: GPU (T4, or A100 if you have Colab Pro)
- Click: Save

### 4. Run Predictions

```python
# This runs on 50 sample images (~10 minutes)
!python3 scripts/predict_sample.py
```

### 5. Evaluate

```python
# Compare Exp1 vs Exp2
!python3 scripts/evaluate.py
```

### 6. View Results

```python
import json

# Load results
with open('evaluation_results.json') as f:
    results = json.load(f)

# Print comparison
print(f"Exp1 (Random):       {results['exp1_random']['accuracy']:.2f}%")
print(f"Exp2 (Qwen Ordered): {results['exp2_qwen_ordered']['accuracy']:.2f}%")
print(f"Difference:          {results['difference']:+.2f}%")
```

---

## 📊 What's in the Package

```
colab_package/
├── exp1_model/          # 464MB - Exp1 trained model
├── exp2_model/          # 501MB - Exp2 trained model
├── sample_images/       # 14MB - 50 sample test images
├── test_data.jsonl      # 688KB - Full test set (8,984 samples)
├── scripts/
│   ├── predict_sample.py   # Generate predictions
│   ├── evaluate.py         # Evaluate & compare
│   └── requirements.txt    # Dependencies
└── README.md            # Detailed instructions
```

---

## ⏱️ Expected Runtime

**On Colab with GPU:**
- Model loading: ~2 minutes (per model)
- Predictions (50 images): ~5-8 minutes (per experiment)
- Evaluation: <1 minute
- **Total: ~15-20 minutes**

**On Colab without GPU (not recommended):**
- Much slower (1+ hour)

---

## 💡 Tips

1. **Use GPU!** CPU will be very slow
2. **Keep session alive** - Colab disconnects after inactivity
3. **Save results** - Download JSON files before closing
4. **Pro tip**: If you have Colab Pro, use A100 for faster inference

---

## 🎯 What You'll Get

### Results for Advisor:

- ✅ Exp1 accuracy (Random ordering)
- ✅ Exp2 accuracy (Qwen clinical ordering)
- ✅ Direct comparison: Which is better?
- ✅ Per-category breakdown
- ✅ Answer to research question: **Does clinical ordering help?**

---

## ⚠️ Limitations

**This package uses 50 sample images** (not all 8,984)
- Good for: Quick demo, proof of concept
- Shows: The approach works
- Not: Complete evaluation (for that, need all images = 1.7GB more)

**For full results**, wait for cluster predictions (Jobs 155442, 155443)

---

## 📧 Troubleshooting

**"Out of memory" error?**
- Make sure GPU is enabled
- Try restarting runtime
- Reduce batch size if needed

**Models not loading?**
- Check you unzipped the file
- Check you're in the colab_package directory
- Reinstall dependencies

**Slow predictions?**
- Enable GPU (see step 3 above)
- Colab free tier is slower than cluster

---

## 🎉 You're All Set!

Download the file, upload to Colab, and run the scripts!

Results in ~15-20 minutes for your advisor meeting! 🚀

---

**Created**: Nov 12, 2025  
**Package size**: 896MB  
**Sample images**: 50  
**Models**: Qwen3-VL-8B checkpoint-642 (1 epoch, 768×768)





