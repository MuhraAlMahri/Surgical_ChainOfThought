# Exp2 Evaluation Package for Google Colab

This package contains everything needed to run Experiment 2 evaluation in Google Colab.

## 📦 Package Contents

- `scripts/evaluate_exp2.py` - Main evaluation script
- `Exp2_Evaluation_Colab.ipynb` - Ready-to-use Colab notebook
- `requirements.txt` - Python dependencies
- `README.md` - This file

## 🚀 Quick Start

### Option 1: Use the Colab Notebook (Recommended)

1. **Upload this zip to Google Colab**
   - Open Google Colab: https://colab.research.google.com/
   - Upload `Exp2_Evaluation_Colab.ipynb` or the entire zip file

2. **Extract files** (if uploaded as zip)
   ```python
   !unzip exp2_colab_package.zip
   %cd exp2_colab_package
   ```

3. **Follow the notebook cells** - The notebook will guide you through:
   - Installing dependencies
   - Uploading model checkpoint
   - Uploading test data
   - Uploading images
   - Running evaluation
   - Viewing results

### Option 2: Manual Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data**
   - Model checkpoint: LoRA adapter folder (from `models/exp2_qwen_reordered/`)
   - Test data: JSON file with test samples
   - Images: Directory containing all test images

3. **Run evaluation**
   ```bash
   python3 scripts/evaluate_exp2.py \
       --model_path /path/to/exp2_qwen_reordered \
       --test_data /path/to/test.json \
       --image_dir /path/to/images \
       --output /path/to/results/exp2_evaluation.json \
       --base_model Qwen/Qwen3-VL-8B-Instruct
   ```

## 📋 What You Need

### 1. Model Checkpoint
- **Path**: `models/exp2_qwen_reordered/`
- **Size**: ~15-20 MB (LoRA adapter only)
- **Contents**: Should contain `adapter_model.safetensors` and `adapter_config.json`

### 2. Test Dataset
- **Format**: JSON file (not JSONL)
- **Size**: ~8,984 samples
- **Structure**: Each item should have:
  ```json
  {
    "image_id": "...",
    "image_filename": "...",
    "question": "...",
    "instruction": "...",
    "answer": "...",
    "stage": 1,
    "question_type": "..."
  }
  ```

### 3. Images Directory
- **Location**: Directory containing all test images
- **Naming**: Images should match `image_filename` from test data
- **Format**: JPG, PNG, etc.

## ⚙️ Configuration

### Base Model
The script uses `Qwen/Qwen3-VL-8B-Instruct` by default. This will be downloaded automatically from HuggingFace (requires internet connection).

### GPU Requirements
- **Recommended**: T4 GPU or better (free tier in Colab)
- **Memory**: ~16GB GPU memory
- **Time**: ~2-3 hours for full evaluation (8,984 samples)

### Quick Testing
To test with a subset of samples:
```bash
python3 scripts/evaluate_exp2.py \
    ... \
    --max_samples 100  # Only evaluate first 100 samples
```

## 📊 Output Format

The evaluation produces a JSON file with:
```json
{
  "total": 8984,
  "correct": 8336,
  "accuracy": 92.79,
  "by_stage": {
    "Stage 1": {
      "total": 3275,
      "correct": 3021,
      "accuracy": 92.24
    },
    ...
  },
  "by_question_type": {
    "multi_label": {
      "total": 2232,
      "correct": 2056,
      "accuracy": 92.11
    },
    ...
  },
  "predictions": [
    {
      "image_id": "...",
      "question": "...",
      "instruction": "...",
      "question_type": "...",
      "prediction": "...",
      "ground_truth": "...",
      "correct": true,
      "stage": 1
    },
    ...
  ]
}
```

## 🔧 Troubleshooting

### Out of Memory
- Reduce batch size (not applicable - processes one at a time)
- Use `--max_samples` to test with fewer samples first
- Use Colab Pro for better GPU

### Model Download Issues
- Ensure you have internet connection
- Check HuggingFace access (may need to login)
- Try downloading model manually first

### Image Not Found Errors
- Verify image paths match `image_filename` in test data
- Check image directory path is correct
- Ensure images are in the specified directory

### Slow Performance
- Normal: ~1-2 seconds per sample
- Full evaluation takes 2-3 hours
- Use `--max_samples` for quick testing

## 📝 Notes

- The script automatically merges LoRA weights for faster inference
- Results include `instruction` and `question_type` fields (fixed version)
- Evaluation uses smart matching (exact, substring, fuzzy similarity)
- All predictions are saved in the output JSON

## 🆘 Support

If you encounter issues:
1. Check that all paths are correct
2. Verify model checkpoint structure
3. Ensure test data format matches expected structure
4. Check Colab GPU is allocated (Runtime → Change runtime type → GPU)








