# Quick Fix for "Can't find adapter_config.json" Error

## The Problem
The error occurs because the notebook is using placeholder paths (`/path/to/exp2_qwen_reordered`) instead of your actual model location.

## Solution

### Step 1: Upload Your Model Checkpoint
You need to upload the `exp2_qwen_reordered` folder that contains:
- `adapter_model.safetensors`
- `adapter_config.json`

**Option A: Upload as Zip**
1. Zip the `exp2_qwen_reordered` folder on your computer
2. In Colab, run the "Step 3: Upload Files" cell
3. Upload the zip file
4. It will automatically extract to `/content/models/`

**Option B: Upload to Google Drive**
1. Upload the `exp2_qwen_reordered` folder to Google Drive
2. Mount Google Drive in Colab (Step 2)
3. Update `MODEL_PATH` to point to your Drive location, e.g.:
   ```python
   MODEL_PATH = "/content/drive/MyDrive/path/to/exp2_qwen_reordered"
   ```

### Step 2: Update the Paths
In the notebook cell "Step 4: Configure Paths", update `MODEL_PATH` to match where you uploaded the model:

```python
# If uploaded to /content/models/exp2_qwen_reordered/
MODEL_PATH = "/content/models/exp2_qwen_reordered"

# OR if in Google Drive:
MODEL_PATH = "/content/drive/MyDrive/your/path/exp2_qwen_reordered"
```

### Step 3: Verify Paths
Run the path verification cell - it will check if:
- The model folder exists
- `adapter_config.json` is found
- `adapter_model.safetensors` is found

**All checks should show ✓ before proceeding!**

### Step 4: Run Evaluation
Once paths are verified, proceed to Step 5 to run the evaluation.

## Common Issues

**Issue: "Can't find adapter_config.json"**
- **Fix**: Make sure `MODEL_PATH` points to the folder containing `adapter_config.json`, not a parent folder
- **Check**: Run `!ls /content/models/exp2_qwen_reordered/` to see contents

**Issue: Model folder not found**
- **Fix**: Check where you uploaded it - use the file listing cell to find it
- **Tip**: If uploaded to current directory, it might be in `/content/exp2_colab_package/` or similar

**Issue: Files in Google Drive**
- **Fix**: Make sure Google Drive is mounted first (Step 2)
- **Fix**: Use full path: `/content/drive/MyDrive/...`

## Example: If Model is in Google Drive

```python
# Step 2: Mount Drive (uncomment)
from google.colab import drive
drive.mount('/content/drive')

# Step 4: Update path
MODEL_PATH = "/content/drive/MyDrive/Surgical_COT/models/exp2_qwen_reordered"
TEST_DATA = "/content/drive/MyDrive/Surgical_COT/data/test.json"
IMAGE_DIR = "/content/drive/MyDrive/Surgical_COT/images"
```







