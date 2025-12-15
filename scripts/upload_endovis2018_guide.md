# Upload EndoVis2018 Dataset to Cluster

## 📋 Overview

This guide helps you upload the EndoVis2018 dataset to the cluster and organize it similar to the Kvasir-VQA structure.

## 📁 Target Structure

After organization, the dataset will be structured as:
```
datasets/EndoVis2018/
├── dataset_info.json          # Dataset metadata
└── raw/
    ├── images/                # All images in flat structure
    │   ├── endovis_seq1_frame000.png
    │   ├── endovis_seq1_frame001.png
    │   └── ...
    └── metadata/
        ├── image_mapping.json  # Mapping of image IDs to source paths
        └── splits.json         # Train/val/test splits
```

## 🚀 Step 1: Upload Images to Cluster

### Option A: Upload from Local Machine (if you have images locally)

```bash
# From your local machine, upload images to the cluster
# Replace <cluster-address> with your cluster address

# Upload all sequence directories
scp -r /local/path/to/EndoVis2018/data/images/seq_* \
    muhra.almahri@<cluster-address>:/l/users/muhra.almahri/Surgical_COT/EndoVis2018/data/images/

# Or upload specific sequences
scp -r /local/path/to/EndoVis2018/data/images/seq_1 \
    muhra.almahri@<cluster-address>:/l/users/muhra.almahri/Surgical_COT/EndoVis2018/data/images/
```

### Option B: Download from Official Source

If you don't have the images locally, download them from:
- **MICCAI EndoVis 2018 Challenge**: https://endovissub-instrument.grand-challenge.org/
- **GitHub Repository**: https://github.com/mli0603/EndoVis2018

Then upload using Option A.

### Option C: Use rsync (for large datasets)

```bash
# From your local machine
rsync -avz --progress \
    /local/path/to/EndoVis2018/data/images/ \
    muhra.almahri@<cluster-address>:/l/users/muhra.almahri/Surgical_COT/EndoVis2018/data/images/
```

## 📂 Step 2: Verify Images Are Uploaded

SSH into the cluster and verify:

```bash
cd /l/users/muhra.almahri/Surgical_COT

# Check if images exist
find EndoVis2018/data/images -name "*.png" | wc -l

# Check specific sequence
ls -lh EndoVis2018/data/images/seq_1/left_frames/ | head -10
```

## 🔧 Step 3: Organize the Dataset

Once images are uploaded, run the organization script:

```bash
cd /l/users/muhra.almahri/Surgical_COT

# Run the organization script
python scripts/organize_endovis2018.py
```

This script will:
- ✅ Copy images to `datasets/EndoVis2018/raw/images/`
- ✅ Create unique image IDs (e.g., `endovis_seq1_frame000.png`)
- ✅ Generate metadata files
- ✅ Create split information from index files

## ✅ Step 4: Verify Organization

```bash
# Check organized structure
ls -lh datasets/EndoVis2018/raw/images/ | head -10

# Check metadata
cat datasets/EndoVis2018/dataset_info.json

# Count images
find datasets/EndoVis2018/raw/images -name "*.png" | wc -l
```

## 📊 Expected Results

After organization, you should have:
- Images in `datasets/EndoVis2018/raw/images/` with naming like `endovis_seq1_frame000.png`
- Metadata files in `datasets/EndoVis2018/raw/metadata/`
- Dataset info in `datasets/EndoVis2018/dataset_info.json`

## 🔍 Troubleshooting

### No images found
- Make sure images are in `EndoVis2018/data/images/seq_*/left_frames/`
- Check file extensions (should be .png, .jpg, or .jpeg)
- Verify permissions on uploaded files

### Permission errors
```bash
# Fix permissions if needed
chmod -R 755 EndoVis2018/data/images/
```

### Missing index files
The script uses index files from `EndoVis2018/data/index/`. If they're missing:
- Check that `train_data.txt`, `validation_data.txt`, and `test_data_final.txt` exist
- The script will still organize images even if index files are missing

## 📝 Next Steps

After organizing:
1. Convert to VQA format (if needed):
   ```bash
   python scripts/endovis2018_loader.py --action convert --output datasets/endovis2018_vqa
   ```

2. Use in training pipeline similar to Kvasir-VQA





















