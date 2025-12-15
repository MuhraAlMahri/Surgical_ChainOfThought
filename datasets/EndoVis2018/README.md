# EndoVis2018 Dataset

This directory contains the organized EndoVis2018 dataset, structured similarly to Kvasir-VQA.

## 📁 Structure

```
datasets/EndoVis2018/
├── dataset_info.json          # Dataset metadata and information
└── raw/
    ├── images/                # All images in flat structure
    │   ├── endovis_seq1_frame000.png
    │   ├── endovis_seq1_frame001.png
    │   └── ...
    └── metadata/
        ├── image_mapping.json  # Mapping of image IDs to source paths
        └── splits.json         # Train/val/test splits
```

## 🚀 Quick Start

### 1. Upload Images (if not already done)

From your local machine:
```bash
scp -r /local/path/to/images/seq_* \
    muhra.almahri@<cluster>:/l/users/muhra.almahri/Surgical_COT/EndoVis2018/data/images/
```

### 2. Organize Dataset

On the cluster:
```bash
cd /l/users/muhra.almahri/Surgical_COT
python scripts/organize_endovis2018.py
```

### 3. Verify

```bash
# Check images
ls -lh datasets/EndoVis2018/raw/images/ | head -10

# Check metadata
cat datasets/EndoVis2018/dataset_info.json
```

## 📊 Dataset Information

- **Source**: https://github.com/mli0603/EndoVis2018
- **Type**: Surgical scene segmentation dataset
- **Classes**: 12 segmentation classes
- **Format**: Images organized by sequence and frame

## 🔗 Related Files

- Upload guide: `scripts/upload_endovis2018_guide.md`
- Organization script: `scripts/organize_endovis2018.py`
- Upload helper: `scripts/upload_endovis2018.sh`
- Original loader: `scripts/endovis2018_loader.py`

## 📝 Notes

- Images are organized in a flat structure with unique IDs
- Original sequence structure is preserved in metadata
- Compatible with Kvasir-VQA pipeline structure





















