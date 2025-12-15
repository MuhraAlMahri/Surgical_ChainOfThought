#!/bin/bash
# Quick script to help upload EndoVis2018 dataset to cluster

set -e

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
SOURCE_DIR="${BASE_DIR}/EndoVis2018/data/images"
TARGET_DIR="${BASE_DIR}/datasets/EndoVis2018"

echo "=========================================="
echo "EndoVis2018 Dataset Upload Helper"
echo "=========================================="
echo ""

# Check if we're on the cluster
if [ ! -d "$BASE_DIR" ]; then
    echo "❌ Error: Base directory not found: $BASE_DIR"
    echo "This script should be run on the cluster."
    exit 1
fi

echo "📋 Current Status:"
echo ""

# Check for existing images
IMAGE_COUNT=$(find "$SOURCE_DIR" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" 2>/dev/null | wc -l)
echo "  Images found: $IMAGE_COUNT"

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo ""
    echo "⚠️  No images found in $SOURCE_DIR"
    echo ""
    echo "📤 To upload images from your local machine:"
    echo ""
    echo "   scp -r /local/path/to/images/seq_* \\"
    echo "       muhra.almahri@<cluster>:/l/users/muhra.almahri/Surgical_COT/EndoVis2018/data/images/"
    echo ""
    echo "   Or use rsync for large datasets:"
    echo ""
    echo "   rsync -avz --progress /local/path/to/images/ \\"
    echo "       muhra.almahri@<cluster>:/l/users/muhra.almahri/Surgical_COT/EndoVis2018/data/images/"
    echo ""
    exit 0
fi

echo ""
echo "✅ Images found! Ready to organize."
echo ""
echo "🔧 To organize the dataset, run:"
echo ""
echo "   python scripts/organize_endovis2018.py"
echo ""
echo "This will:"
echo "  - Copy images to datasets/EndoVis2018/raw/images/"
echo "  - Create metadata files"
echo "  - Organize similar to Kvasir-VQA structure"
echo ""

# Check if already organized
if [ -d "${TARGET_DIR}/raw/images" ]; then
    ORGANIZED_COUNT=$(find "${TARGET_DIR}/raw/images" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" 2>/dev/null | wc -l)
    if [ "$ORGANIZED_COUNT" -gt 0 ]; then
        echo "📊 Already organized: $ORGANIZED_COUNT images in ${TARGET_DIR}/raw/images/"
        echo ""
        echo "✅ Dataset is ready to use!"
    fi
fi





















