#!/bin/bash
# Extract EndoVis2018 images from zip files
# Run this on a compute node if needed (large files)

set -e

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
ZIP_DIR="${BASE_DIR}/EndoVis2018/EndoVis2018/training data 1"
TARGET_DIR="${BASE_DIR}/EndoVis2018/data/images"

echo "=========================================="
echo "Extracting EndoVis2018 Images"
echo "=========================================="
echo "Source: ${ZIP_DIR}"
echo "Target: ${TARGET_DIR}"
echo ""

# Create target directory
mkdir -p "${TARGET_DIR}"

# Extract zip files
cd "${ZIP_DIR}"

for zip_file in *.zip; do
    if [ -f "$zip_file" ]; then
        echo "Extracting: $zip_file"
        unzip -q -o "$zip_file" -d "${TARGET_DIR}" || echo "Warning: Failed to extract $zip_file"
    fi
done

echo ""
echo "=========================================="
echo "Extraction Complete"
echo "=========================================="
echo "Images should now be in: ${TARGET_DIR}"
echo ""

# Count extracted images
IMAGE_COUNT=$(find "${TARGET_DIR}" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
echo "Found ${IMAGE_COUNT} images"




















