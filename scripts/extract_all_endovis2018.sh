#!/bin/bash
# Extract all EndoVis2018 zip files
# Run this on a compute node if needed (large files)

set -e

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
ZIP_DIRS=(
    "${BASE_DIR}/EndoVis2018/EndoVis2018/training data 1"
    "${BASE_DIR}/EndoVis2018/EndoVis2018/traning data 2"
)
TARGET_DIR="${BASE_DIR}/EndoVis2018/data/images"

echo "=========================================="
echo "Extracting All EndoVis2018 Zip Files"
echo "=========================================="
echo "Target: ${TARGET_DIR}"
echo ""

# Create target directory
mkdir -p "${TARGET_DIR}"

# Extract zip files from all directories
for zip_dir in "${ZIP_DIRS[@]}"; do
    if [ -d "$zip_dir" ]; then
        echo "Processing directory: $zip_dir"
        cd "$zip_dir"
        
        for zip_file in *.zip; do
            if [ -f "$zip_file" ]; then
                echo "  Extracting: $zip_file"
                # Extract preserving directory structure
                unzip -q -o "$zip_file" -d "${TARGET_DIR}" 2>&1 | grep -v "inflating:" | grep -v "extracting:" || echo "    (extraction may have warnings)"
            fi
        done
    fi
done

echo ""
echo "=========================================="
echo "Extraction Complete"
echo "=========================================="

# Count extracted images
IMAGE_COUNT=$(find "${TARGET_DIR}" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
echo "Total images found: ${IMAGE_COUNT}"

# List sequences
echo ""
echo "Sequences found:"
find "${TARGET_DIR}" -type d -name "seq_*" | sort | while read seq_dir; do
    seq_name=$(basename "$seq_dir")
    img_count=$(find "${seq_dir}/left_frames" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
    if [ "$img_count" -gt 0 ]; then
        echo "  ${seq_name}: ${img_count} images"
    fi
done

echo ""
echo "Images extracted to: ${TARGET_DIR}"
echo "=========================================="




















