#!/bin/bash
# Organize EndoVis2018 images from various locations to the standard structure
# This script finds images and organizes them properly

set -e

BASE_DIR="/l/users/muhra.almahri/Surgical_COT"
SOURCE_DIRS=(
    "${BASE_DIR}/EndoVis2018/EndoVis2018/test data"
    "${BASE_DIR}/EndoVis2018/data/images"
)
TARGET_DIR="${BASE_DIR}/EndoVis2018/data/images"

echo "=========================================="
echo "Organizing EndoVis2018 Images"
echo "=========================================="
echo "Target: ${TARGET_DIR}"
echo ""

# Create target structure
mkdir -p "${TARGET_DIR}"

# Find and copy images from source directories
for source_dir in "${SOURCE_DIRS[@]}"; do
    if [ -d "$source_dir" ]; then
        echo "Searching in: $source_dir"
        
        # Find all sequence directories
        find "$source_dir" -type d -name "seq_*" | while read seq_dir; do
            seq_name=$(basename "$seq_dir")
            left_frames_dir="${seq_dir}/left_frames"
            
            if [ -d "$left_frames_dir" ]; then
                # Count images
                img_count=$(find "$left_frames_dir" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
                
                if [ "$img_count" -gt 0 ]; then
                    echo "  Found ${img_count} images in ${seq_name}/left_frames"
                    
                    # Create target sequence directory
                    target_seq_dir="${TARGET_DIR}/${seq_name}"
                    mkdir -p "${target_seq_dir}/left_frames"
                    
                    # Copy images (use hard links to save space, or cp for actual copy)
                    find "$left_frames_dir" -name "*.png" -o -name "*.jpg" | while read img_file; do
                        img_name=$(basename "$img_file")
                        target_file="${target_seq_dir}/left_frames/${img_name}"
                        
                        if [ ! -f "$target_file" ]; then
                            cp "$img_file" "$target_file"
                        fi
                    done
                fi
            fi
        done
    fi
done

echo ""
echo "=========================================="
echo "Organization Complete"
echo "=========================================="

# Count total images
TOTAL_IMAGES=$(find "${TARGET_DIR}" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
echo "Total images organized: ${TOTAL_IMAGES}"

# List sequences
echo ""
echo "Sequences found:"
find "${TARGET_DIR}" -type d -name "seq_*" | while read seq_dir; do
    seq_name=$(basename "$seq_dir")
    img_count=$(find "${seq_dir}/left_frames" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
    echo "  ${seq_name}: ${img_count} images"
done

echo ""
echo "Images are now in: ${TARGET_DIR}"
echo "=========================================="




















