#!/bin/bash
# Cache Cleanup and Management Script for LLaVA-Med Training
# This script cleans up old job-specific cache directories and sets up a shared cache

echo "========================================="
echo "HuggingFace Cache Cleanup & Setup"
echo "========================================="
echo ""

# Set shared cache location
SHARED_CACHE="/l/users/muhra.almahri/.cache/hf_shared"
OLD_CACHE_DIR="/l/users/muhra.almahri/.cache"

echo "Step 1: Checking current disk usage..."
du -sh $OLD_CACHE_DIR
echo ""

echo "Step 2: Finding old job-specific cache directories..."
echo "Old cache directories (job-specific):"
find $OLD_CACHE_DIR -maxdepth 1 -name "hf_cache_*" -type d 2>/dev/null | while read dir; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  $dir: $size"
    fi
done
echo ""

# Ask for confirmation before deletion
read -p "Do you want to delete old job-specific cache directories (hf_cache_*)? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting old job-specific cache directories..."
    find $OLD_CACHE_DIR -maxdepth 1 -name "hf_cache_*" -type d -exec rm -rf {} + 2>/dev/null
    echo "✓ Old cache directories deleted"
    echo ""
    
    echo "New disk usage:"
    du -sh $OLD_CACHE_DIR
    echo ""
else
    echo "Skipping deletion."
    echo ""
fi

echo "Step 3: Setting up shared cache directory..."
mkdir -p $SHARED_CACHE
echo "✓ Created shared cache: $SHARED_CACHE"
echo ""

echo "Step 4: Pre-downloading required models to shared cache..."
export HF_HOME=$SHARED_CACHE
export TRANSFORMERS_CACHE=$SHARED_CACHE/transformers
export HF_DATASETS_CACHE=$SHARED_CACHE/datasets
export HF_HUB_CACHE=$SHARED_CACHE

# Pre-download CLIP processor (the one causing issues)
echo "Downloading CLIP processor..."
python3 << 'EOF'
import os
os.environ["HF_HOME"] = "/l/users/muhra.almahri/.cache/hf_shared"
os.environ["TRANSFORMERS_CACHE"] = "/l/users/muhra.almahri/.cache/hf_shared/transformers"
os.environ["HF_HUB_CACHE"] = "/l/users/muhra.almahri/.cache/hf_shared"

from transformers import CLIPImageProcessor

print("  Downloading openai/clip-vit-large-patch14...")
try:
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # Set to 336x336 for LLaVA-Med compatibility
    processor.size = {"height": 336, "width": 336}
    processor.crop_size = {"height": 336, "width": 336}
    print("  ✓ CLIP processor downloaded successfully")
except Exception as e:
    print(f"  ✗ Failed to download: {e}")
    print("  You may need to free up more space first")
EOF

echo ""

echo "Step 5: Pre-downloading LLaVA-Med model components..."
python3 << 'EOF'
import os
os.environ["HF_HOME"] = "/l/users/muhra.almahri/.cache/hf_shared"
os.environ["TRANSFORMERS_CACHE"] = "/l/users/muhra.almahri/.cache/hf_shared/transformers"
os.environ["HF_HUB_CACHE"] = "/l/users/muhra.almahri/.cache/hf_shared"

from transformers import AutoConfig, AutoTokenizer

model_name = "microsoft/llava-med-v1.5-mistral-7b"

print("  Downloading LLaVA-Med config...")
try:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print("  ✓ Config downloaded")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print("  Downloading LLaVA-Med tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    print("  ✓ Tokenizer downloaded")
except Exception as e:
    print(f"  ✗ Failed: {e}")
EOF

echo ""

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""

echo "Shared cache location: $SHARED_CACHE"
echo "Current cache size:"
du -sh $SHARED_CACHE 2>/dev/null || echo "  (cache not yet created)"
echo ""

echo "To use this cache in your jobs, your SLURM scripts should set:"
echo "  export HF_HOME=$SHARED_CACHE"
echo "  export TRANSFORMERS_CACHE=$SHARED_CACHE/transformers"
echo "  export HF_DATASETS_CACHE=$SHARED_CACHE/datasets"
echo "  export HF_HUB_CACHE=$SHARED_CACHE"
echo ""





