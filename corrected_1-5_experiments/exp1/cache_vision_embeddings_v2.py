#!/usr/bin/env python3
"""
Robust vision embedding caching with stable keys and atomic writes.
"""

import json
import torch
import os
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor
import argparse
import sys

# Add exp1 to path
sys.path.insert(0, str(Path(__file__).parent))
from cache_utils import cache_key, atomic_save_tensor, is_cached


def cache_vision_embeddings(model_name, jsonl_path, image_root, output_cache_dir, resolution_id="full"):
    """Pre-compute and cache vision embeddings with robust caching."""
    
    print(f"Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print("✅ Processor loaded")
    
    # Load samples
    samples = [json.loads(line) for line in open(jsonl_path)]
    print(f"Processing {len(samples)} samples...")
    print(f"Image root: {image_root}")
    print(f"Cache dir: {output_cache_dir}")
    print()
    
    output_cache_dir = Path(output_cache_dir)
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    
    cached_count = 0
    skipped_count = 0
    error_count = 0
    
    # Qwen3-VL processor requires text with images
    # Use minimal placeholder that gets replaced during training
    DEFAULT_TEXT = " "
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DEFAULT_TEXT}]}]
    dummy_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    
    for idx, sample in enumerate(tqdm(samples, desc=f"Caching {output_cache_dir.name}")):
        # Get image path
        img_file = sample.get('image') or sample.get('image_filename') or sample.get('image_id')
        if not img_file:
            error_count += 1
            continue
            
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_file = f"{img_file}.jpg"
        
        img_path = Path(image_root) / img_file
        img_path_abs = str(img_path.resolve()).replace("//", "/")
        
        # Generate stable cache key
        key = cache_key(img_path_abs, model_name, resolution_id)
        cache_file = output_cache_dir / f"{key}.pt"
        
        # Robust check: file exists AND has content
        if is_cached(str(cache_file)):
            skipped_count += 1
            continue
        
        try:
            # Verify image file exists
            if not os.path.exists(img_path_abs):
                if error_count < 10:  # Only print first 10
                    print(f"\n⚠️  Image not found: {img_path_abs}")
                error_count += 1
                continue
            
            # Load image
            img = Image.open(img_path_abs).convert("RGB")
            
            # Process with Qwen3-VL (needs text)
            inputs = processor(text=[dummy_text], images=[img], return_tensors="pt")
            
            # Extract vision tensors
            vision_data = {
                'pixel_values': inputs.get('pixel_values'),
                'image_grid_thw': inputs.get('image_grid_thw'),
                'image_path': img_path_abs,  # Store for debugging
            }
            
            # Validate we got the tensors
            if vision_data['pixel_values'] is None or vision_data['image_grid_thw'] is None:
                if error_count < 10:
                    print(f"\n⚠️  Missing tensors for: {img_file}")
                error_count += 1
                continue
            
            # Atomic save
            atomic_save_tensor(vision_data, str(cache_file))
            cached_count += 1
            
            if cached_count % 1000 == 0:
                print(f"\n  ✅ Cached {cached_count} new images (skipped {skipped_count}, errors {error_count})")
            
        except Exception as e:
            if error_count < 10:  # Only print first 10 errors
                print(f"\n⚠️  Error processing {img_file}: {e}")
            error_count += 1
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ Caching complete for {output_cache_dir.name}!")
    print(f"{'='*80}")
    print(f"Total samples in jsonl: {len(samples)}")
    print(f"Already cached (skipped): {skipped_count}")
    print(f"Newly cached: {cached_count}")
    print(f"Errors/missing: {error_count}")
    print(f"Cache directory: {output_cache_dir}")
    
    # Final verification
    actual_files = len(list(output_cache_dir.glob("*.pt")))
    print(f"Actual cache files on disk: {actual_files}")
    print(f"{'='*80}\n")
    
    return cached_count, skipped_count, error_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--cache_dir", default="vision_cache")
    parser.add_argument("--resolution_id", default="full", help="Resolution identifier for cache key")
    args = parser.parse_args()
    
    print("="*80)
    print("VISION EMBEDDING CACHING V2 (ROBUST)")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Resolution ID: {args.resolution_id}")
    print(f"Cache dir: {args.cache_dir}")
    print("="*80)
    print()
    
    # Cache training set
    print("▶ Caching training set...")
    cache_train_dir = Path(args.cache_dir) / "train"
    train_cached, train_skipped, train_errors = cache_vision_embeddings(
        args.model_name,
        args.train_jsonl,
        args.image_root,
        cache_train_dir,
        args.resolution_id
    )
    
    print()
    
    # Cache validation set
    print("▶ Caching validation set...")
    cache_val_dir = Path(args.cache_dir) / "val"
    val_cached, val_skipped, val_errors = cache_vision_embeddings(
        args.model_name,
        args.val_jsonl,
        args.image_root,
        cache_val_dir,
        args.resolution_id
    )
    
    print()
    print("="*80)
    print("✅ ALL CACHING COMPLETE!")
    print("="*80)
    print(f"Train: {train_cached} new, {train_skipped} skipped, {train_errors} errors")
    print(f"Val:   {val_cached} new, {val_skipped} skipped, {val_errors} errors")
    print()
    print("Next: Run training with cached embeddings")
    print(f"  sbatch exp1/slurm/train_exp1_category_based.slurm")
    print("="*80)






