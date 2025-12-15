#!/usr/bin/env python3
"""
Cache vision tower embeddings for faster training.
Since vision tower is frozen, we only need to compute embeddings once!
This can give 2-5x speedup.
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import argparse

def cache_vision_embeddings(model_name, jsonl_path, image_root, output_cache_dir):
    """Pre-compute and cache vision embeddings."""
    
    print(f"Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # We don't actually need the full model - just the processor to get vision tensors!
    # The vision tower processes images during the processor() call
    print("✅ Processor loaded (no need to load full model)")
    
    # Load samples
    samples = [json.loads(line) for line in open(jsonl_path)]
    print(f"Processing {len(samples)} samples...")
    
    output_cache_dir = Path(output_cache_dir)
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    
    cached_count = 0
    skipped_count = 0
    
    for idx, sample in enumerate(tqdm(samples, desc="Caching vision")):
        # Get image path
        img_file = sample.get('image') or sample.get('image_filename') or sample.get('image_id')
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_file = f"{img_file}.jpg"
        
        img_path = Path(image_root) / img_file
        
        # Check if already cached
        cache_file = output_cache_dir / f"{Path(img_file).stem}.pt"
        if cache_file.exists():
            skipped_count += 1
            if skipped_count % 1000 == 0:
                print(f"  Skipped {skipped_count} already cached images...")
            continue
        
        try:
            # Load and process image
            img = Image.open(str(img_path).replace("//", "/")).convert("RGB")
            
            # Qwen3-VL processor requires text to be provided with images
            # Use minimal dummy text just to get vision tensors
            conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "dummy"}]}]
            dummy_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            
            # Process image with text (processor requirement)
            inputs = processor(text=[dummy_text], images=[img], return_tensors="pt")
            
            # Save the vision-related tensors
            # These are exactly what the model needs for the image
            vision_cache = {
                'pixel_values': inputs.get('pixel_values'),
                'image_grid_thw': inputs.get('image_grid_thw'),
            }
            
            # Validate that we got the tensors
            if vision_cache['pixel_values'] is None or vision_cache['image_grid_thw'] is None:
                print(f"\n⚠️  Warning: Missing tensors for {img_file}")
                continue
            
            # Save to disk (keep on CPU to save memory)
            torch.save(vision_cache, cache_file)
            cached_count += 1
            
            if cached_count % 1000 == 0:
                print(f"  Cached {cached_count} images...")
            
        except Exception as e:
            print(f"\n⚠️  Error processing {img_file}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ Caching complete!")
    print(f"{'='*80}")
    print(f"Total samples: {len(samples)}")
    print(f"Already cached (skipped): {skipped_count}")
    print(f"Newly cached: {cached_count}")
    print(f"Cache directory: {output_cache_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--cache_dir", default="vision_cache")
    args = parser.parse_args()
    
    print("="*80)
    print("VISION EMBEDDING CACHING")
    print("="*80)
    print()
    
    # Cache training set
    print("Caching training set...")
    cache_vision_embeddings(
        args.model_name,
        args.train_jsonl,
        args.image_root,
        f"{args.cache_dir}/train"
    )
    
    print()
    
    # Cache validation set
    print("Caching validation set...")
    cache_vision_embeddings(
        args.model_name,
        args.val_jsonl,
        args.image_root,
        f"{args.cache_dir}/val"
    )
    
    print()
    print("="*80)
    print("✅ ALL EMBEDDINGS CACHED!")
    print("="*80)
    print()
    print("Next: Update your training script to load from cache instead of processing images.")

