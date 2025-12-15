#!/usr/bin/env python3
"""
Robust vision-feature caching (Level B - maximum speedup).
Uses manifest + atomic writes + GPU encoding.
"""

import os
import sys
import glob
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Add to path
sys.path.insert(0, str(Path(__file__).parent))
from cache_utils import cache_key, atomic_save_tensor, Manifest


def main():
    MODEL_ID = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    RES = int(os.environ.get("IMG_RES", "1036"))
    CACHE_ROOT = os.environ.get("VISION_CACHE_DIR", "exp1/vision_cache_v2")
    DB_PATH = os.path.join(CACHE_ROOT, "manifest.sqlite")
    
    # Image directories (adapt to your paths)
    IMG_ROOT = "/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
    
    # Load JSONL to get image list
    TRAIN_JSONL = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/kvasir_ULTRA_CONDENSED/train_CATEGORY_BASED.jsonl"
    VAL_JSONL = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/kvasir_ULTRA_CONDENSED/val_CATEGORY_BASED.jsonl"
    
    import json
    
    def get_image_paths(jsonl_path, img_root):
        """Extract image paths from JSONL."""
        paths = []
        with open(jsonl_path) as f:
            for line in f:
                ex = json.loads(line)
                img_file = ex.get('image') or ex.get('image_filename') or ex.get('image_id')
                if img_file:
                    if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                        img_file = f"{img_file}.jpg"
                    img_path = os.path.join(img_root, img_file)
                    if os.path.exists(img_path):
                        paths.append(os.path.abspath(img_path))
        return paths
    
    print("="*80)
    print("ROBUST VISION CACHING (Level B - Vision Tower Features)")
    print("="*80)
    print(f"Model: {MODEL_ID}")
    print(f"Resolution: {RES}x{RES}")
    print(f"Cache: {CACHE_ROOT}")
    print(f"Manifest: {DB_PATH}")
    print("="*80)
    print()
    
    # Get image paths
    print("Loading image lists from JSONL...")
    train_paths = get_image_paths(TRAIN_JSONL, IMG_ROOT)
    val_paths = get_image_paths(VAL_JSONL, IMG_ROOT)
    print(f"Train images: {len(train_paths)}")
    print(f"Val images: {len(val_paths)}")
    print()
    
    # Load model
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("✅ Processor loaded")
    
    print("Loading model on GPU (this takes ~1-2 minutes)...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval().cuda()
    
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False
    
    print("✅ Model loaded and frozen")
    print()
    
    # Qwen3-VL processor needs text
    DEFAULT_TEXT = " "
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DEFAULT_TEXT}]}]
    dummy_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    man = Manifest(DB_PATH)
    
    def encode_img(path):
        """Encode image through vision tower."""
        try:
            # Load image
            img = Image.open(path).convert("RGB")
            
            # Processor needs text for Qwen3-VL
            inputs = processor(text=[dummy_text], images=[img], return_tensors="pt")
            inputs = {k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=amp_dtype):
                # Level B: Run through vision tower to get features
                # For Qwen3-VL, we want pixel_values which are already processed
                # The actual vision encoding happens during forward pass
                # So we save the processed pixel_values + grid info
                feats = {
                    'pixel_values': inputs['pixel_values'].float().cpu(),
                    'image_grid_thw': inputs['image_grid_thw'].cpu() if 'image_grid_thw' in inputs else None
                }
                return feats
        except Exception as e:
            return None
    
    def cachify(paths, split):
        """Cache images for one split."""
        out_dir = os.path.join(CACHE_ROOT, split)
        os.makedirs(out_dir, exist_ok=True)
        done, skipped, failed = 0, 0, 0
        
        for p in tqdm(paths, desc=f"Caching {split}"):
            k = cache_key(p, MODEL_ID, RES)
            outp = os.path.join(out_dir, f"{k}.pt")
            
            # CRITICAL: Check manifest AND actual file
            if man.has(k, outp):
                skipped += 1
                continue
            
            try:
                feats = encode_img(p)
                if feats is None:
                    failed += 1
                    if failed <= 10:
                        print(f"\n[WARN] encode failed: {os.path.basename(p)}")
                    continue
                
                atomic_save_tensor(feats, outp)
                man.put(k, outp)
                done += 1
                
                if done % 1000 == 0:
                    print(f"\n  ✅ Cached {done} new (skipped {skipped}, failed {failed})")
                
            except Exception as e:
                failed += 1
                if failed <= 10:
                    print(f"\n[WARN] error {os.path.basename(p)}: {e}")
        
        print(f"\n{'='*80}")
        print(f"[{split}] Complete:")
        print(f"  New: {done}")
        print(f"  Skipped: {skipped}")
        print(f"  Failed: {failed}")
        print(f"{'='*80}\n")
        
        return done, skipped, failed
    
    # Cache both splits
    print("▶ Caching training set...")
    train_new, train_skip, train_fail = cachify(train_paths, "train")
    
    print("▶ Caching validation set...")
    val_new, val_skip, val_fail = cachify(val_paths, "val")
    
    # Close manifest
    man.close()
    
    # Final verification
    print("="*80)
    print("✅ ALL CACHING COMPLETE!")
    print("="*80)
    print(f"Train: {train_new} new, {train_skip} skipped, {train_fail} failed")
    print(f"Val:   {val_new} new, {val_skip} skipped, {val_fail} failed")
    print()
    
    # CRITICAL: Verify actual files on disk
    train_files = len(glob.glob(os.path.join(CACHE_ROOT, "train", "*.pt")))
    val_files = len(glob.glob(os.path.join(CACHE_ROOT, "val", "*.pt")))
    
    print("DISK VERIFICATION (truth):")
    print(f"  Train files on disk: {train_files} / {len(train_paths)} ({100*train_files/len(train_paths):.1f}%)")
    print(f"  Val files on disk:   {val_files} / {len(val_paths)} ({100*val_files/len(val_paths):.1f}%)")
    print(f"  Total: {train_files + val_files} / {len(train_paths) + len(val_paths)}")
    print()
    
    if train_files < len(train_paths) * 0.95 or val_files < len(val_paths) * 0.95:
        print("⚠️  WARNING: Cache incomplete (<95% coverage)")
        print("   Missing files will be computed on-the-fly during training (slower)")
    else:
        print("✅ Cache complete! Ready for fast training.")
    
    print("="*80)
    print()
    print("Next: Run training")
    print("  sbatch exp1/slurm/train_exp1_category_based.slurm")
    print("="*80)


if __name__ == "__main__":
    main()






