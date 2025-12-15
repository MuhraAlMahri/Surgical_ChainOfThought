#!/usr/bin/env python3
"""
Create a subset of the Kvasir ULTRA_CONDENSED dataset for Google Colab experiments
"""
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

def create_subset(
    train_file,
    val_file, 
    test_file,
    image_dir,
    output_dir,
    subset_size=500,  # Number of training samples
    seed=42
):
    """
    Create a subset of the dataset with balanced categories
    
    Args:
        train_file: Path to train.jsonl
        val_file: Path to val.jsonl  
        test_file: Path to test.jsonl
        image_dir: Path to images directory
        output_dir: Output directory for subset
        subset_size: Number of training samples to include
        seed: Random seed
    """
    random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"Creating subset with {subset_size} training samples")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    with open(train_file) as f:
        train_data = [json.loads(line) for line in f]
    with open(val_file) as f:
        val_data = [json.loads(line) for line in f]
    with open(test_file) as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"Original train: {len(train_data)} samples")
    print(f"Original val: {len(val_data)} samples")
    print(f"Original test: {len(test_data)} samples")
    
    # Group train data by image_id to keep all questions for selected images
    print("\n📊 Grouping by image...")
    image_to_samples = defaultdict(list)
    for sample in train_data:
        img_id = sample.get('image_id', sample.get('image_filename', ''))
        image_to_samples[img_id].append(sample)
    
    unique_images = list(image_to_samples.keys())
    print(f"Unique images: {len(unique_images)}")
    
    # Calculate how many images we need to get ~subset_size samples
    avg_questions_per_image = len(train_data) / len(unique_images)
    target_images = int(subset_size / avg_questions_per_image)
    
    print(f"Avg questions per image: {avg_questions_per_image:.1f}")
    print(f"Target images: {target_images}")
    
    # Sample images randomly
    selected_images = random.sample(unique_images, min(target_images, len(unique_images)))
    
    # Get all samples for selected images
    train_subset = []
    for img_id in selected_images:
        train_subset.extend(image_to_samples[img_id])
    
    print(f"\n✓ Selected {len(selected_images)} images")
    print(f"✓ Total training samples: {len(train_subset)}")
    
    # For val and test, also sample proportionally
    val_size = int(len(val_data) * (len(train_subset) / len(train_data)))
    test_size = int(len(test_data) * (len(train_subset) / len(train_data)))
    
    val_subset = random.sample(val_data, min(val_size, len(val_data)))
    test_subset = random.sample(test_data, min(test_size, len(test_data)))
    
    print(f"✓ Val samples: {len(val_subset)}")
    print(f"✓ Test samples: {len(test_subset)}")
    
    # Collect all unique images needed
    all_images = set()
    for sample in train_subset + val_subset + test_subset:
        img_filename = sample.get('image_filename', sample.get('image_id', ''))
        if not img_filename.endswith('.jpg'):
            img_filename += '.jpg'
        all_images.add(img_filename)
    
    print(f"\n📷 Total unique images needed: {len(all_images)}")
    
    # Save subset files
    print("\n💾 Saving subset files...")
    
    with open(output_dir / "train_subset.jsonl", 'w') as f:
        for sample in train_subset:
            f.write(json.dumps(sample) + '\n')
    print(f"✓ Saved train_subset.jsonl")
    
    with open(output_dir / "val_subset.jsonl", 'w') as f:
        for sample in val_subset:
            f.write(json.dumps(sample) + '\n')
    print(f"✓ Saved val_subset.jsonl")
    
    with open(output_dir / "test_subset.jsonl", 'w') as f:
        for sample in test_subset:
            f.write(json.dumps(sample) + '\n')
    print(f"✓ Saved test_subset.jsonl")
    
    # Copy images
    print(f"\n🖼️  Copying {len(all_images)} images...")
    image_dir = Path(image_dir)
    images_output = output_dir / "images"
    images_output.mkdir(exist_ok=True)
    
    copied = 0
    missing = 0
    for img_file in all_images:
        src = image_dir / img_file
        if src.exists():
            dst = images_output / img_file
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing += 1
            print(f"⚠️  Missing: {img_file}")
    
    print(f"\n✓ Copied {copied} images")
    if missing > 0:
        print(f"⚠️  Missing {missing} images")
    
    # Create summary
    summary = {
        "train_samples": len(train_subset),
        "val_samples": len(val_subset),
        "test_samples": len(test_subset),
        "total_samples": len(train_subset) + len(val_subset) + len(test_subset),
        "unique_images": len(all_images),
        "images_copied": copied,
        "images_missing": missing,
        "subset_size_requested": subset_size,
        "seed": seed
    }
    
    with open(output_dir / "subset_info.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ SUBSET CREATION COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    print("  - train_subset.jsonl")
    print("  - val_subset.jsonl")
    print("  - test_subset.jsonl")
    print("  - images/ (directory)")
    print("  - subset_info.json")
    print("\n📦 To upload to Colab:")
    print(f"  cd {output_dir}")
    print(f"  zip -r colab_subset.zip .")
    print("  # Upload colab_subset.zip to Google Colab")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create subset of Kvasir ULTRA_CONDENSED dataset")
    parser.add_argument("--train", required=True, help="Path to train.jsonl")
    parser.add_argument("--val", required=True, help="Path to val.jsonl")
    parser.add_argument("--test", required=True, help="Path to test.jsonl")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=500, help="Target number of training samples (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    create_subset(
        train_file=args.train,
        val_file=args.val,
        test_file=args.test,
        image_dir=args.images,
        output_dir=args.output,
        subset_size=args.size,
        seed=args.seed
    )




