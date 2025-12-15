#!/usr/bin/env python3
"""
Test different image resolutions to find optimal speed/accuracy balance.
Runs quick training on subset and evaluates quality.
"""

import argparse
import yaml
import os
import sys
import json
import time
from pathlib import Path

# Add exp1 to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForImageTextToText, TrainingArguments, Trainer, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from dataset import collate
from dataset_resolution_test import VQASFTDatasetResTest
import torch


def test_resolution(cfg_path, min_pixels, max_pixels, test_name, max_steps=50):
    """
    Test a specific resolution configuration.
    
    Args:
        cfg_path: Path to config file
        min_pixels: Min pixels for Qwen2-VL image processor
        max_pixels: Max pixels for Qwen2-VL image processor
        test_name: Name for this test (e.g., "low_res", "medium_res")
        max_steps: Number of training steps to test (default 50)
    """
    print(f"\n{'='*80}")
    print(f"Testing: {test_name}")
    print(f"Min pixels: {min_pixels:,} ({int(min_pixels**0.5)}x{int(min_pixels**0.5)})")
    print(f"Max pixels: {max_pixels:,} ({int(max_pixels**0.5)}x{int(max_pixels**0.5)})")
    print(f"{'='*80}\n")
    
    cfg = yaml.safe_load(open(cfg_path))
    model_name = cfg["model_name"]
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent
    train_jsonl = base_dir / cfg["data"]["train_jsonl"]
    val_jsonl = base_dir / cfg["data"]["val_jsonl"]
    image_root = cfg["data"]["image_root"]
    
    # Load model
    print("Loading model...")
    start_load = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, 
        trust_remote_code=True,
        device_map="auto"
    )
    
    if cfg["train"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    
    if cfg.get("vision_frozen", True):
        for n, p in model.named_parameters():
            if "vision_tower" in n or "visual" in n:
                p.requires_grad = False
    
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    
    # Create dataset with custom max_seq_len based on expected tokens
    # Estimate: text_tokens (~250) + image_tokens (varies by resolution)
    # Qwen2-VL uses dynamic patching, roughly: tokens = (pixels / 2.6) / 196
    estimated_image_tokens = int(max_pixels / 600)  # More accurate estimate
    max_seq_len = 250 + estimated_image_tokens + 200  # Text + image + margin
    max_seq_len = min(max_seq_len, 4096)  # Cap at 4096
    
    print(f"Estimated image tokens: ~{estimated_image_tokens}")
    print(f"Using max_seq_len: {max_seq_len}")
    
    train_ds = VQASFTDatasetResTest(
        str(train_jsonl),
        image_root,
        model_name,
        max_seq_len,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    
    val_ds = VQASFTDatasetResTest(
        str(val_jsonl),
        image_root,
        model_name,
        max_seq_len,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    
    # Create small subset for testing
    print(f"Creating test subset (100 train, 50 val samples)...")
    train_subset = torch.utils.data.Subset(train_ds, range(min(100, len(train_ds))))
    val_subset = torch.utils.data.Subset(val_ds, range(min(50, len(val_ds))))
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / "resolution_tests" / test_name
    
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Smaller for quick test
        learning_rate=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        max_steps=max_steps,  # Just test 50 steps
        warmup_steps=5,
        bf16=cfg["train"]["bf16"],
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        eval_strategy="steps",
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        report_to="none",
        save_total_limit=1
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_subset,
        eval_dataset=val_subset,
        data_collator=collate
    )
    
    # Time the training
    print(f"\nStarting training for {max_steps} steps...")
    start_train = time.time()
    
    try:
        trainer.train()
        train_time = time.time() - start_train
        
        # Get final metrics
        metrics = trainer.evaluate()
        
        # Calculate average time per step
        avg_time_per_step = train_time / max_steps
        
        # Estimate full training time
        total_steps = 7704  # Your full training
        estimated_full_time_hours = (total_steps * avg_time_per_step) / 3600
        
        results = {
            "test_name": test_name,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "max_seq_len": max_seq_len,
            "train_time_seconds": train_time,
            "avg_time_per_step": avg_time_per_step,
            "estimated_full_training_hours": estimated_full_time_hours,
            "eval_loss": metrics.get("eval_loss", None),
            "train_samples": len(train_subset),
            "eval_samples": len(val_subset),
            "max_steps_tested": max_steps
        }
        
        # Save results
        results_file = script_dir / "resolution_tests" / f"{test_name}_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"✅ {test_name} COMPLETED")
        print(f"{'='*80}")
        print(f"Time per step: {avg_time_per_step:.2f}s")
        print(f"Estimated full training: {estimated_full_time_hours:.1f} hours")
        print(f"Eval loss: {metrics.get('eval_loss', 'N/A')}")
        print(f"Results saved to: {results_file}")
        print(f"{'='*80}\n")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Critical: Clean up GPU memory
        print("\nCleaning up GPU memory...")
        del model
        del trainer
        del train_ds
        del val_ds
        del train_subset
        del val_subset
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ GPU memory cleaned")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="exp1/config_exp1_category_based.yaml")
    parser.add_argument("--max_steps", type=int, default=50, help="Steps to test per resolution")
    args = parser.parse_args()
    
    # Define resolutions to test
    # Format: (name, min_pixels, max_pixels, description)
    resolutions = [
        ("low_res_448", 200704, 200704, "448x448 (like previous experiments)"),
        ("medium_res_768", 589824, 589824, "768x768 (balanced)"),
        ("high_res_1024", 1048576, 1048576, "1024x1024 (high quality)"),
        ("full_res_current", 3136, 12845056, "Full adaptive (current, ~1036x1288)"),
    ]
    
    print("="*80)
    print("RESOLUTION COMPARISON TEST")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Steps per test: {args.max_steps}")
    print(f"Tests to run: {len(resolutions)}")
    print("="*80)
    
    all_results = []
    
    for name, min_pix, max_pix, desc in resolutions:
        print(f"\n\n{'#'*80}")
        print(f"TEST: {desc}")
        print(f"{'#'*80}")
        
        result = test_resolution(
            args.config,
            min_pix,
            max_pix,
            name,
            max_steps=args.max_steps
        )
        
        if result:
            all_results.append(result)
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
    
    # Create comparison summary
    if all_results:
        summary_file = Path(__file__).parent / "resolution_tests" / "comparison_summary.json"
        with open(summary_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "="*80)
        print("FINAL COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Test Name':<25} {'Time/Step':<12} {'Est. Full Train':<18} {'Eval Loss':<12}")
        print("-"*80)
        
        for r in all_results:
            print(f"{r['test_name']:<25} {r['avg_time_per_step']:>10.2f}s  "
                  f"{r['estimated_full_training_hours']:>14.1f}h  "
                  f"{r['eval_loss']:>10.4f}")
        
        print("="*80)
        print(f"Full results saved to: {summary_file}")
        print("="*80)


if __name__ == "__main__":
    main()

