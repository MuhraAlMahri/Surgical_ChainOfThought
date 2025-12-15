#!/usr/bin/env python3
"""
Combined training script for all multi-head CoT configurations.
Runs all model-dataset combinations in a single job using 2 GPUs.
"""

import torch
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict
import logging
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training(
    model_type: str,
    dataset: str,
    base_checkpoint: str,
    data_path: str,
    image_base_path: str,
    output_dir: str,
    gpu_id: int
):
    """Run training for a single model-dataset combination."""
    logger.info(f"Starting training: {model_type} on {dataset} (GPU {gpu_id})")
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Create output directory for this combination
    combo_output_dir = Path(output_dir) / f"{model_type}_{dataset}_cot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    combo_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training as subprocess
    cmd = [
        sys.executable, 'train_multihead_cot.py',
        '--model_type', model_type,
        '--dataset', dataset,
        '--base_checkpoint', base_checkpoint,
        '--data_path', data_path,
        '--image_base_path', image_base_path,
        '--output_dir', str(combo_output_dir),
        '--learning_rate', '5e-5',
        '--epochs', '5',
        '--batch_size', '1',
        '--grad_accum', '16',
        '--bf16',
        '--gradient_checkpointing',
        '--lora_r', '4',
        '--lora_alpha', '8',
        '--weight_decay', '0.01'
    ]
    
    # Add question categories if available
    question_categories = "results/multihead_cot/question_categories.json"
    if Path(question_categories).exists():
        cmd.extend(['--question_categories', question_categories])
    
    try:
        # Pass environment variables (especially HF_TOKEN) to subprocess
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Debug: log the command being run
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), env=env)
        if result.returncode == 0:
            logger.info(f"✅ Completed: {model_type} on {dataset}")
            return True, combo_output_dir
        else:
            logger.error(f"❌ Failed: {model_type} on {dataset}")
            logger.error(f"Error output: {result.stderr[:500]}")
            if result.stdout:
                logger.error(f"Stdout: {result.stdout[:500]}")
            return False, combo_output_dir
    except Exception as e:
        logger.error(f"❌ Exception: {model_type} on {dataset}: {e}")
        return False, combo_output_dir


def main():
    parser = argparse.ArgumentParser(description="Run all CoT training in parallel on 2 GPUs")
    parser.add_argument("--output-dir", default="results/multihead_cot/all_combined")
    parser.add_argument("--gpu-0", type=int, default=0, help="First GPU ID")
    parser.add_argument("--gpu-1", type=int, default=1, help="Second GPU ID")
    
    args = parser.parse_args()
    
    # All combinations
    combinations = [
        {
            'model_type': 'qwen3vl',
            'dataset': 'kvasir',
            'base_checkpoint': 'Qwen/Qwen3-VL-8B-Instruct',
            'data_path': 'datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json',
            'image_path': 'datasets/Kvasir-VQA/raw/images'
        },
        {
            'model_type': 'qwen3vl',
            'dataset': 'endovis',
            'base_checkpoint': 'Qwen/Qwen3-VL-8B-Instruct',
            'data_path': 'corrected_1-5_experiments/datasets/endovis2018_vqa/train.jsonl',
            'image_path': 'datasets/EndoVis2018/raw/images'
        },
        {
            'model_type': 'medgemma',
            'dataset': 'kvasir',
            'base_checkpoint': 'google/medgemma-4b',
            'data_path': 'datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json',
            'image_path': 'datasets/Kvasir-VQA/raw/images'
        },
        {
            'model_type': 'medgemma',
            'dataset': 'endovis',
            'base_checkpoint': 'google/medgemma-4b',
            'data_path': 'corrected_1-5_experiments/datasets/endovis2018_vqa/train.jsonl',
            'image_path': 'datasets/EndoVis2018/raw/images'
        },
        {
            'model_type': 'llava_med',
            'dataset': 'kvasir',
            'base_checkpoint': 'corrected_1-5_experiments/qlora_experiments/models/llava_med_kvasir_instruction/best_model',
            'data_path': 'datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json',
            'image_path': 'datasets/Kvasir-VQA/raw/images'
        },
        {
            'model_type': 'llava_med',
            'dataset': 'endovis',
            'base_checkpoint': 'corrected_1-5_experiments/qlora_experiments/models/llava_med_endovis_instruction/best_model',
            'data_path': 'corrected_1-5_experiments/datasets/endovis2018_vqa/train.jsonl',
            'image_path': 'datasets/EndoVis2018/raw/images'
        }
    ]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run trainings sequentially but on different GPUs
    # Process combinations in pairs (one per GPU)
    results = {}
    
    for i in range(0, len(combinations), 2):
        combo1 = combinations[i]
        combo2 = combinations[i+1] if i+1 < len(combinations) else None
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing batch {i//2 + 1}: {combo1['model_type']}_{combo1['dataset']}" + 
                   (f" and {combo2['model_type']}_{combo2['dataset']}" if combo2 else ""))
        logger.info(f"{'='*80}\n")
        
        # Run first on GPU 0
        logger.info(f"GPU {args.gpu_0}: {combo1['model_type']} on {combo1['dataset']}")
        success1, output_dir1 = run_training(
            combo1['model_type'],
            combo1['dataset'],
            combo1['base_checkpoint'],
            combo1['data_path'],
            combo1['image_path'],
            str(output_dir),
            args.gpu_0
        )
        results[f"{combo1['model_type']}_{combo1['dataset']}"] = {
            'success': success1,
            'output_dir': str(output_dir1)
        }
        
        # Run second on GPU 1 if available
        if combo2:
            logger.info(f"GPU {args.gpu_1}: {combo2['model_type']} on {combo2['dataset']}")
            success2, output_dir2 = run_training(
                combo2['model_type'],
                combo2['dataset'],
                combo2['base_checkpoint'],
                combo2['data_path'],
                combo2['image_path'],
                str(output_dir),
                args.gpu_1
            )
            results[f"{combo2['model_type']}_{combo2['dataset']}"] = {
                'success': success2,
                'output_dir': str(output_dir2)
            }
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    for key, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{key}: {status}")
        if result['success']:
            print(f"  Output: {result['output_dir']}")
    print("="*80)
    
    # Save results summary
    summary_file = output_dir / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved training summary to {summary_file}")


if __name__ == "__main__":
    main()

