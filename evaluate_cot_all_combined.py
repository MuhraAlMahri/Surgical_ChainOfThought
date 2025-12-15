#!/usr/bin/env python3
"""
Combined evaluation script for all CoT configurations.
Runs all model-dataset combinations in a single job using 2 GPUs.
"""

import torch
import json
import os
from pathlib import Path
from typing import Dict
import logging
import argparse
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_evaluation(
    model_type: str,
    dataset: str,
    test_data: str,
    image_base_path: str,
    cot_checkpoint: str,
    baseline_results: str,
    output_dir: str,
    gpu_id: int
):
    """Run evaluation for a single model-dataset combination."""
    logger.info(f"Starting evaluation: {model_type} on {dataset} (GPU {gpu_id})")
    
    # Run as subprocess to ensure clean GPU isolation
    cmd = [
        sys.executable, 'evaluate_cot_only.py',
        '--model-type', model_type,
        '--dataset', dataset,
        '--test-data', test_data,
        '--image-base-path', image_base_path,
        '--output', output_dir,
        '--baseline-results', baseline_results
    ]
    
    if cot_checkpoint:
        cmd.extend(['--cot-checkpoint', cot_checkpoint])
    
    try:
        # Pass environment variables (especially HF_TOKEN) to subprocess
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), env=env)
        if result.returncode == 0:
            logger.info(f"✅ Completed: {model_type} on {dataset}")
            return True
        else:
            logger.error(f"❌ Failed: {model_type} on {dataset}")
            logger.error(f"Error output: {result.stderr[:500]}")
            if result.stdout:
                logger.error(f"Stdout: {result.stdout[:500]}")
            return False
    except Exception as e:
        logger.error(f"❌ Exception: {model_type} on {dataset}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all CoT evaluations in parallel")
    parser.add_argument("--baseline-results", default="results/baseline_results.json")
    parser.add_argument("--output-dir", default="results/cot_evaluation")
    parser.add_argument("--gpu-0", type=int, default=0, help="First GPU ID")
    parser.add_argument("--gpu-1", type=int, default=1, help="Second GPU ID")
    
    args = parser.parse_args()
    
    # All combinations
    combinations = [
        {
            'model_type': 'qwen3vl',
            'dataset': 'kvasir',
            'test_data': 'datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json',
            'image_path': 'datasets/Kvasir-VQA/raw/images',
            'cot_checkpoint': 'results/multihead_cot/qwen3vl_kvasir_cot_20251208_233609/checkpoint_epoch_3.pt'
        },
        {
            'model_type': 'qwen3vl',
            'dataset': 'endovis',
            'test_data': 'corrected_1-5_experiments/datasets/endovis2018_vqa/test.jsonl',
            'image_path': 'datasets/EndoVis2018/raw/images',
            'cot_checkpoint': None
        },
        {
            'model_type': 'medgemma',
            'dataset': 'kvasir',
            'test_data': 'datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json',
            'image_path': 'datasets/Kvasir-VQA/raw/images',
            'cot_checkpoint': None
        },
        {
            'model_type': 'medgemma',
            'dataset': 'endovis',
            'test_data': 'corrected_1-5_experiments/datasets/endovis2018_vqa/test.jsonl',
            'image_path': 'datasets/EndoVis2018/raw/images',
            'cot_checkpoint': None
        },
        {
            'model_type': 'llava_med',
            'dataset': 'kvasir',
            'test_data': 'datasets/Kvasir-VQA/raw/metadata/raw_complete_metadata.json',
            'image_path': 'datasets/Kvasir-VQA/raw/images',
            'cot_checkpoint': None
        },
        {
            'model_type': 'llava_med',
            'dataset': 'endovis',
            'test_data': 'corrected_1-5_experiments/datasets/endovis2018_vqa/test.jsonl',
            'image_path': 'datasets/EndoVis2018/raw/images',
            'cot_checkpoint': None
        }
    ]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluations sequentially but on different GPUs
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
        result1 = run_evaluation(
            combo1['model_type'],
            combo1['dataset'],
            combo1['test_data'],
            combo1['image_path'],
            combo1['cot_checkpoint'],
            args.baseline_results,
            str(output_dir),
            args.gpu_0
        )
        results[f"{combo1['model_type']}_{combo1['dataset']}"] = result1
        
        # Run second on GPU 1 if available
        if combo2:
            logger.info(f"GPU {args.gpu_1}: {combo2['model_type']} on {combo2['dataset']}")
            result2 = run_evaluation(
                combo2['model_type'],
                combo2['dataset'],
                combo2['test_data'],
                combo2['image_path'],
                combo2['cot_checkpoint'],
                args.baseline_results,
                str(output_dir),
                args.gpu_1
            )
            results[f"{combo2['model_type']}_{combo2['dataset']}"] = result2
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    for key, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{key}: {status}")
    print("="*80)


if __name__ == "__main__":
    main()

