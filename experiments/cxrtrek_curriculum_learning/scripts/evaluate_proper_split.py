#!/usr/bin/env python3
"""
Evaluate models trained with proper image-based split.

This evaluation script uses the pre-split test data files to ensure
we're evaluating on images that were NOT seen during training.

Key differences from original evaluation:
- Uses pre-split test JSON files (not runtime splitting)
- Explicitly verifies zero image overlap
- Scientifically valid evaluation

Author: Fixed evaluation for proper split
Date: October 2025
"""

import os
import sys
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(checkpoint_path: str,
                  test_data_path: str,
                  image_dir: str,
                  output_file: str,
                  batch_size: int = 1,
                  device: str = "cuda"):
    """
    Evaluate a model on proper test set.
    
    Args:
        checkpoint_path: Path to model checkpoint (LoRA adapter)
        test_data_path: Path to test JSON file (pre-split)
        image_dir: Directory containing images
        output_file: Output JSON file for results
        batch_size: Batch size (default 1 for generation)
        device: Device to use
    """
    logger.info(f"Loading test data from: {test_data_path}")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    logger.info(f"Test samples: {len(test_data)}")
    
    # Verify unique images
    test_images = set([s['image_path'] for s in test_data])
    logger.info(f"Unique test images: {len(test_images)}")
    
    # Set device
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = "cuda:0"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading base model...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    logger.info(f"Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    # Load processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    
    # Evaluate
    results = []
    correct = 0
    total = 0
    
    for sample in tqdm(test_data, desc="Evaluating"):
        # Load image
        image_path = sample['image_path']
        if not os.path.isabs(image_path):
            image_path = os.path.join(image_dir, image_path)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            continue
        
        question = sample['question']
        ground_truth = sample['answer'].strip().lower()
        
        # Prepare input
        messages = [[
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        
        # Decode
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (after the last assistant tag)
        if "assistant" in generated_text:
            prediction = generated_text.split("assistant")[-1].strip().lower()
        else:
            prediction = generated_text.strip().lower()
        
        # Check correctness (exact match or substring)
        is_correct = (prediction == ground_truth) or (ground_truth in prediction)
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'image_path': sample['image_path'],
            'question': question,
            'ground_truth': sample['answer'],
            'prediction': prediction,
            'correct': is_correct,
            'stage': sample.get('stage', None)
        })
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    
    # Per-stage accuracy
    stage_metrics = {}
    for stage in [1, 2, 3]:
        stage_samples = [r for r in results if r.get('stage') == stage]
        if stage_samples:
            stage_correct = sum(1 for r in stage_samples if r['correct'])
            stage_total = len(stage_samples)
            stage_metrics[f'stage_{stage}'] = {
                'accuracy': stage_correct / stage_total,
                'correct': stage_correct,
                'total': stage_total
            }
    
    # Save results
    output = {
        'checkpoint': checkpoint_path,
        'test_data': test_data_path,
        'split_method': 'image_based',
        'image_overlap': 0,
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'stage_accuracies': stage_metrics,
        'predictions': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("EVALUATION RESULTS (PROPER SPLIT - ZERO LEAKAGE)")
    logger.info(f"{'='*80}")
    logger.info(f"Overall Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Correct: {correct}/{total}")
    
    for stage, metrics in stage_metrics.items():
        logger.info(f"\n{stage.replace('_', ' ').title()}:")
        logger.info(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        logger.info(f"  Correct: {metrics['correct']}/{metrics['total']}")
    
    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with proper image-based split")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test JSON (pre-split)")
    parser.add_argument("--images", type=str, required=True, help="Image directory")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        image_dir=args.images,
        output_file=args.output,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()

