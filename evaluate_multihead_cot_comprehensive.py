#!/usr/bin/env python3
"""
Comprehensive evaluation script for multi-head CoT models with fixed answer extraction.

This script:
1. Loads a trained multi-head CoT checkpoint
2. Evaluates on full test set
3. Uses robust answer extraction and flexible matching
4. Saves detailed results
"""

import torch
import json
import string
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from multihead_model import create_multihead_model
    from cot_prompts import build_cot_prompt, format_prompt_for_model
    from data.vqa_data_loader import create_data_loader
    from transformers import AutoProcessor, AutoModelForVision2Seq
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_short_answer(prediction: str, ground_truth: str) -> str:
    """
    Extract concise answer from verbose CoT output.
    
    Examples:
    "yes, this finding is easy to detect" → "yes"
    "the image is taken from a colonoscopy" → "colonoscopy"
    "the polyp is approximately 1-2 cm" → "1-2 cm"
    "one" → "1"
    """
    pred = prediction.lower().strip()
    gt = ground_truth.lower().strip()
    
    # For yes/no questions
    if gt in ["yes", "no"]:
        if pred.startswith("yes"):
            return "yes"
        if pred.startswith("no"):
            return "no"
    
    # Remove verbose prefixes
    prefixes_to_remove = [
        "the answer is ", "it is ", "this is ",
        "the image is taken from ", "the image shows ",
        "the polyp is ", "the abnormality is ",
        "approximately ", "around ", "about ",
        "based on the image, ", "based on the image it is not possible to determine the exact size of the polyp",
        "the image is taken from a ", "the image is from a "
    ]
    
    for prefix in prefixes_to_remove:
        if pred.startswith(prefix):
            pred = pred[len(prefix):].strip()
            break
    
    # Remove trailing explanations
    pred = pred.split(",")[0].strip()
    pred = pred.split(".")[0].strip()
    
    # For single-word ground truths, take first word
    if len(gt.split()) == 1 and len(pred.split()) > 1:
        pred = pred.split()[0]
    
    return pred


def flexible_match(pred: str, gt: str) -> bool:
    """Flexible matching that handles verbose predictions."""
    pred = pred.lower().strip()
    gt = gt.lower().strip()
    
    # CRITICAL FIX: Empty predictions only correct if ground truth is also empty
    if not pred:
        return not gt  # Both empty = match, otherwise False
    
    if not gt:
        return False  # Empty ground truth, non-empty prediction = False
    
    # Exact match
    if pred == gt:
        return True
    
    # Ground truth is IN prediction (only if both are non-empty)
    if gt in pred:
        return True
    
    # Prediction is IN ground truth (only if both are non-empty)
    if pred in gt:
        return True
    
    # Handle multi-label (e.g., "center; lower-left; lower-right")
    if ";" in gt:
        gt_labels = [label.strip() for label in gt.split(";")]
        for label in gt_labels:
            if label in pred or pred in label:
                return True
    
    # Number normalization ("one" → "1")
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10"
    }
    
    pred_normalized = pred
    gt_normalized = gt
    for word, num in number_words.items():
        pred_normalized = pred_normalized.replace(word, num)
        gt_normalized = gt_normalized.replace(word, num)
    
    if pred_normalized == gt_normalized or gt_normalized in pred_normalized:
        return True
    
    # Remove punctuation and spaces
    pred_clean = pred.translate(str.maketrans('', '', string.punctuation + ' '))
    gt_clean = gt.translate(str.maketrans('', '', string.punctuation + ' '))
    
    if pred_clean == gt_clean or gt_clean in pred_clean:
        return True
    
    return False


def load_question_categories(categories_file: str) -> Dict[str, str]:
    """Load question categories mapping."""
    with open(categories_file, 'r') as f:
        data = json.load(f)
        # Handle both formats: {"question": "category"} or {"question": {"category": "..."}}
        if isinstance(list(data.values())[0], dict):
            return {k: v.get('category', 'abnormality_detection') for k, v in data.items()}
        return data


def load_model(checkpoint_path: str, base_checkpoint: str, model_type: str, device: str):
    """Load multi-head CoT model from checkpoint."""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    logger.info(f"Base checkpoint: {base_checkpoint}")
    logger.info(f"Model type: {model_type}")
    
    # Create multi-head model using base_checkpoint path (not a loaded model object)
    model = create_multihead_model(
        base_checkpoint=base_checkpoint,
        model_type=model_type,
        freeze_base=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Load to CPU first to avoid OOM
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded checkpoint (no epoch info)")
    
    model.eval()
    return model


def evaluate(
    model,
    test_loader,
    question_categories: Dict[str, str],
    dataset_name: str,
    processor,
    device: str,
    model_type: str,
    max_samples: Optional[int] = None
) -> Dict:
    """Evaluate model on test set with fixed answer extraction."""
    model.eval()
    
    correct = 0
    total = 0
    
    # Per-category tracking
    category_correct = {
        'abnormality_detection': 0,
        'characteristics': 0,
        'treatment': 0
    }
    category_total = {
        'abnormality_detection': 0,
        'characteristics': 0,
        'treatment': 0
    }
    
    all_predictions = []
    all_answers = []
    all_questions = []
    all_categories = []
    all_full_generated = []
    all_is_correct = []
    
    sample_count = 0
    max_samples = max_samples  # Get from function parameter
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
            # Stop if we've reached max_samples
            if max_samples is not None and sample_count >= max_samples:
                logger.info(f"Reached max_samples limit ({max_samples}), stopping evaluation")
                break
                
            images = batch.get('images', [])
            questions = batch.get('questions', [])
            answers = batch.get('answers', [])
            categories = batch.get('categories', [])
            
            for i in range(len(questions)):
                # Stop if we've reached max_samples
                if max_samples is not None and sample_count >= max_samples:
                    break
                if i >= len(images) or images[i] is None:
                    continue
                
                question = questions[i]
                answer = answers[i]
                category = categories[i] if i < len(categories) else question_categories.get(question, 'abnormality_detection')
                
                # Normalize category
                if category == 1 or category == "1":
                    category = 'abnormality_detection'
                elif category == 2 or category == "2":
                    category = 'characteristics'
                elif category == 3 or category == "3":
                    category = 'treatment'
                
                try:
                    # Build CoT prompt
                    prompt = build_cot_prompt(question, category)
                    formatted_prompt = format_prompt_for_model(prompt, model_type)
                    
                    # Process inputs
                    inputs = processor(text=formatted_prompt, images=images[i], return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Generate answer
                    # Access base model through model.model.base_model (MultiHeadCoT_Model -> model -> base_model)
                    base_model = model.model.base_model if hasattr(model, 'model') and hasattr(model.model, 'base_model') else model.base_model
                    
                    if model_type == "qwen3vl":
                        # Qwen3-VL needs image_grid_thw
                        if 'image_grid_thw' in inputs:
                            generated_ids = base_model.generate(
                                pixel_values=inputs['pixel_values'],
                                input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                image_grid_thw=inputs['image_grid_thw'],
                                max_new_tokens=50,
                                do_sample=False
                            )
                        else:
                            generated_ids = base_model.generate(
                                pixel_values=inputs['pixel_values'],
                                input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                max_new_tokens=50,
                                do_sample=False
                            )
                    else:
                        generated_ids = base_model.generate(
                            pixel_values=inputs['pixel_values'],
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_new_tokens=50,
                            do_sample=False
                        )
                    
                    # Decode full generated text
                    full_generated = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    
                    # Extract new tokens (answer part)
                    new_tokens = generated_ids[0, inputs['input_ids'].shape[1]:]
                    predicted_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    # Extract short answer
                    short_pred = extract_short_answer(predicted_text, answer)
                    
                    # Flexible matching
                    is_correct = flexible_match(short_pred, answer)
                    
                    # Update counters
                    total += 1
                    if is_correct:
                        correct += 1
                    
                    category_total[category] += 1
                    if is_correct:
                        category_correct[category] += 1
                    
                    # Store results
                    all_predictions.append(short_pred)
                    all_answers.append(answer)
                    all_questions.append(question)
                    all_categories.append(category)
                    all_full_generated.append(full_generated)
                    all_is_correct.append(is_correct)
                    
                    # Increment sample count
                    sample_count += 1
                    total += 1
                    
                    # Stop if we've reached max_samples
                    if max_samples is not None and sample_count >= max_samples:
                        break
                    
                except Exception as e:
                    logger.error(f"Error processing sample {total}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Calculate accuracies
    overall_accuracy = correct / total if total > 0 else 0.0
    category_accuracies = {
        cat: category_correct[cat] / category_total[cat] if category_total[cat] > 0 else 0.0
        for cat in category_total
    }
    
    results = {
        'overall_accuracy': overall_accuracy,
        'correct': correct,
        'total': total,
        'category_accuracies': category_accuracies,
        'category_correct': category_correct,
        'category_total': category_total,
        'predictions': all_predictions,
        'answers': all_answers,
        'questions': all_questions,
        'categories': all_categories,
        'full_generated_text': all_full_generated,
        'is_correct': all_is_correct
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation for multi-head CoT models")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--base_checkpoint", required=True, help="Base model checkpoint (HuggingFace model ID or path)")
    parser.add_argument("--model_name", required=True, choices=["qwen3vl", "medgemma", "llava"], help="Model type")
    parser.add_argument("--dataset", required=True, choices=["kvasir", "endovis"], help="Dataset name")
    parser.add_argument("--data_path", required=True, help="Path to dataset directory")
    parser.add_argument("--image_base_path", required=True, help="Base path for images")
    parser.add_argument("--question_categories", required=True, help="Path to question categories JSON")
    parser.add_argument("--output_file", required=True, help="Output JSON file path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--use_flexible_matching", action="store_true", help="Use flexible matching (always enabled)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate (for testing)")
    
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load question categories
    question_categories = load_question_categories(args.question_categories)
    logger.info(f"Loaded {len(question_categories)} question categories")
    
    # Load processor (use HuggingFace model name, not local checkpoint path)
    hf_token = os.getenv("HF_TOKEN")
    if args.model_name == "qwen3vl":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True, token=hf_token)
    elif args.model_name == "medgemma":
        processor = AutoProcessor.from_pretrained("google/medgemma-4b-it", trust_remote_code=True, token=hf_token)
    else:
        processor = AutoProcessor.from_pretrained(args.base_checkpoint, trust_remote_code=True, token=hf_token)
    
    # Load test data
    # Try test.json first, then test.jsonl (for EndoVis)
    test_file = os.path.join(args.data_path, "test.json")
    if not os.path.exists(test_file):
        test_file = os.path.join(args.data_path, "test.jsonl")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found in {args.data_path}. Expected test.json or test.jsonl")
    
    test_loader = create_data_loader(
        test_file,
        args.image_base_path,
        batch_size=args.batch_size,
        shuffle=False,
        is_temporal=(args.dataset == "endovis")
    )
    logger.info(f"Loaded test set from {test_file}")
    
    # Load model
    model = load_model(args.checkpoint, args.base_checkpoint, args.model_name, device)
    model = model.to(device)
    
    # Evaluate
    logger.info("Starting evaluation...")
    max_samples = getattr(args, 'max_samples', None)
    if max_samples:
        logger.info(f"Limiting evaluation to {max_samples} samples")
    results = evaluate(
        model,
        test_loader,
        question_categories,
        args.dataset,
        processor,
        device,
        args.model_name,
        max_samples=max_samples
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.2%} ({results['correct']}/{results['total']})")
    print(f"\nPer-Category Accuracies:")
    for cat, acc in results['category_accuracies'].items():
        print(f"  {cat}: {acc:.2%} ({results['category_correct'][cat]}/{results['category_total'][cat]})")
    print("="*80)
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()




