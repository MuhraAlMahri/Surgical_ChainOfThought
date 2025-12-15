#!/usr/bin/env python3
"""
Improved evaluation script for surgical VQA that properly handles different question types.
Uses strict matching and extracts key information from verbose predictions.
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from tqdm import tqdm


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip().replace(".", "").replace(",", "").replace(";", "")


def extract_binary_answer(prediction: str) -> Optional[str]:
    """
    Extract yes/no from a potentially verbose prediction.
    Returns: 'yes', 'no', or None if ambiguous
    """
    pred_lower = prediction.lower()
    
    # Direct match at start
    if pred_lower.startswith('yes'):
        return 'yes'
    if pred_lower.startswith('no'):
        return 'no'
    
    # Look for clear yes/no in first sentence
    first_sentence = pred_lower.split('.')[0]
    
    # Count occurrences
    yes_count = first_sentence.count('yes')
    no_count = first_sentence.count('no')
    
    if yes_count > no_count:
        return 'yes'
    elif no_count > yes_count:
        return 'no'
    
    return None


def extract_number(prediction: str) -> Optional[float]:
    """
    Extract a number from prediction text.
    Handles: "3", "three", "2.5", "approximately 5"
    """
    # Word to number mapping
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10
    }
    
    pred_lower = prediction.lower().strip()
    
    # Try direct number extraction
    numbers = re.findall(r'\b\d+\.?\d*\b', prediction)
    if numbers:
        try:
            return float(numbers[0])
        except:
            pass
    
    # Try word numbers
    for word, num in word_to_num.items():
        if word in pred_lower:
            return float(num)
    
    return None


def extract_first_word(prediction: str) -> str:
    """
    Extract the first meaningful word from a prediction.
    Useful for color, size, and short answer questions.
    """
    # Remove common prefixes
    pred = prediction.lower().strip()
    
    # Remove leading articles and common phrases
    prefixes_to_remove = [
        'the ', 'a ', 'an ', 'it is ', 'it appears to be ',
        'the abnormality is ', 'the color is ', 'the size is '
    ]
    
    for prefix in prefixes_to_remove:
        if pred.startswith(prefix):
            pred = pred[len(prefix):]
    
    # Get first word
    words = pred.split()
    if words:
        # Remove punctuation from first word
        first = words[0].strip('.,!?;:')
        return first
    
    return pred


def evaluate_prediction(
    prediction: str,
    ground_truth: str,
    question_type: str
) -> Tuple[bool, str, str]:
    """
    Evaluate a prediction based on question type.
    
    Returns:
        (is_correct, extracted_answer, reason)
    """
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    
    if question_type == 'binary':
        extracted = extract_binary_answer(prediction)
        if extracted is None:
            return False, prediction[:50], "Could not extract yes/no"
        is_correct = extracted == gt_norm
        return is_correct, extracted, "Extracted binary answer"
    
    elif question_type == 'numeric':
        extracted_num = extract_number(prediction)
        try:
            gt_num = float(ground_truth)
            if extracted_num is None:
                return False, prediction[:50], "Could not extract number"
            # Allow small tolerance for floating point
            is_correct = abs(extracted_num - gt_num) < 0.01
            return is_correct, str(extracted_num), "Extracted number"
        except ValueError:
            return False, prediction[:50], "Ground truth not numeric"
    
    elif question_type in ['color', 'size']:
        # Extract first meaningful word
        extracted = extract_first_word(prediction)
        is_correct = extracted == gt_norm
        return is_correct, extracted, "Extracted first word"
    
    elif question_type == 'open_short':
        # For short answers, extract first 1-3 words
        extracted = ' '.join(prediction.split()[:3]).lower().strip('.,!?;:')
        # Check if ground truth is in extracted portion
        is_correct = gt_norm in extracted or extracted in gt_norm
        return is_correct, extracted, "Extracted short answer"
    
    elif question_type == 'mcq':
        # For MCQ, check if any ground truth option appears in prediction
        gt_options = [opt.strip() for opt in ground_truth.lower().split(',')]
        pred_lower = prediction.lower()
        
        for option in gt_options:
            if option in pred_lower:
                return True, option, "Found MCQ option"
        
        return False, prediction[:50], "No matching MCQ option"
    
    else:  # open_long or unknown
        # For longer answers, use more lenient matching
        is_correct = gt_norm in pred_norm or pred_norm in gt_norm
        return is_correct, prediction[:100], "Substring match"


def load_model_with_lora(model_path: str, base_model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
    """Load vision-language model with LoRA adapter."""
    print(f"Loading base model: {base_model_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Find best checkpoint
    checkpoints = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')]
    if checkpoints:
        # Sort by checkpoint number
        checkpoint_nums = [int(c.split('-')[1]) for c in checkpoints]
        best_checkpoint = checkpoints[checkpoint_nums.index(max(checkpoint_nums))]
        adapter_path = os.path.join(model_path, best_checkpoint)
        print(f"Loading LoRA adapter from: {adapter_path}")
    else:
        adapter_path = model_path
        print(f"Loading LoRA adapter from: {adapter_path}")
    
    # Load and merge LoRA
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    
    return model, processor, device


def evaluate_model(
    model,
    processor,
    test_data: List[Dict],
    image_dir: str,
    device,
    max_new_tokens: int = 128
):
    """
    Evaluate model on test dataset with question-type-aware metrics.
    """
    results = []
    
    # Overall metrics
    total_correct = 0
    total_samples = 0
    
    # Per-question-type metrics
    type_metrics = {}
    
    print(f"\nEvaluating {len(test_data)} samples...")
    
    for item in tqdm(test_data, desc="Evaluating"):
        question = item.get('question', '')
        ground_truth = item.get('answer', '')
        image_filename = item.get('image_filename', item.get('image_path', ''))
        question_type = item.get('question_type', 'unknown')
        instruction = item.get('instruction', question)  # Use instruction if available
        
        # Initialize metrics for this question type
        if question_type not in type_metrics:
            type_metrics[question_type] = {'correct': 0, 'total': 0}
        
        # Load image
        if image_filename.startswith('/'):
            full_image_path = image_filename
        else:
            full_image_path = os.path.join(image_dir, image_filename)
        
        if not os.path.exists(full_image_path):
            print(f"\nWarning: Image not found: {full_image_path}")
            continue
        
        try:
            image = Image.open(full_image_path).convert('RGB')
        except Exception as e:
            print(f"\nWarning: Could not load image {full_image_path}: {e}")
            continue
        
        # Prepare conversation with instruction if available
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction}  # Use instruction, not raw question
            ]
        }]
        
        # Generate prediction
        text_prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        # Decode only new tokens
        prompt_len = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][prompt_len:]
        prediction = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Evaluate based on question type
        is_correct, extracted_answer, eval_reason = evaluate_prediction(
            prediction, ground_truth, question_type
        )
        
        # Update metrics
        if is_correct:
            total_correct += 1
            type_metrics[question_type]['correct'] += 1
        
        total_samples += 1
        type_metrics[question_type]['total'] += 1
        
        # Store result
        results.append({
            'image': image_filename,
            'question': question,
            'question_type': question_type,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'extracted_answer': extracted_answer,
            'correct': is_correct,
            'eval_reason': eval_reason
        })
    
    # Calculate metrics
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    
    # Calculate per-type accuracies
    type_accuracies = {}
    for qtype, metrics in type_metrics.items():
        if metrics['total'] > 0:
            type_accuracies[qtype] = {
                'accuracy': (metrics['correct'] / metrics['total']) * 100,
                'correct': metrics['correct'],
                'total': metrics['total']
            }
    
    return {
        'accuracy': overall_accuracy,
        'correct': total_correct,
        'total': total_samples,
        'by_question_type': type_accuracies,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate surgical VQA model with question-type awareness")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data JSON')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2-VL-7B-Instruct', help='Base model name')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Max tokens to generate')
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from: {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Load model
    model, processor, device = load_model_with_lora(args.model_path, args.base_model)
    
    # Evaluate
    results = evaluate_model(
        model,
        processor,
        test_data,
        args.image_dir,
        device,
        max_new_tokens=args.max_new_tokens
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {results['accuracy']:.2f}%")
    print(f"Correct: {results['correct']}/{results['total']}")
    
    print("\n" + "-"*60)
    print("Per Question Type Accuracy:")
    print("-"*60)
    
    for qtype, metrics in sorted(results['by_question_type'].items()):
        print(f"{qtype:15s}: {metrics['accuracy']:6.2f}% ({metrics['correct']}/{metrics['total']})")
    
    print("\n" + "="*60)
    print(f"Results saved to: {args.output}")
    print("="*60 + "\n")
    
    # Show some example predictions
    print("\nSample Predictions:")
    print("-"*60)
    for i, result in enumerate(results['results'][:5]):
        print(f"\nExample {i+1}:")
        print(f"  Question Type: {result['question_type']}")
        print(f"  Question: {result['question']}")
        print(f"  Ground Truth: {result['ground_truth']}")
        print(f"  Extracted: {result['extracted_answer']}")
        print(f"  Full Prediction: {result['prediction'][:100]}...")
        print(f"  Correct: {'✓' if result['correct'] else '✗'}")


if __name__ == "__main__":
    main()
