#!/usr/bin/env python3
"""
Evaluation Script for Experiment 3: CXRTReK Sequential
Evaluates 3 separate models (one for each stage)
"""

import json
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from tqdm import tqdm
import os
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import argparse


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip().replace(".", "").replace(",", "").replace(";", "")


def parse_labels(text: str) -> set:
    """Parse labels from text (handles semicolon-separated multi-label format)."""
    if not text:
        return set()
    labels = [normalize_text(label) for label in text.split(';')]
    return set(label for label in labels if label)


def calculate_precision_recall_f1(pred_set: set, gt_set: set) -> tuple:
    """Calculate precision, recall, and F1 score for two sets of labels."""
    if not gt_set:
        return (0.0, 0.0, 0.0)
    if not pred_set:
        return (0.0, 0.0, 0.0)
    intersection = pred_set & gt_set
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(gt_set) if gt_set else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return (precision, recall, f1)


def smart_match(prediction: str, ground_truth: str, threshold: float = 0.7) -> bool:
    """Smart matching with multiple strategies."""
    pred_n = normalize_text(prediction)
    gt_n = normalize_text(ground_truth)
    
    if not gt_n:
        return False
    
    # Exact match
    if pred_n == gt_n:
        return True
    
    # Substring match
    if gt_n in pred_n or pred_n in gt_n:
        return True
    
    # Fuzzy similarity
    similarity = SequenceMatcher(None, pred_n, gt_n).ratio()
    return similarity >= threshold


def load_model_with_lora(model_path: str, base_model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
    """Load vision-language model with LoRA adapter."""
    print(f"Loading base model: {base_model_name}")
    
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Find best checkpoint or use final model
    adapter_path = model_path
    if os.path.isdir(model_path):
        # Check if there's a final adapter_model.safetensors in the root
        if os.path.exists(os.path.join(model_path, "adapter_model.safetensors")):
            print(f"Using final model from: {model_path}")
            adapter_path = model_path
        else:
            # Find best checkpoint
            checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoint_numbers = [int(c.split("-")[1]) for c in checkpoints]
                best_checkpoint = f"checkpoint-{max(checkpoint_numbers)}"
                adapter_path = os.path.join(model_path, best_checkpoint)
                print(f"Using checkpoint: {best_checkpoint}")
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    # Merge LoRA for faster inference
    model = model.merge_and_unload()
    model.eval()
    
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    
    device = next(model.parameters()).device
    print(f"✓ Model loaded on device: {device}")
    
    return model, processor, device


def extract_answer_from_response(generated_text: str) -> str:
    """Extract the answer from generated text."""
    # Remove common prefixes
    prefixes = ["assistant", "### Response:", "Response:", "Answer:", "Answer"]
    for prefix in prefixes:
        if generated_text.strip().lower().startswith(prefix.lower()):
            generated_text = generated_text.strip()[len(prefix):].strip()
    
    # Remove trailing special tokens
    for token in ["</s>", "<|endoftext|>", "<|im_end|>"]:
        if generated_text.endswith(token):
            generated_text = generated_text[:-len(token)].strip()
    
    # Take first line if multiple lines
    lines = generated_text.split('\n')
    if lines:
        generated_text = lines[0].strip()
    
    return generated_text.strip()


def evaluate_stage_model(model, processor, device, test_data, image_base_path, stage_num, max_samples=None):
    """Evaluate a single stage model."""
    
    # Filter for this stage
    stage_data = [item for item in test_data if item.get('stage') == stage_num]
    
    if max_samples:
        stage_data = stage_data[:max_samples]
    
    print(f"\nEvaluating Stage {stage_num} - {len(stage_data)} samples")
    
    results = {
        'total': 0,
        'correct': 0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'predictions': [],
        'errors': []
    }
    
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    for item in tqdm(stage_data, desc=f"Stage {stage_num}"):
        # Use instruction field if available (for instructed datasets), otherwise use question
        question = item.get('instruction', item.get('question', ''))
        ground_truth = item.get('answer', '').strip()
        image_filename = item.get('image_filename', '')
        image_id = item.get('image_id', '')
        question_type = item.get('question_type', 'unknown')
        
        # Get image path
        if not image_filename:
            image_filename = f"{image_id}.jpg"
        
        image_path = os.path.join(image_base_path, image_filename)
        
        if not os.path.exists(image_path):
            results['errors'].append(f"Image not found: {image_path}")
            continue
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            results['errors'].append(f"Image load error: {str(e)}")
            continue
        
        # Generate prediction
        try:
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }]
            
            text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            input_len = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128)
            
            # Decode only the generated tokens
            generated_ids = outputs[0][input_len:]
            prediction = processor.decode(generated_ids, skip_special_tokens=True)
            prediction = extract_answer_from_response(prediction)
            
        except Exception as e:
            results['errors'].append(f"Generation error: {str(e)}")
            prediction = ""
        
        # Evaluate
        correct = smart_match(prediction, ground_truth)
        
        # Calculate precision, recall, F1
        pred_set = parse_labels(prediction)
        gt_set = parse_labels(ground_truth)
        precision, recall, f1 = calculate_precision_recall_f1(pred_set, gt_set)
        
        results['total'] += 1
        results['correct'] += int(correct)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        
        results['predictions'].append({
            'image_id': image_id,
            'question': item.get('question', ''),  # Original question
            'instruction': question,  # The instruction/prompt actually used
            'question_type': question_type,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'correct': correct,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'stage': stage_num
        })
    
    # Calculate metrics
    if results['total'] > 0:
        results['accuracy'] = (results['correct'] / results['total']) * 100
        results['precision'] = (total_precision / results['total']) * 100
        results['recall'] = (total_recall / results['total']) * 100
        results['f1'] = (total_f1 / results['total']) * 100
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Experiment 3: CXRTReK Sequential")
    parser.add_argument("--model_stage1", type=str, required=True, help="Path to Stage 1 model")
    parser.add_argument("--model_stage2", type=str, required=True, help="Path to Stage 2 model")
    parser.add_argument("--model_stage3", type=str, required=True, help="Path to Stage 3 model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Base model name")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per stage (for testing)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXPERIMENT 3 EVALUATION: CXRTReK Sequential (3 Separate Models)")
    print("=" * 80)
    
    # Load test data
    print(f"\nLoading test data from: {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    print(f"Total test samples: {len(test_data)}")
    
    all_results = {
        'by_stage': {},
        'by_question_type': defaultdict(lambda: {
            'total': 0, 'correct': 0, 'accuracy': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0
        }),
        'overall': {
            'total': 0,
            'correct': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        },
        'all_predictions': []
    }
    
    # Accumulators for overall metrics
    overall_precision = 0.0
    overall_recall = 0.0
    overall_f1 = 0.0
    qtype_precision = defaultdict(float)
    qtype_recall = defaultdict(float)
    qtype_f1 = defaultdict(float)
    
    # Evaluate each stage
    for stage_num, model_path in [(1, args.model_stage1), (2, args.model_stage2), (3, args.model_stage3)]:
        print(f"\n{'=' * 80}")
        print(f"Stage {stage_num}")
        print(f"Model: {model_path}")
        print(f"{'=' * 80}")
        
        # Load model for this stage
        model, processor, device = load_model_with_lora(model_path, args.base_model)
        
        # Evaluate this stage
        stage_results = evaluate_stage_model(model, processor, device, test_data, args.image_dir, stage_num, args.max_samples)
        
        # Store results
        all_results['by_stage'][f'Stage {stage_num}'] = {
            'total': stage_results['total'],
            'correct': stage_results['correct'],
            'accuracy': stage_results['accuracy'],
            'precision': stage_results['precision'],
            'recall': stage_results['recall'],
            'f1': stage_results['f1'],
            'errors': len(stage_results['errors'])
        }
        
        all_results['all_predictions'].extend(stage_results['predictions'])
        all_results['overall']['total'] += stage_results['total']
        all_results['overall']['correct'] += stage_results['correct']
        overall_precision += stage_results['precision'] * stage_results['total']
        overall_recall += stage_results['recall'] * stage_results['total']
        overall_f1 += stage_results['f1'] * stage_results['total']
        
        # Aggregate by question type
        for pred in stage_results['predictions']:
            qtype = pred.get('question_type', 'unknown')
            all_results['by_question_type'][qtype]['total'] += 1
            if pred.get('correct', False):
                all_results['by_question_type'][qtype]['correct'] += 1
            qtype_precision[qtype] += pred.get('precision', 0.0)
            qtype_recall[qtype] += pred.get('recall', 0.0)
            qtype_f1[qtype] += pred.get('f1', 0.0)
        
        print(f"\nStage {stage_num} Results:")
        print(f"  Total: {stage_results['total']}")
        print(f"  Correct: {stage_results['correct']}")
        print(f"  Accuracy: {stage_results['accuracy']:.2f}%")
        print(f"  Precision: {stage_results['precision']:.2f}%")
        print(f"  Recall: {stage_results['recall']:.2f}%")
        print(f"  F1: {stage_results['f1']:.2f}%")
        
        if stage_results['errors']:
            print(f"  Errors: {len(stage_results['errors'])}")
        
        # Clean up to free memory
        del model, processor
        torch.cuda.empty_cache()
    
    # Calculate overall metrics
    if all_results['overall']['total'] > 0:
        all_results['overall']['accuracy'] = (all_results['overall']['correct'] / all_results['overall']['total']) * 100
        all_results['overall']['precision'] = (overall_precision / all_results['overall']['total'])
        all_results['overall']['recall'] = (overall_recall / all_results['overall']['total'])
        all_results['overall']['f1'] = (overall_f1 / all_results['overall']['total'])
    
    # Calculate metrics by question type
    for qtype, qtype_data in all_results['by_question_type'].items():
        if qtype_data['total'] > 0:
            qtype_data['accuracy'] = (qtype_data['correct'] / qtype_data['total']) * 100
            qtype_data['precision'] = (qtype_precision[qtype] / qtype_data['total']) * 100
            qtype_data['recall'] = (qtype_recall[qtype] / qtype_data['total']) * 100
            qtype_data['f1'] = (qtype_f1[qtype] / qtype_data['total']) * 100
    
    # Print summary
    print("\n" + "=" * 80)
    print("OVERALL EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nTotal samples: {all_results['overall']['total']}")
    print(f"Correct: {all_results['overall']['correct']}")
    print(f"Overall Accuracy: {all_results['overall']['accuracy']:.2f}%")
    print(f"Overall Precision: {all_results['overall']['precision']:.2f}%")
    print(f"Overall Recall: {all_results['overall']['recall']:.2f}%")
    print(f"Overall F1: {all_results['overall']['f1']:.2f}%")
    
    print(f"\nBy Stage:")
    for stage in sorted(all_results['by_stage'].keys()):
        stage_data = all_results['by_stage'][stage]
        print(f"  {stage}:")
        print(f"    Accuracy: {stage_data['accuracy']:.2f}% ({stage_data['correct']}/{stage_data['total']})")
        print(f"    Precision: {stage_data['precision']:.2f}%")
        print(f"    Recall: {stage_data['recall']:.2f}%")
        print(f"    F1: {stage_data['f1']:.2f}%")
    
    print(f"\nBy Question Type:")
    for qtype in sorted(all_results['by_question_type'].keys()):
        qtype_data = all_results['by_question_type'][qtype]
        print(f"  {qtype}:")
        print(f"    Accuracy: {qtype_data['accuracy']:.2f}% ({qtype_data['correct']}/{qtype_data['total']})")
        print(f"    Precision: {qtype_data['precision']:.2f}%")
        print(f"    Recall: {qtype_data['recall']:.2f}%")
        print(f"    F1: {qtype_data['f1']:.2f}%")
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Convert defaultdict to dict for JSON serialization
    all_results['by_question_type'] = dict(all_results['by_question_type'])
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("✓ Evaluation complete!")


if __name__ == "__main__":
    main()



