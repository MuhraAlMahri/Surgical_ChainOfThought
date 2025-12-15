#!/usr/bin/env python3
"""
Evaluation Script for Experiment 1: Random Baseline
Uses Qwen2-VL-7B-Instruct vision-language model with images + questions
"""

import json
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from tqdm import tqdm
import os
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import argparse
import sys
import os

# Add the evaluation scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_utils import calculate_text_generation_metrics


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip().replace(".", "").replace(",", "").replace(";", "")


def parse_labels(text: str) -> set:
    """Parse labels from text (handles semicolon-separated multi-label format)."""
    if not text:
        return set()
    # Split by semicolon and normalize
    labels = [normalize_text(label) for label in text.split(';')]
    # Remove empty labels
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


def load_model_with_lora(model_path: str, base_model_name: str = "Qwen/Qwen3-VL-8B-Instruct", hf_token: str = None):
    """Load vision-language model with LoRA adapter."""
    print(f"Loading base model: {base_model_name}")
    
    # Handle Hugging Face authentication for gated models (e.g., MedGemma-4B)
    if hf_token:
        from huggingface_hub import login
        print("Authenticating with Hugging Face...")
        login(token=hf_token)
    
    # Get token from environment if not provided
    token = hf_token or os.getenv("HF_TOKEN")
    
    # Determine model class based on model name
    # MedGemma uses AutoModelForImageTextToText, Qwen uses AutoModelForVision2Seq
    is_medgemma = 'medgemma' in base_model_name.lower() or 'pali' in base_model_name.lower()
    
    if is_medgemma:
        print("Detected MedGemma/PaliGemma model - using AutoModelForImageTextToText")
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
    else:
        print("Using AutoModelForVision2Seq for vision-language model")
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=token
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
    # Don't merge - use LoRA directly to save memory
    model.eval()
    
    processor = AutoProcessor.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        token=token
    )
    
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


def evaluate_model(model, processor, device, test_data, image_base_path, max_samples=None, use_question_only=False):
    """Evaluate model on test data."""
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    results = {
        'total': 0,
        'correct': 0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'bleu': 0.0,
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
        'meteor': 0.0,
        'by_stage': defaultdict(lambda: {
            'total': 0, 'correct': 0, 'accuracy': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'meteor': 0.0
        }),
        'by_question_type': defaultdict(lambda: {
            'total': 0, 'correct': 0, 'accuracy': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'meteor': 0.0
        }),
        'predictions': [],
        'errors': []
    }
    
    # Accumulators for precision/recall/F1
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_bleu = 0.0
    total_rouge1 = 0.0
    total_rouge2 = 0.0
    total_rougeL = 0.0
    total_meteor = 0.0
    
    stage_precision = defaultdict(float)
    stage_recall = defaultdict(float)
    stage_f1 = defaultdict(float)
    stage_bleu = defaultdict(float)
    stage_rouge1 = defaultdict(float)
    stage_rouge2 = defaultdict(float)
    stage_rougeL = defaultdict(float)
    stage_meteor = defaultdict(float)
    
    qtype_precision = defaultdict(float)
    qtype_recall = defaultdict(float)
    qtype_f1 = defaultdict(float)
    qtype_bleu = defaultdict(float)
    qtype_rouge1 = defaultdict(float)
    qtype_rouge2 = defaultdict(float)
    qtype_rougeL = defaultdict(float)
    qtype_meteor = defaultdict(float)
    
    for item in tqdm(test_data, desc="Evaluating"):
        # Use question only if flag is set, otherwise use instruction field if available
        if use_question_only:
            question = item.get('question', '')
        else:
            question = item.get('instruction', item.get('question', ''))
        ground_truth = item.get('answer', '').strip()
        image_filename = item.get('image_filename', '')
        image_id = item.get('image_id', '')
        stage = item.get('stage', 0)
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
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=20,  # Reduced from 128 for instructed responses
                    do_sample=False,     # Deterministic for consistency
                    temperature=None,    # Disable sampling
                    top_p=None
                )
            
            # Decode only the generated tokens (skip the input prompt)
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
        
        # Calculate text generation metrics (BLEU, ROUGE, METEOR)
        text_metrics = calculate_text_generation_metrics(prediction, ground_truth)
        bleu = text_metrics['bleu']
        rouge1 = text_metrics['rouge1']
        rouge2 = text_metrics['rouge2']
        rougeL = text_metrics['rougeL']
        meteor = text_metrics['meteor']
        
        results['total'] += 1
        results['correct'] += int(correct)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_bleu += bleu
        total_rouge1 += rouge1
        total_rouge2 += rouge2
        total_rougeL += rougeL
        total_meteor += meteor
        
        # Track by stage
        stage_key = f"Stage {stage}" if stage in [1, 2, 3] else "Unknown"
        results['by_stage'][stage_key]['total'] += 1
        results['by_stage'][stage_key]['correct'] += int(correct)
        stage_precision[stage_key] += precision
        stage_recall[stage_key] += recall
        stage_f1[stage_key] += f1
        stage_bleu[stage_key] += bleu
        stage_rouge1[stage_key] += rouge1
        stage_rouge2[stage_key] += rouge2
        stage_rougeL[stage_key] += rougeL
        stage_meteor[stage_key] += meteor
        
        # Track by question type
        results['by_question_type'][question_type]['total'] += 1
        results['by_question_type'][question_type]['correct'] += int(correct)
        qtype_precision[question_type] += precision
        qtype_recall[question_type] += recall
        qtype_f1[question_type] += f1
        qtype_bleu[question_type] += bleu
        qtype_rouge1[question_type] += rouge1
        qtype_rouge2[question_type] += rouge2
        qtype_rougeL[question_type] += rougeL
        qtype_meteor[question_type] += meteor
        
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
            'bleu': bleu,
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
            'meteor': meteor,
            'stage': stage
        })
    
    # Calculate metrics
    if results['total'] > 0:
        results['accuracy'] = (results['correct'] / results['total']) * 100
        results['precision'] = (total_precision / results['total']) * 100
        results['recall'] = (total_recall / results['total']) * 100
        results['f1'] = (total_f1 / results['total']) * 100
        results['bleu'] = (total_bleu / results['total']) * 100
        results['rouge1'] = (total_rouge1 / results['total']) * 100
        results['rouge2'] = (total_rouge2 / results['total']) * 100
        results['rougeL'] = (total_rougeL / results['total']) * 100
        results['meteor'] = (total_meteor / results['total']) * 100
        
        for stage_key, stage_data in results['by_stage'].items():
            if stage_data['total'] > 0:
                stage_data['accuracy'] = (stage_data['correct'] / stage_data['total']) * 100
                stage_data['precision'] = (stage_precision[stage_key] / stage_data['total']) * 100
                stage_data['recall'] = (stage_recall[stage_key] / stage_data['total']) * 100
                stage_data['f1'] = (stage_f1[stage_key] / stage_data['total']) * 100
                stage_data['bleu'] = (stage_bleu[stage_key] / stage_data['total']) * 100
                stage_data['rouge1'] = (stage_rouge1[stage_key] / stage_data['total']) * 100
                stage_data['rouge2'] = (stage_rouge2[stage_key] / stage_data['total']) * 100
                stage_data['rougeL'] = (stage_rougeL[stage_key] / stage_data['total']) * 100
                stage_data['meteor'] = (stage_meteor[stage_key] / stage_data['total']) * 100
        
        for qtype, qtype_data in results['by_question_type'].items():
            if qtype_data['total'] > 0:
                qtype_data['accuracy'] = (qtype_data['correct'] / qtype_data['total']) * 100
                qtype_data['precision'] = (qtype_precision[qtype] / qtype_data['total']) * 100
                qtype_data['recall'] = (qtype_recall[qtype] / qtype_data['total']) * 100
                qtype_data['f1'] = (qtype_f1[qtype] / qtype_data['total']) * 100
                qtype_data['bleu'] = (qtype_bleu[qtype] / qtype_data['total']) * 100
                qtype_data['rouge1'] = (qtype_rouge1[qtype] / qtype_data['total']) * 100
                qtype_data['rouge2'] = (qtype_rouge2[qtype] / qtype_data['total']) * 100
                qtype_data['rougeL'] = (qtype_rougeL[qtype] / qtype_data['total']) * 100
                qtype_data['meteor'] = (qtype_meteor[qtype] / qtype_data['total']) * 100
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Experiment 1: Random Baseline")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Base model name")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (for testing)")
    parser.add_argument("--use_question_only", action="store_true", help="Use only question field (no instructions)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    if args.use_question_only:
        print("EXPERIMENT 1 EVALUATION: Random Baseline (WITHOUT INSTRUCTIONS)")
    else:
        print("EXPERIMENT 1 EVALUATION: Random Baseline (WITH INSTRUCTIONS)")
    print("=" * 80)
    
    # Load model
    # Get HF token from environment for gated models (e.g., MedGemma-4B)
    hf_token = os.getenv("HF_TOKEN")
    model, processor, device = load_model_with_lora(args.model_path, args.base_model, hf_token=hf_token)
    
    # Load test data (supports both JSON and JSONL formats)
    print(f"\nLoading test data from: {args.test_data}")
    test_data = []
    with open(args.test_data, 'r') as f:
        if args.test_data.endswith('.jsonl'):
            # JSONL format: one JSON object per line
            for line in f:
                line = line.strip()
                if line:
                    test_data.append(json.loads(line))
        else:
            # JSON format: single JSON array/object
            data = json.load(f)
            if isinstance(data, list):
                test_data = data
            elif isinstance(data, dict):
                test_data = list(data.values()) if data else []
            else:
                test_data = [data]
    print(f"Test samples: {len(test_data)}")
    if args.use_question_only:
        print("Mode: Using question field only (no instructions)")
    else:
        print("Mode: Using instruction field if available")
    
    # Evaluate
    results = evaluate_model(model, processor, device, test_data, args.image_dir, args.max_samples, args.use_question_only)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nTotal samples: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"\n--- Classification Metrics ---")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Precision: {results['precision']:.2f}%")
    print(f"Recall: {results['recall']:.2f}%")
    print(f"F1 Score: {results['f1']:.2f}%")
    print(f"\n--- Text Generation Metrics ---")
    print(f"BLEU: {results['bleu']:.2f}%")
    print(f"ROUGE-1: {results['rouge1']:.2f}%")
    print(f"ROUGE-2: {results['rouge2']:.2f}%")
    print(f"ROUGE-L: {results['rougeL']:.2f}%")
    print(f"METEOR: {results['meteor']:.2f}%")
    
    print(f"\nBy Stage:")
    for stage in sorted(results['by_stage'].keys()):
        stage_data = results['by_stage'][stage]
        print(f"  {stage}:")
        print(f"    Accuracy: {stage_data['accuracy']:.2f}% ({stage_data['correct']}/{stage_data['total']})")
        print(f"    Precision: {stage_data['precision']:.2f}% | Recall: {stage_data['recall']:.2f}% | F1: {stage_data['f1']:.2f}%")
        print(f"    BLEU: {stage_data['bleu']:.2f}% | ROUGE-1: {stage_data['rouge1']:.2f}% | ROUGE-2: {stage_data['rouge2']:.2f}% | ROUGE-L: {stage_data['rougeL']:.2f}% | METEOR: {stage_data['meteor']:.2f}%")
    
    print(f"\nBy Question Type:")
    for qtype in sorted(results['by_question_type'].keys()):
        qtype_data = results['by_question_type'][qtype]
        print(f"  {qtype:15s}:")
        print(f"    Accuracy: {qtype_data['accuracy']:.2f}% ({qtype_data['correct']}/{qtype_data['total']})")
        print(f"    Precision: {qtype_data['precision']:.2f}% | Recall: {qtype_data['recall']:.2f}% | F1: {qtype_data['f1']:.2f}%")
        print(f"    BLEU: {qtype_data['bleu']:.2f}% | ROUGE-1: {qtype_data['rouge1']:.2f}% | ROUGE-2: {qtype_data['rouge2']:.2f}% | ROUGE-L: {qtype_data['rougeL']:.2f}% | METEOR: {qtype_data['meteor']:.2f}%")
    
    if results['errors']:
        print(f"\nErrors: {len(results['errors'])}")
        for err in results['errors'][:5]:
            print(f"  - {err}")
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Evaluation complete!")


if __name__ == "__main__":
    main()



