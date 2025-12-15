#!/usr/bin/env python3
"""
Evaluation Script for Experiment 2: Qwen-Reordered
Uses Qwen2-VL-7B-Instruct vision-language model with images + questions
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


def evaluate_model(model, processor, device, test_data, image_base_path, max_samples=None):
    """Evaluate model on test data."""
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    results = {
        'total': 0,
        'correct': 0,
        'accuracy': 0.0,
        'by_stage': defaultdict(lambda: {'total': 0, 'correct': 0, 'accuracy': 0.0}),
        'by_question_type': defaultdict(lambda: {'total': 0, 'correct': 0, 'accuracy': 0.0}),
        'predictions': [],
        'errors': []
    }
    
    for item in tqdm(test_data, desc="Evaluating"):
        # Use instruction field if available (for instructed datasets), otherwise use question
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
        
        results['total'] += 1
        results['correct'] += int(correct)
        
        # Track by stage
        stage_key = f"Stage {stage}" if stage in [1, 2, 3] else "Unknown"
        results['by_stage'][stage_key]['total'] += 1
        results['by_stage'][stage_key]['correct'] += int(correct)
        
        # Track by question type
        results['by_question_type'][question_type]['total'] += 1
        results['by_question_type'][question_type]['correct'] += int(correct)
        
        results['predictions'].append({
            'image_id': image_id,
            'question': item.get('question', ''),  # Original question
            'instruction': question,  # The instruction/prompt actually used
            'question_type': question_type,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'correct': correct,
            'stage': stage
        })
    
    # Calculate accuracy
    if results['total'] > 0:
        results['accuracy'] = (results['correct'] / results['total']) * 100
        
        for stage_data in results['by_stage'].values():
            if stage_data['total'] > 0:
                stage_data['accuracy'] = (stage_data['correct'] / stage_data['total']) * 100
        
        for qtype_data in results['by_question_type'].values():
            if qtype_data['total'] > 0:
                qtype_data['accuracy'] = (qtype_data['correct'] / qtype_data['total']) * 100
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Experiment 2: Qwen-Reordered")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Base model name")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (for testing)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXPERIMENT 2 EVALUATION: Qwen-Reordered")
    print("=" * 80)
    
    # Load model
    model, processor, device = load_model_with_lora(args.model_path, args.base_model)
    
    # Load test data
    print(f"\nLoading test data from: {args.test_data}")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    print(f"Test samples: {len(test_data)}")
    
    # Evaluate
    results = evaluate_model(model, processor, device, test_data, args.image_dir, args.max_samples)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nTotal samples: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    
    print(f"\nBy Stage:")
    for stage in sorted(results['by_stage'].keys()):
        stage_data = results['by_stage'][stage]
        print(f"  {stage}: {stage_data['correct']}/{stage_data['total']} ({stage_data['accuracy']:.2f}%)")
    
    print(f"\nBy Question Type:")
    for qtype in sorted(results['by_question_type'].keys()):
        qtype_data = results['by_question_type'][qtype]
        print(f"  {qtype}: {qtype_data['correct']}/{qtype_data['total']} ({qtype_data['accuracy']:.2f}%)")
    
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



