#!/usr/bin/env python3
"""
Evaluation Script for Experiment 3: CXRTReK Sequential
Evaluates 3 separate models on their respective stage-specific test sets
CORRECTED VERSION - Uses proper stage-specific test datasets
"""

import json
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from tqdm import tqdm
import os
from pathlib import Path
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


def load_model_with_lora(model_path: str, base_model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
    """Load vision-language model with LoRA adapter."""
    print(f"Loading base model: {base_model_name}")
    
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()
    model.eval()
    
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    
    device = next(model.parameters()).device
    print(f"✓ Model loaded on device: {device}")
    
    return model, processor, device


def evaluate_stage(model, processor, device, test_data, image_base_path, stage_num, max_samples=None):
    """Evaluate a single stage model on its test set."""
    
    if max_samples:
        test_data = test_data[:max_samples]
    
    results = {
        'stage': stage_num,
        'total': 0,
        'correct': 0,
        'accuracy': 0.0,
        'predictions': [],
        'errors': []
    }
    
    for item in tqdm(test_data, desc=f"Stage {stage_num}"):
        question = item.get('question', '')
        ground_truth = item.get('answer', '').strip()
        image_filename = item.get('image_filename', '')
        image_id = item.get('image_id', '')
        
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
            
            generated_ids = outputs[0][input_len:]
            prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()
            
        except Exception as e:
            results['errors'].append(f"Generation error: {str(e)}")
            prediction = ""
        
        # Evaluate
        correct = smart_match(prediction, ground_truth)
        
        results['total'] += 1
        results['correct'] += int(correct)
        
        results['predictions'].append({
            'image_id': image_id,
            'question': question,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'correct': correct,
            'stage': stage_num
        })
    
    if results['total'] > 0:
        results['accuracy'] = (results['correct'] / results['total']) * 100
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Experiment 3: CXRTReK Sequential (CORRECTED)")
    parser.add_argument("--model_stage1", type=str, required=True, help="Path to Stage 1 model")
    parser.add_argument("--model_stage2", type=str, required=True, help="Path to Stage 2 model")
    parser.add_argument("--model_stage3", type=str, required=True, help="Path to Stage 3 model")
    parser.add_argument("--test_stage1", type=str, required=True, help="Path to Stage 1 test JSON")
    parser.add_argument("--test_stage2", type=str, required=True, help="Path to Stage 2 test JSON")
    parser.add_argument("--test_stage3", type=str, required=True, help="Path to Stage 3 test JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Base model name")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per stage")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EXPERIMENT 3 EVALUATION (CORRECTED): CXRTReK Sequential")
    print("Using stage-specific test datasets")
    print("=" * 80)
    
    all_results = {
        'by_stage': {},
        'overall': {
            'total': 0,
            'correct': 0,
            'accuracy': 0.0
        },
        'all_predictions': [],
        'errors': []
    }
    
    # Evaluate each stage with its corresponding model and test set
    stage_configs = [
        (1, args.model_stage1, args.test_stage1),
        (2, args.model_stage2, args.test_stage2),
        (3, args.model_stage3, args.test_stage3)
    ]
    
    for stage_num, model_path, test_path in stage_configs:
        print(f"\n{'=' * 80}")
        print(f"Stage {stage_num} - Clinical {'Initial Assessment' if stage_num==1 else 'Findings Identification' if stage_num==2 else 'Clinical Context'}")
        print(f"Model: {model_path}")
        print(f"Test Data: {test_path}")
        print(f"{'=' * 80}")
        
        # Load test data for this stage
        with open(test_path, 'r') as f:
            stage_test_data = json.load(f)
        print(f"Test samples: {len(stage_test_data)}")
        
        # Load model
        model, processor, device = load_model_with_lora(model_path, args.base_model)
        
        # Evaluate
        stage_results = evaluate_stage(model, processor, device, stage_test_data, args.image_dir, stage_num, args.max_samples)
        
        # Store results
        all_results['by_stage'][f'Stage {stage_num}'] = {
            'total': stage_results['total'],
            'correct': stage_results['correct'],
            'accuracy': stage_results['accuracy']
        }
        
        all_results['all_predictions'].extend(stage_results['predictions'])
        all_results['overall']['total'] += stage_results['total']
        all_results['overall']['correct'] += stage_results['correct']
        all_results['errors'].extend(stage_results['errors'])
        
        print(f"\nStage {stage_num} Results:")
        print(f"  Total: {stage_results['total']}")
        print(f"  Correct: {stage_results['correct']}")
        print(f"  Accuracy: {stage_results['accuracy']:.2f}%")
        
        if stage_results['errors']:
            print(f"  Errors: {len(stage_results['errors'])}")
        
        # Free memory
        del model, processor
        torch.cuda.empty_cache()
    
    # Calculate overall accuracy
    if all_results['overall']['total'] > 0:
        all_results['overall']['accuracy'] = (all_results['overall']['correct'] / all_results['overall']['total']) * 100
    
    # Print final results
    print(f"\n{'=' * 80}")
    print("OVERALL RESULTS")
    print(f"{'=' * 80}")
    print(f"\nTotal samples: {all_results['overall']['total']}")
    print(f"Correct: {all_results['overall']['correct']}")
    print(f"Overall Accuracy: {all_results['overall']['accuracy']:.2f}%")
    
    print(f"\nBy Stage:")
    for stage_name, stage_data in all_results['by_stage'].items():
        print(f"  {stage_name}: {stage_data['correct']}/{stage_data['total']} ({stage_data['accuracy']:.2f}%)")
    
    # Save results
    print(f"\nSaving results to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("✓ Evaluation complete!")


if __name__ == "__main__":
    main()


