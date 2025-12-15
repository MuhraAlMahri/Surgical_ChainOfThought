#!/usr/bin/env python3
"""
Experiment 5: Sequential Chain-of-Thought Evaluation
Cascading inference where each stage uses previous predictions as context.

Stage 1: Input → Model → Pre 1 (prediction)
Stage 2: Input + Pre 1 (from Stage 1) → Model → Pre 2 (prediction)
Stage 3: Input + Pre 1 (from Stage 1) + Pre 2 (from Stage 2) → Model → Pre 3 (final prediction)
"""

import json
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from tqdm import tqdm
import os
import argparse
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher


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


def load_model_with_lora(model_path: str = None, base_model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
    """Load vision-language model with optional LoRA adapter."""
    print(f"Loading base model: {base_model_name}")
    
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Check if model_path is provided and contains valid checkpoint
    use_lora = False
    if model_path and os.path.exists(model_path):
        # Find best checkpoint
        adapter_path = model_path
        if os.path.isdir(model_path):
            checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
            if checkpoints:
                checkpoint_numbers = [int(c.split("-")[1]) for c in checkpoints]
                best_checkpoint = f"checkpoint-{max(checkpoint_numbers)}"
                adapter_path = os.path.join(model_path, best_checkpoint)
                print(f"Using checkpoint: {best_checkpoint}")
        
        # Check if adapter config exists
        adapter_config = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(adapter_config):
            print(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            # Merge LoRA for inference
            model = model.merge_and_unload()
            use_lora = True
        else:
            print(f"Warning: No LoRA checkpoint found at {adapter_path}, using base model only")
            model = base_model
    else:
        print("No model checkpoint provided, using base model only")
        model = base_model
    
    model.eval()
    
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    
    device = next(model.parameters()).device
    print(f"✓ Model loaded on device: {device} ({'with merged LoRA' if use_lora else 'base model only'})")
    
    return model, processor, device


def extract_answer_from_response(generated_text: str, prompt_text: str) -> str:
    """Extract the answer from generated text."""
    # Remove prompt if present
    if prompt_text in generated_text:
        generated_text = generated_text.split(prompt_text)[-1].strip()
    
    # Remove common prefixes
    prefixes = ["### Response:", "Response:", "Answer:", "Answer", "assistant"]
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


def cascade_inference(model, processor, image, questions_by_stage, device, max_new_tokens=128):
    """
    Sequential cascading inference with context from previous stages.
    
    Args:
        model: Trained VL model
        processor: Model processor
        image: PIL Image
        questions_by_stage: Dict with keys 1, 2, 3 containing question text
        device: Device to use
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        predictions: Dict with keys 1, 2, 3 containing predictions [Pre1, Pre2, Pre3]
    """
    predictions = {}
    
    # Stage 1: Input → Model → Pre 1
    q1 = questions_by_stage.get(1, "")
    conversation_1 = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": q1}
        ]
    }]
    
    text_1 = processor.apply_chat_template(conversation_1, tokenize=False, add_generation_prompt=True)
    inputs_1 = processor(text=[text_1], images=[image], return_tensors="pt")
    inputs_1 = {k: v.to(device) for k, v in inputs_1.items()}
    
    with torch.no_grad():
        outputs_1 = model.generate(**inputs_1, max_new_tokens=max_new_tokens)
    
    prediction_1 = processor.decode(outputs_1[0], skip_special_tokens=True)
    prediction_1 = extract_answer_from_response(prediction_1, text_1)
    predictions[1] = prediction_1
    
    # Stage 2: Input + Pre 1 → Model → Pre 2
    q2 = questions_by_stage.get(2, "")
    stage_2_text = f"""Previous observation (Stage 1): {prediction_1}

Current question: {q2}"""
    
    conversation_2 = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": stage_2_text}
        ]
    }]
    
    text_2 = processor.apply_chat_template(conversation_2, tokenize=False, add_generation_prompt=True)
    inputs_2 = processor(text=[text_2], images=[image], return_tensors="pt")
    inputs_2 = {k: v.to(device) for k, v in inputs_2.items()}
    
    with torch.no_grad():
        outputs_2 = model.generate(**inputs_2, max_new_tokens=max_new_tokens)
    
    prediction_2 = processor.decode(outputs_2[0], skip_special_tokens=True)
    prediction_2 = extract_answer_from_response(prediction_2, text_2)
    predictions[2] = prediction_2
    
    # Stage 3: Input + Pre 1 + Pre 2 → Model → Pre 3
    q3 = questions_by_stage.get(3, "")
    stage_3_text = f"""Previous observations:
- Stage 1: {prediction_1}
- Stage 2: {prediction_2}

Current question: {q3}"""
    
    conversation_3 = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": stage_3_text}
        ]
    }]
    
    text_3 = processor.apply_chat_template(conversation_3, tokenize=False, add_generation_prompt=True)
    inputs_3 = processor(text=[text_3], images=[image], return_tensors="pt")
    inputs_3 = {k: v.to(device) for k, v in inputs_3.items()}
    
    with torch.no_grad():
        outputs_3 = model.generate(**inputs_3, max_new_tokens=max_new_tokens)
    
    prediction_3 = processor.decode(outputs_3[0], skip_special_tokens=True)
    prediction_3 = extract_answer_from_response(prediction_3, text_3)
    predictions[3] = prediction_3
    
    return predictions


def evaluate_sequential_cot(
    model_path: str,
    test_data_path: str,
    image_base_path: str,
    output_path: str,
    base_model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
):
    """Evaluate Sequential Chain-of-Thought model."""
    
    # Load model
    model, processor, device = load_model_with_lora(model_path, base_model_name)
    
    # Load test data
    print(f"\nLoading test data from: {test_data_path}")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"Test samples: {len(test_data)}")
    
    # Group questions by image and stage
    image_to_stages = defaultdict(lambda: {1: None, 2: None, 3: None})
    for item in test_data:
        image_id = item.get('image_id', '')
        stage = item.get('stage')
        if stage in [1, 2, 3] and image_id:
            image_to_stages[image_id][stage] = item
    
    print(f"Unique images: {len(image_to_stages)}")
    
    # Evaluate
    results = {
        'total_images': 0,
        'stage1': {'total': 0, 'correct': 0, 'accuracy': 0.0},
        'stage2': {'total': 0, 'correct': 0, 'accuracy': 0.0},
        'stage3': {'total': 0, 'correct': 0, 'accuracy': 0.0},
        'predictions': [],
        'errors': []
    }
    
    # Process each image
    for image_id, stages_dict in tqdm(image_to_stages.items(), desc="Evaluating"):
        # Check if we have all 3 stages
        if not all(stages_dict[i] for i in [1, 2, 3]):
            missing = [i for i in [1, 2, 3] if not stages_dict[i]]
            results['errors'].append(f"Image {image_id}: Missing stages {missing}")
            continue
        
        # Get questions by stage
        questions_by_stage = {
            1: stages_dict[1].get('question', ''),
            2: stages_dict[2].get('question', ''),
            3: stages_dict[3].get('question', '')
        }
        
        # Get ground truth answers
        gt_by_stage = {
            1: stages_dict[1].get('answer', '').strip(),
            2: stages_dict[2].get('answer', '').strip(),
            3: stages_dict[3].get('answer', '').strip()
        }
        
        # Get image path
        image_filename = stages_dict[1].get('image_filename', '')
        if not image_filename:
            image_filename = f"{image_id}.jpg"
        
        image_path = os.path.join(image_base_path, image_filename)
        
        if not os.path.exists(image_path):
            results['errors'].append(f"Image {image_id}: File not found - {image_path}")
            continue
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            results['errors'].append(f"Image {image_id}: Load error - {str(e)}")
            continue
        
        # Run cascading inference
        try:
            predictions = cascade_inference(model, processor, image, questions_by_stage, device)
            
            pred1 = predictions.get(1, "")
            pred2 = predictions.get(2, "")
            pred3 = predictions.get(3, "")
            
            # Evaluate each stage
            results['total_images'] += 1
            
            # Stage 1
            correct1 = smart_match(pred1, gt_by_stage[1])
            results['stage1']['total'] += 1
            results['stage1']['correct'] += int(correct1)
            
            # Stage 2
            correct2 = smart_match(pred2, gt_by_stage[2])
            results['stage2']['total'] += 1
            results['stage2']['correct'] += int(correct2)
            
            # Stage 3
            correct3 = smart_match(pred3, gt_by_stage[3])
            results['stage3']['total'] += 1
            results['stage3']['correct'] += int(correct3)
            
            # Store prediction details
            results['predictions'].append({
                'image_id': image_id,
                'stage1': {
                    'question': questions_by_stage[1],
                    'prediction': pred1,
                    'ground_truth': gt_by_stage[1],
                    'correct': correct1
                },
                'stage2': {
                    'question': questions_by_stage[2],
                    'prediction': pred2,
                    'ground_truth': gt_by_stage[2],
                    'correct': correct2,
                    'context_used': {'pre1': pred1}
                },
                'stage3': {
                    'question': questions_by_stage[3],
                    'prediction': pred3,
                    'ground_truth': gt_by_stage[3],
                    'correct': correct3,
                    'context_used': {'pre1': pred1, 'pre2': pred2}
                }
            })
            
        except Exception as e:
            results['errors'].append(f"Image {image_id}: Inference error - {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate accuracies
    if results['stage1']['total'] > 0:
        results['stage1']['accuracy'] = (results['stage1']['correct'] / results['stage1']['total']) * 100
    if results['stage2']['total'] > 0:
        results['stage2']['accuracy'] = (results['stage2']['correct'] / results['stage2']['total']) * 100
    if results['stage3']['total'] > 0:
        results['stage3']['accuracy'] = (results['stage3']['correct'] / results['stage3']['total']) * 100
    
    # Overall accuracy (average of all 3 stages)
    overall_correct = results['stage1']['correct'] + results['stage2']['correct'] + results['stage3']['correct']
    overall_total = results['stage1']['total'] + results['stage2']['total'] + results['stage3']['total']
    results['overall'] = {
        'total': overall_total,
        'correct': overall_correct,
        'accuracy': (overall_correct / overall_total * 100) if overall_total > 0 else 0.0
    }
    
    # Save results
    print(f"\nSaving results to: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("SEQUENTIAL CHAIN-OF-THOUGHT EVALUATION RESULTS")
    print("="*70)
    print(f"\nTotal images evaluated: {results['total_images']}")
    print(f"\nStage 1 (Input → Pre1):")
    print(f"  Accuracy: {results['stage1']['correct']}/{results['stage1']['total']} ({results['stage1']['accuracy']:.2f}%)")
    print(f"\nStage 2 (Input + Pre1 → Pre2):")
    print(f"  Accuracy: {results['stage2']['correct']}/{results['stage2']['total']} ({results['stage2']['accuracy']:.2f}%)")
    print(f"\nStage 3 (Input + Pre1 + Pre2 → Pre3):")
    print(f"  Accuracy: {results['stage3']['correct']}/{results['stage3']['total']} ({results['stage3']['accuracy']:.2f}%)")
    print(f"\nOverall Accuracy: {results['overall']['correct']}/{results['overall']['total']} ({results['overall']['accuracy']:.2f}%)")
    
    if results['errors']:
        print(f"\nErrors: {len(results['errors'])}")
        for err in results['errors'][:10]:
            print(f"  - {err}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sequential Chain-of-Thought Model")
    parser.add_argument("--model_path", type=str, default=None, 
                       help="Path to trained model checkpoint (e.g., exp1_random_baseline or exp2_qwen_reordered)")
    parser.add_argument("--test_data", type=str, required=True, 
                       help="Path to test JSON file (must have stage field: 1, 2, or 3)")
    parser.add_argument("--image_dir", type=str, required=True, 
                       help="Directory containing images")
    parser.add_argument("--output", type=str, default="exp5_sequential_cot_results.json", 
                       help="Output JSON file")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-7B-Instruct", 
                       help="Base model name")
    parser.add_argument("--max_new_tokens", type=int, default=128, 
                       help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    evaluate_sequential_cot(
        model_path=args.model_path,
        test_data_path=args.test_data,
        image_base_path=args.image_dir,
        output_path=args.output,
        base_model_name=args.base_model
    )


if __name__ == "__main__":
    main()

