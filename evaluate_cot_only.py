#!/usr/bin/env python3
"""
Evaluate only CoT configurations:
1. CoT zero-shot (CoT, no instruction FT)
2. CoT with instruction fine-tuning

Combines with existing baseline results.
"""

import torch
import json
import os
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

from multihead_model import create_multihead_model
from cot_prompts import build_cot_prompt, format_prompt_for_model
from data.vqa_data_loader import create_data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_base_model(model_name: str, device: str = "cuda"):
    """Load base model without any adapters."""
    logger.info(f"Loading base model: {model_name}")
    
    # Get HuggingFace token from environment
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    if "qwen3vl" in model_name.lower() or "qwen3" in model_name.lower():
        from transformers import AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True,
            token=hf_token
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
        return model, processor, device
    elif "medgemma" in model_name.lower():
        processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True,
            token=hf_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
        return model, processor, device
    elif "llava" in model_name.lower():
        from transformers import AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True,
            token=hf_token
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
        return model, processor, device
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_cot_zeroshot(
    model,
    processor,
    test_loader,
    device: str,
    model_type: str
) -> Dict:
    """Evaluate CoT zero-shot (base model with CoT prompts)."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="CoT zero-shot evaluation"):
            images = batch.get('images', [])
            questions = batch.get('questions', [])
            answers = batch.get('answers', [])
            
            for i in range(len(questions)):
                if i >= len(images) or images[i] is None:
                    continue
                
                question = questions[i]
                answer = answers[i]
                
                try:
                    # Build CoT prompt
                    prompt = build_cot_prompt(question, 'abnormality_detection', model_type=model_type)
                    
                    # Process inputs based on model type
                    if model_type == "qwen3vl":
                        # Qwen3-VL expects list of messages with image embedded
                        if isinstance(prompt, list):
                            # Create a copy to avoid modifying the original
                            messages = []
                            for msg in prompt:
                                msg_copy = msg.copy()
                                if msg_copy.get('role') == 'user':
                                    content = msg_copy.get('content', '')
                                    if isinstance(content, str):
                                        msg_copy['content'] = [
                                            {"type": "image", "image": images[i]},
                                            {"type": "text", "text": content}
                                        ]
                                    elif isinstance(content, list):
                                        # Already formatted, just ensure image is included
                                        has_image = any(isinstance(item, dict) and item.get('type') == 'image' for item in content)
                                        if not has_image:
                                            content = [{"type": "image", "image": images[i]}] + content
                                        msg_copy['content'] = content
                                messages.append(msg_copy)
                            inputs = processor(text=messages, images=[images[i]], return_tensors="pt")
                        else:
                            # Convert to message format
                            messages = [{"role": "user", "content": [
                                {"type": "image", "image": images[i]},
                                {"type": "text", "text": str(prompt)}
                            ]}]
                            inputs = processor(text=messages, images=[images[i]], return_tensors="pt")
                    else:
                        if isinstance(prompt, list):
                            text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                        else:
                            text = format_prompt_for_model(prompt, model_type=model_type, processor=processor)
                        inputs = processor(text=[text], images=[images[i]], return_tensors="pt")
                    
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    
                    # Generate
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False
                    )
                    
                    # Decode
                    generated_text = processor.batch_decode(
                        generated_ids[:, inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )[0]
                    
                    prediction = generated_text.strip()
                    
                    # Check correctness
                    if check_correctness(prediction, answer):
                        correct += 1
                    total += 1
                    
                except Exception as e:
                    logger.warning(f"Error evaluating sample {i}: {e}")
                    continue
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def evaluate_cot_finetuned(
    cot_model,
    test_loader,
    device: str,
    model_type: str,
    dataset: str
) -> Dict:
    """Evaluate CoT with instruction fine-tuning."""
    cot_model.eval()
    correct = 0
    total = 0
    
    # Load question categories
    categories_file = Path("results/multihead_cot/question_categories.json")
    if categories_file.exists():
        with open(categories_file, 'r') as f:
            all_categories = json.load(f)
        question_categories = all_categories.get(dataset, {})
    else:
        question_categories = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="CoT + instruction FT evaluation"):
            images = batch.get('images', [])
            questions = batch.get('questions', [])
            answers = batch.get('answers', [])
            
            for i in range(len(questions)):
                if i >= len(images) or images[i] is None:
                    continue
                
                question = questions[i]
                answer = answers[i]
                
                # Get category
                category = question_categories.get(question, 'abnormality_detection')
                if category == 1 or category == "1":
                    category = 'abnormality_detection'
                elif category == 2 or category == "2":
                    category = 'characteristics'
                elif category == 3 or category == "3":
                    category = 'treatment'
                
                try:
                    # Generate using multi-head model
                    result = cot_model.generate(
                        images=[images[i]],
                        prompt=question,
                        category=category,
                        max_new_tokens=256,
                        temperature=0.7
                    )
                    
                    prediction = result.get('generated_text', '').strip()
                    if not prediction and 'answer' in result:
                        prediction = result['answer'].strip()
                    
                    # Check correctness
                    if check_correctness(prediction, answer):
                        correct += 1
                    total += 1
                    
                except Exception as e:
                    logger.warning(f"Error evaluating sample {i}: {e}")
                    continue
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def check_correctness(prediction: str, answer: str) -> bool:
    """Check if prediction matches answer."""
    pred_lower = prediction.lower().strip()
    ans_lower = answer.lower().strip()
    
    # Exact match
    if pred_lower == ans_lower:
        return True
    
    # Check if answer is contained in prediction
    if ans_lower in pred_lower or pred_lower in ans_lower:
        return True
    
    # For multi-label answers
    if ';' in ans_lower:
        ans_parts = [a.strip() for a in ans_lower.split(';')]
        return all(part in pred_lower for part in ans_parts)
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate CoT configurations only")
    parser.add_argument("--model-type", choices=["qwen3vl", "medgemma", "llava_med"], required=True)
    parser.add_argument("--dataset", choices=["kvasir", "endovis"], required=True)
    parser.add_argument("--test-data", required=True, help="Path to test data JSON")
    parser.add_argument("--image-base-path", required=True, help="Base path for images")
    parser.add_argument("--cot-checkpoint", help="Path to CoT checkpoint (for CoT + instruction FT)")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--baseline-results", help="Path to JSON file with existing baseline results")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    # Determine base model name
    if args.model_type == "qwen3vl":
        base_model = "Qwen/Qwen3-VL-8B-Instruct"
    elif args.model_type == "medgemma":
        base_model = "google/medgemma-4b"
    elif args.model_type == "llava_med":
        base_model = "microsoft/llava-med-v1.5-mistral-7b"
    
    # Create test loader
    test_loader = create_data_loader(
        data_file=args.test_data,
        image_base_path=args.image_base_path,
        batch_size=1,
        shuffle=False,
        is_temporal=(args.dataset == "endovis")
    )
    
    key = f"{args.model_type}_{args.dataset}"
    
    # Load existing baseline results if provided
    baseline_results = {}
    if args.baseline_results and Path(args.baseline_results).exists():
        with open(args.baseline_results, 'r') as f:
            baseline_results = json.load(f)
        logger.info(f"Loaded baseline results from {args.baseline_results}")
    
    # Initialize results with baseline if available
    if key in baseline_results:
        results[key] = baseline_results[key].copy()
    else:
        results[key] = {}
    
    # 1. CoT zero-shot (CoT, no instruction FT)
    logger.info("Evaluating CoT zero-shot...")
    model, processor, device = load_base_model(base_model, device)
    cot_zeroshot_result = evaluate_cot_zeroshot(model, processor, test_loader, device, args.model_type)
    results[key]['cot_zeroshot'] = cot_zeroshot_result['accuracy'] * 100
    results[key]['cot_zeroshot_correct'] = cot_zeroshot_result['correct']
    results[key]['cot_zeroshot_total'] = cot_zeroshot_result['total']
    logger.info(f"CoT zero-shot accuracy: {cot_zeroshot_result['accuracy']:.2%} ({cot_zeroshot_result['correct']}/{cot_zeroshot_result['total']})")
    
    # 2. CoT with instruction fine-tuning
    if args.cot_checkpoint:
        logger.info("Evaluating CoT with instruction fine-tuning...")
        # Load base model first, then load checkpoint
        if args.model_type == "qwen3vl":
            base_model_name = "Qwen/Qwen3-VL-8B-Instruct"
        elif args.model_type == "medgemma":
            base_model_name = "google/medgemma-4b"
        elif args.model_type == "llava_med":
            base_model_name = "microsoft/llava-med-v1.5-mistral-7b"
        
        # Create model with base model name
        # Load checkpoint first, then create model to avoid OOM
        checkpoint_path = Path(args.cot_checkpoint)
        checkpoint_data = None
        if checkpoint_path.exists() and checkpoint_path.suffix == '.pt':
            logger.info(f"Loading checkpoint from {args.cot_checkpoint} (CPU)")
            checkpoint_data = torch.load(args.cot_checkpoint, map_location='cpu')
            logger.info("Checkpoint loaded to CPU")
        
        # Create model - it will load on GPU via device_map="auto"
        logger.info("Creating multi-head model...")
        cot_model = create_multihead_model(
            base_checkpoint=base_model_name,
            model_type=args.model_type,
            freeze_base=True
        )
        
        # Load checkpoint state dict if available
        if checkpoint_data is not None:
            logger.info("Loading checkpoint state dict...")
            if 'model_state_dict' in checkpoint_data:
                cot_model.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
            else:
                cot_model.load_state_dict(checkpoint_data, strict=False)
            logger.info("Checkpoint loaded successfully")
            # Clear checkpoint from memory
            del checkpoint_data
            torch.cuda.empty_cache()
        
        cot_finetuned_result = evaluate_cot_finetuned(
            cot_model, test_loader, device, args.model_type, args.dataset
        )
        results[key]['cot_finetuned'] = cot_finetuned_result['accuracy'] * 100
        results[key]['cot_finetuned_correct'] = cot_finetuned_result['correct']
        results[key]['cot_finetuned_total'] = cot_finetuned_result['total']
        logger.info(f"CoT + instruction FT accuracy: {cot_finetuned_result['accuracy']:.2%} ({cot_finetuned_result['correct']}/{cot_finetuned_result['total']})")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"cot_results_{args.model_type}_{args.dataset}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to {results_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("CoT EVALUATION RESULTS")
    print("="*80)
    print(f"\nModel: {args.model_type}, Dataset: {args.dataset}")
    print(f"\nCoT Zero-shot: {results[key].get('cot_zeroshot', 0):.2f}% ({results[key].get('cot_zeroshot_correct', 0)}/{results[key].get('cot_zeroshot_total', 0)})")
    if 'cot_finetuned' in results[key]:
        print(f"CoT + Instruction FT: {results[key]['cot_finetuned']:.2f}% ({results[key]['cot_finetuned_correct']}/{results[key]['cot_finetuned_total']})")
    print("="*80)


if __name__ == "__main__":
    main()

