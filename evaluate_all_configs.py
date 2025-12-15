#!/usr/bin/env python3
"""
Comprehensive evaluation script for all configurations:
1. Zero-shot (no CoT, no instruction FT)
2. Instruction fine-tuning (no CoT)
3. CoT zero-shot (CoT, no instruction FT)
4. CoT with instruction fine-tuning

Outputs results table matching the user's format.
"""

import torch
import json
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from peft import PeftModel

from multihead_model import create_multihead_model
from cot_prompts import build_cot_prompt, format_prompt_for_model
from data.vqa_data_loader import create_data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_base_model(model_name: str, device: str = "cuda"):
    """Load base model without any adapters."""
    logger.info(f"Loading base model: {model_name}")
    
    if "qwen3vl" in model_name.lower() or "qwen" in model_name.lower():
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        return model, processor, device
    elif "medgemma" in model_name.lower():
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        return model, processor, device
    elif "llava" in model_name.lower():
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        return model, processor, device
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_finetuned_model(checkpoint_path: str, model_type: str, device: str = "cuda"):
    """Load fine-tuned model (instruction FT, no CoT)."""
    logger.info(f"Loading fine-tuned model from: {checkpoint_path}")
    
    # Determine base model name
    if model_type == "qwen3vl":
        base_model = "Qwen/Qwen3-VL-8B-Instruct"
    elif model_type == "medgemma":
        base_model = "google/medgemma-4b"
    elif model_type == "llava_med":
        base_model = "microsoft/llava-med-v1.5-mistral-7b"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA adapters if checkpoint is a directory
    if Path(checkpoint_path).is_dir():
        try:
            model = PeftModel.from_pretrained(model, checkpoint_path)
            logger.info("Loaded LoRA adapters from checkpoint")
        except:
            logger.warning("Could not load LoRA adapters, using base model")
    elif Path(checkpoint_path).exists():
        # Try loading state dict
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'], strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded model weights from checkpoint")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}, using base model")
    
    return model, processor, device


def evaluate_zeroshot(
    model,
    processor,
    test_loader,
    device: str,
    use_cot: bool = False
) -> Dict:
    """Evaluate zero-shot (base model, no fine-tuning)."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Zero-shot evaluation"):
            images = batch.get('images', [])
            questions = batch.get('questions', [])
            answers = batch.get('answers', [])
            
            for i in range(len(questions)):
                if i >= len(images) or images[i] is None:
                    continue
                
                question = questions[i]
                answer = answers[i]
                
                try:
                    if use_cot:
                        # Build CoT prompt
                        prompt = build_cot_prompt(question, 'abnormality_detection', model_type="qwen3vl")
                        if isinstance(prompt, list):
                            text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                        else:
                            text = format_prompt_for_model(prompt, model_type="qwen3vl", processor=processor)
                    else:
                        # Simple prompt without CoT
                        text = question
                    
                    # Process inputs
                    inputs = processor(text=[text], images=[images[i]], return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
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


def evaluate_finetuned(
    model,
    processor,
    test_loader,
    device: str,
    use_cot: bool = False
) -> Dict:
    """Evaluate fine-tuned model (instruction FT, optionally with CoT)."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Fine-tuned evaluation"):
            images = batch.get('images', [])
            questions = batch.get('questions', [])
            answers = batch.get('answers', [])
            
            for i in range(len(questions)):
                if i >= len(images) or images[i] is None:
                    continue
                
                question = questions[i]
                answer = answers[i]
                
                try:
                    if use_cot:
                        # Use multi-head CoT model
                        prompt = build_cot_prompt(question, 'abnormality_detection', model_type="qwen3vl")
                        if isinstance(prompt, list):
                            text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                        else:
                            text = format_prompt_for_model(prompt, model_type="qwen3vl", processor=processor)
                    else:
                        # Standard instruction format
                        text = question
                    
                    # Process inputs
                    inputs = processor(text=[text], images=[images[i]], return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
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


def generate_results_table(results: Dict[str, Dict]) -> str:
    """Generate markdown table from results matching user's format."""
    # Organize results by model-dataset combination
    organized = {}
    
    for key, result in results.items():
        if '_cot_zeroshot' in key:
            base_key = key.replace('_cot_zeroshot', '')
            if base_key not in organized:
                organized[base_key] = {}
            organized[base_key]['cot_zeroshot'] = result
        else:
            if key not in organized:
                organized[key] = {}
            organized[key]['no_cot'] = result
    
    # Model name mapping
    model_names = {
        'qwen3vl': 'Qwen3-vl-8b-instruct',
        'medgemma': 'MedGemma-4B',
        'llava_med': 'LLaVA-Med v1.5 (Mistral-7B)'
    }
    
    dataset_names = {
        'kvasir': 'Kavisr',
        'endovis': 'Endovis18'
    }
    
    table = []
    table.append("| Model | dataset | COT | Zeroshot | Instruction fine tuning |")
    table.append("| :--- | :--- | :--- | :--- | :--- |")
    
    for key in sorted(organized.keys()):
        model_type, dataset = key.split('_', 1)
        model_name = model_names.get(model_type, model_type)
        dataset_name = dataset_names.get(dataset, dataset)
        data = organized[key]
        
        # Row 1: No CoT
        no_cot = data.get('no_cot', {})
        zeroshot_val = no_cot.get('zeroshot')
        finetuned_val = no_cot.get('finetuned')
        
        if zeroshot_val is not None:
            correct = no_cot.get('zeroshot_correct', 0)
            total = no_cot.get('zeroshot_total', 0)
            zeroshot_str = f"{zeroshot_val:.2f}% ({correct:,}/{total:,})"
        else:
            zeroshot_str = ""
        
        if finetuned_val is not None:
            correct = no_cot.get('finetuned_correct', 0)
            total = no_cot.get('finetuned_total', 0)
            finetuned_str = f"{finetuned_val:.2f}% ({correct:,}/{total:,})"
        else:
            finetuned_str = ""
        
        table.append(f"| {model_name} | {dataset_name} | no | {zeroshot_str} | {finetuned_str} |")
        
        # Row 2: CoT yes
        cot_zeroshot = data.get('cot_zeroshot', {})
        cot_finetuned = data.get('no_cot', {}).get('cot_finetuned')
        
        if cot_zeroshot.get('zeroshot') is not None:
            correct = cot_zeroshot.get('zeroshot_correct', 0)
            total = cot_zeroshot.get('zeroshot_total', 0)
            cot_zeroshot_str = f"{cot_zeroshot['zeroshot']:.2f}% ({correct:,}/{total:,})"
        else:
            cot_zeroshot_str = ""
        
        if cot_finetuned is not None:
            correct = data.get('no_cot', {}).get('cot_finetuned_correct', 0)
            total = data.get('no_cot', {}).get('cot_finetuned_total', 0)
            cot_finetuned_str = f"{cot_finetuned:.2f}% ({correct:,}/{total:,})"
        else:
            cot_finetuned_str = ""
        
        table.append(f"| {model_name} | {dataset_name} | yes | {cot_zeroshot_str} | {cot_finetuned_str} |")
    
    return "\n".join(table)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all configurations")
    parser.add_argument("--model-type", choices=["qwen3vl", "medgemma", "llava_med"], required=True)
    parser.add_argument("--dataset", choices=["kvasir", "endovis"], required=True)
    parser.add_argument("--test-data", required=True, help="Path to test data JSON")
    parser.add_argument("--image-base-path", required=True, help="Base path for images")
    parser.add_argument("--finetuned-checkpoint", help="Path to fine-tuned checkpoint (for instruction FT)")
    parser.add_argument("--cot-checkpoint", help="Path to CoT checkpoint (for CoT + instruction FT)")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--use-cot-zeroshot", action="store_true", default=True, help="Evaluate CoT zero-shot")
    
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
    
    # 1. Zero-shot (no CoT, no instruction FT)
    logger.info("Evaluating zero-shot (no CoT)...")
    model, processor, device = load_base_model(base_model, device)
    zeroshot_result = evaluate_zeroshot(model, processor, test_loader, device, use_cot=False)
    results[key] = {
        'zeroshot': zeroshot_result['accuracy'] * 100,
        'zeroshot_correct': zeroshot_result['correct'],
        'zeroshot_total': zeroshot_result['total'],
        'use_cot': False
    }
    logger.info(f"Zero-shot accuracy: {zeroshot_result['accuracy']:.2%} ({zeroshot_result['correct']}/{zeroshot_result['total']})")
    
    # 2. Instruction fine-tuning (no CoT)
    if args.finetuned_checkpoint:
        logger.info("Evaluating instruction fine-tuning (no CoT)...")
        model, processor, device = load_finetuned_model(args.finetuned_checkpoint, args.model_type, device)
        finetuned_result = evaluate_finetuned(model, processor, test_loader, device, use_cot=False)
        results[key]['finetuned'] = finetuned_result['accuracy'] * 100
        results[key]['finetuned_correct'] = finetuned_result['correct']
        results[key]['finetuned_total'] = finetuned_result['total']
        logger.info(f"Instruction FT accuracy: {finetuned_result['accuracy']:.2%} ({finetuned_result['correct']}/{finetuned_result['total']})")
    
    # 3. CoT zero-shot (CoT, no instruction FT)
    if args.use_cot_zeroshot:
        logger.info("Evaluating CoT zero-shot...")
        model, processor, device = load_base_model(base_model, device)
        cot_zeroshot_result = evaluate_zeroshot(model, processor, test_loader, device, use_cot=True)
        results[f"{key}_cot_zeroshot"] = {
            'zeroshot': cot_zeroshot_result['accuracy'] * 100,
            'zeroshot_correct': cot_zeroshot_result['correct'],
            'zeroshot_total': cot_zeroshot_result['total'],
            'use_cot': True
        }
        logger.info(f"CoT zero-shot accuracy: {cot_zeroshot_result['accuracy']:.2%} ({cot_zeroshot_result['correct']}/{cot_zeroshot_result['total']})")
    
    # 4. CoT with instruction fine-tuning
    if args.cot_checkpoint:
        logger.info("Evaluating CoT with instruction fine-tuning...")
        cot_model = create_multihead_model(
            base_checkpoint=args.cot_checkpoint,
            model_type=args.model_type,
            freeze_base=True
        )
        cot_model = cot_model.to(device)
        cot_model.eval()
        
        # Load question categories for CoT
        categories_file = Path("results/multihead_cot/question_categories.json")
        if categories_file.exists():
            with open(categories_file, 'r') as f:
                all_categories = json.load(f)
            question_categories = all_categories.get(args.dataset, {})
        else:
            question_categories = {}
        
        correct = 0
        total = 0
        
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
                        # Build CoT prompt
                        prompt = build_cot_prompt(question, category, model_type=args.model_type)
                        
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
        
        cot_finetuned_accuracy = correct / total if total > 0 else 0.0
        results[key]['cot_finetuned'] = cot_finetuned_accuracy * 100
        results[key]['cot_finetuned_correct'] = correct
        results[key]['cot_finetuned_total'] = total
        logger.info(f"CoT + instruction FT accuracy: {cot_finetuned_accuracy:.2%} ({correct}/{total})")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"evaluation_all_configs_{args.model_type}_{args.dataset}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate table
    table = generate_results_table(results)
    table_file = output_dir / f"results_table_{args.model_type}_{args.dataset}.md"
    with open(table_file, 'w') as f:
        f.write("# Evaluation Results\n\n")
        f.write(table)
    
    logger.info(f"Saved results to {results_file}")
    logger.info(f"Saved table to {table_file}")
    print("\n" + table)


if __name__ == "__main__":
    main()


