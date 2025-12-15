#!/usr/bin/env python3
"""
Evaluate epoch 3 checkpoint for multi-head CoT model.
"""

import argparse
import torch
import json
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
import logging
import os
import sys
import traceback

# Set HuggingFace cache to workspace directory
workspace_dir = Path(__file__).parent.absolute()
hf_cache_dir = workspace_dir / ".hf_cache"
hf_cache_dir.mkdir(exist_ok=True)
os.environ["HF_HOME"] = str(hf_cache_dir)
os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "transformers")
os.environ["HF_HUB_CACHE"] = str(hf_cache_dir / "hub")

from multihead_model import create_multihead_model
from cot_prompts import build_cot_prompt, format_prompt_for_model
from data.vqa_data_loader import create_data_loader
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Using HuggingFace cache directory: {hf_cache_dir}")


def load_question_categories(categories_file: str) -> Dict[str, Dict[str, str]]:
    """Load question categories mapping."""
    with open(categories_file, 'r') as f:
        return json.load(f)


def evaluate(
    model,
    test_loader,
    question_categories: Dict[str, str],
    dataset_name: str,
    processor,
    device: str,
    model_type: str
) -> Dict:
    """
    Evaluate model on test set.
    """
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
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
            images = batch.get('images', [])
            questions = batch.get('questions', [])
            answers = batch.get('answers', [])
            categories = batch.get('categories', [])
            
            for i in range(len(questions)):
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
                    prompt = build_cot_prompt(question, category, model_type=model_type)
                    
                    # Format for model
                    if isinstance(prompt, list):
                        # For qwen3vl, add image to messages
                        for msg in prompt:
                            if msg.get('role') == 'user' and isinstance(msg.get('content'), str):
                                msg['content'] = [
                                    {"type": "image", "image": images[i]},
                                    {"type": "text", "text": msg['content']}
                                ]
                        text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                    else:
                        text = format_prompt_for_model(prompt, model_type=model_type, processor=processor)
                    
                    # Process inputs
                    inputs = processor(text=[text], images=[images[i]], return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Forward pass through multi-head model
                    with torch.no_grad():
                        # Use the model's forward method which handles category routing
                        model_kwargs = {
                            'pixel_values': inputs.get('pixel_values'),
                            'input_ids': inputs.get('input_ids'),
                            'attention_mask': inputs.get('attention_mask'),
                            'category': category,
                            'previous_context': None
                        }
                        # Add image_grid_thw if present (for Qwen3-VL)
                        if 'image_grid_thw' in inputs:
                            model_kwargs['image_grid_thw'] = inputs['image_grid_thw']
                        
                        output = model(**model_kwargs)
                        
                        # Get logits and prediction
                        logits = output.get('logits', output)
                        if isinstance(logits, dict):
                            logits = logits.get('logits', list(logits.values())[0])
                        
                        # Get predicted token
                        predicted_token_id = logits.argmax(dim=-1).item()
                        predicted_token = processor.tokenizer.decode([predicted_token_id], skip_special_tokens=True)
                    
                    # Simple answer matching (can be improved)
                    answer_str = str(answer).lower().strip()
                    predicted_str = predicted_token.lower().strip()
                    
                    # Check if prediction matches answer
                    is_correct = answer_str in predicted_str or predicted_str in answer_str
                    
                    if is_correct:
                        correct += 1
                        category_correct[category] += 1
                    
                    total += 1
                    category_total[category] += 1
                    
                    all_predictions.append(predicted_str)
                    all_answers.append(answer_str)
                    
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue
    
    # Calculate accuracies
    overall_accuracy = correct / total if total > 0 else 0.0
    category_accuracies = {
        cat: category_correct[cat] / category_total[cat] if category_total[cat] > 0 else 0.0
        for cat in category_correct.keys()
    }
    
    return {
        'overall_accuracy': overall_accuracy,
        'correct': correct,
        'total': total,
        'category_accuracies': category_accuracies,
        'category_correct': category_correct,
        'category_total': category_total,
        'predictions': all_predictions[:10],  # First 10 for debugging
        'answers': all_answers[:10]
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate epoch 3 checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to epoch 3 checkpoint")
    parser.add_argument("--base_checkpoint", required=True, help="Path to base model checkpoint")
    parser.add_argument("--model_type", choices=["qwen3vl", "medgemma", "llava_med"], required=True)
    parser.add_argument("--dataset", choices=["kvasir", "endovis"], required=True)
    parser.add_argument("--test_data", required=True, help="Path to test data JSON")
    parser.add_argument("--image_base_path", required=True, help="Base path for images")
    parser.add_argument("--question_categories", default="question_categories.json")
    parser.add_argument("--output", required=True, help="Output directory for results")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("EVALUATING EPOCH 3 CHECKPOINT")
    logger.info("="*80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info("="*80)
    
    try:
        # Load question categories
        all_categories = load_question_categories(args.question_categories)
        question_categories = all_categories.get(args.dataset, {})
        
        # Load checkpoint
        logger.info("\n[1/4] Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        logger.info(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Load base model and create multi-head architecture
        logger.info("\n[2/4] Loading base model and creating multi-head architecture...")
        model = create_multihead_model(
            base_checkpoint=args.base_checkpoint,
            model_type=args.model_type,
            freeze_base=True
        )
        
        # Load trained weights
        state_dict = checkpoint['model_state_dict']
        # Handle DataParallel wrapper
        if any(k.startswith('module.') for k in state_dict.keys()):
            if not any(k.startswith('module.') for k in model.state_dict().keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        logger.info("✓ Model weights loaded")
        
        # Get processor
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if args.model_type == "qwen3vl":
            processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True, token=hf_token)
        elif args.model_type == "medgemma":
            processor = AutoProcessor.from_pretrained("google/medgemma-4b-it", trust_remote_code=True, token=hf_token)
        else:  # llava_med
            processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", trust_remote_code=True, token=hf_token)
        
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"✓ Model moved to {device}")
        
        # Create test loader
        logger.info("\n[3/4] Loading test data...")
        test_loader = create_data_loader(
            data_file=args.test_data,
            image_base_path=args.image_base_path,
            batch_size=1,
            shuffle=False,
            is_temporal=(args.dataset == "endovis")
        )
        logger.info(f"✓ Test data loaded: {len(test_loader.dataset)} samples")
        
        # Evaluate
        logger.info("\n[4/4] Evaluating model...")
        results = evaluate(
            model=model,
            test_loader=test_loader,
            question_categories=question_categories,
            dataset_name=args.dataset,
            processor=processor,
            device=device,
            model_type=args.model_type
        )
        
        # Print results
        logger.info("\n" + "="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
        logger.info(f"\nOverall Accuracy: {results['overall_accuracy']:.2%}")
        logger.info(f"Correct: {results['correct']} / {results['total']}")
        logger.info("\nPer-Category Accuracies:")
        for category, acc in results['category_accuracies'].items():
            logger.info(f"  {category}: {acc:.2%} ({results['category_correct'][category]}/{results['category_total'][category]})")
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"evaluation_epoch3_{args.model_type}_{args.dataset}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Saved results to {results_file}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("ERROR OCCURRED")
        logger.error("="*80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("\nFull traceback:")
        traceback.print_exc(file=sys.stderr)
        logger.error("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()


