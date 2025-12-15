#!/usr/bin/env python3
"""
Evaluate epoch checkpoint for multi-head CoT model - FIXED VERSION.

FIXED: Uses model.generate() instead of single-token argmax for proper answer generation.
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
import time
from datetime import datetime

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


def load_checkpoint(checkpoint_file: Path) -> Optional[Dict]:
    """Load evaluation checkpoint if it exists."""
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            logger.info(f"✓ Found checkpoint: {checkpoint['processed_samples']} samples already processed")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
    return None


def save_checkpoint(
    checkpoint_file: Path,
    processed_samples: int,
    correct: int,
    total: int,
    category_correct: Dict[str, int],
    category_total: Dict[str, int],
    all_predictions: List[str],
    all_answers: List[str],
    processed_indices: List[int],
    all_questions: List[str] = None,
    all_cot_reasoning: List[str] = None,
    all_full_generated: List[str] = None,
    all_categories: List[str] = None
):
    """Save evaluation checkpoint."""
    checkpoint = {
        'processed_samples': processed_samples,
        'correct': correct,
        'total': total,
        'category_correct': category_correct,
        'category_total': category_total,
        'all_predictions': all_predictions,
        'all_answers': all_answers,
        'processed_indices': processed_indices,
        'last_saved': datetime.now().isoformat()
    }
    if all_questions is not None:
        checkpoint['all_questions'] = all_questions
    if all_cot_reasoning is not None:
        checkpoint['all_cot_reasoning'] = all_cot_reasoning
    if all_full_generated is not None:
        checkpoint['all_full_generated'] = all_full_generated
    if all_categories is not None:
        checkpoint['all_categories'] = all_categories
    try:
        # Save to temporary file first, then rename (atomic write)
        temp_file = checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        temp_file.replace(checkpoint_file)
        logger.debug(f"✓ Saved checkpoint: {processed_samples} samples")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def evaluate(
    model,
    test_loader,
    question_categories: Dict[str, str],
    dataset_name: str,
    processor,
    device: str,
    model_type: str,
    checkpoint_file: Optional[Path] = None,
    save_interval: int = 100  # Save checkpoint every N samples
) -> Dict:
    """
    Evaluate model on test set - FIXED to use generate() for full answers.
    Supports resume from checkpoint.
    """
    model.eval()
    
    # Try to load checkpoint
    checkpoint_data = None
    processed_indices = set()
    if checkpoint_file:
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data:
            processed_indices = set(checkpoint_data.get('processed_indices', []))
            logger.info(f"Resuming from checkpoint: {len(processed_indices)} samples already processed")
    
    # Initialize counters from checkpoint or zero
    if checkpoint_data:
        correct = checkpoint_data.get('correct', 0)
        total = checkpoint_data.get('total', 0)
        category_correct = checkpoint_data.get('category_correct', {
            'abnormality_detection': 0,
            'characteristics': 0,
            'treatment': 0
        })
        category_total = checkpoint_data.get('category_total', {
            'abnormality_detection': 0,
            'characteristics': 0,
            'treatment': 0
        })
        all_predictions = checkpoint_data.get('all_predictions', [])
        all_answers = checkpoint_data.get('all_answers', [])
        all_questions = checkpoint_data.get('all_questions', [])
        all_cot_reasoning = checkpoint_data.get('all_cot_reasoning', [])
        all_full_generated = checkpoint_data.get('all_full_generated', [])
        all_categories = checkpoint_data.get('all_categories', [])
        start_sample_idx = checkpoint_data.get('processed_samples', 0)
    else:
        correct = 0
        total = 0
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
        all_cot_reasoning = []
        all_full_generated = []
        all_categories = []
        start_sample_idx = 0
    
    # Track current sample index globally
    global_sample_idx = start_sample_idx
    last_save_time = time.time()
    save_interval_seconds = 300  # Also save every 5 minutes
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {dataset_name}", initial=start_sample_idx)):
            images = batch.get('images', [])
            questions = batch.get('questions', [])
            answers = batch.get('answers', [])
            categories = batch.get('categories', [])
            
            for i in range(len(questions)):
                if i >= len(images) or images[i] is None:
                    global_sample_idx += 1
                    continue
                
                # Check if this sample was already processed
                if global_sample_idx in processed_indices:
                    logger.debug(f"Skipping already processed sample {global_sample_idx}")
                    global_sample_idx += 1
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
                    
                    # FIXED: Use model.generate() instead of forward + argmax
                    # This generates full answers instead of single tokens
                    # For Qwen3-VL, we need to process inputs first to get image_grid_thw
                    if model_type == "qwen3vl":
                        # Process inputs to get image_grid_thw
                        inputs = processor(text=[text], images=[images[i]], return_tensors="pt")
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Use the base model's generate method directly with proper inputs
                        # Access the underlying model
                        base_model = model.model.base_model if hasattr(model.model, 'base_model') else model.model
                        
                        # Generate with proper Qwen3-VL parameters
                        with torch.no_grad():
                            generate_kwargs = {
                                'pixel_values': inputs['pixel_values'],
                                'input_ids': inputs['input_ids'],
                                'attention_mask': inputs.get('attention_mask'),
                                'max_new_tokens': 50,
                                'do_sample': False,
                                'temperature': 0.0
                            }
                            # Add image_grid_thw if present (required for Qwen3-VL)
                            if 'image_grid_thw' in inputs:
                                generate_kwargs['image_grid_thw'] = inputs['image_grid_thw']
                            
                            generated_ids = base_model.generate(**generate_kwargs)
                            
                            # Decode only the new tokens (excluding input prompt)
                            input_length = inputs['input_ids'].shape[1]
                            new_tokens = generated_ids[0, input_length:]
                            full_generated_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
                            
                            generated_output = {'answer': full_generated_text, 'generated_text': full_generated_text, 'full_text': full_generated_text}
                    else:
                        # For other models, use the model's generate method
                        generated_output = model.generate(
                            images=[images[i]],
                            prompt=text,
                            category=category,
                            previous_context=None,
                            max_new_tokens=50,
                            temperature=0.0,
                            do_sample=False
                        )
                    
                    # Extract full generated text and COT reasoning
                    if isinstance(generated_output, dict):
                        full_generated_text = generated_output.get('full_text', generated_output.get('generated_text', generated_output.get('answer', '')))
                        cot_reasoning = generated_output.get('reasoning', '')
                        predicted_answer = generated_output.get('answer', '')
                    else:
                        full_generated_text = str(generated_output)
                        cot_reasoning = ''
                        predicted_answer = ''
                    
                    # Store full generated text (contains COT + answer)
                    full_generated_text = str(full_generated_text).strip()
                    
                    # Extract COT reasoning and answer if not already extracted
                    if not cot_reasoning and not predicted_answer and full_generated_text:
                        # Try to extract reasoning from common patterns
                        if "Reasoning:" in full_generated_text:
                            parts = full_generated_text.split("Reasoning:", 1)
                            if len(parts) == 2:
                                cot_reasoning = parts[1].split("Answer:")[0].strip()
                                if "Answer:" in parts[1]:
                                    predicted_answer = parts[1].split("Answer:")[1].strip()
                                else:
                                    predicted_answer = parts[1].strip()
                            else:
                                cot_reasoning = ""
                                predicted_answer = full_generated_text
                        elif "Answer:" in full_generated_text:
                            # If there's an Answer: marker, everything before is reasoning
                            parts = full_generated_text.split("Answer:", 1)
                            cot_reasoning = parts[0].strip()
                            predicted_answer = parts[1].strip() if len(parts) > 1 else ""
                        else:
                            # No clear markers, try to extract reasoning from structure
                            lines = full_generated_text.split('\n')
                            if len(lines) > 1:
                                # Assume last line is answer, rest is reasoning
                                cot_reasoning = '\n'.join(lines[:-1]).strip()
                                predicted_answer = lines[-1].strip()
                            else:
                                cot_reasoning = ""
                                predicted_answer = full_generated_text
                    elif not predicted_answer:
                        # COT already extracted but no answer, get from dict or use full text
                        if isinstance(generated_output, dict):
                            predicted_answer = generated_output.get('answer', full_generated_text)
                        else:
                            predicted_answer = full_generated_text
                    
                    # Clean up the extracted answer
                    predicted_answer = str(predicted_answer).strip()
                    
                    # Try to extract just the answer if CoT format is present (fallback)
                    if "Answer:" in predicted_answer:
                        predicted_answer = predicted_answer.split("Answer:")[-1].strip()
                    if "answer:" in predicted_answer.lower():
                        predicted_answer = predicted_answer.split("answer:")[-1].strip()
                    
                    # Take first line or first sentence if multiple lines (for matching)
                    predicted_answer_clean = predicted_answer.split('\n')[0].split('.')[0].strip()
                    
                    # If COT is still empty but we have full text, use full text as COT
                    if not cot_reasoning and full_generated_text:
                        # If predicted_answer is extracted, COT is the rest
                        if predicted_answer and predicted_answer in full_generated_text:
                            cot_reasoning = full_generated_text.replace(predicted_answer, '').strip()
                        else:
                            cot_reasoning = full_generated_text
                    
                    # Simple answer matching (can be improved)
                    answer_str = str(answer).lower().strip()
                    predicted_str = predicted_answer_clean.lower().strip()
                    
                    # Check if prediction matches answer (exact or substring match)
                    is_correct = (
                        answer_str == predicted_str or
                        answer_str in predicted_str or
                        predicted_str in answer_str or
                        predicted_str.startswith(answer_str) or
                        answer_str.startswith(predicted_str)
                    )
                    
                    if is_correct:
                        correct += 1
                        category_correct[category] += 1
                    
                    total += 1
                    category_total[category] += 1
                    
                    # Store all information for detailed results
                    all_predictions.append(predicted_answer_clean)  # Clean extracted answer
                    all_answers.append(str(answer))  # Original answer (not lowercased)
                    all_questions.append(question)
                    all_cot_reasoning.append(cot_reasoning)
                    all_full_generated.append(full_generated_text)
                    all_categories.append(category)
                    
                    # Mark as processed
                    processed_indices.add(global_sample_idx)
                    global_sample_idx += 1
                    
                    # Log first few examples for debugging
                    if total <= 5:
                        logger.info(f"Sample {total}:")
                        logger.info(f"  Question: {question[:50]}...")
                        logger.info(f"  Answer: {answer_str}")
                        logger.info(f"  Predicted: {predicted_str}")
                        logger.info(f"  Correct: {is_correct}")
                    
                    # Save checkpoint periodically
                    current_time = time.time()
                    should_save = (
                        (total % save_interval == 0) or  # Every N samples
                        (current_time - last_save_time >= save_interval_seconds)  # Every 5 minutes
                    )
                    if checkpoint_file and should_save:
                        save_checkpoint(
                            checkpoint_file,
                            global_sample_idx,
                            correct,
                            total,
                            category_correct,
                            category_total,
                            all_predictions,
                            all_answers,
                            list(processed_indices),
                            all_questions,
                            all_cot_reasoning,
                            all_full_generated,
                            all_categories
                        )
                        last_save_time = current_time
                    
                except Exception as e:
                    logger.warning(f"Error processing sample {global_sample_idx}: {e}")
                    if total <= 5:
                        traceback.print_exc()
                    # Still mark as processed to avoid infinite retry
                    processed_indices.add(global_sample_idx)
                    global_sample_idx += 1
                    continue
    
    # Calculate accuracies
    overall_accuracy = correct / total if total > 0 else 0.0
    category_accuracies = {
        cat: category_correct[cat] / category_total[cat] if category_total[cat] > 0 else 0.0
        for cat in category_correct.keys()
    }
    
    # Final checkpoint save
    if checkpoint_file:
        save_checkpoint(
            checkpoint_file,
            global_sample_idx,
            correct,
            total,
            category_correct,
            category_total,
            all_predictions,
            all_answers,
            list(processed_indices),
            all_questions,
            all_cot_reasoning,
            all_full_generated,
            all_categories
        )
    
    # Create detailed results with all information
    detailed_results = []
    for i in range(len(all_questions)):
        detailed_results.append({
            'question': all_questions[i],
            'category': all_categories[i],
            'cot_reasoning': all_cot_reasoning[i],
            'full_generated_text': all_full_generated[i],
            'predicted_answer': all_predictions[i],
            'ground_truth_answer': all_answers[i],
            'is_correct': (
                str(all_answers[i]).lower().strip() == all_predictions[i].lower().strip() or
                str(all_answers[i]).lower().strip() in all_predictions[i].lower().strip() or
                all_predictions[i].lower().strip() in str(all_answers[i]).lower().strip()
            )
        })
    
    return {
        'overall_accuracy': overall_accuracy,
        'correct': correct,
        'total': total,
        'category_accuracies': category_accuracies,
        'category_correct': category_correct,
        'category_total': category_total,
        'detailed_results': detailed_results,  # All samples with full details
        # Keep backward compatibility
        'predictions': all_predictions,
        'answers': all_answers,
        'questions': all_questions,
        'cot_reasoning': all_cot_reasoning,
        'full_generated_text': all_full_generated,
        'categories': all_categories
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint - FIXED VERSION")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--base_checkpoint", required=True, help="Path to base model checkpoint")
    parser.add_argument("--model_type", choices=["qwen3vl", "medgemma", "llava_med"], required=True)
    parser.add_argument("--dataset", choices=["kvasir", "endovis"], required=True)
    parser.add_argument("--test_data", required=True, help="Path to test data JSON")
    parser.add_argument("--image_base_path", required=True, help="Base path for images")
    parser.add_argument("--question_categories", default="question_categories.json")
    parser.add_argument("--output", required=True, help="Output directory for results")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("EVALUATION - FIXED VERSION (Using model.generate())")
    logger.info("="*80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Base model: {args.base_checkpoint}")
    logger.info(f"Model type: {args.model_type}")
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
            shuffle=False
        )
        logger.info(f"✓ Loaded {len(test_loader)} batches")
        
        # Set up checkpoint file for resume functionality
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine checkpoint epoch from filename or checkpoint
        checkpoint_epoch = checkpoint.get('epoch', 'unknown')
        if checkpoint_epoch == 'unknown':
            # Try to extract from filename
            if 'epoch_5' in args.checkpoint:
                checkpoint_epoch = 5
            elif 'epoch_3' in args.checkpoint:
                checkpoint_epoch = 3
        
        checkpoint_file = output_dir / f"eval_checkpoint_epoch{checkpoint_epoch}_{args.model_type}_{args.dataset}.json"
        logger.info(f"Checkpoint file: {checkpoint_file}")
        if checkpoint_file.exists():
            logger.info("✓ Found existing checkpoint - will resume from last saved progress")
        
        # Run evaluation
        logger.info("\n[4/4] Running evaluation...")
        results = evaluate(
            model=model,
            test_loader=test_loader,
            question_categories=question_categories,
            dataset_name=args.dataset,
            processor=processor,
            device=device,
            model_type=args.model_type,
            checkpoint_file=checkpoint_file,
            save_interval=100  # Save checkpoint every 100 samples
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
        
        # Print sample results
        logger.info("\n" + "="*80)
        logger.info("SAMPLE RESULTS (First 3)")
        logger.info("="*80)
        for i in range(min(3, len(results.get('detailed_results', [])))):
            sample = results['detailed_results'][i]
            logger.info(f"\nSample {i+1}:")
            logger.info(f"  Question: {sample['question']}")
            logger.info(f"  Category: {sample['category']}")
            logger.info(f"  COT Reasoning: {sample['cot_reasoning'][:100]}..." if len(sample['cot_reasoning']) > 100 else f"  COT Reasoning: {sample['cot_reasoning']}")
            logger.info(f"  Predicted Answer: {sample['predicted_answer']}")
            logger.info(f"  Ground Truth: {sample['ground_truth_answer']}")
            logger.info(f"  Correct: {sample['is_correct']}")
        
        # Save final results
        results_file = output_dir / f"evaluation_epoch{checkpoint_epoch}_{args.model_type}_{args.dataset}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Saved detailed results to {results_file}")
        logger.info(f"  Total samples: {len(results.get('detailed_results', []))}")
        logger.info(f"  Each sample includes: question, category, COT reasoning, full generated text, predicted answer, ground truth")
        
        # Clean up checkpoint file after successful completion
        if checkpoint_file and checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                logger.info(f"✓ Removed checkpoint file (evaluation complete)")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint file: {e}")
        
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

