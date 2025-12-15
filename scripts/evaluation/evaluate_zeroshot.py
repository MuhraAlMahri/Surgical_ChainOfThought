#!/usr/bin/env python3
"""
Zero-Shot Evaluation Script
Evaluates the base model WITHOUT any fine-tuning (no LoRA adapters)
This provides a baseline to compare against fine-tuned models.
"""

import json
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm
import os
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import argparse
from huggingface_hub import login


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


def load_base_model(base_model_name: str = "Qwen/Qwen3-VL-8B-Instruct", hf_token: str = None):
    """Load base model WITHOUT any LoRA adapters (zero-shot)."""
    print(f"Loading base model (ZERO-SHOT): {base_model_name}")
    print("⚠️  No fine-tuning - using base model as-is")
    
    try:
        # Handle Hugging Face authentication for gated models (e.g., MedGemma-4B)
        if hf_token:
            print("Authenticating with Hugging Face...")
            login(token=hf_token)
        
        # Get token from environment if not provided
        token = hf_token or os.getenv("HF_TOKEN")
        
        # Determine model class based on model name
        # MedGemma and LLaVA use AutoModelForImageTextToText, Qwen uses AutoModelForVision2Seq
        is_medgemma = 'medgemma' in base_model_name.lower() or 'pali' in base_model_name.lower()
        is_llava = 'llava' in base_model_name.lower()
        
        if is_medgemma or is_llava:
            model_type = "MedGemma/PaliGemma" if is_medgemma else "LLaVA"
            print(f"Detected {model_type} model - using AutoModelForImageTextToText")
            
            # For LLaVA models, try multiple loading strategies including LlavaForConditionalGeneration
            if is_llava:
                print("Attempting to load LLaVA model...")
                base_model = None
                
                # Strategy 1: Try LlavaForConditionalGeneration (if available)
                try:
                    from transformers import LlavaForConditionalGeneration
                    print("  Trying LlavaForConditionalGeneration (bf16)...")
                    base_model = LlavaForConditionalGeneration.from_pretrained(
                        base_model_name,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True,
                        token=token
                    )
                    print("  ✓ Loaded using LlavaForConditionalGeneration (bf16)")
                except (ImportError, RuntimeError, ValueError) as e:
                    print(f"  ⚠️  LlavaForConditionalGeneration failed: {str(e)[:200]}")
                    base_model = None
                
                # Strategy 2: Try AutoModelForImageTextToText with different dtypes
                if base_model is None:
                    loading_strategies = [
                        ("AutoModelForImageTextToText with bf16", torch.bfloat16),
                        ("AutoModelForImageTextToText with fp16", torch.float16),
                        ("AutoModelForImageTextToText with default dtype", None),
                    ]
                    
                    for strategy_name, dtype in loading_strategies:
                        try:
                            print(f"  Trying {strategy_name}...")
                            base_model = AutoModelForImageTextToText.from_pretrained(
                                base_model_name,
                                torch_dtype=dtype,
                                device_map="auto",
                                trust_remote_code=True,
                                token=token
                            )
                            print(f"  ✓ Loaded using {strategy_name}")
                            break
                        except Exception as e:
                            print(f"  ⚠️  {strategy_name} failed: {str(e)[:200]}")
                            base_model = None
                
                if base_model is None:
                    raise RuntimeError(f"Failed to load LLaVA model {base_model_name} with any strategy. The transformers library may not support llava_mistral architecture yet.")
            else:
                # MedGemma/PaliGemma
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
        
        base_model.eval()
        print("✓ Model loaded successfully")
        
        # Load processor - try LlavaProcessor first for LLaVA models
        print("Loading processor...")
        is_llava = 'llava' in base_model_name.lower()
        
        if is_llava:
            # LLaVA models - try multiple approaches (same as training script)
            processor = None
            try:
                # First try LlavaProcessor
                from transformers import LlavaProcessor
                processor = LlavaProcessor.from_pretrained(
                    base_model_name, trust_remote_code=True, token=token
                )
                print("✓ Loaded using LlavaProcessor")
            except (ImportError, OSError, ValueError) as e:
                print(f"⚠️  LlavaProcessor failed: {str(e)[:200]}")
                try:
                    # Try AutoProcessor with trust_remote_code
                    processor = AutoProcessor.from_pretrained(
                        base_model_name, trust_remote_code=True, token=token
                    )
                    print("✓ Loaded using AutoProcessor")
                except Exception as e2:
                    print(f"⚠️  AutoProcessor failed: {str(e2)[:200]}")
                    # Try loading components separately
                    try:
                        from transformers import AutoTokenizer, CLIPImageProcessor
                        tokenizer = AutoTokenizer.from_pretrained(
                            base_model_name, trust_remote_code=True, token=token
                        )
                        # For LLaVA, try to get image processor from vision model
                        try:
                            image_processor = CLIPImageProcessor.from_pretrained(
                                "openai/clip-vit-large-patch14-336",  # Common CLIP model for LLaVA
                                trust_remote_code=True
                            )
                        except:
                            # Fallback: try to load from model repo
                            image_processor = CLIPImageProcessor.from_pretrained(
                                base_model_name, trust_remote_code=True, token=token
                            )
                        # Create processor-like object that mimics LlavaProcessor interface
                        class LlavaProcessorWrapper:
                            def __init__(self, image_processor, tokenizer):
                                self.image_processor = image_processor
                                self.tokenizer = tokenizer
                                # Note: We don't need to find image_token_id anymore
                                # The model will handle <image> placeholders in its forward pass
                                # We just need to tokenize the text as-is with <image> placeholders
                                pass
                            
                            def __call__(self, text=None, images=None, return_tensors=None, padding=None, max_length=None, truncation=None, **kwargs):
                                # For LLaVA, the model expects <image> placeholders in the text
                                # The model's forward pass will handle replacing them with image embeddings
                                # We should NOT manually replace tokens - let the model handle it
                                
                                # Handle text tokenization
                                if text is not None:
                                    # Ensure text is a string (not a list)
                                    if isinstance(text, list):
                                        text = text[0] if text else ""
                                    
                                    # Simply tokenize the text as-is, including <image> placeholders
                                    # The model will handle the image token embedding in its forward pass
                                    text_inputs = self.tokenizer(
                                        text,
                                        return_tensors=return_tensors,
                                        padding=padding if padding else False,
                                        max_length=max_length,
                                        truncation=truncation,
                                        **kwargs
                                    )
                                else:
                                    text_inputs = {}
                                
                                # Handle image processing
                                if images is not None:
                                    # Convert single image to list if needed
                                    if not isinstance(images, list):
                                        images = [images]
                                    image_inputs = self.image_processor(images, return_tensors=return_tensors, **kwargs)
                                else:
                                    image_inputs = {}
                                
                                # Combine inputs - LLaVA model expects both text and image inputs
                                # The model will match image features to <image> tokens in forward pass
                                combined = {**text_inputs, **image_inputs}
                                return combined
                            
                            def apply_chat_template(self, *args, **kwargs):
                                return self.tokenizer.apply_chat_template(*args, **kwargs)
                        
                        processor = LlavaProcessorWrapper(image_processor, tokenizer)
                        print("✓ Loaded components separately and wrapped")
                    except Exception as e3:
                        raise RuntimeError(f"Failed to load processor: {e3}")
        else:
            processor = AutoProcessor.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                token=token
            )
            print("✓ Processor loaded successfully")
    
        device = next(base_model.parameters()).device
        print(f"✓ Base model loaded on device: {device}")
        print(f"✓ Model type: Zero-shot (no fine-tuning)")
        
        return base_model, processor, device
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load model {base_model_name}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        raise


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


def evaluate_model(model, processor, device, test_data, image_base_path, max_samples=None, use_instruction=True):
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
        'by_stage': defaultdict(lambda: {
            'total': 0, 'correct': 0, 'accuracy': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0
        }),
        'by_question_type': defaultdict(lambda: {
            'total': 0, 'correct': 0, 'accuracy': 0.0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0
        }),
        'predictions': [],
        'errors': []
    }
    
    # Accumulators for precision/recall/F1
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    stage_precision = defaultdict(float)
    stage_recall = defaultdict(float)
    stage_f1 = defaultdict(float)
    qtype_precision = defaultdict(float)
    qtype_recall = defaultdict(float)
    qtype_f1 = defaultdict(float)
    
    for item in tqdm(test_data, desc="Evaluating (Zero-Shot)"):
        # Use instruction field if available, otherwise use question
        if use_instruction:
            question = item.get('instruction', item.get('question', ''))
        else:
            question = item.get('question', '')
        
        ground_truth = item.get('answer', '').strip()
        image_filename = item.get('image_filename', '').strip()  # Strip whitespace
        image_id = item.get('image_id', '')
        stage = item.get('stage', 0)
        question_type = item.get('question_type', 'unknown')
        
        # Get image path - ALWAYS prefer image_filename if available
        image_path = None
        
        if image_filename:
            # Use image_filename from dataset (most reliable)
            image_path = os.path.join(image_base_path, image_filename)
            if not os.path.exists(image_path):
                # Try alternative extension if original doesn't exist
                base_name = os.path.splitext(image_filename)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    alt_path = os.path.join(image_base_path, f"{base_name}{ext}")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
                else:
                    results['errors'].append(f"Image not found: {image_path} (from image_filename: {image_filename})")
                    continue
        else:
            # Fallback: try to construct from image_id with common extensions
            possible_extensions = ['.png', '.jpg', '.jpeg']
            for ext in possible_extensions:
                candidate = os.path.join(image_base_path, f"{image_id}{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            if not image_path:
                results['errors'].append(f"Image not found: {image_id} (tried {possible_extensions}, no image_filename provided)")
                continue
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            results['errors'].append(f"Image load error: {str(e)}")
            continue
        
        # Generate prediction
        try:
            # Check if this is a LLaVA model
            is_llava = 'llava' in str(processor.__class__).lower() or 'llava' in str(type(processor)).lower()
            
            if is_llava:
                # LLaVA models need <image> placeholder in text
                text_prompt = f"USER: <image>\n{question}\nASSISTANT:"
                inputs = processor(text=text_prompt, images=image, return_tensors="pt")
            else:
                # Qwen/MedGemma format
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
            
            # Get tokenizer (handle both regular processor and wrapper)
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=128,  # More tokens for zero-shot (model may be verbose)
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
                )
            
            generated_ids = outputs[0][input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            prediction = extract_answer_from_response(generated_text)
            
        except Exception as e:
            results['errors'].append(f"Generation error: {str(e)}")
            prediction = ""
        
        # Evaluate
        results['total'] += 1
        
        # Determine if correct
        is_correct = False
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        
        # Check if multi-label
        if ';' in ground_truth or question_type == 'multi_label':
            pred_set = parse_labels(prediction)
            gt_set = parse_labels(ground_truth)
            is_correct = (pred_set == gt_set) or (pred_set & gt_set == gt_set)
            precision, recall, f1 = calculate_precision_recall_f1(pred_set, gt_set)
        else:
            is_correct = smart_match(prediction, ground_truth)
            if is_correct:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
        
        if is_correct:
            results['correct'] += 1
        
        # Update accumulators
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        
        # Stage tracking
        stage_name = f"Stage {stage}" if stage else "Unknown"
        results['by_stage'][stage_name]['total'] += 1
        if is_correct:
            results['by_stage'][stage_name]['correct'] += 1
        stage_precision[stage_name] += precision
        stage_recall[stage_name] += recall
        stage_f1[stage_name] += f1
        
        # Question type tracking
        results['by_question_type'][question_type]['total'] += 1
        if is_correct:
            results['by_question_type'][question_type]['correct'] += 1
        qtype_precision[question_type] += precision
        qtype_recall[question_type] += recall
        qtype_f1[question_type] += f1
        
        # Store prediction
        results['predictions'].append({
            'image_id': image_id,
            'question': item.get('question', ''),
            'instruction': item.get('instruction', ''),
            'question_type': question_type,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'correct': is_correct,
            'stage': stage
        })
    
    # Calculate final metrics
    if results['total'] > 0:
        results['accuracy'] = (results['correct'] / results['total']) * 100
        results['precision'] = (total_precision / results['total']) * 100
        results['recall'] = (total_recall / results['total']) * 100
        results['f1'] = (total_f1 / results['total']) * 100
        
        # Calculate stage metrics
        for stage_name in results['by_stage']:
            stage_data = results['by_stage'][stage_name]
            if stage_data['total'] > 0:
                stage_data['accuracy'] = (stage_data['correct'] / stage_data['total']) * 100
                stage_data['precision'] = (stage_precision[stage_name] / stage_data['total']) * 100
                stage_data['recall'] = (stage_recall[stage_name] / stage_data['total']) * 100
                stage_data['f1'] = (stage_f1[stage_name] / stage_data['total']) * 100
        
        # Calculate question type metrics
        for qtype in results['by_question_type']:
            qtype_data = results['by_question_type'][qtype]
            if qtype_data['total'] > 0:
                qtype_data['accuracy'] = (qtype_data['correct'] / qtype_data['total']) * 100
                qtype_data['precision'] = (qtype_precision[qtype] / qtype_data['total']) * 100
                qtype_data['recall'] = (qtype_recall[qtype] / qtype_data['total']) * 100
                qtype_data['f1'] = (qtype_f1[qtype] / qtype_data['total']) * 100
    
    # Convert defaultdict to dict for JSON serialization
    results['by_stage'] = dict(results['by_stage'])
    results['by_question_type'] = dict(results['by_question_type'])
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Evaluation: Base Model Without Fine-Tuning")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Base model name")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate (for testing)")
    parser.add_argument("--use_instruction", action="store_true", default=True, help="Use instruction field if available")
    parser.add_argument("--no_instruction", dest="use_instruction", action="store_false", help="Use question field only")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ZERO-SHOT EVALUATION: Base Model (No Fine-Tuning)")
    print("=" * 80)
    print(f"Base Model: {args.base_model}")
    print(f"Test Data: {args.test_data}")
    print(f"Image Directory: {args.image_dir}")
    print(f"Output: {args.output}")
    if args.use_instruction:
        print("Mode: Using instruction field if available")
    else:
        print("Mode: Using question field only")
    print("=" * 80)
    
    try:
        # Load base model (no LoRA adapters)
        # Get HF token from environment for gated models (e.g., MedGemma-4B)
        hf_token = os.getenv("HF_TOKEN")
        print("\n" + "=" * 80)
        print("STEP 1: Loading Base Model")
        print("=" * 80)
        model, processor, device = load_base_model(args.base_model, hf_token=hf_token)
        
        # Load test data
        print("\n" + "=" * 80)
        print("STEP 2: Loading Test Data")
        print("=" * 80)
        print(f"Loading test data from: {args.test_data}")
        with open(args.test_data, 'r') as f:
            test_data = json.load(f)
        print(f"✓ Loaded {len(test_data)} test samples")
        
        # Evaluate
        print("\n" + "=" * 80)
        print("STEP 3: Running Evaluation")
        print("=" * 80)
        print("Starting evaluation...")
        results = evaluate_model(model, processor, device, test_data, args.image_dir, args.max_samples, args.use_instruction)
        print("✓ Evaluation completed")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: Evaluation failed")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("Evaluation terminated due to error")
        print("=" * 80)
        raise
    
    # Print summary
    print("\n" + "=" * 80)
    print("ZERO-SHOT EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nTotal samples: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Precision: {results['precision']:.2f}%")
    print(f"Recall: {results['recall']:.2f}%")
    print(f"F1 Score: {results['f1']:.2f}%")
    
    print(f"\nBy Stage:")
    for stage in sorted(results['by_stage'].keys()):
        stage_data = results['by_stage'][stage]
        print(f"  {stage}:")
        print(f"    Accuracy: {stage_data['accuracy']:.2f}% ({stage_data['correct']}/{stage_data['total']})")
        print(f"    Precision: {stage_data['precision']:.2f}%")
        print(f"    Recall: {stage_data['recall']:.2f}%")
        print(f"    F1: {stage_data['f1']:.2f}%")
    
    print(f"\nBy Question Type:")
    for qtype in sorted(results['by_question_type'].keys()):
        qtype_data = results['by_question_type'][qtype]
        print(f"  {qtype}:")
        print(f"    Accuracy: {qtype_data['accuracy']:.2f}% ({qtype_data['correct']}/{qtype_data['total']})")
        print(f"    Precision: {qtype_data['precision']:.2f}%")
        print(f"    Recall: {qtype_data['recall']:.2f}%")
        print(f"    F1: {qtype_data['f1']:.2f}%")
    
    if results['errors']:
        print(f"\nErrors: {len(results['errors'])}")
        for error in results['errors'][:10]:
            print(f"  - {error}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")
    
    # Save results
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    print("\n" + "=" * 80)
    print("Zero-shot evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

