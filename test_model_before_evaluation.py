#!/usr/bin/env python3
"""
TEST SCRIPT: Validate Model Before Full Evaluation

This script validates a trained model on 100 samples to catch issues early:
- Model loads correctly
- LoRA adapter merged properly
- Generation produces valid text (not gibberish)
- Answer extraction works
- Accuracy is reasonable

Run this BEFORE running full evaluation to catch bugs early!
"""

import argparse
import json
import os
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
import re


# ============================================================================
# CHECKLIST VALIDATION FUNCTIONS
# ============================================================================

def check_model_loading(model, processor, device) -> Tuple[bool, List[str]]:
    """Check 1: Model loading checklist."""
    issues = []
    
    # ✓ Load base model correctly
    if model is None:
        issues.append("❌ Model is None")
        return False, issues
    
    # ✓ Set model to eval mode
    if model.training:
        issues.append("⚠️  Model is in training mode (should be eval)")
        model.eval()
    
    # ✓ Check vision tower loaded
    if hasattr(model, 'vision_tower') or hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
        pass  # Vision tower exists
    else:
        # For some models, vision is integrated differently
        if not hasattr(model, 'get_image_features') and not hasattr(model, 'encode_images'):
            issues.append("⚠️  Cannot verify vision tower loaded")
    
    # ✓ Check processor
    if processor is None:
        issues.append("❌ Processor is None")
        return False, issues
    
    # ✓ Check device
    model_device = next(model.parameters()).device
    if model_device.type != device.type:
        issues.append(f"⚠️  Model on {model_device}, expected {device}")
    
    return len(issues) == 0, issues


def check_generation_params(model, processor, device) -> Tuple[bool, List[str]]:
    """Check 2: Generation parameters."""
    issues = []
    
    # Test generation on dummy input
    try:
        # Create dummy image and text
        dummy_image = Image.new('RGB', (224, 224), color='red')
        dummy_text = "What is this?"
        
        # Process
        if hasattr(processor, 'apply_chat_template'):
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_image},
                    {"type": "text", "text": dummy_text}
                ]
            }]
            text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_prompt], images=[dummy_image], return_tensors="pt")
        else:
            inputs = processor(text=[dummy_text], images=[dummy_image], return_tensors="pt")
        
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Check required parameters
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        
        # ✓ Set pad_token_id and eos_token_id
        pad_token_id = getattr(tokenizer, 'pad_token_id', None)
        eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        
        if pad_token_id is None:
            issues.append("⚠️  pad_token_id is None (may cause warnings)")
        
        if eos_token_id is None:
            issues.append("⚠️  eos_token_id is None (may cause issues)")
        
        # Test generation
        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                "max_new_tokens": 50,  # Small for testing
                "do_sample": True,  # CRITICAL for LLaVA
                "temperature": 0.2,
            }
            if eos_token_id is not None:
                generate_kwargs["eos_token_id"] = eos_token_id
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            
            outputs = model.generate(**generate_kwargs)
        
        # ✓ Decode only NEW tokens (not input)
        input_len = inputs['input_ids'].shape[1]
        if outputs.shape[1] <= input_len:
            issues.append("❌ Generated output length <= input length (no new tokens generated)")
            return False, issues
        
        generated_ids = outputs[0][input_len:]
        
        # Decode
        if hasattr(processor, 'tokenizer'):
            generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            generated_text = processor.decode(generated_ids, skip_special_tokens=True)
        
        if not generated_text or len(generated_text.strip()) == 0:
            issues.append("❌ Generated text is empty")
            return False, issues
        
        if len(generated_text.strip()) < 2:
            issues.append("⚠️  Generated text is very short (< 2 chars)")
        
    except Exception as e:
        issues.append(f"❌ Generation test failed: {str(e)}")
        return False, issues
    
    return True, issues


def check_answer_extraction(prediction: str, ground_truth: str) -> Tuple[bool, str, List[str]]:
    """Check 3: Answer extraction."""
    issues = []
    
    # ✓ Extract short answer from verbose output
    extracted = extract_answer_from_response(prediction)
    
    if not extracted:
        issues.append("⚠️  Extracted answer is empty")
        return False, extracted, issues
    
    # ✓ Handle empty predictions properly
    if not prediction:
        issues.append("❌ Original prediction is empty")
        return False, extracted, issues
    
    # ✓ Filter invalid tokens (check for artifacts)
    artifacts = ['/**', '***', '<|im_start|>', '<|im_end|>', '<image>', '<pad>']
    for artifact in artifacts:
        if artifact in extracted and artifact not in ground_truth:
            issues.append(f"⚠️  Artifact '{artifact}' in extracted answer")
    
    # ✓ Clean up artifacts
    if '/**' in extracted:
        issues.append("⚠️  Artifact pattern '/**' found in answer")
    
    return True, extracted, issues


def extract_answer_from_response(generated_text: str) -> str:
    """Extract the answer from generated text."""
    if not generated_text:
        return ""
    
    # Remove common prefixes
    prefixes = ["assistant:", "assistant", "### Response:", "Response:", "Answer:", "Answer", "ASSISTANT:", "ASSISTANT"]
    for prefix in prefixes:
        if generated_text.strip().lower().startswith(prefix.lower()):
            generated_text = generated_text.strip()[len(prefix):].strip()
            break
    
    # Remove trailing special tokens
    for token in ["</s>", "<|endoftext|>", "<|im_end|>", "<|endoftext|>"]:
        if generated_text.endswith(token):
            generated_text = generated_text[:-len(token)].strip()
    
    # Take first line if multiple lines
    lines = generated_text.split('\n')
    if lines:
        generated_text = lines[0].strip()
    
    # Remove artifacts
    if '/**' in generated_text:
        parts = generated_text.split('/**')
        cleaned_parts = [p.strip() for p in parts if p.strip() and not p.strip().startswith('*')]
        generated_text = ' '.join(cleaned_parts) if cleaned_parts else ""
    
    return generated_text.strip()


def smart_match(prediction: str, ground_truth: str, threshold: float = 0.7) -> bool:
    """Smart matching with multiple strategies."""
    from difflib import SequenceMatcher
    
    pred_n = prediction.lower().strip()
    gt_n = ground_truth.lower().strip()
    
    if not pred_n:
        return not gt_n  # Both empty = match
    
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


# ============================================================================
# MODEL LOADING (WITH ALL FIXES)
# ============================================================================

def load_model_with_checks(base_model_name: str, adapter_path: str, hf_token: str = None):
    """Load model with all checklist items."""
    print(f"\n{'='*80}")
    print("MODEL LOADING CHECKLIST")
    print(f"{'='*80}")
    
    token = hf_token or os.getenv("HF_TOKEN")
    
    # Determine model type
    is_llava = 'llava' in base_model_name.lower()
    
    print(f"Base Model: {base_model_name}")
    print(f"Adapter Path: {adapter_path}")
    print(f"Model Type: {'LLaVA' if is_llava else 'Other'}")
    
    # Load base model
    if is_llava:
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
    else:
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
    print("✓ Base model loaded")
    
    # Load LoRA adapter if exists
    if os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"✓ LoRA adapter found at: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✓ LoRA adapter loaded")
        
        # CRITICAL: Merge adapter
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        print("✓ LoRA weights merged (merge_and_unload())")
    else:
        print("⚠️  No LoRA adapter found, using base model")
        model = base_model
    
    # Set to eval mode
    model.eval()
    print("✓ Model set to eval mode")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        token=token
    )
    print("✓ Processor loaded")
    
    device = next(model.parameters()).device
    print(f"✓ Model on device: {device}")
    
    return model, processor, device


# ============================================================================
# VALIDATION ON 100 SAMPLES
# ============================================================================

def validate_model_on_samples(
    model, processor, device, test_data: List[Dict], image_base_path: str, num_samples: int = 100
) -> Dict:
    """Validate model on N samples."""
    
    print(f"\n{'='*80}")
    print(f"VALIDATING MODEL ON {num_samples} SAMPLES")
    print(f"{'='*80}")
    
    # Limit samples
    test_subset = test_data[:num_samples]
    
    results = {
        'total': 0,
        'correct': 0,
        'gibberish_count': 0,
        'empty_count': 0,
        'errors': [],
        'sample_predictions': [],
        'issues': []
    }
    
    gibberish_patterns = [
        r'/\*\*+',  # /** patterns
        r'\*{3,}',  # Multiple asterisks
        r'<\|im_\w+\|>',  # Special tokens
        r'<image>',  # Image tokens
        r'^\s*$',  # Only whitespace
    ]
    
    for i, item in enumerate(tqdm(test_subset, desc="Validating")):
        question = item.get('instruction', item.get('question', ''))
        ground_truth = item.get('answer', '').strip()
        image_path = item.get('image', '') or item.get('image_filename', '')
        
        if not image_path:
            results['errors'].append(f"Sample {i}: Missing image path")
            continue
        
        full_image_path = os.path.join(image_base_path, image_path)
        if not os.path.exists(full_image_path):
            results['errors'].append(f"Sample {i}: Image not found: {full_image_path}")
            continue
        
        try:
            image = Image.open(full_image_path).convert('RGB')
            
            # Format prompt
            if 'llava' in str(type(model)).lower():
                prompt = f"USER: <image>\n{question}\nASSISTANT:"
            else:
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
                    ]
                }]
                prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            
            # Process inputs
            inputs = processor(text=[prompt], images=[image], return_tensors="pt")
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            
            input_len = inputs['input_ids'].shape[1]
            
            # Generate
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            pad_token_id = getattr(tokenizer, 'pad_token_id', None)
            
            with torch.no_grad():
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": 256,  # Sufficient for answers
                    "do_sample": True,  # CRITICAL for LLaVA
                    "temperature": 0.2,
                }
                if eos_token_id is not None:
                    generate_kwargs["eos_token_id"] = eos_token_id
                if pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = pad_token_id
                
                generated_ids = model.generate(**generate_kwargs)
            
            # Decode only new tokens
            generated_token_ids = generated_ids[0][input_len:]
            
            # Filter invalid tokens
            image_token_id = getattr(model.config, 'image_token_id', None) or getattr(processor, 'image_token_id', None)
            filtered_token_ids = []
            for token_id in generated_token_ids:
                token_id_int = int(token_id.item() if hasattr(token_id, 'item') else token_id)
                
                if image_token_id and token_id_int == image_token_id:
                    continue
                if pad_token_id and token_id_int == pad_token_id:
                    continue
                if eos_token_id and token_id_int == eos_token_id:
                    break
                
                vocab_size = getattr(tokenizer, 'vocab_size', None)
                if vocab_size and token_id_int >= vocab_size:
                    continue
                if token_id_int <= 0:
                    continue
                
                try:
                    test_text = tokenizer.decode([token_id_int], skip_special_tokens=True)
                    if test_text and test_text.strip() and not test_text.startswith('/**'):
                        filtered_token_ids.append(token_id_int)
                except:
                    continue
            
            # Decode
            if filtered_token_ids:
                generated_text = tokenizer.decode(filtered_token_ids, skip_special_tokens=True)
            else:
                generated_text = ""
            
            # Clean artifacts
            if '/**' in generated_text:
                parts = generated_text.split('/**')
                cleaned_parts = [p.strip() for p in parts if p.strip() and not p.strip().startswith('*')]
                generated_text = ' '.join(cleaned_parts) if cleaned_parts else ""
            
            # Extract answer
            prediction = extract_answer_from_response(generated_text)
            
            # Check for gibberish
            is_gibberish = False
            for pattern in gibberish_patterns:
                if re.search(pattern, prediction):
                    is_gibberish = True
                    break
            
            if is_gibberish:
                results['gibberish_count'] += 1
                results['issues'].append(f"Sample {i}: Gibberish detected: {prediction[:50]}")
            
            if not prediction:
                results['empty_count'] += 1
                results['issues'].append(f"Sample {i}: Empty prediction")
            
            # Evaluate
            is_correct = smart_match(prediction, ground_truth)
            
            results['total'] += 1
            if is_correct:
                results['correct'] += 1
            
            # Store sample
            if i < 10:  # Store first 10
                results['sample_predictions'].append({
                    'question': question[:100],
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'full_generated': generated_text[:200],
                    'correct': is_correct,
                    'is_gibberish': is_gibberish
                })
        
        except Exception as e:
            results['errors'].append(f"Sample {i}: {str(e)}")
    
    # Calculate accuracy
    if results['total'] > 0:
        results['accuracy'] = (results['correct'] / results['total']) * 100
        results['gibberish_rate'] = (results['gibberish_count'] / results['total']) * 100
        results['empty_rate'] = (results['empty_count'] / results['total']) * 100
    else:
        results['accuracy'] = 0.0
        results['gibberish_rate'] = 0.0
        results['empty_rate'] = 0.0
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test model before full evaluation")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Image directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--output", type=str, default="test_results.json", help="Output JSON")
    
    args = parser.parse_args()
    
    print("="*80)
    print("MODEL VALIDATION TEST")
    print("="*80)
    
    # Load model
    hf_token = os.getenv("HF_TOKEN")
    model, processor, device = load_model_with_checks(args.base_model, args.adapter_path, hf_token)
    
    # Run checklist checks
    print(f"\n{'='*80}")
    print("RUNNING CHECKLIST CHECKS")
    print(f"{'='*80}")
    
    # Check 1: Model loading
    ok, issues = check_model_loading(model, processor, device)
    print(f"\n[1] Model Loading: {'✓ PASS' if ok else '❌ FAIL'}")
    for issue in issues:
        print(f"    {issue}")
    
    # Check 2: Generation
    ok, issues = check_generation_params(model, processor, device)
    print(f"\n[2] Generation Parameters: {'✓ PASS' if ok else '❌ FAIL'}")
    for issue in issues:
        print(f"    {issue}")
    
    # Load test data
    print(f"\n{'='*80}")
    print("LOADING TEST DATA")
    print(f"{'='*80}")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # Validate on samples
    results = validate_model_on_samples(model, processor, device, test_data, args.image_dir, args.num_samples)
    
    # Print results
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS")
    print(f"{'='*80}")
    print(f"Total samples tested: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Gibberish rate: {results['gibberish_rate']:.2f}%")
    print(f"Empty predictions: {results['empty_rate']:.2f}%")
    print(f"Errors: {len(results['errors'])}")
    
    if results['gibberish_count'] > 0:
        print(f"\n⚠️  WARNING: {results['gibberish_count']} gibberish predictions detected!")
        print("   This indicates a problem with model loading or generation.")
        print("   DO NOT proceed with full evaluation until this is fixed.")
    
    if results['empty_count'] > results['total'] * 0.1:
        print(f"\n⚠️  WARNING: {results['empty_count']} empty predictions ({results['empty_rate']:.1f}%)")
        print("   This may indicate generation issues.")
    
    if results['accuracy'] < 10:
        print(f"\n⚠️  WARNING: Very low accuracy ({results['accuracy']:.1f}%)")
        print("   This may indicate model loading or answer extraction issues.")
    
    # Show sample predictions
    if results['sample_predictions']:
        print(f"\n{'='*80}")
        print("SAMPLE PREDICTIONS (first 5)")
        print(f"{'='*80}")
        for i, sample in enumerate(results['sample_predictions'][:5]):
            print(f"\nSample {i+1}:")
            print(f"  Question: {sample['question']}")
            print(f"  Ground Truth: {sample['ground_truth']}")
            print(f"  Prediction: {sample['prediction']}")
            print(f"  Full Generated: {sample['full_generated']}")
            print(f"  Correct: {'✓' if sample['correct'] else '✗'}")
            if sample['is_gibberish']:
                print(f"  ⚠️  GIBBERISH DETECTED")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {args.output}")
    
    # Final verdict
    print(f"\n{'='*80}")
    if results['gibberish_count'] == 0 and results['accuracy'] > 10:
        print("✓ VALIDATION PASSED - Model appears to be working correctly")
        print("  You can proceed with full evaluation.")
    else:
        print("❌ VALIDATION FAILED - Issues detected")
        print("  DO NOT proceed with full evaluation until issues are fixed.")
        print("  Check the issues list above and fix them first.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()



