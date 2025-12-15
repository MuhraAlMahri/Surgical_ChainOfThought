#!/usr/bin/env python3
"""
Fine-Tuned Model Evaluation Script
Evaluates fine-tuned LLaVA-Med models (with LoRA adapters) on test sets.
"""

import json
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoModelForImageTextToText, AutoProcessor, AutoConfig, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import os
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import argparse
from huggingface_hub import login
from transformers import CLIPImageProcessor


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip().replace(".", "").replace(",", "").replace(";", "")


def parse_labels(text: str) -> set:
    """Parse labels from text (handles semicolon-separated multi-label format)."""
    if not text:
        return set()
    labels = [normalize_text(label) for label in text.split(';')]
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
    
    # CRITICAL FIX: Empty predictions are ALWAYS wrong (unless ground truth is also empty)
    if not pred_n:
        # Only return True if ground truth is also empty
        return not gt_n
    
    if not gt_n:
        return False
    
    if pred_n == gt_n:
        return True
    
    # CRITICAL FIX: Only check substring match if both are non-empty
    # pred_n in gt_n would return True for empty pred_n, which is wrong
    if gt_n in pred_n:
        return True
    
    similarity = SequenceMatcher(None, pred_n, gt_n).ratio()
    return similarity >= threshold


def load_finetuned_model(base_model_name: str, adapter_path: str, hf_token: str = None):
    """Load fine-tuned model with LoRA adapter."""
    print(f"Loading fine-tuned model:")
    print(f"  Base Model: {base_model_name}")
    print(f"  Adapter: {adapter_path}")
    
    try:
        if hf_token:
            print("Authenticating with Hugging Face...")
            login(token=hf_token)
        
        token = hf_token or os.getenv("HF_TOKEN")
        
        # Determine model class
        is_llava = 'llava' in base_model_name.lower()
        
        if is_llava:
            print("Detected LLaVA model - using AutoModelForImageTextToText")
            
            # Load config first - handle llava_mistral not being recognized (same approach as training script)
            config = None
            try:
                config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True, token=token)
            except ValueError as e:
                if "llava_mistral" in str(e):
                    print("⚠️  llava_mistral not recognized, loading config manually...")
                    from huggingface_hub import hf_hub_download
                    from transformers import LlavaConfig
                    import json
                    
                    config_path = hf_hub_download(
                        repo_id=base_model_name,
                        filename="config.json",
                        token=token
                    )
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    
                    if config_dict.get('model_type') == 'llava_mistral':
                        config_dict['model_type'] = 'llava'
                        print("  ⚠️  Changed model_type from 'llava_mistral' to 'llava' for compatibility")
                    
                    config = LlavaConfig.from_dict(config_dict)
                    if hasattr(config, 'auto_map'):
                        config.trust_remote_code = True
                else:
                    print(f"⚠️  Config loading failed: {e}, will try loading model without explicit config...")
                    config = None
            except (KeyError, OSError) as e:
                print(f"⚠️  Config loading failed: {e}, will try loading model without explicit config...")
                config = None
            
            # Load base model - use same approach as training script
            if config is not None:
                base_model = AutoModelForVision2Seq.from_pretrained(
                    base_model_name,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    token=token
                )
            else:
                # Fallback: try without config
                base_model = AutoModelForVision2Seq.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    token=token
                )
                config = base_model.config
            print("✓ Base model loaded")
            
            # CRITICAL: Check adapter config for vocab_size used during training
            import json
            adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
            adapter_vocab_size = None
            if os.path.exists(adapter_config_path):
                try:
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                        adapter_vocab_size = adapter_config.get('base_model_name_or_path', {}).get('vocab_size')
                        if adapter_vocab_size is None:
                            # Try reading from the adapter's base model config
                            adapter_vocab_size = adapter_config.get('vocab_size')
                except Exception as e:
                    print(f"Could not read adapter config: {e}")
            
            # CRITICAL: Resize embedding layer if image_token_id >= vocab_size (same as training script)
            image_token_id = getattr(config, 'image_token_id', 32000)
            vocab_size = config.vocab_size
            
            # If adapter was trained with resized embeddings, use that vocab_size
            if adapter_vocab_size and adapter_vocab_size > vocab_size:
                print(f"Adapter was trained with vocab_size={adapter_vocab_size}, resizing base model to match...")
                target_vocab_size = adapter_vocab_size
            elif image_token_id >= vocab_size:
                target_vocab_size = image_token_id + 1
            else:
                target_vocab_size = None
            
            if target_vocab_size and target_vocab_size > vocab_size:
                print(f"⚠️  Resizing embedding layer from {vocab_size} to {target_vocab_size}...")
                
                # Try standard resize_token_embeddings method first
                if hasattr(base_model, 'resize_token_embeddings'):
                    try:
                        base_model.resize_token_embeddings(target_vocab_size)
                        print("✓ Resized embeddings using resize_token_embeddings")
                    except Exception as e:
                        print(f"resize_token_embeddings failed: {e}, trying manual resize...")
                        # Fall back to manual resize
                        import torch.nn as nn
                        # For LLaVA models, the structure is model.model.embed_tokens
                        base_model_unwrapped = base_model
                        if hasattr(base_model_unwrapped, 'model'):
                            base_model_unwrapped = base_model_unwrapped.model
                        
                        # Resize token embeddings
                        if hasattr(base_model_unwrapped, 'embed_tokens'):
                            old_embedding = base_model_unwrapped.embed_tokens
                            new_embedding = nn.Embedding(target_vocab_size, old_embedding.embedding_dim)
                            new_embedding.weight.data[:vocab_size] = old_embedding.weight.data
                            if target_vocab_size > vocab_size:
                                new_embedding.weight.data[vocab_size:] = torch.randn(
                                    target_vocab_size - vocab_size, 
                                    old_embedding.embedding_dim
                                ) * 0.02
                            base_model_unwrapped.embed_tokens = new_embedding
                            print("✓ Resized embed_tokens manually")
                        
                        # Resize output embeddings (lm_head)
                        if hasattr(base_model_unwrapped, 'lm_head'):
                            old_lm_head = base_model_unwrapped.lm_head
                            new_lm_head = nn.Linear(
                                old_lm_head.in_features, 
                                target_vocab_size, 
                                bias=old_lm_head.bias is not None
                            )
                            new_lm_head.weight.data[:vocab_size] = old_lm_head.weight.data
                            if old_lm_head.bias is not None:
                                new_lm_head.bias.data[:vocab_size] = old_lm_head.bias.data
                            if target_vocab_size > vocab_size:
                                new_lm_head.weight.data[vocab_size:] = torch.randn(
                                    target_vocab_size - vocab_size,
                                    old_lm_head.in_features
                                ) * 0.02
                                if new_lm_head.bias is not None:
                                    new_lm_head.bias.data[vocab_size:] = 0.0
                            base_model_unwrapped.lm_head = new_lm_head
                            print("✓ Resized lm_head manually")
                else:
                    print("⚠️  Model does not have resize_token_embeddings method, skipping resize")
                
                # Update config
                config.vocab_size = target_vocab_size
                print(f"✓ Updated config.vocab_size to {config.vocab_size}")
            
            # Load LoRA adapter
            print(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            print("✓ LoRA adapter loaded")
            
            # CRITICAL FIX: Merge LoRA weights into base model for proper inference
            print("Merging LoRA weights into base model...")
            model = model.merge_and_unload()
            model.eval()
            print("✓ LoRA weights merged and model set to eval mode")
            
            # Load processor
            try:
                processor = AutoProcessor.from_pretrained(
                    base_model_name,
                    trust_remote_code=True,
                    token=token
                )
                print("✓ Loaded native AutoProcessor")
            except Exception as e:
                print(f"AutoProcessor failed: {e}, loading components separately...")
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=False)
                tokenizer.padding_side = "right"
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
                image_processor.size = {"height": 336, "width": 336}
                image_processor.crop_size = {"height": 336, "width": 336}
                
                # Create processor wrapper (same as training script - replaces <image> with 576 image_token_id tokens)
                class SimpleProcessor:
                    def __init__(self, tokenizer, image_processor, image_token_id=32000):
                        self.tokenizer = tokenizer
                        self.image_processor = image_processor
                        self.image_token_id = image_token_id
                        # Get the token sequence for <image>
                        self.image_token_seq = self.tokenizer.encode("<image>", add_special_tokens=False)
                        # Calculate number of image tokens: (image_size / patch_size)^2
                        # For 336x336 with patch_size 14: (336/14)^2 = 576
                        self.num_image_tokens = 576
                        print(f"<image> tokenizes as: {self.image_token_seq}")
                        print(f"Will replace with {self.num_image_tokens} image_token_id tokens: {self.image_token_id}")
                    
                    def __call__(self, text=None, images=None, return_tensors=None, padding=None, max_length=None, truncation=None, **kwargs):
                        outputs = {}
                        
                        if images is not None:
                            if not isinstance(images, list):
                                images = [images]
                            img_outputs = self.image_processor(images, return_tensors=return_tensors or "pt")
                            outputs["pixel_values"] = img_outputs["pixel_values"]
                        
                        if text is not None:
                            # Get padding and max_length from kwargs
                            pad_val = padding if padding else False
                            max_len = max_length if max_length else 2048
                            
                            # Handle single string vs list
                            is_single = isinstance(text, str)
                            if is_single:
                                text_list = [text]
                            else:
                                text_list = text
                            
                            # Tokenize text
                            tokenizer_kwargs = kwargs.copy()
                            if pad_val is False:
                                tokenizer_kwargs["padding"] = False
                            else:
                                tokenizer_kwargs["padding"] = pad_val
                            
                            # Ensure return_tensors is set
                            if return_tensors:
                                tokenizer_kwargs["return_tensors"] = return_tensors
                            
                            text_outputs = self.tokenizer(text_list, **tokenizer_kwargs)
                            
                            # Ensure all outputs are tensors if return_tensors="pt"
                            if return_tensors == "pt":
                                for key, value in text_outputs.items():
                                    if isinstance(value, list):
                                        text_outputs[key] = torch.tensor(value)
                            
                            # CRITICAL: Replace <image> token sequence with 576 image_token_id tokens
                            # The model expects 576 occurrences (one per image patch)
                            if "input_ids" in text_outputs and isinstance(text_outputs["input_ids"], torch.Tensor):
                                input_ids = text_outputs["input_ids"]
                                attention_mask = text_outputs.get("attention_mask", None)
                                
                                # Ensure input_ids is 2D [batch_size, seq_len]
                                if input_ids.dim() == 1:
                                    input_ids = input_ids.unsqueeze(0)
                                    if attention_mask is not None:
                                        attention_mask = attention_mask.unsqueeze(0)
                                
                                # Rebuild tensors with replaced image tokens
                                new_input_ids_list = []
                                new_attention_mask_list = []
                                
                                for batch_idx in range(input_ids.size(0)):
                                    ids = input_ids[batch_idx].tolist()
                                    new_ids = []
                                    i = 0
                                    
                                    while i < len(ids):
                                        if i + len(self.image_token_seq) <= len(ids) and ids[i:i+len(self.image_token_seq)] == self.image_token_seq:
                                            # Replace <image> token sequence with 576 image_token_id tokens
                                            new_ids.extend([self.image_token_id] * self.num_image_tokens)
                                            i += len(self.image_token_seq)
                                        else:
                                            new_ids.append(ids[i])
                                            i += 1
                                    
                                    # Truncate if needed
                                    if len(new_ids) > max_len:
                                        new_ids = new_ids[:max_len]
                                    
                                    # Only pad if padding is explicitly True or "max_length"
                                    if pad_val and len(new_ids) < max_len:
                                        pad_token_id = self.tokenizer.pad_token_id or 0
                                        new_ids.extend([pad_token_id] * (max_len - len(new_ids)))
                                    
                                    # Create tensor
                                    new_input_ids_list.append(torch.tensor(new_ids, dtype=input_ids.dtype))
                                    
                                    # Update attention_mask if present
                                    if attention_mask is not None:
                                        new_mask = [1] * len(new_ids)
                                        if pad_val and len(new_mask) < max_len:
                                            new_mask.extend([0] * (max_len - len(new_mask)))
                                        new_attention_mask_list.append(torch.tensor(new_mask, dtype=attention_mask.dtype))
                                
                                # Stack tensors
                                text_outputs["input_ids"] = torch.stack(new_input_ids_list)
                                
                                if attention_mask is not None:
                                    text_outputs["attention_mask"] = torch.stack(new_attention_mask_list)
                            
                            outputs.update(text_outputs)
                        
                        return outputs
                    
                    def apply_chat_template(self, *args, **kwargs):
                        return self.tokenizer.apply_chat_template(*args, **kwargs)
                
                image_token_id = getattr(config, 'image_token_id', 32000)
                processor = SimpleProcessor(tokenizer, image_processor, image_token_id)
                print("✓ Loaded components separately and wrapped")
        else:
            # For other models (Qwen, etc.)
            base_model = AutoModelForVision2Seq.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                token=token
            )
            print("✓ Base model loaded")
            
            model = PeftModel.from_pretrained(base_model, adapter_path)
            print("✓ LoRA adapter loaded")
            
            # CRITICAL FIX: Merge LoRA weights into base model for proper inference
            print("Merging LoRA weights into base model...")
            model = model.merge_and_unload()
            model.eval()
            print("✓ LoRA weights merged and model set to eval mode")
            
            processor = AutoProcessor.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                token=token
            )
            print("✓ Processor loaded")
        
        device = next(model.parameters()).device
        print(f"✓ Model loaded on device: {device}")
        print(f"✓ Model type: Fine-tuned (with LoRA adapter)")
        
        return model, processor, device
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load fine-tuned model")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        raise


def extract_answer_from_response(generated_text: str) -> str:
    """Extract the answer from generated text."""
    # Remove common prefixes that might appear in generated text
    prefixes = ["assistant:", "assistant", "### Response:", "Response:", "Answer:", "Answer", "ASSISTANT:", "ASSISTANT"]
    for prefix in prefixes:
        if generated_text.strip().lower().startswith(prefix.lower()):
            generated_text = generated_text.strip()[len(prefix):].strip()
    
    # Remove end-of-sequence tokens
    for token in ["</s>", "<|endoftext|>", "<|im_end|>"]:
        if generated_text.endswith(token):
            generated_text = generated_text[:-len(token)].strip()
    
    # Take first line (in case model generates multiple lines)
    lines = generated_text.split('\n')
    if lines:
        generated_text = lines[0].strip()
    
    # Remove "USER:" if it somehow appears (shouldn't happen with proper decoding)
    if generated_text.strip().startswith("USER:"):
        generated_text = generated_text.strip()[5:].strip()
    
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
    
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    stage_precision = defaultdict(float)
    stage_recall = defaultdict(float)
    stage_f1 = defaultdict(float)
    qtype_precision = defaultdict(float)
    qtype_recall = defaultdict(float)
    qtype_f1 = defaultdict(float)
    
    for item in tqdm(test_data, desc="Evaluating (Fine-tuned)"):
        if use_instruction:
            question = item.get('instruction', item.get('question', ''))
        else:
            question = item.get('question', '')
        
        ground_truth = item.get('answer', '').strip()
        # Handle both 'image' and 'image_filename' fields
        image_path = item.get('image', '') or item.get('image_filename', '')
        stage = item.get('stage', 'unknown')
        question_type = item.get('question_type', 'unknown')
        
        if not image_path:
            results['errors'].append(f"Missing image path for item: {item.get('id', item.get('image_id', 'unknown'))}")
            continue
        
        full_image_path = os.path.join(image_base_path, image_path)
        if not os.path.exists(full_image_path):
            results['errors'].append(f"Image not found: {full_image_path}")
            continue
        
        try:
            image = Image.open(full_image_path).convert('RGB')
            
            # Format prompt for LLaVA - MUST match training format exactly
            # Training format: "USER: <image>\n{question}\nASSISTANT: {answer}"
            # Evaluation format: "USER: <image>\n{question}\nASSISTANT:"
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            
            # Process inputs
            inputs = processor(text=[prompt], images=[image], return_tensors="pt")
            # Ensure all values are tensors before moving to device
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else torch.tensor(v).to(device) if isinstance(v, (list, tuple)) else v) for k, v in inputs.items()}
            
            # Get input length to extract only generated tokens
            input_len = inputs['input_ids'].shape[1]
            
            # Get eos_token_id for stopping criteria
            eos_token_id = None
            if hasattr(processor, 'tokenizer'):
                eos_token_id = processor.tokenizer.eos_token_id
            elif hasattr(processor, 'eos_token_id'):
                eos_token_id = processor.eos_token_id
            
            # Generate
            # CRITICAL FIX: LLaVA models need do_sample=True for proper generation
            with torch.no_grad():
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": 512,  # Increased from 256
                    "do_sample": True,  # CRITICAL: Changed from False to True for LLaVA
                    "temperature": 0.2,  # Low temperature for consistency
                    "top_p": 0.9,  # Nucleus sampling
                }
                if eos_token_id is not None:
                    generate_kwargs["eos_token_id"] = eos_token_id
                
                # Add pad_token_id to avoid warnings
                tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
                
                generated_ids = model.generate(**generate_kwargs)
            
            # Decode only the generated tokens (skip the input prompt)
            generated_token_ids = generated_ids[0][input_len:]
            
            # CRITICAL: Filter out image_token_id and other special tokens before decoding
            # The model might generate image_token_id (32000) which cannot be decoded as text
            image_token_id = None
            if hasattr(model, 'config') and hasattr(model.config, 'image_token_id'):
                image_token_id = model.config.image_token_id
            elif hasattr(processor, 'image_token_id'):
                image_token_id = processor.image_token_id
            
            # Get tokenizer for filtering
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            
            # Filter out image_token_id, pad_token_id, and other invalid tokens
            filtered_token_ids = []
            pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else None
            eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None else None
            
            # Get vocab size for validation
            vocab_size = getattr(tokenizer, 'vocab_size', None)
            
            for token_id in generated_token_ids:
                token_id_int = int(token_id.item() if hasattr(token_id, 'item') else token_id)
                
                # Skip image_token_id
                if image_token_id is not None and token_id_int == image_token_id:
                    continue
                
                # Skip pad_token_id
                if pad_token_id is not None and token_id_int == pad_token_id:
                    continue
                
                # Skip eos_token_id (we want to stop at EOS, not include it)
                if eos_token_id is not None and token_id_int == eos_token_id:
                    break  # Stop generation at EOS
                
                # Skip tokens that are >= vocab_size (invalid)
                if vocab_size is not None and token_id_int >= vocab_size:
                    continue
                
                # Skip negative or zero token IDs (invalid)
                if token_id_int <= 0:
                    continue
                
                # CRITICAL FIX: Test if token decodes to valid text
                # Some token IDs might be valid but decode to control characters or artifacts
                try:
                    test_text = tokenizer.decode([token_id_int], skip_special_tokens=True)
                    # Skip tokens that decode to only control characters, whitespace, or artifacts
                    if test_text and test_text.strip() and not test_text.startswith('/**'):
                        filtered_token_ids.append(token_id_int)
                except (KeyError, ValueError, IndexError):
                    # Token ID is invalid, skip it
                    continue
            
            # Decode the filtered tokens
            if filtered_token_ids:
                try:
                    if hasattr(processor, 'tokenizer'):
                        generated_text = processor.tokenizer.decode(filtered_token_ids, skip_special_tokens=True)
                    else:
                        generated_text = processor.decode(filtered_token_ids, skip_special_tokens=True)
                    
                    # CRITICAL FIX: Clean up any remaining artifacts
                    # Remove patterns like /******/ that might slip through
                    if '/**' in generated_text:
                        # Try to extract only valid text parts
                        parts = generated_text.split('/**')
                        # Keep only parts that look like valid text
                        cleaned_parts = [p.strip() for p in parts if p.strip() and not p.strip().startswith('*')]
                        generated_text = ' '.join(cleaned_parts) if cleaned_parts else ""
                    
                except Exception as decode_error:
                    # If decoding fails, log error and use empty string
                    print(f"⚠️  Decoding error for sample {item.get('id', 'unknown')}: {decode_error}")
                    print(f"   Token IDs: {filtered_token_ids[:20]}...")  # Show first 20 tokens
                    generated_text = ""
            else:
                # If all tokens were filtered out, use empty string
                generated_text = ""
            
            # Extract answer
            prediction = extract_answer_from_response(generated_text)
            
            # Evaluate
            is_correct = smart_match(prediction, ground_truth)
            
            # For multi-label questions, use set-based matching
            if ';' in ground_truth:
                pred_set = parse_labels(prediction)
                gt_set = parse_labels(ground_truth)
                is_correct = len(pred_set & gt_set) > 0
                precision, recall, f1 = calculate_precision_recall_f1(pred_set, gt_set)
            else:
                precision = 1.0 if is_correct else 0.0
                recall = 1.0 if is_correct else 0.0
                f1 = 1.0 if is_correct else 0.0
            
            # Update results
            results['total'] += 1
            if is_correct:
                results['correct'] += 1
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            
            results['by_stage'][stage]['total'] += 1
            if is_correct:
                results['by_stage'][stage]['correct'] += 1
            stage_precision[stage] += precision
            stage_recall[stage] += recall
            stage_f1[stage] += f1
            
            results['by_question_type'][question_type]['total'] += 1
            if is_correct:
                results['by_question_type'][question_type]['correct'] += 1
            qtype_precision[question_type] += precision
            qtype_recall[question_type] += recall
            qtype_f1[question_type] += f1
            
            results['predictions'].append({
                'id': item.get('id', 'unknown'),
                'question': question,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'correct': is_correct
            })
            
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            results['errors'].append(error_msg)
            print(f"\n⚠️  Error processing item: {error_msg}")
            import traceback
            traceback.print_exc()
    
    # Calculate final metrics
    if results['total'] > 0:
        results['accuracy'] = (results['correct'] / results['total']) * 100
        results['precision'] = (total_precision / results['total']) * 100
        results['recall'] = (total_recall / results['total']) * 100
        results['f1'] = (total_f1 / results['total']) * 100
    
    # Calculate per-stage and per-question-type metrics
    for stage in results['by_stage']:
        stage_data = results['by_stage'][stage]
        if stage_data['total'] > 0:
            stage_data['accuracy'] = (stage_data['correct'] / stage_data['total']) * 100
            stage_data['precision'] = (stage_precision[stage] / stage_data['total']) * 100
            stage_data['recall'] = (stage_recall[stage] / stage_data['total']) * 100
            stage_data['f1'] = (stage_f1[stage] / stage_data['total']) * 100
    
    for qtype in results['by_question_type']:
        qtype_data = results['by_question_type'][qtype]
        if qtype_data['total'] > 0:
            qtype_data['accuracy'] = (qtype_data['correct'] / qtype_data['total']) * 100
            qtype_data['precision'] = (qtype_precision[qtype] / qtype_data['total']) * 100
            qtype_data['recall'] = (qtype_recall[qtype] / qtype_data['total']) * 100
            qtype_data['f1'] = (qtype_f1[qtype] / qtype_data['total']) * 100
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLaVA-Med model")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name (e.g., microsoft/llava-med-v1.5-mistral-7b)")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter (e.g., models/llava_med_kvasir_instruction/best_model)")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--image_dir", type=str, required=True, help="Base directory for images")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file for results")
    parser.add_argument("--use_instruction", action="store_true", help="Use instruction field if available")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FINE-TUNED MODEL EVALUATION")
    print("=" * 80)
    print(f"Base Model: {args.base_model}")
    print(f"Adapter Path: {args.adapter_path}")
    print(f"Test Data: {args.test_data}")
    print(f"Image Directory: {args.image_dir}")
    print(f"Output: {args.output}")
    if args.use_instruction:
        print("Mode: Using instruction field if available")
    else:
        print("Mode: Using question field only")
    print("=" * 80)
    
    try:
        # Load fine-tuned model
        hf_token = os.getenv("HF_TOKEN")
        print("\n" + "=" * 80)
        print("STEP 1: Loading Fine-Tuned Model")
        print("=" * 80)
        model, processor, device = load_finetuned_model(args.base_model, args.adapter_path, hf_token=hf_token)
        
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
    print("FINE-TUNED MODEL EVALUATION RESULTS")
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
    print("Fine-tuned model evaluation completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

