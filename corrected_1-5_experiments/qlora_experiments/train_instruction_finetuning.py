#!/usr/bin/env python3
"""
Instruction Fine-tuning Script for EndoVis2018
QLoRA Fine-tuning of Qwen3-VL-8B-Instruct with instruction templates
Supports JSON format datasets (Surgery R1 split)
"""
import os
import sys
import json
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Disable DeepSpeed auto-detection to avoid engine errors
# Must be set before any imports that might use DeepSpeed
os.environ.pop('DS_ACCELERATOR', None)
os.environ.pop('DS_SKIP_CUDA_CHECK', None)
os.environ.pop('DEEPSPEED_CONFIG_FILE', None)
os.environ['ACCELERATE_USE_DEEPSPEED'] = 'false'
os.environ['ACCELERATE_USE_CPU'] = 'false'
os.environ['ACCELERATE_USE_XPU'] = 'false'
# Force CUDA backend
os.environ['ACCELERATE_USE_CUDA'] = 'true'

# Configure cuDNN for stability
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from PIL import Image

# ============================================================================
# Dataset - Supports both JSON and JSONL formats
# ============================================================================
class InstructionVQADataset(Dataset):
    def __init__(self, data_path, image_root, processor, max_length=3072):
        self.data = []
        self.image_root = Path(image_root)
        self.processor = processor
        self.max_length = max_length
        
        data_path = Path(data_path)
        
        # Load data (supports both JSON and JSONL)
        if data_path.suffix == '.json':
            with open(data_path) as f:
                self.data = json.load(f)
        else:  # JSONL
            with open(data_path) as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data):,} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get image filename (try different field names)
        image_filename = item.get('image_filename') or item.get('image_name') or item.get('image_id', '')
        if not image_filename.endswith(('.jpg', '.png', '.jpeg')):
            # Try to construct filename from image_id
            image_id = item.get('image_id', '')
            if image_id:
                # Format: endovis_seq2_frame000 -> endovis_seq2_frame000.jpg
                image_filename = f"{image_id}.jpg"
        
        # Load image
        img_path = self.image_root / image_filename
        if not img_path.exists():
            # Try alternative paths
            alt_paths = [
                self.image_root / image_filename.replace('.jpg', '.png'),
                self.image_root / image_filename.replace('endovis_', ''),
                self.image_root / f"seq{item.get('sequence', '')}/frame{item.get('frame', '').zfill(3)}.jpg",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    img_path = alt_path
                    break
        
        image = Image.open(img_path).convert('RGB')
        
        # Get instruction and answer (instruction already includes question)
        user_message = item.get('instruction', item.get('question', ''))
        assistant_message = item.get('answer', '')
        
        # Check if this is a LLaVA model
        is_llava = 'llava' in str(self.processor.__class__).lower() or 'llava' in str(type(self.processor)).lower()
        
        if is_llava:
            # LLaVA models need <image> placeholder in text and images passed together
            # LLaVA-Med format: "USER: <image>\n{question}\nASSISTANT: {answer}"
            # The processor will replace <image> with image token embeddings
            text = f"USER: <image>\n{user_message}\nASSISTANT: {assistant_message}"
            
            # Process with both text and images - processor will handle image token embedding
            inputs = self.processor(
                text=text,
                images=image,  # Pass image directly, not as list
                return_tensors="pt",
                padding=False,
                max_length=self.max_length,
                truncation=True
            )
        else:
            # Qwen3-VL/MedGemma chat template format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_message}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_message}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text) if text else ""
            
            # Process (NO padding here - will pad dynamically per batch)
            inputs = self.processor(
                text=[text] if isinstance(text, str) else text,
                images=[image],
                return_tensors="pt",
                padding=False,
                max_length=self.max_length,
                truncation=True
            )
        
        # Extract tensors (remove batch dimension)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        # Handle different processor outputs (Qwen3-VL has image_grid_thw, MedGemma doesn't)
        image_grid_thw = inputs.get('image_grid_thw')
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.squeeze(0)
        else:
            # MedGemma doesn't use image_grid_thw, create a dummy tensor for compatibility
            image_grid_thw = torch.tensor([])  # Will be handled in collate function
        
        # Labels for causal LM (mask prompt, keep answer)
        labels = input_ids.clone()
        
        # Find assistant response start (model-specific)
        # Check if this is a Qwen model (has image_grid_thw), MedGemma, or LLaVA
        is_qwen = image_grid_thw.numel() > 0 if hasattr(image_grid_thw, 'numel') else False
        is_llava = 'llava' in str(self.processor.__class__).lower() or 'llava' in str(type(self.processor)).lower()
        
        assistant_token_start = -1
        input_ids_list = input_ids.tolist()
        
        if is_qwen:
            # Qwen3-VL chat template format: <|im_start|>assistant\n
            assistant_start_token_ids = [151644, 77091, 198]  # <|im_start|>assistant\n
            for i in range(len(input_ids_list) - len(assistant_start_token_ids) + 1):
                if input_ids_list[i:i+len(assistant_start_token_ids)] == assistant_start_token_ids:
                    assistant_token_start = i + len(assistant_start_token_ids)
                    break
            
            # Fallback: Try alternative pattern
            if assistant_token_start == -1:
                alt_pattern = [151644, 77091]  # <|im_start|>assistant
                for i in range(len(input_ids_list) - len(alt_pattern) + 1):
                    if input_ids_list[i:i+len(alt_pattern)] == alt_pattern:
                        if i + len(alt_pattern) < len(input_ids_list):
                            assistant_token_start = i + len(alt_pattern) + 1
                        break
        elif is_llava:
            # LLaVA chat template format: ASSISTANT: or similar
            # Try to find common LLaVA patterns
            try:
                tokenizer = self.processor.tokenizer
                # LLaVA typically uses "ASSISTANT:" or "### Assistant:"
                assistant_patterns = ["ASSISTANT:", "### Assistant:", "Assistant:"]
                for pattern in assistant_patterns:
                    pattern_tokens = tokenizer.encode(pattern, add_special_tokens=False)
                    if len(pattern_tokens) > 0:
                        for i in range(len(input_ids_list) - len(pattern_tokens) + 1):
                            if input_ids_list[i:i+len(pattern_tokens)] == pattern_tokens:
                                assistant_token_start = i + len(pattern_tokens)
                                break
                        if assistant_token_start > 0:
                            break
            except:
                pass
            
            # Fallback: Use answer text search
            if assistant_token_start == -1:
                try:
                    answer_tokens = self.processor.tokenizer.encode(
                        assistant_message, add_special_tokens=False
                    )
                    if len(answer_tokens) > 0:
                        for i in range(len(input_ids_list) - len(answer_tokens) + 1):
                            if input_ids_list[i:i+len(answer_tokens)] == answer_tokens:
                                assistant_token_start = i
                                break
                except:
                    pass
        else:
            # MedGemma uses Gemma chat template - look for "<start_of_turn>model" pattern
            # Token IDs may vary, so we'll use a heuristic: find where assistant message likely starts
            # by looking for common patterns or using the tokenizer to find assistant role tokens
            try:
                # Try to find assistant role token using tokenizer
                tokenizer = self.processor.tokenizer
                assistant_text = "<start_of_turn>model\n"
                assistant_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
                if len(assistant_tokens) > 0:
                    for i in range(len(input_ids_list) - len(assistant_tokens) + 1):
                        if input_ids_list[i:i+len(assistant_tokens)] == assistant_tokens:
                            assistant_token_start = i + len(assistant_tokens)
                            break
            except:
                pass
            
            # Fallback: If we can't find the pattern, mask based on a heuristic
            # Look for the answer text in the tokenized sequence (less precise but works)
            if assistant_token_start == -1:
                # Tokenize the assistant message to find where it appears
                try:
                    assistant_tokens = self.processor.tokenizer.encode(
                        assistant_message, add_special_tokens=False
                    )
                    # Search for the assistant message tokens in the full sequence
                    for i in range(len(input_ids_list) - len(assistant_tokens) + 1):
                        if input_ids_list[i:i+len(assistant_tokens)] == assistant_tokens:
                            assistant_token_start = i
                            break
                except:
                    pass
        
        # Mask everything before assistant response (if found)
        if assistant_token_start > 0 and assistant_token_start < len(labels):
            labels[:assistant_token_start] = -100
        else:
            # If we can't find the assistant start, mask a conservative portion
            # This is a fallback - ideally we should find the exact position
            # For now, mask the first 80% as a heuristic (user prompt is typically longer)
            mask_point = int(len(labels) * 0.8)
            if mask_point > 0:
                labels[:mask_point] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'labels': labels
        }

# ============================================================================
# Collate Function - DYNAMIC PADDING (batch-wise)
# ============================================================================
def collate_fn(batch):
    """
    Collate batch of samples with DYNAMIC PADDING.
    Each batch is padded only to the longest sequence in THAT batch,
    not to the global max_length. This dramatically speeds up training!
    """
    # Find max length in this batch
    max_len_in_batch = max(x['input_ids'].shape[0] for x in batch)
    
    # Pad each sample to batch max (not global max)
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for x in batch:
        seq_len = x['input_ids'].shape[0]
        pad_len = max_len_in_batch - seq_len
        
        if pad_len > 0:
            # Pad input_ids
            input_ids_padded = torch.cat([
                x['input_ids'],
                torch.full((pad_len,), 0, dtype=x['input_ids'].dtype)  # Pad token ID
            ])
            
            # Pad attention_mask
            attention_mask_padded = torch.cat([
                x['attention_mask'],
                torch.zeros(pad_len, dtype=x['attention_mask'].dtype)
            ])
            
            # Pad labels
            labels_padded = torch.cat([
                x['labels'],
                torch.full((pad_len,), -100, dtype=x['labels'].dtype)  # Ignore index
            ])
        else:
            input_ids_padded = x['input_ids']
            attention_mask_padded = x['attention_mask']
            labels_padded = x['labels']
        
        input_ids_list.append(input_ids_padded)
        attention_mask_list.append(attention_mask_padded)
        labels_list.append(labels_padded)
    
    # Handle image_grid_thw (Qwen3-VL has it, MedGemma doesn't)
    image_grid_thw_list = [x['image_grid_thw'] for x in batch]
    has_image_grid_thw = any(x.numel() > 0 for x in image_grid_thw_list)
    
    result = {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack(labels_list),
    }
    
    # Only include image_grid_thw if it exists (Qwen3-VL models)
    if has_image_grid_thw:
        result['image_grid_thw'] = torch.stack(image_grid_thw_list)
    
    return result

# ============================================================================
# Main Training
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python train_instruction_finetuning.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("="*80)
    print(f"Instruction Fine-tuning: {config.get('experiment_name', 'Unknown')}")
    print("="*80)
    print(f"Config: {config_path}")
    print(f"Model: {config['model_name']}")
    print(f"Epochs: {config['train']['epochs']}")
    print(f"QLoRA rank: {config['lora']['r']}")
    print(f"Target modules: {config['lora']['target_modules']}")
    print("="*80)
    
    # Setup
    model_name = config['model_name']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Authenticate with Hugging Face if token is provided (for gated models)
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token:
        print("\nAuthenticating with Hugging Face...")
        from huggingface_hub import login
        login(token=hf_token)
        print("✓ Authentication successful")
    
    # Load processor
    print("\nLoading processor...")
    
    # Check if this is a LLaVA model
    is_llava = 'llava' in model_name.lower()
    
    if is_llava:
        # LLaVA models - use proper LLaVA loading with trust_remote_code=True
        processor = None
        try:
            # First try AutoProcessor with trust_remote_code=True (proper method)
            processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                token=hf_token
            )
            print("✓ Loaded using AutoProcessor with trust_remote_code")
            
            # Get image_token_id from processor if available (proper method)
            if hasattr(processor, 'image_token') and processor.image_token:
                try:
                    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
                    if hasattr(processor, 'image_token_id'):
                        processor.image_token_id = image_token_id
                        print(f"  ✓ Set image_token_id from processor.image_token: {image_token_id}")
                except Exception as e:
                    print(f"  ⚠️  Could not get image_token_id from processor.image_token: {e}")
        except Exception as e:
            print(f"⚠️  AutoProcessor failed: {e}")
            try:
                # Fallback: try LlavaProcessor
                from transformers import LlavaProcessor
                processor = LlavaProcessor.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
                print("✓ Loaded using LlavaProcessor")
                
                # Get image_token_id from processor if available
                if hasattr(processor, 'image_token') and processor.image_token:
                    try:
                        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
                        if hasattr(processor, 'image_token_id'):
                            processor.image_token_id = image_token_id
                            print(f"  ✓ Set image_token_id from processor.image_token: {image_token_id}")
                    except Exception as e2:
                        print(f"  ⚠️  Could not get image_token_id from processor.image_token: {e2}")
            except Exception as e3:
                print(f"⚠️  LlavaProcessor failed: {e3}")
                # Try loading components separately
                try:
                    from transformers import AutoTokenizer, CLIPImageProcessor
                    # Use slow tokenizer (use_fast=False) to avoid issues with image tokens
                    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token, use_fast=False)
                    # For LLaVA, try to get image processor from vision model
                    try:
                        image_processor = CLIPImageProcessor.from_pretrained(
                            "openai/clip-vit-large-patch14-336",  # Common CLIP model for LLaVA
                            trust_remote_code=True
                        )
                    except:
                        # Fallback: try to load from model repo
                        image_processor = CLIPImageProcessor.from_pretrained(
                            model_name, trust_remote_code=True, token=hf_token
                        )
                    # Create processor-like object that mimics LlavaProcessor interface
                    class LlavaProcessorWrapper:
                        def __init__(self, image_processor, tokenizer):
                            self.image_processor = image_processor
                            self.tokenizer = tokenizer
                            # Image token ID will be set after model loads
                            # For LLaVA-Med (Mistral-based), it's typically vocab_size (32000)
                            self.image_token_id = None
                        
                        def __call__(self, text=None, images=None, return_tensors=None, padding=None, max_length=None, truncation=None, **kwargs):
                            # For LLaVA, we MUST replace <image> with the actual image token ID
                            # The native LlavaProcessor does this automatically, we need to replicate it
                            
                            # Handle text tokenization
                            if text is not None:
                                # Ensure text is a string (not a list)
                                if isinstance(text, list):
                                    text = text[0] if text else ""
                                
                                # Count <image> placeholders
                                image_count = text.count('<image>')
                                
                                # Tokenize text - this will tokenize <image> as regular tokens
                                text_inputs = self.tokenizer(
                                    text,
                                    return_tensors=return_tensors,
                                    padding=padding if padding else False,
                                    max_length=max_length,
                                    truncation=truncation,
                                    **kwargs
                                )
                                
                                # Replace <image> token sequences with actual image token ID
                                if image_count > 0 and self.image_token_id is not None:
                                    input_ids = text_inputs['input_ids']
                                    if return_tensors == "pt":
                                        import torch
                                        input_ids_list = input_ids[0].tolist()
                                    else:
                                        input_ids_list = input_ids[0] if isinstance(input_ids[0], list) else input_ids
                                    
                                    # Find what <image> tokenizes to
                                    image_token_seq = self.tokenizer.encode('<image>', add_special_tokens=False)
                                    
                                    if image_token_seq:
                                        print(f"Debug: <image> tokenizes to: {image_token_seq}, image_token_id={self.image_token_id}")
                                        print(f"Debug: input_ids_list length: {len(input_ids_list)}, first 20 tokens: {input_ids_list[:20]}")
                                        # Replace all occurrences of image_token_seq with image_token_id
                                        new_input_ids = []
                                        i = 0
                                        replaced_count = 0
                                        while i < len(input_ids_list):
                                            # Check if we're at the start of an image token sequence
                                            if i + len(image_token_seq) <= len(input_ids_list):
                                                if input_ids_list[i:i+len(image_token_seq)] == image_token_seq:
                                                    # Replace with single image token ID
                                                    new_input_ids.append(self.image_token_id)
                                                    i += len(image_token_seq)
                                                    replaced_count += 1
                                                    continue
                                            new_input_ids.append(input_ids_list[i])
                                            i += 1
                                        
                                        print(f"Debug: Replaced {replaced_count} image token sequences out of {image_count} expected")
                                        print(f"Debug: After replacement, first 20 tokens: {new_input_ids[:20]}")
                                        print(f"Debug: Checking if {self.image_token_id} is in new_input_ids: {self.image_token_id in new_input_ids}")
                                        
                                        # Update input_ids
                                        if return_tensors == "pt":
                                            text_inputs['input_ids'] = torch.tensor([new_input_ids], dtype=input_ids.dtype)
                                            # Verify the replacement worked
                                            if self.image_token_id in new_input_ids:
                                                print(f"Debug: ✓ Confirmed {self.image_token_id} is in the tensor")
                                            else:
                                                print(f"Debug: ✗ WARNING: {self.image_token_id} NOT found in new_input_ids!")
                                        else:
                                            text_inputs['input_ids'] = [new_input_ids]
                                        
                                        if replaced_count != image_count:
                                            print(f"Warning: Expected {image_count} image tokens, replaced {replaced_count}")
                                    else:
                                        print(f"Warning: <image> tokenizes to empty sequence!")
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
                            combined = {**text_inputs, **image_inputs}
                            return combined
                        
                        def apply_chat_template(self, *args, **kwargs):
                            return self.tokenizer.apply_chat_template(*args, **kwargs)
                    
                    processor = LlavaProcessorWrapper(image_processor, tokenizer)
                    print("✓ Loaded components separately and wrapped")
                except Exception as e3:
                    print(f"✗ All loading methods failed: {e3}")
                    raise RuntimeError(f"Could not load processor for {model_name}. Tried LlavaProcessor, AutoProcessor, and component loading.")
    else:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    
    # Load model - try multiple strategies for LLaVA models
    hf_token = os.environ.get('HF_TOKEN')
    model = None
    
    if is_llava:
        print("\nLoading LLaVA model (using proper LLaVA loading method)...")
        
        # Use proper LLaVA loading with trust_remote_code=True
        # Strategy 1: Try loading with 4-bit quantization
        try:
            print("  Strategy 1: Loading with 4-bit quantization...")
            from transformers import LlavaForConditionalGeneration
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,  # Key for custom architectures
                quantization_config=bnb_config,
                token=hf_token
            )
            print("  ✓ Loaded with LlavaForConditionalGeneration + 4-bit")
        except (RuntimeError, OSError, Exception) as e:
            print(f"  ⚠️  4-bit quantization failed: {type(e).__name__}")
            print("  Strategy 2: Loading without quantization (bf16)...")
            
            # Strategy 2: Load without quantization, use bf16
            try:
                from transformers import LlavaForConditionalGeneration
                model = LlavaForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,  # Key for custom architectures
                    token=hf_token
                )
                print("  ✓ Loaded with LlavaForConditionalGeneration (bf16, no quant)")
            except Exception as e3:
                print(f"  ⚠️  bf16 loading failed: {type(e3).__name__}")
                print("  Strategy 3: Loading with fp16...")
                
                # Strategy 3: Try fp16
                try:
                    model = AutoModelForImageTextToText.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        device_map="auto",
                        token=hf_token
                    )
                    print("  ✓ Loaded with AutoModelForImageTextToText (fp16)")
                except Exception as e4:
                    print(f"  ⚠️  fp16 loading failed: {type(e4).__name__}")
                    print("  Strategy 4: Loading with default dtype...")
                    
                    # Strategy 4: Default dtype - try loading config manually first, then model
                    try:
                        from transformers import LlavaForConditionalGeneration, AutoConfig
                        import json
                        from huggingface_hub import hf_hub_download
                        
                        # Try to load config manually to bypass model type check
                        print("  Attempting to load config manually...")
                        try:
                            # Download config file directly
                            config_path = hf_hub_download(
                                repo_id=model_name,
                                filename="config.json",
                                token=hf_token
                            )
                            with open(config_path, 'r') as f:
                                config_dict = json.load(f)
                            
                            # Temporarily change model_type to 'llava' if it's 'llava_mistral'
                            if config_dict.get('model_type') == 'llava_mistral':
                                config_dict['model_type'] = 'llava'
                                print("  ⚠️  Changed model_type from 'llava_mistral' to 'llava' for compatibility")
                            
                            # Create config from dict using LlavaConfig
                            from transformers import LlavaConfig
                            config = LlavaConfig.from_dict(config_dict)
                            # Set trust_remote_code attribute if needed
                            if hasattr(config, 'auto_map'):
                                config.trust_remote_code = True
                            
                            # Now load model with the config
                            model = LlavaForConditionalGeneration.from_pretrained(
                                model_name,
                                config=config,
                                trust_remote_code=True,
                                device_map="auto",
                                token=hf_token
                            )
                            print("  ✓ Loaded with LlavaForConditionalGeneration (default dtype, manual config)")
                        except Exception as config_e:
                            print(f"  ⚠️  Manual config loading failed: {type(config_e).__name__}")
                            # Fallback: try direct loading without config
                            model = LlavaForConditionalGeneration.from_pretrained(
                                model_name,
                                trust_remote_code=True,
                                device_map="auto",
                                token=hf_token
                            )
                            print("  ✓ Loaded with LlavaForConditionalGeneration (default dtype)")
                    except Exception as e5:
                        print(f"  ⚠️  LlavaForConditionalGeneration failed: {type(e5).__name__}")
                        # Final fallback: try AutoModelForImageTextToText with manual config
                        try:
                            from transformers import AutoConfig
                            import json
                            from huggingface_hub import hf_hub_download
                            
                            # Try manual config loading for AutoModel too
                            try:
                                config_path = hf_hub_download(
                                    repo_id=model_name,
                                    filename="config.json",
                                    token=hf_token
                                )
                                with open(config_path, 'r') as f:
                                    config_dict = json.load(f)
                                
                                if config_dict.get('model_type') == 'llava_mistral':
                                    config_dict['model_type'] = 'llava'
                                
                                # Use LlavaConfig for llava model type
                                from transformers import LlavaConfig
                                config = LlavaConfig.from_dict(config_dict)
                                # Set trust_remote_code attribute if needed
                                if hasattr(config, 'auto_map'):
                                    config.trust_remote_code = True
                                model = AutoModelForImageTextToText.from_pretrained(
                                    model_name,
                                    config=config,
                                    trust_remote_code=True,
                                    device_map="auto",
                                    token=hf_token
                                )
                                print("  ✓ Loaded with AutoModelForImageTextToText (default dtype, manual config)")
                            except Exception as config_e2:
                                # Final fallback: direct loading
                                model = AutoModelForImageTextToText.from_pretrained(
                                    model_name,
                                    trust_remote_code=True,
                                    device_map="auto",
                                    token=hf_token
                                )
                                print("  ✓ Loaded with AutoModelForImageTextToText (default dtype)")
                        except Exception as e6:
                            print(f"  ⚠️  AutoModelForImageTextToText failed: {type(e6).__name__}")
                            raise RuntimeError(f"All model loading strategies failed. Last error: {e6}")
        
        if model is None:
            raise RuntimeError("Failed to load LLaVA model with all strategies")
        
        # Set image token ID for LLaVA models after model loads
        # Priority: 1) model.config.image_token_id, 2) processor.image_token, 3) manual detection
        if is_llava:
            # First, try to get from model config (most reliable)
            if hasattr(model.config, 'image_token_id') and model.config.image_token_id is not None:
                processor.image_token_id = model.config.image_token_id
                print(f"  ✓ Set image_token_id from model.config.image_token_id: {processor.image_token_id}")
            # Second, try to get from processor.image_token (proper method)
            elif not hasattr(processor, 'image_token_id') or processor.image_token_id is None:
                if hasattr(processor, 'image_token') and processor.image_token:
                    try:
                        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
                        processor.image_token_id = image_token_id
                        print(f"  ✓ Set image_token_id from processor.image_token: {image_token_id}")
                    except Exception as e:
                        print(f"  ⚠️  Could not get image_token_id from processor.image_token: {e}")
            
            # If still not set, try manual detection (fallback)
            if not hasattr(processor, 'image_token_id') or processor.image_token_id is None:
                try:
                    # For LLaVA-Med (Mistral-based), image token is typically at vocab_size
                    # Check if embedding layer is extended to include it
                    if hasattr(model, 'get_input_embeddings'):
                        emb = model.get_input_embeddings()
                        if hasattr(emb, 'num_embeddings') and hasattr(model.config, 'vocab_size'):
                            print(f"  Debug: embedding.num_embeddings={emb.num_embeddings}, config.vocab_size={model.config.vocab_size}")
                            # If embedding is extended beyond vocab_size, image token is at vocab_size
                            if emb.num_embeddings > model.config.vocab_size:
                                processor.image_token_id = model.config.vocab_size
                                print(f"  ✓ Set image_token_id to vocab_size (extended embedding): {processor.image_token_id}")
                            elif emb.num_embeddings == model.config.vocab_size:
                                # Embedding exactly matches vocab_size - NOT extended, use vocab_size - 1
                                # Valid indices are 0 to (num_embeddings - 1), so vocab_size (32000) is out of bounds
                                processor.image_token_id = model.config.vocab_size - 1
                                print(f"  ⚠️  Set image_token_id to vocab_size - 1 (embedding not extended, size={emb.num_embeddings}): {processor.image_token_id}")
                            else:
                                # Embedding smaller than vocab_size - use vocab_size - 1
                                processor.image_token_id = model.config.vocab_size - 1
                                print(f"  ⚠️  Set image_token_id to vocab_size - 1 (embedding smaller): {processor.image_token_id}")
                    elif hasattr(model.config, 'vocab_size'):
                        # Fallback: try vocab_size first (most LLaVA models extend the embedding)
                        processor.image_token_id = model.config.vocab_size
                        print(f"  ⚠️  Set image_token_id to vocab_size (fallback): {processor.image_token_id}")
                except Exception as e:
                    print(f"  ⚠️  Could not set image_token_id from model: {e}")
                    # Last resort: use tokenizer vocab_size
                    if hasattr(processor.tokenizer, 'vocab_size'):
                        processor.image_token_id = processor.tokenizer.vocab_size
                        print(f"  ⚠️  Using tokenizer vocab_size as fallback: {processor.image_token_id}")
        
        # Only apply prepare_model_for_kbit_training if we used quantization
        if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
            model = prepare_model_for_kbit_training(model)
            print("  ✓ Prepared model for 4-bit training")
        else:
            print("  ⚠️  Model loaded without quantization - will use more memory")
            # Enable gradient checkpointing to save memory
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                print("  ✓ Enabled gradient checkpointing")
    else:
        # Non-LLaVA models: use standard 4-bit quantization
        print("\nLoading model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            token=hf_token
        )
        model = prepare_model_for_kbit_training(model)
    
    print("\n*** NEW TRAINING (Instruction Fine-tuning) ***")
    print("Creating new LoRA adapter...")
    # LoRA configuration
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("✓ Created new LoRA adapter")
    
    model.print_trainable_parameters()
    
    # Load datasets (supports both JSON and JSONL formats)
    print("\nLoading datasets...")
    # Support both train_json/train_jsonl and val_json/val_jsonl
    train_data_path = config['data'].get('train_json') or config['data'].get('train_jsonl')
    val_data_path = config['data'].get('val_json') or config['data'].get('val_jsonl')
    
    train_dataset = InstructionVQADataset(
        train_data_path,
        config['data']['image_root'],
        processor,
        max_length=config['train']['max_seq_len']
    )
    
    val_dataset = InstructionVQADataset(
        val_data_path,
        config['data']['image_root'],
        processor,
        max_length=config['train']['max_seq_len']
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['train']['epochs'],
        per_device_train_batch_size=config['train']['train_bs'],
        per_device_eval_batch_size=config['train']['eval_bs'],
        gradient_accumulation_steps=config['train']['grad_accum'],
        learning_rate=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
        warmup_ratio=config['train']['warmup_ratio'],
        bf16=config['train']['bf16'],
        logging_steps=config['train']['logging_steps'],
        save_steps=config['train']['save_steps'],
        eval_steps=config['train']['eval_steps'],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=config['train']['gradient_checkpointing'],
        dataloader_num_workers=config['train'].get('dataloader_num_workers', 16),
        remove_unused_columns=False,
        report_to="none",
        seed=config['train']['seed'],
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Train
    print("\nStarting training...")
    print("="*80)
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model()
    print(f"✓ Model saved to {output_dir}")
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)

if __name__ == "__main__":
    main()

