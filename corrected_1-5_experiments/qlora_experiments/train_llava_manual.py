#!/usr/bin/env python3
"""
LLaVA-Med training with manual training loop (no Trainer = no DeepSpeed issues).
"""

import os

# CRITICAL: Set shared cache BEFORE any other imports
# This prevents disk quota issues by reusing a single cache location
SHARED_CACHE = os.environ.get("HF_HOME", "/l/users/muhra.almahri/.cache/hf_shared")
os.environ["HF_HOME"] = SHARED_CACHE
os.environ["TRANSFORMERS_CACHE"] = f"{SHARED_CACHE}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{SHARED_CACHE}/datasets"
os.environ["HF_HUB_CACHE"] = SHARED_CACHE
os.environ["TORCH_HOME"] = f"{SHARED_CACHE}/torch"

import torch
import sys
import logging
import time
import glob

# Disable DeepSpeed at PyTorch level
# Note: cuDNN deterministic can cause CUDNN_STATUS_INTERNAL_ERROR on some GPUs
# Using benchmark mode for better performance and stability
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

# Prevent DeepSpeed from being imported or initialized
if 'deepspeed' in sys.modules:
    del sys.modules['deepspeed']
from transformers import (
    AutoTokenizer,
    AutoModelForVision2Seq,
    AutoConfig,
    AutoImageProcessor,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
from tqdm import tqdm
from pathlib import Path

# CRITICAL: Aggressively disable DeepSpeed - must be before any imports
# Remove all DeepSpeed-related environment variables
for key in list(os.environ.keys()):
    if 'DEEPSPEED' in key.upper() or 'DS_' in key.upper():
        del os.environ[key]

# Set explicit environment variables to disable DeepSpeed
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
os.environ["DEEPSPEED_DISABLED"] = "true"
os.environ["ACCELERATE_USE_CUDA"] = "true"
os.environ["ACCELERATE_USE_CPU"] = "false"
os.environ["ACCELERATE_USE_XPU"] = "false"
os.environ["CUDNN_DETERMINISTIC"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class LlavaProcessorWrapper:
    """Simple processor for LLaVA with correct image token replacement."""
    
    def __init__(self, image_processor, tokenizer, image_token_id):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        logger.info(f"Image token ID: {image_token_id}")
    
    def __call__(self, text=None, images=None, return_tensors=None, 
                 padding=False, truncation=False, max_length=None):
        """Process text and images."""
        
        # Process images
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            image_outputs = self.image_processor(images, return_tensors=return_tensors)
            pixel_values = image_outputs["pixel_values"]
        else:
            pixel_values = None
        
        # Process text
        if text is not None:
            text_outputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
            )
            
            # CRITICAL: Replace <image> tokens with image_token_id
            if return_tensors == "pt":
                input_ids = text_outputs["input_ids"]
                attention_mask = text_outputs.get("attention_mask", None)
                
                # Try to find <image> token - check if it's a single token or multi-token
                image_token_text = "<image>"
                image_token_ids = self.tokenizer.encode(image_token_text, add_special_tokens=False)
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                original_length = input_ids.size(1)
                
                logger.info(f"Image token '{image_token_text}' encodes to: {image_token_ids}")
                logger.info(f"Image token ID to use: {self.image_token_id}")
                
                for batch_idx in range(input_ids.size(0)):
                    ids = input_ids[batch_idx].tolist()
                    new_ids = []
                    i = 0
                    replaced_count = 0
                    
                    while i < len(ids):
                        # Check if we found the image token sequence
                        if i + len(image_token_ids) <= len(ids) and ids[i:i+len(image_token_ids)] == image_token_ids:
                            # Replace with single image_token_id
                            new_ids.append(self.image_token_id)
                            replaced_count += 1
                            i += len(image_token_ids)
                            logger.info(f"  ✓ Found and replaced <image> token sequence at position {i - len(image_token_ids)}")
                        else:
                            new_ids.append(ids[i])
                            i += 1
                    
                    if replaced_count == 0:
                        logger.warning(f"⚠️  No <image> tokens found in batch {batch_idx}! Text might not contain <image> token.")
                        logger.warning(f"  Looking for sequence {image_token_ids} in first 30 tokens: {ids[:30]}")
                        # Try to find where image should be (after "USER:") and insert it
                        user_ids = self.tokenizer.encode("USER:", add_special_tokens=False)
                        logger.info(f"  Looking for 'USER:' sequence: {user_ids}")
                        for j in range(len(new_ids) - len(user_ids)):
                            if new_ids[j:j+len(user_ids)] == user_ids:
                                # Insert image token after "USER:"
                                new_ids.insert(j + len(user_ids), self.image_token_id)
                                replaced_count += 1
                                logger.info(f"  ✓ Inserted image token after 'USER:' at position {j + len(user_ids)}")
                                break
                        if replaced_count == 0:
                            logger.error(f"  ✗ Could not find 'USER:' or <image> tokens! This will cause an error.")
                    
                    # Handle length changes: pad or truncate to maintain original length
                    if len(new_ids) < original_length:
                        # Pad to original length
                        new_ids.extend([pad_token_id] * (original_length - len(new_ids)))
                    elif len(new_ids) > original_length:
                        # Truncate to original length
                        new_ids = new_ids[:original_length]
                    
                    input_ids[batch_idx] = torch.tensor(new_ids, dtype=input_ids.dtype)
                    
                    if replaced_count > 0:
                        logger.info(f"  ✓ Successfully processed {replaced_count} image token(s) in batch {batch_idx}")
                    else:
                        logger.error(f"  ✗ FAILED to process image tokens in batch {batch_idx} - this will cause an error!")
                    
                    # Update attention mask if present
                    if attention_mask is not None:
                        # Keep attention mask the same (1 for real tokens, 0 for padding)
                        # The mask should already be correct since we maintain the same length
                        pass
            
            if pixel_values is not None:
                return {**text_outputs, "pixel_values": pixel_values}
            return text_outputs
        
        return {"pixel_values": pixel_values} if pixel_values is not None else {}


class SurgicalVQADataset(Dataset):
    """Dataset for surgical VQA."""
    
    def __init__(self, data_root: str, dataset_name: str, split: str, 
                 processor, max_length: int = 2048, image_root: str = None):
        self.data_root = data_root
        self.processor = processor
        self.max_length = max_length
        self.image_root = image_root
        
        # Load data - support both JSON and JSONL formats
        json_path = os.path.join(data_root, f"{split}.json")
        jsonl_path = os.path.join(data_root, f"{split}.jsonl")
        
        if os.path.exists(json_path):
            with open(json_path) as f:
                self.data = json.load(f)
        elif os.path.exists(jsonl_path):
            self.data = []
            with open(jsonl_path) as f:
                for line in f:
                    self.data.append(json.loads(line))
        else:
            raise FileNotFoundError(f"Neither {json_path} nor {jsonl_path} found")
        
        logger.info(f"Loaded {len(self.data)} samples from {dataset_name} {split}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different data formats - support image, image_path, image_filename, and image_name
        if "image_filename" in item:
            # Most common format for our datasets
            if self.image_root:
                image_path = os.path.join(self.image_root, item["image_filename"])
            else:
                image_path = os.path.join(self.data_root, item["image_filename"])
        elif "image_name" in item:
            # EndoVis2018 format
            if self.image_root:
                image_path = os.path.join(self.image_root, item["image_name"])
            else:
                image_path = os.path.join(self.data_root, item["image_name"])
        elif "image" in item:
            if self.image_root:
                image_path = os.path.join(self.image_root, item["image"])
            else:
                image_path = os.path.join(self.data_root, item["image"])
        elif "image_path" in item:
            image_path = item["image_path"]
        else:
            raise KeyError(f"No 'image', 'image_path', 'image_filename', or 'image_name' field found in data. Available keys: {list(item.keys())}")
        
        # Load image
        if not os.path.exists(image_path):
            # Try relative to data_root as fallback
            if "image_filename" in item:
                image_path = os.path.join(self.data_root, item["image_filename"])
            elif "image_name" in item:
                image_path = os.path.join(self.data_root, item["image_name"])
            elif "image" in item:
                image_path = os.path.join(self.data_root, item["image"])
        
        image = Image.open(image_path).convert("RGB")
        
        # Format conversation (LLaVA format) - support both 'question' and 'instruction' fields
        question = item.get("instruction") or item.get("question", "")
        answer = item.get("answer", "")
        text = f"USER: <image>\n{question}\nASSISTANT: {answer}"
        
        # Process
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        
        # Squeeze and create labels
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()
        
        # Mask everything before ASSISTANT:
        assistant_ids = self.processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
        input_ids_list = inputs["input_ids"].tolist()
        
        for i in range(len(input_ids_list) - len(assistant_ids)):
            if input_ids_list[i:i+len(assistant_ids)] == assistant_ids:
                labels[:i+len(assistant_ids)] = -100
                break
        
        # Mask padding
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        
        return inputs


def collate_fn(batch):
    """Collate function."""
    pixel_values = torch.stack([item.pop("pixel_values") for item in batch])
    batch_dict = {key: torch.stack([item[key] for item in batch]) for key in batch[0].keys()}
    batch_dict["pixel_values"] = pixel_values
    return batch_dict


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, gradient_accumulation_steps=8):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps  # Scale loss
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Log
        total_loss += loss.item() * gradient_accumulation_steps  # Unscale for logging
        num_batches += 1
        
        if batch_idx % 10 == 0:
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}", 
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
    
    return total_loss / num_batches


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    
    # Configuration
    model_name = config_dict.get("model_name", "microsoft/llava-med-v1.5-mistral-7b")
    dataset_name = config_dict.get("dataset_name", "kvasir")
    
    # Handle different config formats
    data_config = config_dict.get("data", {})
    train_data_path = data_config.get("train_jsonl") or config_dict.get("train_data_path")
    val_data_path = data_config.get("val_jsonl") or config_dict.get("val_data_path")
    image_root = data_config.get("image_root") or config_dict.get("image_root", "")
    
    # Extract data root from paths
    if train_data_path:
        data_root = os.path.dirname(train_data_path)
    else:
        data_root = os.getenv("DATA_ROOT", "/path/to/data")
    
    output_dir = config_dict.get("output_dir", f"./outputs/llava_med_{dataset_name}_manual")
    
    # Hyperparameters
    train_config = config_dict.get("train", {})
    num_epochs = train_config.get("epochs") or config_dict.get("num_epochs", 5)
    batch_size = train_config.get("train_bs") or config_dict.get("train_batch_size", 1)
    gradient_accumulation_steps = train_config.get("grad_accum") or config_dict.get("gradient_accumulation_steps", 8)
    learning_rate = train_config.get("lr") or config_dict.get("learning_rate", 2e-4)
    warmup_ratio = train_config.get("warmup_ratio", 0.03)
    max_length = train_config.get("max_seq_len") or config_dict.get("max_seq_length", 2048)
    
    # Calculate warmup steps (will be calculated later based on dataset size)
    warmup_steps = config_dict.get("warmup_steps", 100)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("MANUAL TRAINING LOOP (NO DEEPSPEED)")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load model
    logger.info("Loading model...")
    
    # Handle llava_mistral model type issue
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except ValueError as e:
        if "llava_mistral" in str(e):
            logger.info("⚠️  llava_mistral not recognized, loading config manually...")
            from huggingface_hub import hf_hub_download
            import json
            from transformers import LlavaConfig
            
            # Download config file directly
            hf_token = os.environ.get('HF_TOKEN')
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
                logger.info("  ⚠️  Changed model_type from 'llava_mistral' to 'llava' for compatibility")
            
            # Create config from dict using LlavaConfig
            config = LlavaConfig.from_dict(config_dict)
            if hasattr(config, 'auto_map'):
                config.trust_remote_code = True
        else:
            raise
    
    # CRITICAL: Get initial image_token_id from config (will be corrected after model load)
    image_token_id = getattr(config, "image_token_id", config.vocab_size)
    logger.info(f"Initial image token ID from config: {image_token_id}")
    
    # Wait for model files to finish downloading (fix incomplete download issue)
    logger.info("Checking for incomplete downloads...")
    cache_dir = os.environ.get("HF_HUB_CACHE", SHARED_CACHE)
    incomplete_files = list(glob.glob(f"{cache_dir}/**/*.incomplete", recursive=True))
    if incomplete_files:
        logger.info(f"Found {len(incomplete_files)} incomplete download(s), waiting for completion...")
        max_wait = 600  # 10 minutes max wait
        wait_time = 0
        while incomplete_files and wait_time < max_wait:
            time.sleep(5)
            wait_time += 5
            incomplete_files = list(glob.glob(f"{cache_dir}/**/*.incomplete", recursive=True))
            if incomplete_files:
                logger.info(f"  Still waiting... ({wait_time}s elapsed, {len(incomplete_files)} incomplete)")
        if incomplete_files:
            logger.warning(f"⚠️  Still have {len(incomplete_files)} incomplete files after {max_wait}s, proceeding anyway...")
        else:
            logger.info("✓ All downloads completed")
    
    # Load model with the config (with retry for incomplete downloads)
    max_retries = 3
    retry_delay = 10
    model = None
    
    for attempt in range(max_retries):
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="cuda",  # Changed from "auto" to avoid DeepSpeed auto-detection
                trust_remote_code=True,
            )
            break  # Success, exit retry loop
        except Exception as e:
            error_str = str(e)
            if "InvalidHeaderDeserialization" in error_str or "incomplete" in error_str.lower():
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️  Model load failed (attempt {attempt + 1}/{max_retries}): {error_str}")
                    logger.info(f"  Waiting {retry_delay}s for downloads to complete...")
                    time.sleep(retry_delay)
                    # Clear any incomplete files
                    incomplete_files = list(glob.glob(f"{cache_dir}/**/*.incomplete", recursive=True))
                    if incomplete_files:
                        logger.info(f"  Found {len(incomplete_files)} incomplete files, waiting longer...")
                        time.sleep(retry_delay * 2)
                    continue
                else:
                    logger.error(f"✗ Model load failed after {max_retries} attempts: {error_str}")
                    raise
            else:
                # Different error, try fallback
                logger.info(f"⚠️  AutoModelForVision2Seq failed: {e}")
                logger.info("Trying LlavaForConditionalGeneration...")
                from transformers import LlavaForConditionalGeneration
                try:
                    model = LlavaForConditionalGeneration.from_pretrained(
                        model_name,
                        config=config,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    break  # Success
                except Exception as e2:
                    if attempt < max_retries - 1 and ("InvalidHeaderDeserialization" in str(e2) or "incomplete" in str(e2).lower()):
                        logger.warning(f"⚠️  Retry {attempt + 1}/{max_retries} needed...")
                        time.sleep(retry_delay)
                        continue
                    raise e2
    
    # CRITICAL FIX: Use image_token_id from config directly (32000)
    # DO NOT calculate as vocab_size - 1 (31999)
    # The model's get_placeholder_mask() function specifically checks for config.image_token_id
    # This is the fix after 40+ failed attempts - using the wrong token ID causes "tokens: 0" error
    image_token_id = getattr(model.config, "image_token_id", None)
    
    if image_token_id is None:
        logger.error("✗ CRITICAL: model.config.image_token_id not found!")
        logger.error("  This model may not be properly configured for LLaVA")
        raise ValueError("model.config.image_token_id is required")
    
    logger.info("=" * 80)
    logger.info("✓ CRITICAL FIX: Using config.image_token_id directly")
    logger.info(f"  - Image token ID: {image_token_id}")
    logger.info(f"  - Config vocab_size: {model.config.vocab_size}")
    logger.info(f"  - NOT using vocab_size - 1 ({model.config.vocab_size - 1})")
    logger.info("=" * 80)
    
    # Apply LoRA
    logger.info("Applying LoRA...")
    lora_r = config_dict.get("lora_r", 8)
    lora_alpha = config_dict.get("lora_alpha", 16)
    lora_target_modules = config_dict.get("lora_target_modules", 
                                          ["q_proj", "v_proj", "k_proj", "o_proj", 
                                           "gate_proj", "up_proj", "down_proj"])
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load tokenizer and processor
    logger.info("Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try to load image processor - handle cases where it's not available
    try:
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        logger.info("✓ Loaded image processor using AutoImageProcessor")
    except (OSError, Exception) as e:
        logger.info(f"⚠️  AutoImageProcessor failed: {e}")
        logger.info("Trying CLIPImageProcessor from llava-v1.5-7b...")
        from transformers import CLIPImageProcessor
        try:
            # Try loading from a standard LLaVA model that has the processor
            image_processor = CLIPImageProcessor.from_pretrained(
                "liuhaotian/llava-v1.5-7b", trust_remote_code=True
            )
            logger.info("✓ Loaded CLIPImageProcessor from llava-v1.5-7b")
        except Exception as e2:
            logger.info(f"⚠️  CLIPImageProcessor from llava-v1.5-7b failed: {e2}")
            logger.info("Trying CLIPImageProcessor from model repo directly...")
            try:
                image_processor = CLIPImageProcessor.from_pretrained(
                    model_name, trust_remote_code=True
                )
                logger.info("✓ Loaded CLIPImageProcessor from model repo")
            except Exception as e3:
                logger.info(f"⚠️  CLIPImageProcessor from model repo failed: {e3}")
                logger.info("Creating CLIPImageProcessor with default settings...")
                try:
                    # Final fallback: create CLIPImageProcessor with default settings
                    # LLaVA-Med uses 336x336 image size (not 224x224)
                    from transformers import CLIPImageProcessor
                    image_processor = CLIPImageProcessor.from_pretrained(
                        "openai/clip-vit-large-patch14",
                        trust_remote_code=True
                    )
                    # Override image size to 336x336 for LLaVA-Med compatibility
                    image_processor.size = {"height": 336, "width": 336}
                    image_processor.crop_size = {"height": 336, "width": 336}
                    logger.info("✓ Created CLIPImageProcessor with 336x336 size for LLaVA-Med compatibility")
                except Exception as e4:
                    logger.error(f"✗ All image processor loading methods failed. Last error: {e4}")
                    raise
    
    processor = LlavaProcessorWrapper(image_processor, tokenizer, image_token_id)
    
    # Load datasets
    logger.info("Loading datasets...")
    
    # Helper function to extract split name from file path
    def get_split_name(file_path):
        """Extract split name from file path, removing .json or .jsonl extension."""
        if not file_path:
            return None
        basename = os.path.basename(file_path)
        # Remove extension - handle both .json and .jsonl (check .jsonl first to avoid partial removal)
        if basename.endswith('.jsonl'):
            return basename[:-6]  # Remove .jsonl
        elif basename.endswith('.json'):
            return basename[:-5]  # Remove .json
        return basename
    
    # Use full paths if provided, otherwise construct from data_root
    if train_data_path and os.path.isabs(train_data_path):
        # Full path provided
        train_split = get_split_name(train_data_path)
        train_dataset = SurgicalVQADataset(os.path.dirname(train_data_path), dataset_name, 
                                          train_split, processor, max_length, image_root)
    else:
        train_split = get_split_name(train_data_path) if train_data_path else "train"
        train_dataset = SurgicalVQADataset(data_root, dataset_name, train_split, processor, max_length, image_root)
    
    if val_data_path and os.path.isabs(val_data_path):
        # Full path provided
        val_split = get_split_name(val_data_path)
        eval_dataset = SurgicalVQADataset(os.path.dirname(val_data_path), dataset_name,
                                         val_split, processor, max_length, image_root)
    else:
        val_split = get_split_name(val_data_path) if val_data_path else "val"
        eval_dataset = SurgicalVQADataset(data_root, dataset_name, val_split, processor, max_length, image_root)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    # Optimizer and scheduler
    logger.info("Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    # Calculate warmup steps from ratio if not provided
    if warmup_steps == 100 and warmup_ratio > 0:
        warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    best_eval_loss = float('inf')
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, 
            epoch + 1, gradient_accumulation_steps
        )
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        eval_loss = evaluate(model, eval_loader, device)
        logger.info(f"Eval loss: {eval_loss:.4f}")
        
        # Save best model
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_path = os.path.join(output_dir, "best_model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"✓ Saved best model (eval_loss: {eval_loss:.4f})")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best eval loss: {best_eval_loss:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

