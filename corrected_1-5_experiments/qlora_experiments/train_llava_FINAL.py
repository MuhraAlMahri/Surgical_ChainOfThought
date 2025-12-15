#!/usr/bin/env python3
"""
LLaVA-Med training - FINAL CORRECTED VERSION
Uses config.image_token_id (32000) directly from config

This is the critical fix after 40+ failed attempts:
- WRONG: image_token_id = vocab_size - 1 (31999)
- CORRECT: image_token_id = config.image_token_id (32000)

The model's get_placeholder_mask() function specifically checks for config.image_token_id.
Using any other value (even if it's in valid range) causes "tokens: 0" error.
"""

import os

# CRITICAL: Set cache locations BEFORE any other imports
SHARED_CACHE = os.environ.get("HF_HOME", "/l/users/muhra.almahri/.cache/hf_shared")
os.environ["HF_HOME"] = SHARED_CACHE
os.environ["TRANSFORMERS_CACHE"] = f"{SHARED_CACHE}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{SHARED_CACHE}/datasets"
os.environ["HF_HUB_CACHE"] = SHARED_CACHE
os.environ["TORCH_HOME"] = f"{SHARED_CACHE}/torch"

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

import torch
import sys

# Disable DeepSpeed at PyTorch level
# Note: cuDNN deterministic can cause CUDNN_STATUS_INTERNAL_ERROR on some GPUs
# Using benchmark mode for better performance and stability
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

# Prevent DeepSpeed from being imported or initialized
if 'deepspeed' in sys.modules:
    del sys.modules['deepspeed']
if 'accelerate' in sys.modules:
    # Re-import accelerate after setting env vars
    import importlib
    if 'accelerate' in sys.modules:
        importlib.reload(sys.modules['accelerate'])
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForVision2Seq,
    AutoConfig,
    AutoImageProcessor,
    CLIPImageProcessor,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
from tqdm import tqdm
from pathlib import Path
import time
import glob

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class LlavaProcessorWrapper:
    """Processor for LLaVA with correct image token replacement."""
    
    def __init__(self, image_processor, tokenizer, image_token_id):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        
        # Get how <image> is tokenized (should be 3 tokens: [523, 4075, 28767])
        self.image_token_seq = self.tokenizer.encode("<image>", add_special_tokens=False)
        
        logger.info("=" * 80)
        logger.info("✓ LlavaProcessorWrapper initialized")
        logger.info(f"  - Image token ID (from config): {image_token_id}")
        logger.info(f"  - <image> tokenizes as: {self.image_token_seq}")
        logger.info("=" * 80)
    
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
            
            # CRITICAL: Replace <image> token sequence with image_token_id
            if return_tensors == "pt":
                input_ids = text_outputs["input_ids"]
                
                for batch_idx in range(input_ids.size(0)):
                    ids = input_ids[batch_idx].tolist()
                    new_ids = []
                    i = 0
                    replacements = 0
                    
                    while i < len(ids):
                        # Check if we found the image token sequence [523, 4075, 28767]
                        if i + len(self.image_token_seq) <= len(ids) and ids[i:i+len(self.image_token_seq)] == self.image_token_seq:
                            new_ids.append(self.image_token_id)
                            i += len(self.image_token_seq)
                            replacements += 1
                        else:
                            new_ids.append(ids[i])
                            i += 1
                    
                    input_ids[batch_idx] = torch.tensor(new_ids, dtype=input_ids.dtype)
                
                # Verify replacement worked (only log first batch to avoid spam)
                if batch_idx == 0 and replacements > 0:
                    image_token_count = (input_ids == self.image_token_id).sum().item()
                    logger.debug(f"✓ Batch 0: Replaced {replacements} sequences → {image_token_count} tokens with ID {self.image_token_id}")
            
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
        
        logger.info(f"✓ Loaded {len(self.data)} samples from {dataset_name} {split}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different data formats
        if "image_filename" in item:
            if self.image_root:
                image_path = os.path.join(self.image_root, item["image_filename"])
            else:
                image_path = os.path.join(self.data_root, item["image_filename"])
        elif "image_name" in item:
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
            raise KeyError(f"No 'image', 'image_path', 'image_filename', or 'image_name' field found. Available keys: {list(item.keys())}")
        
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
        
        # Format conversation (LLaVA format)
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
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
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


def load_image_processor_with_retry(model_name, max_retries=3):
    """Load image processor with retry logic."""
    logger.info("Loading image processor...")
    
    # Try AutoImageProcessor first
    for attempt in range(max_retries):
        try:
            image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            logger.info(f"✓ Loaded image processor from {model_name}")
            return image_processor
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠ Attempt {attempt+1}: {e}, retrying in 5s...")
                time.sleep(5)
            else:
                logger.warning(f"⚠ Could not load from model: {e}")
    
    # Fallback to CLIP with 336x336 size
    logger.info("Trying CLIP processor with 336x336 size...")
    try:
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        image_processor.size = {"height": 336, "width": 336}
        image_processor.crop_size = {"height": 336, "width": 336}
        logger.info("✓ Loaded CLIP processor with 336x336 size")
        return image_processor
    except Exception as e:
        logger.error(f"✗ Failed to load image processor: {e}")
        raise RuntimeError("Could not load image processor")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    
    model_name = config_dict.get("model_name", "microsoft/llava-med-v1.5-mistral-7b")
    dataset_name = config_dict.get("dataset_name", "kvasir")
    data_root = config_dict.get("data_root")
    output_dir = config_dict.get("output_dir", "./outputs")
    image_root = config_dict.get("image_root", None)
    
    num_epochs = config_dict.get("num_epochs", 5)
    batch_size = config_dict.get("batch_size", 1)
    gradient_accumulation_steps = config_dict.get("gradient_accumulation_steps", 8)
    learning_rate = config_dict.get("learning_rate", 2e-4)
    warmup_steps = config_dict.get("warmup_steps", 100)
    max_length = config_dict.get("max_length", 2048)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("LLAVA-MED TRAINING - FINAL CORRECTED VERSION")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Cache: {SHARED_CACHE}")
    logger.info("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load config
    logger.info("Loading model configuration...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # CRITICAL FIX: Use image_token_id from config directly (32000)
    # DO NOT calculate as vocab_size - 1 (31999)
    # The model's get_placeholder_mask() function specifically checks for config.image_token_id
    image_token_id = getattr(config, "image_token_id", None)
    
    if image_token_id is None:
        logger.error("✗ CRITICAL: config.image_token_id not found!")
        logger.error("  This model may not be properly configured for LLaVA")
        raise ValueError("config.image_token_id is required")
    
    logger.info("=" * 80)
    logger.info("✓ CRITICAL FIX APPLIED: Using config.image_token_id directly")
    logger.info(f"  - Image token ID: {image_token_id}")
    logger.info(f"  - Config vocab_size: {config.vocab_size}")
    logger.info(f"  - NOT using vocab_size - 1 ({config.vocab_size - 1})")
    logger.info("=" * 80)
    
    # Wait for incomplete downloads
    logger.info("Checking for incomplete downloads...")
    cache_dir = os.environ.get("HF_HUB_CACHE", SHARED_CACHE)
    incomplete_files = list(glob.glob(f"{cache_dir}/**/*.incomplete", recursive=True))
    if incomplete_files:
        logger.info(f"Found {len(incomplete_files)} incomplete download(s), waiting...")
        max_wait = 600
        wait_time = 0
        while incomplete_files and wait_time < max_wait:
            time.sleep(5)
            wait_time += 5
            incomplete_files = list(glob.glob(f"{cache_dir}/**/*.incomplete", recursive=True))
        if incomplete_files:
            logger.warning(f"⚠️  Still have {len(incomplete_files)} incomplete files after {max_wait}s")
        else:
            logger.info("✓ All downloads completed")
    
    # Load model with retry - explicitly disable DeepSpeed
    logger.info("Loading model...")
    logger.info("  Ensuring DeepSpeed is disabled...")
    
    # Double-check DeepSpeed is disabled
    if 'deepspeed' in sys.modules:
        logger.warning("⚠️  DeepSpeed module found, removing...")
        del sys.modules['deepspeed']
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use device_map="cuda" instead of "auto" to avoid DeepSpeed auto-detection
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map="cuda",  # Changed from "auto" to avoid DeepSpeed
                trust_remote_code=True,
            )
            logger.info("✓ Model loaded successfully (DeepSpeed disabled)")
            break
        except Exception as e:
            if "InvalidHeaderDeserialization" in str(e) or "Disk quota exceeded" in str(e):
                logger.warning(f"⚠️  Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    model_cache_path = os.path.join(SHARED_CACHE, "models", model_name.replace("/", "--"))
                    if os.path.exists(model_cache_path):
                        import shutil
                        shutil.rmtree(model_cache_path)
                        logger.info(f"  Cleared potentially corrupted cache: {model_cache_path}")
                    time.sleep(30)
                else:
                    raise
            else:
                raise
    
    # Verify model's image_token_id matches
    model_image_token_id = getattr(model.config, "image_token_id", None)
    logger.info(f"✓ Model's image_token_id: {model_image_token_id}")
    if model_image_token_id != image_token_id:
        logger.warning(f"⚠️  Warning: Config image_token_id ({image_token_id}) != Model image_token_id ({model_image_token_id})")
        logger.warning(f"  Using model's value: {model_image_token_id}")
        image_token_id = model_image_token_id
    
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
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("✓ Tokenizer loaded")
    
    # Load image processor
    image_processor = load_image_processor_with_retry(model_name)
    
    # Create processor wrapper
    processor = LlavaProcessorWrapper(image_processor, tokenizer, image_token_id)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = SurgicalVQADataset(data_root, dataset_name, "train", processor, max_length, image_root)
    eval_dataset = SurgicalVQADataset(data_root, dataset_name, "val", processor, max_length, image_root)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=4)
    
    # CRITICAL VERIFICATION: Test first batch to ensure image tokens are present
    logger.info("=" * 80)
    logger.info("CRITICAL VERIFICATION: Testing image token replacement")
    logger.info("=" * 80)
    test_batch = next(iter(train_loader))
    test_image_tokens = (test_batch["input_ids"] == image_token_id).sum().item()
    logger.info(f"✓ First batch contains {test_image_tokens} image tokens with ID {image_token_id}")
    
    if test_image_tokens == 0:
        logger.error("✗ CRITICAL ERROR: No image tokens found in first batch!")
        logger.error("  This means image token replacement is NOT working correctly")
        logger.error("  Training will fail with 'tokens: 0' error")
        raise RuntimeError("Image token replacement verification failed")
    
    logger.info("✓ VERIFICATION PASSED: Image tokens are correctly present")
    logger.info("=" * 80)
    
    # Optimizer
    logger.info("Setting up optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    best_eval_loss = float('inf')
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, 
                                epoch + 1, gradient_accumulation_steps)
        logger.info(f"Train loss: {train_loss:.4f}")
        
        eval_loss = evaluate(model, eval_loader, device)
        logger.info(f"Eval loss: {eval_loss:.4f}")
        
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_path = os.path.join(output_dir, "best_model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"✓ Saved best model (eval_loss: {eval_loss:.4f})")
        
        checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    
    logger.info("=" * 80)
    logger.info(f"TRAINING COMPLETED - Best loss: {best_eval_loss:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

