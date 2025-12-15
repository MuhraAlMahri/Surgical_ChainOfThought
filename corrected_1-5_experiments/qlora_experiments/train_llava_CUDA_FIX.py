#!/usr/bin/env python3
"""
LLaVA-Med Training - CUDA Assert Fix

Issue: Image tokens (32000) being passed to vision encoder
Fix: Vision encoder only processes pixel_values, not input_ids with image tokens
"""

import os
import torch
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

# Cache configuration
SHARED_CACHE = "/l/users/muhra.almahri/.cache/hf_shared"
os.environ["HF_HOME"] = SHARED_CACHE
os.environ["TRANSFORMERS_CACHE"] = f"{SHARED_CACHE}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{SHARED_CACHE}/datasets"
os.environ["TORCH_HOME"] = f"{SHARED_CACHE}/torch"

# Disable DeepSpeed
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
os.environ["DEEPSPEED_DISABLED"] = "true"

# Enable synchronous CUDA for better error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)




class LlavaProcessorWrapper:
    """Processor for LLaVA with correct image token replacement."""
    
    def __init__(self, image_processor, tokenizer, image_token_id, replace_image_tokens=True):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.replace_image_tokens = replace_image_tokens
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
            
            # Replace <image> tokens with image_token_id (if enabled)
            if return_tensors == "pt" and self.replace_image_tokens:
                input_ids = text_outputs["input_ids"]
                attention_mask = text_outputs.get("attention_mask")
                
                for batch_idx in range(input_ids.size(0)):
                    ids = input_ids[batch_idx].tolist()
                    original_length = len(ids)
                    new_ids = []
                    i = 0
                    replacements = 0
                    
                    while i < len(ids):
                        if ids[i:i+len(self.image_token_seq)] == self.image_token_seq:
                            new_ids.append(self.image_token_id)
                            i += len(self.image_token_seq)
                            replacements += 1
                        else:
                            new_ids.append(ids[i])
                            i += 1
                    
                    # Pad or truncate to original length
                    pad_token_id = self.tokenizer.pad_token_id
                    if len(new_ids) < original_length:
                        new_ids.extend([pad_token_id] * (original_length - len(new_ids)))
                    elif len(new_ids) > original_length:
                        new_ids = new_ids[:original_length]
                    
                    input_ids[batch_idx] = torch.tensor(new_ids, dtype=input_ids.dtype)
                    
                    # Update attention mask if present
                    if attention_mask is not None:
                        new_mask = [1 if token_id != pad_token_id else 0 for token_id in new_ids]
                        attention_mask[batch_idx] = torch.tensor(new_mask, dtype=attention_mask.dtype)
                
                # Verify and log
                image_token_count = (input_ids == self.image_token_id).sum().item()
                if image_token_count > 0:
                    logger.debug(f"✓ Replaced {replacements} sequences → {image_token_count} tokens (ID: {self.image_token_id})")
            
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
        
        # Try JSONL first, then JSON
        jsonl_path = os.path.join(data_root, f"{split}_CATEGORY_BASED.jsonl")
        json_path = os.path.join(data_root, f"{split}_CATEGORY_BASED.json")
        
        if os.path.exists(jsonl_path):
            logger.info(f"Loading from JSONL: {jsonl_path}")
            self.data = []
            with open(jsonl_path) as f:
                for line in f:
                    self.data.append(json.loads(line))
        elif os.path.exists(json_path):
            logger.info(f"Loading from JSON: {json_path}")
            with open(json_path) as f:
                self.data = json.load(f)
        else:
            raise FileNotFoundError(f"Neither {jsonl_path} nor {json_path} found")
        
        logger.info(f"✓ Loaded {len(self.data)} samples from {dataset_name} {split}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different data formats - support image, image_path, image_filename, and image_name
        image_filename_field = item.get("image_filename") or item.get("image_name") or item.get("image") or item.get("image_path")
        if not image_filename_field:
            raise KeyError(f"No 'image', 'image_path', 'image_filename', or 'image_name' field found in data. Available keys: {list(item.keys())}")
        
        image_path = os.path.join(self.data_root, image_filename_field)
        
        # Load image
        if not os.path.exists(image_path):
            # Fallback: try relative to data_root if image_root was specified but path not found
            if self.image_root and os.path.exists(os.path.join(self.image_root, image_filename_field)):
                image_path = os.path.join(self.image_root, image_filename_field)
            else:
                raise FileNotFoundError(f"Image not found at {image_path} or {os.path.join(self.data_root, image_filename_field)}")
        
        image = Image.open(image_path).convert("RGB")
        
        # Format conversation (LLaVA format) - support both 'question' and 'instruction' fields
        question = item.get("instruction") or item.get("question", "")
        answer = item.get("answer", "")
        text = f"USER: <image>\n{question}\nASSISTANT: {answer}"
        
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        
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
        
        # Mask image tokens (image_token_id may equal vocab_size, which is invalid)
        image_token_id = self.processor.image_token_id
        labels[labels == image_token_id] = -100
        
        # CRITICAL FIX: Validate labels
        # Ensure all labels are either -100 or valid token IDs
        vocab_size = self.processor.tokenizer.vocab_size
        valid_mask = (labels == -100) | ((labels >= 0) & (labels < vocab_size))
        if not valid_mask.all():
            logger.error(f"Invalid labels found in sample {idx}!")
            logger.error(f"Labels range: {labels.min()} to {labels.max()}")
            logger.error(f"Vocab size: {vocab_size}")
            logger.error(f"Image token ID: {image_token_id}")
            raise ValueError(f"Invalid label values in sample {idx}")
        
        inputs["labels"] = labels
        
        return inputs




def collate_fn(batch):
    """Collate function."""
    pixel_values = torch.stack([item.pop("pixel_values") for item in batch])
    batch_dict = {key: torch.stack([item[key] for item in batch]) for key in batch[0].keys()}
    batch_dict["pixel_values"] = pixel_values
    return batch_dict




def validate_batch(batch, image_token_id, vocab_size):
    """Validate batch before forward pass to catch issues early."""
    
    # Check input_ids
    input_ids = batch["input_ids"]
    if input_ids.min() < 0:
        raise ValueError(f"Negative token ID found: {input_ids.min()}")
    # Allow image_token_id even if it equals vocab_size (special token)
    invalid_mask = (input_ids >= vocab_size) & (input_ids != image_token_id)
    if invalid_mask.any():
        invalid_ids = input_ids[invalid_mask].unique().tolist()
        raise ValueError(f"Token IDs {invalid_ids} exceed vocab_size {vocab_size} (and are not image_token_id {image_token_id})")
    
    # Check labels
    labels = batch["labels"]
    valid_labels = (labels == -100) | ((labels >= 0) & (labels < vocab_size))
    if not valid_labels.all():
        invalid_count = (~valid_labels).sum().item()
        raise ValueError(f"Found {invalid_count} invalid label values")
    
    # Check image tokens
    image_token_count = (input_ids == image_token_id).sum().item()
    
    # Check pixel values
    pixel_values = batch["pixel_values"]
    if pixel_values.isnan().any():
        raise ValueError("NaN values in pixel_values")
    if pixel_values.isinf().any():
        raise ValueError("Inf values in pixel_values")
    
    return image_token_count




def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, 
                gradient_accumulation_steps, image_token_id, vocab_size):
    """Train for one epoch with validation."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Validate batch BEFORE moving to device
            image_token_count = validate_batch(batch, image_token_id, vocab_size)
            
            if batch_idx == 0:
                logger.info(f"✓ First batch validation passed")
                logger.info(f"  - Image tokens in batch: {image_token_count}")
                logger.info(f"  - Input IDs range: {batch['input_ids'].min()} to {batch['input_ids'].max()}")
                logger.info(f"  - Pixel values shape: {batch['pixel_values'].shape}")
            
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            # Try passing image_sizes if model expects it
            try:
                outputs = model(**batch)
            except ValueError as e:
                if "Image features and image tokens do not match" in str(e):
                    # Try without replacing image tokens - let model handle it
                    logger.warning("⚠️  Image token mismatch detected. Trying alternative approach...")
                    # This might require not replacing <image> tokens manually
                    raise RuntimeError(
                        "Image token mismatch. The model expects a different number of image tokens. "
                        "This may require using the native LlavaProcessor or adjusting image token handling."
                    ) from e
                raise
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass
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
                
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            logger.error(f"Batch keys: {batch.keys()}")
            if 'input_ids' in batch:
                logger.error(f"Input IDs shape: {batch['input_ids'].shape}")
                logger.error(f"Input IDs range: {batch['input_ids'].min()} to {batch['input_ids'].max()}")
            if 'labels' in batch:
                logger.error(f"Labels shape: {batch['labels'].shape}")
                logger.error(f"Labels range: {batch['labels'].min()} to {batch['labels'].max()}")
            raise
    
    return total_loss / num_batches




def evaluate(model, dataloader, device, image_token_id, vocab_size):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Validate batch
            validate_batch(batch, image_token_id, vocab_size)
            
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
    
    for attempt in range(max_retries):
        try:
            image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            # Ensure size is 336x336 for LLaVA-Med
            if hasattr(image_processor, 'size'):
                image_processor.size = {"height": 336, "width": 336}
            if hasattr(image_processor, 'crop_size'):
                image_processor.crop_size = {"height": 336, "width": 336}
            logger.info(f"✓ Loaded image processor from {model_name} (size: 336x336)")
            return image_processor
        except Exception as e:
            if "safetensors" in str(e).lower() and attempt < max_retries - 1:
                logger.warning(f"⚠ Attempt {attempt+1}: Incomplete download, retrying in 5s...")
                time.sleep(5)
                continue
            logger.warning(f"⚠ Could not load from model: {e}")
            break
    
    # Fallback to CLIP
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading CLIP processor (attempt {attempt+1})...")
            image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
            # CRITICAL: Set size to 336x336 for LLaVA-Med
            image_processor.size = {"height": 336, "width": 336}
            image_processor.crop_size = {"height": 336, "width": 336}
            logger.info(f"✓ Loaded CLIP processor (resized to 336x336)")
            return image_processor
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠ Attempt {attempt+1} failed, retrying in 5s...")
                time.sleep(5)
            else:
                raise RuntimeError("Could not load image processor")




def main():
    model_name = "microsoft/llava-med-v1.5-mistral-7b"
    dataset_name = os.getenv("DATASET_NAME", "kvasir")
    data_root = os.getenv("DATA_ROOT", "/l/users/muhra.abdalsamad/corrected_1-5_experiments/datasets/kvasir_vqa")
    output_dir = f"./outputs/llava_med_{dataset_name}_cuda_fixed"
    
    num_epochs = 3
    batch_size = 1
    gradient_accumulation_steps = 8
    learning_rate = 2e-4
    warmup_steps = 100
    max_length = 2048
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("LLAVA-MED TRAINING - CUDA ASSERT FIX")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load config
    logger.info("Loading configuration...")
    
    # Handle llava_mistral model type issue
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except ValueError as e:
        if "llava_mistral" in str(e):
            logger.info("⚠️  llava_mistral not recognized, loading config manually...")
            from huggingface_hub import hf_hub_download
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
    
    # Get image_token_id from config - CRITICAL: Use config value directly (32000)
    # The model's get_placeholder_mask() function specifically checks for config.image_token_id
    vocab_size = config.vocab_size
    image_token_id = getattr(config, 'image_token_id', None)
    
    if image_token_id is None:
        logger.warning(f"⚠️  config.image_token_id not found, using vocab_size - 1 ({vocab_size - 1})")
        image_token_id = vocab_size - 1
    elif image_token_id >= vocab_size:
        logger.warning(f"⚠️  image_token_id ({image_token_id}) >= vocab_size ({vocab_size})")
        logger.info("  Will resize embedding layer to accommodate image_token_id")
    
    logger.info("=" * 80)
    logger.info(f"✓ IMAGE TOKEN ID: {image_token_id}")
    logger.info(f"✓ VOCAB SIZE: {vocab_size}")
    logger.info("=" * 80)
    
    # Load model
    logger.info("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    logger.info("✓ Model loaded")
    
    # CRITICAL: Resize embedding layer if image_token_id >= vocab_size
    if image_token_id >= vocab_size:
        logger.info("=" * 80)
        logger.info("RESIZING EMBEDDING LAYER TO ACCOMMODATE IMAGE TOKEN")
        logger.info("=" * 80)
        
        # Get current embedding size
        embedding_layer = model.get_input_embeddings()
        old_size = embedding_layer.num_embeddings
        new_size = image_token_id + 1
        
        logger.info(f"  Current embedding size: {old_size}")
        logger.info(f"  Resizing to: {new_size} (to accommodate token {image_token_id})")
        
        # Use model's resize_token_embeddings method (not embedding layer's)
        if hasattr(model, 'resize_token_embeddings'):
            model.resize_token_embeddings(new_size)
            logger.info(f"  ✓ Embedding layer resized from {old_size} to {new_size}")
        else:
            # Manual resize: create new embedding with extended size
            logger.info("  Using manual embedding resize...")
            import torch.nn as nn
            
            # Get embedding dimension
            embedding_dim = embedding_layer.embedding_dim
            
            # Create new embedding layer with extended size (on same device)
            device = embedding_layer.weight.device
            new_embedding = nn.Embedding(new_size, embedding_dim, dtype=embedding_layer.weight.dtype)
            new_embedding = new_embedding.to(device)
            
            # Copy existing weights
            new_embedding.weight.data[:old_size] = embedding_layer.weight.data
            
            # Initialize new token (image_token_id) with small random values
            nn.init.normal_(new_embedding.weight.data[image_token_id], mean=0.0, std=0.02)
            
            # Replace embedding layer
            if hasattr(model, 'set_input_embeddings'):
                model.set_input_embeddings(new_embedding)
            elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                model.model.embed_tokens = new_embedding
            elif hasattr(model, 'language_model') and hasattr(model.language_model, 'embed_tokens'):
                model.language_model.embed_tokens = new_embedding
            
            # Also resize output embeddings (lm_head) if it exists
            if hasattr(model, 'get_output_embeddings'):
                output_embeddings = model.get_output_embeddings()
                if output_embeddings is not None:
                    # Create new output layer (on same device)
                    output_dim = output_embeddings.out_features if hasattr(output_embeddings, 'out_features') else output_embeddings.weight.shape[0]
                    device = output_embeddings.weight.device
                    dtype = output_embeddings.weight.dtype
                    new_output = nn.Linear(output_embeddings.in_features, new_size)
                    new_output = new_output.to(device=device, dtype=dtype)
                    
                    # Copy existing weights
                    new_output.weight.data[:old_size] = output_embeddings.weight.data[:old_size]
                    if hasattr(output_embeddings, 'bias') and output_embeddings.bias is not None:
                        new_output.bias.data[:old_size] = output_embeddings.bias.data[:old_size]
                    
                    # Initialize new token
                    nn.init.normal_(new_output.weight.data[image_token_id], mean=0.0, std=0.02)
                    if hasattr(new_output, 'bias') and new_output.bias is not None:
                        new_output.bias.data[image_token_id] = 0.0
                    
                    # Replace output layer
                    if hasattr(model, 'set_output_embeddings'):
                        model.set_output_embeddings(new_output)
                    elif hasattr(model, 'lm_head'):
                        model.lm_head = new_output
                    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
                        model.language_model.lm_head = new_output
                    
                    logger.info(f"  ✓ Resized output embeddings to {new_size}")
            
            logger.info(f"  ✓ Embedding layer resized from {old_size} to {new_size}")
        
        logger.info("=" * 80)
    
    # Apply LoRA
    logger.info("Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
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
    
    # Create processor
    # Try NOT replacing image tokens - let model handle it internally
    # This might fix the "Image features and image tokens do not match" error
    replace_tokens = os.getenv("REPLACE_IMAGE_TOKENS", "True").lower() == "true"
    processor = LlavaProcessorWrapper(image_processor, tokenizer, image_token_id, replace_image_tokens=replace_tokens)
    if not replace_tokens:
        logger.info("⚠️  Image token replacement DISABLED - model will handle <image> tokens internally")
    
    # Load datasets
    logger.info("Loading datasets...")
    # Images are in Kvasir-VQA/raw/images directory
    image_root = "/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
    train_dataset = SurgicalVQADataset(data_root, dataset_name, "train", processor, max_length, image_root=image_root)
    eval_dataset = SurgicalVQADataset(data_root, dataset_name, "val", processor, max_length, image_root=image_root)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=collate_fn, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=4)
    
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
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, 
            epoch + 1, gradient_accumulation_steps, image_token_id, vocab_size
        )
        logger.info(f"Train loss: {train_loss:.4f}")
        
        eval_loss = evaluate(model, eval_loader, device, image_token_id, vocab_size)
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

