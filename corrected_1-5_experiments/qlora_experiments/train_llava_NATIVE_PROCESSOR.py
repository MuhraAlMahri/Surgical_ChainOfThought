#!/usr/bin/env python3

"""
LLaVA-Med Training - CORRECT Image Token Handling

Issue: LLaVA uses MULTIPLE image tokens (576), not just one

Fix: Don't replace <image> manually - let the model handle it with prepare_inputs_for_generation
"""

import os
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
from tqdm import tqdm
from pathlib import Path

# Cache configuration
SHARED_CACHE = "/l/users/muhra.almahri/.cache/hf_shared"
os.environ["HF_HOME"] = SHARED_CACHE
os.environ["TRANSFORMERS_CACHE"] = f"{SHARED_CACHE}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{SHARED_CACHE}/datasets"

# Disable DeepSpeed
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
os.environ["DEEPSPEED_DISABLED"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class SurgicalVQADataset(Dataset):
    """Dataset for surgical VQA using native processor."""
    
    def __init__(self, data_root: str, dataset_name: str, split: str, 
                 processor, max_length: int = 2048, image_root: str = None):
        self.data_root = data_root
        self.image_root = image_root or data_root
        self.processor = processor
        self.max_length = max_length
        
        # Handle JSONL files - support both Kvasir and EndoVis2018 formats
        # Kvasir: {split}_CATEGORY_BASED.jsonl
        # EndoVis2018: {split}.jsonl or validation.jsonl (for val split)
        jsonl_paths = [
            os.path.join(data_root, f"{split}_CATEGORY_BASED.jsonl"),  # Kvasir format
            os.path.join(data_root, f"{split}.jsonl"),  # EndoVis2018 format
        ]
        # For EndoVis2018, also check for "validation" instead of "val"
        if split == "val":
            jsonl_paths.insert(1, os.path.join(data_root, "validation.jsonl"))
        json_path = os.path.join(data_root, f"{split}.json")
        
        loaded = False
        for jsonl_path in jsonl_paths:
            if os.path.exists(jsonl_path):
                self.data = []
                with open(jsonl_path) as f:
                    for line in f:
                        if line.strip():
                            self.data.append(json.loads(line))
                logger.info(f"✓ Loaded {len(self.data)} samples from {jsonl_path}")
                loaded = True
                break
        
        if not loaded and os.path.exists(json_path):
            with open(json_path) as f:
                self.data = json.load(f)
            logger.info(f"✓ Loaded {len(self.data)} samples from {json_path}")
            loaded = True
        
        if not loaded:
            raise FileNotFoundError(f"Could not find data file. Tried: {jsonl_paths + [json_path]}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get image path - handle multiple field names
        image_path = None
        if "image_filename" in item:
            image_path = os.path.join(self.image_root, item["image_filename"])
        elif "image_name" in item:
            image_path = os.path.join(self.image_root, item["image_name"])
        elif "image" in item:
            image_path = os.path.join(self.image_root, item["image"])
        elif "image_path" in item:
            image_path = item["image_path"]
        else:
            raise KeyError(f"No image field found. Available keys: {list(item.keys())}")
        
        # Fallback: try data_root if image_root path doesn't exist
        if not os.path.exists(image_path) and self.image_root != self.data_root:
            if "image_filename" in item:
                fallback_path = os.path.join(self.data_root, item["image_filename"])
                if os.path.exists(fallback_path):
                    image_path = fallback_path
            elif "image_name" in item:
                fallback_path = os.path.join(self.data_root, item["image_name"])
                if os.path.exists(fallback_path):
                    image_path = fallback_path
            elif "image" in item:
                fallback_path = os.path.join(self.data_root, item["image"])
                if os.path.exists(fallback_path):
                    image_path = fallback_path
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Format as conversation
        question = item.get("question", item.get("instruction", ""))
        answer = item.get("answer", "")
        
        # Use LLaVA's expected format
        text = f"USER: <image>\n{question}\nASSISTANT: {answer}"
        
        # CRITICAL: Let the processor handle everything
        # Don't manually replace <image> tokens
        # NO padding here - will pad dynamically per batch in collate_fn
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=False,  # Don't pad to max_length - will pad dynamically per batch
            truncation=True,
            max_length=self.max_length,
        )
        
        # Squeeze batch dimension
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        # Create labels - mask everything before ASSISTANT:
        labels = inputs["input_ids"].clone()
        
        # Mask image tokens (image_token_id) - they should not be predicted
        image_token_id = getattr(self.processor, 'image_token_id', None)
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
        
        # Find ASSISTANT: and mask everything before it
        input_ids_list = inputs["input_ids"].tolist()
        assistant_str = " ASSISTANT:"
        assistant_ids = self.processor.tokenizer.encode(assistant_str, add_special_tokens=False)
        
        # Find the assistant marker
        for i in range(len(input_ids_list) - len(assistant_ids)):
            if input_ids_list[i:i+len(assistant_ids)] == assistant_ids:
                labels[:i+len(assistant_ids)] = -100
                break
        
        # Mask padding
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        
        inputs["labels"] = labels
        
        return inputs


def make_collate_fn(pad_token_id):
    """
    Create a collate function with DYNAMIC PADDING.
    Each batch is padded only to the longest sequence in THAT batch,
    not to the global max_length. This dramatically speeds up training!
    
    Args:
        pad_token_id: The padding token ID to use (extracted from processor before multiprocessing)
    """
    def collate_fn(batch):
        # Find max length in this batch
        max_len_in_batch = max(x['input_ids'].shape[0] for x in batch)
        
        # Use the pad_token_id passed in (no need to access processor)
        pad_token_id_value = pad_token_id if pad_token_id is not None else 0
        
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
                    torch.full((pad_len,), pad_token_id_value, dtype=x['input_ids'].dtype)
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
        
        # Stack all tensors
        batch_dict = {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels': torch.stack(labels_list),
        }
        
        # Add pixel_values and any other keys
        for key in batch[0].keys():
            if key not in ['input_ids', 'attention_mask', 'labels']:
                if isinstance(batch[0][key], torch.Tensor):
                    batch_dict[key] = torch.stack([item[key] for item in batch])
                else:
                    batch_dict[key] = [item[key] for item in batch]
        
        return batch_dict
    
    return collate_fn


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, 
                gradient_accumulation_steps):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Log first batch info
            if batch_idx == 0:
                logger.info(f"First batch keys: {batch.keys()}")
                if 'input_ids' in batch:
                    logger.info(f"  input_ids shape: {batch['input_ids'].shape}")
                if 'pixel_values' in batch:
                    logger.info(f"  pixel_values shape: {batch['pixel_values'].shape}")
                if 'attention_mask' in batch:
                    logger.info(f"  attention_mask shape: {batch['attention_mask'].shape}")
            
            # Forward pass
            outputs = model(**batch)
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
            if 'input_ids' in batch:
                logger.error(f"  input_ids shape: {batch['input_ids'].shape}")
                logger.error(f"  input_ids sample: {batch['input_ids'][0][:50]}")
            if 'pixel_values' in batch:
                logger.error(f"  pixel_values shape: {batch['pixel_values'].shape}")
            raise
    
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


def _resize_embeddings_manual(model, old_vocab_size, new_vocab_size):
    """Manually resize embedding layers (same approach as evaluation script)."""
    import torch.nn as nn
    
    # For LLaVA models, the structure is model.model.embed_tokens
    # Access the base model structure correctly
    base_model_unwrapped = model
    if hasattr(base_model_unwrapped, 'model'):
        base_model_unwrapped = base_model_unwrapped.model
    
    # Resize token embeddings
    if hasattr(base_model_unwrapped, 'embed_tokens'):
        old_embedding = base_model_unwrapped.embed_tokens
        new_embedding = nn.Embedding(new_vocab_size, old_embedding.embedding_dim)
        new_embedding.weight.data[:old_vocab_size] = old_embedding.weight.data
        # Initialize new token embedding with small random values
        if new_vocab_size > old_vocab_size:
            new_embedding.weight.data[old_vocab_size:] = torch.randn(
                new_vocab_size - old_vocab_size, 
                old_embedding.embedding_dim
            ) * 0.02
        base_model_unwrapped.embed_tokens = new_embedding
        logger.info("✓ Resized embed_tokens manually")
    
    # Resize output embeddings (lm_head)
    if hasattr(base_model_unwrapped, 'lm_head'):
        old_lm_head = base_model_unwrapped.lm_head
        new_lm_head = nn.Linear(
            old_lm_head.in_features, 
            new_vocab_size, 
            bias=old_lm_head.bias is not None
        )
        new_lm_head.weight.data[:old_vocab_size] = old_lm_head.weight.data
        if old_lm_head.bias is not None:
            new_lm_head.bias.data[:old_vocab_size] = old_lm_head.bias.data
        # Initialize new token output weights
        if new_vocab_size > old_vocab_size:
            new_lm_head.weight.data[old_vocab_size:] = torch.randn(
                new_vocab_size - old_vocab_size,
                old_lm_head.in_features
            ) * 0.02
            if new_lm_head.bias is not None:
                new_lm_head.bias.data[old_vocab_size:] = 0.0
        base_model_unwrapped.lm_head = new_lm_head
        logger.info("✓ Resized lm_head manually")


def main():
    model_name = "microsoft/llava-med-v1.5-mistral-7b"
    dataset_name = os.getenv("DATASET_NAME", "kvasir")
    
    # Set default data root based on dataset
    if dataset_name == "endovis" or "endovis" in dataset_name.lower():
        default_data_root = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/endovis2018_vqa"
    else:
        default_data_root = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/kvasir_ULTRA_CONDENSED"
    
    data_root = os.getenv("DATA_ROOT", default_data_root)
    output_dir = os.getenv("OUTPUT_DIR", f"./outputs/llava_med_{dataset_name}_native_processor")
    
    num_epochs = 5
    batch_size = 1
    gradient_accumulation_steps = 16
    learning_rate = 5.0e-5
    weight_decay = 0.01
    warmup_steps = 100
    max_length = 3072
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for existing checkpoints to resume from
    resume_from_checkpoint = None
    start_epoch = 0
    
    if os.path.exists(output_dir):
        # Find all checkpoint directories
        checkpoints = []
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-epoch-") and os.path.isdir(os.path.join(output_dir, item)):
                try:
                    epoch_num = int(item.split("-")[-1])
                    checkpoint_path = os.path.join(output_dir, item)
                    # Verify checkpoint has adapter files
                    if os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")) or \
                       os.path.exists(os.path.join(checkpoint_path, "adapter_model.bin")):
                        checkpoints.append((epoch_num, checkpoint_path))
                except ValueError:
                    continue
        
        if checkpoints:
            # Sort by epoch number and get the latest
            checkpoints.sort(key=lambda x: x[0])
            latest_epoch, latest_checkpoint = checkpoints[-1]
            resume_from_checkpoint = latest_checkpoint
            start_epoch = latest_epoch  # Will continue from next epoch
            logger.info("=" * 80)
            logger.info(f"🔄 RESUME MODE: Found checkpoint at epoch {latest_epoch}")
            logger.info(f"   Checkpoint: {latest_checkpoint}")
            logger.info(f"   Will continue from epoch {start_epoch + 1}")
            logger.info("=" * 80)
    
    logger.info("=" * 80)
    logger.info("LLAVA-MED TRAINING - NATIVE PROCESSOR (CORRECT FIX)")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Data root: {data_root}")
    if resume_from_checkpoint:
        logger.info(f"Resume: Yes (from epoch {start_epoch})")
    else:
        logger.info("Resume: No (starting fresh)")
    logger.info("=" * 80)
    
    # Device will be determined after model loading when using device_map="auto"
    # For now, set a default that will be updated after model load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initial device setting: {device} (will be updated after model load)")
    
    # Load config
    logger.info("Loading configuration...")
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except ValueError as e:
        if "llava_mistral" in str(e):
            logger.info("⚠️  llava_mistral not recognized, loading config manually...")
            from huggingface_hub import hf_hub_download
            from transformers import LlavaConfig
            import json
            
            hf_token = os.environ.get('HF_TOKEN')
            config_path = hf_hub_download(
                repo_id=model_name,
                filename="config.json",
                token=hf_token
            )
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            if config_dict.get('model_type') == 'llava_mistral':
                config_dict['model_type'] = 'llava'
                logger.info("  ⚠️  Changed model_type from 'llava_mistral' to 'llava' for compatibility")
            
            config = LlavaConfig.from_dict(config_dict)
            if hasattr(config, 'auto_map'):
                config.trust_remote_code = True
        else:
            raise
    
    logger.info(f"Config image_token_id: {getattr(config, 'image_token_id', 'NOT SET')}")
    logger.info(f"Config vocab_size: {config.vocab_size}")
    
    # Load model
    logger.info("Loading model...")
    # Use device_map="cuda" for training to ensure all layers are on GPU
    # This avoids device mismatch errors when moving inputs to CUDA
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",  # Changed from "auto" to "cuda" to ensure all layers on GPU
        trust_remote_code=True,
    )
    logger.info("✓ Model loaded")
    
    # Get the actual device from the model to ensure consistency
    first_param = next(model.parameters())
    model_device = first_param.device
    logger.info(f"Model device: {model_device}")
    # Update device variable to match model's device
    device = model_device
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled")
    elif hasattr(model, 'model') and hasattr(model.model, 'gradient_checkpointing_enable'):
        model.model.gradient_checkpointing_enable()
        logger.info("✓ Gradient checkpointing enabled (via model.model)")
    
    # CRITICAL: Resize embedding layer if image_token_id >= vocab_size (same as evaluation script)
    image_token_id = getattr(config, 'image_token_id', 32000)
    vocab_size = config.vocab_size
    
    # Determine target vocab size
    if image_token_id >= vocab_size:
        target_vocab_size = image_token_id + 1
    else:
        target_vocab_size = None
    
    if target_vocab_size and target_vocab_size > vocab_size:
        logger.info(f"⚠️  Resizing embedding layer from {vocab_size} to {target_vocab_size}...")
        logger.info(f"  (image_token_id={image_token_id} >= vocab_size={vocab_size})")
        
        # Try standard resize_token_embeddings method first
        if hasattr(model, 'resize_token_embeddings'):
            try:
                model.resize_token_embeddings(target_vocab_size)
                logger.info("✓ Resized embeddings using resize_token_embeddings")
            except Exception as e:
                logger.warning(f"resize_token_embeddings failed: {e}, trying manual resize...")
                # Fall back to manual resize
                _resize_embeddings_manual(model, vocab_size, target_vocab_size)
        else:
            logger.info("Model does not have resize_token_embeddings method, using manual resize...")
            # Manual resize
            _resize_embeddings_manual(model, vocab_size, target_vocab_size)
        
        # Update config
        config.vocab_size = target_vocab_size
        logger.info(f"✓ Updated config.vocab_size to {config.vocab_size}")
    
    # Apply LoRA or load from checkpoint
    if resume_from_checkpoint:
        logger.info(f"Loading LoRA adapter from checkpoint: {resume_from_checkpoint}")
        # PeftModel.from_pretrained will automatically load the adapter config and weights
        model = PeftModel.from_pretrained(model, resume_from_checkpoint, is_trainable=True)
        logger.info("✓ Loaded adapter from checkpoint - continuing training")
        model.print_trainable_parameters()
    else:
        logger.info("Applying LoRA...")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # CRITICAL: Use AutoProcessor (native LLaVA processor)
    logger.info("Loading processor...")
    try:
        # Try to load the native processor
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        logger.info("✓ Loaded native AutoProcessor")
    except Exception as e:
        logger.warning(f"AutoProcessor failed: {e}")
        logger.info("Loading tokenizer and image processor separately...")
        
        # Fallback: load components separately
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        image_processor.size = {"height": 336, "width": 336}
        image_processor.crop_size = {"height": 336, "width": 336}
        
        # Get image_token_id from config
        image_token_id = getattr(config, 'image_token_id', 32000)
        logger.info(f"Using image_token_id: {image_token_id}")
        
        # Create a simple processor wrapper that replaces <image> with 576 image_token_id tokens
        # The model expects 576 occurrences of image_token_id (one per image patch token)
        # For 336x336 images with patch size 14: (336/14)^2 = 24^2 = 576 patches
        class SimpleProcessor:
            def __init__(self, image_processor, tokenizer, image_token_id):
                self.image_processor = image_processor
                self.tokenizer = tokenizer
                self.image_token_id = image_token_id
                # Get the token sequence for <image>
                self.image_token_seq = self.tokenizer.encode("<image>", add_special_tokens=False)
                # Calculate number of image tokens: (image_size / patch_size)^2
                # For 336x336 with patch_size 14: (336/14)^2 = 576
                self.num_image_tokens = 576
                logger.info(f"<image> tokenizes as: {self.image_token_seq}")
                logger.info(f"Will replace with {self.num_image_tokens} image_token_id tokens: {self.image_token_id}")
            
            def __call__(self, text=None, images=None, **kwargs):
                outputs = {}
                if images is not None:
                    if not isinstance(images, list):
                        images = [images]
                    img_outputs = self.image_processor(images, return_tensors="pt")
                    outputs["pixel_values"] = img_outputs["pixel_values"]
                
                if text is not None:
                    # Get padding and max_length from kwargs
                    padding = kwargs.get("padding", False)
                    max_length = kwargs.get("max_length", 2048)
                    
                    # Remove padding from kwargs if False (let tokenizer handle it naturally)
                    tokenizer_kwargs = kwargs.copy()
                    if padding is False:
                        tokenizer_kwargs["padding"] = False
                    
                    # Handle single string vs list
                    is_single = isinstance(text, str)
                    if is_single:
                        text_list = [text]
                    else:
                        text_list = text
                    
                    text_outputs = self.tokenizer(text_list, **tokenizer_kwargs)
                    
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
                                    # Replace <image> (3 tokens) with 576 image_token_id tokens
                                    new_ids.extend([self.image_token_id] * self.num_image_tokens)
                                    i += len(self.image_token_seq)
                                else:
                                    new_ids.append(ids[i])
                                    i += 1
                            
                            # Truncate if needed, but only pad if padding=True
                            if len(new_ids) > max_length:
                                new_ids = new_ids[:max_length]
                            
                            # Only pad if padding is explicitly True or "max_length"
                            if padding and len(new_ids) < max_length:
                                pad_token_id = self.tokenizer.pad_token_id or 0
                                new_ids.extend([pad_token_id] * (max_length - len(new_ids)))
                            
                            # Create tensor with actual length (not padded to max_length if padding=False)
                            new_input_ids_list.append(torch.tensor(new_ids, dtype=input_ids.dtype))
                            
                            # Update attention_mask if present
                            if attention_mask is not None:
                                new_mask = [1] * len(new_ids)
                                if padding and len(new_mask) < max_length:
                                    new_mask.extend([0] * (max_length - len(new_mask)))
                                new_attention_mask_list.append(torch.tensor(new_mask, dtype=attention_mask.dtype))
                        
                        # Stack tensors (always keep batch dimension)
                        text_outputs["input_ids"] = torch.stack(new_input_ids_list)
                        
                        if attention_mask is not None:
                            text_outputs["attention_mask"] = torch.stack(new_attention_mask_list)
                    
                    outputs.update(text_outputs)
                
                return outputs
        
        processor = SimpleProcessor(image_processor, tokenizer, image_token_id)
        logger.info("✓ Created fallback processor with image token replacement (model expands 1 token → 576 tokens)")
    
    # Load datasets
    logger.info("Loading datasets...")
    # Set image root based on dataset
    if dataset_name == "endovis" or "endovis" in dataset_name.lower():
        image_root = "/l/users/muhra.almahri/Surgical_COT/datasets/EndoVis2018/raw/images"
    else:
        image_root = "/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
    
    logger.info(f"Image root: {image_root}")
    train_dataset = SurgicalVQADataset(data_root, dataset_name, "train", processor, max_length, image_root=image_root)
    eval_dataset = SurgicalVQADataset(data_root, dataset_name, "val", processor, max_length, image_root=image_root)
    
    # Extract pad_token_id before creating collate function (for multiprocessing safety)
    if hasattr(processor, 'tokenizer'):
        pad_token_id = processor.tokenizer.pad_token_id
    elif hasattr(processor, 'pad_token_id'):
        pad_token_id = processor.pad_token_id
    else:
        pad_token_id = 0
    
    if pad_token_id is None:
        pad_token_id = 0
    
    logger.info(f"Using pad_token_id: {pad_token_id}")
    
    # Create collate function with dynamic padding (pass pad_token_id, not processor)
    collate_fn = make_collate_fn(pad_token_id)
    
    # Use num_workers=0 to avoid multiprocessing issues with tokenizers/processors
    # This is safer and avoids serialization issues
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with tokenizers
        pin_memory=False,  # Disable pin_memory when num_workers=0
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with tokenizers
        pin_memory=False,  # Disable pin_memory when num_workers=0
    )
    
    # Test first batch
    logger.info("=" * 80)
    logger.info("TESTING FIRST BATCH")
    logger.info("=" * 80)
    test_batch = next(iter(train_loader))
    logger.info(f"Batch keys: {test_batch.keys()}")
    for key, value in test_batch.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    logger.info("=" * 80)
    
    # Optimizer
    logger.info("Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    # Calculate total steps based on remaining epochs
    remaining_epochs = num_epochs - start_epoch
    total_steps = len(train_loader) * remaining_epochs // gradient_accumulation_steps
    if resume_from_checkpoint:
        logger.info(f"Resuming: {remaining_epochs} epochs remaining, {total_steps} total steps")
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training
    logger.info("=" * 80)
    if resume_from_checkpoint:
        logger.info(f"RESUMING TRAINING (from epoch {start_epoch + 1})")
    else:
        logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    # Load best eval loss if resuming
    best_eval_loss = float('inf')
    if resume_from_checkpoint:
        best_model_path = os.path.join(output_dir, "best_model")
        if os.path.exists(best_model_path):
            # Try to read the best eval loss from a saved file, or keep it as inf
            # For now, we'll keep it as inf and let it update naturally
            logger.info("Note: Best model checkpoint found, but will recalculate best eval loss")
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, 
            epoch + 1, gradient_accumulation_steps
        )
        logger.info(f"Train loss: {train_loss:.4f}")
        
        eval_loss = evaluate(model, eval_loader, device)
        logger.info(f"Eval loss: {eval_loss:.4f}")
        
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            save_path = os.path.join(output_dir, "best_model")
            model.save_pretrained(save_path)
            if hasattr(processor, 'tokenizer'):
                processor.tokenizer.save_pretrained(save_path)
            logger.info(f"✓ Saved best model (eval_loss: {eval_loss:.4f})")
        
        checkpoint_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(checkpoint_path)
        if hasattr(processor, 'tokenizer'):
            processor.tokenizer.save_pretrained(checkpoint_path)
    
    logger.info("=" * 80)
    logger.info(f"TRAINING COMPLETED - Best loss: {best_eval_loss:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

