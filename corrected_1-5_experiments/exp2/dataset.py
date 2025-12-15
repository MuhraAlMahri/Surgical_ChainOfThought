import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from pathlib import Path
from transformers import AutoProcessor
from templates import build_conversation


def letterbox_to_square(img, target_size=768):
    """
    Resize image preserving aspect ratio (letterbox), then pad to square.
    This avoids distortion of medical image features.
    
    Args:
        img: PIL Image
        target_size: Target square size (default 768x768)
    
    Returns:
        PIL Image of size (target_size, target_size)
    """
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize with BICUBIC for high quality
    img = img.resize((new_w, new_h), Image.BICUBIC)
    
    # Calculate padding to make square
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)  # L, T, R, B
    
    # Pad with black (0) to target_size x target_size
    return ImageOps.expand(img, border=padding, fill=0)


class VQASFTDataset(Dataset):
    def __init__(self, jsonl_path, image_root, model_name, max_len=512, 
                 use_letterbox=False, target_size=768):
        # Handle both JSONL and JSON array formats
        with open(jsonl_path, 'r') as f:
            content = f.read().strip()
            if content.startswith('['):
                # JSON array format
                self.samples = json.loads(content)
            else:
                # JSONL format (one JSON per line)
                f.seek(0)
                self.samples = [json.loads(l) for l in f]
        self.image_root = Path(image_root)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.max_len = max_len
        self.use_letterbox = use_letterbox
        self.target_size = target_size
        
        # If using letterbox, set fixed resolution for processor
        if use_letterbox:
            # Set processor to handle fixed resolution
            target_pixels = target_size * target_size
            if hasattr(self.processor, 'image_processor'):
                self.processor.image_processor.min_pixels = target_pixels
                self.processor.image_processor.max_pixels = target_pixels
                print(f"✓ Letterbox mode enabled: {target_size}×{target_size} (preserves aspect ratio)")
        else:
            # Keep default adaptive resolution
            print(f"✓ Using adaptive resolution (default Qwen2-VL behavior)")
        
        # Ensure right padding for training
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'right'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ex = self.samples[i]
        
        # Load image
        img_file = ex.get('image') or ex.get('image_filename') or ex.get('image_id')
        if not img_file:
            raise KeyError(f"No image field found in sample. Available fields: {ex.keys()}")
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_file = f"{img_file}.jpg"
        img_path = self.image_root / img_file
        img = Image.open(str(img_path).replace("//", "/")).convert("RGB")
        
        # Apply letterbox if enabled (preserves aspect ratio, no warping)
        if self.use_letterbox:
            img = letterbox_to_square(img, target_size=self.target_size)
        
        # Check if 'instruction' field exists (CATEGORY_BASED dataset)
        if 'instruction' in ex and ex['instruction']:
            # Use pre-built instruction with <ANS> sentinels for training
            instruction_text = ex['instruction']
            answer_text = ex['answer']
            
            # Build conversation with image placeholder
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": f"<ANS>{answer_text}</ANS>"
                }
            ]
            
            # Apply chat template
            full_text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fall back to old method for legacy datasets
            conversation = build_conversation(
                ex["question_type"],
                ex["question"],
                ex.get("answer_candidates"),
                answer=ex["answer"],  # Include ground truth with <ANS> sentinels
                for_training=True
            )
            
            # Apply chat template (renders system/user/assistant turns)
            full_text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False  # We already have assistant response
            )
        
        enc = self.processor(
            text=[full_text],
            images=[img],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_len,
            truncation=True
        )
        
        # Extract tensors (remove batch dimension added by processor)
        input_ids = enc["input_ids"].squeeze(0)  # [batch, seq] -> [seq]
        attention_mask = enc["attention_mask"].squeeze(0)
        pixel_values = enc.get("pixel_values")  # Keep as is for vision processing
        image_grid_thw = enc.get("image_grid_thw")  # Keep as is
        
        # SENTINEL-BASED LABEL MASKING
        # Tokenize sentinels to find their IDs
        tok = self.processor.tokenizer
        ans_start_tokens = tok("<ANS>", add_special_tokens=False)["input_ids"]
        ans_end_tokens = tok("</ANS>", add_special_tokens=False)["input_ids"]
        
        # Initialize labels: all -100 (masked)
        labels = torch.full_like(input_ids, fill_value=-100)
        
        # Find sentinel positions in input_ids
        input_ids_list = input_ids.tolist()
        
        # Search for <ANS> start
        ans_start_idx = None
        ans_end_idx = None
        
        for idx in range(len(input_ids_list) - len(ans_start_tokens) + 1):
            if input_ids_list[idx:idx+len(ans_start_tokens)] == ans_start_tokens:
                ans_start_idx = idx + len(ans_start_tokens)  # Start AFTER <ANS>
                break
        
        # Search for </ANS> end
        if ans_start_idx:
            for idx in range(ans_start_idx, len(input_ids_list) - len(ans_end_tokens) + 1):
                if input_ids_list[idx:idx+len(ans_end_tokens)] == ans_end_tokens:
                    ans_end_idx = idx  # End BEFORE </ANS>
                    break
        
        # Apply masking: only supervise answer tokens (between sentinels)
        # Set labels=-100 except for tokens strictly inside the answer span
        if ans_start_idx and ans_end_idx and ans_start_idx < ans_end_idx:
            # Unmask answer span (between sentinels, not including sentinels themselves)
            labels[ans_start_idx:ans_end_idx] = input_ids[ans_start_idx:ans_end_idx]
            
            # Also supervise EOS if it's right after </ANS>
            eos_idx = ans_end_idx + len(ans_end_tokens)
            if eos_idx < len(input_ids_list) and input_ids_list[eos_idx] == tok.eos_token_id:
                labels[eos_idx] = input_ids[eos_idx]
        
        # Build result dictionary
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        # Add vision-related tensors if available
        # These should maintain their processor-generated shapes
        if pixel_values is not None:
            # Remove batch dimension: [1, C, H, W] -> [C, H, W]
            result["pixel_values"] = pixel_values.squeeze(0) if len(pixel_values.shape) == 4 else pixel_values
        if image_grid_thw is not None:
            # Remove batch dimension but keep image dimension: [1, num_imgs, 3] -> [num_imgs, 3]
            result["image_grid_thw"] = image_grid_thw.squeeze(0) if len(image_grid_thw.shape) > 2 else image_grid_thw
        
        return result


def collate(batch):
    """Collate function for Qwen2-VL vision-language model."""
    keys = batch[0].keys()
    out = {}
    
    for k in keys:
        items = [b[k] for b in batch]
        
        if not isinstance(items[0], torch.Tensor):
            out[k] = items
            continue
        
        if k == "pixel_values":
            # For pixel_values, just stack along batch dimension
            # Input: list of [C, H, W] -> Output: [batch, C, H, W]
            out[k] = torch.stack(items)
            
        elif k == "image_grid_thw":
            # For image_grid_thw, stack and then reshape to [batch*num_images, 3]
            # Input: list of [num_images, 3] -> Stack to [batch, num_images, 3] -> Reshape to [batch*num_images, 3]
            stacked = torch.stack(items)  # [batch, num_images, 3]
            out[k] = stacked.view(-1, 3)  # [batch*num_images, 3]
            
        elif k in ["input_ids", "attention_mask", "labels"]:
            # For sequences, pad to same length if needed
            max_len = max(item.size(0) for item in items)
            pad_value = -100 if k == "labels" else 0
            
            padded = []
            for item in items:
                if item.size(0) < max_len:
                    padding = torch.full((max_len - item.size(0),), pad_value, dtype=item.dtype)
                    padded.append(torch.cat([item, padding]))
                else:
                    padded.append(item)
            out[k] = torch.stack(padded)
        else:
            # For other tensors, try simple stacking
            try:
                out[k] = torch.stack(items)
            except:
                out[k] = items
    
    return out
