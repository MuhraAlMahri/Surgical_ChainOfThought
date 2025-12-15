"""
Dataset class for resolution testing - allows custom min/max pixels.
"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor


class VQASFTDatasetResTest(Dataset):
    """VQA Dataset with custom resolution support for testing."""
    
    def __init__(self, jsonl_path, image_root, model_name, max_len=512, min_pixels=None, max_pixels=None):
        self.samples = [json.loads(l) for l in open(jsonl_path)]
        self.image_root = Path(image_root)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.max_len = max_len
        
        # Override image processor resolution if specified
        if min_pixels is not None and max_pixels is not None:
            if hasattr(self.processor, 'image_processor'):
                self.processor.image_processor.min_pixels = min_pixels
                self.processor.image_processor.max_pixels = max_pixels
                print(f"✅ Dataset processor set: min_pixels={min_pixels:,}, max_pixels={max_pixels:,}")
        
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
            # Legacy format (shouldn't hit this for CATEGORY_BASED)
            from templates import build_conversation
            conversation = build_conversation(
                ex["question_type"],
                ex["question"],
                ex.get("answer_candidates"),
                answer=ex["answer"],
                for_training=True
            )
            
            full_text = self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
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
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        pixel_values = enc.get("pixel_values")
        image_grid_thw = enc.get("image_grid_thw")
        
        # SENTINEL-BASED LABEL MASKING
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
                ans_start_idx = idx + len(ans_start_tokens)
                break
        
        # Search for </ANS> end
        if ans_start_idx:
            for idx in range(ans_start_idx, len(input_ids_list) - len(ans_end_tokens) + 1):
                if input_ids_list[idx:idx+len(ans_end_tokens)] == ans_end_tokens:
                    ans_end_idx = idx
                    break
        
        # Apply masking: only supervise answer tokens (between sentinels)
        if ans_start_idx and ans_end_idx and ans_start_idx < ans_end_idx:
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
        if pixel_values is not None:
            result["pixel_values"] = pixel_values.squeeze(0) if len(pixel_values.shape) == 4 else pixel_values
        if image_grid_thw is not None:
            result["image_grid_thw"] = image_grid_thw.squeeze(0) if len(image_grid_thw.shape) > 2 else image_grid_thw
        
        return result







