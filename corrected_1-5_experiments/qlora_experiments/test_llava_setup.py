#!/usr/bin/env python3
"""
Quick test script to verify LLaVA-Med setup before running full training.
Run this BEFORE submitting your training job to catch any issues early!
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoConfig, AutoImageProcessor
from PIL import Image
import json

# Disable DeepSpeed
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
os.environ["DEEPSPEED_DISABLED"] = "true"

print("=" * 80)
print("LLAVA-MED SETUP VERIFICATION")
print("=" * 80)

# Configuration
model_name = "microsoft/llava-med-v1.5-mistral-7b"
dataset_name = os.getenv("DATASET_NAME", "kvasir")
data_root = os.getenv("DATA_ROOT", "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/kvasir_ULTRA_CONDENSED")

print(f"\nModel: {model_name}")
print(f"Dataset: {dataset_name}")
print(f"Data root: {data_root}")
print()

# Step 1: Load configuration
print("Step 1: Loading configuration...")
try:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    image_token_id = getattr(config, "image_token_id", config.vocab_size)
    print(f"✓ Config loaded successfully")
    print(f"  - image_token_id: {image_token_id}")
    print(f"  - vocab_size: {config.vocab_size}")
except Exception as e:
    print(f"✗ Failed to load config: {e}")
    sys.exit(1)

# Step 2: Load model
print("\nStep 2: Loading model (this may take a minute)...")
try:
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✓ Model loaded successfully")
    print(f"  - Device: {model.device}")
    print(f"  - Dtype: {model.dtype}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Step 3: Load tokenizer and processor
print("\nStep 3: Loading tokenizer and image processor...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try to load image processor - handle cases where it's not available
    try:
        image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        print("✓ Loaded image processor using AutoImageProcessor")
    except (OSError, Exception) as e:
        print(f"⚠️  AutoImageProcessor failed: {e}")
        print("Trying CLIPImageProcessor from llava-v1.5-7b...")
        from transformers import CLIPImageProcessor
        try:
            image_processor = CLIPImageProcessor.from_pretrained(
                "liuhaotian/llava-v1.5-7b", trust_remote_code=True
            )
            print("✓ Loaded CLIPImageProcessor from llava-v1.5-7b")
        except Exception as e2:
            print(f"⚠️  CLIPImageProcessor from llava-v1.5-7b failed: {e2}")
            print("Trying CLIPImageProcessor from model repo directly...")
            try:
                image_processor = CLIPImageProcessor.from_pretrained(
                    model_name, trust_remote_code=True
                )
                print("✓ Loaded CLIPImageProcessor from model repo")
            except Exception as e3:
                print(f"⚠️  CLIPImageProcessor from model repo failed: {e3}")
                print("Creating CLIPImageProcessor with default settings...")
                try:
                    # Final fallback: create CLIPImageProcessor with default settings
                    from transformers import CLIPImageProcessor
                    image_processor = CLIPImageProcessor.from_pretrained(
                        "openai/clip-vit-large-patch14",
                        trust_remote_code=True
                    )
                    print("✓ Created CLIPImageProcessor with default settings from clip-vit-large-patch14")
                except Exception as e4:
                    print(f"✗ All image processor loading methods failed. Last error: {e4}")
                    sys.exit(1)
    
    print(f"✓ Tokenizer and processor loaded successfully")
    print(f"  - Vocab size: {len(tokenizer)}")
    print(f"  - Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
except Exception as e:
    print(f"✗ Failed to load tokenizer/processor: {e}")
    sys.exit(1)

# Step 4: Load a sample from dataset
print("\nStep 4: Loading a sample from dataset...")
try:
    # Try JSONL first (most common format)
    jsonl_path = os.path.join(data_root, "train.jsonl")
    json_path = os.path.join(data_root, "train.json")
    
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            sample = json.loads(f.readline())
    elif os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
            sample = data[0]
    else:
        raise FileNotFoundError(f"Neither {jsonl_path} nor {json_path} found")
    
    # Handle different image path formats
    if "image" in sample:
        image_path = os.path.join(data_root, sample["image"])
    elif "image_path" in sample:
        image_path = sample["image_path"]
    else:
        raise KeyError("No 'image' or 'image_path' field found")
    
    image = Image.open(image_path).convert("RGB")
    question = sample["question"]
    answer = sample["answer"]
    
    print(f"✓ Sample loaded successfully")
    print(f"  - Image: {sample.get('image', sample.get('image_path', 'N/A'))}")
    print(f"  - Image size: {image.size}")
    print(f"  - Question: {question[:50]}...")
    print(f"  - Answer: {answer[:50]}...")
except Exception as e:
    print(f"✗ Failed to load sample: {e}")
    print(f"  Make sure DATA_ROOT is correct: {data_root}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Process image
print("\nStep 5: Processing image...")
try:
    image_inputs = image_processor(image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"]
    print(f"✓ Image processed successfully")
    print(f"  - pixel_values shape: {pixel_values.shape}")
except Exception as e:
    print(f"✗ Failed to process image: {e}")
    sys.exit(1)

# Step 6: Test image token replacement
print("\nStep 6: Testing image token replacement...")
try:
    text = f"USER: <image>\n{question}\nASSISTANT: {answer}"
    
    # Tokenize
    text_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = text_inputs["input_ids"]
    
    print(f"  - Text contains <image>: {text.count('<image>')} times")
    
    # Get how <image> is tokenized
    image_token_seq = tokenizer.encode("<image>", add_special_tokens=False)
    print(f"  - <image> tokenizes to: {image_token_seq}")
    
    # Replace
    for batch_idx in range(input_ids.size(0)):
        ids = input_ids[batch_idx].tolist()
        new_ids = []
        i = 0
        replacements = 0
        while i < len(ids):
            if ids[i:i+len(image_token_seq)] == image_token_seq:
                new_ids.append(image_token_id)
                i += len(image_token_seq)
                replacements += 1
            else:
                new_ids.append(ids[i])
                i += 1
        input_ids[batch_idx] = torch.tensor(new_ids, dtype=input_ids.dtype)
    
    # Verify
    image_token_count = (input_ids == image_token_id).sum().item()
    
    print(f"✓ Image token replacement successful")
    print(f"  - Replacements made: {replacements}")
    print(f"  - Image tokens in input_ids: {image_token_count}")
    
    if image_token_count == 0:
        print(f"✗ WARNING: No image tokens found after replacement!")
        sys.exit(1)
    
except Exception as e:
    print(f"✗ Failed image token replacement: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 7: Test forward pass
print("\nStep 7: Testing forward pass...")
try:
    labels = input_ids.clone()
    assistant_ids = tokenizer.encode("ASSISTANT:", add_special_tokens=False)
    input_ids_list = input_ids[0].tolist()
    
    for i in range(len(input_ids_list) - len(assistant_ids)):
        if input_ids_list[i:i+len(assistant_ids)] == assistant_ids:
            labels[0, :i+len(assistant_ids)] = -100
            break
    
    labels[labels == tokenizer.pad_token_id] = -100
    
    input_ids = input_ids.to(model.device)
    pixel_values = pixel_values.to(model.device)
    attention_mask = text_inputs["attention_mask"].to(model.device)
    labels = labels.to(model.device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
        )
    
    loss = outputs.loss
    
    print(f"✓ Forward pass successful")
    print(f"  - Loss: {loss.item():.4f}")
    print(f"  - Loss is finite: {torch.isfinite(loss).item()}")
    
    if not torch.isfinite(loss):
        print(f"✗ WARNING: Loss is not finite!")
        sys.exit(1)
    
except Exception as e:
    print(f"✗ Failed forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print("\n✓ All tests passed! Your setup is ready for training.")
print("\nNext steps:")
print("1. Run training: python train_llava_manual.py --config <config.yaml>")
print("2. Monitor: tail -f logs/llava_manual_*.out")
print("\nGood luck! 🚀")
print("=" * 80)


