#!/usr/bin/env python3
"""Test the trained model on a few examples to see if instruction tuning worked."""
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image
import json
import sys

print("="*80)
print("TESTING INSTRUCTION-TUNED MODEL")
print("="*80)

# Load test data
print("\n1. Loading test data...")
test_file = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/kvasir_instructed/test_instructed.json"
with open(test_file) as f:
    test_data = json.load(f)

# Get 3 diverse examples
binary_sample = next(s for s in test_data if s.get('question_type') == 'binary')
color_sample = next(s for s in test_data if s.get('question_type') == 'color')
numeric_sample = next(s for s in test_data if s.get('question_type') == 'numeric')
samples = [binary_sample, color_sample, numeric_sample]

print(f"✓ Loaded {len(test_data)} test samples")
print(f"✓ Selected 3 test cases (binary, color, numeric)")

# Load model
print("\n2. Loading trained model...")
print("   This may take a minute...")
base_model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model_path = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/models/exp1_random_baseline_instructed/checkpoint-7704"
print(f"   Loading LoRA adapter from: {model_path}")
model = PeftModel.from_pretrained(base_model, model_path)
# Don't merge to save memory
model.eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
print("✓ Model loaded successfully")

# Test each sample
image_dir = "/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"

for i, sample in enumerate(samples, 1):
    print("\n" + "="*80)
    print(f"TEST CASE {i}: {sample['question_type'].upper()}")
    print("="*80)
    
    # Load image
    image_path = f"{image_dir}/{sample['image_filename']}"
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"✗ Could not load image: {e}")
        continue
    
    # Show context
    print(f"\nQuestion Type: {sample['question_type']}")
    print(f"Ground Truth Answer: '{sample['answer']}'")
    print(f"\nInstruction (first 150 chars):\n{sample['instruction'][:150]}...")
    
    # Prepare conversation
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": sample['instruction']}
        ]
    }]
    
    text_prompt = processor.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text_prompt], 
        images=[image], 
        return_tensors="pt"
    ).to(model.device)
    
    # Generate prediction
    print("\nGenerating prediction...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            do_sample=False
        )
    
    prompt_len = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][prompt_len:]
    prediction = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Show results
    print("\n" + "-"*80)
    print("RESULT:")
    print("-"*80)
    print(f"Expected:   '{sample['answer']}'")
    print(f"Predicted:  '{prediction}'")
    print(f"Word count: {len(prediction.split())} words")
    
    # Quick assessment
    if len(prediction.split()) <= 3:
        print("✓ Format: GOOD (concise)")
    else:
        print("✗ Format: BAD (too verbose)")
    
    if prediction.lower().strip() == sample['answer'].lower().strip():
        print("✓ Accuracy: CORRECT")
    elif sample['answer'].lower() in prediction.lower():
        print("~ Accuracy: PARTIAL (contains answer)")
    else:
        print("✗ Accuracy: WRONG")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nINTERPRETATION:")
print("-"*80)
print("If predictions are 1-3 words → Instruction tuning WORKED! ✓")
print("If predictions are 20+ words → Instruction tuning FAILED! ✗")
print("="*80)



