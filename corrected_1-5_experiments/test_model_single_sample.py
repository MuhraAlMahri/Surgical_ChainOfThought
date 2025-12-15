#!/usr/bin/env python3
# Test the trained instructed model on a single sample

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image
import json

# Load test data to get a real example
with open('/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/datasets/kvasir_instructed/test_instructed.json') as f:
    test_data = json.load(f)

# Get first binary question
sample = test_data[0]

print("="*80)
print("TESTING TRAINED MODEL")
print("="*80)
print(f"\nGround Truth Answer: {sample['answer']}")
print(f"Question Type: {sample['question_type']}")
print(f"\nInstruction (first 200 chars):\n{sample['instruction'][:200]}...")

# Load model
print("\n" + "="*80)
print("Loading model...")
base_model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model_path = "/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/models/exp1_random_baseline_instructed/checkpoint-7704"
model = PeftModel.from_pretrained(base_model, model_path)
# Don't merge to save memory - use LoRA adapter directly
model.eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

# Load image
image_path = f"/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images/{sample['image_filename']}"
image = Image.open(image_path).convert('RGB')

# Prepare conversation with instruction
conversation = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": sample['instruction']}
    ]
}]

text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(model.device)

# Generate
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

prompt_len = inputs['input_ids'].shape[1]
generated_tokens = outputs[0][prompt_len:]
prediction = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

print("\n" + "="*80)
print("MODEL PREDICTION:")
print("="*80)
print(prediction)
print("\n" + "="*80)
print(f"Expected: {sample['answer']}")
print(f"Got: {prediction}")
print(f"Word count: {len(prediction.split())}")
print("="*80)


