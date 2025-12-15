#!/usr/bin/env python3
"""
Standalone test to debug label creation without needing GPU/SLURM
"""

import sys
import json

# Load one sample from the training data
data_file = "/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/datasets/kvasir_raw_6500_image_level_70_15_15/train.json"

print("=" * 80)
print("DEBUGGING LABEL CREATION")
print("=" * 80)

print("\n1. Loading one training sample...")
with open(data_file) as f:
    data = json.load(f)

if isinstance(data, list) and len(data) > 0:
    sample = data[0]
    print(f"✓ Loaded sample")
    print(f"  Keys: {sample.keys()}")
    print(f"  Question: {sample.get('question', '')[:100]}...")
    print(f"  Answer: {sample.get('answer', '')[:100]}...")
    
    question = sample.get('question', '')
    answer = sample.get('answer', '')
    
    print(f"\n2. Text lengths:")
    print(f"  Question length: {len(question)} characters")
    print(f"  Answer length: {len(answer)} characters")
    
    # Simulate what the collator does
    prompt_text = f"### Instruction:\n{question}\n\n### Response:"
    full_text = f"{prompt_text}{answer}"
    
    print(f"\n3. After formatting:")
    print(f"  Prompt text: {len(prompt_text)} characters")
    print(f"  Full text: {len(full_text)} characters")
    print(f"  Difference (answer): {len(full_text) - len(prompt_text)} characters")
    
    print("\n" + "=" * 80)
    print("ISSUE HYPOTHESIS:")
    print("=" * 80)
    
    print("\nThe LazyVQACollator tokenizes:")
    print("  1. prompt_texts WITH images → includes vision embeddings")
    print("  2. full_texts WITH images → ALSO includes vision embeddings")
    
    print("\n🔴 THE PROBLEM:")
    print("  Both tokenizations include the SAME image!")
    print("  Vision tokens are identical in both!")
    
    print("\n  prompt_inputs = processor(prompt_texts, images=images)")
    print("  → [IMAGE_TOKENS] + [QUESTION_TOKENS] + [RESPONSE_PREFIX]")
    print(f"  → ~256 image tokens + ~{len(question)//4} question tokens ≈ 300-800 tokens")
    
    print("\n  inputs = processor(full_texts, images=images)")
    print("  → [IMAGE_TOKENS] + [QUESTION_TOKENS] + [RESPONSE_PREFIX] + [ANSWER_TOKENS]")
    print(f"  → SAME image tokens + question + answer")
    
    print("\n  BUT when truncation=False on full_texts:")
    print("  → Sequence can be VERY LONG (>2000 tokens)")
    
    print("\n  prompt_len = len(prompt_inputs)")
    print(f"  → ~300-800 tokens (image + question)")
    
    print("\n  full_len = len(inputs)")
    print(f"  → ~300-800 + {len(answer)//4} = ???")
    
    print("\n  CRITICAL QUESTION:")
    print("  Does processor() add image tokens to BOTH calls?")
    print("  If yes, they cancel out when we subtract!")
    
    print("\n" + "=" * 80)
    print("🔧 POSSIBLE SOLUTION:")
    print("=" * 80)
    
    print("\nWe should tokenize WITHOUT images for the prompt:")
    print("  prompt_inputs = processor(prompt_texts)  # NO images")
    print("  inputs = processor(full_texts, images=images)  # WITH images")
    
    print("\nOr better yet:")
    print("  1. Tokenize full text with image")
    print("  2. Find where 'Response:' ends in tokens")
    print("  3. Mask everything before that")
    print("  4. Don't use separate prompt tokenization")
    
    print("\n" + "=" * 80)

else:
    print("❌ Couldn't load training data")

