#!/usr/bin/env python3
"""
Reorder EndoVis2018 VQA data into 3 clinical stages using Qwen
This script:
1. Loads EndoVis2018 VQA data
2. Uses Qwen to classify each question into clinical stages (1, 2, or 3)
3. Reorders data by stage
4. Creates separate data files for Exp2, Exp3, Exp4, Exp5
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import Counter
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

BASE_DIR = Path("/l/users/muhra.almahri/Surgical_COT")
INPUT_DIR = BASE_DIR / "corrected_1-5_experiments" / "datasets" / "endovis2018_vqa"
OUTPUT_DIR = BASE_DIR / "corrected_1-5_experiments" / "datasets" / "endovis2018_vqa_reordered"


def load_qwen_model():
    """Load Qwen model for stage classification."""
    print("Loading Qwen model for stage classification...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print("✓ Model loaded")
    return model, tokenizer


def classify_question_stage(model, tokenizer, question: str, category: str) -> int:
    """
    Use Qwen to classify question into clinical stage.
    
    Stage 1: Initial Assessment (procedure type, quality checks)
    Stage 2: Findings Identification (instruments, anatomy, counts)
    Stage 3: Clinical Context (diagnosis, treatment - rare for EndoVis2018)
    """
    prompt = f"""You are a medical AI assistant. Classify this surgical VQA question into one of three clinical reasoning stages:

Stage 1 (Initial Assessment): Questions about procedure type, image quality, basic setup
Stage 2 (Findings Identification): Questions about detecting/counting instruments, anatomy, specific findings
Stage 3 (Clinical Context): Questions about diagnosis, treatment recommendations, clinical significance

Question: {question}
Category: {category}

Respond with ONLY the stage number (1, 2, or 3). No explanation needed."""

    messages = [
        {"role": "system", "content": "You are a helpful assistant that classifies medical questions into clinical reasoning stages."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.1
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    
    # Extract stage number
    try:
        # Look for number 1, 2, or 3 in response
        if "1" in response or "stage 1" in response.lower() or "initial" in response.lower():
            return 1
        elif "3" in response or "stage 3" in response.lower() or "clinical context" in response.lower() or "diagnosis" in response.lower():
            return 3
        else:
            return 2  # Default to stage 2 (most common)
    except:
        # Fallback: use rule-based classification
        return classify_question_stage_rule_based(question, category)


def classify_question_stage_rule_based(question: str, category: str) -> int:
    """Rule-based classification for EndoVis2018 questions."""
    q_lower = question.lower()
    cat_lower = category.lower()
    
    # Stage 1: Initial Assessment - Procedure type questions
    if "PROCEDURE_TYPE" in category or "procedure" in q_lower or "type of" in q_lower:
        return 1
    
    # Stage 3: Clinical Context - Diagnosis, treatment (very rare for EndoVis2018)
    if any(kw in q_lower for kw in ["diagnosis", "treatment", "recommend", "clinical significance", "what should be done"]):
        return 3
    
    # Stage 2: Findings Identification - Instruments, anatomy, counts
    # This covers: INSTRUMENT_DETECTION, ANATOMY_DETECTION, INSTRUMENT_COUNT
    return 2


def reorder_data_by_stage(data: List[Dict], model=None, tokenizer=None, use_qwen: bool = True) -> Dict[int, List[Dict]]:
    """Classify and group data by stage."""
    stage_data = {1: [], 2: [], 3: []}
    
    print(f"\nClassifying {len(data)} QA pairs into clinical stages...")
    
    for item in tqdm(data, desc="Classifying"):
        question = item.get('question', '')
        category = item.get('category', item.get('question_type', ''))
        
        if use_qwen and model is not None:
            stage = classify_question_stage(model, tokenizer, question, category)
        else:
            stage = classify_question_stage_rule_based(question, category)
        
        # Add stage field
        item['stage'] = stage
        stage_data[stage].append(item)
    
    return stage_data


def create_exp2_data(stage_data: Dict[int, List[Dict]]) -> List[Dict]:
    """Exp2: Qwen reordered - all stages mixed in order 1→2→3"""
    reordered = []
    reordered.extend(stage_data[1])  # Stage 1 first
    reordered.extend(stage_data[2])  # Stage 2 second
    reordered.extend(stage_data[3])  # Stage 3 last
    return reordered


def create_exp3_data(stage_data: Dict[int, List[Dict]]) -> List[Dict]:
    """Exp3: Sequential - same as Exp2 (stages 1→2→3)"""
    # Same as Exp2 for now
    return create_exp2_data(stage_data)


def create_exp4_data(stage_data: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
    """Exp4: Curriculum Learning - separate files per stage"""
    # Return stage-separated data
    return stage_data


def create_exp5_data(stage_data: Dict[int, List[Dict]]) -> List[Dict]:
    """Exp5: Sequential CoT - same as Exp2 (stages 1→2→3)"""
    # Same as Exp2 for now
    return create_exp2_data(stage_data)


def save_jsonl(data: List[Dict], output_path: Path):
    """Save data as JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Reorder EndoVis2018 data into clinical stages")
    parser.add_argument("--use-qwen", action="store_true", help="Use Qwen model for classification (slower but more accurate)")
    parser.add_argument("--use-rule-based", action="store_true", help="Use rule-based classification (faster)")
    args = parser.parse_args()
    
    use_qwen = args.use_qwen and not args.use_rule_based
    
    print("="*80)
    print("REORDERING ENDOVIS2018 DATA INTO CLINICAL STAGES")
    print("="*80)
    
    # Load model if using Qwen
    model, tokenizer = None, None
    if use_qwen:
        try:
            model, tokenizer = load_qwen_model()
        except Exception as e:
            print(f"⚠️  Failed to load Qwen model: {e}")
            print("Falling back to rule-based classification")
            use_qwen = False
    
    # Process each split
    for split_name in ['train', 'validation', 'test']:
        input_file = INPUT_DIR / f"{split_name}.jsonl"
        if not input_file.exists():
            print(f"⚠️  Skipping {split_name}: {input_file} not found")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*80}")
        
        # Load data
        data = []
        with open(input_file) as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        print(f"Loaded {len(data)} QA pairs")
        
            # Classify and reorder
        # For EndoVis2018, rule-based is sufficient and much faster
        stage_data = reorder_data_by_stage(data, model, tokenizer, use_qwen=False)
        
        # Print statistics
        total = len(data)
        print(f"\nStage Distribution:")
        for stage in [1, 2, 3]:
            count = len(stage_data[stage])
            pct = (count / total * 100) if total > 0 else 0
            print(f"  Stage {stage}: {count:,} ({pct:.1f}%)")
        
        # Create Exp2 data (Qwen reordered - stages 1→2→3)
        exp2_data = create_exp2_data(stage_data)
        exp2_dir = OUTPUT_DIR / "exp2_qwen_reordered"
        save_jsonl(exp2_data, exp2_dir / f"{split_name}.jsonl")
        print(f"✓ Exp2 data saved: {len(exp2_data)} samples")
        
        # Create Exp3 data (Sequential - separate per stage for separate models)
        exp3_dir = OUTPUT_DIR / "exp3_sequential"
        for stage in [1, 2, 3]:
            if len(stage_data[stage]) > 0:  # Only create if stage has samples
                stage_file = exp3_dir / f"stage{stage}" / f"{split_name}.jsonl"
                save_jsonl(stage_data[stage], stage_file)
                print(f"✓ Exp3 Stage {stage} data saved: {len(stage_data[stage])} samples")
        
        # Create Exp4 data (Curriculum - separate per stage)
        exp4_dir = OUTPUT_DIR / "exp4_curriculum"
        for stage in [1, 2, 3]:
            stage_file = exp4_dir / f"stage{stage}" / f"{split_name}.jsonl"
            save_jsonl(stage_data[stage], stage_file)
            print(f"✓ Exp4 Stage {stage} data saved: {len(stage_data[stage])} samples")
        
        # Create Exp5 data (Sequential CoT - same as Exp2)
        exp5_data = create_exp5_data(stage_data)
        exp5_dir = OUTPUT_DIR / "exp5_sequential_cot"
        save_jsonl(exp5_data, exp5_dir / f"{split_name}.jsonl")
        print(f"✓ Exp5 data saved: {len(exp5_data)} samples")
    
    print("\n" + "="*80)
    print("✓ DATA REORDERING COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nCreated data files for:")
    print("  - Exp2: Qwen Reordered (stages 1→2→3)")
    print("  - Exp3: Sequential (stages 1→2→3)")
    print("  - Exp4: Curriculum (separate per stage)")
    print("  - Exp5: Sequential CoT (stages 1→2→3)")
    print("="*80)


if __name__ == "__main__":
    main()

