#!/usr/bin/env python3
"""
Example Usage of Multi-Head Temporal CoT System
Demonstrates how to use the system for training and inference.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.question_categorizer import QuestionCategorizer
from data.temporal_linker import TemporalLinker
from models.multi_head_model import create_model
from prompts.cot_templates import build_cot_prompt, build_stage_dependent_prompt
import json


def example_question_categorization():
    """Example: Categorize questions into clinical stages."""
    print("=" * 80)
    print("Example 1: Question Categorization")
    print("=" * 80)
    
    # Sample questions
    questions = [
        {"question": "Is there a polyp?", "category": "detection"},
        {"question": "What is the color of the polyp?", "category": "characteristics"},
        {"question": "What treatment is recommended?", "category": "treatment"}
    ]
    
    # Create categorizer
    categorizer = QuestionCategorizer(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        use_cache=True
    )
    
    # Categorize
    for qa in questions:
        result = categorizer.classify_question(
            question=qa["question"],
            category=qa.get("category")
        )
        print(f"\nQuestion: {qa['question']}")
        print(f"Stage: {result['stage']} ({result['category']})")
        print(f"Confidence: {result['confidence']}")


def example_temporal_structure():
    """Example: Create temporal structure for video sequences."""
    print("\n" + "=" * 80)
    print("Example 2: Temporal Structure Creation")
    print("=" * 80)
    
    # Sample sequence data
    sequence_dir = Path("/path/to/sequences")
    qa_pairs = [
        {
            "frame_id": "frame_001",
            "question": "What instruments are present?",
            "answer": "scissors, forceps"
        },
        {
            "frame_id": "frame_002",
            "question": "How many instruments are there?",
            "answer": "2"
        }
    ]
    
    # Create temporal linker
    linker = TemporalLinker(sequence_dir)
    
    # Create temporal structure
    temporal_data = linker.create_temporal_structure(
        sequence_id="seq_1",
        qa_pairs=qa_pairs,
        compute_motion=True
    )
    
    print(f"\nCreated temporal structure for {len(temporal_data)} frames")
    if temporal_data:
        print(f"First frame: {temporal_data[0]['frame_id']}")
        if temporal_data[0].get('motion_info'):
            print(f"Motion description: {temporal_data[0]['motion_info'].get('description')}")


def example_cot_prompt():
    """Example: Build CoT prompts."""
    print("\n" + "=" * 80)
    print("Example 3: CoT Prompt Generation")
    print("=" * 80)
    
    # Example 1: Basic prompt
    prompt1 = build_cot_prompt(
        question="What is the color of the polyp?",
        category="characteristics"
    )
    print("\n--- Basic Prompt ---")
    print(prompt1)
    
    # Example 2: With temporal context
    previous_frame_info = {
        "observations": {"polyp_detected": "yes", "location": "upper left"},
        "motion_description": "Camera moved closer to lesion"
    }
    
    prompt2 = build_cot_prompt(
        question="What is the color of the polyp?",
        category="characteristics",
        previous_frame_info=previous_frame_info
    )
    print("\n--- Prompt with Temporal Context ---")
    print(prompt2)
    
    # Example 3: Stage-dependent prompt
    previous_stage_predictions = {
        1: {"polyp_detected": "yes", "location": "upper left quadrant"}
    }
    
    prompt3 = build_stage_dependent_prompt(
        question="What is the color of the polyp?",
        stage=2,
        previous_stage_predictions=previous_stage_predictions
    )
    print("\n--- Stage-Dependent Prompt ---")
    print(prompt3)


def example_model_creation():
    """Example: Create and use the multi-head model."""
    print("\n" + "=" * 80)
    print("Example 4: Model Creation")
    print("=" * 80)
    
    # Create model
    model = create_model(
        base_model_name="Qwen/Qwen2-VL-2B-Instruct",
        use_lora=True,
        lora_r=8,
        lora_alpha=16
    )
    
    print(f"\nModel created: {model.base_model_name}")
    print(f"Has vision: {model.has_vision}")
    print(f"Hidden dimension: {model.hidden_dim}")
    print(f"Vocab size: {model.vocab_size}")
    
    # Example generation (would need actual image and tokenization)
    print("\nModel ready for training/inference")


def example_training_workflow():
    """Example: Complete training workflow."""
    print("\n" + "=" * 80)
    print("Example 5: Training Workflow")
    print("=" * 80)
    
    workflow = """
    1. Categorize questions:
       python data/question_categorizer.py \\
           --input datasets/Kvasir-VQA/train.json \\
           --output data/categorized \\
           --dataset kvasir
    
    2. Create temporal structure (EndoVis only):
       python data/temporal_linker.py \\
           --sequence-dir datasets/EndoVis2018/sequences \\
           --qa-file datasets/EndoVis2018/train.json \\
           --output data/temporal_structure.json \\
           --sequence-id seq_1
    
    3. Train model:
       python train_multihead_temporal_cot.py \\
           --base-model Qwen/Qwen2-VL-2B-Instruct \\
           --train-data data/categorized/train_categorized.json \\
           --val-data data/categorized/val_categorized.json \\
           --image-base-path datasets/Kvasir-VQA/raw/images \\
           --dataset kvasir \\
           --training-mode unified \\
           --num-epochs 10 \\
           --batch-size 4 \\
           --use-lora \\
           --output-dir checkpoints/kvasir_unified
    
    4. Evaluate:
       python evaluation/baseline_comparison.py \\
           --model-path checkpoints/kvasir_unified/best_model.pt \\
           --test-data data/categorized/test_categorized.json \\
           --output results/kvasir_evaluation
    """
    
    print(workflow)


if __name__ == "__main__":
    print("Multi-Head Temporal CoT System - Usage Examples")
    print("=" * 80)
    
    # Run examples
    try:
        example_question_categorization()
    except Exception as e:
        print(f"\nExample 1 failed (may need model download): {e}")
    
    try:
        example_temporal_structure()
    except Exception as e:
        print(f"\nExample 2 failed (may need data): {e}")
    
    example_cot_prompt()
    
    try:
        example_model_creation()
    except Exception as e:
        print(f"\nExample 4 failed (may need model download): {e}")
    
    example_training_workflow()
    
    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)














