#!/usr/bin/env python3
"""
Post-Processing Script: Extract Concise Answers from Verbose VQA Predictions
Handles both single-stage and multi-stage experiments
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip punctuation)."""
    return text.lower().strip(" .,;!?")


def extract_concise_answer(prediction: str, question: str) -> str:
    """
    Extract concise answer from verbose prediction based on question type.
    
    Rules:
    1. Yes/No questions → return "yes" or "no"
    2. Counting questions → extract first digit
    3. Medical term questions → extract first relevant medical term
    4. Fallback → first word
    """
    pred = prediction.lower().strip()
    q = question.lower()
    
    # Rule 1: YES/NO QUESTIONS
    yes_no_keywords = ["is", "are", "was", "were", "does", "do", "did", 
                       "have", "has", "had", "can", "could", "will", "would"]
    if any(q.startswith(kw) or f" {kw} " in q for kw in yes_no_keywords):
        if "yes" in pred[:50]:  # Check first 50 chars for faster detection
            return "yes"
        if "no" in pred[:50]:
            return "no"
    
    # Rule 2: COUNTING QUESTIONS
    if "how many" in q or "number of" in q:
        # Extract first digit
        digits = re.findall(r'\b\d+\b', pred)
        if digits:
            return digits[0]
        
        # Handle word numbers
        word_to_digit = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10"
        }
        for word, digit in word_to_digit.items():
            if word in pred:
                return digit
    
    # Rule 3: MEDICAL TERM EXTRACTION
    medical_terms = [
        "polyp", "polyps",
        "ulcer", "ulcers",
        "bleeding", "hemorrhage",
        "instrument", "instruments", "tool", "tools",
        "colon", "colonoscopy",
        "stomach", "gastric",
        "esophagus", "esophageal",
        "text", "label", "metadata",
        "artifact", "artifacts",
        "inflammation", "inflamed",
        "normal", "healthy",
        "abnormality", "abnormalities",
        "lesion", "lesions",
        "tissue"
    ]
    
    # Find first occurrence of any medical term
    for term in medical_terms:
        if term in pred:
            # Return singular form
            if term.endswith("s") and term[:-1] in medical_terms:
                return term[:-1]
            return term
    
    # Rule 4: COLOR EXTRACTION (for questions like "what color")
    if "color" in q or "colour" in q:
        colors = ["red", "pink", "white", "yellow", "orange", "brown", "green", "blue", "black"]
        for color in colors:
            if color in pred:
                return color
    
    # Rule 5: FALLBACK - First alphanumeric word
    words = re.findall(r'\b\w+\b', pred)
    if words:
        return words[0]
    
    return pred.strip()


def process_predictions(predictions: List[Dict]) -> Tuple[List[Dict], float]:
    """
    Process predictions: extract concise answers and compute accuracy.
    
    Returns:
        Tuple of (enhanced_predictions, accuracy)
    """
    enhanced_predictions = []
    correct_count = 0
    total_count = 0
    
    for item in predictions:
        question = item.get('question', '')
        ground_truth = item.get('ground_truth', '')
        prediction = item.get('prediction', '')
        
        # Extract concise answer
        concise_answer = extract_concise_answer(prediction, question)
        
        # Normalize for comparison
        concise_norm = normalize_text(concise_answer)
        gt_norm = normalize_text(ground_truth)
        
        # Check if correct
        is_correct = (concise_norm == gt_norm) or (concise_norm in gt_norm) or (gt_norm in concise_norm)
        
        if is_correct:
            correct_count += 1
        total_count += 1
        
        # Create enhanced entry
        enhanced_entry = {
            **item,  # Keep original fields
            'prediction_concise': concise_answer,
            'correct_concise': is_correct
        }
        
        enhanced_predictions.append(enhanced_entry)
    
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
    
    return enhanced_predictions, accuracy


def process_single_stage_experiment(exp_name: str, results_dir: Path) -> Dict:
    """Process single-stage experiment (exp1, exp2, exp4)."""
    
    # Try multiple possible filenames
    possible_files = [
        f"{exp_name}_evaluation_results.json",
        f"{exp_name}_predictions.json",
        f"{exp_name}_corrected_evaluation_results.json"
    ]
    
    predictions = None
    filepath = None
    
    for filename in possible_files:
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Extract predictions based on file structure
            if isinstance(data, dict):
                if 'predictions' in data:
                    predictions = data['predictions']
                elif 'all_predictions' in data:
                    predictions = data['all_predictions']
            elif isinstance(data, list):
                predictions = data
            
            if predictions:
                break
    
    if not predictions:
        return None
    
    # Process predictions
    enhanced_predictions, accuracy = process_predictions(predictions)
    
    # Save enhanced results
    output_file = results_dir / f"{exp_name}_predictions_concise.json"
    with open(output_file, 'w') as f:
        json.dump({
            'experiment': exp_name,
            'total': len(enhanced_predictions),
            'correct': sum(1 for p in enhanced_predictions if p['correct_concise']),
            'accuracy': accuracy,
            'predictions': enhanced_predictions
        }, f, indent=2)
    
    return {
        'experiment': exp_name,
        'total': len(enhanced_predictions),
        'correct': sum(1 for p in enhanced_predictions if p['correct_concise']),
        'accuracy': accuracy,
        'output_file': str(output_file)
    }


def process_multi_stage_experiment(exp_name: str, results_dir: Path) -> Dict:
    """Process multi-stage experiment (exp3, exp5)."""
    
    # Stage sample counts for weighted average
    stage_weights = {
        1: 3275,
        2: 5703,
        3: 6
    }
    
    stage_results = {}
    all_enhanced_predictions = []
    total_correct = 0
    total_samples = 0
    
    for stage in [1, 2, 3]:
        # Try multiple possible filenames
        possible_files = [
            f"{exp_name}_stage{stage}_evaluation_results.json",
            f"{exp_name}_stage{stage}_predictions.json",
            f"{exp_name}_corrected_evaluation_results.json"  # Might contain all stages
        ]
        
        predictions = None
        
        for filename in possible_files:
            filepath = results_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Extract predictions for this stage
                if isinstance(data, dict):
                    # Check if it's a multi-stage result file
                    if 'all_predictions' in data:
                        # Filter by stage
                        all_preds = data['all_predictions']
                        predictions = [p for p in all_preds if p.get('stage') == stage]
                    elif 'predictions' in data:
                        predictions = data['predictions']
                elif isinstance(data, list):
                    predictions = data
                
                if predictions:
                    break
        
        if not predictions:
            print(f"  Warning: No predictions found for {exp_name} Stage {stage}")
            continue
        
        # Process predictions
        enhanced_predictions, accuracy = process_predictions(predictions)
        
        # Save stage-specific results
        output_file = results_dir / f"{exp_name}_stage{stage}_predictions_concise.json"
        with open(output_file, 'w') as f:
            json.dump({
                'experiment': exp_name,
                'stage': stage,
                'total': len(enhanced_predictions),
                'correct': sum(1 for p in enhanced_predictions if p['correct_concise']),
                'accuracy': accuracy,
                'predictions': enhanced_predictions
            }, f, indent=2)
        
        stage_results[stage] = {
            'total': len(enhanced_predictions),
            'correct': sum(1 for p in enhanced_predictions if p['correct_concise']),
            'accuracy': accuracy,
            'output_file': str(output_file)
        }
        
        all_enhanced_predictions.extend(enhanced_predictions)
        total_correct += sum(1 for p in enhanced_predictions if p['correct_concise'])
        total_samples += len(enhanced_predictions)
    
    # Compute weighted overall accuracy
    weighted_accuracy = 0.0
    total_weight = sum(stage_weights.values())
    
    for stage, results in stage_results.items():
        weight = stage_weights[stage]
        weighted_accuracy += (results['accuracy'] * weight / total_weight)
    
    # Save combined results
    combined_output = results_dir / f"{exp_name}_all_stages_predictions_concise.json"
    with open(combined_output, 'w') as f:
        json.dump({
            'experiment': exp_name,
            'by_stage': stage_results,
            'overall': {
                'total': total_samples,
                'correct': total_correct,
                'accuracy': (total_correct / total_samples * 100) if total_samples > 0 else 0.0,
                'weighted_accuracy': weighted_accuracy
            },
            'all_predictions': all_enhanced_predictions
        }, f, indent=2)
    
    return {
        'experiment': exp_name,
        'by_stage': stage_results,
        'overall_accuracy': (total_correct / total_samples * 100) if total_samples > 0 else 0.0,
        'weighted_accuracy': weighted_accuracy,
        'output_file': str(combined_output)
    }


def main():
    print()
    print("=" * 80)
    print("SURGICAL VQA - CONCISE ANSWER EXTRACTION & ACCURACY COMPUTATION")
    print("=" * 80)
    print()
    
    # Set paths
    results_dir = Path("/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments/results")
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print(f"Results directory: {results_dir}")
    print()
    
    all_results = {}
    
    # Process single-stage experiments
    print("=" * 80)
    print("SINGLE-STAGE EXPERIMENTS")
    print("=" * 80)
    print()
    
    for exp_name in ['exp1', 'exp2', 'exp4']:
        print(f"Processing {exp_name}...")
        result = process_single_stage_experiment(exp_name, results_dir)
        
        if result:
            all_results[exp_name] = result
            print(f"  ✓ {exp_name}: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
            print(f"    Saved to: {Path(result['output_file']).name}")
        else:
            print(f"  ✗ {exp_name}: No predictions found")
        print()
    
    # Process multi-stage experiments
    print("=" * 80)
    print("MULTI-STAGE EXPERIMENTS")
    print("=" * 80)
    print()
    
    for exp_name in ['exp3', 'exp5']:
        print(f"Processing {exp_name}...")
        result = process_multi_stage_experiment(exp_name, results_dir)
        
        if result:
            all_results[exp_name] = result
            print(f"  ✓ {exp_name} Overall: {result['overall_accuracy']:.2f}%")
            print(f"    Weighted Accuracy: {result['weighted_accuracy']:.2f}%")
            
            if 'by_stage' in result:
                for stage, stage_data in result['by_stage'].items():
                    print(f"    Stage {stage}: {stage_data['accuracy']:.2f}% ({stage_data['correct']}/{stage_data['total']})")
            
            print(f"    Saved to: {Path(result['output_file']).name}")
        else:
            print(f"  ✗ {exp_name}: No predictions found")
        print()
    
    # Final summary
    print("=" * 80)
    print("SUMMARY - CONCISE ANSWER ACCURACY")
    print("=" * 80)
    print()
    
    for exp_name, result in all_results.items():
        if 'accuracy' in result:
            print(f"  {exp_name:6s}: {result['accuracy']:6.2f}%")
        elif 'weighted_accuracy' in result:
            print(f"  {exp_name:6s}: {result['weighted_accuracy']:6.2f}% (weighted)")
    
    print()
    print("=" * 80)
    print("✓ Post-processing complete!")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()




