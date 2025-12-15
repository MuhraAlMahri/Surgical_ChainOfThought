#!/usr/bin/env python3
"""
Generate final results table from all evaluation results.
Matches the user's desired format.
"""

import json
from pathlib import Path
from typing import Dict, List
import argparse


def load_all_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all evaluation results from directory."""
    all_results = {}
    
    for result_file in results_dir.glob("evaluation_all_configs_*.json"):
        with open(result_file, 'r') as f:
            data = json.load(f)
            all_results.update(data)
    
    return all_results


def generate_table(results: Dict[str, Dict]) -> str:
    """Generate markdown table in the user's exact format."""
    # Organize by model and dataset
    organized = {}
    
    for key, result in results.items():
        if '_cot_zeroshot' in key:
            base_key = key.replace('_cot_zeroshot', '')
            if base_key not in organized:
                organized[base_key] = {}
            organized[base_key]['cot_zeroshot'] = result
        else:
            if key not in organized:
                organized[key] = {}
            organized[key]['no_cot'] = result
    
    # Model name mapping
    model_names = {
        'qwen3vl': 'Qwen3-vl-8b-instruct',
        'medgemma': 'MedGemma-4B',
        'llava_med': 'LLaVA-Med v1.5 (Mistral-7B)'
    }
    
    dataset_names = {
        'kvasir': 'Kavisr',
        'endovis': 'Endovis18'
    }
    
    # Generate table
    table_lines = []
    table_lines.append("| Model | dataset | COT | Zeroshot | Instruction fine tuning |")
    table_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    for key in sorted(organized.keys()):
        model_type, dataset = key.split('_', 1)
        model_name = model_names.get(model_type, model_type)
        dataset_name = dataset_names.get(dataset, dataset)
        data = organized[key]
        
        # Row 1: No CoT
        no_cot = data.get('no_cot', {})
        zeroshot_val = no_cot.get('zeroshot')
        finetuned_val = no_cot.get('finetuned')
        
        if zeroshot_val is not None:
            correct = no_cot.get('zeroshot_correct', 0)
            total = no_cot.get('zeroshot_total', 0)
            zeroshot_str = f"{zeroshot_val:.2f}% ({correct:,}/{total:,})"
        else:
            zeroshot_str = ""
        
        if finetuned_val is not None:
            correct = no_cot.get('finetuned_correct', 0)
            total = no_cot.get('finetuned_total', 0)
            finetuned_str = f"{finetuned_val:.2f}% ({correct:,}/{total:,})"
        else:
            finetuned_str = ""
        
        table_lines.append(f"| {model_name} | {dataset_name} | no | {zeroshot_str} | {finetuned_str} |")
        
        # Row 2: CoT yes
        cot_zeroshot = data.get('cot_zeroshot', {})
        cot_finetuned = data.get('no_cot', {}).get('cot_finetuned')
        
        if cot_zeroshot.get('zeroshot') is not None:
            correct = cot_zeroshot.get('zeroshot_correct', 0)
            total = cot_zeroshot.get('zeroshot_total', 0)
            cot_zeroshot_str = f"{cot_zeroshot['zeroshot']:.2f}% ({correct:,}/{total:,})"
        else:
            cot_zeroshot_str = ""
        
        if cot_finetuned is not None:
            correct = data.get('no_cot', {}).get('cot_finetuned_correct', 0)
            total = data.get('no_cot', {}).get('cot_finetuned_total', 0)
            cot_finetuned_str = f"{cot_finetuned:.2f}% ({correct:,}/{total:,})"
        else:
            cot_finetuned_str = ""
        
        table_lines.append(f"| {model_name} | {dataset_name} | yes | {cot_zeroshot_str} | {cot_finetuned_str} |")
    
    return "\n".join(table_lines)


def main():
    parser = argparse.ArgumentParser(description="Generate results table")
    parser.add_argument("--results-dir", required=True, help="Directory containing evaluation results")
    parser.add_argument("--output", required=True, help="Output markdown file")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Load all results
    results = load_all_results(results_dir)
    
    if not results:
        print(f"No results found in {results_dir}")
        return
    
    # Generate table
    table = generate_table(results)
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Evaluation Results Table\n\n")
        f.write(table)
        f.write("\n")
    
    print("Generated results table:")
    print("\n" + table)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()


