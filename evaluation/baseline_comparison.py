#!/usr/bin/env python3
"""
Baseline Comparison and Evaluation Scripts
Compares multi-head temporal CoT model against baselines.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineEvaluator:
    """Evaluate model performance and compare with baselines."""
    
    def __init__(
        self,
        model,
        tokenizer,
        processor=None,
        device: str = "cuda"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: The model to evaluate
            tokenizer: Tokenizer
            processor: Optional processor for vision models
            device: Device to use
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.model.eval()
    
    def evaluate_dataset(
        self,
        data_loader,
        dataset_name: str = "kvasir"
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            dataset_name: Name of dataset
            
        Returns:
            Dictionary with metrics
        """
        all_predictions = []
        all_answers = []
        all_categories = []
        all_stages = []
        
        correct_by_stage = defaultdict(int)
        total_by_stage = defaultdict(int)
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                images = batch['images']
                questions = batch['questions']
                answers = batch['answers']
                categories = batch['categories']
                stages = batch['stages']
                temporal_contexts = batch.get('temporal_contexts', [None] * len(images))
                
                for i in range(len(images)):
                    image = images[i]
                    question = questions[i]
                    answer = answers[i]
                    category = categories[i]
                    stage = stages[i]
                    temporal_context = temporal_contexts[i]
                    
                    # Generate prediction
                    try:
                        prediction = self._generate_answer(
                            image,
                            question,
                            category,
                            temporal_context
                        )
                        
                        all_predictions.append(prediction)
                        all_answers.append(answer)
                        all_categories.append(category)
                        all_stages.append(stage)
                        
                        # Check correctness
                        is_correct = self._check_correctness(prediction, answer)
                        if is_correct:
                            correct_by_stage[stage] += 1
                        total_by_stage[stage] += 1
                        
                    except Exception as e:
                        logger.warning(f"Error evaluating sample: {e}")
                        continue
        
        # Compute metrics
        overall_accuracy = sum(correct_by_stage.values()) / max(sum(total_by_stage.values()), 1)
        
        stage_accuracies = {
            stage: correct_by_stage[stage] / max(total_by_stage[stage], 1)
            for stage in [1, 2, 3]
        }
        
        return {
            'overall_accuracy': overall_accuracy,
            'stage_1_accuracy': stage_accuracies[1],
            'stage_2_accuracy': stage_accuracies[2],
            'stage_3_accuracy': stage_accuracies[3],
            'total_samples': sum(total_by_stage.values()),
            'stage_counts': dict(total_by_stage),
            'predictions': all_predictions,
            'answers': all_answers
        }
    
    def _generate_answer(
        self,
        image,
        question: str,
        category: str,
        temporal_context: Optional[Dict]
    ) -> str:
        """Generate answer for a question."""
        # This is simplified - actual implementation needs proper model forward
        # For now, return a placeholder
        if hasattr(self.model, 'generate'):
            result = self.model.generate(
                images=[image],
                prompt=question,
                category=category,
                previous_frame_info=temporal_context
            )
            return result.get('answer', '')
        else:
            # Fallback
            return ""
    
    def _check_correctness(self, prediction: str, answer: str) -> bool:
        """Check if prediction matches answer."""
        # Normalize strings
        pred_lower = prediction.lower().strip()
        ans_lower = answer.lower().strip()
        
        # Exact match
        if pred_lower == ans_lower:
            return True
        
        # Check if answer is contained in prediction (for longer answers)
        if ans_lower in pred_lower or pred_lower in ans_lower:
            return True
        
        # For multi-label answers (semicolon-separated)
        if ';' in ans_lower:
            ans_parts = [a.strip() for a in ans_lower.split(';')]
            pred_parts = [p.strip() for p in pred_lower.split(';')]
            # Check if all answer parts are in prediction
            return all(part in pred_lower for part in ans_parts)
        
        return False
    
    def compare_with_baselines(
        self,
        results: Dict[str, Dict],
        baseline_results: Optional[Dict[str, Dict]] = None
    ) -> Dict:
        """
        Compare results with baseline models.
        
        Args:
            results: Results from current model
            baseline_results: Results from baseline models
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            'current_model': results,
            'baselines': baseline_results or {},
            'improvements': {}
        }
        
        if baseline_results:
            for baseline_name, baseline_result in baseline_results.items():
                improvements = {}
                for metric in ['overall_accuracy', 'stage_1_accuracy', 'stage_2_accuracy', 'stage_3_accuracy']:
                    if metric in results and metric in baseline_result:
                        improvement = results[metric] - baseline_result[metric]
                        improvements[metric] = improvement
                
                comparison['improvements'][baseline_name] = improvements
        
        return comparison


def create_comparison_table(
    results: Dict[str, Dict],
    output_file: Optional[str] = None
) -> str:
    """
    Create a comparison table in markdown format.
    
    Args:
        results: Dictionary mapping model names to results
        output_file: Optional file to save table
        
    Returns:
        Markdown table string
    """
    # Extract metrics
    models = list(results.keys())
    metrics = ['overall_accuracy', 'stage_1_accuracy', 'stage_2_accuracy', 'stage_3_accuracy']
    
    # Create table
    table = "| Model | Overall Acc | Stage 1 Acc | Stage 2 Acc | Stage 3 Acc |\n"
    table += "|-------|-------------|-------------|-------------|-------------|\n"
    
    for model_name in models:
        model_results = results[model_name]
        row = f"| {model_name} |"
        for metric in metrics:
            value = model_results.get(metric, 0.0)
            row += f" {value:.3f} |"
        table += row + "\n"
    
    # Save if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(table)
        logger.info(f"Saved comparison table to {output_file}")
    
    return table


def run_ablation_study(
    model,
    tokenizer,
    data_loaders: Dict[str, any],
    processor=None,
    device: str = "cuda"
) -> Dict[str, Dict]:
    """
    Run ablation study comparing different components.
    
    Args:
        model: Base model
        tokenizer: Tokenizer
        data_loaders: Dictionary with different data loader configurations
        processor: Optional processor
        device: Device
        
    Returns:
        Dictionary with results for each configuration
    """
    results = {}
    
    # 1. Baseline (no CoT, no temporal, no multi-head)
    logger.info("Running baseline (no CoT, no temporal, no multi-head)")
    evaluator = BaselineEvaluator(model, tokenizer, processor, device)
    # results['baseline'] = evaluator.evaluate_dataset(data_loaders['baseline'])
    
    # 2. Temporal only
    logger.info("Running temporal CoT only")
    # results['temporal_only'] = evaluator.evaluate_dataset(data_loaders['temporal'])
    
    # 3. Multi-head only
    logger.info("Running multi-head only")
    # results['multihead_only'] = evaluator.evaluate_dataset(data_loaders['multihead'])
    
    # 4. Full system (temporal + multi-head)
    logger.info("Running full system (temporal + multi-head)")
    # results['full_system'] = evaluator.evaluate_dataset(data_loaders['full'])
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and compare baselines")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--test-data", required=True, help="Path to test data JSON")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--baseline-results", help="Path to baseline results JSON")
    
    args = parser.parse_args()
    
    # Load model (simplified - actual implementation needed)
    # model = load_model(args.model_path)
    
    # Load test data
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    # Run evaluation
    # results = evaluate_model(model, test_data)
    
    # Compare with baselines
    # if args.baseline_results:
    #     with open(args.baseline_results, 'r') as f:
    #         baseline_results = json.load(f)
    #     comparison = compare_with_baselines(results, baseline_results)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Evaluation complete")














