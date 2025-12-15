#!/usr/bin/env python3
"""
Analysis Script for Surgical CoT Evaluation Results

This script consolidates results from human and automatic evaluations
and produces summary statistics and visualizations.
"""

import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoTAnalyzer:
    """Analyzes and visualizes CoT generation and evaluation results."""
    
    def __init__(self, config_path: str):
        """Initialize the analyzer."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(self.config['output']['evaluation_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        if HAS_SEABORN:
            plt.style.use('seaborn-v0_8')
        else:
            plt.style.use('default')
    
    def load_generated_cot(self, cot_file: str) -> List[Dict]:
        """Load generated CoT data."""
        samples = []
        with open(cot_file, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples
    
    def load_llm_judge_results(self, judge_file: str) -> Dict:
        """Load LLM Judge evaluation results."""
        with open(judge_file, 'r') as f:
            return json.load(f)
    
    def load_human_eval_results(self, human_file: str) -> Optional[Dict]:
        """Load human evaluation results (if available)."""
        if not Path(human_file).exists():
            logger.warning(f"Human evaluation file not found: {human_file}")
            return None
        
        # For now, we'll assume human evaluation is in markdown format
        # In a real implementation, you'd parse the filled evaluation form
        return {'status': 'pending', 'file': human_file}
    
    def analyze_generation_quality(self, samples: List[Dict]) -> Dict:
        """Analyze the quality of generated CoT data."""
        analysis = {
            'total_samples': len(samples),
            'successful_generations': len([s for s in samples if 'error' not in s]),
            'failed_generations': len([s for s in samples if 'error' in s]),
            'dataset_breakdown': {},
            'quality_metrics': {}
        }
        
        # Dataset breakdown
        for sample in samples:
            dataset = sample['dataset_name']
            if dataset not in analysis['dataset_breakdown']:
                analysis['dataset_breakdown'][dataset] = {'total': 0, 'successful': 0, 'failed': 0}
            
            analysis['dataset_breakdown'][dataset]['total'] += 1
            if 'error' in sample:
                analysis['dataset_breakdown'][dataset]['failed'] += 1
            else:
                analysis['dataset_breakdown'][dataset]['successful'] += 1
        
        # Quality metrics for successful generations
        successful_samples = [s for s in samples if 'error' not in s and 'validation' in s]
        
        if successful_samples:
            quality_scores = [s['validation']['quality_score'] for s in successful_samples]
            completeness_rates = [1.0 if s['validation']['is_valid'] else 0.0 for s in successful_samples]
            
            analysis['quality_metrics'] = {
                'average_quality_score': np.mean(quality_scores),
                'quality_score_std': np.std(quality_scores),
                'completeness_rate': np.mean(completeness_rates),
                'high_quality_samples': len([s for s in quality_scores if s >= 0.8]),
                'low_quality_samples': len([s for s in quality_scores if s < 0.5])
            }
        
        return analysis
    
    def analyze_llm_judge_results(self, judge_results: Dict) -> Dict:
        """Analyze LLM Judge evaluation results."""
        if 'statistics' not in judge_results:
            return {'error': 'No statistics found in judge results'}
        
        stats = judge_results['statistics']
        analysis = {
            'evaluation_metadata': judge_results.get('metadata', {}),
            'overall_performance': {},
            'criterion_analysis': {},
            'dataset_performance': {}
        }
        
        # Overall performance
        if 'overall_score' in stats:
            analysis['overall_performance'] = {
                'mean_score': stats['overall_score']['mean'],
                'std_score': stats['overall_score']['std'],
                'min_score': stats['overall_score']['min'],
                'max_score': stats['overall_score']['max'],
                'grade_distribution': self._calculate_grade_distribution(stats['overall_score']['mean'])
            }
        
        # Criterion analysis
        if 'criterion_scores' in stats:
            for criterion, scores in stats['criterion_scores'].items():
                analysis['criterion_analysis'][criterion] = {
                    'mean': scores['mean'],
                    'std': scores['std'],
                    'strength_level': self._assess_strength_level(scores['mean'])
                }
        
        # Dataset performance (if available)
        evaluations = judge_results.get('evaluations', [])
        if evaluations:
            dataset_scores = {}
            for eval_result in evaluations:
                if eval_result['status'] == 'success' and 'overall_score' in eval_result['evaluation']:
                    dataset = eval_result['dataset_name']
                    if dataset not in dataset_scores:
                        dataset_scores[dataset] = []
                    dataset_scores[dataset].append(eval_result['evaluation']['overall_score'])
            
            for dataset, scores in dataset_scores.items():
                analysis['dataset_performance'][dataset] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'sample_count': len(scores)
                }
        
        return analysis
    
    def _calculate_grade_distribution(self, mean_score: float) -> str:
        """Calculate grade distribution based on mean score."""
        if mean_score >= 4.5:
            return "Excellent (A+)"
        elif mean_score >= 4.0:
            return "Very Good (A)"
        elif mean_score >= 3.5:
            return "Good (B+)"
        elif mean_score >= 3.0:
            return "Satisfactory (B)"
        elif mean_score >= 2.5:
            return "Below Average (C)"
        else:
            return "Poor (D/F)"
    
    def _assess_strength_level(self, mean_score: float) -> str:
        """Assess strength level for a criterion."""
        if mean_score >= 4.0:
            return "Strong"
        elif mean_score >= 3.0:
            return "Moderate"
        elif mean_score >= 2.0:
            return "Weak"
        else:
            return "Very Weak"
    
    def create_visualizations(self, generation_analysis: Dict, judge_analysis: Dict) -> None:
        """Create visualization plots for the analysis."""
        
        # 1. Generation Success Rate by Dataset
        if 'dataset_breakdown' in generation_analysis:
            datasets = list(generation_analysis['dataset_breakdown'].keys())
            success_rates = []
            
            for dataset in datasets:
                breakdown = generation_analysis['dataset_breakdown'][dataset]
                success_rate = breakdown['successful'] / breakdown['total'] if breakdown['total'] > 0 else 0
                success_rates.append(success_rate)
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(datasets, success_rates, color='skyblue', alpha=0.7)
            plt.title('CoT Generation Success Rate by Dataset', fontsize=14, fontweight='bold')
            plt.xlabel('Dataset', fontsize=12)
            plt.ylabel('Success Rate', fontsize=12)
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'generation_success_by_dataset.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. LLM Judge Criterion Scores
        if 'criterion_analysis' in judge_analysis and judge_analysis['criterion_analysis']:
            criteria = list(judge_analysis['criterion_analysis'].keys())
            scores = [judge_analysis['criterion_analysis'][c]['mean'] for c in criteria]
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(criteria, scores, color='lightcoral', alpha=0.7)
            plt.title('LLM Judge Evaluation Scores by Criterion', fontsize=14, fontweight='bold')
            plt.xlabel('Mean Score (1-5)', fontsize=12)
            plt.ylabel('Evaluation Criterion', fontsize=12)
            plt.xlim(0, 5)
            
            # Add value labels
            for bar, score in zip(bars, scores):
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{score:.2f}', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'llm_judge_criterion_scores.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Dataset Performance Comparison
        if 'dataset_performance' in judge_analysis and judge_analysis['dataset_performance']:
            datasets = list(judge_analysis['dataset_performance'].keys())
            mean_scores = [judge_analysis['dataset_performance'][d]['mean_score'] for d in datasets]
            std_scores = [judge_analysis['dataset_performance'][d]['std_score'] for d in datasets]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(datasets, mean_scores, yerr=std_scores, capsize=5, 
                          color='lightgreen', alpha=0.7, error_kw={'elinewidth': 2})
            plt.title('Dataset Performance in LLM Judge Evaluation', fontsize=14, fontweight='bold')
            plt.xlabel('Dataset', fontsize=12)
            plt.ylabel('Mean Score (1-5)', fontsize=12)
            plt.ylim(0, 5)
            
            # Add value labels
            for bar, score in zip(bars, mean_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'dataset_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def generate_final_report(self, generation_analysis: Dict, judge_analysis: Dict, 
                            human_eval_status: Optional[Dict]) -> str:
        """Generate a comprehensive final analysis report."""
        
        report = f"""# Surgical CoT Dataset Generation and Evaluation Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents the results of generating and evaluating a novel surgical Vision Chain-of-Thought (CoT) dataset using Qwen2.5-VL-72B-Instruct on AMD MI210 GPUs.

### Key Findings

"""
        
        # Add generation results
        if generation_analysis:
            total_samples = generation_analysis['total_samples']
            successful = generation_analysis['successful_generations']
            success_rate = successful / total_samples if total_samples > 0 else 0
            
            report += f"""
**Generation Results:**
- Total samples processed: {total_samples:,}
- Successful generations: {successful:,} ({success_rate:.1%})
- Failed generations: {generation_analysis['failed_generations']:,}
"""
            
            if 'quality_metrics' in generation_analysis:
                qm = generation_analysis['quality_metrics']
                report += f"""
- Average quality score: {qm['average_quality_score']:.2f}
- Completeness rate: {qm['completeness_rate']:.1%}
- High quality samples (≥0.8): {qm['high_quality_samples']:,}
- Low quality samples (<0.5): {qm['low_quality_samples']:,}
"""
        
        # Add evaluation results
        if judge_analysis and 'overall_performance' in judge_analysis:
            op = judge_analysis['overall_performance']
            report += f"""

**Evaluation Results:**
- Overall mean score: {op['mean_score']:.2f} ± {op['std_score']:.2f}
- Grade distribution: {op['grade_distribution']}
- Score range: {op['min_score']:.2f} - {op['max_score']:.2f}
"""
        
        # Add detailed analysis
        report += f"""

## Detailed Analysis

### 1. Dataset Performance

"""
        
        if 'dataset_breakdown' in generation_analysis:
            report += "**Generation Success by Dataset:**\n\n"
            for dataset, breakdown in generation_analysis['dataset_breakdown'].items():
                success_rate = breakdown['successful'] / breakdown['total'] if breakdown['total'] > 0 else 0
                report += f"- **{dataset}**: {breakdown['successful']}/{breakdown['total']} ({success_rate:.1%})\n"
        
        if 'dataset_performance' in judge_analysis:
            report += "\n**Evaluation Performance by Dataset:**\n\n"
            for dataset, perf in judge_analysis['dataset_performance'].items():
                report += f"- **{dataset}**: {perf['mean_score']:.2f} ± {perf['std_score']:.2f} (n={perf['sample_count']})\n"
        
        # Add criterion analysis
        if 'criterion_analysis' in judge_analysis:
            report += f"""

### 2. Evaluation Criteria Analysis

"""
            for criterion, analysis in judge_analysis['criterion_analysis'].items():
                report += f"- **{criterion.replace('_', ' ').title()}**: {analysis['mean']:.2f} ({analysis['strength_level']})\n"
        
        # Add recommendations
        report += f"""

### 3. Recommendations

Based on the analysis results, the following recommendations are made:

1. **Quality Improvement**: Focus on improving the completeness and structure of generated CoT analyses
2. **Dataset Balance**: Ensure balanced representation across all surgical datasets
3. **Clinical Validation**: Conduct thorough clinical validation of the generated analyses
4. **Template Refinement**: Consider refining the clinical CoT template based on evaluation feedback

### 4. Next Steps

1. Review human evaluation results when available
2. Iterate on the generation pipeline based on findings
3. Scale up generation to full dataset
4. Prepare dataset for publication and distribution

## Technical Details

- **Model**: Qwen2.5-VL-72B-Instruct
- **Hardware**: AMD MI210 GPUs
- **Quantization**: 4-bit with BitsAndBytes
- **Evaluation**: GPT-4-Turbo via OpenAI API

---
*This report was generated automatically by the Surgical CoT pipeline.*
"""
        
        return report
    
    def run_analysis(self, cot_file: str, judge_file: str, human_file: str, output_file: str) -> None:
        """Run the complete analysis pipeline."""
        logger.info("Starting comprehensive analysis...")
        
        # Load data
        samples = self.load_generated_cot(cot_file)
        judge_results = self.load_llm_judge_results(judge_file)
        human_eval = self.load_human_eval_results(human_file)
        
        # Analyze generation quality
        generation_analysis = self.analyze_generation_quality(samples)
        
        # Analyze evaluation results
        judge_analysis = self.analyze_llm_judge_results(judge_results)
        
        # Create visualizations
        self.create_visualizations(generation_analysis, judge_analysis)
        
        # Generate final report
        report = self.generate_final_report(generation_analysis, judge_analysis, human_eval)
        
        # Save results
        analysis_results = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'generation_analysis': generation_analysis,
                'judge_analysis': judge_analysis,
                'human_eval_status': human_eval
            },
            'report': report
        }
        
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Save report as markdown
        report_file = self.output_dir / 'final_analysis_report.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Analysis complete! Results saved to: {output_file}")
        logger.info(f"Report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Generation analysis: {len(samples)} samples processed")
        print(f"Evaluation analysis: {judge_results.get('metadata', {}).get('evaluated_samples', 0)} samples evaluated")
        print(f"Visualizations: {len(list(self.output_dir.glob('*.png')))} plots generated")
        print(f"Report: {report_file}")
        print("="*60)

def main():
    """Main function to run analysis."""
    parser = argparse.ArgumentParser(description="Analyze CoT generation and evaluation results")
    parser.add_argument("--generated-cot", required=True,
                       help="Path to generated CoT JSONL file")
    parser.add_argument("--llm-judge", required=True,
                       help="Path to LLM Judge results JSON file")
    parser.add_argument("--human-eval", default="data/evaluation/human_eval_sample.md",
                       help="Path to human evaluation file")
    parser.add_argument("--output", required=True,
                       help="Path to output analysis JSON file")
    parser.add_argument("--config", default="configs/paths.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CoTAnalyzer(args.config)
    
    # Run analysis
    analyzer.run_analysis(args.generated_cot, args.llm_judge, args.human_eval, args.output)

if __name__ == "__main__":
    import yaml
    main()
