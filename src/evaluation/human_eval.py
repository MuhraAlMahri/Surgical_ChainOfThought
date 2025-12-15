#!/usr/bin/env python3
"""
Human Evaluation Script for Generated CoT Data

This script randomly samples generated CoT pairs and formats them
into a readable format for manual review by experts.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HumanEvaluator:
    """Creates human evaluation samples for manual review."""
    
    def __init__(self, input_file: str, output_file: str, sample_size: int = 100):
        """Initialize the human evaluator."""
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.sample_size = sample_size
        
        # Create output directory
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def load_generated_cot(self) -> List[Dict]:
        """Load generated CoT data from JSONL file."""
        samples = []
        
        with open(self.input_file, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        logger.info(f"Loaded {len(samples)} generated CoT samples")
        return samples
    
    def sample_for_evaluation(self, samples: List[Dict]) -> List[Dict]:
        """Randomly sample samples for human evaluation."""
        # Filter out failed samples
        valid_samples = [s for s in samples if 'error' not in s]
        
        if len(valid_samples) < self.sample_size:
            logger.warning(f"Only {len(valid_samples)} valid samples available, using all")
            return valid_samples
        
        # Stratified sampling by dataset
        dataset_samples = {}
        for sample in valid_samples:
            dataset = sample['dataset_name']
            if dataset not in dataset_samples:
                dataset_samples[dataset] = []
            dataset_samples[dataset].append(sample)
        
        # Sample proportionally from each dataset
        selected_samples = []
        for dataset, dataset_list in dataset_samples.items():
            n_samples = max(1, int(self.sample_size * len(dataset_list) / len(valid_samples)))
            selected = random.sample(dataset_list, min(n_samples, len(dataset_list)))
            selected_samples.extend(selected)
        
        # If we need more samples, add randomly
        if len(selected_samples) < self.sample_size:
            remaining = [s for s in valid_samples if s not in selected_samples]
            additional = random.sample(remaining, min(self.sample_size - len(selected_samples), len(remaining)))
            selected_samples.extend(additional)
        
        # Shuffle and limit to sample_size
        random.shuffle(selected_samples)
        return selected_samples[:self.sample_size]
    
    def format_sample_for_review(self, sample: Dict, index: int) -> str:
        """Format a single sample for human review."""
        content = f"""
## Sample {index + 1}: {sample['sample_id']}

**Dataset:** {sample['dataset_name']}
**Image Path:** {sample['image_path']}
**Generated At:** {sample.get('timestamp', 'Unknown')}

### Original Context
"""
        
        # Add VQA context if available
        if 'metadata' in sample and sample['metadata'].get('vqa_question'):
            content += f"**Original Question:** {sample['metadata']['vqa_question']}\n"
            if sample['metadata'].get('vqa_answer'):
                content += f"**Reference Answer:** {sample['metadata']['vqa_answer']}\n"
        
        content += f"""
### Generated Clinical CoT Analysis
"""
        
        # Format the generated analysis using new hierarchical structure
        if 'clinical_reasoning' in sample:
            for stage_key, stage_data in sample['clinical_reasoning'].items():
                if isinstance(stage_data, dict):
                    stage_name = stage_data.get('stage_name', stage_key.replace('_', ' ').title())
                    content += f"\n**{stage_name}:**\n"
                    
                    # Format questions and answers
                    if 'questions' in stage_data and stage_data['questions']:
                        for question, qa_data in stage_data['questions'].items():
                            if isinstance(qa_data, dict):
                                content += f"- **{qa_data.get('question', question)}**\n"
                                content += f"  {qa_data.get('answer', 'No answer provided')}\n"
                            else:
                                content += f"- **{question}:** {qa_data}\n"
                    elif 'content' in stage_data:
                        content += f"{stage_data['content']}\n"
                else:
                    content += f"\n**{stage_key}:**\n{stage_data}\n"
        elif 'stages' in sample:
            # Fallback for old format
            for stage_key, stage_data in sample['stages'].items():
                if isinstance(stage_data, dict) and 'title' in stage_data:
                    content += f"\n**{stage_data['title']}:**\n{stage_data['content']}\n"
                else:
                    content += f"\n**{stage_key}:**\n{stage_data}\n"
        else:
            content += f"\n{sample.get('raw_response', 'No analysis generated')}\n"
        
        # Add surgical note if available
        if 'surgical_note' in sample and sample['surgical_note']:
            content += f"""
### Surgical Note Summary
{sample['surgical_note']}
"""
        
        # Add validation info if available
        if 'validation' in sample:
            validation = sample['validation']
            content += f"""
### Quality Assessment
- **Valid Structure:** {'Yes' if validation.get('is_valid', False) else 'No'}
- **Quality Score:** {validation.get('quality_score', 0):.2f}
- **Missing Stages:** {', '.join(validation.get('missing_stages', []))}
- **Missing Questions:** {len(validation.get('missing_questions', []))} total
- **Issues:** {', '.join(validation.get('issues', []))}
"""
        
        content += "\n" + "="*80 + "\n"
        return content
    
    def create_evaluation_template(self, samples: List[Dict]) -> str:
        """Create the complete evaluation template."""
        content = f"""# Human Evaluation of Generated Surgical CoT Data

**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Samples:** {len(samples)}
**Source File:** {self.input_file}

## Evaluation Instructions

Please review each generated clinical Chain-of-Thought analysis and rate it on the following criteria:

### Rating Scale (1-5)
- **1:** Poor - Incorrect, irrelevant, or missing critical information
- **2:** Below Average - Some correct elements but significant issues
- **3:** Average - Generally correct but missing some important details
- **4:** Good - Correct and comprehensive with minor issues
- **5:** Excellent - Highly accurate, clinically relevant, and well-structured

### Evaluation Criteria

1. **Clinical Accuracy:** Are the anatomical observations and surgical assessments correct?
2. **Completeness:** Does the analysis cover all 7 required clinical stages and answer all questions?
3. **Relevance:** Is the analysis relevant to the specific surgical context?
4. **Structure:** Does it follow the prescribed hierarchical question-driven template?
5. **Language Quality:** Is the language clear, professional, and clinically appropriate?
6. **Sequential Reasoning:** Does the analysis build logically from basic assessment to final recommendation?
7. **Surgical Note Quality:** Is the final surgical note summary concise and clinically useful?

### Sample Reviews

"""
        
        # Add each sample
        for i, sample in enumerate(samples):
            content += self.format_sample_for_review(sample, i)
        
        # Add evaluation form
        content += """
## Evaluation Form

For each sample, please provide:

| Sample | Clinical Accuracy | Completeness | Relevance | Structure | Language | Sequential Reasoning | Surgical Note | Overall | Notes |
|--------|------------------|--------------|-----------|-----------|----------|---------------------|---------------|---------|-------|
"""
        
        for i in range(len(samples)):
            content += f"| {i+1} | ___/5 | ___/5 | ___/5 | ___/5 | ___/5 | ___/5 | ___/5 | ___/5 | |\n"
        
        content += """
## Summary Questions

1. **Overall Quality:** What is your overall assessment of the generated CoT data?
2. **Strengths:** What are the main strengths of the generated analyses?
3. **Weaknesses:** What are the main areas for improvement?
4. **Clinical Utility:** How useful would this data be for training surgical AI systems?
5. **Recommendations:** What specific improvements would you recommend?

## Additional Comments

[Space for additional feedback and suggestions]

---
*This evaluation was generated automatically by the Surgical CoT pipeline.*
"""
        
        return content
    
    def create_evaluation_samples(self) -> None:
        """Create human evaluation samples."""
        logger.info("Creating human evaluation samples...")
        
        # Load generated CoT data
        samples = self.load_generated_cot()
        
        if not samples:
            logger.error("No samples found in input file")
            return
        
        # Sample for evaluation
        eval_samples = self.sample_for_evaluation(samples)
        
        # Create evaluation template
        evaluation_content = self.create_evaluation_template(eval_samples)
        
        # Save to file
        with open(self.output_file, 'w') as f:
            f.write(evaluation_content)
        
        logger.info(f"Human evaluation samples saved to: {self.output_file}")
        logger.info(f"Created {len(eval_samples)} samples for review")
        
        # Print summary
        print("\n" + "="*60)
        print("HUMAN EVALUATION SAMPLES CREATED")
        print("="*60)
        print(f"Output file: {self.output_file}")
        print(f"Sample count: {len(eval_samples)}")
        print(f"File size: {self.output_file.stat().st_size / 1024:.1f} KB")
        print("="*60)

def main():
    """Main function to create human evaluation samples."""
    parser = argparse.ArgumentParser(description="Create human evaluation samples")
    parser.add_argument("--input", required=True,
                       help="Path to generated CoT JSONL file")
    parser.add_argument("--output", default="data/evaluation/human_eval_sample.md",
                       help="Path to output evaluation file")
    parser.add_argument("--sample-size", type=int, default=100,
                       help="Number of samples to include in evaluation")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create evaluator
    evaluator = HumanEvaluator(args.input, args.output, args.sample_size)
    
    # Create evaluation samples
    evaluator.create_evaluation_samples()

if __name__ == "__main__":
    main()
