#!/usr/bin/env python3
"""
LLM Judge Script for Automatic Evaluation of Generated CoT Data

This script uses a powerful LLM (GPT-4-Turbo) to automatically evaluate
the quality of generated CoT against clinical criteria.
"""

import json
import argparse
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import time
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMJudge:
    """Automatic evaluation using LLM Judge (GPT-4-Turbo)."""
    
    def __init__(self, config_path: str, api_key: str = None):
        """Initialize the LLM Judge."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass --api-key")
        
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-4-turbo-preview"
        self.max_retries = 3
        self.rate_limit_delay = 1.0  # seconds between requests
        
        # Evaluation criteria
        self.criteria = {
            'clinical_accuracy': {
                'description': 'Are the anatomical observations and surgical assessments medically accurate?',
                'weight': 0.25
            },
            'completeness': {
                'description': 'Does the analysis cover all 6 required clinical stages comprehensively?',
                'weight': 0.20
            },
            'relevance': {
                'description': 'Is the analysis relevant to the specific surgical context and image?',
                'weight': 0.20
            },
            'structure': {
                'description': 'Does it follow the prescribed clinical reasoning template structure?',
                'weight': 0.15
            },
            'language_quality': {
                'description': 'Is the language clear, professional, and clinically appropriate?',
                'weight': 0.10
            },
            'clinical_utility': {
                'description': 'How useful would this analysis be for surgical training or decision-making?',
                'weight': 0.10
            }
        }
    
    def create_evaluation_prompt(self, sample: Dict) -> str:
        """Create evaluation prompt for a single sample."""
        prompt = f"""You are an expert surgical AI evaluator. Please evaluate the following generated clinical Chain-of-Thought analysis for a surgical image.

**Sample Information:**
- Dataset: {sample['dataset_name']}
- Sample ID: {sample['sample_id']}
- Image Path: {sample['image_path']}

**Generated Analysis:**
"""
        
        # Add the generated analysis
        if 'stages' in sample:
            for stage_key, stage_data in sample['stages'].items():
                if isinstance(stage_data, dict) and 'title' in stage_data:
                    prompt += f"\n**{stage_data['title']}:**\n{stage_data['content']}\n"
                else:
                    prompt += f"\n**{stage_key}:**\n{stage_data}\n"
        else:
            prompt += f"\n{sample.get('raw_response', 'No analysis generated')}\n"
        
        # Add VQA context if available
        if 'metadata' in sample and sample['metadata'].get('vqa_question'):
            prompt += f"\n**Original Question:** {sample['metadata']['vqa_question']}\n"
            if sample['metadata'].get('vqa_answer'):
                prompt += f"**Reference Answer:** {sample['metadata']['vqa_answer']}\n"
        
        prompt += f"""

**Evaluation Criteria:**
Please rate each criterion on a scale of 1-5 (1=Poor, 5=Excellent) and provide a brief justification:

"""
        
        for criterion, info in self.criteria.items():
            prompt += f"- **{criterion.replace('_', ' ').title()}** (Weight: {info['weight']}): {info['description']}\n"
        
        prompt += f"""

**Output Format:**
Please provide your evaluation in the following JSON format:
{{
    "clinical_accuracy": {{
        "score": <1-5>,
        "justification": "<brief explanation>"
    }},
    "completeness": {{
        "score": <1-5>,
        "justification": "<brief explanation>"
    }},
    "relevance": {{
        "score": <1-5>,
        "justification": "<brief explanation>"
    }},
    "structure": {{
        "score": <1-5>,
        "justification": "<brief explanation>"
    }},
    "language_quality": {{
        "score": <1-5>,
        "justification": "<brief explanation>"
    }},
    "clinical_utility": {{
        "score": <1-5>,
        "justification": "<brief explanation>"
    }},
    "overall_score": <weighted average>,
    "summary": "<overall assessment>"
}}

Please ensure your response is valid JSON."""
        
        return prompt
    
    async def evaluate_single_sample(self, session: aiohttp.ClientSession, sample: Dict) -> Dict:
        """Evaluate a single sample using the LLM Judge."""
        prompt = self.create_evaluation_prompt(sample)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert surgical AI evaluator with deep knowledge of clinical reasoning, surgical procedures, and medical imaging analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        for attempt in range(self.max_retries):
            try:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        
                        # Parse JSON response
                        try:
                            evaluation = json.loads(content)
                            
                            # Calculate weighted overall score
                            overall_score = sum(
                                evaluation[criterion]['score'] * self.criteria[criterion]['weight']
                                for criterion in self.criteria.keys()
                                if criterion in evaluation
                            )
                            evaluation['overall_score'] = round(overall_score, 2)
                            
                            return {
                                'sample_id': sample['sample_id'],
                                'dataset_name': sample['dataset_name'],
                                'evaluation': evaluation,
                                'timestamp': datetime.now().isoformat(),
                                'status': 'success'
                            }
                            
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON response for {sample['sample_id']}: {e}")
                            return {
                                'sample_id': sample['sample_id'],
                                'dataset_name': sample['dataset_name'],
                                'error': f"JSON parsing failed: {e}",
                                'raw_response': content,
                                'timestamp': datetime.now().isoformat(),
                                'status': 'error'
                            }
                    
                    elif response.status == 429:  # Rate limit
                        wait_time = self.rate_limit_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"API error {response.status} for {sample['sample_id']}: {error_text}")
                        return {
                            'sample_id': sample['sample_id'],
                            'dataset_name': sample['dataset_name'],
                            'error': f"API error {response.status}: {error_text}",
                            'timestamp': datetime.now().isoformat(),
                            'status': 'error'
                        }
            
            except Exception as e:
                logger.error(f"Request failed for {sample['sample_id']} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.rate_limit_delay * (2 ** attempt))
                else:
                    return {
                        'sample_id': sample['sample_id'],
                        'dataset_name': sample['dataset_name'],
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                        'status': 'error'
                    }
        
        return {
            'sample_id': sample['sample_id'],
            'dataset_name': sample['dataset_name'],
            'error': 'Max retries exceeded',
            'timestamp': datetime.now().isoformat(),
            'status': 'error'
        }
    
    async def evaluate_batch(self, samples: List[Dict], batch_size: int = 10) -> List[Dict]:
        """Evaluate a batch of samples asynchronously."""
        results = []
        
        async with aiohttp.ClientSession() as session:
            # Process in batches to avoid overwhelming the API
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                logger.info(f"Evaluating batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size}")
                
                # Create tasks for concurrent evaluation
                tasks = [self.evaluate_single_sample(session, sample) for sample in batch]
                
                # Wait for batch completion
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Task failed with exception: {result}")
                        results.append({
                            'sample_id': 'unknown',
                            'dataset_name': 'unknown',
                            'error': str(result),
                            'timestamp': datetime.now().isoformat(),
                            'status': 'error'
                        })
                    else:
                        results.append(result)
                
                # Rate limiting delay between batches
                if i + batch_size < len(samples):
                    await asyncio.sleep(self.rate_limit_delay)
        
        return results
    
    def load_generated_cot(self, input_file: str) -> List[Dict]:
        """Load generated CoT data from JSONL file."""
        samples = []
        
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        
        logger.info(f"Loaded {len(samples)} samples for evaluation")
        return samples
    
    def calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate evaluation statistics."""
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            return {'error': 'No successful evaluations'}
        
        # Extract scores
        scores = {}
        for criterion in self.criteria.keys():
            scores[criterion] = [r['evaluation'][criterion]['score'] for r in successful_results if criterion in r['evaluation']]
        
        # Calculate statistics
        stats = {
            'total_samples': len(results),
            'successful_evaluations': len(successful_results),
            'failed_evaluations': len(results) - len(successful_results),
            'success_rate': len(successful_results) / len(results) if results else 0,
            'criterion_scores': {}
        }
        
        for criterion, score_list in scores.items():
            if score_list:
                stats['criterion_scores'][criterion] = {
                    'mean': sum(score_list) / len(score_list),
                    'min': min(score_list),
                    'max': max(score_list),
                    'std': (sum((x - sum(score_list)/len(score_list))**2 for x in score_list) / len(score_list))**0.5
                }
        
        # Overall score statistics
        overall_scores = [r['evaluation']['overall_score'] for r in successful_results if 'overall_score' in r['evaluation']]
        if overall_scores:
            stats['overall_score'] = {
                'mean': sum(overall_scores) / len(overall_scores),
                'min': min(overall_scores),
                'max': max(overall_scores),
                'std': (sum((x - sum(overall_scores)/len(overall_scores))**2 for x in overall_scores) / len(overall_scores))**0.5
            }
        
        return stats
    
    async def evaluate_dataset(self, input_file: str, output_file: str, max_samples: int = None) -> None:
        """Evaluate the entire dataset."""
        logger.info("Starting LLM Judge evaluation...")
        
        # Load samples
        samples = self.load_generated_cot(input_file)
        
        if max_samples:
            samples = samples[:max_samples]
            logger.info(f"Limited to {max_samples} samples for evaluation")
        
        # Filter out failed samples
        valid_samples = [s for s in samples if 'error' not in s]
        logger.info(f"Evaluating {len(valid_samples)} valid samples")
        
        # Evaluate samples
        results = await self.evaluate_batch(valid_samples)
        
        # Calculate statistics
        stats = self.calculate_statistics(results)
        
        # Create output
        output_data = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'model_used': self.model,
                'total_samples': len(samples),
                'evaluated_samples': len(valid_samples),
                'successful_evaluations': len([r for r in results if r['status'] == 'success'])
            },
            'statistics': stats,
            'evaluations': results
        }
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Evaluation complete! Results saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("LLM JUDGE EVALUATION SUMMARY")
        print("="*60)
        print(f"Total samples: {len(samples)}")
        print(f"Evaluated: {len(valid_samples)}")
        print(f"Successful: {stats['successful_evaluations']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        
        if 'overall_score' in stats:
            print(f"Overall score: {stats['overall_score']['mean']:.2f} ± {stats['overall_score']['std']:.2f}")
        
        print("="*60)

def main():
    """Main function to run LLM Judge evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate CoT data using LLM Judge")
    parser.add_argument("--input", required=True,
                       help="Path to generated CoT JSONL file")
    parser.add_argument("--output", required=True,
                       help="Path to output evaluation results file")
    parser.add_argument("--config", default="configs/paths.yaml",
                       help="Path to configuration file")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to evaluate")
    
    args = parser.parse_args()
    
    # Initialize judge
    judge = LLMJudge(args.config, args.api_key)
    
    # Run evaluation
    asyncio.run(judge.evaluate_dataset(args.input, args.output, args.max_samples))

if __name__ == "__main__":
    import os
    main()
