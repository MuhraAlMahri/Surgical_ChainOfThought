#!/usr/bin/env python3
"""
Hybrid Prompt Builder for Surgical CoT Generation

This module implements a hybrid few-shot/zero-shot approach:
- Zero-shot mode: For datasets WITH Q&A (Kvasir-VQA)
- Few-shot mode: For datasets WITHOUT Q&A (Surg-396k, EgoSurgery, etc.)
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from .few_shot_engine import FewShotEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridPromptBuilder:
    """Builds clinical CoT prompts using hybrid few-shot/zero-shot approach."""
    
    def __init__(self, config_path: str, manifest_path: str):
        """Initialize with CoT template configuration and data manifest."""
        # Load the CoT template config, not the paths config
        with open('configs/cot_template.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize few-shot engine
        self.few_shot_engine = FewShotEngine(manifest_path)
        
        self.system_prompt = self.config['system_prompt']
        self.clinical_flow = self.config['clinical_reasoning_flow']
        self.output_format = self.config['output_format']
        
        # Build the base prompt structure
        self._build_base_prompt()
    
    def _build_base_prompt(self) -> None:
        """Build the base prompt structure from the hierarchical template."""
        self.base_prompt = f"{self.system_prompt}\n\n"
        
        # Add the clinical reasoning flow as sequential questions
        self.base_prompt += "**Clinical Reasoning Flow - Answer each question in sequence:**\n\n"
        
        for i, stage in enumerate(self.clinical_flow, 1):
            stage_name = stage['stage']
            questions = stage['questions']
            
            self.base_prompt += f"**{i}. {stage_name}:**\n"
            for j, question in enumerate(questions, 1):
                self.base_prompt += f"   {j}. {question}\n"
            self.base_prompt += "\n"
    
    def _determine_mode(self, sample: Dict[str, Any]) -> str:
        """Determine if sample should use few-shot or zero-shot mode."""
        dataset_name = sample['dataset_name']
        
        # Zero-shot mode for datasets WITH Q&A pairs
        if dataset_name == 'kvasirvqa' and sample.get('vqa_question') and sample.get('vqa_answer'):
            return 'zero_shot'
        
        # Few-shot mode for datasets WITHOUT Q&A pairs
        return 'few_shot'
    
    def _build_zero_shot_prompt(self, sample: Dict[str, Any]) -> str:
        """Build zero-shot prompt for datasets with existing Q&A (Kvasir-VQA)."""
        prompt = self.base_prompt
        
        # Add dataset context
        context = self._get_dataset_context(sample['dataset_name'])
        if context:
            prompt += f"\n**Dataset Context:** {context}\n"
        
        # Add the specific question from the dataset
        if 'vqa_question' in sample and sample['vqa_question']:
            prompt += f"\n**Question to Answer:** {sample['vqa_question']}\n"
        
        # Add output format instructions
        prompt += f"""
**Output Format Requirements:**
- Answer each question in the clinical reasoning flow sequentially
- Use clear, clinical language appropriate for surgical documentation
- Be specific and detailed in your observations
- Include anatomical landmarks and precise descriptions
- Provide clinical reasoning for your assessments
- Finally, provide a concise answer to the specific question asked

**Image to Analyze:** Please analyze the surgical image provided and follow the clinical reasoning flow to answer the question.
"""
        
        return prompt
    
    def _build_few_shot_prompt(self, sample: Dict[str, Any], n_examples: int = 3) -> str:
        """Build few-shot prompt for datasets without Q&A (Surg-396k, EgoSurgery, etc.)."""
        prompt = self.base_prompt
        
        # Add dataset context
        context = self._get_dataset_context(sample['dataset_name'])
        if context:
            prompt += f"\n**Dataset Context:** {context}\n"
        
        # Get few-shot examples from Kvasir-VQA
        examples = self.few_shot_engine.get_examples_for_dataset(
            sample['dataset_name'], n_examples
        )
        
        if examples:
            prompt += f"\n**Few-Shot Examples (from Kvasir-VQA dataset):**\n\n"
            
            for i, example in enumerate(examples, 1):
                prompt += f"**Example {i}:**\n"
                prompt += f"Question: {example['question']}\n"
                prompt += f"Reasoning:\n{example['reasoning']}\n"
                prompt += f"Answer: {example['answer']}\n\n"
        
        # Add the target task
        prompt += f"""
**Your Task:**
Now analyze the provided surgical image following the same structured approach as the examples above.

**Question:** Describe the surgical scene and actions. What do you observe in this image?

**Instructions:**
- Follow the clinical reasoning flow step by step
- Use the same detailed, structured approach as shown in the examples
- Provide a comprehensive analysis of the surgical scene
- End with a clear, concise answer about what you observe

**Image to Analyze:** Please analyze the surgical image provided.
"""
        
        return prompt
    
    def _get_dataset_context(self, dataset_name: str) -> str:
        """Get dataset-specific context information."""
        contexts = {
            'cataract1k': 'Ophthalmic surgery videos showing cataract procedures',
            'egosurgery': 'Egocentric open surgery images from first-person perspective',
            'heichole': 'Laparoscopic cholecystectomy videos with surgical instruments',
            'kvasirvqa': 'Endoscopic images with visual question-answering pairs',
            'pitvis': 'Endoscopic neurosurgery videos showing pituitary procedures',
            'surg396k': 'General endoscopic surgery images from various procedures'
        }
        return contexts.get(dataset_name, 'Surgical procedure images')
    
    def build_prompt(self, sample: Dict[str, Any], include_image: bool = True, n_examples: int = 3) -> str:
        """
        Build a complete prompt for a given sample using hybrid approach.
        
        Args:
            sample: Sample dictionary from data manifest
            include_image: Whether to include image instruction
            n_examples: Number of few-shot examples to include (for few-shot mode)
            
        Returns:
            Complete prompt string
        """
        mode = self._determine_mode(sample)
        logger.info(f"Building {mode} prompt for {sample['dataset_name']} sample {sample['sample_id']}")
        
        if mode == 'zero_shot':
            prompt = self._build_zero_shot_prompt(sample)
        else:
            prompt = self._build_few_shot_prompt(sample, n_examples)
        
        # Add final instructions
        prompt += f"\n{self.output_format}\n"
        
        return prompt
    
    def format_response(self, response: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format the model response into structured output."""
        # Extract the reasoning and answer from the response
        lines = response.strip().split('\n')
        
        # Initialize structured output
        formatted = {
            'sample_id': sample['sample_id'],
            'dataset_name': sample['dataset_name'],
            'image_path': sample['image_path'],
            'timestamp': None,
            'clinical_reasoning': {
                'unstructured': {
                    'title': 'Complete Analysis',
                    'content': response
                }
            },
            'surgical_note': None,
            'raw_response': response
        }
        
        # Try to extract structured stages (basic implementation)
        stages = {}
        current_stage = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('**') and line.endswith('**'):
                # Save previous stage
                if current_stage and current_content:
                    stages[current_stage] = '\n'.join(current_content)
                
                # Start new stage
                current_stage = line[2:-2].lower().replace(' ', '_').replace('&', 'and')
                current_content = []
            elif current_stage and line:
                current_content.append(line)
        
        # Save last stage
        if current_stage and current_content:
            stages[current_stage] = '\n'.join(current_content)
        
        # Add structured stages if found
        if stages:
            formatted['clinical_reasoning']['structured'] = stages
        
        return formatted
    
    def validate_response(self, formatted_response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the formatted response for completeness and quality."""
        required_stages = [
            'basic_assessment',
            'findings_recognition', 
            'localization_&_description',
            'synthesis_&_correlation',
            'temporal_analysis_if_applicable',
            'prognosis_&_risk',
            'conclusion_&_recommendation'
        ]
        
        validation = {
            'is_valid': True,
            'missing_stages': [],
            'missing_questions': [],
            'quality_score': 0.0,
            'issues': []
        }
        
        # Check for structured stages
        if 'structured' in formatted_response['clinical_reasoning']:
            stages = formatted_response['clinical_reasoning']['structured']
            missing_stages = [stage for stage in required_stages if stage not in stages]
            validation['missing_stages'] = missing_stages
            
            if missing_stages:
                validation['is_valid'] = False
                validation['issues'].append(f"Missing stages: {', '.join(missing_stages)}")
        
        # Check for surgical note
        if not formatted_response.get('surgical_note'):
            validation['issues'].append("Missing surgical note summary")
        
        # Calculate quality score
        if validation['missing_stages']:
            validation['quality_score'] = (len(required_stages) - len(validation['missing_stages'])) / len(required_stages)
        else:
            validation['quality_score'] = 1.0
        
        return validation
