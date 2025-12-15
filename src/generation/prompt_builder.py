#!/usr/bin/env python3
"""
Prompt Builder for Surgical CoT Generation

This module builds prompts that force the VLM to reason through each clinical stage
following the structured template defined in configs/cot_template.yaml.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptBuilder:
    """Builds clinical CoT prompts for surgical vision analysis using hierarchical question-driven structure."""
    
    def __init__(self, config_path: str):
        """Initialize with CoT template configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.system_prompt = self.config['system_prompt']
        self.clinical_flow = self.config['clinical_reasoning_flow']
        self.few_shot_examples = self.config['few_shot_examples']
        self.output_format = self.config['output_format']
        self.final_instruction = self.config['final_instruction']
        
        # Build the base prompt structure
        self._build_base_prompt()
    
    def _build_base_prompt(self) -> None:
        """Build the base prompt structure from the new hierarchical template."""
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
        
        # Add output format instructions
        self.base_prompt += f"""
**Output Format Requirements:**
- Answer each question sequentially in the order presented
- Use clear, clinical language appropriate for surgical documentation
- Be specific and detailed in your observations
- Include anatomical landmarks and precise descriptions
- Provide clinical reasoning for your assessments
- Follow the hierarchical structure shown in the example below

**Example Analysis:**
"""
        
        # Add few-shot example
        if 'kvasir_vqa' in self.few_shot_examples and self.few_shot_examples['kvasir_vqa']:
            example = self.few_shot_examples['kvasir_vqa'][0]
            self.base_prompt += f"""
Question: {example['question']}

Answer:
{example['answer']}

Now analyze the provided surgical image following this same structured approach.
"""
    
    def build_prompt(self, sample: Dict, include_image: bool = True) -> str:
        """Build a complete prompt for a given sample."""
        
        # Start with base prompt
        prompt = self.base_prompt
        
        # Add sample-specific context
        dataset_name = sample['dataset_name']
        sample_id = sample['sample_id']
        
        # Add dataset-specific context
        context = self._get_dataset_context(dataset_name)
        if context:
            prompt += f"\n**Dataset Context:** {context}\n"
        
        # Add VQA context if available (for Kvasir-VQA)
        if 'vqa_question' in sample and sample['vqa_question']:
            prompt += f"\n**Original Question:** {sample['vqa_question']}\n"
            if 'vqa_answer' in sample and sample['vqa_answer']:
                prompt += f"**Reference Answer:** {sample['vqa_answer']}\n"
        
        # Add image instruction
        if include_image:
            prompt += f"\n**Image to Analyze:** Please analyze the surgical image provided.\n"
        else:
            prompt += f"\n**Image Path:** {sample['image_path']}\n"
            prompt += f"**Note:** Load and analyze this image following the clinical reasoning framework above.\n"
        
        # Add final instruction
        prompt += f"\n**Your Analysis:**\n"
        
        return prompt
    
    def _get_dataset_context(self, dataset_name: str) -> str:
        """Get dataset-specific context information."""
        context_map = {
            'cataract1k': "Ophthalmic surgery videos showing cataract procedures with clear anatomical structures and surgical instruments.",
            'egosurgery': "Egocentric open surgery images captured from first-person perspective during various open surgical procedures.",
            'heichole': "Laparoscopic cholecystectomy videos showing gallbladder removal procedures with endoscopic views.",
            'kvasirvqa': "Endoscopy images with associated VQA pairs, focusing on gastrointestinal tract examination and pathology identification.",
            'pitvis': "Endoscopic neurosurgery videos of pituitary surgery procedures showing delicate brain anatomy and surgical techniques.",
            'surg396k': "General endoscopic surgery images covering various laparoscopic and endoscopic procedures across different anatomical regions."
        }
        return context_map.get(dataset_name, "Surgical image requiring clinical analysis.")
    
    def build_batch_prompts(self, samples: List[Dict]) -> List[Dict]:
        """Build prompts for a batch of samples."""
        batch_prompts = []
        
        for sample in samples:
            prompt = self.build_prompt(sample, include_image=False)
            batch_prompts.append({
                'sample_id': sample['sample_id'],
                'dataset_name': sample['dataset_name'],
                'image_path': sample['image_path'],
                'prompt': prompt,
                'metadata': {
                    'split': sample['split'],
                    'original_path': sample['original_path'],
                    'vqa_question': sample.get('vqa_question'),
                    'vqa_answer': sample.get('vqa_answer')
                }
            })
        
        return batch_prompts
    
    def format_response(self, response: str, sample: Dict) -> Dict:
        """Format the model response according to the new hierarchical structure."""
        
        # Parse the response to extract each stage and question
        formatted_response = {
            'sample_id': sample['sample_id'],
            'dataset_name': sample['dataset_name'],
            'image_path': sample['image_path'],
            'timestamp': None,  # Will be set by generation script
            'clinical_reasoning': {},
            'surgical_note': None
        }
        
        # Try to parse the structured response
        lines = response.strip().split('\n')
        current_stage = None
        current_question = None
        current_content = []
        question_counter = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new stage
            for i, stage in enumerate(self.clinical_flow, 1):
                stage_name = stage['stage']
                
                if (line.startswith(f"**{i}. {stage_name}**") or 
                    line.startswith(f"{i}. {stage_name}") or
                    stage_name.lower() in line.lower()):
                    
                    # Save previous stage if exists
                    if current_stage and current_content:
                        self._save_stage_content(formatted_response, current_stage, current_question, current_content)
                    
                    # Start new stage
                    current_stage = stage_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
                    current_question = None
                    current_content = [line]
                    question_counter = 0
                    break
            
            # Check if this line starts a new question within a stage
            if current_stage and line.startswith('- '):
                # Save previous question if exists
                if current_question and current_content:
                    self._save_question_content(formatted_response, current_stage, current_question, current_content)
                
                # Start new question
                question_counter += 1
                current_question = f"Question {question_counter}"
                current_content = [line]
            elif current_stage:
                current_content.append(line)
        
        # Save final stage/question
        if current_stage and current_content:
            if current_question:
                self._save_question_content(formatted_response, current_stage, current_question, current_content)
            else:
                self._save_stage_content(formatted_response, current_stage, current_question, current_content)
        
        # Extract surgical note if present
        surgical_note = self._extract_surgical_note(response)
        if surgical_note:
            formatted_response['surgical_note'] = surgical_note
        
        # If no structured parsing worked, store as raw response
        if not formatted_response['clinical_reasoning']:
            formatted_response['raw_response'] = response
            formatted_response['clinical_reasoning']['unstructured'] = {
                'title': 'Complete Analysis',
                'content': response
            }
        
        return formatted_response
    
    def _save_stage_content(self, formatted_response: Dict, stage: str, question: str, content: List[str]) -> None:
        """Save stage content to formatted response."""
        if stage not in formatted_response['clinical_reasoning']:
            formatted_response['clinical_reasoning'][stage] = {
                'stage_name': stage.replace('_', ' ').title(),
                'questions': {}
            }
        
        if question:
            formatted_response['clinical_reasoning'][stage]['questions'][question] = {
                'question': question,
                'answer': '\n'.join(content).strip()
            }
        else:
            formatted_response['clinical_reasoning'][stage]['content'] = '\n'.join(content).strip()
    
    def _save_question_content(self, formatted_response: Dict, stage: str, question: str, content: List[str]) -> None:
        """Save question content to formatted response."""
        if stage not in formatted_response['clinical_reasoning']:
            formatted_response['clinical_reasoning'][stage] = {
                'stage_name': stage.replace('_', ' ').title(),
                'questions': {}
            }
        
        formatted_response['clinical_reasoning'][stage]['questions'][question] = {
            'question': question,
            'answer': '\n'.join(content).strip()
        }
    
    def _extract_surgical_note(self, response: str) -> Optional[str]:
        """Extract surgical note summary from response."""
        lines = response.strip().split('\n')
        in_surgical_note = False
        surgical_note_lines = []
        
        for line in lines:
            line = line.strip()
            if 'surgical note' in line.lower() or 'summary:' in line.lower():
                in_surgical_note = True
                # Extract the content after the colon
                if ':' in line:
                    note_content = line.split(':', 1)[1].strip()
                    if note_content:
                        surgical_note_lines.append(note_content)
                continue
            elif in_surgical_note and (line.startswith('**') or line.startswith('#')):
                break
            elif in_surgical_note and line:
                surgical_note_lines.append(line)
        
        if surgical_note_lines:
            return '\n'.join(surgical_note_lines).strip()
        return None
    
    def validate_response(self, response: Dict) -> Dict:
        """Validate that the response follows the expected hierarchical structure."""
        validation_result = {
            'is_valid': True,
            'missing_stages': [],
            'missing_questions': [],
            'quality_score': 0.0,
            'issues': []
        }
        
        # Get expected stages and questions from clinical flow
        expected_stages = [stage['stage'].lower().replace(' ', '_').replace('(', '').replace(')', '') 
                          for stage in self.clinical_flow]
        found_stages = list(response.get('clinical_reasoning', {}).keys())
        
        # Check for missing stages
        for stage in expected_stages:
            if stage not in found_stages:
                validation_result['missing_stages'].append(stage)
                validation_result['is_valid'] = False
        
        # Check for missing questions within each stage
        total_expected_questions = 0
        total_found_questions = 0
        
        for stage in self.clinical_flow:
            stage_key = stage['stage'].lower().replace(' ', '_').replace('(', '').replace(')', '')
            expected_questions = stage['questions']
            total_expected_questions += len(expected_questions)
            
            if stage_key in found_stages:
                stage_data = response['clinical_reasoning'][stage_key]
                found_questions = list(stage_data.get('questions', {}).keys())
                total_found_questions += len(found_questions)
                
                # Check for missing questions in this stage (more flexible matching)
                for question in expected_questions:
                    # Extract key terms from the question for matching
                    question_keywords = set(question.lower().split())
                    found_match = False
                    
                    # Check against both question keys and answer content
                    for found_q in found_questions:
                        found_keywords = set(found_q.lower().split())
                        # Check if there's significant overlap in keywords
                        overlap = len(question_keywords.intersection(found_keywords))
                        if overlap >= 3:  # At least 3 common keywords
                            found_match = True
                            break
                        
                        # Also check the answer content for keywords
                        if 'questions' in stage_data and found_q in stage_data['questions']:
                            answer_content = stage_data['questions'][found_q].get('answer', '').lower()
                            answer_keywords = set(answer_content.split())
                            overlap = len(question_keywords.intersection(answer_keywords))
                            if overlap >= 3:  # At least 3 common keywords
                                found_match = True
                                break
                    
                    if not found_match:
                        validation_result['missing_questions'].append(f"{stage['stage']}: {question}")
        
        # Calculate quality score based on completeness
        stage_completeness = len(found_stages) / len(expected_stages) if expected_stages else 0
        question_completeness = total_found_questions / total_expected_questions if total_expected_questions else 0
        validation_result['quality_score'] = (stage_completeness + question_completeness) / 2
        
        # Check for minimum content length
        total_content_length = 0
        for stage_data in response.get('clinical_reasoning', {}).values():
            if 'content' in stage_data:
                total_content_length += len(stage_data['content'])
            for question_data in stage_data.get('questions', {}).values():
                total_content_length += len(question_data.get('answer', ''))
        
        if total_content_length < 200:
            validation_result['issues'].append("Response too short")
            validation_result['quality_score'] *= 0.7
        
        # Check for surgical note
        if not response.get('surgical_note'):
            validation_result['issues'].append("Missing surgical note summary")
            validation_result['quality_score'] *= 0.9
        
        if validation_result['missing_stages']:
            validation_result['issues'].append(f"Missing stages: {', '.join(validation_result['missing_stages'])}")
        
        if validation_result['missing_questions']:
            validation_result['issues'].append(f"Missing questions: {len(validation_result['missing_questions'])} total")
        
        return validation_result

def main():
    """Test the PromptBuilder functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PromptBuilder")
    parser.add_argument("--config", default="configs/cot_template.yaml",
                       help="Path to CoT template configuration")
    parser.add_argument("--sample", help="Path to sample JSON file for testing")
    
    args = parser.parse_args()
    
    # Initialize prompt builder
    builder = PromptBuilder(args.config)
    
    if args.sample:
        # Test with specific sample
        with open(args.sample, 'r') as f:
            sample = json.load(f)
        
        prompt = builder.build_prompt(sample)
        print("Generated Prompt:")
        print("="*80)
        print(prompt)
        print("="*80)
    else:
        # Test with dummy sample
        dummy_sample = {
            'sample_id': 'test_sample_001',
            'dataset_name': 'kvasirvqa',
            'image_path': '/path/to/test/image.jpg',
            'split': 'train',
            'vqa_question': 'What is visible in this endoscopic image?',
            'vqa_answer': 'Normal duodenal mucosa'
        }
        
        prompt = builder.build_prompt(dummy_sample)
        print("Generated Prompt (Dummy Sample):")
        print("="*80)
        print(prompt)
        print("="*80)

if __name__ == "__main__":
    main()
