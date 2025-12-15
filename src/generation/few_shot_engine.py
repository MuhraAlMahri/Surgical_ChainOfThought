#!/usr/bin/env python3
"""
Few-Shot Example Engine for Surgical CoT Generation

This module provides few-shot examples from Kvasir-VQA dataset to teach
the model how to generate high-quality CoT for datasets without Q&A pairs.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FewShotEngine:
    """Generates few-shot examples from Kvasir-VQA for teaching CoT generation."""
    
    def __init__(self, manifest_path: str):
        """Initialize with the data manifest."""
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Filter Kvasir-VQA samples that have Q&A pairs
        self.kvasir_samples = [
            sample for sample in self.manifest['samples']
            if sample['dataset_name'] == 'kvasirvqa' 
            and sample.get('vqa_question') is not None
            and sample.get('vqa_answer') is not None
        ]
        
        logger.info(f"Loaded {len(self.kvasir_samples)} Kvasir-VQA samples with Q&A pairs")
    
    def _generate_synthetic_cot(self, question: str, answer: str, sample: Dict[str, Any]) -> str:
        """Generate synthetic CoT reasoning from Q&A pair using 7-stage clinical template."""
        
        # Extract context from sample metadata
        procedure_type = sample.get('procedure_type', 'surgical procedure')
        anatomy = sample.get('anatomy', 'surgical anatomy')
        
        # Generate synthetic CoT based on the question and answer
        if "abnormality" in question.lower() or "pathology" in question.lower():
            cot = f"""**Basic Assessment:** The endoscopic image shows clear visualization of the {anatomy} with adequate lighting and minimal artifacts.

**Findings Recognition:** The primary finding is {answer.lower()}, which is clearly visible in the image.

**Localization & Description:** The {answer.lower()} is located in the {anatomy} and appears to be well-defined with characteristic features typical of this condition.

**Synthesis & Correlation:** This finding is consistent with {answer.lower()} and correlates with the clinical presentation. The appearance suggests a {self._get_risk_assessment(answer)}.

**Temporal Analysis:** The lesion appears to be in a stable state with no signs of recent changes or progression.

**Prognosis & Risk:** The {answer.lower()} has a {self._get_risk_assessment(answer)} risk profile and requires appropriate clinical management.

**Conclusion & Recommendation:** The findings are consistent with {answer.lower()}. Recommended next steps include appropriate follow-up and management based on clinical guidelines."""
        
        elif "visible" in question.lower() or "see" in question.lower():
            cot = f"""**Basic Assessment:** The {procedure_type} image provides excellent visualization of the {anatomy} with clear anatomical landmarks.

**Findings Recognition:** The image clearly shows {answer.lower()} as the primary finding, with good contrast and resolution.

**Localization & Description:** The {answer.lower()} is well-positioned within the {anatomy} and demonstrates characteristic features that are clearly identifiable.

**Synthesis & Correlation:** The visual findings are consistent with {answer.lower()} and align with expected anatomical structures and surgical context.

**Temporal Analysis:** The image captures the current state of the {anatomy} during the {procedure_type} procedure.

**Prognosis & Risk:** The identified {answer.lower()} presents with standard risk characteristics for this type of finding.

**Conclusion & Recommendation:** The {answer.lower()} is clearly visible and well-documented. Continue with standard procedural protocols."""
        
        else:
            # Generic CoT for other question types
            cot = f"""**Basic Assessment:** The {procedure_type} image shows clear visualization of the {anatomy} with adequate quality for analysis.

**Findings Recognition:** The primary finding is {answer.lower()}, which is clearly identifiable in the image.

**Localization & Description:** The {answer.lower()} is located in the {anatomy} and demonstrates characteristic features typical of this condition.

**Synthesis & Correlation:** The findings are consistent with {answer.lower()} and correlate well with the clinical context of the {procedure_type}.

**Temporal Analysis:** The image captures the current state during the procedure with no signs of recent changes.

**Prognosis & Risk:** The {answer.lower()} presents with standard characteristics for this type of finding.

**Conclusion & Recommendation:** The findings are consistent with {answer.lower()}. Proceed with appropriate clinical management."""
        
        return cot
    
    def _get_risk_assessment(self, answer: str) -> str:
        """Get risk assessment based on the answer."""
        high_risk_terms = ['cancer', 'malignant', 'tumor', 'carcinoma', 'adenocarcinoma']
        medium_risk_terms = ['polyp', 'lesion', 'mass', 'growth']
        low_risk_terms = ['normal', 'healthy', 'benign', 'inflammation']
        
        answer_lower = answer.lower()
        
        if any(term in answer_lower for term in high_risk_terms):
            return "high"
        elif any(term in answer_lower for term in medium_risk_terms):
            return "medium"
        elif any(term in answer_lower for term in low_risk_terms):
            return "low"
        else:
            return "standard"
    
    def get_examples(self, n: int = 3) -> List[Dict[str, Any]]:
        """
        Get n random few-shot examples from Kvasir-VQA dataset.
        
        Args:
            n: Number of examples to return (default: 3)
            
        Returns:
            List of example dictionaries with image, question, reasoning, and answer
        """
        if len(self.kvasir_samples) < n:
            logger.warning(f"Only {len(self.kvasir_samples)} Kvasir samples available, returning all")
            selected_samples = self.kvasir_samples
        else:
            selected_samples = random.sample(self.kvasir_samples, n)
        
        examples = []
        for sample in selected_samples:
            question = sample['vqa_question']
            answer = sample['vqa_answer']
            reasoning = self._generate_synthetic_cot(question, answer, sample)
            
            example = {
                'image': sample['image_path'],
                'question': question,
                'reasoning': reasoning,
                'answer': answer,
                'dataset': sample['dataset_name'],
                'sample_id': sample['sample_id']
            }
            examples.append(example)
        
        logger.info(f"Generated {len(examples)} few-shot examples from Kvasir-VQA")
        return examples
    
    def get_examples_for_dataset(self, target_dataset: str, n: int = 3) -> List[Dict[str, Any]]:
        """
        Get few-shot examples specifically tailored for a target dataset.
        
        Args:
            target_dataset: Name of the target dataset (e.g., 'surg396k', 'egosurgery')
            n: Number of examples to return
            
        Returns:
            List of tailored examples
        """
        examples = self.get_examples(n)
        
        # Tailor examples based on target dataset
        if target_dataset == 'surg396k':
            # Focus on endoscopic findings
            examples = [ex for ex in examples if 'endoscopic' in ex['reasoning'].lower() or 'polyp' in ex['answer'].lower()]
        elif target_dataset == 'egosurgery':
            # Focus on surgical procedures
            examples = [ex for ex in examples if 'surgical' in ex['reasoning'].lower()]
        elif target_dataset == 'heichole':
            # Focus on laparoscopic procedures
            examples = [ex for ex in examples if 'laparoscopic' in ex['reasoning'].lower()]
        
        return examples






