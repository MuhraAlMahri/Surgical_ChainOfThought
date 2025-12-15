#!/usr/bin/env python3
"""
Question Categorizer for Surgical VQA
Uses LLM-based semantic classification to categorize questions into 3 clinical stages:
1. Abnormality/Instrument Detection
2. Characteristics (color, type, location, count)
3. Treatment/Clinical Context
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionCategorizer:
    """Categorize questions into clinical reasoning stages using LLM."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        use_cache: bool = True,
        cache_file: Optional[str] = None
    ):
        """
        Initialize the question categorizer.
        
        Args:
            model_name: HuggingFace model name for classification
            use_cache: Whether to cache classification results
            cache_file: Path to cache file for storing classifications
        """
        self.model_name = model_name
        self.use_cache = use_cache
        self.cache_file = cache_file or "question_category_cache.json"
        self.cache: Dict[str, Dict] = {}
        self.model = None
        self.tokenizer = None
        
        # Load cache if exists
        if use_cache and Path(self.cache_file).exists():
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
            logger.info(f"Loaded {len(self.cache)} cached classifications")
    
    def load_model(self):
        """Load the LLM model for classification."""
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            logger.info("Model loaded successfully")
    
    def classify_question(
        self,
        question: str,
        category: Optional[str] = None,
        dataset: str = "kvasir"
    ) -> Dict[str, any]:
        """
        Classify a question into one of three clinical stages.
        
        Args:
            question: The question text
            category: Optional category/type of question
            dataset: Dataset name (kvasir or endovis)
            
        Returns:
            Dictionary with 'stage', 'category_name', 'confidence'
        """
        # Check cache first
        cache_key = f"{question}_{category}_{dataset}"
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load model if needed
        if self.model is None:
            self.load_model()
        
        # Build classification prompt
        prompt = self._build_classification_prompt(question, category, dataset)
        
        # Classify using LLM
        messages = [
            {
                "role": "system",
                "content": "You are a medical AI assistant that classifies surgical VQA questions into clinical reasoning stages."
            },
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.1
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Parse response
        result = self._parse_classification_response(response, question, category)
        
        # Cache result
        if self.use_cache:
            self.cache[cache_key] = result
            self._save_cache()
        
        return result
    
    def _build_classification_prompt(
        self,
        question: str,
        category: Optional[str],
        dataset: str
    ) -> str:
        """Build the classification prompt."""
        prompt = f"""Classify this surgical/endoscopic VQA question into one of three clinical reasoning stages:

Stage 1 (Abnormality/Instrument Detection): Questions about detecting abnormalities, instruments, polyps, lesions, or anatomical structures. Examples:
- "Is there a polyp?"
- "What instruments are present?"
- "Are there any abnormalities?"
- "What anatomical landmarks are visible?"

Stage 2 (Characteristics): Questions about properties, attributes, or details of detected findings. Examples:
- "What is the color of the polyp?"
- "Where is the lesion located?"
- "How many instruments are there?"
- "What type of instrument is this?"
- "What is the size of the abnormality?"

Stage 3 (Treatment/Clinical Context): Questions about diagnosis, treatment recommendations, clinical significance, or next steps. Examples:
- "What is the diagnosis?"
- "What treatment is recommended?"
- "What is the clinical significance?"
- "Have all polyps been removed?"
- "What should be done next?"

Question: {question}
"""
        if category:
            prompt += f"Question Category: {category}\n"
        prompt += "\nRespond with ONLY the stage number (1, 2, or 3) and category name. Format: 'Stage X: CategoryName'"
        
        return prompt
    
    def _parse_classification_response(
        self,
        response: str,
        question: str,
        category: Optional[str]
    ) -> Dict[str, any]:
        """Parse the LLM response to extract stage and category."""
        response_lower = response.lower()
        
        # Try to extract stage number
        stage = None
        if "stage 1" in response_lower or "1" in response_lower[:10]:
            stage = 1
            category_name = "abnormality_detection"
        elif "stage 3" in response_lower or ("3" in response_lower[:10] and "treatment" in response_lower):
            stage = 3
            category_name = "treatment"
        elif "stage 2" in response_lower or "2" in response_lower[:10]:
            stage = 2
            category_name = "characteristics"
        else:
            # Fallback to rule-based
            stage, category_name = self._rule_based_classify(question, category)
        
        return {
            "stage": stage,
            "category": category_name,
            "confidence": "high" if stage else "low",
            "raw_response": response
        }
    
    def _rule_based_classify(
        self,
        question: str,
        category: Optional[str]
    ) -> Tuple[int, str]:
        """Rule-based fallback classification."""
        q_lower = question.lower()
        cat_lower = (category or "").lower()
        
        # Stage 3 keywords
        if any(kw in q_lower for kw in [
            "diagnosis", "treatment", "recommend", "clinical significance",
            "what should be done", "next step", "removed", "complete"
        ]):
            return 3, "treatment"
        
        # Stage 1 keywords (detection)
        if any(kw in q_lower for kw in [
            "is there", "are there", "what instruments", "what anatomical",
            "detect", "present", "visible", "abnormality", "polyp", "lesion"
        ]) or "detection" in cat_lower:
            return 1, "abnormality_detection"
        
        # Stage 2 keywords (characteristics)
        if any(kw in q_lower for kw in [
            "what is the color", "where is", "how many", "what type",
            "what size", "location", "count", "characteristics", "properties"
        ]) or "characteristics" in cat_lower or "count" in cat_lower:
            return 2, "characteristics"
        
        # Default to stage 2 (most common)
        return 2, "characteristics"
    
    def categorize_dataset(
        self,
        data: List[Dict],
        dataset_name: str = "kvasir"
    ) -> Dict[int, List[Dict]]:
        """
        Categorize an entire dataset.
        
        Args:
            data: List of QA pairs with 'question' and optionally 'category' fields
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary mapping stage (1,2,3) to list of categorized items
        """
        stage_data = {1: [], 2: [], 3: []}
        
        logger.info(f"Categorizing {len(data)} questions...")
        
        for item in tqdm(data, desc="Categorizing questions"):
            question = item.get('question', '')
            category = item.get('category') or item.get('question_type')
            
            if not question:
                logger.warning(f"Skipping item without question: {item}")
                continue
            
            # Classify
            classification = self.classify_question(question, category, dataset_name)
            
            # Add classification info to item
            item['stage'] = classification['stage']
            item['category'] = classification['category']
            item['classification_confidence'] = classification['confidence']
            
            # Add to appropriate stage
            stage_data[classification['stage']].append(item)
        
        logger.info(f"Classification complete:")
        logger.info(f"  Stage 1 (Abnormality Detection): {len(stage_data[1])} questions")
        logger.info(f"  Stage 2 (Characteristics): {len(stage_data[2])} questions")
        logger.info(f"  Stage 3 (Treatment): {len(stage_data[3])} questions")
        
        return stage_data
    
    def _save_cache(self):
        """Save classification cache to file."""
        if self.use_cache:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
    
    def save_categorized_data(
        self,
        stage_data: Dict[int, List[Dict]],
        output_dir: Path,
        split_name: str = "train"
    ):
        """Save categorized data to JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save by stage
        for stage, items in stage_data.items():
            output_file = output_dir / f"{split_name}_stage{stage}.json"
            with open(output_file, 'w') as f:
                json.dump(items, f, indent=2)
            logger.info(f"Saved {len(items)} items to {output_file}")
        
        # Save combined with stage info
        all_items = []
        for stage, items in stage_data.items():
            all_items.extend(items)
        
        output_file = output_dir / f"{split_name}_categorized.json"
        with open(output_file, 'w') as f:
            json.dump(all_items, f, indent=2)
        logger.info(f"Saved {len(all_items)} total items to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Categorize questions into clinical stages")
    parser.add_argument("--input", required=True, help="Input JSON file with QA pairs")
    parser.add_argument("--output", required=True, help="Output directory for categorized data")
    parser.add_argument("--dataset", default="kvasir", choices=["kvasir", "endovis"], help="Dataset name")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model for classification")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    # Load data
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Categorize
    categorizer = QuestionCategorizer(
        model_name=args.model,
        use_cache=not args.no_cache
    )
    
    stage_data = categorizer.categorize_dataset(data, args.dataset)
    
    # Save results
    output_dir = Path(args.output)
    categorizer.save_categorized_data(stage_data, output_dir)














