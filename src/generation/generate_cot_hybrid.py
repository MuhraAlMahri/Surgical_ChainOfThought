#!/usr/bin/env python3
"""
Hybrid CoT Generation Script for Surgical Vision Analysis

This script implements a hybrid few-shot/zero-shot approach:
- Zero-shot mode: For datasets WITH Q&A (Kvasir-VQA)
- Few-shot mode: For datasets WITHOUT Q&A (Surg-396k, EgoSurgery, etc.)
"""

import os
import json
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import argparse
from tqdm import tqdm
import gc

# Import required libraries for model loading
try:
    from transformers import (
        Qwen2VLForConditionalGeneration, 
        Qwen2VLProcessor,
        BitsAndBytesConfig
    )
    from PIL import Image
    import bitsandbytes as bnb
except ImportError as e:
    print(f"Required libraries not installed: {e}")
    print("Please install: pip install transformers torch bitsandbytes pillow")
    exit(1)

# Import hybrid prompt builder
from .prompt_builder_hybrid import HybridPromptBuilder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridCoTGenerator:
    """Generates clinical Chain-of-Thought reasoning using hybrid few-shot/zero-shot approach."""
    
    def __init__(self, config_path: str, manifest_path: str):
        """Initialize the hybrid CoT generator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        self.model_config = self.config['model']
        self.output_dir = Path(self.config['output']['generated_cot_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and processor
        self.model = None
        self.processor = None
        self.device = None
        
        # Load hybrid prompt builder
        self.prompt_builder = HybridPromptBuilder(config_path, manifest_path)
        
        logger.info("Hybrid CoT Generator initialized")
    
    def setup_model(self) -> None:
        """Setup the Qwen2.5-VL model with quantization for A100 GPUs."""
        logger.info("Setting up Qwen2.5-VL model...")
        
        try:
            # Configure quantization for A100 GPUs
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load model and processor
            model_name = self.model_config['name']
            logger.info(f"Loading model: {model_name}")
            
            self.processor = Qwen2VLProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            # Set device
            self.device = next(self.model.parameters()).device
            logger.info(f"Model loaded on device: {self.device}")
            
            # Set generation parameters
            self.generation_config = {
                'max_new_tokens': self.model_config.get('max_length', 2048),
                'temperature': 0.7,
                'do_sample': True,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'pad_token_id': self.processor.tokenizer.eos_token_id
            }
            
            logger.info("Model setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and preprocess image."""
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return None
            
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def generate_cot(self, sample: Dict[str, Any], n_examples: int = 3) -> Dict[str, Any]:
        """Generate CoT for a single sample using hybrid approach."""
        try:
            # Load image
            image = self.load_image(sample['image_path'])
            if image is None:
                return None
            
            # Build prompt using hybrid approach
            prompt = self.prompt_builder.build_prompt(sample, include_image=True, n_examples=n_examples)
            
            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = self.processor(
                text=[text], 
                images=[image], 
                return_tensors="pt"
            )
            
            # Move to device
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **image_inputs,
                    **self.generation_config
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (remove the prompt)
            if "###" in response:
                response = response.split("###")[-1].strip()
            elif "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            # Format response
            formatted_response = self.prompt_builder.format_response(response, sample)
            formatted_response['timestamp'] = datetime.now().isoformat()
            
            # Add validation
            validation = self.prompt_builder.validate_response(formatted_response)
            formatted_response['validation'] = validation
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Failed to generate CoT for {sample['sample_id']}: {e}")
            return None
    
    def generate_batch(self, samples: List[Dict[str, Any]], n_examples: int = 3) -> List[Dict[str, Any]]:
        """Generate CoT for a batch of samples."""
        results = []
        
        for sample in tqdm(samples, desc="Generating CoT"):
            result = self.generate_cot(sample, n_examples)
            if result:
                results.append(result)
            
            # Memory management
            if len(results) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """Save results to JSONL file."""
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        logger.info(f"Saved {len(results)} results to {output_path}")
    
    def get_dataset_samples(self, dataset_name: str, split: str = 'train', max_samples: int = None) -> List[Dict[str, Any]]:
        """Get samples from a specific dataset and split."""
        samples = [
            sample for sample in self.manifest['samples']
            if sample['dataset_name'] == dataset_name and sample['split'] == split
        ]
        
        if max_samples:
            samples = samples[:max_samples]
        
        return samples
    
    def generate_for_dataset(self, dataset_name: str, split: str = 'train', max_samples: int = None, n_examples: int = 3) -> List[Dict[str, Any]]:
        """Generate CoT for all samples in a specific dataset."""
        logger.info(f"Generating CoT for {dataset_name} dataset ({split} split)")
        
        samples = self.get_dataset_samples(dataset_name, split, max_samples)
        logger.info(f"Found {len(samples)} samples")
        
        if not samples:
            logger.warning(f"No samples found for {dataset_name} {split}")
            return []
        
        # Generate CoT
        results = self.generate_batch(samples, n_examples)
        
        # Save results
        output_file = f"generated_cot_{dataset_name}_{split}.jsonl"
        self.save_results(results, output_file)
        
        return results

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Generate CoT using hybrid approach')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--manifest', required=True, help='Path to data manifest')
    parser.add_argument('--dataset', help='Specific dataset to process')
    parser.add_argument('--split', default='train', help='Data split to process')
    parser.add_argument('--max-samples', type=int, help='Maximum number of samples to process')
    parser.add_argument('--n-examples', type=int, default=3, help='Number of few-shot examples')
    parser.add_argument('--real-mode', action='store_true', help='Use real model (not demo mode)')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = HybridCoTGenerator(args.config, args.manifest)
    
    if args.real_mode:
        # Setup model for real generation
        generator.setup_model()
        
        if args.dataset:
            # Process specific dataset
            results = generator.generate_for_dataset(
                args.dataset, 
                args.split, 
                args.max_samples, 
                args.n_examples
            )
            logger.info(f"Generated CoT for {len(results)} samples from {args.dataset}")
        else:
            # Process all datasets
            datasets = ['cataract1k', 'egosurgery', 'heichole', 'kvasirvqa', 'pitvis', 'surg396k']
            for dataset in datasets:
                results = generator.generate_for_dataset(
                    dataset, 
                    args.split, 
                    args.max_samples, 
                    args.n_examples
                )
                logger.info(f"Generated CoT for {len(results)} samples from {dataset}")
    else:
        # Demo mode - just test the prompt building
        logger.info("Demo mode: Testing prompt building")
        
        # Test with a few samples
        test_samples = generator.get_dataset_samples('kvasirvqa', 'train', 2)
        test_samples.extend(generator.get_dataset_samples('surg396k', 'train', 2))
        
        for sample in test_samples:
            prompt = generator.prompt_builder.build_prompt(sample, include_image=False, n_examples=args.n_examples)
            logger.info(f"Generated prompt for {sample['dataset_name']} sample {sample['sample_id']}")
            logger.info(f"Prompt length: {len(prompt)} characters")

if __name__ == "__main__":
    main()
