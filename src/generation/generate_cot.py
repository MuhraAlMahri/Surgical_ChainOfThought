#!/usr/bin/env python3
"""
CoT Generation Script for Surgical Vision Analysis

This script uses Qwen2.5-VL-72B-Instruct on MI210 GPUs to generate
clinical Chain-of-Thought reasoning for surgical images.
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoTGenerator:
    """Generates clinical Chain-of-Thought reasoning for surgical images."""
    
    def __init__(self, config_path: str, manifest_path: str):
        """Initialize the CoT generator."""
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
        
        # Load prompt builder
        from .prompt_builder import PromptBuilder
        self.prompt_builder = PromptBuilder("configs/cot_template.yaml")
        
        logger.info("CoT Generator initialized")
    
    def setup_model(self) -> None:
        """Setup the Qwen2.5-VL model with quantization for MI210 GPUs."""
        logger.info("Setting up Qwen2.5-VL model...")
        
        try:
            # Configure quantization for MI210 GPUs
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
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'pad_token_id': self.processor.tokenizer.eos_token_id
            }
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            raise
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and preprocess an image."""
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return None
            
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def generate_single_cot(self, sample: Dict) -> Dict:
        """Generate CoT for a single sample."""
        try:
            # Load image
            image = self.load_image(sample['image_path'])
            if image is None:
                return {
                    'sample_id': sample['sample_id'],
                    'error': 'Failed to load image',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Build prompt
            prompt = self.prompt_builder.build_prompt(sample)
            
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
            image_inputs, video_inputs = self.processor.process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config
                )
            
            # Decode response
            response = self.processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Format response
            formatted_response = self.prompt_builder.format_response(response, sample)
            formatted_response['timestamp'] = datetime.now().isoformat()
            
            # Validate response
            validation = self.prompt_builder.validate_response(formatted_response)
            formatted_response['validation'] = validation
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error generating CoT for {sample['sample_id']}: {e}")
            return {
                'sample_id': sample['sample_id'],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_batch_cot(self, samples: List[Dict], batch_size: int = None) -> List[Dict]:
        """Generate CoT for a batch of samples."""
        if batch_size is None:
            batch_size = self.model_config.get('batch_size', 4)
        
        results = []
        
        # Process samples in batches
        for i in tqdm(range(0, len(samples), batch_size), desc="Generating CoT"):
            batch = samples[i:i + batch_size]
            batch_results = []
            
            # Process each sample in the batch
            for sample in batch:
                result = self.generate_single_cot(sample)
                batch_results.append(result)
                
                # Clear cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            results.extend(batch_results)
            
            # Save intermediate results
            if (i + batch_size) % 10 == 0:  # Save every 10 batches
                self._save_intermediate_results(results, i + batch_size)
        
        return results
    
    def _save_intermediate_results(self, results: List[Dict], processed_count: int) -> None:
        """Save intermediate results to prevent data loss."""
        output_file = self.output_dir / f"generated_cot_intermediate_{processed_count}.jsonl"
        
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        logger.info(f"Saved intermediate results: {processed_count} samples")
    
    def generate_cot_dataset(self, split: str = 'train', max_samples: int = None) -> None:
        """Generate CoT for the entire dataset split."""
        logger.info(f"Generating CoT for {split} split...")
        
        # Filter samples by split
        samples = [s for s in self.manifest['samples'] if s['split'] == split]
        
        if max_samples:
            samples = samples[:max_samples]
            logger.info(f"Limited to {max_samples} samples for testing")
        
        logger.info(f"Processing {len(samples)} samples")
        
        # Generate CoT
        results = self.generate_batch_cot(samples)
        
        # Save final results
        output_file = self.output_dir / f"generated_cot_{split}.jsonl"
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        # Save summary
        summary = {
            'metadata': {
                'split': split,
                'total_samples': len(samples),
                'processed_samples': len(results),
                'successful_samples': len([r for r in results if 'error' not in r]),
                'failed_samples': len([r for r in results if 'error' in r]),
                'generation_time': datetime.now().isoformat()
            },
            'quality_stats': self._calculate_quality_stats(results)
        }
        
        summary_file = self.output_dir / f"generation_summary_{split}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"CoT generation complete!")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Summary saved to: {summary_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("CoT GENERATION SUMMARY")
        print("="*60)
        print(f"Split: {split}")
        print(f"Total samples: {len(samples)}")
        print(f"Processed: {len(results)}")
        print(f"Successful: {summary['metadata']['successful_samples']}")
        print(f"Failed: {summary['metadata']['failed_samples']}")
        print(f"Success rate: {summary['metadata']['successful_samples']/len(results)*100:.1f}%")
        print("="*60)
    
    def _calculate_quality_stats(self, results: List[Dict]) -> Dict:
        """Calculate quality statistics for the generated results."""
        valid_results = [r for r in results if 'validation' in r and 'error' not in r]
        
        if not valid_results:
            return {'average_quality_score': 0.0, 'completeness_rate': 0.0}
        
        quality_scores = [r['validation']['quality_score'] for r in valid_results]
        completeness_rates = [1.0 if r['validation']['is_valid'] else 0.0 for r in valid_results]
        
        return {
            'average_quality_score': sum(quality_scores) / len(quality_scores),
            'completeness_rate': sum(completeness_rates) / len(completeness_rates),
            'total_valid_samples': len(valid_results)
        }
    
    def run_simulation(self, split: str, max_samples: Optional[int] = None):
        """Run simulation mode for testing without loading the actual model."""
        logger.info(f"Running simulation mode for {split} split")
        
        # Load samples
        samples = self._load_samples(split, max_samples)
        logger.info(f"Loaded {len(samples)} samples for simulation")
        
        # Generate simulated CoT
        results = []
        for i, sample in enumerate(tqdm(samples, desc="Simulating CoT generation")):
            # Create simulated CoT reasoning
            simulated_cot = {
                'sample_id': sample['sample_id'],
                'dataset': sample['dataset'],
                'image_path': sample['image_path'],
                'cot_reasoning': {
                    '1_quality_assessment': f"Simulated quality assessment for {sample['dataset']} sample",
                    '2_anatomical_identification': f"Simulated anatomical identification for {sample['dataset']} sample",
                    '3_pathological_analysis': f"Simulated pathological analysis for {sample['dataset']} sample",
                    '4_surgical_context': f"Simulated surgical context for {sample['dataset']} sample",
                    '5_technical_considerations': f"Simulated technical considerations for {sample['dataset']} sample",
                    '6_clinical_recommendations': f"Simulated clinical recommendations for {sample['dataset']} sample"
                },
                'validation': {
                    'is_valid': True,
                    'quality_score': 0.9,
                    'completeness_score': 0.95,
                    'clinical_relevance_score': 0.88
                },
                'generation_metadata': {
                    'model_name': 'simulation',
                    'generation_time': datetime.now().isoformat(),
                    'simulation_mode': True
                }
            }
            results.append(simulated_cot)
        
        # Save simulation results
        output_file = self.output_dir / f"simulated_cot_{split}.jsonl"
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        # Save simulation summary
        summary = {
            'metadata': {
                'split': split,
                'total_samples': len(samples),
                'simulated_samples': len(results),
                'simulation_time': datetime.now().isoformat(),
                'simulation_mode': True
            },
            'quality_stats': self._calculate_quality_stats(results)
        }
        
        summary_file = self.output_dir / f"simulation_summary_{split}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Simulation complete! Results saved to: {output_file}")
        print(f"\nSimulation completed: {len(results)} samples processed")

def main():
    """Main function to run CoT generation."""
    parser = argparse.ArgumentParser(description="Generate CoT for surgical images")
    parser.add_argument("--config", default="configs/paths.yaml",
                       help="Path to configuration file")
    parser.add_argument("--manifest", default="data/standardized/data_manifest.json",
                       help="Path to data manifest file")
    parser.add_argument("--split", default="train", choices=['train', 'val', 'test'],
                       help="Dataset split to process")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--real-mode", action="store_true",
                       help="Run real model generation instead of simulation")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CoTGenerator(args.config, args.manifest)
    
    if args.real_mode:
        # Setup real model
        generator.setup_model()
        
        # Update batch size if specified
        if args.batch_size:
            generator.model_config['batch_size'] = args.batch_size
        
        # Generate real CoT
        generator.generate_cot_dataset(args.split, args.max_samples)
    else:
        # Run simulation mode
        print("Running in simulation mode...")
        generator.run_simulation(args.split, args.max_samples)

if __name__ == "__main__":
    main()
