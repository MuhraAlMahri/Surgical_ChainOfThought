#!/usr/bin/env python3
"""
Evaluate CXRTrek Sequential Models

Evaluates three specialized models (one per stage) on their respective test data.
Each model is evaluated independently on its own stage.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import defaultdict


class EvaluationDataset(Dataset):
    """Dataset for evaluation."""
    
    def __init__(self, data_path: str, image_dir: str, stage_num: int):
        """
        Args:
            data_path: Path to JSON data file
            image_dir: Directory containing images
            stage_num: Which stage to extract (1, 2, or 3)
        """
        self.image_dir = image_dir
        self.stage_num = stage_num
        
        # Load data
        with open(data_path, 'r') as f:
            all_data = json.load(f)
        
        # Extract QA pairs for this stage
        self.samples = []
        for item in all_data:
            image_path = item.get('image_path', item.get('image', ''))
            
            # Handle image path
            if not os.path.isabs(image_path):
                if image_path.startswith('images/'):
                    image_path = image_path[7:]
                image_path = os.path.join(image_dir, image_path)
            
            # Extract QA pairs
            stages_data = item.get('stages', item.get('clinical_flow_stages', {}))
            qa_pairs = self._extract_stage_qa(stages_data, stage_num)
            
            for qa in qa_pairs:
                self.samples.append({
                    'image_path': image_path,
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'stage': stage_num,
                    'image_id': item.get('image_id', '')
                })
        
        # Use test split (last 10%)
        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(0.9 * len(self.samples))
        self.samples = [self.samples[i] for i in indices[split_idx:]]
        
        print(f"Stage {stage_num}: Loaded {len(self.samples)} test samples")
    
    def _extract_stage_qa(self, stages_data: Dict, stage_num: int) -> List[Dict]:
        """Extract QA pairs for a specific stage."""
        possible_keys = [
            f'stage_{stage_num}',
            f'Stage-{stage_num}',
            f'Stage {stage_num}',
            f'Stage-{stage_num}: Initial Assessment',
            f'Stage-{stage_num}: Findings Identification',
            f'Stage-{stage_num}: Clinical Context'
        ]
        
        for key in possible_keys:
            if key in stages_data:
                stage_data = stages_data[key]
                qa_pairs = []
                
                if isinstance(stage_data, dict):
                    for q_key, answer in stage_data.items():
                        if q_key.startswith('Q'):
                            question = q_key[3:] if len(q_key) > 3 else q_key
                            qa_pairs.append({'question': question, 'answer': str(answer)})
                elif isinstance(stage_data, list):
                    for qa in stage_data:
                        if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                            qa_pairs.append({'question': qa['question'], 'answer': str(qa['answer'])})
                
                return qa_pairs
        
        return []
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'image_path': sample['image_path'],
            'question': sample['question'],
            'answer': sample['answer'],
            'stage': sample['stage'],
            'image_id': sample.get('image_id', '')
        }


def evaluate_model_on_stage(
    model,
    processor,
    dataset,
    stage_num: int,
    device: str = "cuda",
    batch_size: int = 1
) -> Dict:
    """
    Evaluate a single model on a specific stage.
    
    Args:
        model: The model to evaluate
        processor: The processor for the model
        dataset: Dataset containing test samples
        stage_num: Stage number (1, 2, or 3)
        device: Device to use
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with results
    """
    model.eval()
    
    all_predictions = []
    correct = 0
    total = 0
    
    print(f"\nEvaluating Stage {stage_num} model on {len(dataset)} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_samples = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            
            # Load images
            images = []
            for sample in batch_samples:
                try:
                    image = Image.open(sample['image_path']).convert('RGB')
                    images.append(image)
                except Exception as e:
                    print(f"Error loading image {sample['image_path']}: {e}")
                    continue
            
            if not images:
                continue
            
            # Prepare prompts
            questions = [sample['question'] for sample in batch_samples]
            ground_truths = [sample['answer'] for sample in batch_samples]
            
            # Create messages
            messages_batch = []
            for q in questions:
                messages_batch.append([{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q}
                    ]
                }])
            
            # Process batch
            texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                     for msg in messages_batch]
            
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            # Generate predictions
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
            
            # Decode predictions
            generated_ids = [
                output_ids[i][len(inputs.input_ids[i]):] 
                for i in range(len(output_ids))
            ]
            predictions = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Compare predictions with ground truth
            for pred, gt, sample in zip(predictions, ground_truths, batch_samples):
                pred_clean = pred.strip().lower()
                gt_clean = gt.strip().lower()
                
                is_correct = (pred_clean == gt_clean)
                
                all_predictions.append({
                    'image_id': sample.get('image_id', ''),
                    'question': sample['question'],
                    'prediction': pred.strip(),
                    'ground_truth': gt,
                    'correct': is_correct,
                    'stage': stage_num
                })
                
                if is_correct:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"Stage {stage_num} Results:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    
    return {
        'stage': stage_num,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': all_predictions
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate CXRTrek Sequential Models')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data JSON')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--stage1_checkpoint', type=str, required=True, help='Path to Stage 1 model checkpoint')
    parser.add_argument('--stage2_checkpoint', type=str, required=True, help='Path to Stage 2 model checkpoint')
    parser.add_argument('--stage3_checkpoint', type=str, required=True, help='Path to Stage 3 model checkpoint')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save results JSON')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = "cuda:0"
    else:
        device = "cpu"
    
    print("="*60)
    print("CXRTrek Sequential Evaluation")
    print("="*60)
    print(f"Device: {device}")
    print(f"Data path: {args.data_path}")
    print(f"Image directory: {args.image_dir}")
    print(f"Output file: {args.output_file}")
    print("="*60)
    
    # Load processor (shared across all models)
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True
    )
    
    # Results storage
    all_results = {
        'experiment_name': 'CXRTrek Sequential Training',
        'evaluation_date': '2025-10-18',
        'model_base': 'Qwen2-VL-2B-Instruct',
        'training_approach': 'Three specialized models (independent training)',
        'stage_results': {},
        'overall_results': {},
        'all_predictions': []
    }
    
    # Evaluate each stage
    for stage_num in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"EVALUATING STAGE {stage_num}")
        print(f"{'='*60}")
        
        # Get checkpoint path
        if stage_num == 1:
            checkpoint_path = args.stage1_checkpoint
        elif stage_num == 2:
            checkpoint_path = args.stage2_checkpoint
        else:
            checkpoint_path = args.stage3_checkpoint
        
        print(f"Checkpoint: {checkpoint_path}")
        
        # Load dataset for this stage
        dataset = EvaluationDataset(args.data_path, args.image_dir, stage_num)
        
        if len(dataset) == 0:
            print(f"No samples found for stage {stage_num}, skipping...")
            continue
        
        # Load base model
        print(f"Loading base model...")
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        print(f"Loading LoRA adapter from {checkpoint_path}...")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.eval()
        
        # Evaluate
        stage_results = evaluate_model_on_stage(
            model=model,
            processor=processor,
            dataset=dataset,
            stage_num=stage_num,
            device=device,
            batch_size=args.batch_size
        )
        
        # Store results
        all_results['stage_results'][f'stage_{stage_num}'] = {
            'stage_name': ['Initial Assessment', 'Findings Identification', 'Clinical Context'][stage_num-1],
            'accuracy': stage_results['accuracy'],
            'accuracy_percentage': f"{stage_results['accuracy']*100:.2f}%",
            'correct_predictions': stage_results['correct'],
            'total_samples': stage_results['total'],
            'sample_percentage': f"{stage_results['total']/41130*100:.1f}%",  # Approximate
            'checkpoint': checkpoint_path
        }
        
        all_results['all_predictions'].extend(stage_results['predictions'])
        
        # Clean up
        del model
        del base_model
        torch.cuda.empty_cache()
    
    # Calculate overall results
    total_correct = sum(r['correct_predictions'] for r in all_results['stage_results'].values())
    total_samples = sum(r['total_samples'] for r in all_results['stage_results'].values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    all_results['overall_results'] = {
        'total_samples': total_samples,
        'correct_predictions': total_correct,
        'accuracy': overall_accuracy,
        'accuracy_percentage': f"{overall_accuracy*100:.2f}%"
    }
    
    # Save results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\nOverall Accuracy: {overall_accuracy*100:.2f}% ({total_correct}/{total_samples})")
    print(f"\nPer-Stage Results:")
    for stage_key, stage_data in all_results['stage_results'].items():
        print(f"  {stage_data['stage_name']}: {stage_data['accuracy_percentage']} "
              f"({stage_data['correct_predictions']}/{stage_data['total_samples']})")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {args.output_file}")
    print("="*60)


if __name__ == "__main__":
    main()















