#!/usr/bin/env python3
"""
Main Training Script for Multi-Head Temporal CoT Model
Supports both sequential curriculum learning and unified training.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import torch

from models.multi_head_model import create_model
from models.qwen3vl_multihead import create_qwen3vl_multihead
from models.medgemma_multihead import create_medgemma_multihead
from models.llava_med_multihead import create_llava_med_multihead
from data.vqa_data_loader import create_data_loader
from data.question_categorizer import QuestionCategorizer
from data.temporal_linker import TemporalLinker
from training.sequential_trainer import SequentialCurriculumTrainer
from training.temporal_trainer import TemporalSequenceTrainer
from training.unified_trainer import TemporalCoTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train multi-head temporal CoT model for surgical VQA"
    )
    
    # Model arguments
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Base vision-language model"
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for fine-tuning"
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    
    # Data arguments
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data JSON"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        help="Path to validation data JSON"
    )
    parser.add_argument(
        "--image-base-path",
        type=str,
        required=True,
        help="Base path for images"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kvasir", "endovis"],
        default="kvasir",
        help="Dataset name"
    )
    parser.add_argument(
        "--temporal-data",
        type=str,
        help="Path to temporal structure JSON (for EndoVis)"
    )
    
    # Training arguments
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["sequential", "unified"],
        default="unified",
        help="Training mode: sequential curriculum or unified"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--use-temporal",
        action="store_true",
        default=True,
        help="Use temporal context"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--categorize-questions",
        action="store_true",
        help="Categorize questions before training"
    )
    parser.add_argument(
        "--category-cache",
        type=str,
        help="Path to question category cache"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Categorize questions if needed
    if args.categorize_questions:
        logger.info("Categorizing questions...")
        categorizer = QuestionCategorizer(
            cache_file=args.category_cache
        )
        
        with open(args.train_data, 'r') as f:
            train_data = json.load(f)
        
        stage_data = categorizer.categorize_dataset(train_data, args.dataset)
        categorizer.save_categorized_data(stage_data, output_dir / "categorized", "train")
        
        # Update train_data with categories
        train_data = []
        for stage, items in stage_data.items():
            train_data.extend(items)
        
        # Save categorized data
        categorized_file = output_dir / "train_categorized.json"
        with open(categorized_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        args.train_data = str(categorized_file)
        logger.info(f"Saved categorized data to {categorized_file}")
    
    # Step 2: Create temporal structure if needed (EndoVis)
    temporal_data_file = None
    if args.dataset == "endovis" and args.temporal_data:
        logger.info("Creating temporal structure...")
        # Load QA pairs
        with open(args.train_data, 'r') as f:
            qa_pairs = json.load(f)
        
        # Group by sequence
        sequences = {}
        for qa in qa_pairs:
            seq_id = qa.get('sequence_id', 'unknown')
            if seq_id not in sequences:
                sequences[seq_id] = []
            sequences[seq_id].append(qa)
        
        # Create temporal structure for each sequence
        linker = TemporalLinker(args.image_base_path)
        all_temporal_data = []
        
        for seq_id, seq_qa_pairs in sequences.items():
            temporal_data = linker.create_temporal_structure(
                seq_id,
                seq_qa_pairs,
                compute_motion=True
            )
            all_temporal_data.extend(temporal_data)
        
        # Save temporal structure
        temporal_data_file = output_dir / "temporal_structure.json"
        linker.save_temporal_structure(all_temporal_data, temporal_data_file)
        logger.info(f"Saved temporal structure to {temporal_data_file}")
    
    # Step 3: Create model
    logger.info(f"Creating model: {args.base_model}")
    
    # Select appropriate model class based on base model name
    if "Qwen3-VL" in args.base_model or "Qwen3VL" in args.base_model:
        model = create_qwen3vl_multihead(
            base_model_name=args.base_model,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha
        )
    elif "medgemma" in args.base_model.lower() or "MedGemma" in args.base_model:
        model = create_medgemma_multihead(
            base_model_name=args.base_model,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha
        )
    elif "llava-med" in args.base_model.lower() or "LLaVA-Med" in args.base_model:
        model = create_llava_med_multihead(
            base_model_name=args.base_model,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            freeze_vision_tower=True
        )
    else:
        # Fallback to generic model
        logger.warning(f"Unknown model type, using generic model wrapper")
        model = create_model(
            base_model_name=args.base_model,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha
        )
    
    # Step 4: Create data loaders
    logger.info("Creating data loaders...")
    train_loader = create_data_loader(
        data_file=args.train_data,
        image_base_path=args.image_base_path,
        batch_size=args.batch_size,
        shuffle=True,
        is_temporal=(args.dataset == "endovis"),
        temporal_data_file=str(temporal_data_file) if temporal_data_file else None
    )
    
    val_loader = None
    if args.val_data:
        # Check if validation file exists
        val_file_path = Path(args.val_data)
        if val_file_path.exists():
            val_loader = create_data_loader(
                data_file=args.val_data,
                image_base_path=args.image_base_path,
                batch_size=args.batch_size,
                shuffle=False,
                is_temporal=(args.dataset == "endovis"),
                temporal_data_file=str(temporal_data_file) if temporal_data_file else None
            )
            logger.info(f"Validation loader created from {args.val_data}")
        else:
            logger.warning(f"Validation file not found: {args.val_data}")
            logger.warning("Continuing without validation set")
            val_loader = None
    
    # Step 5: Create trainer
    if args.training_mode == "sequential":
        logger.info("Using sequential curriculum learning")
        
        # Create stage-specific data loaders
        stage_loaders = {}
        val_loaders_dict = {}
        
        # Check for stage-specific files
        train_data_path = Path(args.train_data)
        base_dir = train_data_path.parent
        
        for stage in [1, 2, 3]:
            stage_file = base_dir / f"train_stage{stage}.json"
            if stage_file.exists():
                logger.info(f"Found stage {stage} data: {stage_file}")
                stage_loaders[stage] = create_data_loader(
                    data_file=str(stage_file),
                    image_base_path=args.image_base_path,
                    batch_size=args.batch_size,
                    shuffle=True,
                    is_temporal=(args.dataset == "endovis"),
                    temporal_data_file=str(temporal_data_file) if temporal_data_file else None
                )
                
                # Create validation loader for this stage if val data exists
                if args.val_data:
                    val_stage_file = Path(args.val_data).parent / f"val_stage{stage}.json"
                    if val_stage_file.exists():
                        val_loaders_dict[stage] = create_data_loader(
                            data_file=str(val_stage_file),
                            image_base_path=args.image_base_path,
                            batch_size=args.batch_size,
                            shuffle=False,
                            is_temporal=(args.dataset == "endovis"),
                            temporal_data_file=str(temporal_data_file) if temporal_data_file else None
                        )
                    elif val_loader:
                        val_loaders_dict[stage] = val_loader
            else:
                logger.warning(f"Stage {stage} data file not found: {stage_file}")
                logger.warning("Using full dataset for all stages")
                # Fallback: use full dataset
                stage_loaders[stage] = train_loader
                if val_loader:
                    val_loaders_dict[stage] = val_loader
        
        # If no stage files found, use full dataset for all stages
        if not stage_loaders:
            logger.warning("No stage-specific files found, using full dataset for all stages")
            stage_loaders = {1: train_loader, 2: train_loader, 3: train_loader}
            if val_loader:
                val_loaders_dict = {1: val_loader, 2: val_loader, 3: val_loader}
        
        trainer = SequentialCurriculumTrainer(
            model=model,
            stage_data_loaders=stage_loaders,
            val_loaders=val_loaders_dict if val_loaders_dict else None,
            device=device
        )
        
        trainer.train_all_stages(epochs_per_stage={1: args.num_epochs, 2: args.num_epochs, 3: args.num_epochs})
    
    else:
        logger.info("Using unified multi-head training")
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01
        )
        
        trainer = TemporalCoTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        trainer.train(
            num_epochs=args.num_epochs,
            use_temporal=args.use_temporal,
            save_dir=str(output_dir),
            save_every=1
        )
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

