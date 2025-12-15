"""
Experiment 2: Qwen Reordered Training
Train on QA pairs reordered by Qwen into 3 clinical stages (but all mixed together)
Model: Qwen3-VL-8B-Instruct
Resolution: 768x768 letterbox
Instructions: ULTRA_CONDENSED
"""
import argparse
import yaml
import os
import sys
from pathlib import Path

# Add exp2 to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForImageTextToText, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from dataset import VQASFTDataset, collate
import torch

# Performance optimizations
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✅ TF32 enabled for faster training")

# Distributed training support
def is_distributed():
    return int(os.environ.get('WORLD_SIZE', 1)) > 1

def get_rank():
    return int(os.environ.get('RANK', 0))

def is_main_process():
    return get_rank() == 0


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    model_name = cfg["model_name"]
    
    # Initialize distributed training if using torchrun
    if is_distributed() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    
    if is_distributed():
        print(f"[Rank {get_rank()}] Running in distributed mode with {os.environ.get('WORLD_SIZE')} GPUs")
    
    print(f"\n{'='*80}")
    print(f"EXP2: QWEN REORDERED TRAINING")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Strategy: Train on Qwen-reordered QA pairs (stages 1,2,3 mixed)")
    print(f"Instructions: ULTRA_CONDENSED")
    print(f"{'='*80}\n")
    
    # Load model
    print(f"Loading model: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Enable gradient checkpointing
    if cfg["train"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    
    # Freeze vision tower
    if cfg.get("vision_frozen", True):
        for n, p in model.named_parameters():
            if "vision_tower" in n or "visual" in n:
                p.requires_grad = False
    
    # Apply LoRA
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)
    
    # Create datasets
    base_dir = Path(__file__).parent.parent
    train_json = base_dir / cfg["data"]["train_json"]
    val_json = base_dir / cfg["data"]["val_json"]
    
    use_letterbox = cfg["data"].get("use_letterbox", False)
    target_size = cfg["data"].get("target_size", 768)
    
    train_ds = VQASFTDataset(
        str(train_json),
        cfg["data"]["image_root"],
        model_name,
        cfg["train"]["max_seq_len"],
        use_letterbox=use_letterbox,
        target_size=target_size
    )
    val_ds = VQASFTDataset(
        str(val_json),
        cfg["data"]["image_root"],
        model_name,
        cfg["train"]["max_seq_len"],
        use_letterbox=use_letterbox,
        target_size=target_size
    )
    
    # Output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs"
    
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["train"]["train_bs"],
        per_device_eval_batch_size=cfg["train"]["eval_bs"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        num_train_epochs=cfg["train"]["epochs"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        bf16=cfg["train"]["bf16"],
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        eval_strategy="steps",
        eval_steps=cfg["train"]["save_steps"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        report_to="none",
        # Performance optimizations
        optim="adamw_torch_fused",
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        # Distributed settings
        ddp_find_unused_parameters=False,
        ddp_backend="nccl" if is_distributed() else None,
        # Save only final checkpoint
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate
    )
    
    print(f"\nStarting training...")
    trainer.train()
    
    print(f"\n{'='*80}")
    print(f"EXP2 TRAINING COMPLETED")
    print(f"Checkpoint saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")
    args = parser.parse_args()
    
    main(args.config)






