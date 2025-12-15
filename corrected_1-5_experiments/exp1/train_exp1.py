import argparse
import yaml
import os
import sys
from pathlib import Path

# Add exp1 to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForVision2Seq, AutoModelForImageTextToText, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from dataset import VQASFTDataset, collate
from dataset_cached import VQASFTDatasetCached
import torch

# Note: Qwen3-VL uses AutoModelForVision2Seq instead of AutoModelForVision2Seq

# ============================================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================================
# Enable TF32 for Ampere+ GPUs (A100, RTX 30xx/40xx) - 1.2-1.5x speedup
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✅ TF32 enabled for faster training")

# Support for distributed training
def is_distributed():
    """Check if running in distributed mode"""
    return int(os.environ.get('WORLD_SIZE', 1)) > 1

def get_rank():
    """Get current process rank"""
    return int(os.environ.get('RANK', 0))

def is_main_process():
    """Check if this is the main process"""
    return get_rank() == 0


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    model_name = cfg["model_name"]
    
    # Initialize distributed training if using torchrun
    # (Trainer will also initialize, but we need it earlier for barriers)
    if is_distributed() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    
    # Print distributed info
    if is_distributed():
        print(f"[Rank {get_rank()}] Running in distributed mode with {os.environ.get('WORLD_SIZE')} GPUs")
    
    # preprocess jsonl to add question_type, candidates, normalized answers
    # Import with proper path handling
    exp1_dir = Path(__file__).parent
    data_dir = exp1_dir / "data"
    sys.path.insert(0, str(data_dir))
    
    # Import directly without relative imports
    import schema
    import json
    import re
    from pathlib import Path as PathlibPath
    
    def normalize_answer_local(ans):
        x = ans.strip().lower()
        x = re.sub(r"[^\w\.\-\% ]+", "", x)
        return x
    
    def enrich_jsonl_local(in_path, out_path):
        out = []
        with open(in_path, "r") as f:
            for line in f:
                ex = json.loads(line)
                q = ex["question"]
                gt = normalize_answer_local(ex["answer"])
                qtype = ex.get("question_type") or schema.infer_question_type(q)
                ex["question_type"] = qtype
                ex["answer"] = gt
                ex["answer_candidates"] = schema.build_candidates(qtype, ex)
                out.append(ex)
        PathlibPath(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for ex in out:
                f.write(json.dumps(ex) + "\n")
    
    enrich_jsonl = enrich_jsonl_local
    
    # Resolve paths relative to exp1 parent (corrected_1-5_experiments)
    base_dir = Path(__file__).parent.parent
    
    # Only main process should enrich data to avoid race conditions
    for split in ["train_jsonl", "val_jsonl"]:
        inp = base_dir / cfg["data"][split]
        outp = str(inp).replace(".jsonl", ".enriched.jsonl")
        if not os.path.exists(outp):
            if is_main_process() or not is_distributed():
                print(f"Enriching {inp} -> {outp}")
                enrich_jsonl(str(inp), outp)
            # Wait for main process to finish enriching (in distributed mode)
            if is_distributed():
                torch.distributed.barrier()
        cfg["data"][split] = outp
    
    # Use appropriate model class based on model name
    if "Qwen3" in model_name:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    
    # Enable gradient checkpointing before LoRA for memory savings
    if cfg["train"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    
    if cfg.get("vision_frozen", True):
        for n, p in model.named_parameters():
            if "vision_tower" in n or "visual" in n:
                p.requires_grad = False
    
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_cfg)
    
    # Check if we should use cached vision embeddings for faster training
    use_cache = cfg["data"].get("use_vision_cache", False)
    vision_cache_dir = cfg["data"].get("vision_cache_dir", None)
    
    if use_cache and vision_cache_dir:
        print("="*80)
        print("🚀 USING CACHED VISION EMBEDDINGS (2-5x speedup expected!)")
        print("="*80)
        cache_train = base_dir / vision_cache_dir / "train"
        cache_val = base_dir / vision_cache_dir / "val"
        resolution_id = cfg["data"].get("resolution_id", "full")
        
        train_ds = VQASFTDatasetCached(
            cfg["data"]["train_jsonl"],
            cfg["data"]["image_root"],
            model_name,
            cfg["train"]["max_seq_len"],
            vision_cache_dir=str(cache_train),
            resolution_id=resolution_id
        )
        val_ds = VQASFTDatasetCached(
            cfg["data"]["val_jsonl"],
            cfg["data"]["image_root"],
            model_name,
            cfg["train"]["max_seq_len"],
            vision_cache_dir=str(cache_val),
            resolution_id=resolution_id
        )
    else:
        print("Using on-the-fly image processing (slower)")
        # Get letterbox settings from config
        use_letterbox = cfg["data"].get("use_letterbox", False)
        target_size = cfg["data"].get("target_size", 768)
        
        train_ds = VQASFTDataset(
            cfg["data"]["train_jsonl"],
            cfg["data"]["image_root"],
            model_name,
            cfg["train"]["max_seq_len"],
            use_letterbox=use_letterbox,
            target_size=target_size
        )
        val_ds = VQASFTDataset(
            cfg["data"]["val_jsonl"],
            cfg["data"]["image_root"],
            model_name,
            cfg["train"]["max_seq_len"],
            use_letterbox=use_letterbox,
            target_size=target_size
        )
    
    # Use absolute path for output
    script_dir = Path(__file__).parent
    
    args = TrainingArguments(
        output_dir=str(script_dir / "outputs"),
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
        optim="adamw_torch_fused",  # Fused optimizer for ~5-10% speedup
        dataloader_num_workers=4,   # Parallel data loading
        dataloader_pin_memory=True,  # Faster GPU transfer
        dataloader_prefetch_factor=2,  # Prefetch batches
        # Distributed training settings
        ddp_find_unused_parameters=False,  # Faster DDP
        ddp_backend="nccl" if is_distributed() else None,
    )
    
    def compute_metrics(eval_pred):
        # exact-match on normalized strings; numeric tolerance from cfg
        # We won't decode here; evaluation will be done by a separate generate+eval script.
        return {}
    
    # Use custom collate function optimized for vision-language models
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate
    )
    
    trainer.train()


if __name__ == "__main__":
    import sys
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "corrected 1-5 experiments/exp1/config_exp1.yaml"
    main(cfg_file)

