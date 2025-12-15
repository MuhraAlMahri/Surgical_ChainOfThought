import argparse
import yaml
import os
import sys
from pathlib import Path

# Add exp1 to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoModelForVision2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from dataset import VQASFTDataset, collate
import torch


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    model_name = cfg["model_name"]
    
    # Initialize distributed training if available
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        print(f"[Rank {local_rank}/{world_size}] Distributed training initialized")
    
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
    
    # Only main process should enrich data
    if local_rank in [-1, 0]:
        for split in ["train_jsonl", "val_jsonl"]:
            inp = base_dir / cfg["data"][split]
            outp = str(inp).replace(".jsonl", ".enriched.jsonl")
            if not os.path.exists(outp):
                print(f"Enriching {inp} -> {outp}")
                enrich_jsonl(str(inp), outp)
            cfg["data"][split] = outp
    
    # Sync all processes before loading data
    if local_rank != -1:
        torch.distributed.barrier()
        # Update config on all ranks
        for split in ["train_jsonl", "val_jsonl"]:
            inp = base_dir / cfg["data"][split]
            outp = str(inp).replace(".jsonl", ".enriched.jsonl")
            cfg["data"][split] = outp
    
    model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
    
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
    
    train_ds = VQASFTDataset(
        cfg["data"]["train_jsonl"],
        cfg["data"]["image_root"],
        model_name,
        cfg["train"]["max_seq_len"]
    )
    val_ds = VQASFTDataset(
        cfg["data"]["val_jsonl"],
        cfg["data"]["image_root"],
        model_name,
        cfg["train"]["max_seq_len"]
    )
    
    # Use absolute path for output
    script_dir = Path(__file__).parent
    
    args = TrainingArguments(
        output_dir=str(script_dir / "outputs_4gpu"),
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
        # Distributed training settings
        ddp_find_unused_parameters=False,
        local_rank=local_rank,
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

