# EndoVis2018 Experiments - Quick Start

## 🚀 Run Everything (Recommended)

Submit the mega job that runs zero-shot + training in parallel:

```bash
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/endovis2018_experiments
./SUBMIT_MEGA_JOB.sh
```

This runs:
- **GPU 0**: Zero-shot evaluation (baseline)
- **GPU 1**: Training Exp1 (Random Baseline)
- **GPU 2**: Training Exp2 (Qwen Reordered)
- **GPU 3**: Training Exp3 (Sequential)

## ⚙️ Configuration

All settings match Kvasir-VQA experiments exactly:

### QLoRA Settings
- **r**: 4 (LoRA rank)
- **alpha**: 8 (LoRA alpha)
- **dropout**: 0.05
- **target_modules**: [q_proj, k_proj, v_proj, o_proj] (attention layers only)

### Training Settings
- **epochs**: 5
- **train_bs**: 1 (per-GPU batch size)
- **grad_accum**: 16 (effective batch size = 16)
- **lr**: 5.0e-5
- **weight_decay**: 0.01
- **max_seq_len**: 3072
- **bf16**: true
- **gradient_checkpointing**: true

### Model
- **Base Model**: Qwen/Qwen3-VL-8B-Instruct

### Batch Padding
- **Dynamic per-batch padding** (implemented in `collate_fn`)
- Each batch padded only to longest sequence in that batch
- Reduces wasted computation

## 📊 Expected Outputs

After job completion:
- Zero-shot results: `results/endovis2018_zeroshot.json`
- Trained models:
  - `models/exp1_random/` (Exp1)
  - `models/exp2_qwen_reordered/` (Exp2)
  - `models/exp3_sequential/` (Exp3)
- Logs: `slurm/logs/mega_job_*.out`

## 🔍 Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View logs
tail -f slurm/logs/mega_job_*.out

# Check specific GPU output
grep "\[GPU 0\]" slurm/logs/mega_job_*.out  # Zero-shot
grep "\[GPU 1\]" slurm/logs/mega_job_*.out  # Training Exp1
grep "\[GPU 2\]" slurm/logs/mega_job_*.out  # Training Exp2
grep "\[GPU 3\]" slurm/logs/mega_job_*.out  # Training Exp3
```

## ⚠️ Important

- All jobs run on **compute nodes** via SLURM (MBZUAI HPC policy)
- Data must be prepared first (see README.md Step 1)
- Instruction fine-tuning will be added after template approval

## 📝 Notes

- Same config as Kvasir-VQA for fair comparison
- Batch padding already implemented in training script
- Mega job uses 4 GPUs in parallel for efficiency
- All experiments run simultaneously (zero-shot + 3 training experiments)

