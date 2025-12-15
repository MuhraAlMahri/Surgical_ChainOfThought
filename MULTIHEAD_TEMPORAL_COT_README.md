# Multi-Head Temporal Chain-of-Thought (CoT) for Surgical VQA

This implementation provides a comprehensive multi-stage Chain-of-Thought reasoning system for surgical Visual Question Answering that:

1. **Generates model-driven CoT reasoning** (not hand-crafted)
2. **Follows clinically meaningful structure** (abnormality → characteristics → treatment)
3. **Incorporates temporal information** from previous frames
4. **Uses multi-head architecture** for different clinical reasoning stages

## Architecture Overview

```
Input: Frame Image + Previous Frame Info
         ↓
   Vision Encoder
         ↓
   LLM Backbone
    ↙    ↓    ↘
Head 1  Head 2  Head 3
(Abnorm) (Char) (Treat)
```

### Three Specialized Heads

- **Head 1 (Abnormality Detection)**: Detects abnormalities, instruments, polyps, lesions
- **Head 2 (Characteristics)**: Analyzes properties (color, location, type, count)
- **Head 3 (Treatment)**: Provides clinical context, diagnosis, treatment recommendations

## Installation

```bash
# Install dependencies
pip install torch transformers peft pillow opencv-python numpy tqdm

# For vision-language models
pip install accelerate bitsandbytes
```

## Quick Start

### 1. Categorize Questions

First, categorize questions into clinical stages:

```bash
python data/question_categorizer.py \
    --input datasets/Kvasir-VQA/train.json \
    --output data/categorized \
    --dataset kvasir \
    --model Qwen/Qwen2.5-7B-Instruct
```

### 2. Create Temporal Structure (EndoVis only)

For video sequences, create temporal links:

```bash
python data/temporal_linker.py \
    --sequence-dir datasets/EndoVis2018/sequences \
    --qa-file datasets/EndoVis2018/train.json \
    --output data/temporal_structure.json \
    --sequence-id seq_1
```

### 3. Train Model

#### Unified Multi-Head Training

```bash
python train_multihead_temporal_cot.py \
    --base-model Qwen/Qwen2-VL-2B-Instruct \
    --train-data data/categorized/train_categorized.json \
    --val-data data/categorized/val_categorized.json \
    --image-base-path datasets/Kvasir-VQA/raw/images \
    --dataset kvasir \
    --training-mode unified \
    --num-epochs 10 \
    --batch-size 4 \
    --use-lora \
    --output-dir checkpoints/kvasir_unified
```

#### Sequential Curriculum Learning

```bash
python train_multihead_temporal_cot.py \
    --base-model Qwen/Qwen2-VL-2B-Instruct \
    --train-data data/categorized/train_categorized.json \
    --val-data data/categorized/val_categorized.json \
    --image-base-path datasets/Kvasir-VQA/raw/images \
    --dataset kvasir \
    --training-mode sequential \
    --num-epochs 5 \
    --batch-size 4 \
    --use-lora \
    --output-dir checkpoints/kvasir_sequential
```

### 4. Evaluate

```bash
python evaluation/baseline_comparison.py \
    --model-path checkpoints/kvasir_unified/best_model.pt \
    --test-data data/categorized/test_categorized.json \
    --output results/kvasir_evaluation
```

## Dataset Support

### Kvasir-VQA

- **Type**: Single-frame endoscopic images
- **Format**: JSON with `question`, `answer`, `image_filename`
- **Stages**: All 3 stages supported (abnormality → characteristics → treatment)
- **Temporal**: Not applicable (single frames)

### EndoVis 2018

- **Type**: Video sequences (surgical scenes)
- **Format**: JSON with `question`, `answer`, `frame_id`, `sequence_id`
- **Stages**: All 3 stages supported
- **Temporal**: Full temporal CoT with motion computation
- **Test Split**: Sequences 1, 5, 16 (as specified)

## Key Features

### 1. Model-Generated CoT

The system uses **hybrid CoT prompts** that:
- ✅ Guide clinical structure implicitly
- ✅ Allow model to generate its own reasoning flow
- ❌ Do NOT prescribe step-by-step instructions

Example prompt structure:
```
You are analyzing a surgical/endoscopic image.

Question: What is the color of the polyp?

Context from previous frame:
- Previous observations: A polyp was detected in the upper left quadrant
- Camera movement: Small camera movement (down-right), slight adjustment

Based on the identified findings, analyze their specific properties 
such as color, location, size, type, or quantity.

Think through your reasoning step by step, then provide your answer.

Reasoning: [MODEL GENERATES]
```

### 2. Temporal CoT Integration

For video sequences (EndoVis), the system:
- Computes optical flow between consecutive frames
- Describes camera movement and scene changes
- Passes previous frame predictions as context
- Enables reasoning that considers temporal evolution

### 3. Multi-Head Architecture

Each head is specialized:
- **Head 1**: Focuses on detection tasks
- **Head 2**: Reuses Head 1 predictions, focuses on characterization
- **Head 3**: Reuses Head 1 & 2 predictions, focuses on clinical context

### 4. Training Strategies

**Sequential Curriculum Learning**:
1. Train Head 1 (abnormality detection)
2. Freeze Head 1, train Head 2 (characteristics) with Head 1 predictions
3. Freeze Head 1 & 2, train Head 3 (treatment) with Head 1 & 2 predictions

**Unified Multi-Head Training**:
- Train all heads simultaneously
- Weighted loss across stages
- Shared vision encoder and LLM backbone

## File Structure

```
Surgical_COT/
├── data/
│   ├── question_categorizer.py      # LLM-based question categorization
│   ├── temporal_linker.py            # Temporal structure creation
│   └── vqa_data_loader.py            # Data loaders with temporal support
├── models/
│   └── multi_head_model.py           # Multi-head architecture
├── prompts/
│   └── cot_templates.py              # Hybrid CoT prompt templates
├── training/
│   └── temporal_trainer.py           # Training infrastructure
├── evaluation/
│   └── baseline_comparison.py        # Evaluation and comparison
└── train_multihead_temporal_cot.py  # Main training script
```

## Configuration

### Model Selection

Supported base models:
- `Qwen/Qwen2-VL-2B-Instruct` (recommended for efficiency)
- `Qwen/Qwen2-VL-7B-Instruct` (better performance)
- `microsoft/llava-med-v1.5-mistral-7b` (medical domain)
- `google/gemma-2-4b-it` (alternative)

### LoRA Configuration

Default LoRA settings:
- Rank (r): 8
- Alpha: 16
- Dropout: 0.05
- Target modules: `["q_proj", "v_proj", "k_proj", "o_proj"]`

### Training Hyperparameters

Recommended settings:
- Learning rate: 2e-5
- Batch size: 4-8 (depending on GPU memory)
- Gradient accumulation: 4-8 steps
- Max gradient norm: 1.0
- Epochs: 5-10 per stage (sequential) or 10-20 (unified)

## Evaluation Metrics

The system tracks:
- Overall accuracy
- Stage-specific accuracy (Stage 1, 2, 3)
- Per-category accuracy
- CoT quality metrics (reasoning coherence, clinical relevance)

## Baseline Comparisons

Compare against:
1. **Baseline (No CoT)**: Standard VQA without CoT reasoning
2. **Temporal Only**: CoT with temporal context, single head
3. **Multi-Head Only**: Multi-head without temporal context
4. **Full System**: Multi-head + temporal CoT

## Ablation Studies

Run ablation to test:
- Contribution of temporal context
- Contribution of multi-head architecture
- Sequential vs. unified training
- Different base models

## Troubleshooting

### Memory Issues

- Reduce batch size
- Increase gradient accumulation steps
- Use smaller base model (2B instead of 7B)
- Enable gradient checkpointing

### Slow Training

- Use smaller image resolution (e.g., 224 instead of 448)
- Reduce LoRA rank (r=4 instead of r=8)
- Use mixed precision training (bfloat16)

### Question Categorization Errors

- Check cache file for incorrect classifications
- Use rule-based fallback if LLM unavailable
- Manually review and correct category mappings

## Citation

If you use this implementation, please cite:

```bibtex
@misc{surgical_cot_vqa,
  title={Multi-Head Temporal Chain-of-Thought for Surgical Visual Question Answering},
  author={Your Name},
  year={2025}
}
```

## License

See LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub.














