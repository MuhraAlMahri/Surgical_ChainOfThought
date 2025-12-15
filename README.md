# Surgical Chain-of-Thought (COT) for Medical Visual Question Answering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Comprehensive evaluation of Vision-Language Models for surgical and endoscopic Visual Question Answering**

This repository contains experimental results and implementations for medical Visual Question Answering (VQA) using state-of-the-art vision-language models, focusing on surgical and endoscopic image analysis.

## 🏆 Key Results

### Experiment Results Summary (Exp1-Exp5)

| Experiment | Zero-Shot Accuracy | Instruction Fine-Tuned Accuracy |
|------------|-------------------|--------------------------------|
| **Exp1** | 53.48% (4,805/8,984) | **92.79%** (8,336/8,984) |
| **Exp2** | 53.48% (4,805/8,984) | **92.76%** (8,334/8,984) |
| **Exp3** | 53.48% (4,805/8,984) | **92.23%** (8,290/8,984) |
| **Exp4** | 53.48% (4,805/8,984) | **92.44%** (8,305/8,984) |
| **Exp5** | 53.48% (4,805/8,984) | **92.62%** (8,321/8,984) |

**All experiments use Qwen3-VL-8B-Instruct as the base model.**

### Model Performance Comparison

| Model | Dataset | COT | Zeroshot | Instruction Fine-Tuned |
|-------|---------|-----|----------|----------------------|
| **Qwen3-VL-8B-Instruct** | Kvasir | No | 53.48% (4,805/8,984) | **92.79%** (8,336/8,984) |
| **Qwen3-VL-8B-Instruct** | EndoVis18 | No | 31.12% (742/2,384) | **95.18%** (2,269/2,384) |
| **MedGemma-4B** | Kvasir | No | 32.05% (2,879/8,984) | **91.90%** (8,256/8,984) |
| **MedGemma-4B** | EndoVis18 | No | 25.08% (598/2,384) | **99.83%** (2,380/2,384) |
| **LLaVA-Med v1.5** | Kvasir | No | 72.01% (6,469/8,984) | 70.27% (6,313/8,984) |
| **LLaVA-Med v1.5** | EndoVis18 | No | - | - |

---

## 📋 Table of Contents

- [Overview](#overview)
- [Experiments](#experiments)
- [Installation](#installation)
- [Datasets](#datasets)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## 🔬 Overview

This project evaluates multiple vision-language models on medical VQA tasks using two surgical/endoscopic datasets:

1. **Kvasir-VQA**: 8,984 test samples
2. **EndoVis2018**: 2,384 test samples

We compare:
- **Zero-shot performance**: Base models without fine-tuning
- **Instruction fine-tuning**: Models fine-tuned with instruction templates using QLoRA

---

## 🧪 Experiments

### Experiment 1: Random Baseline
- **Model**: Qwen3-VL-8B-Instruct
- **Approach**: Standard training with random question ordering
- **Zero-Shot**: 53.48% (4,805/8,984)
- **Instruction Fine-Tuned**: 92.79% (8,336/8,984)

### Experiment 2: Qwen Clinical Reordering
- **Model**: Qwen3-VL-8B-Instruct
- **Approach**: Questions ordered by Qwen clinical stages (Stage1→Stage2→Stage3)
- **Zero-Shot**: 53.48% (4,805/8,984)
- **Instruction Fine-Tuned**: 92.76% (8,334/8,984)

### Experiment 3: CXRTrek Sequential
- **Model**: Qwen3-VL-8B-Instruct
- **Approach**: Three specialized models, one per clinical stage
- **Zero-Shot**: 53.48% (4,805/8,984)
- **Instruction Fine-Tuned**: 92.23% (8,290/8,984)

### Experiment 4: Curriculum Learning
- **Model**: Qwen3-VL-8B-Instruct
- **Approach**: Progressive training through stages
- **Zero-Shot**: 53.48% (4,805/8,984)
- **Instruction Fine-Tuned**: 92.44% (8,305/8,984)

### Experiment 5: Sequential Chain-of-Thought
- **Model**: Qwen3-VL-8B-Instruct
- **Approach**: Sequential CoT training strategy
- **Zero-Shot**: 53.48% (4,805/8,984)
- **Instruction Fine-Tuned**: 92.62% (8,321/8,984)

---

## 🚀 Quick Start

### Reproducing an Experiment

Each experiment has its own directory with a README containing detailed reproduction instructions. For example:

```bash
# Navigate to experiment directory
cd experiments/exp1_random

# Read the experiment-specific README
cat README.md

# Run zero-shot evaluation
python ../scripts/evaluation/evaluate_zeroshot.py \
    --config configs/exp1_random.yaml \
    --output results/exp1_zeroshot.json

# Train the model
python ../scripts/training/train_instruction_finetuning.py \
    --config configs/exp1_random.yaml

# Evaluate after training
python ../scripts/evaluation/evaluate_exp1.py \
    --config configs/exp1_random.yaml \
    --checkpoint <checkpoint_path> \
    --output results/instruction_finetuned.json
```

See each experiment's README for specific instructions.

---

## 🚀 Installation

### Prerequisites

```bash
# Python 3.8+
# CUDA 11.8+ (for GPU support)
# 40GB+ GPU memory (for Qwen3-VL-8B-Instruct)
```

### Setup

```bash
# Clone the repository
git clone https://github.com/MuhraAlMahri/Surgical_ChainOfThought.git
cd Surgical_ChainOfThought

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
torch>=2.0.0
transformers>=4.37.0
peft>=0.8.0
qwen-vl-utils
accelerate>=0.26.0
pillow>=10.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

---

## 📊 Datasets

### Kvasir-VQA
- **Total Test Samples**: 8,984
- **Images**: Endoscopic/surgical images from Kvasir dataset
- **Split**: 80% train, 20% test
- **Location**: `datasets/Kvasir-VQA/`

### EndoVis2018
- **Total Test Samples**: 2,384
- **Images**: Surgical images from EndoVis 2018 challenge
- **Location**: `datasets/EndoVis2018/`

---

## 📈 Results

### Detailed Results Location

All experimental results are available in the repository:

- **Exp1-Exp5 Results**: `corrected_1-5_experiments/qlora_experiments/results/`
  - `exp1_zeroshot.json`, `exp1_evaluation.json`
  - `exp2_zeroshot.json`, `exp2_evaluation.json`
  - `exp3_zeroshot.json`, `exp3_evaluation.json`
  - `exp4_zeroshot.json`, `exp4_evaluation.json`
  - `exp5_zeroshot.json`, `exp5_evaluation.json`

- **Baseline Results**: `results/baseline_results.json`
  - Contains summary results for all models and datasets

- **Qwen3-VL-8B Results**: `results/eval_epoch*_qwen3vl_kvasir/`
  - Detailed evaluation results for Qwen3-VL-8B

- **LLaVA-Med Results**: `corrected_1-5_experiments/qlora_experiments/results/`
  - `kvasir_zeroshot_llava_med_v15.json`
  - `kvasir_finetuned_llava_med_v15.json`
  - `endovis_zeroshot_llava_med_v15.json`
  - `endovis_finetuned_llava_med_v15.json`

### Key Observations

1. **Instruction Fine-Tuning Provides Significant Gains**
   - All experiments show ~40 percentage point improvement over zero-shot
   - Qwen3-VL-8B achieves 92.79% on Kvasir (vs 53.48% zero-shot)

2. **MedGemma-4B Excels on EndoVis**
   - Achieves 99.83% accuracy on EndoVis2018 after fine-tuning
   - Best performing model-dataset combination

3. **LLaVA-Med Shows Strong Zero-Shot Performance**
   - 72.01% zero-shot on Kvasir (highest among all models)
   - However, fine-tuning shows slight decrease (70.27%)

4. **Consistent Zero-Shot Performance**
   - All Exp1-Exp5 show identical 53.48% zero-shot (same base model)
   - Differences appear only after instruction fine-tuning

---

## 🛠️ Technical Details

### Models Evaluated

1. **Qwen3-VL-8B-Instruct** (8 billion parameters)
   - Vision-language model from Qwen team
   - Used in all Exp1-Exp5 experiments

2. **MedGemma-4B** (4 billion parameters)
   - Medical domain specialized model
   - Strong performance on medical VQA tasks

3. **LLaVA-Med v1.5** (Mistral-7B based)
   - Medical adaptation of LLaVA
   - Excellent zero-shot capabilities

### Fine-Tuning Approach

- **Method**: QLoRA (Quantized Low-Rank Adaptation)
- **LoRA Rank**: 4-8 (varies by experiment)
- **LoRA Alpha**: 8-16
- **Epochs**: 5
- **Precision**: bfloat16 (mixed precision training)

---

## 📂 Repository Structure

```
Surgical_ChainOfThought/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
│
├── experiments/                        # All experiments organized by type
│   ├── exp1_random/                   # Experiment 1: Random Baseline
│   │   ├── README.md                  # Reproduction instructions
│   │   ├── configs/
│   │   │   └── exp1_random.yaml      # Training configuration
│   │   └── results/
│   │       ├── exp1_zeroshot.json    # Zero-shot results
│   │       └── instruction_finetuned.json  # Fine-tuned results
│   │
│   ├── exp2_qwen_reordered/           # Experiment 2: Qwen Reordering
│   │   ├── README.md
│   │   ├── configs/
│   │   └── results/
│   │
│   ├── exp3_sequential/               # Experiment 3: CXRTrek Sequential
│   │   ├── README.md
│   │   ├── configs/
│   │   │   ├── exp3_stage1.yaml
│   │   │   ├── exp3_stage2.yaml
│   │   │   └── exp3_stage3.yaml
│   │   └── results/
│   │
│   ├── exp4_curriculum/               # Experiment 4: Curriculum Learning
│   │   ├── README.md
│   │   ├── configs/
│   │   └── results/
│   │
│   └── exp5_sequential_cot/           # Experiment 5: Sequential CoT
│       ├── README.md
│       ├── configs/
│       └── results/
│
├── scripts/                            # Reusable scripts
│   ├── README.md                      # Scripts overview
│   ├── training/                      # Training scripts
│   │   ├── train_instruction_finetuning.py
│   │   └── train_qlora_qwen3vl.py
│   ├── evaluation/                    # Evaluation scripts
│   │   ├── evaluate_exp1.py
│   │   ├── evaluate_exp2.py
│   │   ├── evaluate_exp3.py
│   │   ├── evaluate_exp4.py
│   │   ├── evaluate_exp5.py
│   │   ├── evaluate_zeroshot.py
│   │   ├── evaluate_finetuned_llava.py
│   │   └── metrics_utils.py
│   └── data_preparation/              # Data preparation scripts
│       ├── prepare_all_datasets_qlora.py
│       └── create_stage_splits.py
│
├── results/                            # All experimental results
│   ├── baseline/                      # Baseline model comparisons
│   │   └── baseline_results.json
│   ├── llava_med/                     # LLaVA-Med specific results
│   └── ...                            # Other result files
│
└── datasets/                           # Dataset metadata
    ├── Kvasir-VQA/                    # Kvasir dataset
    └── EndoVis2018/                   # EndoVis dataset
```

---

## 📄 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{surgical_cot_2025,
  title={Surgical Chain-of-Thought: Vision-Language Models for Medical VQA},
  author={Muhra Al Mahri},
  journal={[Conference/Journal]},
  year={2025},
  note={Available at: https://github.com/MuhraAlMahri/Surgical_ChainOfThought}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Qwen Team** for the Qwen3-VL-8B-Instruct model
- **Google** for MedGemma-4B
- **LLaVA Team** for LLaVA-Med v1.5
- **Kvasir Dataset** authors for the medical image dataset
- **EndoVis Challenge** organizers for the EndoVis2018 dataset
- **Hugging Face** for transformers and PEFT libraries

---

## 📧 Contact

For questions or collaborations:
- **GitHub**: [@MuhraAlMahri](https://github.com/MuhraAlMahri)
- **GitHub Issues**: [Create an issue](https://github.com/MuhraAlMahri/Surgical_ChainOfThought/issues)

---

## 🔗 Related Work

- [Qwen3-VL: Vision-Language Models](https://github.com/QwenLM/Qwen3-VL)
- [MedGemma](https://github.com/google-research/google-research/tree/master/medgemma)
- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
- [Kvasir Dataset](https://datasets.simula.no/kvasir/)
- [EndoVis Challenge](https://endovissub-instrument.grand-challenge.org/)

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ for advancing medical AI

</div>
