# Data Directory

This directory contains the training and test data for the CXRTrek Sequential and Curriculum Learning experiments.

## 📊 Dataset Overview

**Source**: Kvasir-VQA (Surgical/Endoscopic Visual Question Answering)
**Total Samples**: 5,266 image-question-answer triplets
**Split**: 80% train (4,213 samples), 20% test (1,053 samples)

## 📁 File Structure

```
data/
├── qa_pairs_train.json      # Original training data (all stages mixed)
├── qa_pairs_test.json       # Original test data (all stages mixed)
├── stage1_train.json        # Stage 1 training samples (Initial Assessment)
├── stage1_test.json         # Stage 1 test samples
├── stage2_train.json        # Stage 2 training samples (Findings)
├── stage2_test.json         # Stage 2 test samples
├── stage3_train.json        # Stage 3 training samples (Clinical Context)
├── stage3_test.json         # Stage 3 test samples
└── README.md                # This file
```

## 📋 Data Format

### Original QA Pairs (`qa_pairs_train.json`, `qa_pairs_test.json`)

```json
[
  {
    "image": "image_001.jpg",
    "question": "What type of procedure is shown in the image?",
    "answer": "Colonoscopy"
  },
  {
    "image": "image_002.jpg",
    "question": "What abnormality is visible?",
    "answer": "Polyp"
  }
]
```

### Stage-Specific Files (`stage{1,2,3}_{train,test}.json`)

```json
[
  {
    "image": "image_001.jpg",
    "question": "What type of procedure is shown in the image?",
    "answer": "Colonoscopy",
    "stage": 1,
    "stage_name": "Initial Assessment"
  }
]
```

## 🏥 Clinical Stage Definitions

### Stage 1: Initial Assessment (~38% of data)
**Purpose**: Quality control, procedure identification, artifact detection

**Example Questions**:
- "What type of procedure is shown in the image?"
- "Is there any text visible in the image?"
- "Are there any artifacts present?"
- "What is the image quality?"

**Sample Counts**:
- Training: ~1,600 samples
- Test: 1,586 samples

### Stage 2: Findings Identification (~55% of data)
**Purpose**: Abnormalities, instruments, anatomical landmarks

**Example Questions**:
- "What abnormality is visible in the image?"
- "Where is the polyp located?"
- "What instruments are present?"
- "Describe the anatomical landmarks visible"
- "Is there evidence of bleeding?"

**Sample Counts**:
- Training: ~2,300 samples
- Test: 2,249 samples

### Stage 3: Clinical Context (~7% of data)
**Purpose**: Diagnosis, clinical reasoning, treatment recommendations

**Example Questions**:
- "What is the most likely diagnosis?"
- "What treatment is recommended?"
- "Have all polyps been removed?"
- "What is the clinical significance of this finding?"
- "What is the next step in management?"

**Sample Counts**:
- Training: ~300 samples
- Test: 279 samples

## 🤖 Stage Categorization Process

Questions were categorized into stages using **Qwen2.5-7B-Instruct** LLM with the following process:

1. **Prompt Engineering**: Structured prompt with clear definitions and examples for each stage
2. **LLM Inference**: Each question processed through Qwen2.5-7B
3. **Validation**: Manual spot-checking of categorizations
4. **Split Creation**: Separate train/test files created for each stage

### Categorization Script

```bash
python scripts/categorize_questions_llm.py \
    --input_file data/qa_pairs_train.json \
    --output_dir data/ \
    --model_name "Qwen/Qwen2.5-7B-Instruct"
```

## 📈 Data Statistics

### Overall Distribution
```
Total Samples: 5,266
├── Training: 4,213 (80%)
└── Test:     1,053 (20%)

Stage Distribution (Test Set):
├── Stage 1: 1,586 samples (38.5%)
├── Stage 2: 2,249 samples (54.6%)
└── Stage 3:   279 samples ( 6.9%)
```

### Question Length Statistics
```
Average question length: ~12 words
Average answer length: ~5 words
Max question length: ~50 words
Max answer length: ~30 words
```

### Image Statistics
```
Total unique images: ~1,000
Images per stage:
├── Stage 1: ~600 images
├── Stage 2: ~800 images
└── Stage 3: ~300 images
```

## 🔒 Data Privacy

**Important**: Medical images should be handled according to HIPAA and institutional guidelines.

- All images are de-identified
- No patient information included
- Public dataset (Kvasir) used for research purposes
- Follow your institution's IRB requirements

## 📥 Data Access

### Downloading the Dataset

```bash
# Download Kvasir-VQA dataset
cd experiments/cxrtrek_curriculum_learning/data
wget [KVASIR_VQA_URL] -O kvasir_vqa.zip
unzip kvasir_vqa.zip

# Verify data
python scripts/verify_data.py
```

### Data Preparation

```bash
# Categorize questions into stages
python scripts/categorize_questions_llm.py \
    --input_file data/qa_pairs_train.json \
    --output_dir data/

# This creates:
# - stage1_train.json, stage1_test.json
# - stage2_train.json, stage2_test.json
# - stage3_train.json, stage3_test.json
```

## 🔄 Data Augmentation (Optional)

Future improvements could include:
- Image augmentation (rotation, brightness, contrast)
- Question paraphrasing
- Synthetic data generation
- Cross-dataset mixing

## 📊 Data Quality

### Quality Checks Performed
✅ No duplicate image-question pairs
✅ All images exist and are readable
✅ All answers are non-empty
✅ Questions are grammatically correct
✅ Stage categorization validated

### Known Issues
- Some questions may be ambiguous between stages
- Stage 3 has fewer samples (clinical questions are less common)
- Some images appear in multiple stages with different questions

## 🛠️ Data Utils

### Verifying Data Integrity

```python
import json
from pathlib import Path

def verify_data_files():
    """Verify all data files exist and are valid."""
    data_dir = Path("data")
    
    files = [
        "qa_pairs_train.json",
        "qa_pairs_test.json",
        "stage1_train.json",
        "stage1_test.json",
        "stage2_train.json",
        "stage2_test.json",
        "stage3_train.json",
        "stage3_test.json"
    ]
    
    for fname in files:
        fpath = data_dir / fname
        assert fpath.exists(), f"Missing: {fname}"
        
        with open(fpath) as f:
            data = json.load(f)
            print(f"✓ {fname}: {len(data)} samples")

verify_data_files()
```

## 📝 Citation

If you use this dataset, please cite:

```bibtex
@article{kvasir_vqa,
  title={Kvasir-VQA: A Visual Question Answering Dataset for Gastrointestinal Endoscopy},
  author={[Kvasir Authors]},
  journal={[Journal]},
  year={2023}
}
```

## 🤝 Contributing

To add new data or improve data quality:
1. Follow the existing data format
2. Validate stage categorizations
3. Update statistics in this README
4. Submit a pull request

## 📧 Contact

Questions about the data:
- Open an issue on GitHub
- GitHub: [@MuhraAlMahri](https://github.com/MuhraAlMahri)

---

**Last Updated**: October 20, 2025

