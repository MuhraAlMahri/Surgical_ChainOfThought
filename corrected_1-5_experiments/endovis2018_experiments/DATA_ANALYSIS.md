# EndoVis2018 Data Analysis

## 📊 Extraction Results

### Zip Files Extracted
1. `miccai_challenge_2018_release_1.zip` - Sequences 1-4
2. `miccai_challenge_release_2.zip` - Sequences 5-7
3. `miccai_challenge_release_3.zip` - Sequences 9-12
4. `miccai_challenge_release_4.zip` - Sequences 13-16
5. `repairs.zip` - Additional data

### Total Images Extracted
- **Total**: 9,499 images
- **Organized**: All images organized into flat structure

### Sequences Available
All sequences 1-16 (except 8, which is missing):
- **seq_1**: 149 images (test)
- **seq_2**: 149 images (test)
- **seq_3**: 149 images (test)
- **seq_4**: 149 images (test)
- **seq_5**: 149 images (train)
- **seq_6**: 149 images (train)
- **seq_7**: 149 images (train)
- **seq_9**: 149 images (validation)
- **seq_10**: 149 images (train)
- **seq_11**: 149 images (train)
- **seq_12**: 149 images (train)
- **seq_13**: 149 images (validation)
- **seq_14**: 149 images (train)
- **seq_15**: 149 images (train)
- **seq_16**: 149 images (train)

## 📋 VQA Data Prepared

### Split Distribution (Sequence-Based, No Overlap)
- **Train**: 5,364 samples (9 sequences: 5, 6, 7, 10, 11, 12, 14, 15, 16)
- **Validation**: 1,192 samples (2 sequences: 9, 13)
- **Test**: 2,384 samples (4 sequences: 1, 2, 3, 4)
- **Total**: 8,940 VQA samples

### Question Categories
1. **INSTRUMENT_DETECTION** (multi_label)
2. **ANATOMY_DETECTION** (multi_label)
3. **INSTRUMENT_COUNT** (single_choice, numeric)
4. **PROCEDURE_TYPE** (single_choice)

### Sample Structure
Each JSONL line contains:
```json
{
  "image_id": "endovis_seq5_frame000",
  "image_filename": "endovis_seq_5_frame000.png",
  "question": "What surgical instruments are visible in this image?",
  "answer": "none",
  "question_type": "instrument_detection",
  "category": "INSTRUMENT_DETECTION",
  "sequence": "5",
  "frame": "000",
  "dataset": "EndoVis2018"
}
```

## ✅ Data Files Created

- `train.jsonl` - 5,364 samples (1.6MB)
- `validation.jsonl` - 1,192 samples (356KB)
- `test.jsonl` - 2,384 samples (709KB)
- `INSTRUCTIONS_PER_CATEGORY.txt` - Instruction templates

## 🚀 Jobs Submitted

**Job ID**: 159676

Running on 4 GPUs in parallel:
- GPU 0: Zero-shot evaluation
- GPU 1: Training Exp1 (Random Baseline)
- GPU 2: Training Exp2 (Qwen Reordered)
- GPU 3: Training Exp3 (Sequential)

## 📝 Notes

- All images extracted and organized
- Proper sequence-based split (no overlap)
- VQA questions generated from segmentation masks
- Ready for training and evaluation




















