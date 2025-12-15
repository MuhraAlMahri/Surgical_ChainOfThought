# EndoVis2018 Dataset Split Information

## ✅ Proper Sequence-Based Split (No Overlap)

This dataset uses a **sequence-based split** following surgical dataset best practices (similar to Surgery R1 paper methodology).

### Split Methodology

- **No sequence overlap** between train/val/test (prevents data leakage)
- **Test sequences preserved**: Sequences 1, 2, 3, 4 (as in original dataset)
- **Train/Val split**: Properly split from remaining sequences

### Split Distribution

| Split | Sequences | Count | Percentage |
|-------|-----------|-------|------------|
| **Train** | 5, 6, 7, 10, 11, 12, 14, 15, 16 | 9 sequences | ~60% |
| **Validation** | 9, 13 | 2 sequences | ~13% |
| **Test** | 1, 2, 3, 4 | 4 sequences | ~27% |
| **Total** | 1-16 (excluding 8) | 15 sequences | 100% |

### Files

- `train_data_sequence_based.txt` - Training samples
- `validation_data_sequence_based.txt` - Validation samples  
- `test_data_sequence_based.txt` - Test samples
- `split_metadata_sequence_based.json` - Split metadata

### Why Sequence-Based Split?

1. **Prevents Data Leakage**: Frames from the same sequence are temporally related
2. **Better Generalization**: Model evaluated on completely unseen sequences
3. **Standard Practice**: Common in surgical/medical datasets
4. **Reproducible**: Fixed random seed (42) ensures consistency

### Usage

The organization script (`scripts/organize_endovis2018.py`) automatically uses these split files when organizing the dataset.

### Verification

```bash
# Check split metadata
cat EndoVis2018/data/index/split_metadata_sequence_based.json

# Verify no overlap
python -c "
import json
with open('EndoVis2018/data/index/split_metadata_sequence_based.json') as f:
    data = json.load(f)
    train = set(data['sequences']['train'])
    val = set(data['sequences']['validation'])
    test = set(data['sequences']['test'])
    print('Overlap check:')
    print(f'  Train-Val: {train & val}')
    print(f'  Train-Test: {train & test}')
    print(f'  Val-Test: {val & test}')
"
```

Expected output: All overlaps should be empty sets `set()`.





















