# ✅ IMPLEMENTATION CHECKLIST

Print this out and check off each item as you complete it!

---

## 📋 PRE-IMPLEMENTATION

- [ ] Downloaded all 6 files from `/mnt/user-data/outputs/`
  - [ ] `preprocess_data_with_instructions.py`
  - [ ] `evaluate_improved.py`
  - [ ] `QUICK_START_GUIDE.md`
  - [ ] `EXACT_CODE_MODIFICATIONS.md`
  - [ ] `DIAGNOSIS_AND_FIX_GUIDE.md`
  - [ ] `COMPLETE_FIX_SUMMARY.md`

- [ ] Read `QUICK_START_GUIDE.md` for overview
- [ ] Understood the problem from `DIAGNOSIS_AND_FIX_GUIDE.md`
- [ ] Have access to cluster
- [ ] Located original data files

---

## 🔄 STEP 1: DATA PREPROCESSING (30 min)

- [ ] Upload `preprocess_data_with_instructions.py` to cluster
  ```bash
  scp preprocess_data_with_instructions.py user@cluster:~/path/
  ```

- [ ] Verify input data exists
  ```bash
  ls datasets/kvasir_raw_6500_image_level_70_15_15/train.json
  ls datasets/kvasir_raw_6500_image_level_70_15_15/val.json
  ls datasets/kvasir_raw_6500_image_level_70_15_15/test.json
  ```

- [ ] Create output directory
  ```bash
  mkdir -p datasets/kvasir_instructed
  ```

- [ ] Run preprocessing script
  ```bash
  python preprocess_data_with_instructions.py \
      --input_dir datasets/kvasir_raw_6500_image_level_70_15_15 \
      --output_dir datasets/kvasir_instructed \
      --files train.json val.json test.json
  ```

- [ ] Verify output files created
  ```bash
  ls -lh datasets/kvasir_instructed/train_instructed.json
  ls -lh datasets/kvasir_instructed/val_instructed.json
  ls -lh datasets/kvasir_instructed/test_instructed.json
  ```

- [ ] Check sample output has required fields
  ```bash
  head -50 datasets/kvasir_instructed/train_instructed.json
  ```
  Should see: `instruction`, `question_type`, `question`, `answer`

---

## 🔧 STEP 2: MODIFY TRAINING SCRIPT (15 min)

- [ ] Backup original training script
  ```bash
  cp train_qwen_lora.py train_qwen_lora.py.backup
  ```

- [ ] Open `train_qwen_lora.py` in editor
  ```bash
  nano train_qwen_lora.py
  # or use your preferred editor
  ```

### Modification A: LazyVQADataset (around line 430-435)

- [ ] Find the section:
  ```python
  self.items.append({
      'question': item.get('question'),
      'answer': item.get('answer'),
      'image_id': item.get('image_id'),
      'image_path': image_path
  })
  ```

- [ ] Add two new fields:
  ```python
  self.items.append({
      'question': item.get('question'),
      'answer': item.get('answer'),
      'instruction': item.get('instruction', item.get('question')),  # NEW
      'question_type': item.get('question_type', 'open_short'),     # NEW
      'image_id': item.get('image_id'),
      'image_path': image_path
  })
  ```

- [ ] Verify change saved

### Modification B: LazyVQACollator (around line 459-520)

- [ ] Find the __call__ method

- [ ] Change variable name (line ~461):
  ```python
  # OLD: questions: List[str] = []
  instructions: List[str] = []  # NEW
  ```

- [ ] Change feature extraction (line ~467):
  ```python
  # OLD: question = feat.get('question', '')
  instruction = feat.get('instruction', feat.get('question', ''))  # NEW
  ```

- [ ] Change append (line ~484):
  ```python
  # OLD: questions.append(question)
  instructions.append(instruction)  # NEW
  ```

- [ ] Change loop (line ~489):
  ```python
  # OLD: for question, answer in zip(questions, answers):
  for instruction, answer in zip(instructions, answers):  # NEW
  ```

- [ ] Change message content (line ~496):
  ```python
  # OLD: {"type": "text", "text": question}
  {"type": "text", "text": instruction}  # NEW
  ```

- [ ] Verify all changes saved

- [ ] Test grep to confirm changes:
  ```bash
  grep "instruction = feat.get('instruction'" train_qwen_lora.py
  grep "instructions: List\[str\]" train_qwen_lora.py
  ```
  Both should return results!

---

## 📝 STEP 3: UPDATE SLURM SCRIPT (5 min)

- [ ] Backup original SLURM script
  ```bash
  cp experiments/exp1_random/train_exp1.slurm \
     experiments/exp1_random/train_exp1.slurm.backup
  ```

- [ ] Open SLURM script
  ```bash
  nano experiments/exp1_random/train_exp1.slurm
  ```

- [ ] Find training command section

- [ ] Update `--train_file` path:
  ```bash
  # OLD: --train_file datasets/kvasir_raw_6500.../train.json
  --train_file datasets/kvasir_instructed/train_instructed.json
  ```

- [ ] Update `--val_file` path:
  ```bash
  # OLD: --val_file datasets/kvasir_raw_6500.../val.json
  --val_file datasets/kvasir_instructed/val_instructed.json
  ```

- [ ] Save changes

- [ ] Verify changes:
  ```bash
  grep "kvasir_instructed" experiments/exp1_random/train_exp1.slurm
  ```
  Should show both train and val files!

---

## 🚀 STEP 4: LAUNCH TRAINING (1-2 hours)

- [ ] Double-check all prerequisites:
  ```bash
  # Data
  ls datasets/kvasir_instructed/train_instructed.json
  ls datasets/kvasir_instructed/val_instructed.json
  
  # Code modifications
  grep "instruction = feat.get" train_qwen_lora.py
  
  # SLURM script
  grep "kvasir_instructed" experiments/exp1_random/train_exp1.slurm
  ```

- [ ] Submit training job
  ```bash
  sbatch experiments/exp1_random/train_exp1.slurm
  ```

- [ ] Note job ID: _________________

- [ ] Check job started
  ```bash
  squeue -u $USER
  ```

- [ ] Monitor training log
  ```bash
  tail -f experiments/exp1_random/logs/train_*.out
  ```

### Training Monitoring Checklist

- [ ] Training started (no immediate errors)
- [ ] Initial loss: ______ (should be 4-6)
- [ ] After 100 steps: loss = ______ (should be dropping)
- [ ] After 300 steps: loss = ______ (should be <3.0)
- [ ] After 500 steps: loss = ______ (should be <2.0) ✓
- [ ] Training completed successfully
- [ ] Final loss: ______ (should be 0.8-1.5)

---

## 📊 STEP 5: EVALUATION (15 min)

- [ ] Upload evaluation script
  ```bash
  scp evaluate_improved.py user@cluster:~/evaluation/
  ```

- [ ] Find best checkpoint
  ```bash
  ls -lht experiments/exp1_random/models/checkpoint-*
  ```
  Best checkpoint: _________________

- [ ] Run evaluation
  ```bash
  python evaluation/evaluate_improved.py \
      --model_path experiments/exp1_random/models/checkpoint-XXXX \
      --test_data datasets/kvasir_instructed/test_instructed.json \
      --image_dir datasets/Kvasir-VQA/raw/images \
      --output results/exp1_improved_results.json \
      --base_model Qwen/Qwen2-VL-7B-Instruct
  ```

- [ ] Evaluation completed

- [ ] Check results file
  ```bash
  ls -lh results/exp1_improved_results.json
  ```

---

## ✅ STEP 6: VERIFY SUCCESS (10 min)

### Check Overall Metrics

- [ ] Overall accuracy: ______% (should be >60%)
- [ ] Binary accuracy: ______% (should be >85%)
- [ ] Numeric accuracy: ______% (should be >65%)
- [ ] Color accuracy: ______% (should be >70%)

### Check Prediction Quality

- [ ] Open results file
  ```bash
  less results/exp1_improved_results.json
  ```

- [ ] Pick 5 random predictions and verify:
  
  1. [ ] Prediction #1: ______ words (should be 1-3)
  2. [ ] Prediction #2: ______ words (should be 1-3)
  3. [ ] Prediction #3: ______ words (should be 1-3)
  4. [ ] Prediction #4: ______ words (should be 1-3)
  5. [ ] Prediction #5: ______ words (should be 1-3)

- [ ] Predictions match expected format (yes/no, colors, numbers, etc.)

### Compare Before/After

Before Fix:
- Loss: 7.26
- Accuracy: 19.6%
- Prediction length: ~50 words

After Fix:
- Loss: ______
- Accuracy: ______%
- Prediction length: ______ words

Improvement: [ ] YES / [ ] NO

---

## 🎯 SUCCESS CRITERIA

Mark ✓ if criterion is met:

- [ ] Training loss < 2.0
- [ ] Overall accuracy > 60%
- [ ] Binary question accuracy > 85%
- [ ] Predictions are concise (1-3 words for short questions)
- [ ] Model follows instruction format
- [ ] No training errors or crashes

**If all checked → SUCCESS! 🎉**

---

## 🔄 STEP 7: APPLY TO OTHER EXPERIMENTS (Optional)

### Experiment 2

- [ ] Update exp2 SLURM script to use instructed data
- [ ] Submit exp2 training
- [ ] Evaluate exp2
- [ ] Compare with exp1

### Experiment 3

- [ ] Update exp3 SLURM script
- [ ] Submit exp3 training
- [ ] Evaluate exp3

### Experiment 4

- [ ] Update exp4 SLURM script
- [ ] Submit exp4 training
- [ ] Evaluate exp4

### Experiment 5

- [ ] Update exp5 SLURM script
- [ ] Submit exp5 training
- [ ] Evaluate exp5

---

## 📝 NOTES SECTION

Use this space to note any issues, observations, or questions:

```
Date: __________

Training notes:
_____________________________________________
_____________________________________________
_____________________________________________

Issues encountered:
_____________________________________________
_____________________________________________
_____________________________________________

Questions to ask advisor:
_____________________________________________
_____________________________________________
_____________________________________________
```

---

## 🎓 FINAL DELIVERABLES CHECKLIST

For thesis/presentation:

- [ ] All 5 experiments completed with improved method
- [ ] Comparison table: Before vs After for all experiments
- [ ] Loss curves plotted
- [ ] Accuracy by question type graphs
- [ ] Sample predictions (good and bad) collected
- [ ] Computational resources documented
- [ ] Code committed to GitHub
- [ ] Results documented in thesis

---

## 🆘 TROUBLESHOOTING LOG

If something goes wrong, note it here:

| Issue | Time | Solution | Status |
|-------|------|----------|--------|
| | | | |
| | | | |
| | | | |

---

## ✨ COMPLETION

Date completed: __________
Time taken: __________
Final accuracy: __________%
Ready for thesis: [ ] YES / [ ] NO

**Congratulations!** 🎉

Signature: ___________________

---

**REMINDER**: Save this checklist and refer back to it when implementing
the same fix for Experiments 2-5!
