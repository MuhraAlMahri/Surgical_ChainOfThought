# 📝 EXACT CODE MODIFICATIONS

This file contains the EXACT code you need to copy-paste into your training script.

---

## File: `train_qwen_lora.py`

### Modification 1: LazyVQADataset.__init__ (around line 417-442)

**FIND THIS SECTION** (lines 430-435):
```python
        self.items.append({
            'question': item.get('question'),
            'answer': item.get('answer'),
            'image_id': item.get('image_id'),
            'image_path': image_path
        })
```

**REPLACE WITH**:
```python
        self.items.append({
            'question': item.get('question'),
            'answer': item.get('answer'),
            'instruction': item.get('instruction', item.get('question')),  # NEW: Use instruction field
            'question_type': item.get('question_type', 'open_short'),     # NEW: Track question type
            'image_id': item.get('image_id'),
            'image_path': image_path
        })
```

---

### Modification 2: LazyVQACollator.__call__ (around line 459-520)

**FIND THIS ENTIRE METHOD**:
```python
def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    images: List[Image.Image] = []
    questions: List[str] = []
    answers: List[str] = []

    # Extract data from features
    for feat in features:
        question = feat.get('question', '')
        answer = feat.get('answer', '')
        image_path = feat.get('image_path')

        # Load image
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert('RGB')
                max_size = self.image_max_size
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                images.append(img)
            except Exception:
                images.append(Image.new('RGB', (self.image_max_size, self.image_max_size), color=(0, 0, 0)))
        else:
            images.append(Image.new('RGB', (self.image_max_size, self.image_max_size), color=(0, 0, 0)))

        questions.append(question)
        answers.append(answer)

    # FIXED: Use PROVEN approach from working train_progressive_stage.py
    # Build FULL conversation including assistant response
    messages_batch = []
    for question, answer in zip(questions, answers):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",  # Include answer in conversation!
                "content": [{"type": "text", "text": answer}]
            }
        ]
        messages_batch.append(messages)

    # Apply chat template to full conversations (user + assistant)
    texts = [self.processor.apply_chat_template(msg, tokenize=False) 
             for msg in messages_batch]
    
    # Tokenize full conversations with images
    inputs = self.processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    # Labels = clone of input_ids (model masks user portion internally)
    inputs['labels'] = inputs['input_ids'].clone()

    return inputs
```

**REPLACE WITH THIS FIXED VERSION**:
```python
def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    images: List[Image.Image] = []
    instructions: List[str] = []  # CHANGED: Use instructions instead of raw questions
    answers: List[str] = []

    # Extract data from features
    for feat in features:
        # CHANGED: Use instruction field (with fallback to question for compatibility)
        instruction = feat.get('instruction', feat.get('question', ''))
        answer = feat.get('answer', '')
        image_path = feat.get('image_path')

        # Load image (no changes to this section)
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert('RGB')
                max_size = self.image_max_size
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                images.append(img)
            except Exception:
                images.append(Image.new('RGB', (self.image_max_size, self.image_max_size), color=(0, 0, 0)))
        else:
            images.append(Image.new('RGB', (self.image_max_size, self.image_max_size), color=(0, 0, 0)))

        # CHANGED: Append instruction instead of question
        instructions.append(instruction)
        answers.append(answer)

    # Build FULL conversation with proper instruction templates
    messages_batch = []
    for instruction, answer in zip(instructions, answers):  # CHANGED: Use instruction
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction}  # CHANGED: Use instruction template
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]
        messages_batch.append(messages)

    # Apply chat template to full conversations (user + assistant)
    texts = [self.processor.apply_chat_template(msg, tokenize=False) 
             for msg in messages_batch]
    
    # Tokenize full conversations with images
    inputs = self.processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    # Labels = clone of input_ids (model masks user portion internally)
    inputs['labels'] = inputs['input_ids'].clone()

    return inputs
```

---

## Summary of Changes

### What Changed:
1. `questions` → `instructions` (variable name)
2. `feat.get('question')` → `feat.get('instruction', feat.get('question'))`
3. Added `'instruction'` and `'question_type'` fields to dataset items

### Why This Matters:
- **Before**: Model saw raw question: `"What is the color?"`
- **After**: Model sees full instruction: `"You are a surgical assistant. Answer with ONLY the color name. Do not provide explanations.\n\nQuestion: What is the color?\nAnswer with only the color:"`

This explicit instruction tells the model EXACTLY how to format its response!

---

## How to Apply These Changes

### Option 1: Manual Edit (Recommended)
1. Open `train_qwen_lora.py` in your editor
2. Find line ~430 (LazyVQADataset.__init__)
3. Copy-paste the first modification
4. Find line ~459 (LazyVQACollator.__call__)
5. Copy-paste the second modification
6. Save the file

### Option 2: Using sed (Advanced)
```bash
# Backup original file
cp train_qwen_lora.py train_qwen_lora.py.backup

# Apply modifications (you'll need to adjust line numbers based on your exact file)
# This is just an example - manual editing is safer!
```

---

## Verification Checklist

After making changes, verify:

- [ ] Line ~432: `'instruction': item.get('instruction', item.get('question')),`
- [ ] Line ~433: `'question_type': item.get('question_type', 'open_short'),`
- [ ] Line ~461: `instructions: List[str] = []` (not `questions`)
- [ ] Line ~467: `instruction = feat.get('instruction', ...)`
- [ ] Line ~484: `instructions.append(instruction)`
- [ ] Line ~489: `for instruction, answer in zip(instructions, answers):`
- [ ] Line ~496: `{"type": "text", "text": instruction}`

All these should reference `instruction` not `question`!

---

## Testing Your Changes

### Quick Test (Before Training):
```python
# Add this at the end of train_qwen_lora.py for testing
if __name__ == "__main__" and False:  # Set to True for testing
    # Test data loading
    from datasets import Dataset
    
    test_item = {
        'question': 'What color?',
        'answer': 'pink',
        'instruction': 'Answer with only the color:',
        'question_type': 'color',
        'image_path': '/path/to/test/image.jpg'
    }
    
    dataset = LazyVQADataset(...)
    print(f"Sample item: {dataset[0]}")
    print(f"Has instruction: {'instruction' in dataset[0]}")
    
    # Test collator
    collator = LazyVQACollator(processor, tokenizer, max_length=512)
    batch = collator([test_item])
    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")
```

### Real Test (After Training Starts):
```bash
# Monitor training log
tail -f experiments/exp1_random/logs/train_*.out

# Look for:
# - Loss starting high (4-6) is okay
# - Loss dropping to <2.0 within first epoch
# - No errors about missing 'instruction' key
```

---

## Common Mistakes to Avoid

### ❌ WRONG:
```python
# Still using 'question' instead of 'instruction'
for feat in features:
    question = feat.get('question', '')  # WRONG!
    instructions.append(question)        # WRONG!
```

### ✅ CORRECT:
```python
# Using 'instruction' field
for feat in features:
    instruction = feat.get('instruction', feat.get('question', ''))  # CORRECT!
    instructions.append(instruction)                                  # CORRECT!
```

---

### ❌ WRONG:
```python
# Forgot to update the zip() call
for question, answer in zip(questions, answers):  # WRONG! Still says 'question'
```

### ✅ CORRECT:
```python
# Updated zip() to use instructions
for instruction, answer in zip(instructions, answers):  # CORRECT!
```

---

## Final Check Before Training

Run this checklist:

```bash
# 1. Preprocessed data exists
ls -lh datasets/kvasir_instructed/train_instructed.json
ls -lh datasets/kvasir_instructed/val_instructed.json

# 2. Training script has been modified
grep "instruction = feat.get('instruction'" train_qwen_lora.py

# 3. SLURM script points to new data
grep "kvasir_instructed" experiments/exp1_random/train_exp1.slurm

# All three should show results!
```

If all three checks pass → You're ready to train! 🚀

---

## Expected Results After Fix

### Training Metrics:
```
Before Fix:
Step 100: loss=7.26, lr=5e-6
Step 500: loss=7.01, lr=5e-6  ← Barely improving!

After Fix:
Step 100: loss=4.12, lr=5e-6
Step 500: loss=1.34, lr=5e-6  ← Much better!
```

### Prediction Quality:
```
Before Fix:
Q: "What is the color?"
A: "The abnormality appears to be pink in color, indicating..."  (27 words)

After Fix:
Q: "What is the color?"
A: "pink"  (1 word) ✓
```

Good luck! 🎉
