# Model Architecture Components

This document describes the five main components of your Surgical CoT model architecture.

## Architecture Overview

```
Input: Image + Question + (Optional) Previous Frame Context
         ↓
┌─────────────────────────────────────────┐
│  1. Visual Encoder                      │
│     - Encodes image pixels              │
│     - Output: Visual features           │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  2. Language Encoder                     │
│     - Encodes question text             │
│     - Output: Text embeddings           │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  3. Fusion Module                       │
│     - Combines visual + language        │
│     - Output: Multimodal features       │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  4. Reasoning Module                    │
│     - Temporal context integration      │
│     - Chain-of-Thought processing      │
│     - Output: Reasoning features       │
└─────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────┐
│  5. Answer Generation Decoder           │
│     - Three specialized heads           │
│     - Predicts answer tokens            │
└─────────────────────────────────────────┘
         ↓
Output: Answer tokens (vocabulary logits)
```

---

## 1. Visual Encoder

**Purpose**: Encodes input images into visual feature representations.

**Implementation**:
- **Qwen3-VL**: `self.base_model.vision_model` 
  - Location: `models/qwen3vl_multihead.py` (line 67-68)
  - Architecture: Vision Transformer (ViT) or similar
  - Input: `pixel_values` (preprocessed images)
  - Output: Visual embeddings

- **LLaVA-Med**: `self.base_model.vision_tower`
  - Location: `models/llava_med_multihead.py` (line 67-68)
  - Architecture: CLIP vision encoder
  - Can be frozen during training (`freeze_vision_tower=True`)

- **MedGemma**: Built into the base model
  - Location: `models/medgemma_multihead.py`
  - Architecture: Integrated vision encoder

**Key Code**:
```python
# Qwen3-VL
if freeze_vision and hasattr(self.base_model, 'vision_model'):
    for param in self.base_model.vision_model.parameters():
        param.requires_grad = False

# LLaVA-Med
if freeze_vision_tower and hasattr(self.base_model, 'vision_tower'):
    for param in self.base_model.vision_tower.parameters():
        param.requires_grad = False
```

---

## 2. Language Encoder

**Purpose**: Encodes question text into text embeddings.

**Implementation**:
- **Qwen3-VL**: `AutoModelForImageTextToText` (transformer backbone)
  - Location: `models/qwen3vl_multihead.py` (line 53)
  - Architecture: Qwen transformer with vision-language capabilities

- **LLaVA-Med**: `AutoModelForVision2Seq` (language model backbone)
  - Location: `models/llava_med_multihead.py` (line 53)
  - Architecture: Mistral-7B or similar LLM

- **MedGemma**: `AutoModelForCausalLM` (Gemma-based)
  - Location: `models/medgemma_multihead.py` (line 51)
  - Architecture: Gemma-4B transformer

**Key Code**:
```python
# All models process text through the base model
outputs = self.base_model(
    pixel_values=pixel_values,  # Visual input
    input_ids=input_ids,        # Text input (tokenized)
    attention_mask=attention_mask,
    output_hidden_states=True
)
```

**Hidden Dimension**:
- Qwen3-VL: 4096 (default) or from `config.hidden_size`
- LLaVA-Med: 4096 (default) or from `config.hidden_size`
- MedGemma: 2048 (default) or from `config.hidden_size`

---

## 3. Fusion Module

**Purpose**: Combines visual and language features into unified multimodal representations.

**Implementation**:
- **Location**: Inside the base model (vision-language model architecture)
- **Method**: The base model handles fusion internally
  - For Qwen3-VL: Vision-language attention mechanism
  - For LLaVA-Med: Multimodal projector + cross-attention
  - For MedGemma: Integrated vision-language processing

**Key Code**:
```python
# Fusion happens inside base_model.forward()
outputs = self.base_model(
    pixel_values=pixel_values,  # Visual features
    input_ids=input_ids,          # Text embeddings
    attention_mask=attention_mask,
    **kwargs
)

# After fusion, get hidden states
hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
# or
hidden_states = outputs.last_hidden_state
```

**Output**: 
- Shape: `[batch_size, sequence_length, hidden_dim]`
- Contains fused visual-textual information

---

## 4. Reasoning Module

**Purpose**: Enhances features with temporal context and Chain-of-Thought reasoning.

**Implementation**:

### A. Temporal Context Encoder
- **Location**: All model files (e.g., `models/qwen3vl_multihead.py` line 101)
- **Architecture**: `nn.Linear(hidden_dim, hidden_dim)`
- **Purpose**: Integrates information from previous frames in video sequences

**Key Code**:
```python
# Temporal context encoder
self.temporal_encoder = nn.Linear(self.hidden_dim, self.hidden_dim)

# In forward pass:
if previous_context is not None:
    temporal_features = self.temporal_encoder(previous_context)
    hidden_state = hidden_state + temporal_features  # Residual connection
```

### B. Chain-of-Thought Reasoning
- **Location**: Prompt-based (see `prompts/cot_builder.py` or `prompts/cot_templates.py`)
- **Method**: Guided by structured prompts, actual reasoning happens in transformer layers
- **Categories**:
  1. `abnormality_detection`: Detects abnormalities, instruments, polyps
  2. `characteristics`: Analyzes color, location, type, count
  3. `treatment`: Provides clinical context and recommendations

**Key Code**:
```python
# CoT is guided by prompts, processed through transformer
# The model generates reasoning text during generation
outputs = self.base_model.generate(
    pixel_values=pixel_values,
    input_ids=input_ids,
    max_new_tokens=max_new_tokens,
    temperature=temperature
)
```

**Output**:
- Enhanced hidden states with temporal context
- Shape: `[batch_size, hidden_dim]` (last token)

---

## 5. Answer Generation Decoder

**Purpose**: Predicts answer tokens from reasoning features.

**Implementation**:
- **Location**: All model files (e.g., `models/qwen3vl_multihead.py` lines 96-98)
- **Architecture**: Three specialized linear heads

**Components**:

### A. Three Specialized Heads

1. **Head 1: Abnormality Detection**
   ```python
   self.head_abnormality = nn.Linear(self.hidden_dim, self.vocab_size)
   ```
   - Purpose: Detects abnormalities, instruments, polyps, lesions
   - Category: `"abnormality_detection"` or `1`

2. **Head 2: Characteristics**
   ```python
   self.head_characteristics = nn.Linear(self.hidden_dim, self.vocab_size)
   ```
   - Purpose: Analyzes properties (color, location, type, count)
   - Category: `"characteristics"` or `2`

3. **Head 3: Treatment**
   ```python
   self.head_treatment = nn.Linear(self.hidden_dim, self.vocab_size)
   ```
   - Purpose: Provides clinical context, diagnosis, treatment
   - Category: `"treatment"` or `3`

### B. Routing Logic

**Key Code**:
```python
# Route to appropriate head based on category
if category == "abnormality_detection" or category == 1:
    logits = self.head_abnormality(hidden_state)
elif category == "characteristics" or category == 2:
    logits = self.head_characteristics(hidden_state)
elif category == "treatment" or category == 3:
    logits = self.head_treatment(hidden_state)
```

### C. Initialization

**Key Code**:
```python
def _init_heads(self):
    """Initialize the specialized heads."""
    nn.init.xavier_uniform_(self.head_abnormality.weight)
    nn.init.xavier_uniform_(self.head_characteristics.weight)
    nn.init.xavier_uniform_(self.head_treatment.weight)
    
    if self.head_abnormality.bias is not None:
        nn.init.zeros_(self.head_abnormality.bias)
    # ... same for other heads
```

**Output**:
- Shape: `[batch_size, vocab_size]`
- Contains logits for each vocabulary token
- Used for token prediction during generation

---

## Complete Forward Pass Flow

```python
def forward(self, images, input_ids, category, previous_context=None):
    # 1. Visual Encoder (inside base_model)
    #    pixel_values → vision_model → visual_features
    
    # 2. Language Encoder (inside base_model)
    #    input_ids → transformer → text_features
    
    # 3. Fusion Module (inside base_model)
    #    visual_features + text_features → multimodal_features
    
    # 4. Get hidden states from base model
    outputs = self.base_model(pixel_values, input_ids, ...)
    hidden_state = outputs.hidden_states[-1][:, -1, :]  # Last token
    
    # 5. Reasoning Module: Add temporal context
    if previous_context is not None:
        temporal_features = self.temporal_encoder(previous_context)
        hidden_state = hidden_state + temporal_features
    
    # 6. Answer Generation Decoder: Route to appropriate head
    if category == "abnormality_detection":
        logits = self.head_abnormality(hidden_state)
    elif category == "characteristics":
        logits = self.head_characteristics(hidden_state)
    elif category == "treatment":
        logits = self.head_treatment(hidden_state)
    
    return {'logits': logits, 'hidden_state': hidden_state}
```

---

## Model-Specific Details

### Qwen3-VL-8B
- **Base Model**: `Qwen/Qwen3-VL-8B-Instruct`
- **Visual Encoder**: `base_model.vision_model`
- **Hidden Dim**: 4096
- **Vocab Size**: From processor tokenizer

### LLaVA-Med v1.5
- **Base Model**: `microsoft/llava-med-v1.5-mistral-7b`
- **Visual Encoder**: `base_model.vision_tower` (CLIP)
- **Hidden Dim**: 4096
- **Vocab Size**: From processor tokenizer

### MedGemma-4B
- **Base Model**: `google/medgemma-4b`
- **Visual Encoder**: Integrated
- **Hidden Dim**: 2048
- **Vocab Size**: From processor tokenizer

---

## Training Configuration

- **LoRA**: Applied to base model (typically `r=8`, `alpha=16`)
- **Frozen Components**: Vision encoder can be frozen
- **Trainable Components**: 
  - LoRA adapters in base model
  - Three specialized heads
  - Temporal encoder

---

## File Locations

- **Qwen3-VL Model**: `models/qwen3vl_multihead.py`
- **LLaVA-Med Model**: `models/llava_med_multihead.py`
- **MedGemma Model**: `models/medgemma_multihead.py`
- **Generic Model**: `models/multi_head_model.py`
- **CoT Prompts**: `prompts/cot_builder.py` or `prompts/cot_templates.py`











