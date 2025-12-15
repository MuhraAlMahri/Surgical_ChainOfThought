#!/usr/bin/env python3
"""
Multi-Head CoT Model for MedGemma-4B
Implements 3 specialized heads for clinical reasoning stages.
"""

import torch
import torch.nn as nn
import os
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiHeadCoT_MedGemma(nn.Module):
    """
    Multi-head architecture for MedGemma-4B with Chain-of-Thought reasoning.
    
    Three specialized heads:
    - Head 1: Abnormality/Instrument Detection
    - Head 2: Characteristics (color, type, location, count)
    - Head 3: Treatment/Clinical Context
    """
    
    def __init__(
        self,
        base_model_name: str = "google/medgemma-4b",
        use_lora: bool = True,
        lora_r: int = 4,
        lora_alpha: int = 16,
        processor_base_model: Optional[str] = None
    ):
        """
        Initialize the multi-head model.
        
        Args:
            base_model_name: Base MedGemma model name
            use_lora: Whether to use LoRA for fine-tuning
            lora_r: LoRA rank (4 for MedGemma)
            lora_alpha: LoRA alpha
        """
        super().__init__()
        
        self.base_model_name = base_model_name
        self.use_lora = use_lora
        
        # Get HuggingFace token from environment
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        
        # Load base model
        logger.info(f"Loading base model: {base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
        
        # Load processor - always use base model name, not checkpoint path
        # If base_model_name is a local path (checkpoint), use processor_base_model or default
        processor_model_name = processor_base_model
        if processor_model_name is None:
            # Check if base_model_name is a local path (contains / and doesn't look like HF model)
            if os.path.exists(base_model_name) or (os.path.sep in base_model_name and not base_model_name.startswith("google/") and not base_model_name.startswith("microsoft/")):
                # It's a checkpoint path, use default base model for processor
                processor_model_name = "google/medgemma-4b-it"
            else:
                # It's a HuggingFace model name, use it directly
                processor_model_name = base_model_name
        
        self.processor = AutoProcessor.from_pretrained(
            processor_model_name,
            trust_remote_code=True,
            token=hf_token
        )
        
        # Get hidden dimension - handle PEFT models and fine-tuned checkpoints
        hidden_dim = None
        
        # Try to get from config (handle PEFT wrapper)
        config = self.base_model.config
        if hasattr(self.base_model, 'get_base_model'):
            # PEFT model - get base model config
            config = self.base_model.get_base_model().config
        
        if hasattr(config, 'hidden_size'):
            hidden_dim = config.hidden_size
        elif hasattr(config, 'd_model'):
            hidden_dim = config.d_model
        
        # If still not found, try to infer from model architecture
        if hidden_dim is None:
            # Try to get from lm_head or embedding layer
            if hasattr(self.base_model, 'lm_head'):
                if hasattr(self.base_model.lm_head, 'in_features'):
                    hidden_dim = self.base_model.lm_head.in_features
            elif hasattr(self.base_model, 'get_base_model'):
                base = self.base_model.get_base_model()
                if hasattr(base, 'lm_head') and hasattr(base.lm_head, 'in_features'):
                    hidden_dim = base.lm_head.in_features
        
        # If still not found, try to get from model output shape (last resort)
        if hidden_dim is None:
            # Try to infer from model's word embeddings
            if hasattr(self.base_model, 'get_input_embeddings'):
                emb = self.base_model.get_input_embeddings()
                if hasattr(emb, 'embedding_dim'):
                    hidden_dim = emb.embedding_dim
            elif hasattr(self.base_model, 'get_base_model'):
                base = self.base_model.get_base_model()
                if hasattr(base, 'get_input_embeddings'):
                    emb = base.get_input_embeddings()
                    if hasattr(emb, 'embedding_dim'):
                        hidden_dim = emb.embedding_dim
        
        # Final fallback - use actual model output dimension if available
        if hidden_dim is None:
            logger.warning("Could not determine hidden_dim from config, trying to infer from model...")
            # Try a dummy forward pass to get the dimension (but this is expensive)
            # Instead, check common MedGemma sizes
            # MedGemma-4B typically has hidden_size=2560, but some variants use 2048
            # Since the error shows 2560, let's use that as default
            hidden_dim = 2560  # MedGemma-4B actual hidden size
            logger.warning(f"Using default hidden_dim={hidden_dim} for MedGemma-4B")
        
        self.hidden_dim = hidden_dim
        logger.info(f"Detected hidden_dim={self.hidden_dim}")
        
        # Get vocab size
        self.vocab_size = len(self.processor.tokenizer)
        
        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            logger.info(f"Applied LoRA (r={lora_r}, alpha={lora_alpha})")
        
        # Three specialized heads
        self.head_abnormality = nn.Linear(self.hidden_dim, self.vocab_size)
        self.head_characteristics = nn.Linear(self.hidden_dim, self.vocab_size)
        self.head_treatment = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # Temporal context encoder
        self.temporal_encoder = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Initialize heads
        self._init_heads()
        
        logger.info(f"Multi-head model initialized: hidden_dim={self.hidden_dim}, vocab_size={self.vocab_size}")
    
    def _init_heads(self):
        """Initialize the specialized heads."""
        nn.init.xavier_uniform_(self.head_abnormality.weight)
        nn.init.xavier_uniform_(self.head_characteristics.weight)
        nn.init.xavier_uniform_(self.head_treatment.weight)
        
        if self.head_abnormality.bias is not None:
            nn.init.zeros_(self.head_abnormality.bias)
        if self.head_characteristics.bias is not None:
            nn.init.zeros_(self.head_characteristics.bias)
        if self.head_treatment.bias is not None:
            nn.init.zeros_(self.head_treatment.bias)
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        category: str = "abnormality_detection",
        previous_context: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            images: Input images (PIL Images or tensors)
            pixel_values: Preprocessed pixel values
            input_ids: Tokenized input text
            attention_mask: Attention mask
            category: Question category
            previous_context: Temporal context from previous frame
            output_hidden_states: Whether to output hidden states
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with logits, hidden states, and model outputs
        """
        # Process images if needed
        if pixel_values is None and images is not None:
            inputs = self.processor(
                images=images,
                text=kwargs.get('text', ''),
                return_tensors="pt",
                padding=True
            )
            pixel_values = inputs['pixel_values'].to(self.base_model.device)
            if input_ids is None:
                input_ids = inputs['input_ids'].to(self.base_model.device)
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.base_model.device)
        
        # Ensure tensors are on correct device
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.base_model.device)
        if input_ids is not None:
            input_ids = input_ids.to(self.base_model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.base_model.device)
        
        # Forward through base model
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
        
        # Get hidden states (last layer, last token)
        if output_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_state = outputs.hidden_states[-1][:, -1, :]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_state = outputs.last_hidden_state[:, -1, :]
        else:
            hidden_state = outputs.logits[:, -1, :]
        
        # Add temporal context if available
        if previous_context is not None:
            if previous_context.device != hidden_state.device:
                previous_context = previous_context.to(hidden_state.device)
            # Ensure temporal_encoder matches hidden_state dtype
            if self.temporal_encoder.weight.dtype != hidden_state.dtype:
                self.temporal_encoder = self.temporal_encoder.to(dtype=hidden_state.dtype)
            temporal_features = self.temporal_encoder(previous_context)
            hidden_state = hidden_state + temporal_features
        
        # Ensure hidden_state and heads have matching dtype
        hidden_state_dtype = hidden_state.dtype
        if self.head_abnormality.weight.dtype != hidden_state_dtype:
            # Convert heads to match hidden_state dtype
            self.head_abnormality = self.head_abnormality.to(dtype=hidden_state_dtype)
            self.head_characteristics = self.head_characteristics.to(dtype=hidden_state_dtype)
            self.head_treatment = self.head_treatment.to(dtype=hidden_state_dtype)
        
        # Route to appropriate head
        if category == "abnormality_detection" or category == 1:
            logits = self.head_abnormality(hidden_state)
        elif category == "characteristics" or category == 2:
            logits = self.head_characteristics(hidden_state)
        elif category == "treatment" or category == 3:
            logits = self.head_treatment(hidden_state)
        else:
            logits = self.head_abnormality(hidden_state)
        
        return {
            'logits': logits,
            'hidden_state': hidden_state,
            'base_outputs': outputs
        }
    
    def generate(
        self,
        images,
        prompt: str,
        category: str = "abnormality_detection",
        previous_context: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate answer with CoT reasoning."""
        inputs = self.processor(
            images=images,
            text=prompt,
            return_tensors="pt",
            padding=True
        )
        
        pixel_values = inputs['pixel_values'].to(self.base_model.device)
        input_ids = inputs['input_ids'].to(self.base_model.device)
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
        
        generated_text = self.processor.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return {
            'generated_text': generated_text,
            'output_ids': outputs
        }
    
    def freeze_head(self, head_name: str):
        """Freeze a specific head."""
        heads = {
            "abnormality": self.head_abnormality,
            "characteristics": self.head_characteristics,
            "treatment": self.head_treatment
        }
        if head_name in heads:
            for param in heads[head_name].parameters():
                param.requires_grad = False
    
    def unfreeze_head(self, head_name: str):
        """Unfreeze a specific head."""
        heads = {
            "abnormality": self.head_abnormality,
            "characteristics": self.head_characteristics,
            "treatment": self.head_treatment
        }
        if head_name in heads:
            for param in heads[head_name].parameters():
                param.requires_grad = True


def create_medgemma_multihead(
    base_model_name: str = "google/medgemma-4b",
    use_lora: bool = True,
    lora_r: int = 4,
    lora_alpha: int = 16,
    processor_base_model: Optional[str] = None
) -> MultiHeadCoT_MedGemma:
    """Factory function to create a MedGemma multi-head model."""
    model = MultiHeadCoT_MedGemma(
        base_model_name=base_model_name,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        processor_base_model=processor_base_model
    )
    return model









