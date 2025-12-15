#!/usr/bin/env python3
"""
Multi-Head CoT Model for Qwen3-VL-8B-Instruct
Implements 3 specialized heads for clinical reasoning stages.
"""

import torch
import torch.nn as nn
import os
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiHeadCoT_Qwen3VL(nn.Module):
    """
    Multi-head architecture for Qwen3-VL-8B with Chain-of-Thought reasoning.
    
    Three specialized heads:
    - Head 1: Abnormality/Instrument Detection
    - Head 2: Characteristics (color, type, location, count)
    - Head 3: Treatment/Clinical Context
    """
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        freeze_vision: bool = False
    ):
        """
        Initialize the multi-head model.
        
        Args:
            base_model_name: Base Qwen3-VL model name
            use_lora: Whether to use LoRA for fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            freeze_vision: Whether to freeze vision encoder
        """
        super().__init__()
        
        self.base_model_name = base_model_name
        self.use_lora = use_lora
        
        # Get HuggingFace token from environment
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        
        # Load base model
        logger.info(f"Loading base model: {base_model_name}")
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            token=hf_token
        )
        
        # Freeze vision encoder if requested
        if freeze_vision and hasattr(self.base_model, 'vision_model'):
            for param in self.base_model.vision_model.parameters():
                param.requires_grad = False
            logger.info("Vision encoder frozen")
        
        # Get hidden dimension
        if hasattr(self.base_model.config, 'hidden_size'):
            self.hidden_dim = self.base_model.config.hidden_size
        elif hasattr(self.base_model.config, 'd_model'):
            self.hidden_dim = self.base_model.config.d_model
        else:
            self.hidden_dim = 4096  # Default fallback
        
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
        image_grid_thw: Optional[torch.Tensor] = None,
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
            category: Question category (abnormality_detection, characteristics, treatment)
            previous_context: Temporal context from previous frame (hidden state)
            output_hidden_states: Whether to output hidden states
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with logits, hidden states, and model outputs
        """
        # Process images if needed
        if pixel_values is None and images is not None:
            if input_ids is None:
                # Process both images and text
                inputs = self.processor(
                    images=images,
                    text=kwargs.get('text', ''),
                    return_tensors="pt",
                    padding=True
                )
                pixel_values = inputs['pixel_values'].to(self.base_model.device)
                input_ids = inputs['input_ids'].to(self.base_model.device)
                attention_mask = inputs.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.base_model.device)
            else:
                # Only process images
                inputs = self.processor(images=images, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.base_model.device)
        
        # Ensure tensors are on correct device
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.base_model.device)
        if input_ids is not None:
            input_ids = input_ids.to(self.base_model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.base_model.device)
        
        # Forward through base model
        base_model_kwargs = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': output_hidden_states,
        }
        # Add image_grid_thw if provided (required for Qwen3-VL)
        if image_grid_thw is not None:
            base_model_kwargs['image_grid_thw'] = image_grid_thw.to(self.base_model.device)
        base_model_kwargs.update(kwargs)
        
        outputs = self.base_model(**base_model_kwargs)
        
        # Get hidden states (last layer, last token)
        if output_hidden_states and hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_state = outputs.hidden_states[-1][:, -1, :]  # [batch_size, hidden_dim]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_state = outputs.last_hidden_state[:, -1, :]
        else:
            # Fallback: use logits to infer hidden state
            logger.warning("Could not extract hidden states, using logits")
            hidden_state = outputs.logits[:, -1, :]
        
        # Add temporal context if available
        if previous_context is not None:
            if previous_context.device != hidden_state.device:
                previous_context = previous_context.to(hidden_state.device)
            temporal_features = self.temporal_encoder(previous_context)
            hidden_state = hidden_state + temporal_features
        
        # Ensure hidden_state dtype matches head dtype (heads are float32, hidden_state might be bfloat16)
        if hidden_state.dtype != next(self.head_abnormality.parameters()).dtype:
            hidden_state = hidden_state.to(next(self.head_abnormality.parameters()).dtype)
        
        # Route to appropriate head
        if category == "abnormality_detection" or category == 1:
            logits = self.head_abnormality(hidden_state)
        elif category == "characteristics" or category == 2:
            logits = self.head_characteristics(hidden_state)
        elif category == "treatment" or category == 3:
            logits = self.head_treatment(hidden_state)
        else:
            # Default to abnormality detection
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
        """
        Generate answer with CoT reasoning.
        
        Args:
            images: Input images
            prompt: Text prompt/question
            category: Question category
            previous_context: Temporal context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
            
        Returns:
            Dictionary with generated text and reasoning
        """
        # Process inputs
        inputs = self.processor(
            images=images,
            text=prompt,
            return_tensors="pt",
            padding=True
        )
        
        pixel_values = inputs['pixel_values'].to(self.base_model.device)
        input_ids = inputs['input_ids'].to(self.base_model.device)
        
        # Generate using base model (heads are used during training, not inference)
        with torch.no_grad():
            outputs = self.base_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
        
        # Decode
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
        if head_name == "abnormality":
            for param in self.head_abnormality.parameters():
                param.requires_grad = False
        elif head_name == "characteristics":
            for param in self.head_characteristics.parameters():
                param.requires_grad = False
        elif head_name == "treatment":
            for param in self.head_treatment.parameters():
                param.requires_grad = False
    
    def unfreeze_head(self, head_name: str):
        """Unfreeze a specific head."""
        if head_name == "abnormality":
            for param in self.head_abnormality.parameters():
                param.requires_grad = True
        elif head_name == "characteristics":
            for param in self.head_characteristics.parameters():
                param.requires_grad = True
        elif head_name == "treatment":
            for param in self.head_treatment.parameters():
                param.requires_grad = True


def create_qwen3vl_multihead(
    base_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16
) -> MultiHeadCoT_Qwen3VL:
    """
    Factory function to create a Qwen3-VL multi-head model.
    
    Args:
        base_model_name: Base model name
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        
    Returns:
        Initialized MultiHeadCoT_Qwen3VL model
    """
    model = MultiHeadCoT_Qwen3VL(
        base_model_name=base_model_name,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )
    return model




