#!/usr/bin/env python3
"""
Multi-head CoT wrapper for surgical VQA models.

Takes a fine-tuned checkpoint and adds 3 specialized heads:
1. Abnormality detection head
2. Characteristics head  
3. Treatment head

Also includes temporal context encoder for video sequences.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from pathlib import Path
import logging

# Import model-specific implementations
from models.qwen3vl_multihead import MultiHeadCoT_Qwen3VL, create_qwen3vl_multihead
from models.medgemma_multihead import MultiHeadCoT_MedGemma, create_medgemma_multihead
from models.llava_med_multihead import MultiHeadCoT_LLaVAMed, create_llava_med_multihead

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_by_type(
    base_checkpoint: str,
    model_type: str,
    freeze_base: bool = True
) -> nn.Module:
    """
    Load fine-tuned base model by type.
    
    Args:
        base_checkpoint: Path to fine-tuned model checkpoint
        model_type: "qwen3vl", "medgemma", or "llava_med"
        freeze_base: Whether to freeze base model weights
        
    Returns:
        Loaded base model
    """
    checkpoint_path = Path(base_checkpoint)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {base_checkpoint}")
    
    logger.info(f"Loading {model_type} checkpoint from {base_checkpoint}")
    
    # Load model-specific base
    if model_type == "qwen3vl":
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            base_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    elif model_type == "medgemma":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            base_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    elif model_type == "llava_med":
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            base_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Freeze base model if requested
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Base model frozen")
    
    return model


class MultiHeadCoT_Model(nn.Module):
    """
    Multi-head CoT wrapper for surgical VQA models.
    
    Takes a fine-tuned checkpoint and adds 3 specialized heads:
    1. Abnormality detection head
    2. Characteristics head  
    3. Treatment head
    
    Also includes temporal context encoder for video sequences.
    """
    
    def __init__(
        self,
        base_checkpoint: str,
        model_type: str = "qwen3vl",
        freeze_base: bool = True,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16
    ):
        """
        Initialize multi-head model.
        
        Args:
            base_checkpoint: Path to fine-tuned model checkpoint
            model_type: "qwen3vl", "medgemma", or "llava_med"
            freeze_base: Whether to freeze base model weights
            use_lora: Whether to use LoRA for heads
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
        """
        super().__init__()
        
        self.model_type = model_type
        self.freeze_base = freeze_base
        
        # Create model-specific multi-head wrapper
        if model_type == "qwen3vl":
            self.model = create_qwen3vl_multihead(
                base_model_name=base_checkpoint,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha
            )
        elif model_type == "medgemma":
            self.model = create_medgemma_multihead(
                base_model_name=base_checkpoint,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha
            )
        elif model_type == "llava_med":
            self.model = create_llava_med_multihead(
                base_model_name=base_checkpoint,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Created multi-head model for {model_type}")
    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        images: Optional[list] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        category: str = "abnormality_detection",
        previous_context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass through multi-head model.
        
        Args:
            pixel_values: Image tensor
            images: List of PIL images (alternative to pixel_values)
            input_ids: Tokenized prompt
            attention_mask: Attention mask
            category: "abnormality_detection", "characteristics", or "treatment"
            previous_context: Hidden state from previous frame (optional)
        
        Returns:
            dict with 'logits' and 'hidden_state'
        """
        return self.model(
            pixel_values=pixel_values,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            category=category,
            previous_context=previous_context,
            **kwargs
        )
    
    def generate(
        self,
        images: list,
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
            images: List of PIL images
            prompt: Input prompt/question
            category: Question category
            previous_context: Temporal context from previous frame
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Dictionary with 'answer', 'reasoning', etc.
        """
        return self.model.generate(
            images=images,
            prompt=prompt,
            category=category,
            previous_context=previous_context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )


def create_multihead_model(
    base_checkpoint: str,
    model_type: str = "qwen3vl",
    freeze_base: bool = True,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16
) -> MultiHeadCoT_Model:
    """
    Factory function to create multi-head model.
    
    Args:
        base_checkpoint: Path to fine-tuned checkpoint
        model_type: "qwen3vl", "medgemma", or "llava_med"
        freeze_base: Whether to freeze base model
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        
    Returns:
        Multi-head model instance
    """
    return MultiHeadCoT_Model(
        base_checkpoint=base_checkpoint,
        model_type=model_type,
        freeze_base=freeze_base,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to fine-tuned checkpoint")
    parser.add_argument("--model-type", choices=["qwen3vl", "medgemma", "llava_med"], required=True)
    parser.add_argument("--freeze-base", action="store_true", default=True)
    
    args = parser.parse_args()
    
    model = create_multihead_model(
        base_checkpoint=args.checkpoint,
        model_type=args.model_type,
        freeze_base=args.freeze_base
    )
    
    print(f"Created multi-head model: {model}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")













