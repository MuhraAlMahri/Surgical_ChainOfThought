#!/usr/bin/env python3
"""
Multi-Head Surgical CoT Model
Implements 3 specialized heads for different clinical reasoning stages.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurgicalCoTModel(nn.Module):
    """
    Multi-head architecture for surgical VQA with Chain-of-Thought reasoning.
    
    Three specialized heads:
    - Head 1: Abnormality/Instrument Detection
    - Head 2: Characteristics (color, type, location, count)
    - Head 3: Treatment/Clinical Context
    """
    
    def __init__(
        self,
        base_model_name: str,
        hidden_dim: int = 4096,
        vocab_size: Optional[int] = None,
        use_lora: bool = True,
        lora_config: Optional[Dict] = None
    ):
        """
        Initialize the multi-head model.
        
        Args:
            base_model_name: Base vision-language model name
            hidden_dim: Hidden dimension size
            vocab_size: Vocabulary size (auto-detected if None)
            use_lora: Whether to use LoRA for fine-tuning
            lora_config: LoRA configuration dictionary
        """
        super().__init__()
        
        self.base_model_name = base_model_name
        self.hidden_dim = hidden_dim
        self.use_lora = use_lora
        
        # Load base model
        self._load_base_model()
        
        # Get hidden dimension from model if not specified
        if hidden_dim is None or hidden_dim == 4096:
            self.hidden_dim = self._get_hidden_dim()
        
        # Get vocab size
        if vocab_size is None:
            self.vocab_size = len(self.tokenizer) if hasattr(self, 'tokenizer') else 32000
        else:
            self.vocab_size = vocab_size
        
        # Three specialized heads
        self.head_abnormality = nn.Linear(self.hidden_dim, self.vocab_size)
        self.head_characteristics = nn.Linear(self.hidden_dim, self.vocab_size)
        self.head_treatment = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # CoT reasoning projector (optional, for explicit CoT generation)
        self.cot_projector = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Initialize heads
        self._init_heads()
        
        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_config)
    
    def _load_base_model(self):
        """Load the base vision-language model."""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        try:
            # Try loading as vision-language model first
            self.processor = AutoProcessor.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
            self.has_vision = True
            logger.info("Loaded as vision-language model")
        except Exception as e:
            logger.warning(f"Failed to load as VL model: {e}, trying as LLM")
            # Fallback to language model only
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.has_vision = False
            logger.info("Loaded as language model")
    
    def _get_hidden_dim(self) -> int:
        """Get hidden dimension from model."""
        # Try to infer from model config
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'hidden_size'):
                return self.model.config.hidden_size
            elif hasattr(self.model.config, 'd_model'):
                return self.model.config.d_model
            elif hasattr(self.model.config, 'n_embd'):
                return self.model.config.n_embd
        
        # Default fallback
        return 4096
    
    def _init_heads(self):
        """Initialize the specialized heads."""
        # Use Xavier initialization
        nn.init.xavier_uniform_(self.head_abnormality.weight)
        nn.init.xavier_uniform_(self.head_characteristics.weight)
        nn.init.xavier_uniform_(self.head_treatment.weight)
        
        # Initialize biases to zero
        if self.head_abnormality.bias is not None:
            nn.init.zeros_(self.head_abnormality.bias)
        if self.head_characteristics.bias is not None:
            nn.init.zeros_(self.head_characteristics.bias)
        if self.head_treatment.bias is not None:
            nn.init.zeros_(self.head_treatment.bias)
    
    def _apply_lora(self, lora_config: Optional[Dict]):
        """Apply LoRA to the base model."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            if lora_config is None:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
                )
            else:
                lora_config = LoraConfig(**lora_config)
            
            self.model = get_peft_model(self.model, lora_config)
            logger.info("Applied LoRA to base model")
        except ImportError:
            logger.warning("PEFT not available, skipping LoRA")
        except Exception as e:
            logger.warning(f"Failed to apply LoRA: {e}")
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        category: str = "abnormality_detection",
        previous_frame_info: Optional[Dict] = None,
        generate_cot: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.
        
        Args:
            images: Input images (if using vision model)
            input_ids: Tokenized input text
            attention_mask: Attention mask
            pixel_values: Preprocessed pixel values
            category: Question category (abnormality_detection, characteristics, treatment)
            previous_frame_info: Temporal context from previous frame
            generate_cot: Whether to generate CoT reasoning
            **kwargs: Additional arguments for model forward
            
        Returns:
            Dictionary with logits, hidden states, and optionally CoT text
        """
        # Prepare inputs
        if self.has_vision:
            if pixel_values is None and images is not None:
                # Process images if needed
                inputs = self.processor(
                    images=images,
                    text=kwargs.get('text', ''),
                    return_tensors="pt",
                    padding=True
                )
                pixel_values = inputs['pixel_values']
                if input_ids is None:
                    input_ids = inputs['input_ids']
                if attention_mask is None:
                    attention_mask = inputs['attention_mask']
            
            # Forward through base model
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        else:
            # Language model only
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # Get hidden states (last layer)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            # Fallback: use logits to infer hidden states
            logger.warning("Could not extract hidden states, using logits")
            hidden_states = outputs.logits
        
        # Get the last token's hidden state (for generation)
        if len(hidden_states.shape) == 3:
            last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_dim]
        else:
            last_hidden = hidden_states
        
        # Route to appropriate head
        if category == "abnormality_detection" or category == 1:
            logits = self.head_abnormality(last_hidden)
        elif category == "characteristics" or category == 2:
            logits = self.head_characteristics(last_hidden)
        elif category == "treatment" or category == 3:
            logits = self.head_treatment(last_hidden)
        else:
            # Default to abnormality detection
            logits = self.head_abnormality(last_hidden)
        
        result = {
            "logits": logits,
            "hidden_states": hidden_states,
            "base_outputs": outputs
        }
        
        # Generate CoT reasoning if requested
        if generate_cot:
            cot_text = self._generate_cot_reasoning(
                hidden_states,
                category,
                previous_frame_info
            )
            result["cot_reasoning"] = cot_text
        
        return result
    
    def _generate_cot_reasoning(
        self,
        hidden_states: torch.Tensor,
        category: str,
        previous_frame_info: Optional[Dict]
    ) -> str:
        """
        Generate CoT reasoning from hidden states.
        This is a placeholder - actual CoT generation happens in the prompt.
        """
        # In practice, CoT is generated by the model during text generation
        # This method can be used for post-processing or explicit CoT extraction
        return ""
    
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        prompt: str = "",
        category: str = "abnormality_detection",
        previous_frame_info: Optional[Dict] = None,
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
            previous_frame_info: Temporal context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
            
        Returns:
            Dictionary with generated text and reasoning
        """
        # Build CoT prompt
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from prompts.cot_templates import build_cot_prompt
        
        full_prompt = build_cot_prompt(
            question=prompt,
            category=category,
            previous_frame_info=previous_frame_info
        )
        
        # Prepare inputs
        if self.has_vision and images is not None:
            inputs = self.processor(
                images=images,
                text=full_prompt,
                return_tensors="pt",
                padding=True
            )
            pixel_values = inputs['pixel_values'].to(self.model.device)
            input_ids = inputs['input_ids'].to(self.model.device)
        else:
            input_ids = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True
            ).input_ids.to(self.model.device)
            pixel_values = None
        
        # Generate
        with torch.no_grad():
            if self.has_vision:
                outputs = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs
                )
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs
                )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract reasoning and answer (simple parsing)
        reasoning, answer = self._parse_cot_output(generated_text)
        
        return {
            "generated_text": generated_text,
            "reasoning": reasoning,
            "answer": answer
        }
    
    def _parse_cot_output(self, text: str) -> Tuple[str, str]:
        """Parse CoT output to extract reasoning and answer."""
        # Simple parsing - look for "Reasoning:" and "Answer:" markers
        if "Reasoning:" in text:
            parts = text.split("Reasoning:", 1)
            if len(parts) == 2:
                reasoning = parts[1].split("Answer:")[0].strip()
                answer = parts[1].split("Answer:")[1].strip() if "Answer:" in parts[1] else ""
                return reasoning, answer
        
        # Fallback: return all as reasoning, try to extract last sentence as answer
        lines = text.strip().split('\n')
        if len(lines) > 1:
            reasoning = '\n'.join(lines[:-1])
            answer = lines[-1]
        else:
            reasoning = ""
            answer = text
        
        return reasoning, answer


def create_model(
    base_model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05
) -> SurgicalCoTModel:
    """
    Factory function to create a SurgicalCoTModel.
    
    Args:
        base_model_name: Base model name
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        
    Returns:
        Initialized SurgicalCoTModel
    """
    lora_config = {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout
    } if use_lora else None
    
    model = SurgicalCoTModel(
        base_model_name=base_model_name,
        use_lora=use_lora,
        lora_config=lora_config
    )
    
    return model

