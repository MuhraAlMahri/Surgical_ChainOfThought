#!/usr/bin/env python3
"""
Hybrid CoT prompt builder.

Key principle: Provide structure HINTS, not step-by-step instructions.
Let the model generate its own reasoning.

CORRECT:   "Analyze the surgical scene for abnormalities."
INCORRECT: "Step 1: Look for lesions. Step 2: Check color. Step 3: ..."
"""

from typing import Dict, Optional
from prompts.cot_builder import build_cot_prompt as _build_cot_prompt_base


def build_cot_prompt(
    question: str,
    category: str,
    previous_frame_info: Optional[Dict] = None,
    model_type: str = "qwen3vl"
) -> str:
    """
    Build hybrid CoT prompt.
    
    Args:
        question: The question to answer
        category: "abnormality_detection", "characteristics", or "treatment"
        previous_frame_info: Dict with 'summary' and 'motion' from previous frame
        model_type: "qwen3vl", "medgemma", or "llava_med"
    
    Returns:
        Formatted prompt string
    """
    
    # Structure hints (NOT step-by-step instructions!)
    structure_hints = {
        "abnormality_detection": "Analyze the surgical scene for any abnormalities or instruments.",
        "characteristics": "Examine the specific properties of the identified findings.",
        "treatment": "Consider the clinical implications based on your observations."
    }
    
    # Map stage numbers to category names
    category_map = {
        1: "abnormality_detection",
        2: "characteristics",
        3: "treatment",
        "1": "abnormality_detection",
        "2": "characteristics",
        "3": "treatment"
    }
    
    if category in category_map:
        category = category_map[category]
    
    hint = structure_hints.get(category, structure_hints["abnormality_detection"])
    
    # Model-specific formatting
    if model_type == "qwen3vl":
        messages = [
            {"role": "system", "content": "You are a medical AI assistant analyzing surgical images."}
        ]
        
        # Add temporal context if available
        if previous_frame_info:
            messages.append({
                "role": "assistant",
                "content": f"Previous frame: {previous_frame_info.get('summary', 'N/A')}\n"
                          f"Motion: {previous_frame_info.get('motion', 'N/A')}"
            })
        
        # Add current question with structure hint
        messages.append({
            "role": "user",
            "content": f"{hint}\n\n"
                      f"Question: {question}\n\n"
                      f"Think through your reasoning step by step, then provide your answer.\n\n"
                      f"Reasoning:"
        })
        
        # Return messages for chat template (will be formatted by processor)
        return messages
    
    elif model_type == "medgemma":
        prompt = "<start_of_turn>user\n"
        
        if previous_frame_info:
            prompt += f"Prior findings: {previous_frame_info.get('summary', 'N/A')}\n"
            prompt += f"Motion detected: {previous_frame_info.get('motion', 'N/A')}\n\n"
        
        prompt += f"{hint}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Provide your clinical reasoning followed by your answer.\n\n"
        prompt += "Reasoning:<end_of_turn>\n<start_of_turn>model\n"
        
        return prompt
    
    else:  # llava_med
        prompt = "USER: <image>\n"
        
        if previous_frame_info:
            prompt += f"Context from previous frame:\n"
            prompt += f"- Observations: {previous_frame_info.get('summary', 'N/A')}\n"
            prompt += f"- Changes: {previous_frame_info.get('motion', 'N/A')}\n\n"
        
        prompt += f"{hint}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Think through your medical reasoning step by step, then provide your answer.\n\n"
        prompt += "ASSISTANT: Let me analyze this systematically.\n\nReasoning: "
        
        return prompt


def format_prompt_for_model(
    prompt: str,
    model_type: str = "qwen3vl",
    processor=None
) -> str:
    """
    Format prompt according to model-specific chat template.
    
    Args:
        prompt: Prompt string or messages list
        model_type: Model type
        processor: Model processor (for chat template)
        
    Returns:
        Formatted prompt string
    """
    if model_type == "qwen3vl" and isinstance(prompt, list) and processor:
        # Use processor's chat template
        return processor.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
    
    return prompt if isinstance(prompt, str) else str(prompt)













