#!/usr/bin/env python3
"""
Hybrid CoT Prompt Templates
Model generates reasoning, but follows clinical structure guidance.
"""

from typing import Dict, Optional, List


def build_cot_prompt(
    question: str,
    category: str,
    previous_frame_info: Optional[Dict] = None,
    dataset: str = "kvasir"
) -> str:
    """
    Build a hybrid CoT prompt that guides structure without prescribing steps.
    
    Args:
        question: The question to answer
        category: Question category (abnormality_detection, characteristics, treatment)
        previous_frame_info: Temporal context from previous frame
        dataset: Dataset name (kvasir or endovis)
        
    Returns:
        Formatted prompt string
    """
    # Base structure hints (NOT step-by-step instructions)
    structure_hints = {
        "abnormality_detection": (
            "Consider what visual features indicate normality or abnormality "
            "in this surgical scene. Look for instruments, anatomical structures, "
            "or pathological findings."
        ),
        "characteristics": (
            "Based on the identified findings, analyze their specific properties "
            "such as color, location, size, type, or quantity."
        ),
        "treatment": (
            "Given the observations, consider the clinical implications, "
            "diagnosis, and potential treatment recommendations."
        )
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
    
    # Normalize category
    if category in category_map:
        category = category_map[category]
    
    # Get structure hint
    hint = structure_hints.get(category, structure_hints["abnormality_detection"])
    
    # Build prompt
    prompt = f"""You are analyzing a surgical/endoscopic image.

Question: {question}
"""
    
    # Add previous frame context if available (TEMPORAL CoT)
    if previous_frame_info:
        prompt += _add_temporal_context(previous_frame_info)
    
    # Add clinical structure hint (NOT step-by-step instructions)
    prompt += f"""
{hint}

Think through your reasoning step by step, then provide your answer.

Reasoning: """
    
    return prompt


def _add_temporal_context(previous_frame_info: Dict) -> str:
    """
    Add temporal context from previous frame to prompt.
    
    Args:
        previous_frame_info: Dictionary with previous frame information
        
    Returns:
        Formatted temporal context string
    """
    context_parts = []
    
    # Previous observations
    if "observations" in previous_frame_info:
        obs = previous_frame_info["observations"]
        if isinstance(obs, dict):
            # Format as readable text
            obs_text = ", ".join([f"{k}: {v}" for k, v in obs.items() if v])
        elif isinstance(obs, list):
            obs_text = ", ".join([str(o) for o in obs])
        else:
            obs_text = str(obs)
        
        if obs_text:
            context_parts.append(f"Previous observations: {obs_text}")
    
    # Motion description
    if "motion_description" in previous_frame_info:
        motion = previous_frame_info["motion_description"]
        if motion:
            context_parts.append(f"Camera movement: {motion}")
    
    # Motion info from temporal linker
    if "motion_info" in previous_frame_info:
        motion_info = previous_frame_info["motion_info"]
        if isinstance(motion_info, dict) and "description" in motion_info:
            context_parts.append(f"Scene changes: {motion_info['description']}")
    
    if context_parts:
        return "Context from previous frame:\n" + "\n".join([f"- {part}" for part in context_parts]) + "\n\n"
    
    return ""


def build_stage_dependent_prompt(
    question: str,
    stage: int,
    previous_stage_predictions: Optional[Dict] = None,
    previous_frame_info: Optional[Dict] = None
) -> str:
    """
    Build prompt that incorporates previous stage predictions.
    
    For stage 2 (characteristics), reuse stage 1 predictions.
    For stage 3 (treatment), reuse stage 1 and 2 predictions.
    
    Args:
        question: Current question
        stage: Current stage (1, 2, or 3)
        previous_stage_predictions: Predictions from previous stages
        previous_frame_info: Temporal context
        
    Returns:
        Formatted prompt
    """
    prompt = build_cot_prompt(
        question=question,
        category=stage,
        previous_frame_info=previous_frame_info
    )
    
    # Add previous stage context
    if previous_stage_predictions and stage > 1:
        context = "Based on the previous analysis:\n"
        
        if stage == 2 and 1 in previous_stage_predictions:
            # Stage 2: Use stage 1 (abnormality detection) results
            stage1_preds = previous_stage_predictions[1]
            if isinstance(stage1_preds, dict):
                context += f"- Detected findings: {stage1_preds}\n"
            else:
                context += f"- Previous detection: {stage1_preds}\n"
        
        elif stage == 3:
            # Stage 3: Use both stage 1 and 2 results
            if 1 in previous_stage_predictions:
                stage1_preds = previous_stage_predictions[1]
                context += f"- Detected findings: {stage1_preds}\n"
            if 2 in previous_stage_predictions:
                stage2_preds = previous_stage_predictions[2]
                context += f"- Characteristics: {stage2_preds}\n"
        
        # Insert context before reasoning
        if "Reasoning:" in prompt:
            prompt = prompt.replace("Reasoning:", context + "\nReasoning:")
        else:
            prompt = prompt + "\n" + context
    
    return prompt


def get_category_examples(category: str) -> List[str]:
    """Get example questions for a category."""
    examples = {
        "abnormality_detection": [
            "Is there a polyp?",
            "What instruments are present?",
            "Are there any abnormalities?",
            "What anatomical landmarks are visible?"
        ],
        "characteristics": [
            "What is the color of the polyp?",
            "Where is the lesion located?",
            "How many instruments are there?",
            "What type of instrument is this?"
        ],
        "treatment": [
            "What is the diagnosis?",
            "What treatment is recommended?",
            "Have all polyps been removed?",
            "What is the clinical significance?"
        ]
    }
    return examples.get(category, [])



