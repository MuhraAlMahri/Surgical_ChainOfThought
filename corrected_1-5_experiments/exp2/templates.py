"""
Templates for Exp1 instruction fine-tuning with LLaVA-style conversations.
"""

def build_conversation(question_type, question, answer_candidates=None, answer=None, for_training=True):
    """
    Build LLaVA-style conversation with system/user/assistant turns.
    
    Args:
        question_type: Type of question (yes_no, color, etc.)
        question: The question text
        answer_candidates: Optional list of valid answers
        answer: Ground truth answer (only for training)
        for_training: If True, include assistant response with <ANS> sentinels
    
    Returns:
        List of conversation turns in LLaVA format
    """
    # Build user prompt
    user_text = f"Question type: {question_type}\nQuestion: {question}"
    if answer_candidates:
        cand = ", ".join(answer_candidates)
        user_text += f"\nValid answers: {cand}"
    
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a surgical VQA assistant. Answer with a single word/number when possible."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text}
            ]
        }
    ]
    
    if for_training and answer is not None:
        # Add assistant response with sentinels for training
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": f"<ANS>{answer}</ANS>"}]
        })
    
    return conversation


def build_conversation_inference(question_type, question, answer_candidates=None):
    """Convenience function for inference (no answer, no sentinels)."""
    return build_conversation(question_type, question, answer_candidates, answer=None, for_training=False)
