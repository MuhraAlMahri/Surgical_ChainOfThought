"""
Generate predictions using Exp2 checkpoint-642 (completed model from Nov 12)
Uses test set for evaluation
"""
import os
import sys
import json
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoProcessor, AutoModelForImageTextToText, GenerationConfig
from peft import PeftModel
from PIL import Image
import torch


def load_model(checkpoint_dir):
    """Load model with LoRA adapter from checkpoint"""
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
    
    print(f"Loading base model: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    print(f"Loading LoRA adapter from: {checkpoint_dir}")
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model = model.merge_and_unload()  # Merge LoRA for faster inference
    
    model.eval().cuda()
    
    proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tok = proc.tokenizer
    
    return model, tok, proc


def generate_answer(model, proc, tok, img_path, question, max_new=50):
    """Generate answer for a single question"""
    # Simple conversation format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Apply chat template
    text = proc.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process with image
    enc = proc(
        text=[text],
        images=[Image.open(img_path).convert("RGB")],
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id
        )
    
    # Decode only the generated part
    prompt_len = enc["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    answer = tok.decode(generated_ids, skip_special_tokens=True).strip()
    
    return answer


def main():
    script_dir = Path(__file__).parent
    checkpoint_dir = script_dir / "outputs/checkpoint-642"
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint not found: {checkpoint_dir}")
        return
    
    print("="*80)
    print("EXP2 PREDICTION - Checkpoint 642 (768×768, 1 epoch, Qwen Reordered)")
    print("="*80)
    
    # Load model
    model, tok, proc = load_model(checkpoint_dir)
    
    # Load test data
    test_path = script_dir.parent / "datasets/kvasir_ULTRA_CONDENSED/test_CATEGORY_BASED.jsonl"
    img_root = "/l/users/muhra.almahri/Surgical_COT/datasets/Kvasir-VQA/raw/images"
    
    print(f"\nGenerating predictions on test set: {test_path}")
    preds = []
    
    with open(test_path) as f:
        for idx, line in enumerate(f):
            if idx % 100 == 0:
                print(f"  Processed {idx} samples...")
            
            ex = json.loads(line)
            img_file = ex.get('image_filename') or ex.get('image_id')
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                img_file = f"{img_file}.jpg"
            img_path = f"{img_root}/{img_file}"
            
            try:
                ans = generate_answer(model, proc, tok, img_path, ex["question"])
            except Exception as e:
                print(f"  Error on sample {idx}: {e}")
                ans = ""
            
            preds.append({
                "id": ex.get("image_id"),
                "question": ex["question"],
                "pred": ans,
                "gt": ex["answer"],
                "category": ex.get("category", "unknown")
            })
    
    # Save predictions
    output_file = script_dir / "outputs/predictions_checkpoint642.jsonl"
    print(f"\nSaving predictions to {output_file}...")
    with open(output_file, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    
    print(f"✅ Done! Generated {len(preds)} predictions")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()





