"""
Diagnostic script to verify LLaVA-style conversation with sentinel-based label masking.
Tests that:
1. Conversation template is correctly applied
2. Sentinels <ANS> and </ANS> are present
3. Labels are properly masked (only answer tokens between sentinels unmasked)
4. Decoded text shows correct structure
5. Vision tensors are properly aligned
"""
import sys
from pathlib import Path
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoProcessor
from dataset import VQASFTDataset
import yaml


def test_llava_sentinel_masking():
    print("=" * 80)
    print("LLAVA-STYLE CONVERSATION + SENTINEL MASKING TEST")
    print("=" * 80)
    
    # Load config
    script_dir = Path(__file__).parent
    cfg_path = script_dir / "config_exp1.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    
    model_name = cfg["model_name"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tok = processor.tokenizer
    
    # Test sentinel tokenization
    print("\n1. Testing sentinel tokenization:")
    print("-" * 80)
    
    ans_start = tok("<ANS>", add_special_tokens=False)
    ans_end = tok("</ANS>", add_special_tokens=False)
    
    print(f"<ANS> tokens: {ans_start['input_ids']} -> '{tok.decode(ans_start['input_ids'])}'")
    print(f"</ANS> tokens: {ans_end['input_ids']} -> '{tok.decode(ans_end['input_ids'])}'")
    
    # Load dataset
    print("\n2. Loading enriched dataset:")
    print("-" * 80)
    
    base_dir = script_dir.parent
    train_input = base_dir / cfg["data"]["train_jsonl"]
    train_enriched = str(train_input).replace(".jsonl", ".enriched.jsonl")
    
    # Create enriched data if needed
    if not Path(train_enriched).exists():
        print(f"Creating enriched data from {train_input}...")
        sys.path.insert(0, str(script_dir / "data"))
        import schema
        import json
        import re
        
        def normalize_answer_local(ans):
            x = ans.strip().lower()
            x = re.sub(r"[^\w\.\-\% ]+", "", x)
            return x
        
        out = []
        with open(train_input, "r") as f:
            for line in f:
                ex = json.loads(line)
                q = ex["question"]
                gt = normalize_answer_local(ex["answer"])
                qtype = ex.get("question_type") or schema.infer_question_type(q)
                ex["question_type"] = qtype
                ex["answer"] = gt
                ex["answer_candidates"] = schema.build_candidates(qtype, ex)
                out.append(ex)
        
        Path(train_enriched).parent.mkdir(parents=True, exist_ok=True)
        with open(train_enriched, "w") as f:
            for ex in out:
                f.write(json.dumps(ex) + "\n")
        print(f"Created {train_enriched}")
    
    # Load dataset
    ds = VQASFTDataset(
        train_enriched,
        cfg["data"]["image_root"],
        model_name,
        max_len=512
    )
    
    print(f"Dataset loaded: {len(ds)} samples")
    
    # Test first 5 samples
    print("\n3. Testing LLaVA conversation + sentinel masking:")
    print("=" * 80)
    
    for i in range(min(5, len(ds))):
        print(f"\nSample {i}:")
        print("-" * 80)
        
        sample = ds[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        
        # Decode full input
        full_text = tok.decode(input_ids, skip_special_tokens=False)
        
        # Check if sentinels are present
        has_ans_start = "<ANS>" in full_text
        has_ans_end = "</ANS>" in full_text
        
        print(f"Sentinels present: <ANS>={has_ans_start}, </ANS>={has_ans_end}")
        
        # Show truncated text
        print(f"\nFull decoded input (truncated to 500 chars):")
        print(full_text[:500] + "..." if len(full_text) > 500 else full_text)
        
        # Find unmasked positions
        unmasked_positions = (labels != -100).nonzero(as_tuple=True)[0]
        
        if len(unmasked_positions) > 0:
            print(f"\nUnmasked positions: {unmasked_positions.tolist()}")
            print(f"Unmasked token count: {len(unmasked_positions)}")
            print(f"Decoded answer (from labels): '{tok.decode(labels[unmasked_positions])}'")
            
            # Show context around answer
            start = max(0, unmasked_positions[0].item() - 15)
            end = min(len(input_ids), unmasked_positions[-1].item() + 15)
            context = tok.decode(input_ids[start:end], skip_special_tokens=False)
            print(f"\nContext around answer:")
            print(f"  {context}")
        else:
            print("\n⚠️  WARNING: No unmasked positions found!")
            print("   Sentinel detection may have failed.")
        
        # Check shapes
        print(f"\nShapes:")
        print(f"  input_ids: {input_ids.shape}")
        print(f"  labels: {labels.shape}")
        print(f"  attention_mask: {sample['attention_mask'].shape}")
        if 'pixel_values' in sample:
            print(f"  pixel_values: {sample['pixel_values'].shape}")
        if 'image_grid_thw' in sample:
            print(f"  image_grid_thw: {sample['image_grid_thw'].shape}")
        
        # Calculate masking ratio
        total_tokens = (input_ids != tok.pad_token_id).sum().item()
        answer_tokens = len(unmasked_positions)
        mask_ratio = (total_tokens - answer_tokens) / total_tokens if total_tokens > 0 else 0
        print(f"\nMasking statistics:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Masked tokens: {total_tokens - answer_tokens}")
        print(f"  Answer tokens: {answer_tokens}")
        print(f"  Mask ratio: {mask_ratio:.2%}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("\nExpected behavior:")
    print("  ✓ Sentinels <ANS> and </ANS> should be present in decoded text")
    print("  ✓ Only tokens BETWEEN sentinels should be unmasked (labels != -100)")
    print("  ✓ Sentinel tokens themselves should be MASKED (labels = -100)")
    print("  ✓ Decoded answer should match ground truth")
    print("  ✓ All shapes should be consistent")
    print("  ✓ Mask ratio should be 97-99% (most tokens masked)")
    print("\nIf any ⚠️ warnings appear, check:")
    print("  - Are sentinels in the conversation format?")
    print("  - Are sentinel tokens being found correctly?")
    print("  - Is truncation cutting off the answer?")


if __name__ == "__main__":
    test_llava_sentinel_masking()

















