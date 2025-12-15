#!/usr/bin/env python3
"""
Test script to verify the evaluation matching fix works correctly.

Tests the critical bug fix where empty predictions were incorrectly marked as correct.
"""

import sys
from pathlib import Path
import string
from difflib import SequenceMatcher

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the fixed matching function directly
from evaluate_multihead_cot_comprehensive import flexible_match

# Also test the logic directly
def test_flexible_match_logic(pred: str, gt: str) -> bool:
    """Test the fixed logic directly"""
    pred = pred.lower().strip()
    gt = gt.lower().strip()
    
    # CRITICAL FIX: Empty predictions only correct if ground truth is also empty
    if not pred:
        return not gt  # Both empty = match, otherwise False
    
    if not gt:
        return False  # Empty ground truth, non-empty prediction = False
    
    # Exact match
    if pred == gt:
        return True
    
    # Ground truth is IN prediction (only if both are non-empty)
    if gt in pred:
        return True
    
    # Prediction is IN ground truth (only if both are non-empty)
    if pred in gt:
        return True
    
    return False

def test_flexible_match():
    """Test the flexible_match function from evaluate_multihead_cot_comprehensive.py"""
    print("=" * 80)
    print("Testing flexible_match from evaluate_multihead_cot_comprehensive.py")
    print("=" * 80)
    
    test_cases = [
        # (prediction, ground_truth, expected, description)
        ("", "yes", False, "Empty prediction with non-empty GT → False (BUG FIX)"),
        ("", "", True, "Both empty → True"),
        ("yes", "yes", True, "Exact match → True"),
        ("yes", "not", False, "Different answers → False"),
        ("yes", "yes, I agree", True, "GT contains prediction → True"),
        ("I think yes", "yes", True, "Prediction contains GT → True"),
        ("", "no", False, "Empty prediction with 'no' GT → False (BUG FIX)"),
        ("no", "", False, "Non-empty prediction with empty GT → False"),
    ]
    
    passed = 0
    failed = 0
    
    for pred, gt, expected, desc in test_cases:
        result = flexible_match(pred, gt)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status}: {desc}")
        print(f"  Prediction: '{pred}' | Ground Truth: '{gt}'")
        print(f"  Expected: {expected}, Got: {result}")
        print()
    
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def test_flexible_match_logic_direct():
    """Test the fixed logic directly (without imports)"""
    print("=" * 80)
    print("Testing flexible_match logic directly")
    print("=" * 80)
    
    test_cases = [
        # (prediction, ground_truth, expected, description)
        ("", "yes", False, "Empty prediction with non-empty GT → False (BUG FIX)"),
        ("", "", True, "Both empty → True"),
        ("yes", "yes", True, "Exact match → True"),
        ("yes", "not", False, "Different answers → False"),
    ]
    
    passed = 0
    failed = 0
    
    for pred, gt, expected, desc in test_cases:
        result = test_flexible_match_logic(pred, gt)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status}: {desc}")
        print(f"  Prediction: '{pred}' | Ground Truth: '{gt}'")
        print(f"  Expected: {expected}, Got: {result}")
        print()
    
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EVALUATION BUG FIX TEST")
    print("=" * 80)
    print()
    
    test1_passed = test_flexible_match()
    print()
    test2_passed = test_flexible_match_logic_direct()
    
    print()
    print("=" * 80)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED - Bug fix verified!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - Bug fix incomplete!")
        print("=" * 80)
        sys.exit(1)

