#!/usr/bin/env python3
"""
Test script to verify the evaluation matching fix works correctly.

Tests the critical bug fix where empty predictions were incorrectly marked as correct.
"""

def flexible_match_fixed(pred: str, gt: str) -> bool:
    """Fixed flexible_match function - tests the corrected logic"""
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


def flexible_match_broken(pred: str, gt: str) -> bool:
    """BROKEN version - demonstrates the bug"""
    pred = pred.lower().strip()
    gt = gt.lower().strip()
    
    # Exact match
    if pred == gt:
        return True
    
    # BUG: This returns True if pred is empty and gt is "yes"!
    # Because "" in "yes" returns True in Python!
    if gt in pred:
        return True
    
    # BUG: Same issue here
    if pred in gt:
        return True
    
    return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("EVALUATION BUG FIX TEST")
    print("=" * 80)
    print()
    
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
    
    print("Testing FIXED version:")
    print("-" * 80)
    passed_fixed = 0
    failed_fixed = 0
    
    for pred, gt, expected, desc in test_cases:
        result = flexible_match_fixed(pred, gt)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        
        if result == expected:
            passed_fixed += 1
        else:
            failed_fixed += 1
        
        print(f"{status}: {desc}")
        print(f"  Prediction: '{pred}' | Ground Truth: '{gt}'")
        print(f"  Expected: {expected}, Got: {result}")
        print()
    
    print(f"FIXED version: {passed_fixed} passed, {failed_fixed} failed")
    print()
    
    print("Testing BROKEN version (demonstrates bug):")
    print("-" * 80)
    passed_broken = 0
    failed_broken = 0
    
    for pred, gt, expected, desc in test_cases:
        result = flexible_match_broken(pred, gt)
        # For broken version, we expect it to fail on empty prediction cases
        if pred == "" and gt != "":
            # Broken version will incorrectly return True
            expected_broken = True  # This is the bug!
        else:
            expected_broken = expected
        
        status = "⚠️  BUG" if result != expected else "✅"
        
        if result == expected_broken:
            passed_broken += 1
        else:
            failed_broken += 1
        
        print(f"{status}: {desc}")
        print(f"  Prediction: '{pred}' | Ground Truth: '{gt}'")
        print(f"  Expected (correct): {expected}, Got (broken): {result}")
        if pred == "" and gt != "" and result == True:
            print(f"  ⚠️  BUG: Empty prediction incorrectly marked as correct!")
        print()
    
    print(f"BROKEN version: {passed_broken} passed (but wrong on empty predictions)")
    print()
    
    print("=" * 80)
    if failed_fixed == 0:
        print("✅ ALL TESTS PASSED - Bug fix verified!")
        print("=" * 80)
        print()
        print("The fix correctly handles empty predictions:")
        print("  - Empty prediction + non-empty GT → False ✅")
        print("  - Empty prediction + empty GT → True ✅")
        print("  - Non-empty prediction + empty GT → False ✅")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED - Bug fix incomplete!")
        print("=" * 80)
        exit(1)



