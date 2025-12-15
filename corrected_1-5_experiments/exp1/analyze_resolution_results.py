#!/usr/bin/env python3
"""
Analyze resolution test results and provide recommendations.
"""

import json
from pathlib import Path
import sys

def analyze_results():
    results_file = Path(__file__).parent / "resolution_tests" / "comparison_summary.json"
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("Run the resolution test first: sbatch slurm/test_resolutions.slurm")
        sys.exit(1)
    
    with open(results_file) as f:
        results = json.load(f)
    
    if not results:
        print("❌ No results found in the file")
        sys.exit(1)
    
    print("\n" + "="*100)
    print("RESOLUTION TEST RESULTS & ANALYSIS")
    print("="*100)
    print()
    
    # Print detailed table
    print(f"{'Resolution':<25} {'Resolution':<15} {'Time/Step':<12} {'Full Train':<15} {'Eval Loss':<12} {'Speedup':<10}")
    print(f"{'Test Name':<25} {'(approx)':<15} {'(seconds)':<12} {'(hours)':<15} {'(lower=better)':<12} {'vs Full':<10}")
    print("-"*100)
    
    # Find baseline (full res) for comparison
    baseline = next((r for r in results if 'full_res' in r['test_name']), results[-1])
    baseline_time = baseline['estimated_full_training_hours']
    
    for r in sorted(results, key=lambda x: x['estimated_full_training_hours']):
        # Estimate resolution from pixels
        approx_res = int(r['max_pixels'] ** 0.5)
        res_str = f"{approx_res}×{approx_res}"
        
        speedup = baseline_time / r['estimated_full_training_hours']
        
        print(f"{r['test_name']:<25} {res_str:<15} {r['avg_time_per_step']:>10.2f}  "
              f"{r['estimated_full_training_hours']:>13.1f}  "
              f"{r.get('eval_loss', float('nan')):>10.4f}  "
              f"{speedup:>8.2f}x")
    
    print("="*100)
    print()
    
    # Analysis and recommendations
    print("📊 ANALYSIS & RECOMMENDATIONS")
    print("-"*100)
    print()
    
    # Sort by speed
    sorted_by_speed = sorted(results, key=lambda x: x['estimated_full_training_hours'])
    fastest = sorted_by_speed[0]
    slowest = sorted_by_speed[-1]
    
    print(f"⚡ FASTEST: {fastest['test_name']}")
    print(f"   Time: {fastest['estimated_full_training_hours']:.1f} hours")
    print(f"   Speed: {fastest['avg_time_per_step']:.2f}s/step")
    print(f"   Loss: {fastest.get('eval_loss', 'N/A'):.4f}")
    print()
    
    print(f"🐌 SLOWEST: {slowest['test_name']}")
    print(f"   Time: {slowest['estimated_full_training_hours']:.1f} hours")
    print(f"   Speed: {slowest['avg_time_per_step']:.2f}s/step")
    print(f"   Loss: {slowest.get('eval_loss', 'N/A'):.4f}")
    print()
    
    # Find best balance
    # Score based on: speed (weight 0.6) + quality/loss (weight 0.4)
    if all('eval_loss' in r and r['eval_loss'] is not None for r in results):
        min_loss = min(r['eval_loss'] for r in results)
        max_loss = max(r['eval_loss'] for r in results)
        min_time = min(r['estimated_full_training_hours'] for r in results)
        max_time = max(r['estimated_full_training_hours'] for r in results)
        
        for r in results:
            # Normalize metrics to 0-1 (lower is better for both)
            norm_time = (r['estimated_full_training_hours'] - min_time) / (max_time - min_time) if max_time > min_time else 0
            norm_loss = (r['eval_loss'] - min_loss) / (max_loss - min_loss) if max_loss > min_loss else 0
            
            # Combined score (lower is better)
            r['score'] = 0.6 * norm_time + 0.4 * norm_loss
        
        best_balance = min(results, key=lambda x: x['score'])
        
        print(f"⚖️  BEST BALANCE (Speed + Quality):")
        print(f"   {best_balance['test_name']}")
        print(f"   Time: {best_balance['estimated_full_training_hours']:.1f} hours")
        print(f"   Speed: {best_balance['avg_time_per_step']:.2f}s/step")
        print(f"   Loss: {best_balance['eval_loss']:.4f}")
        print(f"   Score: {best_balance['score']:.3f} (lower is better)")
        print()
    
    # Specific recommendations
    print("💡 RECOMMENDATIONS:")
    print()
    
    if fastest['estimated_full_training_hours'] < 20:
        print(f"   ✅ Use '{fastest['test_name']}' if you want FAST iteration (<20 hours)")
    
    medium_res = [r for r in results if 'medium' in r['test_name'].lower()]
    if medium_res and medium_res[0]['estimated_full_training_hours'] < 30:
        print(f"   ⚖️  Use '{medium_res[0]['test_name']}' for BALANCED speed/quality")
    
    if slowest['estimated_full_training_hours'] > 40:
        print(f"   🎯 Use '{slowest['test_name']}' only if you need MAXIMUM quality")
    
    print()
    print("   📝 Next steps:")
    print("      1. Check eval_loss differences - if small, choose faster option")
    print("      2. Run full evaluation on validation set for chosen resolution")
    print("      3. Update config_exp1_category_based.yaml with chosen resolution")
    print()
    
    print("="*100)
    print()
    
    # Show how to apply chosen resolution
    print("🔧 TO APPLY A RESOLUTION:")
    print("-"*100)
    print()
    print("1. Edit exp1/dataset.py to set resolution in processor initialization:")
    print("   processor.image_processor.min_pixels = YOUR_MIN_PIXELS")
    print("   processor.image_processor.max_pixels = YOUR_MAX_PIXELS")
    print()
    print("2. Update max_seq_len in config_exp1_category_based.yaml:")
    print("   For 448x448:  max_seq_len: 800")
    print("   For 768x768:  max_seq_len: 1800")
    print("   For 1024x1024: max_seq_len: 2500")
    print("   For full res:  max_seq_len: 2900")
    print()
    print("3. Resubmit training with new settings")
    print()
    print("="*100)


if __name__ == "__main__":
    analyze_results()







