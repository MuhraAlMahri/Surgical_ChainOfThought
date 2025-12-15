#!/bin/bash
# Safe Cleanup Script for Surgical_COT experiments
# This script identifies and optionally removes large, safe-to-delete files

BASE_DIR="/l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments"
QLORA_DIR="${BASE_DIR}/qlora_experiments"

echo "=========================================="
echo "CLEANUP ANALYSIS - Safe to Delete"
echo "=========================================="
echo ""

# 1. Vision Cache (55GB total - SAFE TO DELETE)
echo "1. VISION CACHE (55GB) - SAFE TO DELETE"
echo "   These are cached image processing files that can be regenerated:"
echo "   - exp1/vision_cache (28GB)"
echo "   - exp1/vision_cache_v2 (27GB)"
echo ""

# 2. Old Checkpoints (Keep only latest)
echo "2. OLD CHECKPOINTS - Keep only latest, delete intermediate"
echo "   Current checkpoints found:"
find "${QLORA_DIR}/models" -name "checkpoint-*" -type d 2>/dev/null | while read dir; do
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    echo "   - $size: $dir"
done | head -10
echo "   Recommendation: Keep only the latest checkpoint per experiment"
echo ""

# 3. Old Log Files
echo "3. OLD LOG FILES (20MB) - Safe to delete old ones"
echo "   Total log files: $(find "${QLORA_DIR}/slurm/logs" -name "*.out" -o -name "*.err" 2>/dev/null | wc -l)"
echo "   Recommendation: Keep last 20-30 logs, delete older ones"
echo ""

# 4. ZIP Files
echo "4. ZIP FILES - Safe to delete if extracted"
echo "   - colab_package.zip (895MB)"
echo "   - exp2_colab_package.zip (if exists)"
echo ""

# 5. Test/Resolution files
echo "5. TEST FILES - Safe to delete"
echo "   - exp1/resolution_tests (698MB)"
echo ""

# 6. Old experiment outputs
echo "6. OLD OUTPUTS - Review before deleting"
echo "   - exp1/outputs (1.9GB) - Check if contains important results"
echo ""

echo "=========================================="
echo "ESTIMATED SPACE TO RECOVER: ~60GB"
echo "=========================================="
echo ""
echo "To proceed with cleanup, run:"
echo "  bash CLEANUP_SCRIPT.sh --execute"
echo ""

if [ "$1" == "--execute" ]; then
    echo "=========================================="
    echo "EXECUTING CLEANUP"
    echo "=========================================="
    echo ""
    
    # Delete vision caches
    echo "Deleting vision caches..."
    rm -rf "${BASE_DIR}/exp1/vision_cache"
    rm -rf "${BASE_DIR}/exp1/vision_cache_v2"
    echo "✓ Deleted ~55GB of vision cache"
    echo ""
    
    # Delete resolution tests
    echo "Deleting resolution test files..."
    rm -rf "${BASE_DIR}/exp1/resolution_tests"
    echo "✓ Deleted ~698MB of test files"
    echo ""
    
    # Delete ZIP files
    echo "Deleting ZIP files..."
    rm -f "${BASE_DIR}/colab_package.zip"
    rm -f "${BASE_DIR}/exp2_colab_package.zip"
    echo "✓ Deleted ZIP files"
    echo ""
    
    # Keep only latest checkpoints (manual review needed)
    echo ""
    echo "⚠️  CHECKPOINT CLEANUP - Manual review recommended"
    echo "   Keep only latest checkpoint per experiment:"
    echo "   - exp1_random: checkpoint-12840 (latest)"
    echo "   - exp2_qwen_reordered: checkpoint-12840 (latest)"
    echo "   - exp3_cxrtrek/stage1: checkpoint-4590 (latest)"
    echo "   - exp3_cxrtrek/stage2: checkpoint-8240 (latest)"
    echo "   - exp3_cxrtrek/stage3: checkpoint-15 (latest)"
    echo "   - exp4_curriculum/stage1: checkpoint-4590 (latest)"
    echo "   - exp4_curriculum/stage2: checkpoint-3294 (latest)"
    echo "   - exp5_sequential_cot: checkpoint-12840 (latest)"
    echo ""
    echo "   To delete old checkpoints, run:"
    echo "   bash CLEANUP_SCRIPT.sh --delete-old-checkpoints"
    echo ""
    
    # Clean old log files (keep last 30)
    echo "Cleaning old log files (keeping last 30)..."
    cd "${QLORA_DIR}/slurm/logs"
    ls -t *.out 2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null
    ls -t *.err 2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null
    echo "✓ Cleaned old log files"
    echo ""
    
    echo "=========================================="
    echo "CLEANUP COMPLETED"
    echo "=========================================="
    echo ""
    echo "Space recovered: ~56GB"
    echo ""
    
elif [ "$1" == "--delete-old-checkpoints" ]; then
    echo "=========================================="
    echo "DELETING OLD CHECKPOINTS"
    echo "=========================================="
    echo ""
    
    # Delete old checkpoints, keep only latest
    echo "Exp1: Keeping checkpoint-12840, deleting others..."
    rm -rf "${QLORA_DIR}/models/exp1_random/checkpoint-7710"
    rm -rf "${QLORA_DIR}/models/exp1_random/checkpoint-10280"
    
    echo "Exp2: Keeping checkpoint-12840, deleting others..."
    rm -rf "${QLORA_DIR}/models/exp2_qwen_reordered/checkpoint-7710"
    rm -rf "${QLORA_DIR}/models/exp2_qwen_reordered/checkpoint-10280"
    
    echo "Exp3 Stage1: Keeping checkpoint-4590, deleting others..."
    rm -rf "${QLORA_DIR}/models/exp3_cxrtrek/stage1/checkpoint-3668"
    rm -rf "${QLORA_DIR}/models/exp3_cxrtrek/stage1/checkpoint-4585"
    
    echo "Exp3 Stage2: Keeping checkpoint-8240, deleting others..."
    rm -rf "${QLORA_DIR}/models/exp3_cxrtrek/stage2/checkpoint-4941"
    rm -rf "${QLORA_DIR}/models/exp3_cxrtrek/stage2/checkpoint-8235"
    
    echo "Exp3 Stage3: Keeping checkpoint-15, deleting others..."
    rm -rf "${QLORA_DIR}/models/exp3_cxrtrek/stage3/checkpoint-9"
    rm -rf "${QLORA_DIR}/models/exp3_cxrtrek/stage3/checkpoint-12"
    
    echo "Exp4 Stage1: Keeping checkpoint-4590, deleting others..."
    rm -rf "${QLORA_DIR}/models/exp4_curriculum/stage1/checkpoint-3668"
    rm -rf "${QLORA_DIR}/models/exp4_curriculum/stage1/checkpoint-4585"
    
    echo "Exp4 Stage2: Keeping checkpoint-3294, deleting others..."
    rm -rf "${QLORA_DIR}/models/exp4_curriculum/stage2/checkpoint-1647"
    
    echo "Exp5: Keeping checkpoint-12840, deleting others..."
    rm -rf "${QLORA_DIR}/models/exp5_sequential_cot/checkpoint-7710"
    rm -rf "${QLORA_DIR}/models/exp5_sequential_cot/checkpoint-10280"
    
    echo ""
    echo "✓ Deleted old checkpoints (~500MB-1GB recovered)"
    echo ""
fi

