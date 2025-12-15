#!/bin/bash
# Quick status check script

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Multi-Head Temporal CoT - Quick Status                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Current jobs
echo "🔵 RUNNING JOBS:"
squeue -u $USER -o "%.8i %-20j %-10T %10M %R" 2>/dev/null | head -5
if [ $? -ne 0 ] || [ -z "$(squeue -u $USER 2>/dev/null)" ]; then
    echo "   (none)"
fi
echo ""

# Categorization status
echo "📊 CATEGORIZATION:"
if [ -f "data/categorized/train_categorized.json" ]; then
    COUNT=$(wc -l < data/categorized/train_categorized.json 2>/dev/null || echo "0")
    SIZE=$(du -h data/categorized/train_categorized.json 2>/dev/null | cut -f1)
    echo "   ✅ Complete - $COUNT samples, $SIZE"
else
    echo "   ❌ Not found - Run: sbatch slurm/01_categorize_questions.slurm"
fi
echo ""

# Checkpoints
echo "💾 CHECKPOINTS:"
if [ -d "checkpoints" ] && [ "$(ls -A checkpoints 2>/dev/null)" ]; then
    ls -1t checkpoints/ | head -3 | while read dir; do
        echo "   📁 $dir"
    done
else
    echo "   (none)"
fi
echo ""

# Latest logs
echo "📋 LATEST LOGS:"
LATEST_OUT=$(ls -t slurm/logs/*.out 2>/dev/null | head -1)
LATEST_ERR=$(ls -t slurm/logs/*.err 2>/dev/null | head -1)
if [ -n "$LATEST_OUT" ]; then
    echo "   Output: $(basename $LATEST_OUT)"
    echo "   Error:  $(basename $LATEST_ERR)"
fi
echo ""

# Next steps
echo "🚀 NEXT STEPS:"
if [ -f "data/categorized/train_categorized.json" ]; then
    echo "   ✅ Ready to train!"
    echo "   Run: sbatch slurm/03_train_unified.slurm kvasir"
    echo "   Or:  sbatch slurm/04_train_sequential.slurm kvasir"
else
    echo "   ⏳ Wait for categorization to complete"
fi
echo ""














