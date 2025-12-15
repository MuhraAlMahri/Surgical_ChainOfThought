#!/bin/bash
# Job Monitoring Script

echo "=========================================="
echo "JOB MONITORING DASHBOARD"
echo "=========================================="
echo ""

# Current running jobs
echo "=== CURRENT JOBS ==="
squeue -u muhra.almahri -o "%.18i %.9P %.12j %.8u %.2t %.10M %.6D %R"
echo ""

# Check zero-shot job progress
ZEROSHOT_JOB=$(squeue -u muhra.almahri -o "%.18i %.j" | grep zeroshot | head -1 | awk '{print $1}')
if [ ! -z "$ZEROSHOT_JOB" ]; then
    echo "=== ZERO-SHOT JOB PROGRESS (Job $ZEROSHOT_JOB) ==="
    LOG_FILE="slurm/logs/zeroshot_batch1_${ZEROSHOT_JOB}.out"
    if [ -f "$LOG_FILE" ]; then
        # Count progress
        EVAL_COUNT=$(grep -c "Evaluating (Zero-Shot)" "$LOG_FILE" 2>/dev/null || echo "0")
        echo "  Evaluation progress: Processing samples..."
        
        # Check for errors
        ERROR_COUNT=$(grep -c "Image not found" "$LOG_FILE" 2>/dev/null || echo "0")
        if [ "$ERROR_COUNT" -gt 0 ]; then
            echo "  ⚠️  Errors found: $ERROR_COUNT image not found errors"
        fi
        
        # Check if completed
        if grep -q "Zero-shot evaluation completed" "$LOG_FILE" 2>/dev/null; then
            echo "  ✓ Job completed!"
            tail -10 "$LOG_FILE" | grep -E "Accuracy|Results saved"
        else
            echo "  Status: Running..."
            tail -3 "$LOG_FILE"
        fi
    fi
    echo ""
fi

# Recent results
echo "=== RECENT RESULTS ==="
cd /l/users/muhra.almahri/Surgical_COT/corrected_1-5_experiments/qlora_experiments
python3 << 'PYEOF'
import json
import os
from datetime import datetime

results_dir = "results"
if os.path.exists(results_dir):
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith("_evaluation.json") or fname.endswith("_zeroshot.json"):
            fpath = os.path.join(results_dir, fname)
            try:
                mtime = os.path.getmtime(fpath)
                mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
                
                with open(fpath, 'r') as f:
                    data = json.load(f)
                
                total = data.get('total', 0)
                correct = data.get('correct', 0)
                accuracy = data.get('accuracy', 0.0)
                
                if total > 0:
                    status = "✓"
                    print(f"{status} {fname:30s} | {mtime_str} | Acc: {accuracy:6.2f}% ({correct}/{total})")
                else:
                    status = "⚠️"
                    errors = len(data.get('errors', []))
                    print(f"{status} {fname:30s} | {mtime_str} | No data (errors: {errors})")
            except:
                pass
PYEOF

echo ""
echo "=========================================="
echo "To watch live: tail -f slurm/logs/zeroshot_batch1_${ZEROSHOT_JOB}.out"
echo "=========================================="

