#!/bin/bash
# Real-time status dashboard for Nov 5th deadline

cd "/l/users/muhra.almahri/Surgical_COT/corrected 1-5 experiments"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║              📊 NOV 5TH DEADLINE - STATUS DASHBOARD 📊            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Current time and deadline
NOW=$(date '+%Y-%m-%d %H:%M:%S')
DEADLINE="2025-11-05 23:59:59"
DEADLINE_EPOCH=$(date -d "$DEADLINE" +%s 2>/dev/null || echo "0")
NOW_EPOCH=$(date +%s)
TIME_LEFT_SEC=$((DEADLINE_EPOCH - NOW_EPOCH))
TIME_LEFT_HOURS=$((TIME_LEFT_SEC / 3600))

echo "⏰ Current Time: $NOW"
echo "🎯 Deadline:     Nov 5, 2025 23:59"
echo "⏳ Time Left:    ~${TIME_LEFT_HOURS} hours"
echo ""

echo "════════════════════════════════════════════════════════════════════"
echo "📋 JOB QUEUE STATUS"
echo "════════════════════════════════════════════════════════════════════"
echo ""

squeue -u muhra.almahri -o "%.10i %.12j %.8T %.10M %.6D %R" 2>/dev/null || echo "No jobs in queue"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "📈 TRAINING PROGRESS"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Function to check training progress
check_progress() {
    local exp_name=$1
    local log_pattern=$2
    
    LOG=$(ls -t logs/${log_pattern}*.out 2>/dev/null | head -1)
    
    if [ -f "$LOG" ]; then
        # Get first and last loss
        FIRST_LOSS=$(grep "{'loss':" "$LOG" | head -1 | grep -oP "'loss':\s*\K[0-9.]+")
        LAST_LOSS=$(grep "{'loss':" "$LOG" | tail -1 | grep -oP "'loss':\s*\K[0-9.]+")
        LAST_EPOCH=$(grep "{'loss':" "$LOG" | tail -1 | grep -oP "'epoch':\s*\K[0-9.]+")
        
        # Check if completed
        COMPLETED=$(grep -c "completed at:" "$LOG")
        
        if [ "$COMPLETED" -gt 0 ]; then
            echo "  ✅ $exp_name: COMPLETED (Final loss: $LAST_LOSS)"
        elif [ -n "$LAST_LOSS" ]; then
            echo "  🔄 $exp_name: Training (Epoch: $LAST_EPOCH, Loss: $LAST_LOSS)"
        else
            echo "  ⏳ $exp_name: Starting..."
        fi
    else
        echo "  ⬜ $exp_name: Not started"
    fi
}

# Check all experiments
check_progress "Exp1 Training" "exp1_random_1526"
check_progress "Exp2 Training" "exp2_qwen_1526"
check_progress "Exp3 Stage 1" "exp3_s1_1526"
check_progress "Exp3 Stage 2" "exp3_s2_1526"
check_progress "Exp3 Stage 3" "exp3_s3"
check_progress "Exp4 Stage 1" "exp4_s1"
check_progress "Exp4 Stage 2" "exp4_s2"
check_progress "Exp4 Stage 3" "exp4_s3"
check_progress "Exp5 Training" "exp5_train"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "🎯 EVALUATION STATUS"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Check evaluations
check_eval() {
    local exp_name=$1
    local log_pattern=$2
    
    LOG=$(ls -t logs/${log_pattern}*.out 2>/dev/null | head -1)
    
    if [ -f "$LOG" ]; then
        COMPLETED=$(grep -c "Evaluation completed" "$LOG")
        ACCURACY=$(grep -oP "Accuracy:\s*\K[0-9.]+%" "$LOG" | tail -1)
        
        if [ "$COMPLETED" -gt 0 ]; then
            echo "  ✅ $exp_name: DONE (Accuracy: ${ACCURACY:-N/A})"
        else
            echo "  🔄 $exp_name: Running..."
        fi
    else
        echo "  ⬜ $exp_name: Not started"
    fi
}

check_eval "Exp1 Eval" "exp1_eval"
check_eval "Exp2 Eval" "exp2_eval"
check_eval "Exp3 Eval" "exp3_eval"
check_eval "Exp4 Eval" "exp4_eval"
check_eval "Exp5 Eval" "exp5_cot"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "📊 OVERALL PROGRESS"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Count completed jobs
TOTAL_JOBS=14
COMPLETED_TRAINING=$(grep -l "Training completed at:" logs/exp*_*.out 2>/dev/null | wc -l)
COMPLETED_EVAL=$(grep -l "Evaluation completed at:" logs/exp*_*.out 2>/dev/null | wc -l)
COMPLETED_TOTAL=$((COMPLETED_TRAINING + COMPLETED_EVAL))
PROGRESS_PCT=$((COMPLETED_TOTAL * 100 / TOTAL_JOBS))

echo "  Total Jobs:     $TOTAL_JOBS"
echo "  Completed:      $COMPLETED_TOTAL"
echo "  Remaining:      $((TOTAL_JOBS - COMPLETED_TOTAL))"
echo "  Progress:       ${PROGRESS_PCT}%"
echo ""

# Progress bar
FILLED=$((PROGRESS_PCT / 5))
EMPTY=$((20 - FILLED))
printf "  ["
printf "%${FILLED}s" | tr ' ' '█'
printf "%${EMPTY}s" | tr ' ' '░'
printf "] ${PROGRESS_PCT}%%\n"

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "🚦 QOS STATUS"
echo "════════════════════════════════════════════════════════════════════"
echo ""

RUNNING=$(squeue -u muhra.almahri -h -t RUNNING 2>/dev/null | wc -l)
PENDING=$(squeue -u muhra.almahri -h -t PENDING 2>/dev/null | wc -l)

echo "  Running Jobs:   $RUNNING / 4"
echo "  Pending Jobs:   $PENDING"
echo ""

if [ "$RUNNING" -lt 4 ] && [ "$PENDING" -gt 0 ]; then
    echo "  ⚠️  WARNING: QOS limit may still be restricting jobs!"
    echo "      Expected: 4 concurrent jobs"
    echo "      Actual:   $RUNNING running, $PENDING pending"
    echo ""
elif [ "$RUNNING" -eq 4 ]; then
    echo "  ✅ All 4 GPU slots being used efficiently!"
    echo ""
fi

echo "════════════════════════════════════════════════════════════════════"
echo "🎯 NEXT ACTIONS"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Determine what needs to be done next
if [ "$RUNNING" -lt 2 ] && [ "$PENDING" -gt 0 ]; then
    echo "  1. Contact IT about QOS limit (URGENT!)"
fi

# Check if Exp3 S2 is done and S3 not submitted
if grep -q "completed at:" logs/exp3_s2_*.out 2>/dev/null; then
    if ! squeue -u muhra.almahri -h -n exp3_s3 >/dev/null 2>&1; then
        echo "  2. Submit Exp3 Stage 3 (Exp3 S2 completed)"
    fi
fi

# Check if Exp4 not started
if ! squeue -u muhra.almahri -h | grep -q "exp4" 2>/dev/null; then
    if ! ls logs/exp4_*.out >/dev/null 2>&1; then
        echo "  3. Run: ./submit_remaining_when_ready.sh"
    fi
fi

echo ""
echo "Run this dashboard again: ./status_dashboard.sh"
echo ""





