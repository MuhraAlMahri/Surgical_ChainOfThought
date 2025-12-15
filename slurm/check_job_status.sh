#!/bin/bash
# Quick script to check job status and provide updates

echo "=========================================="
echo "Multi-Head Temporal CoT - Job Status"
echo "=========================================="
echo "Date: $(date)"
echo ""

# Current running jobs
echo "📊 CURRENT JOBS:"
echo "----------------------------------------"
squeue -u $USER -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "No jobs running"
echo ""

# Recent job history
echo "📜 RECENT JOB HISTORY (Last 10):"
echo "----------------------------------------"
sacct -u $USER --format=JobID,JobName,State,ExitCode,Start,End -S today 2>/dev/null | tail -10 || echo "No recent jobs"
echo ""

# Check categorization output
echo "📁 CATEGORIZATION STATUS:"
echo "----------------------------------------"
if [ -d "data/categorized" ] && [ "$(ls -A data/categorized 2>/dev/null)" ]; then
    echo "✅ Categorized data exists:"
    ls -lh data/categorized/ | tail -5
else
    echo "❌ No categorized data found"
    echo "   Run: sbatch slurm/01_categorize_questions.slurm"
fi
echo ""

# Check temporal structure (EndoVis)
echo "📁 TEMPORAL STRUCTURE STATUS:"
echo "----------------------------------------"
if [ -f "data/temporal/temporal_structure.json" ]; then
    echo "✅ Temporal structure exists"
    echo "   Size: $(du -h data/temporal/temporal_structure.json | cut -f1)"
else
    echo "⚠️  No temporal structure found (only needed for EndoVis)"
fi
echo ""

# Check checkpoints
echo "📁 CHECKPOINTS STATUS:"
echo "----------------------------------------"
if [ -d "checkpoints" ] && [ "$(ls -A checkpoints 2>/dev/null)" ]; then
    echo "✅ Checkpoints found:"
    ls -lht checkpoints/ | head -5
else
    echo "❌ No checkpoints found"
fi
echo ""

# Check recent logs
echo "📋 RECENT LOGS:"
echo "----------------------------------------"
if [ -d "slurm/logs" ]; then
    echo "Latest output logs:"
    ls -lht slurm/logs/*.out 2>/dev/null | head -3 | awk '{print $9, "(" $5 ")"}'
    echo ""
    echo "Latest error logs:"
    ls -lht slurm/logs/*.err 2>/dev/null | head -3 | awk '{print $9, "(" $5 ")"}'
else
    echo "No logs directory found"
fi
echo ""

# Check for errors in recent logs
echo "⚠️  RECENT ERRORS (if any):"
echo "----------------------------------------"
for err_file in $(ls -t slurm/logs/*.err 2>/dev/null | head -3); do
    if [ -s "$err_file" ]; then
        echo "File: $err_file"
        tail -5 "$err_file" | grep -i "error\|fail\|exception" | head -3 || echo "  (no obvious errors)"
        echo ""
    fi
done

echo "=========================================="
echo "Quick Commands:"
echo "  View job output: tail -f slurm/logs/JOBNAME_JOBID.out"
echo "  Cancel job: scancel JOBID"
echo "  Submit pipeline: sbatch slurm/submit_all.sh"
echo "=========================================="














