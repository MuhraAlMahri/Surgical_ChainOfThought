#!/bin/bash
#
# Auto-submit remaining CXRTrek jobs when job slots become available
#

echo "=================================================="
echo "Waiting for job slots to submit CXRTrek jobs..."
echo "=================================================="

# Wait for the tamset job to finish or for job slots to open up
while true; do
    # Count running/pending jobs
    JOB_COUNT=$(squeue -u muhra.almahri | grep -c muhra.al)
    
    echo "Current jobs: $JOB_COUNT (checking every 30 seconds...)"
    
    # If we have fewer than 3 jobs, we can submit more
    if [ $JOB_COUNT -lt 3 ]; then
        echo ""
        echo "Job slots available! Submitting CXRTrek jobs..."
        
        cd /l/users/muhra.almahri/Surgical_COT/experiments/cxrtrek_curriculum_learning
        
        sbatch slurm/retrain_cxrtrek_proper_stage1.slurm
        sleep 1
        sbatch slurm/retrain_cxrtrek_proper_stage2.slurm
        sleep 1
        sbatch slurm/retrain_cxrtrek_proper_stage3.slurm
        
        echo ""
        echo "All CXRTrek jobs submitted!"
        echo ""
        squeue -u muhra.almahri
        break
    fi
    
    sleep 30
done

echo ""
echo "=================================================="
echo "All training jobs are now submitted!"
echo "=================================================="
echo ""
echo "Current job status:"
squeue -u muhra.almahri -o "%.10i %.12P %.30j %.8u %.2t %.10M %.6D %20R"

echo ""
echo "Monitor progress with: squeue -u muhra.almahri"

