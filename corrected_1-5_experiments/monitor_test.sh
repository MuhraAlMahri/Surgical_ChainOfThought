#!/bin/bash
echo "=== Test Job Status ==="
squeue -j 151628 -o "%.10i %.9P %.30j %.8u %.2t %.10M %.6D %R"
echo ""
echo "=== Last 20 lines of output ==="
tail -20 logs/test_training_fix_151628.out
echo ""
echo "=== Checking for loss values ==="
grep -i "loss" logs/test_training_fix_151628.out | tail -10
