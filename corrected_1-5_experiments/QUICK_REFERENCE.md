# 🚀 QUICK REFERENCE - Training Monitoring & Management

## 📊 Check Queue Status
```bash
squeue -u muhra.almahri
```

## 🔍 Monitor Training Progress
```bash
./monitor_and_submit_remaining.sh
```

## 📝 Check Training Logs
```bash
# Check latest logs
tail -50 logs/train_exp1_*.out
tail -50 logs/train_exp2_*.out

# Check for loss values
grep "{'loss':" logs/train_exp1_*.out | tail -20
grep "{'loss':" logs/train_exp2_*.out | tail -20
```

## 🚀 Submit Remaining Jobs (when space available)
```bash
./submit_remaining_jobs.sh
```

## 📂 Important Directories
- **Models**: `models/exp{1,2,3,4}_*/`
- **Logs**: `logs/train_exp*.out`
- **Old Backup**: `models_OLD_UNTRAINED_20251103_170620/`

## ✅ Training Success Indicators
- Loss > 0 (NOT 0.0000)
- Loss decreasing over time
- Example: 16.66 → 15.57 → 14.23 → ...

## ❌ Training Failure Indicators
- Loss = 0.0000 throughout training
- No gradient updates (grad_norm = 0.0)

## 📊 Current Status (Nov 3, 2025)

### ✅ Submitted & In Queue (4/8)
- **Exp 1** (Job 152667) - RUNNING on gpu-08
- **Exp 2** (Job 152668) - RUNNING on gpu-05
- **Exp 3 Stage 1** (Job 152669) - PENDING (waiting for GPU slot)
- **Exp 3 Stage 2** (Job 152670) - PENDING (waiting for GPU slot)

### ⏳ To Submit Later (4/8)
- Exp 3 Stage 3 - Submit after Exp 1/2 complete (~24h)
- Exp 4 Stage 1 - Submit after Exp 1/2 complete (~24h)
- Exp 4 Stage 2 (depends on Stage 1)
- Exp 4 Stage 3 (depends on Stage 2)

**Note**: QOS limits currently allow 4 submitted jobs and 2 running jobs at a time. The pending jobs will start when the running jobs complete.

## 🔄 Next Steps
1. **In 1 hour**: Check logs to verify training is working (loss > 0)
2. **In ~24 hours**: Run `./submit_remaining_jobs.sh` to submit next batch
3. **Monitor regularly**: Use `./monitor_and_submit_remaining.sh`
4. **After all complete**: Re-evaluate all models
5. **Compare results**: Expect different accuracies per experiment

## 📞 Timeline
- **Hour 0 (Now)**: Exp 1 & 2 running, Exp 3 S1/S2 pending
- **Hour 24**: Exp 1/2 done → Exp 3 S1/S2 start → Submit Exp 3 S3 + Exp 4 S1-3
- **Hour 48**: Exp 3 S1/S2 done → Exp 3 S3 + Exp 4 S1 start
- **Hour 72**: Exp 3 S3/Exp 4 S1 done → Exp 4 S2 starts
- **Hour 96 (~4 days)**: All training complete! 🎉

## 🎯 Final Goal
All 8 models trained properly with:
- Non-zero training loss
- Distinct accuracies (not all identical)
- Proper LoRA adapter weights
