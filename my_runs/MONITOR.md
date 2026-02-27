# Training Run Monitoring Guide

## ✅ Current Status: RUNNING SUCCESSFULLY

**Run Name**: run_20260212_134817
**Started**: 2026-02-12 13:48:17
**Output Directory**: `/data/scratch/medhaven/nanochat/my_runs/run_20260212_134817/`

## Fixed Issues

### 1. ✅ Disk Quota Problem - SOLVED
**Issue**: Original script tried to install packages in AFS home directory which has quota limits
**Solution**: Modified scripts to use local `/data/scratch/` directory for:
- UV cache: `.uv_cache/`
- Virtual environment: `.venv/`
- Training cache: `cache/` (data, tokenizer, checkpoints)

### 2. ✅ Dependencies Installation - SOLVED
**Issue**: Missing packages (psutil, requests, torch, etc.)
**Solution**: Properly installed all dependencies with correct cache location

## Current Progress

```
✅ Environment setup (2 seconds)
✅ Dependencies installed (20 seconds)
✅ Report initialized
✅ Tokenizer trained (122 seconds)
⏳ Tokenizer evaluation (in progress)
⏳ Data download (43/370 shards downloaded, continuing in background)
⏳ Base model training (will start after data download completes)
⏳ SFT training
⏳ Final report generation
```

## Real-Time Monitoring Commands

### Watch the log file in real-time:
```bash
# Main log file with timestamps
tail -f my_runs/run_20260212_134817/run.log

# Or find latest run automatically
tail -f $(ls -t my_runs/*/run.log | head -1)

# Watch with color
tail -f my_runs/run_20260212_134817/run.log | ccze -A
```

### Check current stage:
```bash
grep "===" my_runs/run_20260212_134817/run.log | tail -5
```

### Count downloaded data shards:
```bash
ls -1 cache/base_data/*.parquet 2>/dev/null | wc -l
```

### Check disk usage:
```bash
du -sh cache/
du -sh .venv/
du -sh .uv_cache/
```

### Monitor GPU usage (when on GPU node):
```bash
watch -n 1 nvidia-smi
```

### Check if training process is running:
```bash
ps aux | grep -i "base_train\|chat_sft" | grep -v grep
```

## Training Timeline (4 GPUs)

| Stage | Duration | Status |
|-------|----------|--------|
| Environment setup | ~2 min | ✅ Complete |
| Data download | ~15-20 min | ⏳ In progress (43/370) |
| Tokenizer training | ~2 min | ✅ Complete (122s) |
| Tokenizer eval | ~30 sec | ⏳ Running |
| **Base model training** | **~5 hours** | ⏳ Pending |
| Base model eval | ~10 min | ⏳ Pending |
| SFT training | ~25 min | ⏳ Pending |
| SFT eval | ~5 min | ⏳ Pending |
| Report generation | ~1 min | ⏳ Pending |
| **TOTAL** | **~6 hours** | - |

## Expected Output Files

### During Training:
```
my_runs/run_20260212_134817/
├── run.log                    # Live training log ⏳
└── (checkpoints will be copied at end)

cache/
├── base_data/                 # 370 parquet files (~37GB) ⏳
│   ├── shard_00000.parquet   ✅
│   ├── shard_00001.parquet   ✅
│   └── ...
├── tokenizer/                 # Tokenizer files ✅
│   ├── tokenizer.pkl
│   └── token_bytes.pt
└── report/                    # Training reports ✅
    └── header.md
```

### After Training Completes:
```
my_runs/run_20260212_134817/
├── run.log                    # Complete training log
├── report.md                  # Final evaluation report
└── checkpoints/               # Model checkpoints (optional)
    └── ...
```

## What's Cached (Reused in Future Runs)

✅ **Permanent cache** (in `cache/`):
- 370 data shards (~37GB) - Downloaded once
- Tokenizer model - Trained once
- Identity conversations - Downloaded once

❌ **Per-run** (generated fresh each time):
- Model checkpoints
- Training metrics/logs
- Evaluation reports

## Key Metrics to Watch

When base model training starts, watch for:

1. **val_bpb**: Validation loss in bits per byte
   - Should decrease over time
   - Lower is better

2. **core_metric**: DCLM CORE score
   - Target: >0.2565 (to beat GPT-2)
   - Latest repo record: 0.2602

3. **train/mfu**: Model FLOPS Utilization
   - Should be >80% on H100s
   - Measures GPU efficiency

4. **train/tok_per_sec**: Training throughput
   - Tokens processed per second
   - Higher is better

## Troubleshooting

### If training stops unexpectedly:
```bash
# Check the log for errors
grep -i "error\|exception\|failed" my_runs/run_20260212_134817/run.log

# Check if process is still running
ps aux | grep speedrun_myrun
```

### If out of disk space:
```bash
# Check disk usage
df -h /data/scratch/medhaven/

# Clean up old runs if needed
rm -rf my_runs/run_20260212_133930/  # old failed run
```

### To restart training:
```bash
# The script will detect cached data and skip downloads
bash runs/speedrun_myrun.sh
```

## After Training Completes

### View the final report:
```bash
cat my_runs/run_20260212_134817/report.md
```

### Chat with your model:
```bash
source .venv/bin/activate
python -m scripts.chat_web
# Then visit http://<your-ip>:8000/
```

### Run CLI chat:
```bash
python -m scripts.chat_cli -p "Explain quantum computing"
```

## Background Process Info

**Task ID**: b20666a
**Output file**: `/tmp/claude-29036/-data-scratch-medhaven-nanochat/tasks/b20666a.output`

To check background process:
```bash
tail -f /tmp/claude-29036/-data-scratch-medhaven-nanochat/tasks/b20666a.output
```

---

## Quick Status Check

Run this to get a quick status overview:
```bash
echo "=== TRAINING STATUS ==="
echo "Data shards: $(ls -1 cache/base_data/*.parquet 2>/dev/null | wc -l)/370"
echo "Tokenizer: $([ -f cache/tokenizer/tokenizer.pkl ] && echo '✅' || echo '❌')"
echo ""
echo "=== LATEST LOG ENTRIES ==="
tail -10 my_runs/run_20260212_134817/run.log
```

---

**Estimated Completion Time**: ~7:48 PM (assuming 6-hour training)
**Check back in**: ~5 hours for base model training progress
