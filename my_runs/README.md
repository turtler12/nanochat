# My Runs Directory

This directory contains outputs from your nanochat training runs.

## Directory Structure

Each run creates a timestamped subdirectory:
```
my_runs/
├── run_20260212_133045/    # Timestamped run
│   ├── run.log             # Complete training log
│   ├── report.md           # Final training report
│   └── checkpoints/        # Model checkpoints (if saved)
└── README.md               # This file
```

## Running Training

### Option 1: Use the Modified Script (Recommended)
The `speedrun_myrun.sh` script outputs everything to this directory:

```bash
# Basic run
bash runs/speedrun_myrun.sh

# With custom run name
WANDB_RUN=my_experiment bash runs/speedrun_myrun.sh

# In screen session (recommended for long runs)
screen -S training bash runs/speedrun_myrun.sh
```

### Option 2: Monitor Real-Time Progress

While training is running, you can monitor progress in real-time:

```bash
# Watch the log file
tail -f my_runs/run_*/run.log

# Or find the latest run
tail -f $(ls -t my_runs/*/run.log | head -1)
```

## What Gets Cached vs Re-downloaded

### Cached Once (Reused Across Runs) - in `~/.cache/nanochat/`:
- ✅ **Data shards** (~370 parquet files, ~37GB total) - Downloaded once, reused forever
- ✅ **Tokenizer** (tok_32768.model) - Trained once, reused forever
- ✅ **Identity conversations** (2.3MB) - Downloaded once, reused forever

### Generated Per Run:
- ❌ **Model checkpoints** - New for each training run
- ❌ **Training metrics** - New for each run
- ❌ **Reports** - New for each run

**TL;DR**: After the first run, you only need ~10 seconds to check cache before training starts!

## Estimated Timeline

**First Run** (4 GPUs):
- Data download: ~15-30 min (one-time)
- Tokenizer training: ~5 min (one-time)
- Base model training: ~5 hours
- SFT: ~30 min
- **Total: ~6 hours**

**Subsequent Runs** (4 GPUs):
- Cache check: ~10 sec
- Base model training: ~5 hours
- SFT: ~30 min
- **Total: ~5.5 hours**

## Quick Commands

```bash
# List all runs
ls -lt my_runs/

# View latest report
cat $(ls -t my_runs/*/report.md | head -1)

# Compare multiple runs
grep "CORE" my_runs/*/report.md

# Clean up old runs (be careful!)
rm -rf my_runs/run_20260212_*
```
