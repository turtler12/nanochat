# nanochat Training Guide - 4 GPU Setup

## Quick Start

```bash
# Start training with output to my_runs/
bash runs/speedrun_myrun.sh

# Or in screen session (recommended)
screen -S nanochat bash runs/speedrun_myrun.sh

# Monitor progress in real-time
tail -f my_runs/*/run.log
```

## What This Repo Does

**nanochat** trains a GPT-2 capability LLM from scratch in ~6 hours on 4 GPUs.

### Current Optimizer: **Combined MuonAdamW**

The training uses a sophisticated hybrid optimizer:

#### Muon (for 2D weight matrices)
- **Full name**: MomentUm Orthogonalized by Newton-schulz
- **Used for**: All transformer weight matrices (Q/K/V/projection, MLP weights)
- **How it works**:
  1. Standard momentum update
  2. Orthogonalization via Polar Express iteration (5 steps)
  3. Per-neuron variance reduction (NorMuon)
  4. Cautious weight decay (only when update aligns with weights)
- **Settings**:
  - Learning rate: 0.02 × sqrt(max(1.0, rows/cols))
  - Momentum: 0.95 (Nesterov)
  - Weight decay: 0.2 (cautious)

#### AdamW (for embeddings & scalars)
- **Used for**: Embeddings, unembedding layer, scalar parameters
- **Settings**:
  - Unembedding LR: 0.004 × sqrt(768/n_embd)
  - Embedding LR: 0.2 × sqrt(768/n_embd)
  - Betas: (0.8, 0.95)

## Data Caching - You Only Download Once!

### The Good News 🎉
After your first run, almost everything is cached!

### What's Cached in `~/.cache/nanochat/`:

| Item | Size | Downloaded | Purpose |
|------|------|-----------|---------|
| **Data shards** | ~37GB | Once | Pretraining data (370 files) |
| **Tokenizer** | ~1MB | Once | BPE tokenizer with 32K vocab |
| **Identity data** | 2.3MB | Once | SFT conversation data |

**Result**: Subsequent runs skip ~30 minutes of downloads!

### What's Generated Per Run:
- Model checkpoints (~1-2GB)
- Training logs
- Evaluation reports

## Modified for 4 GPUs

The original script was for 8 GPUs. I've modified it to use 4 GPUs:

**Changes made to `speedrun.sh`:**
- Line 73: `--nproc_per_node=4` (base training)
- Line 75: `--nproc_per_node=4` (base eval)
- Line 85: `--nproc_per_node=4` (SFT training)
- Line 86: `--nproc_per_node=4` (SFT eval)

**Impact:**
- ✅ Training time: ~6 hours (vs 3 hours on 8 GPUs)
- ✅ Results: Identical (gradient accumulation maintains batch size)
- ✅ VRAM per GPU: Same as 8 GPU setup
- ✅ Total cost: Still <$100

## Timeline Breakdown (4 GPUs)

### First Run:
```
Environment setup:     ~2 min    (pip install)
Data download:        ~20 min    (one-time, 370 shards)
Tokenizer training:    ~5 min    (one-time)
─────────────────────────────────────────────
Base model training:   ~5 hrs    (depth-26 GPT)
Base model eval:      ~10 min    (CORE metric, samples)
SFT training:         ~25 min    (chat fine-tuning)
SFT eval:             ~5 min     (eval tasks)
Report generation:    ~1 min
─────────────────────────────────────────────
TOTAL:                ~6 hours
```

### Subsequent Runs:
```
Cache verification:   ~10 sec    ✅ Data already downloaded!
Base model training:   ~5 hrs
SFT:                  ~30 min
─────────────────────────────────────────────
TOTAL:                ~5.5 hours
```

## Output Structure

All runs output to `my_runs/` directory:

```
my_runs/
├── run_20260212_140000/
│   ├── run.log          # Complete training log with timestamps
│   ├── report.md        # Final evaluation report (CORE score, etc.)
│   └── checkpoints/     # Model checkpoints (optional)
└── my_experiment_v2/    # Custom name via WANDB_RUN=my_experiment_v2
    └── ...
```

## Monitoring Training

### Watch logs in real-time:
```bash
tail -f my_runs/*/run.log
```

### Check GPU utilization:
```bash
watch -n 1 nvidia-smi
```

### Key metrics to watch in logs:
- `val_bpb`: Validation loss (lower is better)
- `train/mfu`: Model FLOPS utilization (higher = more efficient)
- `core_metric`: DCLM CORE score (target: >0.2565 to beat GPT-2)
- `train/tok_per_sec`: Training throughput

## After Training

### Chat with your model (Web UI):
```bash
source .venv/bin/activate
python -m scripts.chat_web
# Visit http://<your-ip>:8000/
```

### Chat via CLI:
```bash
python -m scripts.chat_cli -p "Why is the sky blue?"
```

### Interactive CLI:
```bash
python -m scripts.chat_cli
# Type your questions, Ctrl+C to exit
```

## Running with WandB Logging (Recommended)

WandB provides beautiful real-time training charts:

```bash
# First time: login to wandb
wandb login

# Run with wandb logging
WANDB_RUN=my_gpt2_run bash runs/speedrun_myrun.sh
```

## Troubleshooting

### Out of Memory (OOM):
Reduce batch size in the commands:
```bash
# Edit runs/speedrun_myrun.sh
# Change --device-batch-size=16 to --device-batch-size=8 (or 4, 2)
```

### Training is slow:
Check GPU utilization with `nvidia-smi`. You should see:
- All 4 GPUs at ~95-100% utilization
- Memory usage ~70-80GB per GPU

### Want to use fewer GPUs:
Change `--nproc_per_node=4` to 2 or 1 in the script.

## Cost Estimate (4 GPUs)

At typical cloud pricing:
- 4x H100 GPUs: ~$12-15/hour
- Training time: ~6 hours first run, ~5.5 hours subsequent
- **Total cost**: ~$72-90 per run

On spot instances:
- ~$6-8/hour
- **Total cost**: ~$36-48 per run

## What You're Training

- **Model**: GPT-2 scale transformer (depth-26, ~1.6B parameters)
- **Training data**: ~10B tokens from FineWeb-Edu
- **Tokenizer**: 32K BPE vocab
- **Context length**: 1024 tokens
- **Target**: Beat GPT-2's CORE score of 0.2565

## File Locations

```
/data/scratch/medhaven/nanochat/
├── runs/
│   ├── speedrun.sh          # Original 8-GPU script (modified to 4)
│   └── speedrun_myrun.sh    # New script with my_runs output
├── my_runs/                 # Your training outputs
├── nanochat/                # Core library code
├── scripts/                 # Training/eval scripts
└── ~/.cache/nanochat/       # Cached data (outside repo)
```

## Quick Commands Reference

```bash
# Start fresh training run
bash runs/speedrun_myrun.sh

# Start with custom name
WANDB_RUN=experiment_v3 bash runs/speedrun_myrun.sh

# Monitor live
tail -f my_runs/*/run.log

# Check GPU usage
nvidia-smi

# List all runs
ls -lt my_runs/

# View latest report
cat $(ls -t my_runs/*/report.md | head -1)

# Chat with model
python -m scripts.chat_web
```

## Next Steps

1. Start your first training run: `bash runs/speedrun_myrun.sh`
2. Monitor progress: `tail -f my_runs/*/run.log`
3. Wait ~6 hours
4. Chat with your GPT-2: `python -m scripts.chat_web`

Enjoy training your own ChatGPT! 🚀
