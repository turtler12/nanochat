# Running nanochat on SLURM Cluster with 4 H100s

## Important: You're on a SLURM Cluster!

You're currently on a **login node** (`slurm-login-0.csail.mit.edu`) which has **NO GPUs**.

To train the model, you need to **submit a SLURM job** to get allocated to GPU nodes.

## Previous Runs (Login Node - No GPUs)

The previous runs completed but **skipped model training** because they ran on the login node:

```
my_runs/
├── run_20260212_133930/  # Failed - disk quota issues
├── run_20260212_134811/  # Failed early
└── run_20260212_134817/  # Completed tokenizer only, no GPU training
```

These runs only did:
- ✅ Tokenizer training (CPU only)
- ✅ Data download
- ❌ Model training (skipped - no GPUs on login node)

## How to Submit a Proper SLURM Job

### Step 1: Submit the job

```bash
# Submit the job to get 4 H100 GPUs
sbatch runs/submit_speedrun.sh
```

This will:
- Request 4 H100 GPUs
- Allocate to lingo-h100 partition
- Run for up to 8 hours
- Save logs to `slurm_logs/job_<ID>.log`

### Step 2: Monitor the job

```bash
# Check job status
squeue -u $USER

# Watch the SLURM log
tail -f slurm_logs/job_*.log

# Watch the training log (once job starts)
tail -f my_runs/run_*/run.log
```

### Step 3: Check job details

```bash
# View job info
scontrol show job <JOB_ID>

# Cancel job if needed
scancel <JOB_ID>
```

## SLURM Job Configuration

The submit script (`runs/submit_speedrun.sh`) requests:

| Resource | Value | Reason |
|----------|-------|--------|
| GPUs | 4 | Training uses 4 H100s |
| CPUs | 32 | For data loading parallelism |
| Memory | 256G | Enough for 4 GPUs |
| Time | 8 hours | Training takes ~6 hours |
| Partition | lingo-h100 | H100 GPU partition |
| QoS | lingo-main | Quality of Service |

## Output Files

### SLURM logs (job-level):
```
slurm_logs/
├── job_<JOBID>.log  # stdout
└── job_<JOBID>.err  # stderr
```

### Training logs (training-level):
```
my_runs/run_<timestamp>/
├── run.log           # Detailed training log
└── report.md         # Final report (after completion)
```

## Timeline with SLURM

**Queue time**: Depends on cluster usage (could be immediate or wait)
**Training time**: ~6 hours once job starts

### What happens when job runs:

1. Job gets allocated to GPU node
2. GPUs are initialized
3. Training starts automatically
4. Logs stream to both:
   - `slurm_logs/job_<ID>.log`
   - `my_runs/run_<timestamp>/run.log`

## Monitoring During Training

### Option 1: SLURM log (recommended while waiting)
```bash
tail -f slurm_logs/job_*.log
```

### Option 2: Training log (once training starts)
```bash
# Find the latest run
tail -f $(ls -t my_runs/*/run.log | head -1)
```

### Option 3: SSH to compute node (advanced)
```bash
# Get node name from squeue
squeue -u $USER

# SSH to the node
ssh <node-name>

# Watch GPUs
watch -n 1 nvidia-smi
```

## After Job Completes

### View results:
```bash
# Latest training report
cat $(ls -t my_runs/*/report.md | head -1)

# Check SLURM logs
cat slurm_logs/job_<ID>.log
```

### Chat with your model:
```bash
# Start interactive session on GPU node for inference
srun --gpus=1 --mem=32G --time=1:00:00 --pty bash

# Then in the interactive session:
cd /data/scratch/medhaven/nanochat
source .venv/bin/activate
python -m scripts.chat_web
```

## Common SLURM Commands

```bash
# Submit job
sbatch runs/submit_speedrun.sh

# Check queue
squeue -u $USER

# Check all jobs
squeue -p lingo-h100

# Job details
scontrol show job <JOBID>

# Cancel job
scancel <JOBID>

# View past jobs
sacct -u $USER --format=JobID,JobName,Partition,State,Start,End,Elapsed

# Interactive GPU session (for testing)
srun --account=lingo --partition=lingo-h100 --qos=lingo-main \
     --gpus=4 --cpus-per-task=32 --mem=256G --time=8:00:00 --pty bash
```

## Troubleshooting

### Job pending for a long time?
```bash
# Check why job is pending
squeue -u $USER -o "%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %.20R"
```

Common reasons:
- `Resources`: Waiting for 4 H100s to become available
- `Priority`: Other jobs have higher priority
- `QOSMaxGRESPerUser`: You've hit GPU limits

### Job failed immediately?
```bash
# Check error log
cat slurm_logs/job_<ID>.err

# Check SLURM output
cat slurm_logs/job_<ID>.log
```

### Need to test quickly?
```bash
# Request interactive session with 1 GPU
srun --account=lingo --partition=lingo-h100 --qos=lingo-main \
     --gpus=1 --mem=64G --time=1:00:00 --pty bash

# Then test commands
cd /data/scratch/medhaven/nanochat
source .venv/bin/activate
python -c "import torch; print(torch.cuda.device_count())"
```

## What's Already Done (Cached)

Good news! The login node runs already downloaded and cached:

✅ **Data shards**: 43/370 downloaded (in `cache/base_data/`)
✅ **Tokenizer**: Trained and saved (in `cache/tokenizer/`)

When the SLURM job runs, it will:
- Skip re-downloading cached data
- Skip re-training tokenizer
- Start directly with model training!

## Quick Start (TL;DR)

```bash
# 1. Submit the job
sbatch runs/submit_speedrun.sh

# 2. Monitor
tail -f slurm_logs/job_*.log

# 3. Wait ~6 hours

# 4. Chat with model (on GPU node)
srun --gpus=1 --mem=32G --time=1:00:00 --pty bash -c \
  "cd /data/scratch/medhaven/nanochat && source .venv/bin/activate && python -m scripts.chat_web"
```

## Submit Script Location

**File**: [runs/submit_speedrun.sh](runs/submit_speedrun.sh)

You can edit this file to:
- Change time limit: `--time=XX:XX:XX`
- Change memory: `--mem=XXXG`
- Add email notifications: `--mail-type=END,FAIL --mail-user=your@email.com`
- Change job name: `--job-name=my_custom_name`
