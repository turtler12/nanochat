# Quick Start - Training GPT-2 on 4 H100s

## You're on a SLURM cluster! Use this command:

```bash
# Submit the training job
sbatch runs/submit_speedrun.sh
```

## That's it! Now monitor progress:

```bash
# Check job status
squeue -u $USER

# Watch SLURM log
tail -f slurm_logs/job_*.log

# Or watch training log (once started)
tail -f my_runs/run_*/run.log
```

## Job will:
- ✅ Request 4 H100 GPUs
- ✅ Train GPT-2 for ~6 hours
- ✅ Save results to `my_runs/run_<timestamp>/`
- ✅ Use cached data/tokenizer (already downloaded!)

## After training completes:

```bash
# View report
cat $(ls -t my_runs/*/report.md | head -1)

# Chat with model (request GPU for inference)
srun --gpus=1 --mem=32G --time=1:00:00 --pty bash
source .venv/bin/activate
python -m scripts.chat_web
```

## Cancel job if needed:

```bash
scancel <JOBID>
```

See [SLURM_GUIDE.md](SLURM_GUIDE.md) for detailed instructions.
