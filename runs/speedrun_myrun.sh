#!/bin/bash

# Modified speedrun script that outputs to my_runs/ directory
# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a 4 GPU node and takes approximately 6 hours to complete.

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create timestamped run directory
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${WANDB_RUN:-run_$RUN_TIMESTAMP}"
RUN_DIR="$PROJECT_ROOT/my_runs/$RUN_NAME"
mkdir -p "$RUN_DIR"

# Create log file
LOG_FILE="$RUN_DIR/run.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "==================================================="
log "Starting nanochat speedrun"
log "Run name: $RUN_NAME"
log "Output directory: $RUN_DIR"
log "==================================================="

# Default intermediate artifacts directory - use /data/scratch to avoid AFS quota
# This is where data shards, tokenizer, and checkpoints are cached
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$PROJECT_ROOT/cache"
mkdir -p $NANOCHAT_BASE_DIR

log "Cache directory: $NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv setup with uv
# Set UV cache to local directory to avoid AFS quota issues
export UV_CACHE_DIR="$PROJECT_ROOT/.uv_cache"
export TMPDIR="/tmp"

log "Setting up Python virtual environment..."
# install uv (if not already installed)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
# create a .venv local virtual environment (if it doesn't exist)
[ -d "$PROJECT_ROOT/.venv" ] || (cd "$PROJECT_ROOT" && uv venv)
# install the repo dependencies
log "Installing dependencies (this may take a few minutes)..."
(cd "$PROJECT_ROOT" && uv sync --extra gpu) 2>&1 | tee -a "$LOG_FILE"
# activate venv so that `python` uses the project's venv instead of system python
source "$PROJECT_ROOT/.venv/bin/activate"

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi
log "WandB run name: $WANDB_RUN"

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
log "Initializing report..."
python -m nanochat.report reset 2>&1 | tee -a "$LOG_FILE"

# -----------------------------------------------------------------------------
# Tokenizer

log "==================================================="
log "Stage 1: Tokenizer Training"
log "==================================================="

# Check if we already have enough data shards
EXISTING_SHARDS=$(ls -1 $NANOCHAT_BASE_DIR/base_data/*.parquet 2>/dev/null | wc -l)
log "Found $EXISTING_SHARDS existing data shards in cache"

if [ $EXISTING_SHARDS -lt 8 ]; then
    log "Downloading initial 8 data shards for tokenizer training..."
    python -m nanochat.dataset -n 8 2>&1 | tee -a "$LOG_FILE"
else
    log "Skipping initial data download (already cached)"
fi

# Download more shards in background if needed
if [ $EXISTING_SHARDS -lt 370 ]; then
    log "Starting background download of remaining data shards (up to 370 total)..."
    python -m nanochat.dataset -n 370 &
    DATASET_DOWNLOAD_PID=$!
else
    log "All required data shards already cached, skipping background download"
    DATASET_DOWNLOAD_PID=""
fi

# Check if tokenizer already exists
if [ -f "$NANOCHAT_BASE_DIR/tok_32768.model" ]; then
    log "Tokenizer already exists, skipping training"
else
    log "Training tokenizer..."
    python -m scripts.tok_train 2>&1 | tee -a "$LOG_FILE"
fi

# Evaluate tokenizer
log "Evaluating tokenizer..."
python -m scripts.tok_eval 2>&1 | tee -a "$LOG_FILE"

# -----------------------------------------------------------------------------
# Base model (pretraining)

log "==================================================="
log "Stage 2: Base Model Pretraining"
log "==================================================="

if [ ! -z "$DATASET_DOWNLOAD_PID" ]; then
    log "Waiting for dataset download to complete..."
    wait $DATASET_DOWNLOAD_PID
fi

log "Starting base model training (depth=26, 4 GPUs)..."
log "This will take approximately 5 hours..."
torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=26 \
    --target-param-data-ratio=8.25 \
    --device-batch-size=16 \
    --fp8 \
    --run=$WANDB_RUN 2>&1 | tee -a "$LOG_FILE"

log "Evaluating base model..."
torchrun --standalone --nproc_per_node=4 -m scripts.base_eval -- \
    --device-batch-size=16 2>&1 | tee -a "$LOG_FILE"

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

log "==================================================="
log "Stage 3: Supervised Fine-Tuning (SFT)"
log "==================================================="

# Download identity conversations if not already cached
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    log "Downloading identity conversations..."
    curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
        https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl 2>&1 | tee -a "$LOG_FILE"
else
    log "Identity conversations already cached"
fi

log "Running SFT training..."
torchrun --standalone --nproc_per_node=4 -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --run=$WANDB_RUN 2>&1 | tee -a "$LOG_FILE"

log "Evaluating chat model..."
torchrun --standalone --nproc_per_node=4 -m scripts.chat_eval -- \
    -i sft 2>&1 | tee -a "$LOG_FILE"

# -----------------------------------------------------------------------------
# Generate the full report

log "==================================================="
log "Stage 4: Report Generation"
log "==================================================="

log "Generating final report..."
python -m nanochat.report generate 2>&1 | tee -a "$LOG_FILE"

# Copy report to run directory
if [ -f "report.md" ]; then
    cp report.md "$RUN_DIR/report.md"
    log "Report saved to: $RUN_DIR/report.md"
fi

# Copy checkpoints to run directory (if they exist)
if [ -d "$NANOCHAT_BASE_DIR/checkpoints" ]; then
    log "Copying checkpoints to run directory..."
    cp -r "$NANOCHAT_BASE_DIR/checkpoints" "$RUN_DIR/"
fi

log "==================================================="
log "Training complete!"
log "Output directory: $RUN_DIR"
log "==================================================="
log ""
log "To chat with your model:"
log "  python -m scripts.chat_web"
log ""
log "Or use CLI:"
log "  python -m scripts.chat_cli -p 'Why is the sky blue?'"
