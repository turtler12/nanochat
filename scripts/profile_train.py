"""
Profile training step breakdown: forward, backward, optimizer, data loading.
Supports both single-GPU and multi-GPU (DDP) modes.

Single GPU:  python -m scripts.profile_train
Multi GPU:   torchrun --standalone --nproc_per_node=4 -m scripts.profile_train
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import time
import argparse
from collections import defaultdict
from contextlib import nullcontext
import statistics

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, print_banner, autodetect_device_type
from nanochat.optim import polar_express_coeffs
from nanochat.tokenizer import get_tokenizer
from nanochat.flash_attention import HAS_FA3
print_banner()

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Profile training step breakdown")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--depth", type=int, default=12, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
parser.add_argument("--embedding-lr", type=float, default=0.3)
parser.add_argument("--unembedding-lr", type=float, default=0.004)
parser.add_argument("--weight-decay", type=float, default=0.2)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--scalar-lr", type=float, default=0.5)
parser.add_argument("--adam-beta1", type=float, default=0.8)
parser.add_argument("--adam-beta2", type=float, default=0.95)
parser.add_argument("--num-profile-steps", type=int, default=50, help="number of steps to profile")
parser.add_argument("--warmup-steps", type=int, default=15, help="warmup steps to skip (includes torch.compile warmup)")
args = parser.parse_args()

_ns_steps = 5

# -----------------------------------------------------------------------------
# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    print0(f"GPU: {gpu_device_name} x {ddp_world_size}")

# -----------------------------------------------------------------------------
# Model
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
config = GPTConfig(
    sequence_len=args.max_seq_len, vocab_size=vocab_size,
    n_layer=args.depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
    window_pattern=args.window_pattern,
)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"  {key:24s}: {value:,}")

orig_model = model
model = torch.compile(model, dynamic=False)

# -----------------------------------------------------------------------------
# Optimizer (auto-selects MuonAdamW or DistMuonAdamW based on DDP)
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    scalar_lr=args.scalar_lr,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
    ns_steps=_ns_steps,
)

optimizer_type = type(optimizer).__name__
print0(f"Optimizer: {optimizer_type}")
for group in optimizer.param_groups:
    n = sum(p.numel() for p in group['params'])
    print0(f"  Group '{group['kind']}': {len(group['params'])} params, {n:,} elements")

# For single-GPU, we can separately time AdamW vs Muon
can_split_optimizer = not ddp

# -----------------------------------------------------------------------------
# DataLoader
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device
)
x, y, dataloader_state_dict = next(train_loader)

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
total_batch_size = args.total_batch_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Batch size: {total_batch_size:,} | Grad accum steps: {grad_accum_steps} | World size: {ddp_world_size}")

# -----------------------------------------------------------------------------
# Profiling loop

total_steps = args.warmup_steps + args.num_profile_steps
print0(f"\nRunning {args.warmup_steps} warmup + {args.num_profile_steps} profiled steps...")

timings = defaultdict(list)

for step in range(total_steps):
    profiling = step >= args.warmup_steps

    # ---- Forward + Backward (all micro-steps) ----
    step_fwd_time = 0.0
    step_bwd_time = 0.0

    for micro_step in range(grad_accum_steps):
        # Forward
        synchronize()
        t0 = time.perf_counter()
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        synchronize()
        t1 = time.perf_counter()
        step_fwd_time += t1 - t0

        # Backward
        synchronize()
        t2 = time.perf_counter()
        loss.backward()
        synchronize()
        t3 = time.perf_counter()
        step_bwd_time += t3 - t2

        # Prefetch next batch
        x, y, dataloader_state_dict = next(train_loader)

    # ---- Optimizer ----
    frac = min(step / 300, 1)
    muon_momentum = (1 - frac) * 0.85 + frac * 0.95
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"]
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = args.weight_decay

    if can_split_optimizer:
        # Single GPU: time AdamW and Muon separately
        synchronize()
        t_adamw_start = time.perf_counter()
        with torch.no_grad():
            for group in optimizer.param_groups:
                if group['kind'] == 'adamw':
                    optimizer._step_adamw(group)
        synchronize()
        t_adamw_end = time.perf_counter()

        synchronize()
        t_muon_start = time.perf_counter()
        with torch.no_grad():
            for group in optimizer.param_groups:
                if group['kind'] == 'muon':
                    optimizer._step_muon(group)
        synchronize()
        t_muon_end = time.perf_counter()

        dt_adamw = t_adamw_end - t_adamw_start
        dt_muon = t_muon_end - t_muon_start
        dt_opt = dt_adamw + dt_muon
    else:
        # Multi-GPU: time full optimizer.step() (includes comm + compute)
        synchronize()
        t_opt_start = time.perf_counter()
        optimizer.step()
        synchronize()
        t_opt_end = time.perf_counter()
        dt_opt = t_opt_end - t_opt_start
        dt_adamw = None
        dt_muon = None

    # ---- Zero grad ----
    synchronize()
    t_zero_start = time.perf_counter()
    model.zero_grad(set_to_none=True)
    synchronize()
    t_zero_end = time.perf_counter()

    dt_zero = t_zero_end - t_zero_start
    dt_total = step_fwd_time + step_bwd_time + dt_opt + dt_zero

    if profiling:
        timings['forward'].append(step_fwd_time)
        timings['backward'].append(step_bwd_time)
        timings['optimizer_total'].append(dt_opt)
        timings['zero_grad'].append(dt_zero)
        timings['total_step'].append(dt_total)
        if can_split_optimizer:
            timings['adamw'].append(dt_adamw)
            timings['muon'].append(dt_muon)

    loss_val = train_loss.item()
    if step % 10 == 0:
        print0(f"step {step:04d} | loss: {loss_val:.4f} | dt: {dt_total*1000:.1f}ms")

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

# -----------------------------------------------------------------------------
# Report
print0("\n" + "=" * 70)
print0("PROFILING RESULTS")
print0("=" * 70)
print0(f"Model: d{args.depth} ({param_counts['total']:,} params)")
print0(f"Batch size: {total_batch_size:,} tokens | Grad accum: {grad_accum_steps} micro-steps")
print0(f"Newton-Schulz iterations: {_ns_steps}")
print0(f"DDP world size: {ddp_world_size} | Optimizer: {optimizer_type}")
print0(f"Profiled over {args.num_profile_steps} steps (after {args.warmup_steps} warmup steps)")
if device_type == "cuda":
    print0(f"GPU: {gpu_device_name} x {ddp_world_size}")
print0("")

total_mean = statistics.mean([t * 1000 for t in timings['total_step']])

print0(f"{'Component':<30s} {'Mean (ms)':>12s} {'Std (ms)':>10s} {'% of step':>10s}")
print0("-" * 65)

rows = [
    ('forward', 'Forward'),
    ('backward', 'Backward'),
]
if can_split_optimizer:
    rows += [
        ('adamw', 'Optimizer: AdamW'),
        ('muon', 'Optimizer: Muon'),
    ]
rows += [
    ('optimizer_total', 'Optimizer: Total (incl. comm)' if ddp else 'Optimizer: Total'),
    ('zero_grad', 'Zero grad'),
    ('total_step', 'Total step'),
]

for key, label in rows:
    vals_ms = [t * 1000 for t in timings[key]]
    mean = statistics.mean(vals_ms)
    std = statistics.stdev(vals_ms) if len(vals_ms) > 1 else 0
    pct = 100 * mean / total_mean
    print0(f"{label:<30s} {mean:>12.2f} {std:>10.2f} {pct:>9.1f}%")

print0("")
opt_total = statistics.mean(timings['optimizer_total'])
fwd_mean = statistics.mean(timings['forward'])
bwd_mean = statistics.mean(timings['backward'])
step_mean = statistics.mean(timings['total_step'])

print0(f"Forward+Backward as % of step:  {100 * (fwd_mean + bwd_mean) / step_mean:.1f}%")
print0(f"Optimizer as % of total step:    {100 * opt_total / step_mean:.1f}%")
if can_split_optimizer:
    muon_mean = statistics.mean(timings['muon'])
    adamw_mean = statistics.mean(timings['adamw'])
    print0(f"Muon as % of optimizer time:     {100 * muon_mean / opt_total:.1f}%")
    print0(f"AdamW as % of optimizer time:     {100 * adamw_mean / opt_total:.1f}%")
    print0(f"Muon as % of total step:          {100 * muon_mean / step_mean:.1f}%")

compute_cleanup()
