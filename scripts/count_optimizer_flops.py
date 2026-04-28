"""
Count and compare optimizer FLOPs for Muon vs NS3+FTRL.

Analytic FLOPs counting for the Newton-Schulz orthogonalization kernel,
plus wall-clock timing to verify the savings are real.

Muon:      5 NS iterations (15 matmuls per layer)
NS3+FTRL:  3 NS iterations (9 matmuls per layer) + cheap elementwise FTRL correction

Run:
  python -m scripts.count_optimizer_flops
  python -m scripts.count_optimizer_flops --depth 20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import time
import statistics
import math

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, print_banner
from nanochat.optim import polar_express_coeffs
from nanochat.optim_ablation import AblationMuonAdamW
from nanochat.tokenizer import get_tokenizer

print_banner()

parser = argparse.ArgumentParser(description="Count optimizer FLOPs: Muon vs NS3+FTRL")
parser.add_argument("--depth", type=int, default=12)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--total-batch-size", type=int, default=524288)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--weight-decay", type=float, default=0.2)
parser.add_argument("--warmup-steps", type=int, default=15, help="steps to skip (includes compile warmup)")
parser.add_argument("--measure-steps", type=int, default=40, help="steps to measure")
parser.add_argument("--ftrl-eta", type=str, default="0p3", help="FTRL eta key, e.g. 0p3 for eta=0.3")
args = parser.parse_args()

# -----------------------------------------------------------------------
# Setup
device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
assert ddp_world_size == 1, "FLOPs counting script is single-GPU only (multi-GPU adds comm overhead)"
master_process = True
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else __import__("contextlib").nullcontext()

# -----------------------------------------------------------------------
# Model
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim

config = GPTConfig(
    sequence_len=args.max_seq_len, vocab_size=vocab_size,
    n_layer=args.depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
    window_pattern="SSSL",
)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
model = torch.compile(model, dynamic=False)

# -----------------------------------------------------------------------
# Analytic FLOPs counting for the NS orthogonalization kernel
#
# Per NS iteration on a (m, n) weight matrix (m >= n, tall):
#   A   = X.mT @ X          : (n,m) x (m,n)  -> (n,n)   => 2*m*n^2 FLOPs
#   A@A : (n,n) x (n,n)     -> (n,n)          => 2*n^3 FLOPs
#   X@B : (m,n) x (n,n)     -> (m,n)          => 2*m*n^2 FLOPs
#   Total per iter = 4*m*n^2 + 2*n^3
#
# For wide matrices (m < n), swap m and n in the formula.
#
# FTRL correction (after NS): only elementwise ops + one dot product per layer.
# The dot product ⟨Q, G⟩ is O(m*n) — negligible vs O(m*n^2) matmuls.

def ns_flops_per_iter(m: int, n: int) -> int:
    """FLOPs for one NS iteration on an (m, n) matrix."""
    r = min(m, n)
    s = max(m, n)
    # s >= r always
    # A = X.mT @ X  (or X @ X.mT for wide): 2*s*r^2
    # A @ A:                                 2*r^3
    # X @ B (or B @ X):                      2*s*r^2
    return 4 * s * r * r + 2 * r * r * r


def ftrl_extra_flops(m: int, n: int) -> int:
    """Extra FLOPs for FTRL correction on top of NS output (elementwise only, no matmuls)."""
    # dot product ⟨Q, G⟩: 2*m*n
    # scale and blend:     ~4*m*n
    return 6 * m * n


def count_matrix_params(model_orig):
    """Return list of (m, n) shapes for all matrix (Muon) params in the model."""
    shapes = []
    for name, p in model_orig.named_parameters():
        if p.ndim == 2 and min(p.shape) >= 2:
            # Match the grouping in setup_optimizer: transformer.h params are Muon
            if "transformer.h" in name:
                shapes.append(tuple(p.shape))
    return shapes


# Get the original (uncompiled) model for parameter inspection
orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model

matrix_shapes = count_matrix_params(orig_model)
print0(f"\nModel: depth={args.depth}, model_dim={model_dim}")
print0(f"Matrix parameter shapes (Muon params): {len(matrix_shapes)} tensors")

# Group by shape for summary
from collections import Counter
shape_counts = Counter(matrix_shapes)
print0("  Shape -> count:")
for shape, count in sorted(shape_counts.items()):
    print0(f"    {shape[0]:5d} x {shape[1]:5d}  x{count}")

# -----------------------------------------------------------------------
# Compute analytic FLOPs

muon_ns_steps = 5
ns3_ftrl_ns_steps = 3

total_muon_ns_flops = 0
total_ns3_ftrl_ns_flops = 0
total_ftrl_extra_flops = 0

for (m, n) in matrix_shapes:
    per_iter = ns_flops_per_iter(m, n)
    total_muon_ns_flops += muon_ns_steps * per_iter
    total_ns3_ftrl_ns_flops += ns3_ftrl_ns_steps * per_iter
    total_ftrl_extra_flops += ftrl_extra_flops(m, n)

total_muon_flops = total_muon_ns_flops
total_ns3_ftrl_flops = total_ns3_ftrl_ns_flops + total_ftrl_extra_flops

print0("\n" + "=" * 65)
print0("ANALYTIC FLOPS COMPARISON (optimizer step only, NS kernel)")
print0("=" * 65)
print0(f"{'Component':<45s} {'GFLOPs':>10s}")
print0("-" * 65)
print0(f"  Muon (5x NS iters)                          {total_muon_flops/1e9:>10.2f}")
print0(f"  NS3+FTRL: 3x NS iters                       {total_ns3_ftrl_ns_flops/1e9:>10.2f}")
print0(f"  NS3+FTRL: FTRL elementwise correction       {total_ftrl_extra_flops/1e9:>10.2f}")
print0(f"  NS3+FTRL total                              {total_ns3_ftrl_flops/1e9:>10.2f}")
print0("-" * 65)
savings = total_muon_flops - total_ns3_ftrl_flops
savings_pct = 100 * savings / total_muon_flops
print0(f"  Savings (NS FLOPs):                         {savings/1e9:>10.2f} ({savings_pct:.1f}%)")
print0(f"  NS3+FTRL is {total_muon_flops / total_ns3_ftrl_flops:.2f}x fewer NS FLOPs than Muon")
print0("")

# Per-iteration breakdown
print0("Per NS iteration FLOPs by shape:")
for shape, count in sorted(shape_counts.items()):
    m, n = shape
    per_iter = ns_flops_per_iter(m, n)
    muon_total = muon_ns_steps * per_iter * count
    ns3_total  = ns3_ftrl_ns_steps * per_iter * count
    print0(f"  {m:5d}x{n:<5d} x{count}: {per_iter/1e6:6.1f} MFLOPs/iter | "
           f"Muon={muon_total/1e9:.3f}G, NS3={ns3_total/1e9:.3f}G")

# -----------------------------------------------------------------------
# Wall-clock timing experiment
print0("\n" + "=" * 65)
print0("WALL-CLOCK TIMING EXPERIMENT")
print0("=" * 65)

from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
assert args.total_batch_size % tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd

train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device
)
x, y, _ = next(train_loader)

def make_optimizer(mode: str):
    """Build AblationMuonAdamW for a given update mode."""
    with torch.device("meta"):
        m2 = GPT(config)
    m2.to_empty(device=device)
    m2.init_weights()
    m2 = torch.compile(m2, dynamic=False)
    orig = m2._orig_mod if hasattr(m2, "_orig_mod") else m2

    matrix_params = [p for n, p in orig.named_parameters() if "transformer.h" in n and p.ndim == 2]
    embedding_params = list(orig.transformer.wte.parameters())
    lm_head_params = list(orig.lm_head.parameters())

    exclude_ids = {id(p) for p in embedding_params + lm_head_params + matrix_params}
    scalar_params = [p for n, p in orig.named_parameters() if id(p) not in exclude_ids]

    groups_by_shape = {}
    for p in matrix_params:
        s = tuple(p.shape)
        groups_by_shape.setdefault(s, []).append(p)

    param_groups = [
        dict(kind='adamw', params=lm_head_params,    lr=0.004, initial_lr=0.004, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=embedding_params,  lr=0.3,   initial_lr=0.3,   betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=scalar_params,     lr=0.5,   initial_lr=0.5,   betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for shape, params in sorted(groups_by_shape.items()):
        param_groups.append(dict(
            kind='muon', params=params,
            lr=args.matrix_lr, initial_lr=args.matrix_lr,
            momentum=0.95, beta2=0.95,
            weight_decay=args.weight_decay,
            update_mode=mode,
        ))

    opt = AblationMuonAdamW(param_groups)
    return m2, opt


def run_timing(mode: str, model_c, optimizer):
    """Run warmup + measurement steps, return muon step times."""
    total_steps = args.warmup_steps + args.measure_steps
    muon_times = []

    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device
    )
    x_t, y_t, _ = next(loader)

    for step in range(total_steps):
        measuring = step >= args.warmup_steps

        # forward + backward
        for _ in range(grad_accum_steps):
            with autocast_ctx:
                loss = model_c(x_t, y_t)
            (loss / grad_accum_steps).backward()
            x_t, y_t, _ = next(loader)

        # update LR
        for g in optimizer.param_groups:
            g["lr"] = g["initial_lr"]

        # time just the Muon step
        synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for g in optimizer.param_groups:
                if g['kind'] == 'muon':
                    optimizer._step_muon(g)
        synchronize()
        t1 = time.perf_counter()

        # AdamW (don't time, same for both)
        with torch.no_grad():
            for g in optimizer.param_groups:
                if g['kind'] == 'adamw':
                    optimizer._step_adamw(g)

        model_c.zero_grad(set_to_none=True)

        if measuring:
            muon_times.append((t1 - t0) * 1000)

    return muon_times


# Run both experiments
print0(f"\nRunning Muon (ns_steps=5)...")
muon_model, muon_opt = make_optimizer("muon")
muon_times = run_timing("muon", muon_model, muon_opt)

print0(f"Running NS3+FTRL (ns3_ftrl_eta{args.ftrl_eta})...")
ftrl_mode = f"ns3_ftrl_eta{args.ftrl_eta}"
ftrl_model, ftrl_opt = make_optimizer(ftrl_mode)
ftrl_times = run_timing(ftrl_mode, ftrl_model, ftrl_opt)

# -----------------------------------------------------------------------
# Results
print0("\n" + "=" * 65)
print0("RESULTS SUMMARY")
print0("=" * 65)

muon_mean = statistics.mean(muon_times)
muon_std  = statistics.stdev(muon_times) if len(muon_times) > 1 else 0
ftrl_mean = statistics.mean(ftrl_times)
ftrl_std  = statistics.stdev(ftrl_times) if len(ftrl_times) > 1 else 0

print0(f"\nAnalytic FLOPs (optimizer NS kernel only):")
print0(f"  Muon (5x NS):          {total_muon_flops/1e9:>8.2f} GFLOPs")
print0(f"  NS3+FTRL:              {total_ns3_ftrl_flops/1e9:>8.2f} GFLOPs")
print0(f"  Savings:               {savings/1e9:>8.2f} GFLOPs ({savings_pct:.1f}% fewer)")
print0(f"  Speedup (analytic):    {total_muon_flops / total_ns3_ftrl_flops:.2f}x")

print0(f"\nWall-clock Muon step time ({args.measure_steps} steps):")
print0(f"  Muon:                  {muon_mean:>8.2f} ms ± {muon_std:.2f}")
print0(f"  NS3+FTRL ({ftrl_mode}):  {ftrl_mean:>8.2f} ms ± {ftrl_std:.2f}")
wallclock_speedup = muon_mean / ftrl_mean if ftrl_mean > 0 else float('nan')
wallclock_savings_pct = 100 * (muon_mean - ftrl_mean) / muon_mean
print0(f"  Wall-clock speedup:    {wallclock_speedup:.2f}x ({wallclock_savings_pct:.1f}% faster)")

print0(f"\nNote: FTRL elementwise correction adds {total_ftrl_extra_flops/1e6:.1f} MFLOPs "
       f"({100*total_ftrl_extra_flops/total_muon_flops:.2f}% of Muon NS FLOPs) — negligible.")

compute_cleanup()
