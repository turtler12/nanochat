"""
Measure forward/backward/optimizer/zero_grad component times from actual training code.
Runs a short training loop (warmup + measurement) using the real training setup.

Usage (2 GPU):
  torchrun --standalone --nproc_per_node=2 -m scripts.time_components -- --method muon --depth 12
  torchrun --standalone --nproc_per_node=2 -m scripts.time_components -- --method taylor --depth 12 --eta 0.1
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import json
import time
import argparse

import torch
import numpy as np

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, print_banner
from nanochat.optim import polar_express_coeffs
from nanochat.tokenizer import get_tokenizer, get_token_bytes

print_banner()

parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, required=True, choices=["muon", "taylor"])
parser.add_argument("--depth", type=int, default=12)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--fp8", action="store_true")
parser.add_argument("--eta", type=float, default=0.1)
parser.add_argument("--warmup-steps", type=int, default=50, help="warmup steps before measurement")
parser.add_argument("--measure-steps", type=int, default=150, help="steps to measure")
parser.add_argument("--ns-mode", type=str, default="baseline", choices=["baseline", "fast4"])
args = parser.parse_args()

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
synchronize = torch.cuda.synchronize

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()

# Build model (same as training scripts)
base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
config = GPTConfig(
    sequence_len=args.max_seq_len, vocab_size=vocab_size,
    n_layer=args.depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

# FP8
if args.fp8 and device_type == "cuda":
    from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
    import torch.nn as nn
    def fp8_module_filter(mod, fqn):
        return isinstance(mod, nn.Linear) and mod.in_features % 16 == 0 and mod.out_features % 16 == 0
    fp8_config = Float8LinearConfig.from_recipe_name("tensorwise")
    convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)

orig_model = model
model = torch.compile(model, dynamic=False)

# Optimizer (same setup as training scripts)
_ns_steps = 5 if args.ns_mode == "baseline" else 4
B_REF = 2**19
total_batch_size = B_REF
batch_lr_scale = 1.0
weight_decay_scaled = 0.2

if args.method == "muon":
    optimizer = orig_model.setup_optimizer(
        unembedding_lr=0.004 * batch_lr_scale,
        embedding_lr=0.3 * batch_lr_scale,
        scalar_lr=0.5 * batch_lr_scale,
        adam_betas=(0.8, 0.95),
        matrix_lr=0.02 * batch_lr_scale,
        weight_decay=weight_decay_scaled,
        ns_steps=_ns_steps,
    )
    print0(f"Method: Muon (baseline)")
elif args.method == "taylor":
    from nanochat.optim_taylor import DistTaylorMuonAdamW, TaylorMuonAdamW

    is_ddp = ddp_world_size > 1
    matrix_params = list(orig_model.transformer.h.parameters())
    embedding_params = list(orig_model.transformer.wte.parameters())
    value_embeds_params = list(orig_model.value_embeds.parameters())
    lm_head_params = list(orig_model.lm_head.parameters())
    resid_params = [orig_model.resid_lambdas]
    x0_params = [orig_model.x0_lambdas]

    param_groups = [
        dict(kind='adamw', params=lm_head_params,
             lr=0.004 * batch_lr_scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=embedding_params,
             lr=0.3 * batch_lr_scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=value_embeds_params,
             lr=0.3 * batch_lr_scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=resid_params,
             lr=0.5 * batch_lr_scale * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=x0_params,
             lr=0.5 * batch_lr_scale, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(
            kind='muon', params=group_params, lr=0.02 * batch_lr_scale,
            momentum=0.95, ns_steps=_ns_steps, beta2=0.95, weight_decay=weight_decay_scaled,
        ))

    Factory = DistTaylorMuonAdamW if is_ddp else TaylorMuonAdamW
    optimizer = Factory(param_groups, eta=args.eta)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    print0(f"Method: TaylorMuon (eta={args.eta})")

# Dataloader
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=None)

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

x, y, _ = next(train_loader)

print0(f"GPUs: {ddp_world_size} | grad_accum_steps: {grad_accum_steps} | depth: {args.depth}")
print0(f"Warmup: {args.warmup_steps} steps | Measure: {args.measure_steps} steps")

# Timing storage
fwd_times = []
bwd_times = []
opt_times = []
zg_times = []
total_times = []

total_steps = args.warmup_steps + args.measure_steps

for step in range(total_steps):
    measuring = step >= args.warmup_steps

    # === Exactly mirroring the training loop ===
    synchronize()
    t_total_start = time.time()

    # Forward + backward (all micro-steps)
    synchronize()
    t_fwd_start = time.time()

    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps

        synchronize()
        if micro_step == 0:
            t_fwd_end = time.time()

        loss.backward()
        x, y, _ = next(train_loader)

    synchronize()
    t_bwd_end = time.time()

    # LR / momentum schedule (mimicking the training loop)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"]

    # Optimizer step
    synchronize()
    t_opt_start = time.time()
    optimizer.step()
    synchronize()
    t_opt_end = time.time()

    # Zero grad
    model.zero_grad(set_to_none=True)
    synchronize()
    t_zg_end = time.time()

    t_total_end = t_zg_end

    if measuring:
        fwd_times.append(t_fwd_end - t_fwd_start)
        bwd_times.append(t_bwd_end - t_fwd_end)
        opt_times.append(t_opt_end - t_opt_start)
        zg_times.append(t_zg_end - t_opt_end)
        total_times.append(t_total_end - t_total_start)

    if master_process and step % 10 == 0:
        dt = t_total_end - t_total_start
        print(f"  step {step:3d}/{total_steps} | dt: {dt*1000:.1f}ms | loss: {train_loss.item():.4f}"
              + (" [measuring]" if measuring else " [warmup]"))

# Report
if master_process:
    def trimmed_mean(arr):
        arr = sorted(arr)
        trim = max(1, len(arr) // 10)
        trimmed = arr[trim:-trim] if trim < len(arr) // 2 else arr
        return sum(trimmed) / len(trimmed)

    fwd_ms = trimmed_mean(fwd_times) * 1000
    bwd_ms = trimmed_mean(bwd_times) * 1000
    opt_ms = trimmed_mean(opt_times) * 1000
    zg_ms = trimmed_mean(zg_times) * 1000
    total_ms = trimmed_mean(total_times) * 1000

    method_label = args.method.upper()
    if args.method == "taylor":
        method_label += f" eta={args.eta}"

    print(f"\n{'='*60}")
    print(f"  COMPONENT TIMING: {method_label} | depth={args.depth} | {ddp_world_size} GPUs")
    print(f"  {args.measure_steps} measurement steps (trimmed mean)")
    print(f"{'='*60}")
    print(f"  {'Component':<30s} {'Mean (ms)':>10s} {'% of step':>10s}")
    print(f"  {'-'*50}")
    print(f"  {'Forward (1 micro-batch)':<30s} {fwd_ms:>9.1f}ms {fwd_ms/total_ms*100:>9.1f}%")
    print(f"  {'Backward (all accum)':<30s} {bwd_ms:>9.1f}ms {bwd_ms/total_ms*100:>9.1f}%")
    print(f"  {'Optimizer (incl. comm)':<30s} {opt_ms:>9.1f}ms {opt_ms/total_ms*100:>9.1f}%")
    print(f"  {'Zero grad':<30s} {zg_ms:>9.1f}ms {zg_ms/total_ms*100:>9.1f}%")
    print(f"  {'-'*50}")
    print(f"  {'Total step':<30s} {total_ms:>9.1f}ms {100.0:>9.1f}%")
    print(f"{'='*60}")

    result = {
        "method": args.method,
        "depth": args.depth,
        "gpus": ddp_world_size,
        "eta": args.eta if args.method == "taylor" else None,
        "fwd_ms": round(fwd_ms, 1),
        "bwd_ms": round(bwd_ms, 1),
        "opt_ms": round(opt_ms, 1),
        "zg_ms": round(zg_ms, 1),
        "total_ms": round(total_ms, 1),
        "measure_steps": args.measure_steps,
    }
    print(f"\nJSON: {json.dumps(result)}")

compute_cleanup()
