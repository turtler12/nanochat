"""
Trace/norm-constrained Muon training with spectral logging.

Supports eight modes:
  --norm-mode baseline       : Standard Muon (no constraint)
  --norm-mode fro            : Variant A - Frobenius normalization every step
  --norm-mode nuclear        : Variant B - Nuclear norm every K steps, Frobenius between
  --norm-mode fixed-fro      : Variant C - Fixed universal Frobenius target (median of init)
  --norm-mode cond-clamp     : Option 1 - Condition number clamping via SVD every K steps
  --norm-mode spectral-cap   : Option 2 - Spectral normalization only (cap sigma_max)
  --norm-mode growing-fro    : Option 3 - Growing Frobenius target matching baseline trajectory
  --norm-mode spectral-penalty : Option 4 - Soft spectral regularization penalty on loss

Run as:
    torchrun --standalone --nproc_per_node=2 -m scripts.trace_norm_train -- --depth 20 --norm-mode fro ...
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import nullcontext, contextmanager

import wandb
import torch
import numpy as np

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.optim_trace_norm import (
    FrobeniusConstraint, NuclearConstraint, FixedFrobeniusConstraint,
    NuclearEveryStepConstraint,
    ConditionClampConstraint, SpectralCapConstraint,
    GrowingFrobeniusConstraint, SpectralPenaltyConstraint,
    SpectralSharpenConstraint, WarmupOnlyConstraint,
    compute_norm_stats, compute_sv_entropy,
)
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Trace/norm-constrained Muon training")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU and torchao)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=float, default=10.5, help="calculate num_iterations to maintain data:param ratio")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens (-1 = auto)")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for Muon")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars")
parser.add_argument("--ns-mode", type=str, default="baseline", choices=["baseline", "fast4"], help="Newton-Schulz coefficient mode")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Norm constraint
parser.add_argument("--norm-mode", type=str, default="baseline",
                    choices=["baseline", "fro", "nuclear", "fixed-fro", "nuclear-every",
                             "cond-clamp", "spectral-cap", "growing-fro", "spectral-penalty",
                             "sharpen", "warmup-only"],
                    help="norm constraint mode")
parser.add_argument("--nuclear-k", type=int, default=10, help="SVD frequency for nuclear/cond-clamp modes (every K steps)")
parser.add_argument("--cond-max-ratio", type=float, default=1.5, help="max ratio for condition clamping")
parser.add_argument("--spectral-cap-ratio", type=float, default=1.5, help="cap ratio for spectral cap (multiple of init spectral norm)")
parser.add_argument("--growth-rate", type=float, default=0.001, help="growth rate for growing-fro mode")
parser.add_argument("--growth-max-ratio", type=float, default=3.5, help="max growth multiplier for growing-fro mode")
parser.add_argument("--penalty-lambda", type=float, default=0.01, help="lambda for spectral penalty mode")
parser.add_argument("--sharpen-k", type=int, default=50, help="sharpen every K steps")
parser.add_argument("--sharpen-alpha", type=float, default=1.05, help="sharpening exponent (>1 sharpens)")
parser.add_argument("--warmup-constraint-steps", type=int, default=500, help="steps to apply Frobenius constraint before releasing")
# Spectral logging
parser.add_argument("--spectral-log-every", type=int, default=250, help="log norm stats and SV entropy every N steps")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy()

# Newton-Schulz / Polar Express: fixed 5 iterations
_ns_steps = 5
print(f"Newton-Schulz mode: baseline ({_ns_steps} iterations, {_ns_steps * 3} matmuls)")

# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

if HAS_FA3:
    print0("Using Flash Attention 3 (Hopper GPU detected)")
else:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")

# -----------------------------------------------------------------------------
# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

def build_model_meta(depth):
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

model = build_model_meta(args.depth)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device)
model.init_weights()

base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"tn_{args.norm_mode}_d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "trace_norm", output_dirname)

# -----------------------------------------------------------------------------
# FP8 training initialization
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8_layers = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        print0(f"FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8_layers} layers")

@contextmanager
def disable_fp8(model):
    import torch.nn as nn
    fp8_locations = []
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield
        return

    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(
            fp8_module.in_features, fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device, dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Initialize norm constraint BEFORE compile (needs access to raw model params)

norm_constraint = None
if args.norm_mode == "fro":
    norm_constraint = FrobeniusConstraint()
    norm_constraint.record_init_norms(model)
    print0(f"[Variant A] Frobenius constraint: recorded init norms for {len(norm_constraint.targets)} layers")
elif args.norm_mode == "nuclear":
    norm_constraint = NuclearConstraint(K=args.nuclear_k)
    norm_constraint.record_init_norms(model)
    print0(f"[Variant B] Nuclear constraint (K={args.nuclear_k}): recorded init norms for {len(norm_constraint.nuc_targets)} layers")
elif args.norm_mode == "fixed-fro":
    norm_constraint = FixedFrobeniusConstraint()
    norm_constraint.record_init_norms(model)
    print0(f"[Variant C] Fixed Frobenius constraint: target={norm_constraint.fixed_target:.6f} for {len(norm_constraint.targets)} layers")
elif args.norm_mode == "cond-clamp":
    norm_constraint = ConditionClampConstraint(K=args.nuclear_k, max_ratio=args.cond_max_ratio)
    norm_constraint.record_init_norms(model)
    print0(f"[Option 1] Condition clamping (K={args.nuclear_k}, max_ratio={args.cond_max_ratio}): {len(norm_constraint.targets)} layers")
elif args.norm_mode == "spectral-cap":
    norm_constraint = SpectralCapConstraint(cap_ratio=args.spectral_cap_ratio)
    norm_constraint.record_init_norms(model)
    print0(f"[Option 2] Spectral cap ({args.spectral_cap_ratio}x init): {len(norm_constraint.targets)} layers")
elif args.norm_mode == "growing-fro":
    norm_constraint = GrowingFrobeniusConstraint(growth_rate=args.growth_rate, max_ratio=args.growth_max_ratio)
    norm_constraint.record_init_norms(model)
    print0(f"[Option 3] Growing Frobenius (rate={args.growth_rate}, max={args.growth_max_ratio}x): {len(norm_constraint.targets)} layers")
elif args.norm_mode == "spectral-penalty":
    norm_constraint = SpectralPenaltyConstraint(penalty_lambda=args.penalty_lambda)
    norm_constraint.record_init_norms(model)
    print0(f"[Option 4] Spectral penalty (lambda={args.penalty_lambda}): {len(norm_constraint.targets)} layers")
elif args.norm_mode == "sharpen":
    norm_constraint = SpectralSharpenConstraint(K=args.sharpen_k, alpha=args.sharpen_alpha)
    norm_constraint.record_init_norms(model)
    print0(f"[Option 5] Spectral sharpening (K={args.sharpen_k}, alpha={args.sharpen_alpha}): {len(norm_constraint.targets)} layers")
elif args.norm_mode == "warmup-only":
    norm_constraint = WarmupOnlyConstraint(warmup_steps=args.warmup_constraint_steps)
    norm_constraint.record_init_norms(model)
    print0(f"[Option 7] Warmup-only Frobenius constraint ({args.warmup_constraint_steps} steps): {len(norm_constraint.targets)} layers")
elif args.norm_mode == "nuclear-every":
    norm_constraint = NuclearEveryStepConstraint()
    norm_constraint.record_init_norms(model)
    print0(f"[Option 8] Nuclear norm every step (full SVD): {len(norm_constraint.targets)} layers")
else:
    print0("[Baseline] No norm constraint applied")

# Representative layers for SV entropy logging
# L0, L10, L19 attention Q and MLP up
n_layer = model_config.n_layer
entropy_layers = []
for li in [0, min(10, n_layer - 1), n_layer - 1]:
    entropy_layers.append(f"transformer.h.{li}.attn.c_q.weight")     # Attention Q
    entropy_layers.append(f"transformer.h.{li}.mlp.c_fc.weight")     # MLP up
# Filter to only layers that actually exist
existing_params = {name for name, _ in model.named_parameters()}
entropy_layers = [l for l in entropy_layers if l in existing_params]
print0(f"SV entropy tracking layers: {entropy_layers}")

# -----------------------------------------------------------------------------
# Compile the model
orig_model = model
model = torch.compile(model, dynamic=False)

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations

param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

def get_scaling_params(m):
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params

num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params)

d12_ref = build_model_meta(12)
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)
B_REF = 2**19

total_batch_size = args.total_batch_size
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size))
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF
if batch_ratio != 1.0:
    batch_lr_scale = batch_ratio ** 0.5
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
    ns_steps=_ns_steps,
)

# -----------------------------------------------------------------------------
# Initialize the DataLoaders
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=None)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# Calculate iterations and schedulers

assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")

total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}")
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
# Training loop

step = 0
val_bpb = None
min_val_bpb = float("inf")
smooth_train_loss = 0
total_training_time = 0

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Log files
train_log_path = os.path.join(checkpoint_dir, "train_log.jsonl")
val_log_path = os.path.join(checkpoint_dir, "val_log.jsonl")
spectral_log_path = os.path.join(checkpoint_dir, "spectral_log.jsonl")

if master_process:
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(train_log_path, "w") as f:
        f.write(json.dumps({"_config": user_config}) + "\n")
    with open(val_log_path, "w") as f:
        f.write(json.dumps({"_config": user_config}) + "\n")
    with open(spectral_log_path, "w") as f:
        f.write(json.dumps({"_config": user_config}) + "\n")

# Log initial spectral stats (step 0)
if master_process:
    print0("Logging initial spectral stats (step 0)...")
    init_stats = compute_norm_stats(orig_model)
    init_entropy = compute_sv_entropy(orig_model, entropy_layers)
    spectral_entry = {"step": 0, "norm_stats": init_stats, "sv_entropy": init_entropy}
    with open(spectral_log_path, "a") as f:
        f.write(json.dumps(spectral_entry) + "\n")
    print0(f"Logged initial spectral stats for {len(init_stats)} layers, entropy for {len(init_entropy)} layers")

while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Evaluate val bpb
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model), autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        if master_process:
            with open(val_log_path, "a") as f:
                f.write(json.dumps({"step": step, "val_bpb": val_bpb, "total_training_time": total_training_time}) + "\n")
        wandb_run.log({
            "step": step, "total_training_flops": flops_so_far, "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # CORE metric
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model), autocast_ctx:
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({"step": step, "total_training_flops": flops_so_far, "core_metric": results["core_metric"]})
        model.train()

    # Sample
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model), autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # Save checkpoint
    if last_step or (step > 0 and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir, step,
            orig_model.state_dict(), optimizer.state_dict(),
            {
                "step": step, "val_bpb": val_bpb,
                "model_config": model_config_kwargs, "user_config": user_config,
                "device_batch_size": args.device_batch_size, "max_seq_len": args.max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {"min_val_bpb": min_val_bpb, "smooth_train_loss": smooth_train_loss, "total_training_time": total_training_time},
            },
            rank=ddp_rank,
        )

    if last_step:
        break

    # =========================================================================
    # Training step
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        # Add spectral penalty if using Option 4
        if isinstance(norm_constraint, SpectralPenaltyConstraint):
            penalty = norm_constraint.compute_penalty(orig_model, step)
            if penalty != 0.0:
                loss = loss + penalty / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)

    # Step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # Apply norm constraint AFTER optimizer step
    if norm_constraint is not None:
        norm_constraint.apply(orig_model, step)

    # =========================================================================
    # Spectral logging (every spectral_log_every steps)
    should_log_spectral = master_process and args.spectral_log_every > 0 and (step % args.spectral_log_every == 0)
    if should_log_spectral:
        norm_stats = compute_norm_stats(orig_model)
        sv_entropy = compute_sv_entropy(orig_model, entropy_layers)
        spectral_entry = {"step": step, "norm_stats": norm_stats, "sv_entropy": sv_entropy}
        with open(spectral_log_path, "a") as f:
            f.write(json.dumps(spectral_entry) + "\n")

        # Log summary to wandb
        for layer_name, stats in norm_stats.items():
            short_name = layer_name.replace("transformer.h.", "L").replace(".weight", "")
            wandb_run.log({
                f"norm/{short_name}/frobenius": stats['frobenius_norm'],
                f"norm/{short_name}/spectral": stats['spectral_norm'],
                f"norm/{short_name}/nuclear": stats['nuclear_norm'],
                "step": step,
            })
        for layer_name, ent in sv_entropy.items():
            short_name = layer_name.replace("transformer.h.", "L").replace(".weight", "")
            wandb_run.log({
                f"entropy/{short_name}": ent,
                "step": step,
            })

    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")

    if master_process:
        with open(train_log_path, "a") as f:
            f.write(json.dumps({"step": step, "loss": debiased_smooth_loss, "dt": dt, "total_training_time": total_training_time}) + "\n")
    if step % 100 == 0:
        wandb_run.log({
            "step": step, "total_training_flops": flops_so_far, "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss, "train/lrm": lrm, "train/dt": dt,
            "train/tok_per_sec": tok_per_sec, "train/mfu": mfu, "train/epoch": epoch,
        })

    first_step_of_run = (step == 0)
    step += 1

    if first_step_of_run:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# Print final stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nanochat.report import get_report
get_report().log(section=f"Trace Norm ({args.norm_mode}) training", data=[
    user_config,
    {"Number of parameters": num_params, "Number of FLOPs per token": f"{num_flops_per_token:e}",
     "Calculated number of iterations": num_iterations, "Number of training tokens": total_tokens},
    {"Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
     "Final validation bpb": val_bpb, "CORE metric estimate": results.get("core_metric", None),
     "MFU %": f"{mfu:.2f}%", "Total training time": f"{total_training_time/60:.2f}m",
     "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB"},
])

wandb_run.finish()
compute_cleanup()
