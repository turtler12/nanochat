"""
WeightSoftmaxMuon training: standard Muon step (Newton-Schulz), then spectral
projection on the weight matrix W.

At each step:
  1. Standard Muon update: momentum → Polar Express → variance reduction → param update
  2. Spectral projection on W: U, S_W, Vh = SVD(W); S_proj = tau_W * softmax(eta * S_W)
     where tau_W = initial nuclear norm of W (stored at step 0)
  3. Reconstruct: W = U @ diag(S_proj) @ Vh

Run as:
    torchrun --standalone --nproc_per_node=2 -m scripts.weight_softmax_muon_train -- --depth 20 --eta 0.5 ...
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

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, get_dist_info
from nanochat.optim_weight_softmax import WeightSoftmaxMuonAdamW, DistWeightSoftmaxMuonAdamW
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="WeightSoftmaxMuon training")
parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--fp8", action="store_true")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"])
parser.add_argument("--depth", type=int, default=20)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--window-pattern", type=str, default="SSSL")
parser.add_argument("--num-iterations", type=int, default=-1)
parser.add_argument("--target-flops", type=float, default=-1.0)
parser.add_argument("--target-param-data-ratio", type=float, default=10.5)
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--total-batch-size", type=int, default=-1)
parser.add_argument("--embedding-lr", type=float, default=0.3)
parser.add_argument("--unembedding-lr", type=float, default=0.004)
parser.add_argument("--weight-decay", type=float, default=0.2)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--scalar-lr", type=float, default=0.5)
parser.add_argument("--adam-beta1", type=float, default=0.8)
parser.add_argument("--adam-beta2", type=float, default=0.95)
parser.add_argument("--warmup-ratio", type=float, default=0.0)
parser.add_argument("--warmdown-ratio", type=float, default=0.5)
parser.add_argument("--final-lr-frac", type=float, default=0.0)
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--eval-tokens", type=int, default=40*524288)
parser.add_argument("--core-metric-every", type=int, default=2000)
parser.add_argument("--core-metric-max-per-task", type=int, default=500)
parser.add_argument("--sample-every", type=int, default=2000)
parser.add_argument("--save-every", type=int, default=-1)
# WeightSoftmaxMuon-specific
parser.add_argument("--eta", type=float, default=0.5, help="softmax temperature for weight spectral projection")
parser.add_argument("--spectral-log-every", type=int, default=100, help="log spectral diagnostics every N steps")
parser.add_argument("--model-tag", type=str, default=None)
args = parser.parse_args()
user_config = vars(args).copy()

_ns_steps = 5

# -----------------------------------------------------------------------------
# Compute init
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

# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Model
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
output_dirname = args.model_tag if args.model_tag else f"wsoftmax_eta{args.eta}_d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)

# FP8
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            return mod.in_features % 16 == 0 and mod.out_features % 16 == 0
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
        linear = nn.Linear(fp8_module.in_features, fp8_module.out_features,
                           bias=fp8_module.bias is not None,
                           device=fp8_module.weight.device, dtype=fp8_module.weight.dtype)
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
# Identify representative layer for spectral logging BEFORE compile
spectral_layer_name = "transformer.h.0.mlp.c_fc.weight"
spectral_layer_param = dict(model.named_parameters())[spectral_layer_name]
print0(f"Spectral logging layer: {spectral_layer_name} (shape: {spectral_layer_param.shape})")

# Compile
orig_model = model
model = torch.compile(model, dynamic=False)

# -----------------------------------------------------------------------------
# Scaling laws
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

def get_scaling_params(m):
    pc = m.num_scaling_params()
    return pc['transformer_matrices'] + pc['lm_head']

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
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,}")

weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f}")

# -----------------------------------------------------------------------------
# Custom optimizer setup
def setup_weight_softmax_optimizer(model, eta):
    model_dim = model.config.n_embd
    ddp_info = get_dist_info()
    is_ddp = ddp_info[0]

    matrix_params = list(model.transformer.h.parameters())
    value_embeds_params = list(model.value_embeds.parameters())
    embedding_params = list(model.transformer.wte.parameters())
    lm_head_params = list(model.lm_head.parameters())
    resid_params = [model.resid_lambdas]
    x0_params = [model.x0_lambdas]

    dmodel_lr_scale = (model_dim / 768) ** -0.5

    param_groups = [
        dict(kind='adamw', params=lm_head_params, lr=args.unembedding_lr * batch_lr_scale * dmodel_lr_scale, betas=(args.adam_beta1, args.adam_beta2), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=embedding_params, lr=args.embedding_lr * batch_lr_scale * dmodel_lr_scale, betas=(args.adam_beta1, args.adam_beta2), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=value_embeds_params, lr=args.embedding_lr * batch_lr_scale * dmodel_lr_scale, betas=(args.adam_beta1, args.adam_beta2), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=resid_params, lr=args.scalar_lr * batch_lr_scale * 0.01, betas=(args.adam_beta1, args.adam_beta2), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=x0_params, lr=args.scalar_lr * batch_lr_scale, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(
            kind='muon', params=group_params, lr=args.matrix_lr * batch_lr_scale,
            momentum=0.95, ns_steps=_ns_steps, beta2=0.95, weight_decay=weight_decay_scaled,
        ))

    Factory = DistWeightSoftmaxMuonAdamW if is_ddp else WeightSoftmaxMuonAdamW
    optimizer = Factory(param_groups, eta=eta)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer

optimizer = setup_weight_softmax_optimizer(orig_model, eta=args.eta)
print0(f"WeightSoftmaxMuon optimizer created with eta={args.eta}")

# DataLoaders
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=None)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader)

# Iterations and schedulers
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
elif args.target_param_data_ratio > 0:
    num_iterations = target_tokens // total_batch_size

total_tokens = total_batch_size * num_iterations
print0(f"Total training tokens: {total_tokens:,} | Iterations: {num_iterations:,}")

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

# Training loop state
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

while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Validate
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
        wandb_run.log({"step": step, "total_training_flops": flops_so_far, "total_training_time": total_training_time, "val/bpb": val_bpb})
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
        prompts = ["The capital of France is", "The chemical symbol of gold is", "If yesterday was Friday, then tomorrow will be"]
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
            checkpoint_dir, step, orig_model.state_dict(), optimizer.state_dict(),
            {"step": step, "val_bpb": val_bpb, "model_config": model_config_kwargs, "user_config": user_config,
             "device_batch_size": args.device_batch_size, "max_seq_len": args.max_seq_len,
             "dataloader_state_dict": dataloader_state_dict,
             "loop_state": {"min_val_bpb": min_val_bpb, "smooth_train_loss": smooth_train_loss, "total_training_time": total_training_time}},
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
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)

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

    # =========================================================================
    # Spectral logging for representative layer
    should_log_spectral = master_process and args.spectral_log_every > 0 and step % args.spectral_log_every == 0
    if should_log_spectral:
        with torch.no_grad():
            W = spectral_layer_param.float()
            S_w = torch.linalg.svdvals(W)
            p_sv = S_w / (S_w.sum() + 1e-8)
            sv_entropy = -(p_sv * torch.log(p_sv + 1e-8)).sum().item()
            nuclear_norm = S_w.sum().item()

        spectral_entry = {
            "step": step,
            "sv_entropy": sv_entropy,
            "nuclear_norm": nuclear_norm,
            "eta": args.eta,
        }
        with open(spectral_log_path, "a") as f:
            f.write(json.dumps(spectral_entry) + "\n")

    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0

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

# Final stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

from nanochat.report import get_report
get_report().log(section=f"WeightSoftmaxMuon (eta={args.eta}) training", data=[
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
