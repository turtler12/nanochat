"""
Ablation: how much does Q quality matter?
Tests 7 update modes: muon, fast4, ns3, ns2, ns1, ns0_normalized, ns0_sign

Run as:
    python -m scripts.ablation_train --update-mode muon --depth 12 --num-iterations 2205
    python -m scripts.ablation_train --update-mode ns0_normalized --depth 12 --num-iterations 2205
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

try:
    import wandb
except ImportError:
    wandb = None
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops
from nanochat.optim_ablation import AblationMuonAdamW, DistAblationMuonAdamW, UPDATE_MODES
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Ablation: NS iteration count vs model quality")
parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--device-type", type=str, default="")
parser.add_argument("--fp8", action="store_true")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"])
parser.add_argument("--depth", type=int, default=12)
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
parser.add_argument("--resume-from-step", type=int, default=-1)
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--eval-tokens", type=int, default=40*524288)
parser.add_argument("--core-metric-every", type=int, default=2000)
parser.add_argument("--core-metric-max-per-task", type=int, default=500)
parser.add_argument("--sample-every", type=int, default=2000)
parser.add_argument("--save-every", type=int, default=-1)
parser.add_argument("--model-tag", type=str, default=None)
# Ablation-specific
parser.add_argument("--update-mode", type=str, default="muon",
                    help=f"Update mode for matrix params. One of: {UPDATE_MODES}")
args = parser.parse_args()
user_config = vars(args).copy()

print0(f"Update mode: {args.update_mode}")

# -----------------------------------------------------------------------------
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

use_dummy_wandb = args.run == "dummy" or not master_process or wandb is None
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

if HAS_FA3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected)")
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}').")
        print0("WARNING: Recommend using --window-pattern L for full context attention.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
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
output_dirname = args.model_tag if args.model_tag else f"ablation_{args.update_mode}_d{args.depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

# -----------------------------------------------------------------------------
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
        num_skipped = sum(1 for m in model.modules() if isinstance(m, nn.Linear)) - num_fp8_layers
        print0(f"✓ FP8 enabled ({args.fp8_recipe}) - {num_fp8_layers} layers converted, {num_skipped} skipped")

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
                parent, attr_name = model, name
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
orig_model = model
model = torch.compile(model, dynamic=False)

# -----------------------------------------------------------------------------
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()

def get_scaling_params(m):
    counts = m.num_scaling_params()
    return counts['transformer_matrices'] + counts['lm_head']

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
    print0(f"Scaling weight decay {args.weight_decay:.6f} -> {weight_decay_scaled:.6f}")

# -----------------------------------------------------------------------------
# Build the optimizer using AblationMuonAdamW.
# We replicate GPT.setup_optimizer here, injecting update_mode into muon groups.
def setup_ablation_optimizer(model, update_mode):
    model_dim = orig_model.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    print0(f"dmodel_lr_scale: {dmodel_lr_scale:.6f}")

    matrix_params    = list(orig_model.transformer.h.parameters())
    value_embeds     = list(orig_model.value_embeds.parameters())
    embedding_params = list(orig_model.transformer.wte.parameters())
    lm_head_params   = list(orig_model.lm_head.parameters())
    resid_params     = [orig_model.resid_lambdas]
    x0_params        = [orig_model.x0_lambdas]

    matrix_lr   = args.matrix_lr  * batch_lr_scale
    embed_lr    = args.embedding_lr * batch_lr_scale
    unembed_lr  = args.unembedding_lr * batch_lr_scale
    scalar_lr   = args.scalar_lr * batch_lr_scale
    adam_betas  = (args.adam_beta1, args.adam_beta2)

    param_groups = [
        dict(kind='adamw', params=lm_head_params,   lr=unembed_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=embedding_params, lr=embed_lr   * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=value_embeds,     lr=embed_lr   * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=resid_params,     lr=scalar_lr * 0.01,             betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=x0_params,        lr=scalar_lr,                    betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(
            kind='muon', params=group_params, lr=matrix_lr,
            momentum=0.95, ns_steps=5, beta2=0.95,
            weight_decay=weight_decay_scaled,
            update_mode=update_mode,
        ))

    Factory = DistAblationMuonAdamW if ddp else AblationMuonAdamW
    optimizer = Factory(param_groups)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer

optimizer = setup_ablation_optimizer(orig_model, args.update_mode)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
else:
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
total_tokens = total_batch_size * num_iterations
print0(f"Total training tokens: {total_tokens:,}")
print0(f"Tokens:Scaling params ratio: {total_tokens / num_scaling_params:.2f}")
print0(f"Total FLOPs estimate: {num_flops_per_token * total_tokens:e}")

def get_lr_multiplier(it):
    warmup_iters   = round(args.warmup_ratio   * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress + (1 - progress) * args.final_lr_frac

import math as _math
_D12_REF_STEPS = 2205  # reference step count from d12 runs
# λ = ln(3)/100 so that η(100) = 0.3 * exp(-λ*100) = 0.1 (reaches 0.1 at ~step 100 / ~10 min)
_ETA_LAMBDA = _math.log(3.0) / 100.0
# λ_zero: η(100) ≈ 0.001 (effectively 0), i.e. λ = ln(300)/100
_ETA_LAMBDA_ZERO_S100 = _math.log(300.0) / 100.0

def get_ftrl_eta(it, mode):
    """Scheduled eta for dynamic-eta FTRL modes."""
    eta0 = 0.3
    t = it / max(num_iterations, 1)
    if mode == "ns3_ftrl_linear_eta0p3":
        return eta0 * (1.0 - t)
    elif mode == "ns3_ftrl_exp_eta0p3":
        return eta0 * _math.exp(-5.0 * t)
    elif mode in ("ns3_ftrl_exp_abs_eta0p3", "ns3_ftrl_exp_abs_then_muon",
                  "ns3_ftrl_exp_abs_then_muon_s100", "ns3_ftrl_exp_abs_then_muon_s250"):
        # η(t) = 0.3 * exp(-λ*step), λ = ln(3)/100 => η(100)=0.1, η(200)≈0.033
        return eta0 * _math.exp(-_ETA_LAMBDA * it)
    elif mode in ("ns3_ftrl_exp_abs_zero_s100", "ns3_ftrl_exp_abs_then_muon_s150"):
        # η(t) = 0.3 * exp(-λ*step), λ = ln(300)/100 => η(100)≈0.001 (~0)
        return eta0 * _math.exp(-_ETA_LAMBDA_ZERO_S100 * it)
    return 0.0

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# -----------------------------------------------------------------------------
if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float("inf")
    smooth_train_loss = 0
    total_training_time = 0
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Grad accum steps: {grad_accum_steps}")

train_log_path = os.path.join(checkpoint_dir, "train_log.jsonl")
val_log_path   = os.path.join(checkpoint_dir, "val_log.jsonl")
if master_process and not resuming:
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(train_log_path, "w") as f:
        f.write(json.dumps({"_config": user_config}) + "\n")
    with open(val_log_path, "w") as f:
        f.write(json.dumps({"_config": user_config}) + "\n")

# -----------------------------------------------------------------------------
while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model), autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | val bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        if master_process:
            with open(val_log_path, "a") as f:
                f.write(json.dumps({"step": step, "val_bpb": val_bpb, "total_training_time": total_training_time}) + "\n")
        wandb_run.log({"step": step, "total_training_flops": flops_so_far,
                       "total_training_time": total_training_time, "val/bpb": val_bpb})
        model.train()

    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model), autocast_ctx:
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({"step": step, "total_training_flops": flops_so_far,
                       "core_metric": results["core_metric"],
                       "centered_results": results["centered_results"]})
        model.train()

    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model), autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir, step,
            orig_model.state_dict(), optimizer.state_dict(),
            {"step": step, "val_bpb": val_bpb, "model_config": model_config_kwargs,
             "user_config": user_config, "device_batch_size": args.device_batch_size,
             "max_seq_len": args.max_seq_len, "dataloader_state_dict": dataloader_state_dict,
             "loop_state": {"min_val_bpb": min_val_bpb, "smooth_train_loss": smooth_train_loss,
                            "total_training_time": total_training_time}},
            rank=ddp_rank,
        )

    if last_step:
        break

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
            if args.update_mode in ("ns3_ftrl_linear_eta0p3", "ns3_ftrl_exp_eta0p3",
                                    "ns3_ftrl_exp_abs_eta0p3", "ns3_ftrl_exp_abs_then_muon",
                                    "ns3_ftrl_exp_abs_then_muon_s100", "ns3_ftrl_exp_abs_zero_s100",
                                    "ns3_ftrl_exp_abs_then_muon_s150", "ns3_ftrl_exp_abs_then_muon_s250"):
                switch_step = {"ns3_ftrl_exp_abs_then_muon_s100": 100,
                               "ns3_ftrl_exp_abs_then_muon_s150": 150,
                               "ns3_ftrl_exp_abs_then_muon":      200,
                               "ns3_ftrl_exp_abs_then_muon_s250": 250}.get(args.update_mode)
                if switch_step is not None and step >= switch_step:
                    group["update_mode"] = "muon"
                else:
                    group["ftrl_eta"] = get_ftrl_eta(step, args.update_mode)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0

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
        avg_dt = total_training_time / steps_done
        eta_str = f" | eta: {(num_iterations - step) * avg_dt / 60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) [{args.update_mode}] | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.2f}ms | tok/s: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {epoch}{eta_str}")
    if master_process:
        with open(train_log_path, "a") as f:
            f.write(json.dumps({"step": step, "loss": debiased_smooth_loss, "dt": dt, "total_training_time": total_training_time}) + "\n")
    if step % 100 == 0:
        wandb_run.log({
            "step": step, "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss, "train/lrm": lrm,
            "train/dt": dt, "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu, "train/epoch": epoch,
        })

    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1
    if first_step_of_run:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# -----------------------------------------------------------------------------
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Min val bpb: {min_val_bpb:.6f}")

from nanochat.report import get_report
get_report().log(section="Ablation training", data=[
    user_config,
    {"update_mode": args.update_mode, "num_params": num_params,
     "num_iterations": num_iterations, "total_tokens": total_tokens,
     "min_val_bpb": min_val_bpb if val_bpb is not None else None,
     "final_val_bpb": val_bpb,
     "core_metric": results.get("core_metric", None)},
])

wandb_run.finish()
compute_cleanup()
