"""
Random singular value mapping experiment for Muon optimizer.

Instead of Polar Express or an affine mapping, this replaces each (normalized)
singular value with an independent uniform random draw from [0.1, 1.0].

The point: if training quality depends on the *shape* of the singular value
spectrum (i.e. which values are large vs small), randomizing it should hurt.
If it doesn't hurt much, the shaping doesn't matter — only the overall scale does.

Usage:
  torchrun --standalone --nproc_per_node=4 -m scripts.rand_sv_train -- \
    --depth 20

From root directory of the project.
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
from torch import Tensor

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, get_dist_info
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.flash_attention import HAS_FA3
from nanochat.optim import adamw_step_fused, polar_express_coeffs, DistMuonAdamW, MuonAdamW
print_banner()

# =============================================================================
# SVD-based random singular value mapping
# =============================================================================

def _svd_map_batched(X: torch.Tensor, map_fn):
    """
    Apply a singular-value mapping to a batch of matrices.
    X: (batch, m, n) or (m, n)
    map_fn: f(singular_vals) -> mapped_singular_vals
    """
    orig_dtype = X.dtype
    X32 = X.to(torch.float32)
    U, S, Vh = torch.linalg.svd(X32, full_matrices=False)
    S_mapped = map_fn(S)
    US = U * S_mapped.unsqueeze(-2)
    Y = US @ Vh
    return Y.to(orig_dtype)


def make_svd_map_random(lo: float = 0.1, hi: float = 1.0):
    """
    Random mapping: replace each (normalized) singular value with
    an independent draw from Uniform[lo, hi].

    This tests whether the *ordering / shape* of the singular value spectrum
    matters. A structured mapping (affine, clamp) preserves the relative ranking;
    this one destroys it entirely.
    """
    lo_val = float(lo)
    hi_val = float(hi)

    def _map(s: torch.Tensor) -> torch.Tensor:
        s = s.to(torch.float32)
        # Identify which singular values are active (nonzero after normalization)
        s_max = s.amax(dim=-1, keepdim=True).clamp_min(1e-12)
        s_norm = s / s_max
        active = (s_norm > 0)
        # Draw uniform random values in [lo, hi] for every position
        rand_vals = torch.empty_like(s).uniform_(lo_val, hi_val)
        # Zero out inactive singular values (keep dead directions dead)
        s_new = torch.where(active, rand_vals, torch.zeros_like(rand_vals))
        return s_new

    return _map


# =============================================================================
# Modified Muon step: replaces Polar Express with random SV mapping
# =============================================================================

# NOTE: torch.linalg.svd is not well supported in torch.compile graphs,
# so we intentionally skip @torch.compile here.
@torch.no_grad()
def muon_step_svd_rand(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
    svd_map_fn,
) -> None:
    """
    Muon step with random SV mapping instead of Polar Express.
    Same structure: momentum -> SVD random mapping -> variance reduction -> cautious update.
    """
    # Nesterov momentum (same as original)
    momentum = momentum_t.item()
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Random SV mapping (replaces Polar Express)
    g = _svd_map_batched(g, svd_map_fn).to(g.dtype)

    # Variance reduction (NorMuon, same as original)
    beta2 = beta2_t.item()
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update (same as original)
    lr = lr_t.item()
    wd = wd_t.item()
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


# =============================================================================
# Modified optimizers that use random SV mapping
# =============================================================================

class MuonAdamW_SVDRand(MuonAdamW):
    """Single-GPU MuonAdamW that uses random SV mapping instead of Polar Express."""

    def __init__(self, param_groups: list[dict], svd_map_fn):
        super().__init__(param_groups)
        self.svd_map_fn = svd_map_fn

    def _step_muon(self, group: dict) -> None:
        params = group['params']
        if not params:
            return

        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        muon_step_svd_rand(
            stacked_grads, stacked_params,
            momentum_buffer, second_momentum_buffer,
            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
            group["ns_steps"], red_dim,
            self.svd_map_fn,
        )

        torch._foreach_copy_(params, list(stacked_params.unbind(0)))


class DistMuonAdamW_SVDRand(DistMuonAdamW):
    """Distributed MuonAdamW that uses random SV mapping instead of Polar Express."""

    def __init__(self, param_groups: list[dict], svd_map_fn):
        super().__init__(param_groups)
        self.svd_map_fn = svd_map_fn

    def _compute_muon(self, group, info, gather_list, rank):
        import torch.distributed as dist

        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_svd_rand(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
                self.svd_map_fn,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))


# =============================================================================
# Monkey-patch GPT.setup_optimizer to use our random SV optimizers
# =============================================================================

def make_setup_optimizer_svd_rand(svd_map_fn):
    """Return a setup_optimizer method that uses random SV mapping."""
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW_SVDRand if ddp else MuonAdamW_SVDRand
        optimizer = Factory(param_groups, svd_map_fn=svd_map_fn)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    return setup_optimizer


# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Random SV mapping experiment for Muon optimizer")
# Experiment-specific args
parser.add_argument("--rand-lo", type=float, default=0.1, help="Lower bound for uniform random SV draw (default 0.1)")
parser.add_argument("--rand-hi", type=float, default=1.0, help="Upper bound for uniform random SV draw (default 1.0)")
parser.add_argument("--results-dir", type=str, default="rand_sv_results", help="Directory to save results")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8
parser.add_argument("--fp8", action="store_true", help="enable FP8 training")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"])
# Model
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--window-pattern", type=str, default="SSSL")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1)
parser.add_argument("--target-flops", type=float, default=-1.0)
parser.add_argument("--target-param-data-ratio", type=float, default=10.5)
# Optimization
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
# Evaluation
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--eval-tokens", type=int, default=40*524288)
parser.add_argument("--core-metric-every", type=int, default=-1)
parser.add_argument("--sample-every", type=int, default=-1)
parser.add_argument("--save-every", type=int, default=-1)
args = parser.parse_args()
user_config = vars(args).copy()

# =============================================================================
# Build the SVD mapping function
# =============================================================================

svd_map_fn = make_svd_map_random(lo=args.rand_lo, hi=args.rand_hi)
run_tag = f"rand_sv_lo{args.rand_lo:.2f}_hi{args.rand_hi:.2f}"
print0(f"=== Random SV mapping: Uniform[{args.rand_lo}, {args.rand_hi}] ===")

# Monkey-patch GPT.setup_optimizer
GPT.setup_optimizer = make_setup_optimizer_svd_rand(svd_map_fn)

# =============================================================================
# Compute init and wandb
# =============================================================================

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
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rand-sv", name=f"{args.run}_{run_tag}", config=user_config)

if HAS_FA3:
    print0("Using Flash Attention 3")
else:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")

# =============================================================================
# Tokenizer
# =============================================================================

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# =============================================================================
# Model init
# =============================================================================

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

# =============================================================================
# FP8
# =============================================================================

if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn
        def fp8_module_filter(mod, fqn):
            if not isinstance(mod, nn.Linear):
                return False
            return mod.in_features % 16 == 0 and mod.out_features % 16 == 0
        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8_layers = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        print0(f"FP8 training enabled - converted {num_fp8_layers} layers")

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
                           bias=fp8_module.bias is not None, device=fp8_module.weight.device, dtype=fp8_module.weight.dtype)
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)
    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# =============================================================================
# Compile the model
# =============================================================================

orig_model = model
model = torch.compile(model, dynamic=False)

# =============================================================================
# Scaling laws
# =============================================================================

param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

def get_scaling_params(m):
    params_counts = m.num_scaling_params()
    return params_counts['transformer_matrices'] + params_counts['lm_head']

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

# =============================================================================
# Optimizer
# =============================================================================

optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

# =============================================================================
# DataLoader
# =============================================================================

train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader)

# =============================================================================
# Iterations and schedulers
# =============================================================================

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
print0(f"Total tokens: {total_tokens:,} | Iterations: {num_iterations:,}")

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

# =============================================================================
# Results logging setup
# =============================================================================

results_dir = os.path.join(args.results_dir, run_tag)
if master_process:
    os.makedirs(results_dir, exist_ok=True)
val_loss_log = []
min_val_bpb = float("inf")

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Gradient accumulation steps: {grad_accum_steps}")

# =============================================================================
# Training loop
# =============================================================================

step = 0
smooth_train_loss = 0
total_training_time = 0

while True:
    last_step = step == num_iterations

    # Evaluation
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model), autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        val_loss_log.append({"step": step, "val_bpb": val_bpb})
        print0(f"[{run_tag}] step {step:05d} | val_bpb: {val_bpb:.6f}")
        wandb_run.log({"step": step, "val/bpb": val_bpb, "total_training_time": total_training_time})
        model.train()

    if last_step:
        break

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
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = get_muon_momentum(step)
            group["weight_decay"] = get_weight_decay(step)

    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    tok_per_sec = int(total_batch_size / dt)
    if step > 10:
        total_training_time += dt

    if step % 50 == 0:
        steps_done = step - 10
        if steps_done > 0:
            avg_time_per_step = total_training_time / steps_done
            eta_str = f" | eta: {(num_iterations - step) * avg_time_per_step / 60:.1f}m"
        else:
            eta_str = ""
        pct_done = 100 * step / num_iterations
        print0(f"[{run_tag}] step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | dt: {dt*1000:.2f}ms | tok/s: {tok_per_sec:,}{eta_str}")

    if step % 100 == 0:
        wandb_run.log({"step": step, "train/loss": debiased_smooth_loss, "train/tok_per_sec": tok_per_sec})

    first_step = step == 0
    step += 1
    if first_step:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# =============================================================================
# Save results
# =============================================================================

if master_process:
    results_file = os.path.join(results_dir, "val_loss.json")
    results_data = {
        "run_tag": run_tag,
        "rand_lo": args.rand_lo,
        "rand_hi": args.rand_hi,
        "depth": args.depth,
        "num_iterations": num_iterations,
        "total_batch_size": total_batch_size,
        "min_val_bpb": min_val_bpb,
        "val_loss_log": val_loss_log,
        "user_config": user_config,
    }
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print0(f"Results saved to {results_file}")

compute_cleanup()
wandb_run.finish()
