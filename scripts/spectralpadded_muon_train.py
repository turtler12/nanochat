"""
Spectral-Padded Muon experiment: blends the raw gradient with the Polar Express
polar factor scaled by sigma_max, giving a hard floor to tail singular values.

    update = (1 - gamma) * g_raw + gamma * sigma_max(g_raw) * P

Environment variables:
  PAD_RATIO=0.0           Blend ratio gamma (0.0 = baseline, >0 = spectral padding)
  POWER_ITERS=1           Power iterations for sigma_max estimation (1-2)
  POWER_BATCH=4           Probe vectors per matrix for sigma_max estimate (4-8)
  SV_LOG=0                Enable SV diagnostics logging ("1" to enable)
  SV_LOG_EVERY=500        Log SV diagnostics every N steps

Two experiment modes:
  1) Baseline:           PAD_RATIO=0.0  (standard Muon, identical to base_train)
  2) Spectral padded:    PAD_RATIO=0.1  (blend raw + sigma_max*P)
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
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.common import (
    compute_init, compute_cleanup, print0, DummyWandb, print_banner,
    get_base_dir, autodetect_device_type, get_peak_flops, get_dist_info,
)
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from nanochat.flash_attention import HAS_FA3
from scripts.muon_spectralpadded import (
    MuonAdamW as MuonAdamW_SpectralPadded,
    DistMuonAdamW as DistMuonAdamW_SpectralPadded,
    polar_express_coeffs,
)
print_banner()

# =============================================================================
# Configuration from environment
# =============================================================================

PAD_RATIO = float(os.environ.get("PAD_RATIO", "0.0"))
POWER_ITERS = int(os.environ.get("POWER_ITERS", "1"))
POWER_BATCH = int(os.environ.get("POWER_BATCH", "4"))
SV_LOG = os.environ.get("SV_LOG", "0") == "1"
SV_LOG_EVERY = int(os.environ.get("SV_LOG_EVERY", "500"))

print0(f"=== Spectral-Padded Muon Experiment ===")
print0(f"  PAD_RATIO={PAD_RATIO}, POWER_ITERS={POWER_ITERS}, POWER_BATCH={POWER_BATCH}")
print0(f"  SV_LOG={SV_LOG}, every={SV_LOG_EVERY}")

# =============================================================================
# SV diagnostics
# =============================================================================

def _sv_stats(mat: Tensor) -> dict:
    """Compute SV summary for a single 2D matrix."""
    sv = torch.linalg.svdvals(mat.float())
    sigma_max = sv[0].clamp_min(1e-12)
    sv_norm = sv / sigma_max
    return {
        "sigma_max": sigma_max.item(),
        "p05": sv_norm.quantile(0.05).item(),
        "p25": sv_norm.quantile(0.25).item(),
        "p50": sv_norm.quantile(0.50).item(),
        "p75": sv_norm.quantile(0.75).item(),
        "p95": sv_norm.quantile(0.95).item(),
    }


@torch.no_grad()
def muon_step_with_sv_logging(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    pad_ratio_t: Tensor,
    power_iters: int,
    power_batch: int,
    ns_steps: int,
    red_dim: int,
    sv_log_idx: int = -1,
) -> dict | None:
    """
    Uncompiled Muon step that mirrors muon_step_fused but captures SV stats
    at three stages:
      - pre:     after Nesterov momentum (raw update direction)
      - polar:   pure Polar Express output (all SVs ~ 1)
      - blended: after spectral padding blend (1-gamma)*g + gamma*sigma_max*P
    Returns SV stats dict if sv_log_idx >= 0, else None.
    """
    sv_result = None

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # --- SV LOG: pre (after momentum, before anything) ---
    if sv_log_idx >= 0:
        sv_result = {"pre": _sv_stats(g[sv_log_idx])}

    # Polar Express
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    P = X

    # --- SV LOG: polar (pure Polar Express output, before blend) ---
    if sv_log_idx >= 0:
        sv_result["polar"] = _sv_stats(P[sv_log_idx])

    # Spectral padding blend
    pad_ratio = pad_ratio_t.to(g.dtype)
    b = power_batch
    Omega = torch.randn(g.shape[0], g.shape[-1], b, device=g.device, dtype=g.dtype)
    Omega = Omega / (Omega.norm(dim=-2, keepdim=True).clamp_min(1e-6))
    for _ in range(power_iters):
        Y = g @ Omega
        Omega = g.mT @ Y
        Omega = Omega / (Omega.norm(dim=-2, keepdim=True).clamp_min(1e-6))
    Y = g @ Omega
    sigma_hat = (Y.norm(dim=-2, keepdim=True).amax(dim=-1, keepdim=True)).clamp_min(1e-8)

    g = g.mul(1 - pad_ratio) + (P.to(g.dtype) * sigma_hat) * pad_ratio

    # --- SV LOG: blended (after spectral padding, before variance reduction) ---
    if sv_log_idx >= 0:
        sv_result["blended"] = _sv_stats(g[sv_log_idx])
        sv_result["sigma_hat"] = sigma_hat[sv_log_idx].squeeze().item()

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
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

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

    return sv_result


# =============================================================================
# SVLogger: writes JSONL with per-stage SV stats
# =============================================================================

class SVLogger:
    """Logs singular-value statistics of the Muon update at stages:
    pre (after momentum), polar (after Polar Express), blended (after spectral blend).

    Picks 2 representative Muon groups (largest and smallest shape) and logs
    one matrix from each.
    """

    def __init__(self, log_every: int, variant: str, log_dir: str | None):
        self.log_every = log_every
        self.variant = variant
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "sv_log.jsonl") if log_dir else None
        self._log_groups: dict[int, str] | None = None

    def should_log(self, step: int) -> bool:
        return self.log_every > 0 and step % self.log_every == 0

    def pick_log_groups(self, optimizer) -> dict[int, str]:
        """Select 2 Muon groups to log: largest and smallest by numel."""
        muon_groups = [g for g in optimizer.param_groups if g['kind'] == 'muon']
        if not muon_groups:
            return {}
        by_size = sorted(muon_groups, key=lambda g: g['params'][0].numel())
        picks = {}
        picks[id(by_size[-1])] = f"largest_{list(by_size[-1]['params'][0].shape)}"
        if len(by_size) > 1:
            picks[id(by_size[0])] = f"smallest_{list(by_size[0]['params'][0].shape)}"
        return picks

    def get_sv_log_idx(self, group) -> int:
        """Return which matrix index to log for this group, or -1."""
        if self._log_groups is None:
            return -1
        if id(group) in self._log_groups:
            return 0
        return -1

    def record(self, step: int, group, sv_result: dict | None, wandb_run=None):
        """Record SV result from muon_step_with_sv_logging."""
        if sv_result is None or self._log_groups is None:
            return
        gid = id(group)
        if gid not in self._log_groups:
            return

        layer_id = self._log_groups[gid]
        shape = list(group['params'][0].shape)

        for stage, stats in sv_result.items():
            if stage == "sigma_hat":
                continue  # scalar, not a stage dict
            entry = {
                "step": step,
                "variant": self.variant,
                "stage": stage,
                "layer_id": layer_id,
                "shape": shape,
                **stats,
            }
            if stage == "blended" and "sigma_hat" in sv_result:
                entry["sigma_hat"] = sv_result["sigma_hat"]

            # Write JSONL
            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")

            # Print summary
            print0(
                f"[SV step={step}] {self.variant}/{stage} {layer_id} {shape} | "
                f"sigma_max={stats['sigma_max']:.4e} | "
                f"p05={stats['p05']:.4f} p25={stats['p25']:.4f} "
                f"p50={stats['p50']:.4f} p75={stats['p75']:.4f} p95={stats['p95']:.4f}"
            )

            # wandb logging
            if wandb_run is not None:
                prefix = f"sv/{layer_id}/{stage}"
                wandb_run.log({
                    "step": step,
                    f"{prefix}/sigma_max": stats["sigma_max"],
                    f"{prefix}/p05": stats["p05"],
                    f"{prefix}/p25": stats["p25"],
                    f"{prefix}/p50": stats["p50"],
                    f"{prefix}/p75": stats["p75"],
                    f"{prefix}/p95": stats["p95"],
                })


# =============================================================================
# Subclassed optimizers with SV logging support
# =============================================================================

class MuonAdamW_SVLogging(MuonAdamW_SpectralPadded):
    """Single-GPU spectral-padded MuonAdamW with optional SV logging."""

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups)
        self.sv_logger: SVLogger | None = None
        self._current_step = -1
        self._wandb_run = None

    def _step_muon(self, group: dict) -> None:
        # If not logging this step, use the fast compiled path
        if self.sv_logger is None or not self.sv_logger.should_log(self._current_step):
            return super()._step_muon(group)

        sv_log_idx = self.sv_logger.get_sv_log_idx(group)
        if sv_log_idx < 0:
            return super()._step_muon(group)

        # Slow uncompiled path with SV diagnostics
        params: list[Tensor] = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        self._muon_pad_ratio_t.fill_(group.get("pad_ratio", 0.0))

        sv_result = muon_step_with_sv_logging(
            stacked_grads, stacked_params,
            state["momentum_buffer"], state["second_momentum_buffer"],
            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
            self._muon_pad_ratio_t,
            int(group.get("power_iters", 1)), int(group.get("power_batch", 4)),
            group["ns_steps"], red_dim,
            sv_log_idx=sv_log_idx,
        )

        if sv_result is not None:
            self.sv_logger.record(self._current_step, group, sv_result, self._wandb_run)

        torch._foreach_copy_(params, list(stacked_params.unbind(0)))


class DistMuonAdamW_SVLogging(DistMuonAdamW_SpectralPadded):
    """Distributed spectral-padded MuonAdamW with optional SV logging."""

    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups)
        self.sv_logger: SVLogger | None = None
        self._current_step = -1
        self._wandb_run = None

    def _compute_muon(self, group, info, gather_list, rank):
        import torch.distributed as dist

        # If not logging this step or not rank 0, use fast path
        if (self.sv_logger is None
                or not self.sv_logger.should_log(self._current_step)
                or rank != 0
                or self.sv_logger.get_sv_log_idx(group) < 0):
            return super()._compute_muon(group, info, gather_list, rank)

        # Slow uncompiled path with SV diagnostics (rank 0 only)
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
            self._muon_pad_ratio_t.fill_(group.get("pad_ratio", 0.0))

            sv_log_idx = self.sv_logger.get_sv_log_idx(group)

            sv_result = muon_step_with_sv_logging(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                self._muon_pad_ratio_t,
                int(group.get("power_iters", 1)), int(group.get("power_batch", 4)),
                group["ns_steps"], red_dim,
                sv_log_idx=sv_log_idx,
            )

            if sv_result is not None:
                self.sv_logger.record(self._current_step, group, sv_result, self._wandb_run)

            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

# =============================================================================
# Monkey-patch GPT.setup_optimizer
# =============================================================================

def make_setup_optimizer_spectralpadded(pad_ratio: float, power_iters: int, power_batch: int):
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
        print0(f"Scaling the LR for the AdamW parameters 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")

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
                pad_ratio=pad_ratio, power_iters=power_iters, power_batch=power_batch,
            ))

        Factory = DistMuonAdamW_SVLogging if ddp else MuonAdamW_SVLogging
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    return setup_optimizer


GPT.setup_optimizer = make_setup_optimizer_spectralpadded(PAD_RATIO, POWER_ITERS, POWER_BATCH)

# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Spectral-padded Muon experiment")
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
parser.add_argument("--core-metric-every", type=int, default=-1)
parser.add_argument("--sample-every", type=int, default=-1)
parser.add_argument("--save-every", type=int, default=-1)
parser.add_argument("--results-dir", type=str, default="spectralpadded_results")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
user_config = vars(args).copy()

# =============================================================================
# Run tag
# =============================================================================

if PAD_RATIO > 0:
    run_tag = f"spectralpad_gamma{PAD_RATIO}_pi{POWER_ITERS}_pb{POWER_BATCH}"
else:
    run_tag = "baseline"

# =============================================================================
# Compute init
# =============================================================================

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
# Fix TF32 legacy/new API conflict that causes torch.compile inductor to crash.
# compute_init sets torch.backends.fp32_precision = "tf32" (new API), but inductor
# checks torch.backends.cuda.matmul.allow_tf32 (legacy API). Set both to be safe.
if device_type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
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
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-spectralpadded-muon", name=f"{args.run}_{run_tag}",
    config={**user_config, "pad_ratio": PAD_RATIO, "power_iters": POWER_ITERS, "power_batch": POWER_BATCH},
)

if HAS_FA3:
    print0("Using Flash Attention 3")
else:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")

# =============================================================================
# Seed
# =============================================================================

torch.manual_seed(args.seed)

# =============================================================================
# Tokenizer
# =============================================================================

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# =============================================================================
# Model
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
# Compile
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
# SV logging setup
# =============================================================================

results_dir = os.path.join(args.results_dir, run_tag)
if master_process:
    os.makedirs(results_dir, exist_ok=True)

sv_logger = SVLogger(
    log_every=SV_LOG_EVERY if SV_LOG else 0,
    variant=run_tag,
    log_dir=results_dir if master_process else None,
)

if SV_LOG:
    optimizer.sv_logger = sv_logger
    sv_logger._log_groups = sv_logger.pick_log_groups(optimizer)
    print0(f"SV logging groups: {sv_logger._log_groups}")

# =============================================================================
# DataLoader
# =============================================================================

train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device,
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device,
)
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
    print0(f"Calculated iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated iterations from target data:param ratio: {num_iterations:,}")
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

val_loss_log = []

# =============================================================================
# Training loop
# =============================================================================

step = 0
val_bpb = None
min_val_bpb = float("inf")
smooth_train_loss = 0
total_training_time = 0

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Gradient accumulation steps: {grad_accum_steps}")

while True:
    last_step = step == num_iterations

    # Evaluate val bpb
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model), autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"[{run_tag}] Step {step:05d} | val_bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        val_loss_log.append({"step": step, "val_bpb": val_bpb})
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
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay

    # Tell the optimizer what step we're on (for SV logging decisions)
    optimizer._current_step = step
    optimizer._wandb_run = wandb_run

    optimizer.step()
    model.zero_grad(set_to_none=True)
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
    if step > 10:
        total_training_time += dt
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_str = f" | eta: {remaining_steps * avg_time_per_step / 60:.1f}m"
    else:
        eta_str = ""

    if step % 50 == 0:
        print0(f"[{run_tag}] step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | dt: {dt*1000:.2f}ms | tok/s: {tok_per_sec:,}{eta_str}")

    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
        })

    first_step_of_run = step == 0
    step += 1

    if first_step_of_run:
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
        "pad_ratio": PAD_RATIO,
        "power_iters": POWER_ITERS,
        "power_batch": POWER_BATCH,
        "sv_log": SV_LOG,
        "sv_log_every": SV_LOG_EVERY,
        "num_iterations": num_iterations,
        "total_batch_size": total_batch_size,
        "min_val_bpb": min_val_bpb,
        "val_loss_log": val_loss_log,
        "user_config": user_config,
    }
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print0(f"Results saved to {results_file}")

print0(f"[{run_tag}] Done. Min val bpb: {min_val_bpb:.6f}")
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")

wandb_run.finish()
compute_cleanup()
