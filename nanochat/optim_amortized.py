"""
Amortized NS optimizer: full Newton-Schulz every N steps, cached Q on others.

On full steps (step % recompute_interval == 0):
    Q = polar_express(G)   # 15 matmuls (5×quintic)
    Q_cached = Q.clone()

On cheap steps:
    Q = Q_cached           # 0 matmuls
    if eta > 0:            # FTRL correction with current momentum-smoothed G
        nuclear_norm = (Q * G_scaled).sum()
        s_bar = nuclear_norm / sqrt(min(m,n))
        Q = (1 - eta * s_bar) * Q + eta * G_scaled
    Q = Q * (sqrt(min(m,n)) / Q.norm())   # renormalize

Modes (encoded in update_mode string passed to the muon param group):
  amort_<N>_e0    - interval N, eta=0.0 (plain cached Q, no correction)
  amort_<N>_e01   - interval N, eta=0.1 (FTRL-corrected Q)

Both always apply variance reduction (NorMuon) and cautious weight decay,
identical to the standard Muon path.

Wall-clock timing is logged separately for full vs. cheap steps via
state["full_step_time"] / state["cheap_step_time"] accumulators on the
first param of each muon group. ablation_train.py reads these to log speedup.
"""

import time
import math
import torch
import torch.distributed as dist
from torch import Tensor
from nanochat.optim import adamw_step_fused, polar_express_coeffs

# ---------------------------------------------------------------------------
# Compiled kernels
# ---------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
def _full_step_kernel(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    red_dim: int,
) -> Tensor:
    """Full Muon step (5×quintic NS). Returns Q for caching."""
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    Q = X

    beta2 = beta2_t.to(Q.dtype)
    v_mean = Q.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = Q.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g_update = Q * final_scale.to(Q.dtype)

    lr = lr_t.to(g_update.dtype)
    wd = wd_t.to(g_update.dtype)
    mask = (g_update * stacked_params) >= 0
    stacked_params.sub_(lr * g_update + lr * wd * stacked_params * mask)

    return Q  # caller clones and stores in Q_cached


def _make_cheap_kernel(eta: float):
    """Build a cheap step kernel for a fixed eta value."""
    _eta = eta

    @torch.compile(dynamic=False, fullgraph=True)
    def _cheap_step(
        stacked_grads: Tensor,
        stacked_params: Tensor,
        momentum_buffer: Tensor,
        second_momentum_buffer: Tensor,
        Q_cached: Tensor,
        momentum_t: Tensor,
        lr_t: Tensor,
        wd_t: Tensor,
        beta2_t: Tensor,
        red_dim: int,
    ) -> None:
        momentum = momentum_t.to(stacked_grads.dtype)
        momentum_buffer.lerp_(stacked_grads, 1 - momentum)
        g = stacked_grads.lerp_(momentum_buffer, momentum)

        # G_scaled: normalized momentum gradient (same normalization as in full step)
        G_scaled = g.bfloat16()
        G_scaled = G_scaled / (G_scaled.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)

        Q = Q_cached.to(G_scaled.dtype)

        if _eta > 0.0:
            m, n = g.size(-2), g.size(-1)
            sqrt_r = float(min(m, n)) ** 0.5
            nuclear_norm = (Q * G_scaled).sum(dim=(-2, -1), keepdim=True)
            s_bar = nuclear_norm / sqrt_r
            Q = (1.0 - _eta * s_bar) * Q + _eta * G_scaled
            Q = Q * (sqrt_r / Q.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6))

        beta2 = beta2_t.to(Q.dtype)
        v_mean = Q.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = Q.size(red_dim)
        v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
        v_norm = v_norm_sq.sqrt()
        second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
        step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
        final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
        g_update = Q * final_scale.to(Q.dtype)

        lr = lr_t.to(g_update.dtype)
        wd = wd_t.to(g_update.dtype)
        mask = (g_update * stacked_params) >= 0
        stacked_params.sub_(lr * g_update + lr * wd * stacked_params * mask)

    return _cheap_step


# Cache compiled cheap kernels keyed by eta value
_cheap_kernels: dict[float, object] = {}


def _get_cheap_kernel(eta: float):
    if eta not in _cheap_kernels:
        _cheap_kernels[eta] = _make_cheap_kernel(eta)
    return _cheap_kernels[eta]


# ---------------------------------------------------------------------------
# Mode parsing
# ---------------------------------------------------------------------------

def _parse_mode(mode: str) -> tuple[int, float]:
    """
    Parse "amort_<N>_e0" → (N, 0.0)
         "amort_<N>_e01" → (N, 0.1)
    Supported eta encodings: e0→0.0, e01→0.1
    """
    _ETA_MAP = {
        "e0":  0.0,
        "e01": 0.1,
    }
    parts = mode.split("_")
    if len(parts) != 3 or parts[0] != "amort":
        raise ValueError(f"Unknown amortized mode '{mode}'. Expected 'amort_<N>_e0' or 'amort_<N>_e01'.")
    try:
        interval = int(parts[1])
    except ValueError:
        raise ValueError(f"Bad interval in mode '{mode}'")
    eta_key = parts[2]
    if eta_key not in _ETA_MAP:
        raise ValueError(f"Unknown eta key '{eta_key}' in mode '{mode}'. Supported: {list(_ETA_MAP)}")
    return interval, _ETA_MAP[eta_key]


AMORT_MODES = [
    f"amort_{n}_{e}"
    for n in [2, 3, 4, 8]
    for e in ["e0", "e01"]
] + ["amort_8_e01"]  # deduped below


# Deduplicate while preserving order
_seen = set()
AMORT_MODES = [m for m in AMORT_MODES if not (m in _seen or _seen.add(m))]


# ---------------------------------------------------------------------------
# Single-GPU optimizer
# ---------------------------------------------------------------------------

class AmortizedMuonAdamW(torch.optim.Optimizer):
    """
    Amortized NS optimizer (single GPU).

    Each muon param group must include:
        update_mode: str   - e.g. "amort_4_e01"

    The global step counter is tracked per-group in the state of params[0].
    Q_cached is stored per-group (stacked over all params in the group).
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t     = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t       = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t       = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        for p in group['params']:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p, p.grad, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        params: list[Tensor] = group['params']
        if not params:
            return

        mode = group.get('update_mode', 'amort_2_e01')
        interval, eta = _parse_mode(mode)
        cheap_fn = _get_cheap_kernel(eta)

        p0 = params[0]
        state = self.state[p0]
        num_params = len(params)
        shape, device, dtype = p0.shape, p0.device, p0.dtype

        # Lazy init
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            ss = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(ss, dtype=dtype, device=device)
        if "Q_cached" not in state:
            state["Q_cached"] = torch.zeros(num_params, *shape, dtype=torch.bfloat16, device=device)
        if "opt_step" not in state:
            state["opt_step"] = 0
        if "full_step_time" not in state:
            state["full_step_time"] = 0.0
            state["cheap_step_time"] = 0.0
            state["full_step_count"] = 0
            state["cheap_step_count"] = 0

        red_dim = -1 if shape[-2] >= shape[-1] else -2
        opt_step = state["opt_step"]
        is_full = (opt_step % interval == 0)

        stacked_grads  = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        torch.cuda.synchronize() if stacked_grads.is_cuda else None
        t0 = time.perf_counter()

        if is_full:
            Q = _full_step_kernel(
                stacked_grads, stacked_params,
                state["momentum_buffer"], state["second_momentum_buffer"],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                red_dim,
            )
            state["Q_cached"].copy_(Q.detach())
        else:
            cheap_fn(
                stacked_grads, stacked_params,
                state["momentum_buffer"], state["second_momentum_buffer"],
                state["Q_cached"],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                red_dim,
            )

        torch.cuda.synchronize() if stacked_grads.is_cuda else None
        dt = time.perf_counter() - t0

        if is_full:
            state["full_step_time"] += dt
            state["full_step_count"] += 1
        else:
            state["cheap_step_time"] += dt
            state["cheap_step_count"] += 1

        state["opt_step"] += 1
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")


# ---------------------------------------------------------------------------
# Distributed optimizer
# ---------------------------------------------------------------------------

class DistAmortizedMuonAdamW(torch.optim.Optimizer):
    """
    Amortized NS optimizer (multi-GPU, ZeRO-2 style comms, same as DistMuonAdamW).

    Each muon param group must include:
        update_mode: str   - e.g. "amort_4_e01"

    Q_cached is stored per-rank (each rank caches only its chunk).
    The full-vs-cheap decision is synchronized across ranks: all ranks
    use the same opt_step counter so they always agree on full vs. cheap.
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t     = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t       = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t       = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group, world_size):
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group, world_size):
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded:
            stacked_grads[len(params):].zero_()
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()
        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group, info, gather_list, rank, world_size):
        for p in group['params']:
            pinfo = info['param_infos'][p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]
            p_slice = p if pinfo['is_small'] else p[rank * (p.shape[0] // world_size):(rank + 1) * (p.shape[0] // world_size)]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group, info, gather_list, rank):
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']

        mode = group.get('update_mode', 'amort_2_e01')
        interval, eta = _parse_mode(mode)
        cheap_fn = _get_cheap_kernel(eta)

        p0 = params[0]
        shape, device, dtype = p0.shape, p0.device, p0.dtype
        state = self.state[p0]

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            ss = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(ss, dtype=dtype, device=device)
        if "Q_cached" not in state:
            state["Q_cached"] = torch.zeros(chunk_size, *shape, dtype=torch.bfloat16, device=device)
        if "opt_step" not in state:
            state["opt_step"] = 0
        if "full_step_time" not in state:
            state["full_step_time"] = 0.0
            state["cheap_step_time"] = 0.0
            state["full_step_count"] = 0
            state["cheap_step_count"] = 0

        red_dim = -1 if shape[-2] >= shape[-1] else -2
        opt_step = state["opt_step"]
        is_full = (opt_step % interval == 0)

        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            torch.cuda.synchronize() if stacked_owned.is_cuda else None
            t0 = time.perf_counter()

            if is_full:
                Q = _full_step_kernel(
                    grad_chunk[:num_owned], stacked_owned,
                    state["momentum_buffer"][:num_owned],
                    state["second_momentum_buffer"][:num_owned],
                    self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                    red_dim,
                )
                state["Q_cached"][:num_owned].copy_(Q.detach())
            else:
                cheap_fn(
                    grad_chunk[:num_owned], stacked_owned,
                    state["momentum_buffer"][:num_owned],
                    state["second_momentum_buffer"][:num_owned],
                    state["Q_cached"][:num_owned],
                    self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                    red_dim,
                )

            torch.cuda.synchronize() if stacked_owned.is_cuda else None
            dt = time.perf_counter() - t0

            if is_full:
                state["full_step_time"] += dt
                state["full_step_count"] += 1
            else:
                state["cheap_step_time"] += dt
                state["cheap_step_count"] += 1

            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        state["opt_step"] += 1

        stacked_params_buf = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params_buf, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params_buf, params=params))

    def _finish_gathers(self, gather_list):
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        reduce_infos = []
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        gather_list = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        self._finish_gathers(gather_list)
