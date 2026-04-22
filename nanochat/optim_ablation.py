"""
Ablation optimizer: multiple update modes to test how much Q quality matters.

Modes (controlled by update_mode in the muon param group):
  muon          - standard Muon: 5×quintic Polar Express (15 matmuls)
  fast4         - optimized 4×quintic σ_lb=0.02 (12 matmuls)
  ns3           - 3 Polar Express iterations (9 matmuls)
  ns2           - 2 Polar Express iterations (6 matmuls)
  ns1           - 1 Polar Express iteration (3 matmuls)
  ns0_normalized - no NS: G / ||G||_F * sqrt(min(m,n))  (0 matmuls)
  ns0_sign      - no NS: sign(G) * sqrt(min(m,n)) / sqrt(m*n)  (0 matmuls)
"""

import torch
import torch.distributed as dist
from torch import Tensor
from nanochat.optim import adamw_step_fused, polar_express_coeffs

# 4×quintic coefficients optimized for σ_lb=0.02
fast4_coeffs = [
    (3.4445, -4.7750,  2.0315),
    (3.0941, -3.6288,  1.2070),
    (2.7418, -3.0511,  0.9272),
    (2.4258, -2.6320,  0.7884),
]

# Map mode → (coeffs, ns_steps).  ns0_* handled separately.
_NS_MODES = {
    "muon":  (polar_express_coeffs, 5),
    "fast4": (fast4_coeffs,         4),
    "ns3":   (polar_express_coeffs, 3),
    "ns2":   (polar_express_coeffs, 2),
    "ns1":   (polar_express_coeffs, 1),
}
UPDATE_MODES = list(_NS_MODES.keys()) + ["ns0_normalized", "ns0_sign"]


# One compiled kernel per NS mode (compile doesn't like dynamic coeff lists)
def _make_ns_kernel(coeffs, ns_steps):
    _c = coeffs[:ns_steps]

    @torch.compile(dynamic=False, fullgraph=True)
    def _step(
        stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
        momentum_t, lr_t, wd_t, beta2_t, red_dim: int,
    ) -> None:
        momentum = momentum_t.to(stacked_grads.dtype)
        momentum_buffer.lerp_(stacked_grads, 1 - momentum)
        g = stacked_grads.lerp_(momentum_buffer, momentum)

        X = g.bfloat16()
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
        if g.size(-2) > g.size(-1):
            for a, b, c in _c:
                A = X.mT @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
        else:
            for a, b, c in _c:
                A = X @ X.mT
                B = b * A + c * (A @ A)
                X = a * X + B @ X
        g = X

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

        lr = lr_t.to(g.dtype)
        wd = wd_t.to(g.dtype)
        mask = (g * stacked_params) >= 0
        stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

    return _step


@torch.compile(dynamic=False, fullgraph=True)
def _ns0_normalized_step(
    stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
    momentum_t, lr_t, wd_t, beta2_t, red_dim: int,
) -> None:
    """No NS: normalized gradient scaled to match Muon's nuclear norm."""
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    m, n = g.size(-2), g.size(-1)
    scale = float(min(m, n)) ** 0.5
    g = g.bfloat16()
    g = g / (g.norm(dim=(-2, -1), keepdim=True) + 1e-6) * scale

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

    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


@torch.compile(dynamic=False, fullgraph=True)
def _ns0_sign_step(
    stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
    momentum_t, lr_t, wd_t, beta2_t, red_dim: int,
) -> None:
    """No NS: elementwise sign, scaled so nuclear norm matches Muon's."""
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    m, n = g.size(-2), g.size(-1)
    # sign(G) has nuclear norm ≈ min(m,n); divide by sqrt(m*n) and multiply by
    # sqrt(min(m,n)) to match Muon's scale: net factor = sqrt(min(m,n)/(m*n))
    scale = (float(min(m, n)) / float(m * n)) ** 0.5
    g = g.bfloat16().sign() * scale

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

    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


def _get_step_fn(mode: str):
    if mode in _NS_MODES:
        coeffs, ns_steps = _NS_MODES[mode]
        return _make_ns_kernel(coeffs, ns_steps)
    elif mode == "ns0_normalized":
        return _ns0_normalized_step
    elif mode == "ns0_sign":
        return _ns0_sign_step
    else:
        raise ValueError(f"Unknown update_mode '{mode}'. Choose from: {UPDATE_MODES}")


class AblationMuonAdamW(torch.optim.Optimizer):
    """
    Ablation optimizer: same as MuonAdamW but the update mode is configurable
    per muon param group via group['update_mode'].
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t     = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t  = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t  = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t     = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Cache compiled kernels per mode string
        self._step_fns: dict[str, object] = {}

    def _get_fn(self, mode: str):
        if mode not in self._step_fns:
            self._step_fns[mode] = _get_step_fn(mode)
        return self._step_fns[mode]

    def _step_adamw(self, group: dict) -> None:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
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
                p, grad, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        params: list[Tensor] = group['params']
        if not params:
            return
        mode = group.get('update_mode', 'muon')
        step_fn = self._get_fn(mode)

        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            ss = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(ss, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads  = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        step_fn(
            stacked_grads, stacked_params,
            state["momentum_buffer"], state["second_momentum_buffer"],
            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
            red_dim,
        )
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


class DistAblationMuonAdamW(torch.optim.Optimizer):
    """Distributed version of AblationMuonAdamW (ZeRO-2 style, same comm pattern as DistMuonAdamW)."""
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t     = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t  = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t  = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t    = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t     = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t      = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t   = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._step_fns: dict[str, object] = {}

    def _get_fn(self, mode: str):
        if mode not in self._step_fns:
            self._step_fns[mode] = _get_step_fn(mode)
        return self._step_fns[mode]

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
        mode = group.get('update_mode', 'muon')
        step_fn = self._get_fn(mode)

        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            ss = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(ss, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            step_fn(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)
        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

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
