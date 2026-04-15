"""
TaylorMuon2 optimizer: second-order Taylor approximation of SoftmaxMuon.

Extends TaylorMuon with a quadratic correction term:
    Q = newton_schulz(M)                    # orthogonal factor (Muon's step)
    M_scaled = M * (sqrt(r) / ||M||_F)      # normalize M to same Frobenius norm as Q
    S2V = M @ M^T @ Q                       # ≈ U·diag(S²)·V^T (quadratic singular value info)
    quad = S2V_scaled - Q * mean(diag(S²))  # centered second-order term
    update = Q + eta * M_scaled + 0.5 * eta² * quad
    update = update * (sqrt(r) / ||update||_F)  # renormalize to Muon scale

The second-order Taylor expansion of softmax(η·S)_i around η=0:
    1/r + η·(S_i - S̄)/r + η²·[(S_i - S̄)² - Var(S)] / (2r) + ...

Extra cost vs TaylorMuon: two matrix multiplications per layer (M·M^T and result·Q).
Zero SVD calls, fully torch.compile-able.
"""

import torch
import torch.distributed as dist
from torch import Tensor

from nanochat.optim import adamw_step_fused, polar_express_coeffs


@torch.compile(dynamic=False, fullgraph=True)
def taylor2_muon_step_fused(
    stacked_grads: Tensor,          # (K, m, n)
    stacked_params: Tensor,         # (K, m, n)
    momentum_buffer: Tensor,        # (K, m, n)
    second_momentum_buffer: Tensor, # (K, m, 1) or (K, 1, n)
    momentum_t: Tensor,             # () 0-D CPU tensor
    lr_t: Tensor,                   # () 0-D CPU tensor
    wd_t: Tensor,                   # () 0-D CPU tensor
    beta2_t: Tensor,                # () 0-D CPU tensor
    eta_t: Tensor,                  # () 0-D CPU tensor
    ns_steps: int,
    red_dim: int,
) -> None:
    """
    Fused TaylorMuon2 step: momentum → polar_express → quadratic_taylor_correction → variance_reduction → update.
    Fully compilable — no SVD.
    """
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Save pre-orthogonalization gradient (for Taylor correction)
    M = g.clone()

    # Polar Express orthogonalization → Q
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):  # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # Wide matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    Q = X  # orthogonal factor, Frobenius norm ≈ sqrt(min(m,n))

    # Taylor correction terms
    eta = eta_t.to(g.dtype)
    r = min(g.size(-2), g.size(-1))
    r_sqrt = r ** 0.5
    M_norm = M.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    M_scaled = M * (r_sqrt / M_norm)  # same Frobenius norm as Q

    # First-order term: η * M_scaled (proportional to U·diag(S - S̄)·Vᵀ)

    # Second-order term: need U·diag(S²)·Vᵀ
    # M·Mᵀ·Q = U·S·Vᵀ·V·S·Uᵀ·U·Vᵀ = U·S²·Vᵀ
    # Rewrite as M_scaled @ (M_scaled.mT @ Q) to avoid large (K,m,m) intermediate
    # and keep matmul shapes consistent with Newton-Schulz (compiler-friendly).
    # M_scaled.mT @ Q = V·S_scaled·Uᵀ·U·Vᵀ = V·S_scaled·Vᵀ  (K, n, n)
    # M_scaled @ (V·S_scaled·Vᵀ) = U·S_scaled²·Vᵀ             (K, m, n)
    S2V = M_scaled @ (M_scaled.mT @ Q)  # (K, m, n) — U·diag(S_scaled²)·Vᵀ

    # mean(S_scaled²) = ||M_scaled||_F² / r = r / r = 1 (since ||M_scaled||_F = sqrt(r))
    # So the centered term is: S2V - Q * mean(S²) = S2V - Q
    # The second-order softmax Taylor term is: [(S_i - S̄)² - Var(S)] / (2r)
    # = [S_i² - 2·S̄·S_i + S̄² - (mean(S²) - S̄²)] / (2r)
    # = [S_i² - 2·S̄·S_i + 2·S̄² - mean(S²)] / (2r)
    # With S̄ ≈ mean of scaled singular values. For the first-order term,
    # U·diag(S - S̄)·Vᵀ = M_scaled - Q·S̄, and S̄ = trace(Qᵀ·M_scaled)/r.
    #
    # Full second-order: quad = (S2V - 2·S̄·M_scaled + (2·S̄² - mean(S²))·Q) / r
    # But S̄ = trace(Qᵀ·M_scaled)/r and mean(S²) = trace(M_scaledᵀ·M_scaled)/r = 1
    S_bar = (Q * M_scaled).sum(dim=(-2, -1), keepdim=True) / r  # mean singular value (scaled)
    # mean(S²) = ||M_scaled||²/r = r/r = 1
    mean_S2 = 1.0

    quad = (S2V - 2 * S_bar * M_scaled + (2 * S_bar * S_bar - mean_S2) * Q) / r

    g = Q + eta * M_scaled + 0.5 * eta * eta * quad

    # Renormalize to Muon's Frobenius norm scale
    g_norm = g.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    g = g * (r_sqrt / g_norm)

    # Variance reduction (same as Muon)
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


class Taylor2MuonAdamW(torch.optim.Optimizer):
    """Single-GPU TaylorMuon2 (quadratic) + AdamW."""
    def __init__(self, param_groups: list[dict], eta: float = 0.5):
        super().__init__(param_groups, defaults={})
        self.eta = eta
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_eta_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

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
        p = params[0]
        state = self.state[p]
        shape = p.shape

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=p.dtype, device=p.device)
        if "second_momentum_buffer" not in state:
            state_shape = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=p.dtype, device=p.device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        self._muon_eta_t.fill_(self.eta)
        taylor2_muon_step_fused(
            stacked_grads, stacked_params,
            state["momentum_buffer"], state["second_momentum_buffer"],
            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
            self._muon_eta_t, group["ns_steps"], red_dim,
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


class DistTaylor2MuonAdamW(torch.optim.Optimizer):
    """Distributed TaylorMuon2 (quadratic) + AdamW."""
    def __init__(self, param_groups: list[dict], eta: float = 0.5):
        super().__init__(param_groups, defaults={})
        self.eta = eta
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_eta_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
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

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

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

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
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
            self._muon_eta_t.fill_(self.eta)
            taylor2_muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                self._muon_eta_t, group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)

        self._finish_gathers(gather_list)
