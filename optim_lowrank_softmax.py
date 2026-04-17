"""
HybridSoftmaxMuon optimizer: Q + rank-k additive softmax correction.

Combines Newton-Schulz orthogonalization (all directions, flat weights) with a
low-rank SVD correction that nudges the top-k singular directions toward their
softmax targets — without discarding any directions.

For each gradient matrix M (after Nesterov momentum):
  1. Q = newton_schulz(M)                     # all r directions, flat weights
  2. U_k, S_k, V_k = svd_lowrank(M, q=k)     # top-k singular structure, O(k·mn)
  3. p_k = softmax(eta * S_k)                 # softmax weights for top-k
  4. delta = p_k - 1/k                        # correction vs flat weights
  5. correction = sqrt(r) * U_k @ diag(delta) @ V_k^T
  6. update = Q + correction                  # all directions preserved, top-k reweighted

Cost: Newton-Schulz (same as Muon) + one cheap low-rank SVD per matrix per step.

Why this works where pure low-rank didn't:
  - Low-rank: only k directions, discards (r-k) → 99%+ of gradient signal lost
  - Hybrid: all r directions via Q, plus targeted correction on top-k
"""

import math
import torch
import torch.distributed as dist
from torch import Tensor

from nanochat.optim import adamw_step_fused, polar_express_coeffs


def hybrid_softmax_muon_step(
    stacked_grads: Tensor,          # (K, m, n)
    stacked_params: Tensor,         # (K, m, n)
    momentum_buffer: Tensor,        # (K, m, n)
    momentum_val: float,
    lr: float,
    wd: float,
    eta: float,
    rank_k: int,
    ns_steps: int,
) -> None:
    """
    HybridSoftmaxMuon step for a stack of same-shape 2D weight matrices.
    Cannot be torch.compiled due to svd_lowrank.
    """
    K, m, n = stacked_grads.shape
    r = min(m, n)
    r_sqrt = math.sqrt(r)

    # Nesterov momentum
    momentum_buffer.lerp_(stacked_grads, 1 - momentum_val)
    g = stacked_grads.lerp_(momentum_buffer, momentum_val)

    # Newton-Schulz orthogonalization → Q (all directions, flat weights)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if m > n:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    Q = X  # (K, m, n), Frobenius norm ≈ sqrt(r)

    # Process each matrix for the low-rank correction
    for i in range(K):
        g_i = g[i].float()     # (m, n)
        Q_i = Q[i]             # (m, n) in bf16
        param_i = stacked_params[i]  # (m, n)

        # Top-k singular structure via randomized SVD
        U_k, S_k, V_k = torch.svd_lowrank(g_i, q=rank_k)  # U_k: (m, k), S_k: (k,), V_k: (n, k)

        # Softmax weights for top-k directions
        p_k = torch.softmax(eta * S_k, dim=0)  # (k,), sums to 1

        # What Q gives each direction: flat weight = 1/k of the softmax budget
        uniform = torch.full_like(p_k, 1.0 / rank_k)

        # Additive correction: push top-k from flat toward softmax target
        delta = p_k - uniform  # positive for top SVs, negative for bottom of top-k
        correction = r_sqrt * (U_k * delta.unsqueeze(0)) @ V_k.mT  # (m, n)

        # Combined update
        update = Q_i.float() + correction  # (m, n)
        del U_k, S_k, V_k, p_k, uniform, delta, correction

        # Cautious weight decay + parameter update
        update = update.to(param_i.dtype)
        mask = (update * param_i) >= 0
        param_i.sub_(lr * update + lr * wd * param_i * mask)
        del update, mask


class LowRankSoftmaxMuonAdamW(torch.optim.Optimizer):
    """Single-GPU HybridSoftmaxMuon + AdamW."""
    def __init__(self, param_groups: list[dict], eta: float = 1.0, rank_k: int = 4):
        super().__init__(param_groups, defaults={})
        self.eta = eta
        self.rank_k = rank_k
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

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

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5

        hybrid_softmax_muon_step(
            stacked_grads, stacked_params,
            state["momentum_buffer"],
            momentum_val=group["momentum"],
            lr=lr, wd=group["weight_decay"],
            eta=self.eta, rank_k=self.rank_k,
            ns_steps=group.get("ns_steps", 5),
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


class DistLowRankSoftmaxMuonAdamW(torch.optim.Optimizer):
    """Distributed HybridSoftmaxMuon + AdamW."""
    def __init__(self, param_groups: list[dict], eta: float = 1.0, rank_k: int = 4):
        super().__init__(param_groups, defaults={})
        self.eta = eta
        self.rank_k = rank_k
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

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

        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5

            hybrid_softmax_muon_step(
                grad_chunk[:num_owned],
                stacked_owned,
                state["momentum_buffer"][:num_owned],
                momentum_val=group["momentum"],
                lr=lr, wd=group["weight_decay"],
                eta=self.eta, rank_k=self.rank_k,
                ns_steps=group.get("ns_steps", 5),
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
