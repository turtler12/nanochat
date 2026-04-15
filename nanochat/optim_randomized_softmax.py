"""
Randomized-SVD SoftmaxMuon optimizer: replaces full SVD with a randomized
rank-k approximation for the softmax reweighting, and uses standard Muon
(Newton-Schulz / Polar Express) for the remaining directions.

Update formula:
  U_k, S_k, Vh_k = randomized_svd(M, k, n_power_iter)
  p_k = softmax(eta * S_k)                        # k-dim
  tau_k = min(m, n) * p_k.sum()                    # rescale budget
  update_topk = U_k @ diag(tau_k * p_k / p_k.sum()) @ Vh_k
  Q_full = newton_schulz(M)
  proj = U_k @ (U_k.T @ Q_full)
  update_rest = (Q_full - proj) * (1 - p_k.sum() / min(m, n))
  update = update_topk + update_rest
  update = update * (sqrt(r) / update.norm())      # normalize
"""

import torch
import torch.distributed as dist
from torch import Tensor

from nanochat.optim import adamw_step_fused, polar_express_coeffs


def randomized_svd(M: Tensor, k: int = 64, n_power_iter: int = 2):
    """Randomized SVD computing top-k singular triplets."""
    m, n = M.shape
    Omega = torch.randn(n, k, device=M.device, dtype=M.dtype)
    Y = M @ Omega
    for _ in range(n_power_iter):
        Y = M @ (M.T @ Y)
    Q_rand, _ = torch.linalg.qr(Y)
    B = Q_rand.T @ M
    U_s, S_s, Vh_s = torch.linalg.svd(B, full_matrices=False)
    U_approx = Q_rand @ U_s
    return U_approx, S_s, Vh_s


def polar_express(X: Tensor, ns_steps: int = 5) -> Tensor:
    """Newton-Schulz / Polar Express orthogonalization."""
    X = X.bfloat16()
    X = X / (X.norm(dim=(-1, -2), keepdim=True) * 1.02 + 1e-6)
    if X.size(-2) > X.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


def randomized_softmax_muon_step(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    momentum_val: float,
    lr: float,
    wd: float,
    eta: float,
    k: int,
    n_power_iter: int,
    ns_steps: int,
) -> None:
    """Randomized-SVD SoftmaxMuon step for a stack of same-shape 2D weight matrices."""
    K, m, n = stacked_grads.shape
    tau = min(m, n)

    # Nesterov momentum
    momentum_buffer.lerp_(stacked_grads, 1 - momentum_val)
    g = stacked_grads.lerp_(momentum_buffer, momentum_val).clone()

    # Polar Express for full orthogonal proxy (in bfloat16, fast)
    Q_full = polar_express(g, ns_steps=ns_steps).float()

    for i in range(K):
        grad_i = g[i].float()
        param_i = stacked_params[i]
        q_i = Q_full[i]

        # Randomized top-k SVD
        actual_k = min(k, min(m, n))
        U_k, S_k, Vh_k = randomized_svd(grad_i, k=actual_k, n_power_iter=n_power_iter)

        # Softmax reweighting of top-k singular values
        p_k = torch.softmax(eta * S_k, dim=0)
        tau_k = tau * p_k.sum()

        # Top-k update with softmax weighting
        update_topk = (U_k * (tau_k * p_k / p_k.sum()).unsqueeze(0)) @ Vh_k

        # Remaining directions: standard Muon on complement
        proj = U_k @ (U_k.T @ q_i)
        update_rest = (q_i - proj) * (1 - p_k.sum().item() / tau)

        # Combined update, normalized
        update = update_topk + update_rest
        r = min(m, n)
        update = update * (r ** 0.5 / (update.norm() + 1e-8))

        # Cautious weight decay + parameter update
        update = update.to(param_i.dtype)
        mask = (update * param_i) >= 0
        param_i.sub_(lr * update + lr * wd * param_i * mask)
        del update, mask, U_k, S_k, Vh_k, p_k, update_topk, proj, update_rest


class RandomizedSoftmaxMuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Randomized-SVD SoftmaxMuon for 2D matrix params, AdamW for others.
    Single GPU version.
    """
    def __init__(self, param_groups: list[dict], eta: float = 1.0, k: int = 64, n_power_iter: int = 2):
        super().__init__(param_groups, defaults={})
        self.eta = eta
        self.k = k
        self.n_power_iter = n_power_iter
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
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5

        randomized_softmax_muon_step(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            momentum_val=group["momentum"],
            lr=lr,
            wd=group["weight_decay"],
            eta=self.eta,
            k=self.k,
            n_power_iter=self.n_power_iter,
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


class DistRandomizedSoftmaxMuonAdamW(torch.optim.Optimizer):
    """
    Distributed version: Randomized-SVD SoftmaxMuon for 2D matrix params, AdamW for others.
    """
    def __init__(self, param_groups: list[dict], eta: float = 1.0, k: int = 64, n_power_iter: int = 2):
        super().__init__(param_groups, defaults={})
        self.eta = eta
        self.k = k
        self.n_power_iter = n_power_iter
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

            randomized_softmax_muon_step(
                grad_chunk[:num_owned],
                stacked_owned,
                state["momentum_buffer"][:num_owned],
                momentum_val=group["momentum"],
                lr=lr,
                wd=group["weight_decay"],
                eta=self.eta,
                k=self.k,
                n_power_iter=self.n_power_iter,
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
