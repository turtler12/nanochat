"""
Cached-SVD SoftmaxMuon optimizer: amortizes the cost of SVD by recomputing it
only every `svd_freq` steps.

On SVD steps (every svd_freq):
  1. Nesterov momentum -> M
  2. Full SVD: U, S, Vh = SVD(M)
  3. Cache U, S, Vh
  4. Compute update from cached values

On non-SVD steps:
  1. Nesterov momentum -> M (still updated every step)
  2. Reuse cached U, S, Vh from the last SVD step
  3. Compute update from cached values

Update formula (every step):
  p = softmax(eta * cached_S)
  x = tau * p
  update = cached_U @ diag(x) @ cached_Vh

When svd_freq=1, this is identical to the original SoftmaxMuon.
"""

import torch
import torch.distributed as dist
from torch import Tensor

from nanochat.optim import adamw_step_fused


def cached_softmax_muon_step(
    stacked_grads: Tensor,          # (K, m, n) - stacked gradients
    stacked_params: Tensor,         # (K, m, n) - stacked parameters
    momentum_buffer: Tensor,        # (K, m, n) - first moment buffer
    cached_U: list[Tensor | None],  # length >= K + cache_offset
    cached_S: list[Tensor | None],  # length >= K + cache_offset
    cached_Vh: list[Tensor | None], # length >= K + cache_offset
    svd_step_counter: Tensor,       # scalar, counts steps since last SVD
    svd_freq: int,
    momentum_val: float,
    lr: float,
    wd: float,
    eta: float,
    cache_offset: int = 0,          # offset into cached_U/S/Vh lists
) -> None:
    """
    Cached-SVD SoftmaxMuon step for a stack of same-shape 2D weight matrices.
    cache_offset allows indexing into larger cache lists (for distributed case).
    """
    K, m, n = stacked_grads.shape
    tau = min(m, n)

    # Nesterov momentum (always updated)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum_val)
    # Clone to get a contiguous tensor not aliasing compiled-graph outputs
    g = stacked_grads.lerp_(momentum_buffer, momentum_val).clone()

    # Check if we should recompute SVD this step
    do_svd = (svd_step_counter.item() % svd_freq == 0)

    for i in range(K):
        ci = cache_offset + i  # index into the full cache lists
        grad_i = g[i]  # (m, n)
        param_i = stacked_params[i]  # (m, n)

        if do_svd:
            # Full SVD recomputation
            U, S, Vh = torch.linalg.svd(grad_i.float(), full_matrices=False)
            cached_U[ci] = U
            cached_S[ci] = S
            cached_Vh[ci] = Vh
        else:
            # Reuse cached decomposition entirely
            U = cached_U[ci]
            S = cached_S[ci]
            Vh = cached_Vh[ci]

        # Softmax weighting of singular values
        p = torch.softmax(eta * S, dim=0)
        x = tau * p

        # Reconstruct update: U @ diag(x) @ Vh
        update = (U * x.unsqueeze(0)) @ Vh
        del p, x

        # Cautious weight decay + parameter update
        update = update.to(param_i.dtype)
        mask = (update * param_i) >= 0
        param_i.sub_(lr * update + lr * wd * param_i * mask)
        del update, mask

    # Increment counter
    svd_step_counter.add_(1)


class CachedSoftmaxMuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Cached-SVD SoftmaxMuon for 2D matrix params, AdamW for others.
    Single GPU version.
    """
    def __init__(self, param_groups: list[dict], eta: float = 1.0, svd_freq: int = 10):
        super().__init__(param_groups, defaults={})
        self.eta = eta
        self.svd_freq = svd_freq
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
            state["cached_U"] = [None] * num_params
            state["cached_S"] = [None] * num_params
            state["cached_Vh"] = [None] * num_params
            state["svd_step_counter"] = torch.tensor(0, dtype=torch.long, device="cpu")

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5

        cached_softmax_muon_step(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["cached_U"],
            state["cached_S"],
            state["cached_Vh"],
            state["svd_step_counter"],
            svd_freq=self.svd_freq,
            momentum_val=group["momentum"],
            lr=lr,
            wd=group["weight_decay"],
            eta=self.eta,
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


class DistCachedSoftmaxMuonAdamW(torch.optim.Optimizer):
    """
    Distributed version: Cached-SVD SoftmaxMuon for 2D matrix params, AdamW for others.
    """
    def __init__(self, param_groups: list[dict], eta: float = 1.0, svd_freq: int = 10):
        super().__init__(param_groups, defaults={})
        self.eta = eta
        self.svd_freq = svd_freq
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
            state["cached_U"] = [None] * chunk_size
            state["cached_S"] = [None] * chunk_size
            state["cached_Vh"] = [None] * chunk_size
            state["svd_step_counter"] = torch.tensor(0, dtype=torch.long, device="cpu")

        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            lr = group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5

            cached_softmax_muon_step(
                grad_chunk[:num_owned],
                stacked_owned,
                state["momentum_buffer"][:num_owned],
                state["cached_U"],
                state["cached_S"],
                state["cached_Vh"],
                state["svd_step_counter"],
                svd_freq=self.svd_freq,
                momentum_val=group["momentum"],
                lr=lr,
                wd=group["weight_decay"],
                eta=self.eta,
                cache_offset=0,
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
