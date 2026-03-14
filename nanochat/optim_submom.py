"""
Subspace Momentum variant of Muon optimizer.

The key idea: orthogonalize each gradient BEFORE adding it to the momentum buffer,
instead of only orthogonalizing the buffer after accumulation.

Two modes:
  - "double": polar_express(G) -> accumulate -> polar_express(buf) -> update
  - "single": polar_express(G) -> accumulate -> normalize(buf) -> update (cheaper)
"""

import torch
from torch import Tensor
from nanochat.optim import (
    active_polar_express_coeffs,
    adamw_step_fused,
)


def _polar_express_inplace(X: Tensor, ns_steps: int) -> Tensor:
    """Apply polar express orthogonalization. X is modified in-place and returned."""
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if X.size(-2) > X.size(-1):  # Tall matrix
        for a, b, c in active_polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # Wide matrix
        for a, b, c in active_polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


@torch.compile(dynamic=False, fullgraph=True)
def submom_step_double(
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
) -> None:
    """
    Subspace momentum (double orthogonalization):
      1. P = polar_express(G)        -- orthogonalize gradient
      2. buf = momentum * buf + P    -- accumulate in orthogonal space
      3. update = polar_express(buf)  -- re-orthogonalize
      4. variance reduction + cautious update
    """
    # Step 1: Orthogonalize the gradient first
    P = _polar_express_inplace(stacked_grads.bfloat16(), ns_steps)

    # Step 2: Accumulate orthogonalized gradient into momentum buffer
    # momentum_buffer is in param dtype (bf16); P is bf16 from polar_express
    momentum = momentum_t.to(momentum_buffer.dtype)
    momentum_buffer.lerp_(P.to(momentum_buffer.dtype), 1 - momentum)
    # Nesterov lookahead: g = (1-m)*P + m*buf = P + m*(buf - P)
    g = P.to(momentum_buffer.dtype)
    g.lerp_(momentum_buffer, momentum)

    # Step 3: Re-orthogonalize the buffer
    g = _polar_express_inplace(g.bfloat16(), ns_steps)

    # Step 4: Variance reduction (same as standard Muon)
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


@torch.compile(dynamic=False, fullgraph=True)
def submom_step_single(
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
) -> None:
    """
    Subspace momentum (single orthogonalization, cheaper):
      1. P = polar_express(G)        -- orthogonalize gradient
      2. buf = momentum * buf + P    -- accumulate in orthogonal space
      3. update = buf / ||buf||       -- just normalize, skip re-orthogonalization
      4. variance reduction + cautious update
    """
    # Step 1: Orthogonalize the gradient first
    P = _polar_express_inplace(stacked_grads.bfloat16(), ns_steps)

    # Step 2: Accumulate orthogonalized gradient into momentum buffer
    momentum = momentum_t.to(momentum_buffer.dtype)
    momentum_buffer.lerp_(P.to(momentum_buffer.dtype), 1 - momentum)
    # Nesterov lookahead
    g = P.to(momentum_buffer.dtype)
    g.lerp_(momentum_buffer, momentum)

    # Step 3: Just normalize (cheap alternative to polar_express)
    g = g / (g.norm(dim=(-2, -1), keepdim=True) + 1e-6)

    # Step 4: Variance reduction (same as standard Muon)
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
