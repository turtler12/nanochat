"""
Trace/norm-constrained Muon optimizer variants.

Three variants that maintain norm constraints from initialization:
  A) Frobenius normalization every step
  B) Nuclear norm normalization every K steps (Frobenius between)
  C) Frobenius normalization with a fixed universal target

These wrap the standard Muon optimizer step and apply post-hoc normalization.
"""

import torch
import torch.distributed as dist
from torch import Tensor

from nanochat.optim import (
    adamw_step_fused,
    muon_step_fused,
)


# =============================================================================
# Norm constraint modes
# =============================================================================

class NormConstraint:
    """Base class for norm constraints applied after each Muon step."""

    def __init__(self):
        # Map: param_name -> target norm value (recorded at init)
        self.targets = {}

    def record_init_norms(self, model):
        """Record target norms from initialization for all Muon params."""
        for name, param in model.named_parameters():
            if param.ndim != 2:
                continue
            if 'wte' in name or 'lm_head' in name or 'value_embeds' in name:
                continue
            self._record_one(name, param)

    def _record_one(self, name, param):
        raise NotImplementedError

    def apply(self, model, step):
        """Apply the constraint after an optimizer step."""
        raise NotImplementedError


class FrobeniusConstraint(NormConstraint):
    """Variant A: Rescale W to match init Frobenius norm every step."""

    def _record_one(self, name, param):
        self.targets[name] = param.detach().float().norm().item()

    @torch.no_grad()
    def apply(self, model, step):
        for name, param in model.named_parameters():
            if name not in self.targets:
                continue
            target = self.targets[name]
            current = param.float().norm().item()
            if current > 1e-10:
                param.mul_(target / current)


class NuclearConstraint(NormConstraint):
    """Variant B: Nuclear norm every K steps, Frobenius between."""

    def __init__(self, K=10):
        super().__init__()
        self.K = K
        self.fro_targets = {}  # Frobenius norms at init
        self.nuc_targets = {}  # Nuclear norms at init

    def _record_one(self, name, param):
        w = param.detach().float()
        self.fro_targets[name] = w.norm().item()
        s = torch.linalg.svdvals(w)
        self.nuc_targets[name] = s.sum().item()

    def record_init_norms(self, model):
        for name, param in model.named_parameters():
            if param.ndim != 2:
                continue
            if 'wte' in name or 'lm_head' in name or 'value_embeds' in name:
                continue
            self._record_one(name, param)
        self.targets = self.nuc_targets  # for compatibility

    @torch.no_grad()
    def apply(self, model, step):
        use_nuclear = (step % self.K == 0)
        for name, param in model.named_parameters():
            if name not in self.nuc_targets:
                continue
            if use_nuclear:
                w = param.detach().float()
                s = torch.linalg.svdvals(w)
                current_nuc = s.sum().item()
                target_nuc = self.nuc_targets[name]
                if current_nuc > 1e-10:
                    param.mul_(target_nuc / current_nuc)
            else:
                # Frobenius approximation between SVD steps
                target_fro = self.fro_targets[name]
                current_fro = param.float().norm().item()
                if current_fro > 1e-10:
                    param.mul_(target_fro / current_fro)


class FixedFrobeniusConstraint(NormConstraint):
    """Variant C: Frobenius normalization with a fixed universal target (median of init norms)."""

    def __init__(self):
        super().__init__()
        self._init_norms = []
        self.fixed_target = None

    def _record_one(self, name, param):
        fro = param.detach().float().norm().item()
        self.targets[name] = fro
        self._init_norms.append(fro)

    def record_init_norms(self, model):
        super().record_init_norms(model)
        # Set target to median of all init Frobenius norms
        sorted_norms = sorted(self._init_norms)
        n = len(sorted_norms)
        if n % 2 == 0:
            self.fixed_target = (sorted_norms[n // 2 - 1] + sorted_norms[n // 2]) / 2.0
        else:
            self.fixed_target = sorted_norms[n // 2]
        print(f"[FixedFrobeniusConstraint] Median init Frobenius norm: {self.fixed_target:.6f} (from {n} layers)")

    @torch.no_grad()
    def apply(self, model, step):
        target = self.fixed_target
        for name, param in model.named_parameters():
            if name not in self.targets:
                continue
            current = param.float().norm().item()
            if current > 1e-10:
                param.mul_(target / current)


class NuclearEveryStepConstraint(NormConstraint):
    """Nuclear norm normalization every step via full SVD.

    Computes SVD every step, sums singular values (nuclear norm),
    and rescales W so nuclear norm matches init value.
    """

    def _record_one(self, name, param):
        w = param.detach().float()
        s = torch.linalg.svdvals(w)
        nuc = s.sum().item()
        if nuc < 1e-10:
            return  # skip zero-init layers
        self.targets[name] = nuc

    @torch.no_grad()
    def apply(self, model, step):
        for name, param in model.named_parameters():
            if name not in self.targets:
                continue
            w = param.detach().float()
            s = torch.linalg.svdvals(w)
            current_nuc = s.sum().item()
            target_nuc = self.targets[name]
            if current_nuc > 1e-10:
                param.mul_(target_nuc / current_nuc)


class ConditionClampConstraint(NormConstraint):
    """Option 1: Condition number clamping via SVD every K steps.

    Clamps singular values to [median/max_ratio, median*max_ratio],
    preventing any SV from getting too far from the pack.
    Norms can grow freely; only the spread is constrained.
    """

    def __init__(self, K=10, max_ratio=1.5):
        super().__init__()
        self.K = K
        self.max_ratio = max_ratio

    def _record_one(self, name, param):
        self.targets[name] = True  # just mark as tracked

    @torch.no_grad()
    def apply(self, model, step):
        if step % self.K != 0:
            return
        for name, param in model.named_parameters():
            if name not in self.targets:
                continue
            w = param.detach().float()
            U, S, Vh = torch.linalg.svd(w, full_matrices=False)
            median_sv = S.median()
            S_clamped = S.clamp(
                min=median_sv / self.max_ratio,
                max=median_sv * self.max_ratio,
            )
            W_new = U @ torch.diag(S_clamped) @ Vh
            param.copy_(W_new.to(param.dtype))


class SpectralCapConstraint(NormConstraint):
    """Option 2: Spectral normalization only (cheapest).

    Only clips sigma_max if it exceeds target (1.5x init spectral norm).
    Everything else grows freely.
    """

    def __init__(self, cap_ratio=1.5):
        super().__init__()
        self.cap_ratio = cap_ratio

    def _record_one(self, name, param):
        w = param.detach().float()
        s = torch.linalg.svdvals(w)
        init_spec = s[0].item()
        if init_spec < 1e-10:
            # Skip zero-initialized layers (e.g. projection layers init'd to zeros)
            return
        self.targets[name] = init_spec * self.cap_ratio

    @torch.no_grad()
    def apply(self, model, step):
        for name, param in model.named_parameters():
            if name not in self.targets:
                continue
            target = self.targets[name]
            w = param.detach().float()
            spec_norm = torch.linalg.svdvals(w)[0].item()
            if target > 1e-10 and spec_norm > target:
                param.mul_(target / spec_norm)


class GrowingFrobeniusConstraint(NormConstraint):
    """Option 3: Frobenius normalization with a growing target.

    target_norm(step) = init_norm * min(max_ratio, 1 + growth_rate * step)
    Allows controlled growth matching baseline Muon's observed trajectory.
    """

    def __init__(self, growth_rate=0.001, max_ratio=3.5):
        super().__init__()
        self.growth_rate = growth_rate
        self.max_ratio = max_ratio

    def _record_one(self, name, param):
        self.targets[name] = param.detach().float().norm().item()

    @torch.no_grad()
    def apply(self, model, step):
        for name, param in model.named_parameters():
            if name not in self.targets:
                continue
            init_norm = self.targets[name]
            target = init_norm * min(self.max_ratio, 1.0 + self.growth_rate * step)
            current = param.float().norm().item()
            if current > 1e-10:
                param.mul_(target / current)


class SpectralPenaltyConstraint(NormConstraint):
    """Option 4: Soft spectral regularization penalty.

    Adds lambda * (spectral_norm(W) / frobenius_per_dim(W)) to loss.
    Does not modify weights directly — instead returns a penalty term
    that must be added to the loss before backward().
    """

    def __init__(self, penalty_lambda=0.01, K=1):
        super().__init__()
        self.penalty_lambda = penalty_lambda
        self.K = K  # compute penalty every K steps

    def _record_one(self, name, param):
        self.targets[name] = True

    @torch.no_grad()
    def apply(self, model, step):
        # No-op: penalty is applied via compute_penalty() in the loss
        pass

    def compute_penalty(self, model, step):
        """Compute spectral penalty to add to loss. Call BEFORE backward()."""
        if step % self.K != 0:
            return 0.0
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name not in self.targets:
                continue
            w = param.float()
            fro = w.norm()
            ndim = min(w.shape)
            fro_per_dim = fro / (ndim ** 0.5)
            # Use power iteration for spectral norm to keep it differentiable
            # Approximate: just use the largest svdval (detached for stability,
            # but scale the penalty by the ratio)
            with torch.no_grad():
                s_max = torch.linalg.svdvals(w)[0]
            if fro_per_dim > 1e-10:
                penalty = penalty + s_max / fro_per_dim
        return self.penalty_lambda * penalty


class SpectralSharpenConstraint(NormConstraint):
    """Option 5: Spectral sharpening — amplify top SVs relative to bottom.

    Every K steps: S_new = S ** alpha (alpha > 1 sharpens).
    Accelerates the low-rank structure Muon naturally develops.
    """

    def __init__(self, K=50, alpha=1.05):
        super().__init__()
        self.K = K
        self.alpha = alpha

    def _record_one(self, name, param):
        w = param.detach().float()
        if w.norm().item() < 1e-10:
            return  # skip zero-init layers
        self.targets[name] = True

    @torch.no_grad()
    def apply(self, model, step):
        if step % self.K != 0 or step == 0:
            return
        for name, param in model.named_parameters():
            if name not in self.targets:
                continue
            w = param.detach().float()
            U, S, Vh = torch.linalg.svd(w, full_matrices=False)
            # Preserve Frobenius norm: sharpen then rescale
            fro_before = w.norm()
            S_sharp = S ** self.alpha
            W_new = U @ torch.diag(S_sharp) @ Vh
            fro_after = W_new.norm()
            if fro_after > 1e-10:
                W_new = W_new * (fro_before / fro_after)
            param.copy_(W_new.to(param.dtype))


class WarmupOnlyConstraint(NormConstraint):
    """Option 7: Frobenius constraint only during warmup, then release.

    Apply Frobenius normalization for the first `warmup_steps` steps,
    then remove constraint entirely and let norms grow freely.
    """

    def __init__(self, warmup_steps=500):
        super().__init__()
        self.warmup_steps = warmup_steps

    def _record_one(self, name, param):
        self.targets[name] = param.detach().float().norm().item()

    @torch.no_grad()
    def apply(self, model, step):
        if step >= self.warmup_steps:
            return
        for name, param in model.named_parameters():
            if name not in self.targets:
                continue
            target = self.targets[name]
            current = param.float().norm().item()
            if current > 1e-10:
                param.mul_(target / current)


# =============================================================================
# Spectral logging utilities
# =============================================================================

def compute_norm_stats(model, top_k=64):
    """Compute per-layer norm stats for all Muon params."""
    stats = {}
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        if 'wte' in name or 'lm_head' in name or 'value_embeds' in name:
            continue
        w = param.detach().float()
        try:
            s = torch.linalg.svdvals(w)
        except Exception:
            continue
        stats[name] = {
            'frobenius_norm': w.norm().item(),
            'spectral_norm': s[0].item(),
            'nuclear_norm': s.sum().item(),
            'effective_rank': (s.sum() / s[0]).item() if s[0] > 0 else 0,
            'stable_rank': (w.norm()**2 / s[0]**2).item() if s[0] > 0 else 0,
            'shape': list(w.shape),
        }
    return stats


def compute_sv_entropy(model, layer_names):
    """Compute singular value entropy for specified layers.

    Entropy = -(p * log(p)).sum() where p = S / S.sum() (normalized SV distribution).
    Higher entropy = more uniform SVs = better rank utilization.
    """
    results = {}
    for name, param in model.named_parameters():
        if name not in layer_names:
            continue
        w = param.detach().float()
        try:
            s = torch.linalg.svdvals(w)
        except Exception:
            continue
        # Normalize to probability distribution
        p = s / s.sum()
        # Clamp to avoid log(0)
        p = p.clamp(min=1e-10)
        entropy = -(p * p.log()).sum().item()
        results[name] = entropy
    return results
