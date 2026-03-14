#!/usr/bin/env python3
"""
Polynomial Schedule Search for Newton-Schulz Orthogonalization.

Goal: Find polynomial iteration schedules that achieve bf16-sufficient accuracy
(max error < 0.01 across σ ∈ [0.001, 0.98]) in fewer matmuls than Muon's
current 5 × quintic (15 matmuls).

Part 1: Coefficient optimization (CPU, scipy)
Part 2: GPU benchmark (CUDA)
Part 3: Training validation
"""

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import differential_evolution, minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

ACCURACY_THRESHOLD = 0.01
SIGMA_GRID = np.logspace(np.log10(0.001), np.log10(0.98), 2000)

# Muon's original quintic coefficients (Polar Express, 5 iterations)
MUON_QUINTIC_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# ─────────────────────────────────────────────────────────────────────────────
# Candidate configurations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolyConfig:
    name: str
    degrees: list[int]       # degree of each polynomial in the composition
    matmuls: int             # total matmuls
    savings_pct: float       # % savings vs baseline 15 matmuls
    coefficients: Optional[list[list[float]]] = None
    max_error: float = float('inf')
    optimization_time: float = 0.0
    gpu_time_ms: dict = field(default_factory=dict)
    baseline_gpu_time_ms: dict = field(default_factory=dict)
    speedup: dict = field(default_factory=dict)
    bf16_max_error: dict = field(default_factory=dict)
    training_validation: dict = field(default_factory=dict)


CONFIGS = [
    PolyConfig("4×quintic", [5,5,5,5], 12, 20.0),
    PolyConfig("3×quintic", [5,5,5], 9, 40.0),
    PolyConfig("3×septic", [7,7,7], 12, 20.0),
    PolyConfig("2×septic+1×quintic", [7,7,5], 11, 27.0),
    PolyConfig("2×nonic", [9,9], 10, 33.0),
    PolyConfig("2×nonic+1×quintic", [9,9,5], 13, 13.0),
    PolyConfig("1×deg13+1×septic", [13,7], 11, 27.0),
    PolyConfig("3×septic+1×quintic", [7,7,7,5], 15, 0.0),
    PolyConfig("4×septic", [7,7,7,7], 16, -7.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Polynomial math & optimization
# ─────────────────────────────────────────────────────────────────────────────

def degree_to_num_coeffs(degree: int) -> int:
    """A degree-(2d+1) polynomial has d+1 coefficients: a1*σ + a2*σ³ + a3*σ⁵ + ..."""
    assert degree % 2 == 1, f"Degree must be odd, got {degree}"
    d = (degree - 1) // 2
    return d + 1


def eval_single_poly(sigma: np.ndarray, coeffs: list[float]) -> np.ndarray:
    """
    Evaluate p(σ) = a₁σ + a₂σ³ + a₃σ⁵ + ...
    with constraint p(1) = 1: a₁ = 1 - sum(a₂, a₃, ...)

    coeffs = [a₂, a₃, ...] (free coefficients, a₁ is derived)
    """
    a1 = 1.0 - sum(coeffs)
    result = a1 * sigma
    sigma_sq = sigma ** 2
    power = sigma_sq.copy()  # σ²
    for c in coeffs:
        result += c * sigma * power  # c * σ^(2k+1)
        power = power * sigma_sq
    return result


def eval_composed_poly(sigma: np.ndarray, all_coeffs: list[list[float]]) -> np.ndarray:
    """Evaluate composition of polynomials: p_n(...p_2(p_1(σ))...)"""
    result = sigma.copy()
    for coeffs in all_coeffs:
        result = eval_single_poly(result, coeffs)
    return result


def minimax_objective(flat_coeffs: np.ndarray, degrees: list[int], sigma: np.ndarray) -> float:
    """Minimize max |composed_poly(σ) - 1| over sigma grid."""
    all_coeffs = unflatten_coeffs(flat_coeffs, degrees)
    composed = eval_composed_poly(sigma, all_coeffs)
    error = np.abs(composed - 1.0)
    # Add penalty for NaN/Inf
    if np.any(~np.isfinite(composed)):
        return 1e10
    # Also penalize if any intermediate result goes negative or > 2 (divergence)
    result = sigma.copy()
    for coeffs in all_coeffs:
        result = eval_single_poly(result, coeffs)
        if np.any(result < -0.5) or np.any(result > 2.0):
            return 1e10 + np.max(np.abs(result))
    return np.max(error)


def unflatten_coeffs(flat: np.ndarray, degrees: list[int]) -> list[list[float]]:
    """Convert flat coefficient array to list of per-polynomial coefficient lists."""
    all_coeffs = []
    idx = 0
    for deg in degrees:
        n = degree_to_num_coeffs(deg) - 1  # free coefficients (excluding a₁)
        all_coeffs.append(list(flat[idx:idx+n]))
        idx += n
    return all_coeffs


def flatten_coeffs(all_coeffs: list[list[float]]) -> np.ndarray:
    """Flatten list of coefficient lists to single array."""
    return np.concatenate([np.array(c) for c in all_coeffs])


def get_total_free_coeffs(degrees: list[int]) -> int:
    """Total number of free coefficients across all polynomials."""
    return sum(degree_to_num_coeffs(d) - 1 for d in degrees)


def get_muon_init(degrees: list[int]) -> Optional[np.ndarray]:
    """Get initialization from Muon's quintic coefficients where applicable."""
    if all(d == 5 for d in degrees):
        # Use first len(degrees) Muon quintic coefficients
        n = min(len(degrees), len(MUON_QUINTIC_COEFFS))
        init_coeffs = []
        for i in range(len(degrees)):
            if i < n:
                a, b, c = MUON_QUINTIC_COEFFS[i]
                # Muon uses p(σ) = aσ + bσ³ + cσ⁵, so free coeffs are [b, c]
                init_coeffs.append([b, c])
            else:
                init_coeffs.append([-2.0, 0.5])  # reasonable default
        return flatten_coeffs(init_coeffs)
    return None


def optimize_config(config: PolyConfig, n_seeds: int = 15, verbose: bool = True) -> PolyConfig:
    """Optimize polynomial coefficients for a given configuration."""
    degrees = config.degrees
    n_free = get_total_free_coeffs(degrees)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Optimizing: {config.name} (degrees={degrees}, {config.matmuls} matmuls, {n_free} free coefficients)")
        print(f"{'='*70}")

    # Bounds for coefficients
    bounds = [(-30.0, 30.0)] * n_free

    best_result = None
    best_error = float('inf')
    t0 = time.time()

    # Try Muon init first if available
    muon_init = get_muon_init(degrees)

    for seed in range(n_seeds):
        if verbose and seed % 5 == 0:
            print(f"  Seed {seed}/{n_seeds}...")

        try:
            # Differential evolution
            result = differential_evolution(
                minimax_objective,
                bounds=bounds,
                args=(degrees, SIGMA_GRID),
                seed=seed,
                maxiter=2000,
                tol=1e-12,
                atol=1e-12,
                mutation=(0.5, 1.5),
                recombination=0.9,
                popsize=25,
                polish=False,
                init='sobol' if seed == 0 else 'latinhypercube',
                x0=muon_init if (seed == 0 and muon_init is not None) else None,
            )

            # Polish with Nelder-Mead
            polished = minimize(
                minimax_objective,
                result.x,
                args=(degrees, SIGMA_GRID),
                method='Nelder-Mead',
                options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )

            if polished.fun < result.fun:
                result = polished

            # Also try L-BFGS-B from this point
            try:
                lbfgs = minimize(
                    minimax_objective,
                    result.x,
                    args=(degrees, SIGMA_GRID),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 5000},
                )
                if lbfgs.fun < result.fun:
                    result = lbfgs
            except Exception:
                pass

            if result.fun < best_error:
                best_error = result.fun
                best_result = result
                if verbose:
                    print(f"    New best: max_error = {best_error:.8f}")

        except Exception as e:
            if verbose:
                print(f"    Seed {seed} failed: {e}")
            continue

    elapsed = time.time() - t0

    if best_result is not None:
        all_coeffs = unflatten_coeffs(best_result.x, degrees)
        # Convert to full coefficients (including a₁)
        full_coeffs = []
        for coeffs in all_coeffs:
            a1 = 1.0 - sum(coeffs)
            full_coeffs.append([a1] + list(coeffs))

        config.coefficients = full_coeffs
        config.max_error = float(best_error)
        config.optimization_time = elapsed

        if verbose:
            print(f"\n  Result: max_error = {best_error:.8f} ({'PASS' if best_error < ACCURACY_THRESHOLD else 'FAIL'})")
            print(f"  Time: {elapsed:.1f}s")
            for i, (deg, coeffs) in enumerate(zip(degrees, full_coeffs)):
                print(f"  Poly {i} (degree {deg}): {coeffs}")
    else:
        config.optimization_time = elapsed
        if verbose:
            print(f"  FAILED: No valid solution found in {elapsed:.1f}s")

    return config


def plot_convergence(config: PolyConfig, save_path: Path):
    """Plot |composed_poly(σ) - 1| for a config."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    if config.coefficients is None:
        return

    sigma = np.logspace(np.log10(0.001), np.log10(0.98), 5000)

    # Convert full coefficients back to free coefficients for evaluation
    free_coeffs = []
    for full in config.coefficients:
        free_coeffs.append(full[1:])  # skip a₁

    composed = eval_composed_poly(sigma, free_coeffs)
    error = np.abs(composed - 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Error plot
    ax1.semilogy(sigma, error, 'b-', linewidth=1.5)
    ax1.axhline(y=ACCURACY_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({ACCURACY_THRESHOLD})')
    ax1.set_xlabel('σ (singular value)')
    ax1.set_ylabel('|p(σ) - 1|')
    ax1.set_title(f'{config.name}\nmax error = {config.max_error:.6f}, {config.matmuls} matmuls')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])

    # Composed polynomial plot
    ax2.plot(sigma, composed, 'b-', linewidth=1.5)
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('σ (singular value)')
    ax2.set_ylabel('p(σ)')
    ax2.set_title(f'Composed polynomial output')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0.9, 1.1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved convergence plot: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: GPU Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def apply_ns_iteration_gpu(X, coefficients, degrees):
    """
    Apply Newton-Schulz iteration on GPU using optimized polynomial coefficients.
    X: (batch, m, n) or (m, n) tensor in bf16
    coefficients: list of list of full coefficients [a₁, a₂, a₃, ...]
    degrees: list of polynomial degrees
    """
    import torch

    for poly_coeffs, deg in zip(coefficients, degrees):
        n_terms = len(poly_coeffs)  # (degree+1)/2 terms

        # Compute X^T X
        if X.dim() == 2:
            A = X.mT @ X  # 1 matmul
        else:
            A = X.mT @ X

        # Build B = a₂*I + a₃*(X^TX) + a₄*(X^TX)² + ...
        # Actually: p(σ) = a₁σ + a₂σ³ + a₃σ⁵ + ...
        # X_{n+1} = a₁X + a₂X(X^TX) + a₃X(X^TX)² + ...
        # = a₁X + X @ [a₂*A + a₃*A² + a₄*A³ + ...]
        # = a₁X + X @ B where B = a₂*A + a₃*A² + ...

        a1 = poly_coeffs[0]

        if n_terms == 1:
            X = a1 * X
            continue

        # Build B incrementally: use Horner-like scheme for A powers
        # B = a₂*A + a₃*A² + a₄*A³ + ...
        # But it's more efficient to compute powers iteratively

        if n_terms == 3:
            # Quintic: B = a₂*A + a₃*A²  (Muon's form)
            a2, a3 = poly_coeffs[1], poly_coeffs[2]
            A2 = A @ A  # 1 matmul for A²
            B = a2 * A + a3 * A2
            X = a1 * X + X @ B  # 1 matmul
        elif n_terms == 4:
            # Septic: B = a₂*A + a₃*A² + a₄*A³
            a2, a3, a4 = poly_coeffs[1], poly_coeffs[2], poly_coeffs[3]
            A2 = A @ A
            A3 = A2 @ A  # 1 more matmul
            B = a2 * A + a3 * A2 + a4 * A3
            X = a1 * X + X @ B
        elif n_terms == 5:
            # Nonic: B = a₂*A + a₃*A² + a₄*A³ + a₅*A⁴
            a2, a3, a4, a5 = poly_coeffs[1], poly_coeffs[2], poly_coeffs[3], poly_coeffs[4]
            A2 = A @ A
            A3 = A2 @ A
            A4 = A3 @ A
            B = a2 * A + a3 * A2 + a4 * A3 + a5 * A4
            X = a1 * X + X @ B
        else:
            # General case
            powers = [A]
            for i in range(n_terms - 2):
                powers.append(powers[-1] @ A)
            B = sum(c * p for c, p in zip(poly_coeffs[1:], powers))
            X = a1 * X + X @ B

    return X


def apply_muon_baseline_gpu(X):
    """Apply Muon's original 5×quintic iteration."""
    import torch

    for a, b, c in MUON_QUINTIC_COEFFS:
        A = X.mT @ X
        B = b * A + c * (A @ A)
        X = a * X + X @ B
    return X


def generate_test_matrix(size, device, dtype):
    """Generate test matrix with realistic gradient spectra."""
    import torch

    m, n = size, size
    U, _ = torch.linalg.qr(torch.randn(m, m, device=device, dtype=torch.float32))
    V, _ = torch.linalg.qr(torch.randn(n, n, device=device, dtype=torch.float32))

    k = min(m, n)
    # Top 3% SVs near 1.0, rest exponentially decaying to ~0.005
    top_k = max(1, int(0.03 * k))
    svs = torch.zeros(k, device=device, dtype=torch.float32)
    svs[:top_k] = 0.9 + 0.1 * torch.rand(top_k, device=device)
    svs[top_k:] = 0.005 + (0.5 - 0.005) * torch.exp(
        -3.0 * torch.linspace(0, 1, k - top_k, device=device)
    )

    S = torch.zeros(m, n, device=device, dtype=torch.float32)
    S[:k, :k] = torch.diag(svs)

    X = (U @ S @ V.mT).to(dtype)
    # Normalize by spectral norm * 1.02 (Polar Express pattern)
    X = X / (torch.linalg.norm(X, ord=2) * 1.02)
    return X


def benchmark_config_gpu(config: PolyConfig, device: str = 'cuda:0', verbose: bool = True):
    """Benchmark a config on GPU."""
    import torch

    if config.coefficients is None or config.max_error >= ACCURACY_THRESHOLD:
        return config

    sizes = [256, 512, 768, 1024, 1536, 2048, 4096, 8192]
    warmup = 10
    repeats = 50

    if verbose:
        print(f"\n  GPU Benchmark: {config.name} on {device}")

    for size in sizes:
        try:
            X = generate_test_matrix(size, device, torch.bfloat16)

            # Warmup + benchmark config
            for _ in range(warmup):
                _ = apply_ns_iteration_gpu(X.clone(), config.coefficients, config.degrees)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(repeats):
                _ = apply_ns_iteration_gpu(X.clone(), config.coefficients, config.degrees)
            end.record()
            torch.cuda.synchronize()

            config_time = start.elapsed_time(end) / repeats
            config.gpu_time_ms[str(size)] = config_time

            # Warmup + benchmark baseline
            for _ in range(warmup):
                _ = apply_muon_baseline_gpu(X.clone())
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(repeats):
                _ = apply_muon_baseline_gpu(X.clone())
            end.record()
            torch.cuda.synchronize()

            baseline_time = start.elapsed_time(end) / repeats
            config.baseline_gpu_time_ms[str(size)] = baseline_time
            config.speedup[str(size)] = baseline_time / config_time if config_time > 0 else 0

            # Verify bf16 accuracy
            result = apply_ns_iteration_gpu(X.clone(), config.coefficients, config.degrees)
            U_r, S_r, V_r = torch.linalg.svd(result.float(), full_matrices=False)
            bf16_error = (S_r - 1.0).abs().max().item()
            config.bf16_max_error[str(size)] = bf16_error

            if verbose:
                status = "OK" if bf16_error < 0.05 else "WARN"
                print(f"    size={size:5d}: config={config_time:.3f}ms, baseline={baseline_time:.3f}ms, "
                      f"speedup={config.speedup[str(size)]:.3f}x, bf16_err={bf16_error:.4f} [{status}]")

        except torch.cuda.OutOfMemoryError:
            if verbose:
                print(f"    size={size}: OOM, skipping")
            continue
        except Exception as e:
            if verbose:
                print(f"    size={size}: Error: {e}")
            continue

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Training Validation
# ─────────────────────────────────────────────────────────────────────────────

def training_validation(config: PolyConfig, device: str = 'cuda:0', verbose: bool = True):
    """Validate config against baseline on random gradient-like matrices."""
    import torch

    if config.coefficients is None or config.max_error >= ACCURACY_THRESHOLD:
        return config

    # Check if actually faster
    has_speedup = any(v > 1.0 for v in config.speedup.values()) if config.speedup else True
    if not has_speedup and config.savings_pct > 0:
        if verbose:
            print(f"  Skipping training validation for {config.name}: not faster than baseline")
        return config

    n_matrices = 100
    size = 1024

    if verbose:
        print(f"\n  Training Validation: {config.name}")

    cosine_sims = []
    norm_ratios = []
    max_errors = []

    for i in range(n_matrices):
        X = generate_test_matrix(size, device, torch.bfloat16)

        result_config = apply_ns_iteration_gpu(X.clone(), config.coefficients, config.degrees)
        result_baseline = apply_muon_baseline_gpu(X.clone())

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            result_config.float().flatten().unsqueeze(0),
            result_baseline.float().flatten().unsqueeze(0)
        ).item()
        cosine_sims.append(cos_sim)

        # Frobenius norm ratio
        norm_ratio = (result_config.float().norm() / result_baseline.float().norm()).item()
        norm_ratios.append(norm_ratio)

        # Max element-wise difference
        max_err = (result_config.float() - result_baseline.float()).abs().max().item()
        max_errors.append(max_err)

    config.training_validation = {
        'mean_cosine_sim': float(np.mean(cosine_sims)),
        'min_cosine_sim': float(np.min(cosine_sims)),
        'mean_norm_ratio': float(np.mean(norm_ratios)),
        'std_norm_ratio': float(np.std(norm_ratios)),
        'mean_max_error': float(np.mean(max_errors)),
        'max_max_error': float(np.max(max_errors)),
        'n_matrices': n_matrices,
    }

    if verbose:
        v = config.training_validation
        print(f"    Cosine similarity: mean={v['mean_cosine_sim']:.6f}, min={v['min_cosine_sim']:.6f}")
        print(f"    Norm ratio: mean={v['mean_norm_ratio']:.6f} ± {v['std_norm_ratio']:.6f}")
        print(f"    Max error: mean={v['mean_max_error']:.6f}, max={v['max_max_error']:.6f}")

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Summary & Output
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(configs: list[PolyConfig]):
    """Plot summary comparison of all configs."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Accuracy vs matmuls plot
    fig, ax = plt.subplots(figsize=(10, 6))

    passed = [c for c in configs if c.max_error < ACCURACY_THRESHOLD]
    failed = [c for c in configs if c.max_error >= ACCURACY_THRESHOLD and c.coefficients is not None]
    no_result = [c for c in configs if c.coefficients is None]

    if passed:
        ax.scatter([c.matmuls for c in passed], [c.max_error for c in passed],
                   c='green', s=100, zorder=5, label='Pass')
        for c in passed:
            ax.annotate(c.name, (c.matmuls, c.max_error),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)

    if failed:
        ax.scatter([c.matmuls for c in failed], [c.max_error for c in failed],
                   c='red', s=100, zorder=5, label='Fail')
        for c in failed:
            ax.annotate(c.name, (c.matmuls, c.max_error),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.axhline(y=ACCURACY_THRESHOLD, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({ACCURACY_THRESHOLD})')
    ax.axvline(x=15, color='b', linestyle='--', alpha=0.5, label='Baseline (15 matmuls)')
    ax.set_xlabel('Total MatMuls')
    ax.set_ylabel('Max Error')
    ax.set_yscale('log')
    ax.set_title('Polynomial Schedule Search: Accuracy vs MatMuls')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'summary_accuracy_vs_matmuls.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary plot: {PLOTS_DIR / 'summary_accuracy_vs_matmuls.png'}")

    # GPU speedup plot
    passed_with_gpu = [c for c in passed if c.gpu_time_ms]
    if passed_with_gpu:
        fig, ax = plt.subplots(figsize=(12, 6))
        sizes = sorted(set(s for c in passed_with_gpu for s in c.speedup.keys()), key=int)

        for c in passed_with_gpu:
            speedups = [c.speedup.get(s, float('nan')) for s in sizes]
            ax.plot([int(s) for s in sizes], speedups, 'o-', label=f'{c.name} ({c.matmuls} mm)', markersize=6)

        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline (1.0x)')
        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Speedup vs Baseline')
        ax.set_title('GPU Speedup by Matrix Size')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'summary_gpu_speedup.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved GPU speedup plot: {PLOTS_DIR / 'summary_gpu_speedup.png'}")


def print_summary_table(configs: list[PolyConfig]):
    """Print summary table sorted by potential speedup."""
    print("\n" + "=" * 120)
    print("SUMMARY TABLE (sorted by matmul savings)")
    print("=" * 120)
    print(f"{'Config':<25s} {'Degrees':<20s} {'MatMuls':>7s} {'Savings':>8s} {'MaxErr':>10s} {'Status':>8s} {'OptTime':>8s}", end="")

    # Check if we have GPU data
    has_gpu = any(c.gpu_time_ms for c in configs)
    if has_gpu:
        print(f" {'GPU 1024':>10s} {'Speedup':>8s}", end="")
    print()
    print("-" * 120)

    for c in sorted(configs, key=lambda x: x.matmuls):
        status = "PASS" if c.max_error < ACCURACY_THRESHOLD else ("FAIL" if c.coefficients else "N/A")
        degrees_str = str(c.degrees)
        opt_time = f"{c.optimization_time:.0f}s" if c.optimization_time > 0 else "N/A"

        print(f"{c.name:<25s} {degrees_str:<20s} {c.matmuls:>7d} {c.savings_pct:>7.0f}% {c.max_error:>10.6f} {status:>8s} {opt_time:>8s}", end="")

        if has_gpu:
            gpu_time = c.gpu_time_ms.get('1024', float('nan'))
            speedup = c.speedup.get('1024', float('nan'))
            print(f" {gpu_time:>9.3f}ms {speedup:>7.3f}x", end="")
        print()

    print("=" * 120)

    # Highlight winners
    winners = [c for c in configs if c.max_error < ACCURACY_THRESHOLD and c.matmuls < 15]
    if winners:
        print(f"\nWINNERS (pass accuracy, fewer than 15 matmuls):")
        for c in sorted(winners, key=lambda x: x.matmuls):
            print(f"  {c.name}: {c.matmuls} matmuls ({c.savings_pct:.0f}% savings), max_error={c.max_error:.6f}")
    else:
        print("\nNo configs achieved both accuracy threshold AND fewer matmuls than baseline.")


def save_results(configs: list[PolyConfig], path: str = "results.json"):
    """Save all results to JSON."""
    results = []
    for c in configs:
        d = {
            'name': c.name,
            'degrees': c.degrees,
            'matmuls': c.matmuls,
            'savings_pct': c.savings_pct,
            'max_error': c.max_error,
            'optimization_time': c.optimization_time,
            'coefficients': c.coefficients,
            'gpu_time_ms': c.gpu_time_ms,
            'baseline_gpu_time_ms': c.baseline_gpu_time_ms,
            'speedup': c.speedup,
            'bf16_max_error': c.bf16_max_error,
            'training_validation': c.training_validation,
        }
        results.append(d)

    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Polynomial Schedule Search for Newton-Schulz Orthogonalization")
    print("=" * 70)
    print(f"Accuracy threshold: max error < {ACCURACY_THRESHOLD}")
    print(f"Sigma range: [{SIGMA_GRID[0]:.4f}, {SIGMA_GRID[-1]:.4f}], {len(SIGMA_GRID)} points")
    print(f"Baseline: 5 × quintic = 15 matmuls")
    print(f"Configs to test: {len(CONFIGS)}")

    # ── Part 1: CPU Optimization ──
    print("\n" + "#" * 70)
    print("# PART 1: Coefficient Optimization (CPU)")
    print("#" * 70)

    for i, config in enumerate(CONFIGS):
        print(f"\n[{i+1}/{len(CONFIGS)}]", end="")
        optimize_config(config, n_seeds=15, verbose=True)
        plot_convergence(config, PLOTS_DIR / f'convergence_{config.name.replace("×","x").replace("+","_")}.png')

    # Print intermediate results
    print_summary_table(CONFIGS)

    # ── Part 2: GPU Benchmark ──
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass

    if gpu_available:
        import torch
        print("\n" + "#" * 70)
        print("# PART 2: GPU Benchmark")
        print("#" * 70)

        device = 'cuda:0'
        print(f"\nUsing device: {device} ({torch.cuda.get_device_name(device)})")

        for i, config in enumerate(CONFIGS):
            if config.max_error < ACCURACY_THRESHOLD:
                print(f"\n[{i+1}/{len(CONFIGS)}]", end="")
                benchmark_config_gpu(config, device=device, verbose=True)

        # ── Part 3: Training Validation ──
        print("\n" + "#" * 70)
        print("# PART 3: Training Validation")
        print("#" * 70)

        for i, config in enumerate(CONFIGS):
            if config.max_error < ACCURACY_THRESHOLD:
                training_validation(config, device=device, verbose=True)
    else:
        print("\n[WARNING] No GPU available. Skipping Parts 2 & 3.")

    # ── Final Summary ──
    print_summary_table(CONFIGS)
    plot_summary(CONFIGS)
    save_results(CONFIGS)

    print("\nDone!")


if __name__ == "__main__":
    main()
