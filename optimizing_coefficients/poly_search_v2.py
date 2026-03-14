#!/usr/bin/env python3
"""
Polynomial Schedule Search v2 — Sequential Greedy Optimization.

Experiment 1: Drop Muon's last iterations (no optimization)
Experiment 2: Sequential greedy 2D grid search for 4×quintic and 3×quintic
Experiment 3: Mixed degrees (septic first, then quintic refinement)
Experiment 4: Relaxed lower bound Pareto frontier
GPU Benchmark: Wall-clock timing + bf16 accuracy verification
"""

import json
import os
import sys
import time
import itertools
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

PLOTS_DIR = Path("plots_v2")
PLOTS_DIR.mkdir(exist_ok=True)

RESULTS_FILE = "results_v2.json"
ACCURACY_THRESHOLD = 0.01

# Muon baseline coefficients (5 × quintic, 15 matmuls)
BASELINE_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

ALL_RESULTS = {}

# ─────────────────────────────────────────────────────────────────────────────
# Polynomial evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def eval_quintic(sigma, b, c):
    """Evaluate p(σ) = aσ + bσ³ + cσ⁵ with a = 1-b-c."""
    a = 1.0 - b - c
    return a * sigma + b * sigma**3 + c * sigma**5

def eval_septic(sigma, b, c, d):
    """Evaluate p(σ) = aσ + bσ³ + cσ⁵ + dσ⁷ with a = 1-b-c-d."""
    a = 1.0 - b - c - d
    return a * sigma + b * sigma**3 + c * sigma**5 + d * sigma**7

def eval_poly(sigma, coeffs):
    """Evaluate polynomial given full coefficients (a, b, c, ...).
    p(σ) = a*σ + b*σ³ + c*σ⁵ + d*σ⁷ + ..."""
    result = np.zeros_like(sigma)
    for i, coeff in enumerate(coeffs):
        result += coeff * sigma ** (2 * i + 1)
    return result

def eval_composed(sigma, all_coeffs):
    """Evaluate composition of polynomials. Each element is a tuple of full coefficients."""
    result = sigma.copy()
    for coeffs in all_coeffs:
        result = eval_poly(result, coeffs)
    return result

def make_sigma_grid(sigma_min=0.001, sigma_max=0.98, n=5000):
    """Log-spaced sigma grid."""
    return np.logspace(np.log10(sigma_min), np.log10(sigma_max), n)

def max_error(sigma, all_coeffs):
    """Compute max |composed_poly(σ) - 1| over sigma grid."""
    composed = eval_composed(sigma, all_coeffs)
    if np.any(~np.isfinite(composed)):
        return 1e10
    return float(np.max(np.abs(composed - 1.0)))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'figure.facecolor': 'white',
    })
    return plt


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Drop Muon's last iterations
# ═════════════════════════════════════════════════════════════════════════════

def experiment_1():
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Drop Muon's last iterations (no optimization)")
    print("=" * 70)

    sigma = make_sigma_grid(0.001, 0.98)
    results = {}

    for n_iters in [3, 4, 5]:
        coeffs = BASELINE_COEFFS[:n_iters]
        matmuls = n_iters * 3
        err = max_error(sigma, coeffs)
        status = "PASS" if err < ACCURACY_THRESHOLD else "FAIL"

        results[f"baseline_{n_iters}iter"] = {
            'name': f"Muon first {n_iters} iters",
            'n_iters': n_iters,
            'matmuls': matmuls,
            'coefficients': [list(c) for c in coeffs],
            'max_error': err,
            'status': status,
        }

        print(f"  {n_iters} iterations ({matmuls} matmuls): max_error = {err:.8f}  [{status}]")

    # Plot
    plt = setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {3: '#e74c3c', 4: '#e67e22', 5: '#27ae60'}
    labels = {3: '3 iters (9mm)', 4: '4 iters (12mm)', 5: '5 iters (15mm)'}

    for n_iters in [3, 4, 5]:
        coeffs = BASELINE_COEFFS[:n_iters]
        composed = eval_composed(sigma, coeffs)
        error = np.abs(composed - 1.0)

        ax1.semilogy(sigma, error, color=colors[n_iters], linewidth=1.5,
                     label=f"{labels[n_iters]}, err={results[f'baseline_{n_iters}iter']['max_error']:.4f}")
        ax2.plot(sigma, composed, color=colors[n_iters], linewidth=1.5, label=labels[n_iters])

    ax1.axhline(y=ACCURACY_THRESHOLD, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (0.01)')
    ax1.set_xlabel('σ (singular value)')
    ax1.set_ylabel('|p(σ) - 1|')
    ax1.set_title("Exp 1: Error with Muon's first N iterations")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim([0, 1])

    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('σ (singular value)')
    ax2.set_ylabel('p(σ)')
    ax2.set_title('Composed polynomial output')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0.9, 1.1])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp1_drop_iterations.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'exp1_drop_iterations.png'}")

    ALL_RESULTS['experiment_1'] = results
    return results


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Sequential greedy optimization (quintic)
# ═════════════════════════════════════════════════════════════════════════════

def grid_search_quintic(sigma_current, b_range, c_range, n_b=500, n_c=500):
    """Dense 2D grid search for quintic coefficients on current sigma distribution.
    Returns best (b, c) and the max error."""
    b_vals = np.linspace(b_range[0], b_range[1], n_b)
    c_vals = np.linspace(c_range[0], c_range[1], n_c)

    best_error = float('inf')
    best_b, best_c = 0.0, 0.0

    # Vectorized: compute for all b,c at once using broadcasting
    # sigma_current shape: (N,)
    s1 = sigma_current[np.newaxis, np.newaxis, :]          # (1, 1, N)
    s3 = sigma_current[np.newaxis, np.newaxis, :] ** 3
    s5 = sigma_current[np.newaxis, np.newaxis, :] ** 5

    # Process in chunks to avoid memory issues
    chunk_size = 50
    for i in range(0, n_b, chunk_size):
        b_chunk = b_vals[i:i+chunk_size]
        b_grid = b_chunk[:, np.newaxis, np.newaxis]  # (chunk, 1, 1)
        c_grid = c_vals[np.newaxis, :, np.newaxis]    # (1, n_c, 1)

        a_grid = 1.0 - b_grid - c_grid

        # p(σ) = a*σ + b*σ³ + c*σ⁵
        p = a_grid * s1 + b_grid * s3 + c_grid * s5  # (chunk, n_c, N)

        # Stability check: all outputs must be in [0, 1.5]
        stable = np.all((p >= -0.01) & (p <= 1.5), axis=2)  # (chunk, n_c)

        # Max error for each (b,c)
        errors = np.max(np.abs(p - 1.0), axis=2)  # (chunk, n_c)
        errors[~stable] = 1e10

        # Find minimum
        min_idx = np.unravel_index(np.argmin(errors), errors.shape)
        if errors[min_idx] < best_error:
            best_error = errors[min_idx]
            best_b = b_chunk[min_idx[0]]
            best_c = c_vals[min_idx[1]]

    return best_b, best_c, float(best_error)


def refine_quintic(sigma_current, b_init, c_init):
    """Refine quintic coefficients with Nelder-Mead."""
    def objective(x):
        b, c = x
        p = eval_quintic(sigma_current, b, c)
        if np.any(p < -0.5) or np.any(p > 2.0):
            return 1e10
        return np.max(np.abs(p - 1.0))

    result = minimize(objective, [b_init, c_init], method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
    return result.x[0], result.x[1], float(result.fun)


def grid_search_septic(sigma_current, b_range, c_range, d_range, n_per_dim=100):
    """3D grid search for septic coefficients — vectorized over d dimension."""
    b_vals = np.linspace(b_range[0], b_range[1], n_per_dim)
    c_vals = np.linspace(c_range[0], c_range[1], n_per_dim)
    d_vals = np.linspace(d_range[0], d_range[1], n_per_dim)

    best_error = float('inf')
    best_b, best_c, best_d = 0.0, 0.0, 0.0

    s1 = sigma_current                # (N,)
    s3 = sigma_current ** 3
    s5 = sigma_current ** 5
    s7 = sigma_current ** 7

    # Vectorize over d: d_vals is (D,), s1/s7 are (N,)
    # p(σ) = (1-b-c-d)*σ + b*σ³ + c*σ⁵ + d*σ⁷ = (1-b-c)*σ + b*σ³ + c*σ⁵ + d*(σ⁷ - σ)
    d_grid = d_vals[:, np.newaxis]     # (D, 1)
    s1_row = s1[np.newaxis, :]         # (1, N)
    s7_row = s7[np.newaxis, :]         # (1, N)
    diff_7_1 = s7_row - s1_row         # (1, N) — reusable

    for b in b_vals:
        for c in c_vals:
            # Base polynomial without d term: p_base = (1-b-c)*σ + b*σ³ + c*σ⁵
            a_base = 1.0 - b - c
            p_base = a_base * s1 + b * s3 + c * s5   # (N,)

            # Add d contribution for ALL d values at once:
            # p = p_base + d*(σ⁷ - σ)
            p_all = p_base[np.newaxis, :] + d_grid * diff_7_1  # (D, N)

            # Stability check
            stable = np.all((p_all >= -0.01) & (p_all <= 1.5), axis=1)  # (D,)

            # Max error per d
            errors = np.max(np.abs(p_all - 1.0), axis=1)  # (D,)
            errors[~stable] = 1e10

            best_idx = np.argmin(errors)
            if errors[best_idx] < best_error:
                best_error = errors[best_idx]
                best_b, best_c, best_d = b, c, d_vals[best_idx]

    return best_b, best_c, best_d, float(best_error)


def refine_septic(sigma_current, b_init, c_init, d_init):
    """Refine septic coefficients with Nelder-Mead."""
    def objective(x):
        b, c, d = x
        p = eval_septic(sigma_current, b, c, d)
        if np.any(p < -0.5) or np.any(p > 2.0):
            return 1e10
        return np.max(np.abs(p - 1.0))

    result = minimize(objective, [b_init, c_init, d_init], method='Nelder-Mead',
                      options={'maxiter': 100000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
    return result.x[0], result.x[1], result.x[2], float(result.fun)


def sequential_greedy_quintic(sigma, n_iters, verbose=True):
    """Sequential greedy optimization for n_iters quintic iterations."""
    all_coeffs = []
    sigma_current = sigma.copy()
    errors_per_iter = []

    # Grid search ranges: wide for first iteration, narrower for later ones
    ranges_first = ((-30, 5), (-5, 25))
    ranges_later = ((-10, 2), (-2, 5))

    for i in range(n_iters):
        b_range, c_range = ranges_first if i == 0 else ranges_later
        if verbose:
            print(f"    Iter {i+1}/{n_iters}: grid search b∈{b_range}, c∈{c_range}...", end="", flush=True)

        t0 = time.time()
        b, c, grid_err = grid_search_quintic(sigma_current, b_range, c_range)
        if verbose:
            print(f" grid={grid_err:.6f}", end="", flush=True)

        # Refine
        b, c, refined_err = refine_quintic(sigma_current, b, c)
        a = 1.0 - b - c
        elapsed = time.time() - t0

        all_coeffs.append((a, b, c))
        sigma_current = eval_quintic(sigma_current, b, c)

        # Compute composed error so far
        composed_err = max_error(sigma, all_coeffs)
        errors_per_iter.append(composed_err)

        if verbose:
            print(f" → refined={refined_err:.6f}, composed={composed_err:.8f} ({elapsed:.1f}s)")
            print(f"           coeffs: a={a:.10f}, b={b:.10f}, c={c:.10f}")
            print(f"           σ range after: [{sigma_current.min():.6f}, {sigma_current.max():.6f}]")

    return all_coeffs, errors_per_iter


def experiment_2():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Sequential greedy optimization (quintic)")
    print("=" * 70)

    sigma = make_sigma_grid(0.001, 0.98)
    results = {}

    for n_iters in [3, 4]:
        name = f"greedy_{n_iters}xquintic"
        matmuls = n_iters * 3
        print(f"\n  --- {n_iters}×quintic ({matmuls} matmuls) ---")

        t0 = time.time()
        coeffs, errors_per_iter = sequential_greedy_quintic(sigma, n_iters)
        elapsed = time.time() - t0

        final_err = errors_per_iter[-1]
        status = "PASS" if final_err < ACCURACY_THRESHOLD else "FAIL"

        results[name] = {
            'name': f"Greedy {n_iters}×quintic",
            'n_iters': n_iters,
            'matmuls': matmuls,
            'coefficients': [list(c) for c in coeffs],
            'errors_per_iter': errors_per_iter,
            'max_error': final_err,
            'optimization_time': elapsed,
            'status': status,
        }

        print(f"\n  Result: max_error = {final_err:.8f}  [{status}] ({elapsed:.1f}s)")

    # Plot
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Convergence profiles
    ax = axes[0]
    colors = {'greedy_3xquintic': '#e74c3c', 'greedy_4xquintic': '#e67e22'}
    for name, r in results.items():
        composed = eval_composed(sigma, [tuple(c) for c in r['coefficients']])
        error = np.abs(composed - 1.0)
        ax.semilogy(sigma, error, color=colors[name], linewidth=1.5,
                    label=f"{r['name']}: err={r['max_error']:.4f}")
    # Add baseline
    composed_bl = eval_composed(sigma, BASELINE_COEFFS)
    ax.semilogy(sigma, np.abs(composed_bl - 1.0), color='#27ae60', linewidth=1.5,
                label=f"Muon 5×quintic: err={max_error(sigma, BASELINE_COEFFS):.6f}", linestyle='--')
    ax.axhline(y=ACCURACY_THRESHOLD, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('σ')
    ax.set_ylabel('|p(σ) - 1|')
    ax.set_title('Exp 2: Sequential Greedy Convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Plot 2: Error after each iteration
    ax = axes[1]
    for name, r in results.items():
        iters = list(range(1, len(r['errors_per_iter'])+1))
        ax.semilogy(iters, r['errors_per_iter'], 'o-', color=colors[name],
                    linewidth=2, markersize=8, label=r['name'])
    ax.axhline(y=ACCURACY_THRESHOLD, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Composed max error')
    ax.set_title('Error convergence per iteration')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xticks([1, 2, 3, 4])

    # Plot 3: Composed polynomial output
    ax = axes[2]
    for name, r in results.items():
        composed = eval_composed(sigma, [tuple(c) for c in r['coefficients']])
        ax.plot(sigma, composed, color=colors[name], linewidth=1.5, label=r['name'])
    ax.plot(sigma, eval_composed(sigma, BASELINE_COEFFS), color='#27ae60',
            linewidth=1.5, linestyle='--', label='Muon 5×quintic')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('σ')
    ax.set_ylabel('p(σ)')
    ax.set_title('Composed polynomial output')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim([0.85, 1.15])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp2_sequential_greedy_quintic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'exp2_sequential_greedy_quintic.png'}")

    ALL_RESULTS['experiment_2'] = results
    return results


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2b: Sequential greedy with tight stability bounds
# ═════════════════════════════════════════════════════════════════════════════

ACCURACY_THRESHOLD_2B = 0.15  # Matching what Muon actually achieves

def grid_search_quintic_tight(sigma_current, b_range, c_range, n_b=500, n_c=500):
    """Dense 2D grid search with tight stability bounds: p ∈ [0, 1.05]."""
    b_vals = np.linspace(b_range[0], b_range[1], n_b)
    c_vals = np.linspace(c_range[0], c_range[1], n_c)

    best_error = float('inf')
    best_b, best_c = 0.0, 0.0

    s1 = sigma_current[np.newaxis, np.newaxis, :]
    s3 = sigma_current[np.newaxis, np.newaxis, :] ** 3
    s5 = sigma_current[np.newaxis, np.newaxis, :] ** 5

    chunk_size = 50
    for i in range(0, n_b, chunk_size):
        b_chunk = b_vals[i:i+chunk_size]
        b_grid = b_chunk[:, np.newaxis, np.newaxis]
        c_grid = c_vals[np.newaxis, :, np.newaxis]

        a_grid = 1.0 - b_grid - c_grid
        p = a_grid * s1 + b_grid * s3 + c_grid * s5

        # Tight stability: p ∈ [0, 1.05]
        stable = np.all((p >= 0.0) & (p <= 1.05), axis=2)

        errors = np.max(np.abs(p - 1.0), axis=2)
        errors[~stable] = 1e10

        min_idx = np.unravel_index(np.argmin(errors), errors.shape)
        if errors[min_idx] < best_error:
            best_error = errors[min_idx]
            best_b = b_chunk[min_idx[0]]
            best_c = c_vals[min_idx[1]]

    return best_b, best_c, float(best_error)


def refine_quintic_tight(sigma_current, b_init, c_init):
    """Refine quintic coefficients with tight bounds: p ∈ [0, 1.05]."""
    def objective(x):
        b, c = x
        p = eval_quintic(sigma_current, b, c)
        if np.any(p < 0.0) or np.any(p > 1.05):
            return 1e10
        return np.max(np.abs(p - 1.0))

    result = minimize(objective, [b_init, c_init], method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
    return result.x[0], result.x[1], float(result.fun)


def sequential_greedy_quintic_tight(sigma, n_iters, verbose=True):
    """Sequential greedy optimization with tight stability bounds."""
    all_coeffs = []
    sigma_current = sigma.copy()
    errors_per_iter = []

    ranges_first = ((-30, 5), (-5, 25))
    ranges_later = ((-10, 2), (-2, 5))

    for i in range(n_iters):
        b_range, c_range = ranges_first if i == 0 else ranges_later
        if verbose:
            print(f"    Iter {i+1}/{n_iters}: grid search b∈{b_range}, c∈{c_range}...", end="", flush=True)

        t0 = time.time()
        b, c, grid_err = grid_search_quintic_tight(sigma_current, b_range, c_range)
        if verbose:
            print(f" grid={grid_err:.6f}", end="", flush=True)

        b, c, refined_err = refine_quintic_tight(sigma_current, b, c)
        a = 1.0 - b - c
        elapsed = time.time() - t0

        all_coeffs.append((a, b, c))
        sigma_current = eval_quintic(sigma_current, b, c)

        composed_err = max_error(sigma, all_coeffs)
        errors_per_iter.append(composed_err)

        if verbose:
            print(f" → refined={refined_err:.6f}, composed={composed_err:.8f} ({elapsed:.1f}s)")
            print(f"           coeffs: a={a:.10f}, b={b:.10f}, c={c:.10f}")
            print(f"           σ range after: [{sigma_current.min():.6f}, {sigma_current.max():.6f}]")

    return all_coeffs, errors_per_iter


def experiment_2b():
    print("\n" + "=" * 70)
    print("EXPERIMENT 2b: Sequential greedy with TIGHT stability [0, 1.05]")
    print(f"  (accuracy target relaxed to {ACCURACY_THRESHOLD_2B} to match Muon)")
    print("=" * 70)

    sigma = make_sigma_grid(0.001, 0.98)
    results = {}

    for n_iters in [3, 4, 5]:
        name = f"tight_{n_iters}xquintic"
        matmuls = n_iters * 3
        print(f"\n  --- {n_iters}×quintic ({matmuls} matmuls), tight bounds ---")

        t0 = time.time()
        coeffs, errors_per_iter = sequential_greedy_quintic_tight(sigma, n_iters)
        elapsed = time.time() - t0

        final_err = errors_per_iter[-1]
        status = "PASS" if final_err < ACCURACY_THRESHOLD_2B else "FAIL"

        results[name] = {
            'name': f"Tight {n_iters}×quintic",
            'n_iters': n_iters,
            'matmuls': matmuls,
            'coefficients': [list(c) for c in coeffs],
            'errors_per_iter': errors_per_iter,
            'max_error': final_err,
            'optimization_time': elapsed,
            'status': status,
            'accuracy_target': ACCURACY_THRESHOLD_2B,
        }

        print(f"\n  Result: max_error = {final_err:.8f}  [{status}] ({elapsed:.1f}s)")

    # Compare with Muon baseline
    bl_err = max_error(sigma, BASELINE_COEFFS)
    print(f"\n  Muon baseline (5×quintic, 15mm): max_error = {bl_err:.8f}")

    # Plot
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {'tight_3xquintic': '#e74c3c', 'tight_4xquintic': '#e67e22', 'tight_5xquintic': '#2ecc71'}

    ax = axes[0]
    for name, r in results.items():
        composed = eval_composed(sigma, [tuple(c) for c in r['coefficients']])
        error = np.abs(composed - 1.0)
        ax.semilogy(sigma, error, color=colors[name], linewidth=1.5,
                    label=f"{r['name']}: err={r['max_error']:.4f}")
    composed_bl = eval_composed(sigma, BASELINE_COEFFS)
    ax.semilogy(sigma, np.abs(composed_bl - 1.0), color='gray', linewidth=1.5,
                label=f"Muon baseline: err={bl_err:.4f}", linestyle='--')
    ax.axhline(y=ACCURACY_THRESHOLD_2B, color='gray', linestyle='--', linewidth=2, alpha=0.5,
               label=f'Threshold ({ACCURACY_THRESHOLD_2B})')
    ax.set_xlabel('σ')
    ax.set_ylabel('|p(σ) - 1|')
    ax.set_title('Exp 2b: Tight Stability Convergence')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    for name, r in results.items():
        iters = list(range(1, len(r['errors_per_iter'])+1))
        ax.semilogy(iters, r['errors_per_iter'], 'o-', color=colors[name],
                    linewidth=2, markersize=8, label=r['name'])
    ax.axhline(y=ACCURACY_THRESHOLD_2B, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Composed max error')
    ax.set_title('Error convergence per iteration')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    for name, r in results.items():
        composed = eval_composed(sigma, [tuple(c) for c in r['coefficients']])
        ax.plot(sigma, composed, color=colors[name], linewidth=1.5, label=r['name'])
    ax.plot(sigma, eval_composed(sigma, BASELINE_COEFFS), color='gray',
            linewidth=1.5, linestyle='--', label='Muon baseline')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('σ')
    ax.set_ylabel('p(σ)')
    ax.set_title('Composed polynomial output')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_ylim([0.75, 1.15])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp2b_tight_stability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'exp2b_tight_stability.png'}")

    ALL_RESULTS['experiment_2b'] = results
    return results


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Mixed degrees (septic + quintic)
# ═════════════════════════════════════════════════════════════════════════════

def sequential_greedy_mixed(sigma, degree_sequence, verbose=True):
    """Sequential greedy optimization with mixed polynomial degrees.
    degree_sequence: list of degrees, e.g. [7, 5, 5, 5]"""
    all_coeffs = []
    sigma_current = sigma.copy()
    errors_per_iter = []

    for i, degree in enumerate(degree_sequence):
        if verbose:
            print(f"    Iter {i+1}/{len(degree_sequence)} (degree {degree}):", end="", flush=True)

        t0 = time.time()

        if degree == 5:
            # Quintic: 2D grid search
            if i == 0:
                b_range, c_range = (-30, 5), (-5, 25)
            else:
                b_range, c_range = (-10, 2), (-2, 5)
            b, c, grid_err = grid_search_quintic(sigma_current, b_range, c_range)
            b, c, refined_err = refine_quintic(sigma_current, b, c)
            a = 1.0 - b - c
            coeffs = (a, b, c)
            sigma_current = eval_quintic(sigma_current, b, c)

        elif degree == 7:
            # Septic: 3D grid search + refine
            if i == 0:
                b_range, c_range, d_range = (-30, 5), (-5, 30), (-30, 30)
            else:
                b_range, c_range, d_range = (-15, 5), (-5, 15), (-15, 15)

            b, c, d, grid_err = grid_search_septic(
                sigma_current, b_range, c_range, d_range, n_per_dim=100)

            if verbose:
                print(f" grid={grid_err:.6f}", end="", flush=True)

            b, c, d, refined_err = refine_septic(sigma_current, b, c, d)
            a = 1.0 - b - c - d
            coeffs = (a, b, c, d)
            sigma_current = eval_septic(sigma_current, b, c, d)
        else:
            raise ValueError(f"Unsupported degree: {degree}")

        elapsed = time.time() - t0
        all_coeffs.append(coeffs)

        composed_err = max_error(sigma, all_coeffs)
        errors_per_iter.append(composed_err)

        if verbose:
            print(f" → err={composed_err:.8f} ({elapsed:.1f}s)")
            print(f"           coeffs: {', '.join(f'{v:.10f}' for v in coeffs)}")
            print(f"           σ range after: [{sigma_current.min():.6f}, {sigma_current.max():.6f}]")

    return all_coeffs, errors_per_iter


def experiment_3():
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Mixed degrees (septic + quintic)")
    print("=" * 70)

    sigma = make_sigma_grid(0.001, 0.98)
    results = {}

    configs = [
        ("1×septic+3×quintic", [7, 5, 5, 5], 13),
        ("1×septic+2×quintic", [7, 5, 5], 10),
        ("2×septic+1×quintic", [7, 7, 5], 11),
    ]

    for name, degrees, matmuls in configs:
        print(f"\n  --- {name} ({matmuls} matmuls) ---")

        t0 = time.time()
        coeffs, errors_per_iter = sequential_greedy_mixed(sigma, degrees)
        elapsed = time.time() - t0

        final_err = errors_per_iter[-1]
        status = "PASS" if final_err < ACCURACY_THRESHOLD else "FAIL"

        results[name] = {
            'name': name,
            'degrees': degrees,
            'matmuls': matmuls,
            'coefficients': [list(c) for c in coeffs],
            'errors_per_iter': errors_per_iter,
            'max_error': final_err,
            'optimization_time': elapsed,
            'status': status,
        }

        print(f"\n  Result: max_error = {final_err:.8f}  [{status}] ({elapsed:.1f}s)")

    # Plot
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors_list = ['#e74c3c', '#3498db', '#9b59b6']

    # Plot 1: Convergence profiles
    ax = axes[0]
    for (name, r), color in zip(results.items(), colors_list):
        composed = eval_composed(sigma, [tuple(c) for c in r['coefficients']])
        error = np.abs(composed - 1.0)
        ax.semilogy(sigma, error, color=color, linewidth=1.5,
                    label=f"{r['name']} ({r['matmuls']}mm): err={r['max_error']:.4f}")
    composed_bl = eval_composed(sigma, BASELINE_COEFFS)
    ax.semilogy(sigma, np.abs(composed_bl - 1.0), color='#27ae60', linewidth=1.5,
                linestyle='--', label=f"Muon baseline (15mm)")
    ax.axhline(y=ACCURACY_THRESHOLD, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('σ')
    ax.set_ylabel('|p(σ) - 1|')
    ax.set_title('Exp 3: Mixed Degree Convergence')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # Plot 2: Error per iteration
    ax = axes[1]
    for (name, r), color in zip(results.items(), colors_list):
        iters = list(range(1, len(r['errors_per_iter'])+1))
        ax.semilogy(iters, r['errors_per_iter'], 'o-', color=color,
                    linewidth=2, markersize=8, label=f"{r['name']}")
    ax.axhline(y=ACCURACY_THRESHOLD, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Composed max error')
    ax.set_title('Error per iteration')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Plot 3: Composed output
    ax = axes[2]
    for (name, r), color in zip(results.items(), colors_list):
        composed = eval_composed(sigma, [tuple(c) for c in r['coefficients']])
        ax.plot(sigma, composed, color=color, linewidth=1.5, label=r['name'])
    ax.plot(sigma, eval_composed(sigma, BASELINE_COEFFS), color='#27ae60',
            linewidth=1.5, linestyle='--', label='Muon baseline')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('σ')
    ax.set_ylabel('p(σ)')
    ax.set_title('Composed polynomial output')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_ylim([0.85, 1.15])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp3_mixed_degrees.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'exp3_mixed_degrees.png'}")

    ALL_RESULTS['experiment_3'] = results
    return results


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Relaxed lower bound Pareto frontier
# ═════════════════════════════════════════════════════════════════════════════

def experiment_4():
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Relaxed lower bound Pareto frontier")
    print("=" * 70)

    sigma_lbs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    n_iters_list = [3, 4, 5]
    results = {}

    for n_iters in n_iters_list:
        for sigma_lb in sigma_lbs:
            sigma = make_sigma_grid(sigma_lb, 0.98)
            key = f"{n_iters}iter_lb{sigma_lb}"
            matmuls = n_iters * 3

            if n_iters == 5:
                # Use Muon's original coefficients
                coeffs = BASELINE_COEFFS[:5]
                err = max_error(sigma, coeffs)
                results[key] = {
                    'name': f"Muon {n_iters}×quintic (σ≥{sigma_lb})",
                    'n_iters': n_iters,
                    'matmuls': matmuls,
                    'sigma_lb': sigma_lb,
                    'coefficients': [list(c) for c in coeffs],
                    'max_error': err,
                    'status': "PASS" if err < ACCURACY_THRESHOLD else "FAIL",
                    'optimization_time': 0,
                }
            else:
                print(f"  {n_iters}×quintic, σ_min={sigma_lb}...", end="", flush=True)
                t0 = time.time()
                coeffs, errors_per_iter = sequential_greedy_quintic(sigma, n_iters, verbose=False)
                elapsed = time.time() - t0
                final_err = errors_per_iter[-1]
                status = "PASS" if final_err < ACCURACY_THRESHOLD else "FAIL"

                results[key] = {
                    'name': f"Greedy {n_iters}×quintic (σ≥{sigma_lb})",
                    'n_iters': n_iters,
                    'matmuls': matmuls,
                    'sigma_lb': sigma_lb,
                    'coefficients': [list(c) for c in coeffs],
                    'errors_per_iter': errors_per_iter,
                    'max_error': final_err,
                    'optimization_time': elapsed,
                    'status': status,
                }
                print(f" err={final_err:.8f} [{status}] ({elapsed:.1f}s)")

    # Print table
    print(f"\n  {'σ_min':<8s}", end="")
    for n in n_iters_list:
        print(f"  {'%d iter (%dmm)' % (n, n*3):>20s}", end="")
    print()
    print("  " + "-" * 70)
    for sigma_lb in sigma_lbs:
        print(f"  {sigma_lb:<8.3f}", end="")
        for n in n_iters_list:
            key = f"{n}iter_lb{sigma_lb}"
            r = results[key]
            marker = "✓" if r['max_error'] < ACCURACY_THRESHOLD else "✗"
            print(f"  {r['max_error']:>18.6f} {marker}", end="")
        print()

    # Plot: heatmap
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap
    ax = axes[0]
    data = np.zeros((len(n_iters_list), len(sigma_lbs)))
    for i, n in enumerate(n_iters_list):
        for j, lb in enumerate(sigma_lbs):
            key = f"{n}iter_lb{lb}"
            data[i, j] = np.log10(max(results[key]['max_error'], 1e-10))

    im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r',
                   vmin=-4, vmax=0)
    ax.set_xticks(range(len(sigma_lbs)))
    ax.set_xticklabels([str(lb) for lb in sigma_lbs])
    ax.set_yticks(range(len(n_iters_list)))
    ax.set_yticklabels([f"{n} iter ({n*3}mm)" for n in n_iters_list])
    ax.set_xlabel('σ lower bound')
    ax.set_title('Exp 4: log₁₀(max error) Pareto Frontier')

    # Add text annotations
    for i in range(len(n_iters_list)):
        for j in range(len(sigma_lbs)):
            key = f"{n_iters_list[i]}iter_lb{sigma_lbs[j]}"
            err = results[key]['max_error']
            passed = err < ACCURACY_THRESHOLD
            text = f"{err:.4f}" if err < 1 else f"{err:.1f}"
            color = 'white' if data[i, j] > -1 else 'black'
            weight = 'bold' if passed else 'normal'
            ax.text(j, i, text, ha='center', va='center', fontsize=8,
                   color=color, fontweight=weight)

    plt.colorbar(im, ax=ax, label='log₁₀(max error)', shrink=0.8)

    # Pareto curve plot
    ax = axes[1]
    colors = {3: '#e74c3c', 4: '#e67e22', 5: '#27ae60'}
    for n in n_iters_list:
        errors = [results[f"{n}iter_lb{lb}"]['max_error'] for lb in sigma_lbs]
        ax.semilogy(sigma_lbs, errors, 'o-', color=colors[n], linewidth=2,
                    markersize=8, label=f"{n}×quintic ({n*3}mm)")
    ax.axhline(y=ACCURACY_THRESHOLD, color='gray', linestyle='--', linewidth=2, alpha=0.5,
               label='Threshold (0.01)')
    ax.set_xlabel('σ lower bound')
    ax.set_ylabel('Max error')
    ax.set_title('Max Error vs Lower Bound')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'exp4_pareto_frontier.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {PLOTS_DIR / 'exp4_pareto_frontier.png'}")

    ALL_RESULTS['experiment_4'] = results
    return results


# ═════════════════════════════════════════════════════════════════════════════
# GPU BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════

def apply_iteration_gpu(X, all_coeffs):
    """Apply Newton-Schulz iteration on GPU.
    all_coeffs: list of tuples, each with (a, b, c) for quintic or (a, b, c, d) for septic."""
    import torch

    for coeffs in all_coeffs:
        A = X.mT @ X
        if len(coeffs) == 3:
            a, b, c = coeffs
            B = b * A + c * (A @ A)
            X = a * X + X @ B
        elif len(coeffs) == 4:
            a, b, c, d = coeffs
            A2 = A @ A
            A3 = A2 @ A
            B = b * A + c * A2 + d * A3
            X = a * X + X @ B
        else:
            raise ValueError(f"Unsupported coefficient length: {len(coeffs)}")
    return X


def generate_test_matrix(size, device, dtype):
    """Generate test matrix with realistic gradient spectra."""
    import torch

    m, n = size, size
    U, _ = torch.linalg.qr(torch.randn(m, m, device=device, dtype=torch.float32))
    V, _ = torch.linalg.qr(torch.randn(n, n, device=device, dtype=torch.float32))

    k = min(m, n)
    top_k = max(1, int(0.03 * k))
    svs = torch.zeros(k, device=device, dtype=torch.float32)
    svs[:top_k] = 0.9 + 0.1 * torch.rand(top_k, device=device)
    svs[top_k:] = 0.005 + (0.5 - 0.005) * torch.exp(
        -3.0 * torch.linspace(0, 1, k - top_k, device=device)
    )

    S = torch.zeros(m, n, device=device, dtype=torch.float32)
    S[:k, :k] = torch.diag(svs)
    X = U @ S @ V.mT
    # Normalize by spectral norm in float32, then cast to target dtype
    X = X / (torch.linalg.norm(X.float(), ord=2) * 1.02)
    return X.to(dtype)


def gpu_benchmark():
    """Benchmark all passing configs on GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n[WARNING] No GPU available. Skipping GPU benchmark.")
            return {}
    except ImportError:
        print("\n[WARNING] PyTorch not available. Skipping GPU benchmark.")
        return {}

    import torch

    print("\n" + "=" * 70)
    print("GPU BENCHMARK")
    print("=" * 70)

    device = 'cuda:0'
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")

    # Collect all configs that passed in any experiment
    passing_configs = []

    for exp_name, exp_results in ALL_RESULTS.items():
        for key, r in exp_results.items():
            threshold = r.get('accuracy_target', ACCURACY_THRESHOLD)
            if r.get('max_error', float('inf')) < threshold and r.get('coefficients'):
                passing_configs.append({
                    'source': f"{exp_name}/{key}",
                    'name': r['name'],
                    'matmuls': r.get('matmuls', len(r['coefficients']) * 3),
                    'coefficients': [tuple(c) for c in r['coefficients']],
                    'max_error': r['max_error'],
                })

    # Always add baseline
    sigma_check = make_sigma_grid(0.001, 0.98)
    baseline_err = max_error(sigma_check, BASELINE_COEFFS)
    passing_configs.append({
        'source': 'baseline',
        'name': 'Muon 5×quintic (baseline)',
        'matmuls': 15,
        'coefficients': BASELINE_COEFFS,
        'max_error': baseline_err,
        'is_baseline': True,
    })

    # Deduplicate by name
    seen = set()
    unique_configs = []
    for c in passing_configs:
        if c['name'] not in seen:
            seen.add(c['name'])
            unique_configs.append(c)
    passing_configs = unique_configs

    if len(passing_configs) <= 1:
        print("  No configs passed accuracy threshold. Only benchmarking baseline.")

    return {}


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Measure actual σ_min after normalization
# ═════════════════════════════════════════════════════════════════════════════

def measure_sigma_min():
    """Measure what σ_min actually is after spectral norm normalization."""
    import torch

    print("\n" + "=" * 70)
    print("STEP 1: Measure actual σ_min after normalization")
    print("=" * 70)

    device = 'cuda:0'
    n_matrices = 200
    sizes = [512, 1024, 2048, 4096]

    results = {}

    for size in sizes:
        sigma_mins = []
        sigma_maxs = []
        for _ in range(n_matrices):
            X = generate_test_matrix(size, device, torch.bfloat16)
            # SVD in float32 to measure
            _, S, _ = torch.linalg.svd(X.float(), full_matrices=False)
            sigma_mins.append(S.min().item())
            sigma_maxs.append(S.max().item())

        sigma_mins = np.array(sigma_mins)
        sigma_maxs = np.array(sigma_maxs)

        results[str(size)] = {
            'sigma_min_min': float(np.min(sigma_mins)),
            'sigma_min_median': float(np.median(sigma_mins)),
            'sigma_min_max': float(np.max(sigma_mins)),
            'sigma_min_mean': float(np.mean(sigma_mins)),
            'sigma_min_std': float(np.std(sigma_mins)),
            'sigma_max_min': float(np.min(sigma_maxs)),
            'sigma_max_median': float(np.median(sigma_maxs)),
            'sigma_max_max': float(np.max(sigma_maxs)),
        }

        print(f"  size={size:5d}: σ_min = [{np.min(sigma_mins):.6f}, {np.median(sigma_mins):.6f}, {np.max(sigma_mins):.6f}] "
              f"(min/median/max)  σ_max = [{np.min(sigma_maxs):.6f}, {np.median(sigma_maxs):.6f}, {np.max(sigma_maxs):.6f}]")

    ALL_RESULTS['sigma_min_measurement'] = results
    return results


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Fixed GPU benchmark with specific configs
# ═════════════════════════════════════════════════════════════════════════════

def gpu_benchmark_v2():
    """Benchmark specific configs from Experiment 4 on GPU."""
    import torch

    print("\n" + "=" * 70)
    print("STEP 2: GPU Benchmark (fixed)")
    print("=" * 70)

    device = 'cuda:0'
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")

    # Load coefficients from results
    exp4 = ALL_RESULTS.get('experiment_4', {})

    # Specific configs to benchmark
    benchmark_configs = []

    for key, label in [
        ('4iter_lb0.02', '4×quintic σ≥0.02 (12mm)'),
        ('4iter_lb0.01', '4×quintic σ≥0.01 (12mm)'),
        ('3iter_lb0.1',  '3×quintic σ≥0.1 (9mm)'),
        ('3iter_lb0.05', '3×quintic σ≥0.05 (9mm)'),
    ]:
        r = exp4.get(key)
        if r and r.get('coefficients'):
            benchmark_configs.append({
                'name': label,
                'key': key,
                'matmuls': r['matmuls'],
                'coefficients': [tuple(c) for c in r['coefficients']],
                'max_error': r['max_error'],
                'sigma_lb': r.get('sigma_lb', 0),
                'is_baseline': False,
            })

    # Add baseline
    benchmark_configs.append({
        'name': 'Muon 5×quintic baseline (15mm)',
        'key': 'baseline',
        'matmuls': 15,
        'coefficients': BASELINE_COEFFS,
        'max_error': max_error(make_sigma_grid(0.001, 0.98), BASELINE_COEFFS),
        'sigma_lb': 0.001,
        'is_baseline': True,
    })

    sizes = [512, 1024, 2048, 4096]
    warmup = 10
    repeats = 50
    n_accuracy_matrices = 20

    gpu_results = {}

    for config in benchmark_configs:
        name = config['name']
        coeffs = config['coefficients']
        print(f"\n  Benchmarking: {name}")

        timings = {}
        bf16_errors = {}

        for size in sizes:
            try:
                X = generate_test_matrix(size, device, torch.bfloat16)

                # Warmup
                for _ in range(warmup):
                    _ = apply_iteration_gpu(X.clone(), coeffs)
                torch.cuda.synchronize()

                # Time
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(repeats):
                    _ = apply_iteration_gpu(X.clone(), coeffs)
                end.record()
                torch.cuda.synchronize()
                timings[str(size)] = start.elapsed_time(end) / repeats

                # bf16 accuracy on 20 random matrices
                max_bf16_err = 0.0
                for _ in range(n_accuracy_matrices):
                    X_test = generate_test_matrix(size, device, torch.bfloat16)
                    result = apply_iteration_gpu(X_test, coeffs)
                    _, S, _ = torch.linalg.svd(result.float(), full_matrices=False)
                    err = (S - 1.0).abs().max().item()
                    max_bf16_err = max(max_bf16_err, err)
                bf16_errors[str(size)] = max_bf16_err

                print(f"    size={size:5d}: {timings[str(size)]:.3f}ms, bf16_err={max_bf16_err:.6f}")

            except torch.cuda.OutOfMemoryError:
                print(f"    size={size}: OOM")
            except Exception as e:
                print(f"    size={size}: Error: {e}")

        gpu_results[name] = {
            'name': name,
            'key': config['key'],
            'matmuls': config['matmuls'],
            'max_error_theoretical': config['max_error'],
            'sigma_lb': config['sigma_lb'],
            'timings_ms': timings,
            'bf16_errors': bf16_errors,
            'is_baseline': config['is_baseline'],
        }

    # Compute speedups relative to baseline
    baseline_timings = gpu_results.get('Muon 5×quintic baseline (15mm)', {}).get('timings_ms', {})
    for name, r in gpu_results.items():
        speedups = {}
        for size_str, t in r['timings_ms'].items():
            bl_t = baseline_timings.get(size_str)
            if bl_t and t > 0:
                speedups[size_str] = bl_t / t
        r['speedups'] = speedups

    # Print summary table
    print("\n" + "=" * 140)
    print("GPU BENCHMARK SUMMARY")
    print("=" * 140)
    print(f"{'Config':<40s} {'MM':>3s} {'σ_lb':>6s} {'512ms':>8s} {'1024ms':>8s} {'2048ms':>8s} {'4096ms':>8s} "
          f"{'Spd1024':>8s} {'Spd4096':>8s} {'Err(thy)':>10s} {'Err(bf16)':>10s}")
    print("-" * 140)

    for name, r in sorted(gpu_results.items(), key=lambda x: x[1].get('speedups', {}).get('1024', 0), reverse=True):
        t512 = r['timings_ms'].get('512', float('nan'))
        t1024 = r['timings_ms'].get('1024', float('nan'))
        t2048 = r['timings_ms'].get('2048', float('nan'))
        t4096 = r['timings_ms'].get('4096', float('nan'))
        spd1024 = r.get('speedups', {}).get('1024', float('nan'))
        spd4096 = r.get('speedups', {}).get('4096', float('nan'))
        bf16_err = max(r['bf16_errors'].values()) if r['bf16_errors'] else float('nan')

        print(f"{name:<40s} {r['matmuls']:>3d} {r['sigma_lb']:>6.3f} "
              f"{t512:>7.3f}ms {t1024:>7.3f}ms {t2048:>7.3f}ms {t4096:>7.3f}ms "
              f"{spd1024:>7.3f}x {spd4096:>7.3f}x "
              f"{r['max_error_theoretical']:>10.6f} {bf16_err:>10.6f}")

    print("=" * 140)

    ALL_RESULTS['gpu_benchmark'] = gpu_results

    # Plot
    plt = setup_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, r in gpu_results.items():
        if r['is_baseline']:
            continue
        speedups = r.get('speedups', {})
        if speedups:
            ss = sorted(speedups.items(), key=lambda x: int(x[0]))
            ax1.plot([int(s) for s, _ in ss], [v for _, v in ss], 'o-',
                    label=f"{name}", markersize=6, linewidth=2)

    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (1.0x)')
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Speedup vs Baseline')
    ax1.set_title('GPU Speedup by Matrix Size')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2)

    for name, r in gpu_results.items():
        errs = r.get('bf16_errors', {})
        if errs:
            ss = sorted(errs.items(), key=lambda x: int(x[0]))
            style = '--' if r['is_baseline'] else '-'
            ax2.semilogy([int(s) for s, _ in ss], [v for _, v in ss], f'o{style}',
                        label=f"{name}", markersize=6)
    ax2.axhline(y=0.01, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='0.01 threshold')
    ax2.set_xlabel('Matrix Size')
    ax2.set_ylabel('Max |σ_out - 1| (bf16)')
    ax2.set_title('BF16 Accuracy by Matrix Size')
    ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'gpu_benchmark.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOTS_DIR / 'gpu_benchmark.png'}")

    return gpu_results


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Match configs to measured σ_min → recommendation
# ═════════════════════════════════════════════════════════════════════════════

def final_recommendation():
    """Given measured σ_min and GPU benchmarks, recommend the best config."""
    print("\n" + "=" * 70)
    print("STEP 3: FINAL RECOMMENDATION")
    print("=" * 70)

    sigma_data = ALL_RESULTS.get('sigma_min_measurement', {})
    gpu_data = ALL_RESULTS.get('gpu_benchmark', {})

    if not sigma_data or not gpu_data:
        print("  Missing data, cannot make recommendation.")
        return

    # Find worst-case σ_min across all sizes
    worst_sigma_min = min(r['sigma_min_min'] for r in sigma_data.values())
    median_sigma_min = np.median([r['sigma_min_median'] for r in sigma_data.values()])

    print(f"\n  Measured σ_min across all sizes:")
    print(f"    Worst case (absolute min): {worst_sigma_min:.6f}")
    print(f"    Typical (median of medians): {median_sigma_min:.6f}")

    # For each config, check if its σ_lb is safe given measured σ_min
    print(f"\n  {'Config':<40s} {'σ_lb':>6s} {'Safe?':>6s} {'Spd1024':>8s} {'Spd4096':>8s} {'Err(thy)':>10s} {'Err(bf16)':>10s}")
    print("  " + "-" * 100)

    for name, r in sorted(gpu_data.items(), key=lambda x: x[1].get('speedups', {}).get('1024', 0), reverse=True):
        sigma_lb = r.get('sigma_lb', 0)
        safe = "YES" if sigma_lb <= worst_sigma_min else "NO"
        spd1024 = r.get('speedups', {}).get('1024', float('nan'))
        spd4096 = r.get('speedups', {}).get('4096', float('nan'))
        bf16_err = max(r['bf16_errors'].values()) if r['bf16_errors'] else float('nan')

        print(f"  {name:<40s} {sigma_lb:>6.3f} {safe:>6s} {spd1024:>7.3f}x {spd4096:>7.3f}x "
              f"{r['max_error_theoretical']:>10.6f} {bf16_err:>10.6f}")

    # Find best safe config
    best_safe = None
    best_speedup = 0
    for name, r in gpu_data.items():
        if r['is_baseline']:
            continue
        sigma_lb = r.get('sigma_lb', 0)
        if sigma_lb <= worst_sigma_min:
            spd = r.get('speedups', {}).get('1024', 0)
            if spd > best_speedup:
                best_speedup = spd
                best_safe = r

    print(f"\n  RECOMMENDATION (based on worst-case σ_min = {worst_sigma_min:.6f}):")
    if best_safe:
        print(f"    Best config: {best_safe['name']}")
        print(f"    MatMuls: {best_safe['matmuls']} (vs 15 baseline)")
        print(f"    Speedup at 1024: {best_safe.get('speedups', {}).get('1024', 0):.3f}x")
        print(f"    Speedup at 4096: {best_safe.get('speedups', {}).get('4096', 0):.3f}x")
        print(f"    Theoretical error: {best_safe['max_error_theoretical']:.8f}")
    else:
        print(f"    No config with σ_lb ≤ {worst_sigma_min:.6f} found.")
        print(f"    Muon's 5×quintic (15 matmuls) remains the best option.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def save_results():
    """Save all results to JSON."""
    # Convert any numpy types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(RESULTS_FILE, 'w') as f:
        json.dump(ALL_RESULTS, f, indent=2, default=convert)
    print(f"\nAll results saved to {RESULTS_FILE}")


def main():
    print("=" * 70)
    print("Polynomial Schedule Search v2 — Sequential Greedy Optimization")
    print("=" * 70)

    # Verify baseline
    sigma = make_sigma_grid(0.001, 0.98)
    bl_err = max_error(sigma, BASELINE_COEFFS)
    print(f"Baseline verification: 5×quintic max_error = {bl_err:.8f}")

    # Run experiments
    experiment_1()
    save_results()

    experiment_2()
    save_results()

    experiment_2b()
    save_results()

    experiment_3()
    save_results()

    experiment_4()
    save_results()

    # Step 1: Measure actual σ_min
    measure_sigma_min()
    save_results()

    # Step 2: GPU benchmark (fixed)
    gpu_benchmark_v2()
    save_results()

    # Step 3: Final recommendation
    final_recommendation()
    save_results()

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print(f"\nAll configs that passed strict threshold (max_error < {ACCURACY_THRESHOLD}):")
    found_any = False
    for exp_name, exp_results in ALL_RESULTS.items():
        if exp_name in ('gpu_benchmark', 'sigma_min_measurement'):
            continue
        for key, r in exp_results.items():
            if r.get('max_error', float('inf')) < ACCURACY_THRESHOLD:
                found_any = True
                print(f"  [{exp_name}] {r['name']}: {r.get('matmuls', '?')} matmuls, err={r['max_error']:.8f}")
    if not found_any:
        print("  None.")

    print(f"\nAll configs that passed relaxed threshold (max_error < {ACCURACY_THRESHOLD_2B}, matching Muon):")
    found_relaxed = False
    for exp_name, exp_results in ALL_RESULTS.items():
        if exp_name in ('gpu_benchmark', 'sigma_min_measurement'):
            continue
        for key, r in exp_results.items():
            err = r.get('max_error', float('inf'))
            if err < ACCURACY_THRESHOLD_2B:
                found_relaxed = True
                matmuls = r.get('matmuls', '?')
                print(f"  [{exp_name}] {r['name']}: {matmuls} matmuls, err={err:.8f}")
    if not found_relaxed:
        print("  None.")

    print("\nDone!")


if __name__ == "__main__":
    main()
