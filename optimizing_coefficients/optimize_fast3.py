#!/usr/bin/env python3
"""
Optimize 3×quintic NS coefficients for Experiment A.

Run on login node (CPU only, ~2-5 minutes):
    python optimizing_coefficients/optimize_fast3.py

Outputs fast3 coefficients for σ_lb ∈ {0.02, 0.03, 0.05}.
Paste the best result into nanochat/optim_ablation.py as fast3_coeffs.
"""

import numpy as np
from scipy.optimize import minimize


def eval_quintic(sigma, b, c):
    a = 1.0 - b - c
    return a * sigma + b * sigma**3 + c * sigma**5


def sequential_greedy(sigma_lb, n_iters=3, n_grid=500):
    sigma = np.logspace(np.log10(sigma_lb), np.log10(0.98), 5000)
    sigma_cur = sigma.copy()
    coeffs = []

    for i in range(n_iters):
        best_err = 1e9
        best_b, best_c = 0.0, 0.0

        # Vectorized 2D grid search
        b_vals = np.linspace(-30, 10, n_grid)
        c_vals = np.linspace(-20, 20, n_grid)
        s1 = sigma_cur[np.newaxis, np.newaxis, :]
        s3 = sigma_cur[np.newaxis, np.newaxis, :] ** 3
        s5 = sigma_cur[np.newaxis, np.newaxis, :] ** 5

        chunk = 50
        for ci in range(0, n_grid, chunk):
            b_chunk = b_vals[ci:ci + chunk]
            b_g = b_chunk[:, np.newaxis, np.newaxis]
            c_g = c_vals[np.newaxis, :, np.newaxis]
            a_g = 1.0 - b_g - c_g
            p = a_g * s1 + b_g * s3 + c_g * s5
            stable = np.all((p >= -0.01) & (p <= 1.5), axis=2)
            errs = np.max(np.abs(p - 1.0), axis=2)
            errs[~stable] = 1e10
            idx = np.unravel_index(np.argmin(errs), errs.shape)
            if errs[idx] < best_err:
                best_err = errs[idx]
                best_b = b_chunk[idx[0]]
                best_c = c_vals[idx[1]]

        # Nelder-Mead refinement
        def obj(x):
            p = eval_quintic(sigma_cur, x[0], x[1])
            if np.any(p < -0.5) or np.any(p > 2.0):
                return 1e9
            return float(np.max(np.abs(p - 1.0)))

        r = minimize(obj, [best_b, best_c], method="Nelder-Mead",
                     options={"maxiter": 50000, "xatol": 1e-14, "fatol": 1e-14, "adaptive": True})
        best_b, best_c = r.x
        best_err = float(r.fun)
        a = 1.0 - best_b - best_c
        coeffs.append((a, best_b, best_c))
        sigma_cur = eval_quintic(sigma_cur, best_b, best_c)
        print(f"  iter {i + 1}: a={a:.6f}, b={best_b:.6f}, c={best_c:.6f}  err_so_far={best_err:.4e}")

    final_err = float(np.max(np.abs(sigma_cur - 1.0)))
    return coeffs, final_err


def main():
    results = {}
    for lb in [0.02, 0.03, 0.05]:
        print(f"\n{'=' * 60}")
        print(f"σ_lb = {lb}  (3×quintic)")
        print("=" * 60)
        coeffs, err = sequential_greedy(lb, n_iters=3)
        results[lb] = (coeffs, err)
        print(f"  Final max error on [σ_lb={lb}, 0.98]: {err:.6e}")
        print(f"  fast3_coeffs (σ_lb={lb}) = [")
        for a, b, c in coeffs:
            print(f"      ({a:.6f}, {b:.6f}, {c:.6f}),")
        print("  ]")

    best_lb = min(results, key=lambda k: results[k][1])
    best_coeffs, best_err = results[best_lb]
    print(f"\n{'=' * 60}")
    print(f"Best: σ_lb={best_lb}  max_err={best_err:.6e}")
    print("Paste into nanochat/optim_ablation.py as fast3_coeffs:")
    print("fast3_coeffs = [")
    for a, b, c in best_coeffs:
        print(f"    ({a:.6f}, {b:.6f}, {c:.6f}),")
    print("]")


if __name__ == "__main__":
    main()
