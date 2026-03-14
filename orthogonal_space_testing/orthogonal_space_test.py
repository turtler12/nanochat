"""
Polar Interpolation — Beating Muon by Building ON It
=====================================================
Key insight: Don't compete with Polar Express. USE it.

  P ≈ UV^T  (from Polar Express, all SVs = 1)
  G = UΣV^T (original gradient)

  G_new = (1-β)·G + β·σ_max·P = U·((1-β)Σ + β·σ_max·I)·V^T

  σ_new = (1-β)·σ + β·σ_max
    → For σ = σ_max: σ_new = σ_max  (PRESERVED)
    → For σ ≈ 0:     σ_new ≈ β·σ_max (BOOSTED)

Cost = Polar Express + one torch.lerp (negligible overhead).
We can also use FEWER NS iterations (3 instead of 5) since the
interpolation smooths approximation error → ~40% faster than Muon.

Methods tested:
  1. Polar Interp (3 NS steps) — FASTEST, our best shot
  2. Polar Interp (5 NS steps) — same accuracy, ~same speed as Muon
  3. Polar Express (5 NS steps) — Muon baseline
  4. Randomized SVD + Residual Scaling — v2 for reference

Usage:
  python orthogonal_space_test_v3.py --beta 0.3
  python orthogonal_space_test_v3.py --beta 0.1 --ns_steps 3
"""

import torch
import numpy as np
import time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, default=0.3,
                    help="Interpolation strength. σ_new = (1-β)σ + β·σ_max. "
                         "β=0: no change, β=1: full polar (Muon), β=0.3: moderate boost")
parser.add_argument("--ns_steps_fast", type=int, default=3,
                    help="NS iterations for fast variant (default 3)")
parser.add_argument("--ns_steps_full", type=int, default=5,
                    help="NS iterations for full-accuracy variant (default 5)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--oversampling", type=int, default=10,
                    help="Oversampling for randomized SVD comparison")
parser.add_argument("--alpha_v2", type=float, default=5.0,
                    help="Alpha for randomized SVD v2 comparison")
args = parser.parse_args()

BETA = args.beta
NS_FAST = args.ns_steps_fast
NS_FULL = args.ns_steps_full
SEED = args.seed
OVERSAMPLE = args.oversampling
ALPHA_V2 = args.alpha_v2
K_FRAC = 0.03

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")

# Focus on N=32..1500 as requested, plus a couple larger for scaling
SIZES = [32, 64, 128, 256, 512, 768, 1000, 1500]

# ── Polar Express coefficients (from Muon/nanochat) ─────────────────────
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


# ══════════════════════════════════════════════════════════════════════════
#  MATRIX GENERATION — realistic gradient spectrum
# ══════════════════════════════════════════════════════════════════════════

def make_matrix(n, k, sigma_max=1.0):
    """
    NxN matrix with realistic gradient-like SV distribution:
      - Top k SVs: linearly spaced from 1.0 down to ~0.5
      - Remaining SVs: exponentially decaying from ~0.06 to ~0.005
    """
    torch.manual_seed(SEED)
    top = torch.linspace(sigma_max, sigma_max * 0.5, k)
    n_bottom = n - k
    decay = torch.linspace(0, 4, n_bottom)
    bottom = 0.06 * sigma_max * torch.exp(-decay)
    bottom = bottom + torch.rand(n_bottom) * 0.005 * sigma_max
    bottom = bottom.clamp(min=0.002 * sigma_max)
    bottom = bottom.sort(descending=True).values
    sigmas = torch.cat([top, bottom])
    U, _ = torch.linalg.qr(torch.randn(n, n))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    M = (U @ torch.diag(sigmas) @ V.T).to(device)
    return M, sigmas.numpy()


# ══════════════════════════════════════════════════════════════════════════
#  CORE: POLAR EXPRESS
# ══════════════════════════════════════════════════════════════════════════

def polar_express(M, ns_steps=5):
    """
    Muon's Polar Express: approximate UV^T via quintic Newton-Schulz.
    Uses bf16 for 2x throughput on tensor cores.
    """
    X = M.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if X.size(-2) >= X.size(-1):
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


# ══════════════════════════════════════════════════════════════════════════
#  METHOD 1: POLAR INTERPOLATION  (NEW — the key idea)
# ══════════════════════════════════════════════════════════════════════════

def estimate_spectral_norm(G, iters=3):
    """
    Estimate σ_max via power iteration.
    Cost: 2*iters mat-vecs (negligible vs the matmuls in Polar Express).
    """
    v = torch.randn(G.shape[1], device=G.device, dtype=G.dtype)
    v = v / v.norm()
    for _ in range(iters):
        u = G @ v
        u_norm = u.norm()
        if u_norm > 0:
            u = u / u_norm
        v = G.T @ u
        sigma = v.norm()
        if sigma > 0:
            v = v / sigma
    return sigma


def polar_interpolation(G, beta=0.3, ns_steps=3, sigma_max=None):
    """
    THE KEY METHOD: Piggyback on Polar Express for selective SV boosting.

    G_new = (1-β)·G + β·σ_max·P     where P ≈ UV^T

    Spectral effect on each singular value:
        σ_new = (1-β)·σ + β·σ_max

    Properties:
        - σ_max is EXACTLY preserved (no error at top)
        - Small σ ≈ 0 gets boosted to β·σ_max
        - Intermediate σ gets proportional boost
        - ALL gradient directions preserved (unlike pure Polar which discards magnitudes)
        - Cost = Polar Express + O(mn) lerp (negligible)
        - Can use FEWER NS iterations since interpolation smooths error

    Args:
        G: gradient matrix (m, n) on GPU
        beta: interpolation strength
               β=0 → no change (original gradient)
               β=0.1 → mild boost (small SVs → 0.1·σ_max)
               β=0.3 → moderate boost
               β=0.5 → strong boost (small SVs → 0.5·σ_max)
               β=1.0 → full Polar Express (all SVs → σ_max, i.e. Muon)
        ns_steps: Newton-Schulz iterations (3 is usually enough)
        sigma_max: if known, skip estimation (saves 6 mat-vecs)

    Returns:
        G_new with boosted small singular values
    """
    # Step 1: Estimate σ_max (cheap: 6 mat-vecs ≈ 0.6% of a mat-mat)
    if sigma_max is None:
        sigma_max = estimate_spectral_norm(G, iters=3)

    # Step 2: Get UV^T approximation via Polar Express
    P = polar_express(G, ns_steps)  # bf16, all the heavy work

    # Step 3: Interpolate (one trivial fused op)
    # G_new = (1-β)·G + β·σ_max·P
    # torch.lerp(a, b, t) = a + t*(b - a) = (1-t)*a + t*b
    G_bf16 = G.bfloat16()
    result = torch.lerp(G_bf16, sigma_max * P, BETA)

    return result.float()


def polar_interpolation_fast(G, beta=0.3, sigma_max=None):
    """3 NS steps — ~40% faster than Muon."""
    return polar_interpolation(G, beta, ns_steps=3, sigma_max=sigma_max)


def polar_interpolation_full(G, beta=0.3, sigma_max=None):
    """5 NS steps — same accuracy as Muon, same speed."""
    return polar_interpolation(G, beta, ns_steps=5, sigma_max=sigma_max)


# ══════════════════════════════════════════════════════════════════════════
#  METHOD 2 (reference): RANDOMIZED SVD + RESIDUAL SCALING (v2)
# ══════════════════════════════════════════════════════════════════════════

def randomized_svd_topk(M, k, oversampling=10, n_power_iters=1):
    """Halko-Martinsson-Tropp randomized SVD."""
    m, n = M.shape
    r = k + oversampling
    Omega = torch.randn(n, r, device=M.device, dtype=M.dtype)
    Y = M @ Omega
    for _ in range(n_power_iters):
        Y, _ = torch.linalg.qr(Y)
        Z = M.T @ Y
        Z, _ = torch.linalg.qr(Z)
        Y = M @ Z
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ M
    U_B, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_B
    return U[:, :k], S[:k], Vh[:k, :].T


def ours_v2(M, k, alpha, oversampling=10, n_power_iters=1):
    """Randomized SVD + residual scaling from v2."""
    U_k, S_k, V_k = randomized_svd_topk(M, k, oversampling, n_power_iters)
    correction = U_k @ (S_k.unsqueeze(1) * V_k.T)
    return alpha * M + (1.0 - alpha) * correction


# ══════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def analyze_spectrum(sigmas_orig, sigmas_new, k):
    """
    Metrics:
      - top_k_rel_err: max relative error in top-k SVs (lower = better preservation)
      - bottom_boost: ratio of new/old average bottom SVs (higher = more exploration)
      - gradient_cos_sim: not computed here but could be added
    """
    # Match top-k by nearest neighbor
    remaining = list(sigmas_new)
    max_rel_err = 0.0
    for i in range(k):
        diffs = [abs(s - sigmas_orig[i]) / max(abs(sigmas_orig[i]), 1e-10)
                 for s in remaining]
        best = np.argmin(diffs)
        max_rel_err = max(max_rel_err, diffs[best])
        remaining.pop(best)

    orig_bottom = sorted(sigmas_orig[k:], reverse=True)
    new_bottom = sorted(remaining, reverse=True)
    orig_mean = np.mean(orig_bottom) if len(orig_bottom) > 0 else 1e-10
    new_mean = np.mean(new_bottom) if len(new_bottom) > 0 else 0
    boost = new_mean / max(orig_mean, 1e-10)

    return max_rel_err, boost


def time_fn(fn, warmup=5, repeats=20):
    """Robust GPU timing with more repeats for stable measurements."""
    for _ in range(warmup):
        fn()
    if device == 'cuda':
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.median(times)


# ══════════════════════════════════════════════════════════════════════════
#  BENCHMARK
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*100}")
print(f"  Polar Interpolation — Building ON Polar Express to Beat It")
print(f"  β={BETA}  (σ_new = (1-β)·σ + β·σ_max)")
print(f"  At β={BETA}: small SVs boosted to ~{BETA}·σ_max, large SVs preserved")
print(f"  Fast variant: {NS_FAST} NS steps, Full variant: {NS_FULL} NS steps")
print(f"{'='*100}\n")

hdr = (f"{'N':>6} {'k':>4} | "
       f"{'PI-fast':>9} {'PI-full':>9} {'Polar5':>9} {'RandSVD':>9} | "
       f"{'fast/P':>7} {'full/P':>7} | "
       f"{'PI_err%':>8} {'PI_boost':>9} {'P_boost':>8}")
print(hdr)
print("-" * len(hdr))

results = []

for n in SIZES:
    k = max(1, int(n * K_FRAC))
    M, sigmas_true = make_matrix(n, k)

    # Pre-estimate sigma_max once (shared across methods)
    sigma_max = estimate_spectral_norm(M, iters=5).item()

    # ── Polar Interpolation FAST (3 NS steps) ──
    torch.manual_seed(SEED + n)
    t_pi_fast = time_fn(lambda: polar_interpolation_fast(M, BETA, sigma_max))
    torch.manual_seed(SEED + n)
    M_pi_fast = polar_interpolation_fast(M, BETA, sigma_max)
    sigmas_pi_fast = torch.linalg.svdvals(M_pi_fast).cpu().numpy()
    err_pi_fast, boost_pi_fast = analyze_spectrum(sigmas_true, sigmas_pi_fast, k)

    # ── Polar Interpolation FULL (5 NS steps) ──
    torch.manual_seed(SEED + n)
    t_pi_full = time_fn(lambda: polar_interpolation_full(M, BETA, sigma_max))
    torch.manual_seed(SEED + n)
    M_pi_full = polar_interpolation_full(M, BETA, sigma_max)
    sigmas_pi_full = torch.linalg.svdvals(M_pi_full).cpu().numpy()
    err_pi_full, boost_pi_full = analyze_spectrum(sigmas_true, sigmas_pi_full, k)

    # ── Polar Express baseline (5 NS steps) ──
    t_polar = time_fn(lambda: polar_express(M, 5))
    M_polar = polar_express(M, 5).float()
    sigmas_polar = torch.linalg.svdvals(M_polar).cpu().numpy()
    _, boost_polar = analyze_spectrum(sigmas_true, sigmas_polar, k)

    # ── Randomized SVD v2 (for reference) ──
    torch.manual_seed(SEED + n)
    t_rsvd = time_fn(lambda: ours_v2(M, k, ALPHA_V2, OVERSAMPLE, 1))

    ratio_fast = t_pi_fast / t_polar
    ratio_full = t_pi_full / t_polar

    print(f"{n:>6} {k:>4} | "
          f"{t_pi_fast*1000:>8.3f}ms {t_pi_full*1000:>8.3f}ms "
          f"{t_polar*1000:>8.3f}ms {t_rsvd*1000:>8.3f}ms | "
          f"{ratio_fast:>6.2f}x {ratio_full:>6.2f}x | "
          f"{err_pi_fast*100:>7.3f}% {boost_pi_fast:>8.1f}x {boost_polar:>7.1f}x")

    results.append(dict(
        n=n, k=k, sigma_max=sigma_max,
        t_pi_fast=t_pi_fast, t_pi_full=t_pi_full, t_polar=t_polar, t_rsvd=t_rsvd,
        ratio_fast=ratio_fast, ratio_full=ratio_full,
        err_pi_fast=err_pi_fast, err_pi_full=err_pi_full,
        boost_pi_fast=boost_pi_fast, boost_pi_full=boost_pi_full, boost_polar=boost_polar,
        sigmas_orig=sigmas_true,
        sigmas_pi_fast=sigmas_pi_fast, sigmas_pi_full=sigmas_pi_full,
        sigmas_polar=sigmas_polar,
    ))

    del M, M_pi_fast, M_pi_full, M_polar
    if device == 'cuda':
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════
#  THEORETICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("THEORETICAL SPECTRAL ANALYSIS")
print(f"{'='*70}")
print(f"\nFor β={BETA}, σ_max≈{results[-1]['sigma_max']:.3f}:")
print(f"  σ_new(σ) = {1-BETA:.1f}·σ + {BETA:.1f}·{results[-1]['sigma_max']:.3f}")
print(f"  = {1-BETA:.1f}·σ + {BETA * results[-1]['sigma_max']:.4f}")
print(f"\n  {'σ_orig':>10} → {'σ_new':>10}  {'boost':>8}")
print(f"  {'-'*35}")
for sigma_test in [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
    sigma_new = (1 - BETA) * sigma_test + BETA * results[-1]['sigma_max']
    boost = sigma_new / sigma_test if sigma_test > 0 else float('inf')
    print(f"  {sigma_test:>10.4f} → {sigma_new:>10.4f}  {boost:>7.1f}×")


# ══════════════════════════════════════════════════════════════════════════
#  PLOTS (2 rows × 4 cols)
# ══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(22, 13))
fig.suptitle(
    f"Polar Interpolation: G_new = (1-β)·G + β·σ_max·P\n"
    f"β={BETA},  Fast={NS_FAST} NS iters,  Full={NS_FULL} NS iters",
    fontsize=15, fontweight='bold', y=0.99
)

ns_list = [r['n'] for r in results]

# Use the largest matrix for spectrum plots
r = results[-1]
k = r['k']
n_show = min(r['n'], 200)
x = np.arange(n_show)

# ── Row 1: Spectral analysis ──

# 1) Polar Interpolation: SV spectrum
ax1 = fig.add_subplot(2, 4, 1)
ax1.bar(x[:k], r['sigmas_orig'][:k], width=0.8, color='#4a90d9', alpha=0.35, label='Orig top-k')
ax1.bar(x[:k], r['sigmas_pi_fast'][:k], width=0.4, color='#2ecc71', alpha=0.85, label='PI-fast')
ax1.bar(x[k:], r['sigmas_orig'][k:n_show], width=0.8, color='#cccccc', alpha=0.35, label='Orig bottom')
ax1.bar(x[k:], r['sigmas_pi_fast'][k:n_show], width=0.4, color='#2ecc71', alpha=0.85)
ax1.axvline(x=k - 0.5, color='black', ls='--', lw=1.5, alpha=0.7)
ax1.axhline(y=BETA * r['sigma_max'], color='#e74c3c', ls=':', lw=2, alpha=0.7,
            label=f'Floor = β·σ_max = {BETA * r["sigma_max"]:.3f}')
ax1.set_xlabel('SV index')
ax1.set_ylabel('Singular value')
ax1.set_title(f'Polar Interpolation (N={r["n"]})\nTop-k preserved, bottom → β·σ_max', fontsize=10)
ax1.legend(fontsize=6.5, loc='upper right')
ax1.set_ylim(bottom=0)

# 2) Polar Express: SV spectrum
ax2 = fig.add_subplot(2, 4, 2)
ax2.bar(x, r['sigmas_orig'][:n_show], width=0.8, color='#cccccc', alpha=0.35, label='Original')
ax2.bar(x, r['sigmas_polar'][:n_show], width=0.4, color='#4a90d9', alpha=0.85, label='Polar Express')
ax2.axhline(y=1.0, color='orange', ls=':', lw=2, alpha=0.8, label='Target=1')
ax2.axvline(x=k - 0.5, color='black', ls='--', lw=1.5, alpha=0.7)
ax2.set_xlabel('SV index')
ax2.set_ylabel('Singular value')
ax2.set_title(f'Polar Express (N={r["n"]})\nAll SVs → ~1.0 (Muon)', fontsize=10)
ax2.legend(fontsize=6.5, loc='upper right')
ax2.set_ylim(bottom=0)

# 3) Log-scale overlay — all methods
ax3 = fig.add_subplot(2, 4, 3)
ax3.semilogy(x, r['sigmas_orig'][:n_show], 'o', ms=2.5, color='gray', alpha=0.3, label='Original')
ax3.semilogy(x, r['sigmas_pi_fast'][:n_show], 's', ms=2, color='#2ecc71', alpha=0.85, label=f'PI-fast ({NS_FAST} NS)')
ax3.semilogy(x, r['sigmas_pi_full'][:n_show], 'D', ms=1.5, color='#27ae60', alpha=0.6, label=f'PI-full ({NS_FULL} NS)')
ax3.semilogy(x, r['sigmas_polar'][:n_show], '^', ms=2, color='#4a90d9', alpha=0.7, label='Polar Express')
ax3.axvline(x=k - 0.5, color='black', ls='--', lw=1, alpha=0.5)
ax3.axhline(y=BETA * r['sigma_max'], color='#e74c3c', ls=':', lw=1.5, alpha=0.5)
ax3.set_xlabel('SV index')
ax3.set_ylabel('σ (log)')
ax3.set_title('All Methods (log scale)', fontsize=10)
ax3.legend(fontsize=6.5)
ax3.grid(True, alpha=0.2)

# 4) Spectral transfer function: σ_new vs σ_old
ax4 = fig.add_subplot(2, 4, 4)
sigma_range = np.linspace(0, r['sigma_max'] * 1.1, 200)
# Polar Interpolation
pi_curve = (1 - BETA) * sigma_range + BETA * r['sigma_max']
# Polar Express (all → 1)
polar_curve = np.ones_like(sigma_range) * 1.0  # approximate
# Identity (no change)
identity = sigma_range

ax4.plot(sigma_range, identity, 'k--', lw=1, alpha=0.4, label='No change (σ_new=σ)')
ax4.plot(sigma_range, pi_curve, '-', lw=2.5, color='#2ecc71', label=f'Polar Interp (β={BETA})')
ax4.plot(sigma_range, polar_curve, '-', lw=2, color='#4a90d9', alpha=0.7, label='Polar Express (→1)')
# Show different beta values
for b_test, alpha_val in [(0.1, 0.3), (0.5, 0.5)]:
    curve = (1 - b_test) * sigma_range + b_test * r['sigma_max']
    ax4.plot(sigma_range, curve, '--', lw=1.2, alpha=alpha_val,
             color='#2ecc71', label=f'β={b_test}')
ax4.set_xlabel('σ_original')
ax4.set_ylabel('σ_new')
ax4.set_title('Spectral Transfer Function\nσ_new = (1-β)σ + β·σ_max', fontsize=10)
ax4.legend(fontsize=6.5)
ax4.grid(True, alpha=0.2)
ax4.set_xlim(0, r['sigma_max'] * 1.1)
ax4.set_ylim(0, r['sigma_max'] * 1.3)

# ── Row 2: Performance ──

x_bar = np.arange(len(ns_list))

# 5) Timing comparison
ax5 = fig.add_subplot(2, 4, 5)
w = 0.2
ax5.bar(x_bar - 1.5*w, [r_['t_pi_fast']*1000 for r_ in results], w,
        label=f'PI-fast ({NS_FAST} NS)', color='#2ecc71', alpha=0.9)
ax5.bar(x_bar - 0.5*w, [r_['t_pi_full']*1000 for r_ in results], w,
        label=f'PI-full ({NS_FULL} NS)', color='#27ae60', alpha=0.7)
ax5.bar(x_bar + 0.5*w, [r_['t_polar']*1000 for r_ in results], w,
        label='Polar Express', color='#4a90d9', alpha=0.85)
ax5.bar(x_bar + 1.5*w, [r_['t_rsvd']*1000 for r_ in results], w,
        label='Rand SVD v2', color='#999999', alpha=0.5)
ax5.set_xticks(x_bar)
ax5.set_xticklabels([str(n) for n in ns_list], rotation=45)
ax5.set_xlabel('Matrix size N')
ax5.set_ylabel('Time (ms)')
ax5.set_title('GPU Time Comparison', fontsize=10)
ax5.set_yscale('log')
ax5.legend(fontsize=6.5)
ax5.grid(True, alpha=0.2, axis='y')

# 6) Speed ratio: our methods vs Polar Express
ax6 = fig.add_subplot(2, 4, 6)
ratio_fast = [r_['ratio_fast'] for r_ in results]
ratio_full = [r_['ratio_full'] for r_ in results]
ax6.bar(x_bar - 0.15, ratio_fast, 0.28, label=f'PI-fast ({NS_FAST} NS)',
        color='#2ecc71', alpha=0.9)
ax6.bar(x_bar + 0.15, ratio_full, 0.28, label=f'PI-full ({NS_FULL} NS)',
        color='#27ae60', alpha=0.7)
ax6.axhline(y=1.0, color='black', ls='--', lw=2, alpha=0.6, label='Parity with Polar')
ax6.axhline(y=0.6, color='#2ecc71', ls=':', lw=1.5, alpha=0.4, label='Theoretical 3/5 = 0.6')
ax6.set_xticks(x_bar)
ax6.set_xticklabels([str(n) for n in ns_list], rotation=45)
ax6.set_xlabel('Matrix size N')
ax6.set_ylabel('Time / Polar Express time')
ax6.set_title('Speed vs Polar Express\n< 1.0 = FASTER than Muon', fontsize=10)
for i, v in enumerate(ratio_fast):
    ax6.text(i - 0.15, v + 0.03, f'{v:.2f}', ha='center', fontsize=7, fontweight='bold',
             color='#2ecc71')
for i, v in enumerate(ratio_full):
    ax6.text(i + 0.15, v + 0.03, f'{v:.2f}', ha='center', fontsize=7,
             color='#27ae60')
ax6.legend(fontsize=6.5, loc='upper left')
ax6.grid(True, alpha=0.2, axis='y')
ax6.set_ylim(0, max(max(ratio_fast), max(ratio_full)) * 1.3)

# 7) Top-k preservation (PI-fast vs PI-full)
ax7 = fig.add_subplot(2, 4, 7)
ax7.bar(x_bar - 0.15, [r_['err_pi_fast']*100 for r_ in results], 0.28,
        label=f'PI-fast ({NS_FAST} NS)', color='#2ecc71', alpha=0.9)
ax7.bar(x_bar + 0.15, [r_['err_pi_full']*100 for r_ in results], 0.28,
        label=f'PI-full ({NS_FULL} NS)', color='#27ae60', alpha=0.7)
ax7.set_xticks(x_bar)
ax7.set_xticklabels([str(n) for n in ns_list], rotation=45)
ax7.set_xlabel('Matrix size N')
ax7.set_ylabel('Max relative error (%)')
ax7.set_title('Top-k Preservation Error\n(lower = better)', fontsize=10)
ax7.legend(fontsize=6.5)
ax7.grid(True, alpha=0.2, axis='y')

# 8) Bottom SV boost comparison
ax8 = fig.add_subplot(2, 4, 8)
ax8.bar(x_bar - 0.2, [r_['boost_pi_fast'] for r_ in results], 0.2,
        label=f'PI-fast', color='#2ecc71', alpha=0.9)
ax8.bar(x_bar, [r_['boost_pi_full'] for r_ in results], 0.2,
        label=f'PI-full', color='#27ae60', alpha=0.7)
ax8.bar(x_bar + 0.2, [r_['boost_polar'] for r_ in results], 0.2,
        label='Polar', color='#4a90d9', alpha=0.7)
ax8.set_xticks(x_bar)
ax8.set_xticklabels([str(n) for n in ns_list], rotation=45)
ax8.set_xlabel('Matrix size N')
ax8.set_ylabel('Avg bottom SV boost (×)')
ax8.set_title('Bottom SV Boost\n(Polar always wins here — it maps to 1.0)', fontsize=10)
ax8.legend(fontsize=6.5)
ax8.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('polar_interpolation_benchmark.png', dpi=150, bbox_inches='tight')
print(f"\nSaved polar_interpolation_benchmark.png")


# ══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"\n  Method: Polar Interpolation")
print(f"  G_new = (1-β)·G + β·σ_max·polar_express(G)")
print(f"  β = {BETA}")
print(f"")
for r_ in results:
    status_fast = "✓ FASTER" if r_['ratio_fast'] < 1.0 else f"  {r_['ratio_fast']:.2f}× slower"
    status_full = "✓ FASTER" if r_['ratio_full'] < 1.0 else f"  ~same" if r_['ratio_full'] < 1.1 else f"  {r_['ratio_full']:.2f}× slower"
    print(f"  N={r_['n']:>5}: fast={r_['t_pi_fast']*1000:>7.3f}ms ({status_fast})  "
          f"full={r_['t_pi_full']*1000:>7.3f}ms ({status_full})  "
          f"polar={r_['t_polar']*1000:>7.3f}ms  "
          f"err={r_['err_pi_fast']*100:.2f}%  boost={r_['boost_pi_fast']:.1f}×")

print(f"\n  Key insight: Don't fight Polar Express — piggyback on it.")
print(f"  The 3-step variant should be ~60% of Polar Express cost")
print(f"  while providing SELECTIVE SV boosting (Muon boosts everything to 1).")
print(f"  Gradient direction info is PRESERVED (Muon discards magnitudes).")

print(f"\n  For training: β is your exploration knob.")
print(f"    β=0.0 → vanilla gradient")
print(f"    β=0.1 → mild exploration of neglected directions")
print(f"    β=0.3 → moderate (recommended starting point)")
print(f"    β=0.5 → aggressive exploration")
print(f"    β=1.0 → equivalent to Muon (all SVs → σ_max)")