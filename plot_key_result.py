#!/usr/bin/env python3
"""Key result plot: 4×quintic with σ_lb=0.02 vs Muon baseline."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Load data ──
with open("results_v2.json") as f:
    data = json.load(f)

baseline_coeffs = data["experiment_4"]["5iter_lb0.001"]["coefficients"]
winner_coeffs = data["experiment_4"]["4iter_lb0.02"]["coefficients"]

def eval_poly(sigma, coeffs):
    result = np.zeros_like(sigma)
    for i, c in enumerate(coeffs):
        result += c * sigma ** (2 * i + 1)
    return result

def eval_composed(sigma, all_coeffs):
    result = sigma.copy()
    for coeffs in all_coeffs:
        result = eval_poly(result, coeffs)
    return result

sigma_min_measured = data["sigma_min_measurement"]["4096"]["sigma_min_min"]  # ~0.029
sigma_max_measured = data["sigma_min_measurement"]["4096"]["sigma_max_max"]  # ~0.98

# ── Figure ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={'width_ratios': [2.5, 1, 1]})

# ── Panel A: Polynomial output on the gradient spectrum ──
ax1 = axes[0]

# Focus on the range that matters: [0.02, 0.98]
sigma = np.linspace(0.015, 0.98, 5000)
baseline_out = eval_composed(sigma, baseline_coeffs)
winner_out = eval_composed(sigma, winner_coeffs)

ax1.plot(sigma, winner_out, color='#2E86C1', linewidth=2.2,
         label='Ours: 4×quintic, σ_lb=0.02  (12 matmuls)', zorder=3)
ax1.plot(sigma, baseline_out, color='#7B68AE', linewidth=2, alpha=0.75,
         label='Muon: 5×quintic  (15 matmuls)', zorder=2)
ax1.axhline(y=1.0, color='#888', linestyle='--', linewidth=1, alpha=0.4)

# Shade measured gradient spectrum
ax1.axvspan(sigma_min_measured, sigma_max_measured, alpha=0.06, color='#27AE60', zorder=0)
ax1.axvline(x=sigma_min_measured, color='#27AE60', linestyle='-', linewidth=1, alpha=0.5)
ax1.text(sigma_min_measured + 0.005, 0.45, f'σ_min = {sigma_min_measured:.3f}\n(measured)',
         fontsize=7.5, color='#1E8449', va='center')

ax1.axvline(x=0.02, color='#E74C3C', linestyle=':', linewidth=1.5, alpha=0.6)
ax1.text(0.021, 0.3, 'σ_lb = 0.02\n(design bound)',
         fontsize=7.5, color='#C0392B', va='center')

ax1.set_xlabel('Singular value σ')
ax1.set_ylabel('Composed polynomial p(σ)  →  should be 1')
ax1.set_title('A.  Newton-Schulz output on gradient spectrum', fontweight='bold', loc='left')
ax1.legend(fontsize=8.5, loc='center right')
ax1.set_xlim(0.015, 0.98)
ax1.set_ylim(0.0, 1.25)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Panel B: Speed ──
ax2 = axes[1]
size = "2048"
baseline_time = data["gpu_benchmark"]["Muon 5×quintic baseline (15mm)"]["timings_ms"][size]
winner_time = data["gpu_benchmark"]["4×quintic σ≥0.02 (12mm)"]["timings_ms"][size]
speedup = baseline_time / winner_time

colors = ['#7B68AE', '#2E86C1']
bars = ax2.bar([0, 1], [baseline_time, winner_time],
               color=colors, width=0.55, edgecolor='white', linewidth=1.5)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Muon\n(15 matmuls)', 'Ours\n(12 matmuls)'], fontsize=9)
ax2.set_ylabel('Wall-clock time (ms)')
ax2.set_title('B.  H100 Speed (2048²)', fontweight='bold', loc='left')

for bar, t in zip(bars, [baseline_time, winner_time]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
             f'{t:.2f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Speedup callout
ax2.text(0.5, winner_time * 0.45, f'{speedup:.2f}×\nfaster',
         fontsize=14, fontweight='bold', color='#27AE60', ha='center', va='center')

ax2.set_ylim(0, baseline_time * 1.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ── Panel C: BF16 accuracy ──
ax3 = axes[2]
baseline_bf16 = data["gpu_benchmark"]["Muon 5×quintic baseline (15mm)"]["bf16_errors"][size]
winner_bf16 = data["gpu_benchmark"]["4×quintic σ≥0.02 (12mm)"]["bf16_errors"][size]

bars = ax3.bar([0, 1], [baseline_bf16, winner_bf16],
               color=colors, width=0.55, edgecolor='white', linewidth=1.5)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Muon\n(15 matmuls)', 'Ours\n(12 matmuls)'], fontsize=9)
ax3.set_ylabel('Max |σ_out − 1|  (bf16)')
ax3.set_title('C.  BF16 Accuracy (2048²)', fontweight='bold', loc='left')

for bar, e in zip(bars, [baseline_bf16, winner_bf16]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{e:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ratio = baseline_bf16 / winner_bf16
ax3.text(0.5, baseline_bf16 * 0.4, f'{ratio:.0f}× lower\nerror',
         fontsize=13, fontweight='bold', color='#27AE60', ha='center', va='center')

ax3.set_ylim(0, baseline_bf16 * 1.4)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ── Title ──
fig.suptitle('Optimized Newton-Schulz: 1.24× faster, 9× more accurate, 12 vs 15 matmuls',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('plots_v2/key_result.png', dpi=180, bbox_inches='tight', facecolor='white')
print("Saved plots_v2/key_result.png")
