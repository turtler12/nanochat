#!/usr/bin/env python3
"""Generate comprehensive analysis plots from poly_search results."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

with open("results.json") as f:
    results = json.load(f)

# Sort by matmuls
results.sort(key=lambda x: x['matmuls'])

# Filter out diverged configs for the main plot
valid = [r for r in results if r['max_error'] < 1e9]
diverged = [r for r in results if r['max_error'] >= 1e9]

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# ── Plot 1: Max error vs matmuls ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter([r['matmuls'] for r in valid], [r['max_error'] for r in valid],
            c='#e74c3c', s=120, zorder=5, edgecolors='black', linewidths=0.5)
for r in valid:
    ax1.annotate(r['name'], (r['matmuls'], r['max_error']),
                textcoords="offset points", xytext=(5, 8), fontsize=7.5,
                ha='left', rotation=15)

# Mark diverged
for r in diverged:
    ax1.scatter(r['matmuls'], 10, c='black', s=120, zorder=5, marker='x', linewidths=2)
    ax1.annotate(f"{r['name']}\n(diverged)", (r['matmuls'], 10),
                textcoords="offset points", xytext=(5, 8), fontsize=7, color='gray',
                ha='left')

ax1.axhline(y=0.01, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.01)')
ax1.axvline(x=15, color='blue', linestyle=':', linewidth=1.5, alpha=0.5, label='Muon baseline (15 mm)')
ax1.set_xlabel('Total MatMuls', fontsize=11)
ax1.set_ylabel('Max |p(σ) - 1|', fontsize=11)
ax1.set_title('Minimax Error vs Computational Cost', fontsize=12, fontweight='bold')
ax1.set_yscale('log')
ax1.set_ylim(0.005, 15)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2)

# ── Plot 2: Error vs number of iterations ──
ax2 = fig.add_subplot(gs[0, 1])
for r in valid:
    n_iters = len(r['degrees'])
    max_deg = max(r['degrees'])
    ax2.scatter(n_iters, r['max_error'], s=120, zorder=5, edgecolors='black', linewidths=0.5,
               c=plt.cm.viridis(max_deg / 13.0))
    ax2.annotate(r['name'], (n_iters, r['max_error']),
                textcoords="offset points", xytext=(5, 5), fontsize=7.5)

ax2.axhline(y=0.01, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax2.set_xlabel('Number of Iterations', fontsize=11)
ax2.set_ylabel('Max |p(σ) - 1|', fontsize=11)
ax2.set_title('Error vs Iteration Count\n(color = max polynomial degree)', fontsize=12, fontweight='bold')
ax2.set_yscale('log')
ax2.set_ylim(0.005, 15)
ax2.grid(True, alpha=0.2)

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=5, vmax=13))
sm.set_array([])
plt.colorbar(sm, ax=ax2, label='Max degree', shrink=0.8)

# ── Plot 3: Error breakdown by degree type ──
ax3 = fig.add_subplot(gs[1, 0])

categories = {
    'Pure quintic': [r for r in valid if all(d == 5 for d in r['degrees'])],
    'Pure septic': [r for r in valid if all(d == 7 for d in r['degrees'])],
    'Mixed': [r for r in valid if len(set(r['degrees'])) > 1 or r['degrees'][0] not in [5,7]],
}
colors = {'Pure quintic': '#3498db', 'Pure septic': '#e67e22', 'Mixed': '#9b59b6'}

x_pos = 0
tick_labels = []
tick_positions = []
for cat_name, cat_results in categories.items():
    for r in sorted(cat_results, key=lambda x: x['matmuls']):
        bar = ax3.bar(x_pos, r['max_error'], color=colors[cat_name], edgecolor='black',
                     linewidth=0.5, width=0.7, label=cat_name if x_pos == 0 or (cat_name == 'Pure septic' and x_pos == 2) or (cat_name == 'Mixed' and x_pos == 4) else '')
        tick_labels.append(f"{r['name']}\n({r['matmuls']}mm)")
        tick_positions.append(x_pos)
        x_pos += 1

ax3.axhline(y=0.01, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target')
ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha='right')
ax3.set_ylabel('Max Error', fontsize=11)
ax3.set_title('Error by Polynomial Type', fontsize=12, fontweight='bold')
ax3.set_yscale('log')
ax3.legend(fontsize=8, loc='upper right')
ax3.grid(True, alpha=0.2, axis='y')

# ── Plot 4: Gap analysis text ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

best = min(valid, key=lambda x: x['max_error'])
gap = best['max_error'] / 0.01

text = f"""EXPERIMENT SUMMARY
{'─'*40}

Configs tested:     9
Passed (err<0.01):  0  ← none passed
Diverged:           2  (3×septic+1×quintic, 4×septic)
Valid but failed:   7

Best result:
  {best['name']}
  {best['matmuls']} matmuls ({best['savings_pct']:.0f}% savings)
  max error = {best['max_error']:.4f}
  gap to target = {gap:.0f}× too high

Muon baseline:
  5×quintic, 15 matmuls
  max error ≈ 0.002 (from paper)

Key finding:
  The minimum matmul count for 0.01
  accuracy appears to be ≥15.
  Every config with <15 matmuls failed
  by 50-90× the threshold.

Optimization budget:
  Total CPU time: {sum(r['optimization_time'] for r in results)/60:.0f} min
  Seeds per config: 15
  Optimizer: DE + Nelder-Mead + L-BFGS-B
"""

ax4.text(0.05, 0.95, text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))

plt.suptitle('Polynomial Schedule Search for Newton-Schulz Orthogonalization',
             fontsize=14, fontweight='bold', y=0.98)
plt.savefig('plots/full_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved plots/full_analysis.png")

# ── Also make a combined convergence plot ──
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

sigma = np.logspace(np.log10(0.001), np.log10(0.98), 5000)

def eval_single_poly(s, coeffs):
    """coeffs = full [a1, a2, a3, ...]"""
    result = coeffs[0] * s
    s_sq = s ** 2
    power = s_sq.copy()
    for c in coeffs[1:]:
        result += c * s * power
        power *= s_sq
    return result

def eval_composed(s, all_coeffs):
    result = s.copy()
    for coeffs in all_coeffs:
        result = eval_single_poly(result, coeffs)
    return result

for idx, r in enumerate(results):
    ax = axes[idx]
    if r['coefficients'] is not None and r['max_error'] < 1e9:
        composed = eval_composed(sigma, r['coefficients'])
        error = np.abs(composed - 1.0)
        ax.semilogy(sigma, error, 'b-', linewidth=1, alpha=0.8)
        ax.axhline(y=0.01, color='r', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_ylim(1e-5, 2)
        color = '#e74c3c'
    else:
        ax.text(0.5, 0.5, 'DIVERGED', transform=ax.transAxes, ha='center', va='center',
               fontsize=16, color='red', fontweight='bold')
        color = 'gray'

    ax.set_title(f"{r['name']}\n{r['matmuls']}mm | err={r['max_error']:.4f}" if r['max_error'] < 1e9
                 else f"{r['name']}\n{r['matmuls']}mm | DIVERGED",
                 fontsize=9, color=color, fontweight='bold')
    ax.set_xlabel('σ', fontsize=8)
    ax.set_ylabel('|p(σ)-1|', fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)

plt.suptitle('Convergence Profiles: All Configurations', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('plots/all_convergence.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved plots/all_convergence.png")
