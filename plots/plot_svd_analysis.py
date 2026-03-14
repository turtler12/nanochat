"""
Comprehensive SVD spectrum analysis for baseline Muon training.
Generates plots to understand the implicit geometry Muon prefers.
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

SVD_DIR = "cache/base_checkpoints/baseline_muon/svd_logs"
OUT_DIR = "plots/baseline_muon"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

# ─────────────────────────────────────────────────────────────────────────────
# Load all data
# ─────────────────────────────────────────────────────────────────────────────

def load_svd_files(prefix):
    """Load all SVD json files of a given type (weight/gradient/update)."""
    files = sorted(glob.glob(os.path.join(SVD_DIR, f"{prefix}_svd_step*.json")))
    data = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        data[d['step']] = d['spectra']
    return data

weight_data = load_svd_files("weight")
gradient_data = load_svd_files("gradient")
update_data = load_svd_files("update")
steps = sorted(weight_data.keys())

print(f"Loaded {len(steps)} snapshots: steps {steps}")

# Get all matrix names, excluding tiny ve_gate matrices
all_names = sorted([n for n in weight_data[steps[0]].keys() if 've_gate' not in n])

# Categorize by type
attn_types = ['c_q', 'c_k', 'c_v', 'c_proj']
mlp_types = ['c_fc', 'c_proj']

def get_layer_and_type(name):
    """Parse 'transformer.h.5.attn.c_q.weight' -> (5, 'attn.c_q')"""
    parts = name.replace('.weight', '').split('.')
    layer = int(parts[2])
    subtype = '.'.join(parts[3:])
    return layer, subtype

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Stable rank over training (Stiefel proximity)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Stable Rank of Weight Matrices Over Training\n(Stiefel manifold → stable_rank = min(m,n))", fontsize=14, fontweight='bold')

subtypes = ['attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
subtype_titles = ['Attention Q', 'Attention K', 'Attention V', 'Attention Out Proj', 'MLP Up (c_fc)', 'MLP Down (c_proj)']

for ax_idx, (subtype, title) in enumerate(zip(subtypes, subtype_titles)):
    ax = axes.flat[ax_idx]
    colors = cm.viridis(np.linspace(0, 1, 20))

    for layer in range(20):
        name = f"transformer.h.{layer}.{subtype}.weight"
        if name not in weight_data[steps[0]]:
            continue
        stable_ranks = [weight_data[s][name]['stable_rank'] for s in steps]
        ax.plot(steps, stable_ranks, color=colors[layer], alpha=0.7, linewidth=1.2, label=f"L{layer}" if layer % 5 == 0 else None)

    # Reference line: min(m,n) for full-rank orthogonal
    shape = weight_data[steps[0]][name]['shape']
    min_dim = min(shape)
    ax.axhline(y=min_dim, color='red', linestyle='--', alpha=0.5, label=f'Stiefel ({min_dim})')

    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel('Stable Rank')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_stable_rank_over_training.png"))
plt.close()
print("Saved 01_stable_rank_over_training.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Effective rank over training
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Effective Rank of Weight Matrices Over Training\n(nuclear_norm / spectral_norm — measures spectrum spread)", fontsize=14, fontweight='bold')

for ax_idx, (subtype, title) in enumerate(zip(subtypes, subtype_titles)):
    ax = axes.flat[ax_idx]
    colors = cm.viridis(np.linspace(0, 1, 20))

    for layer in range(20):
        name = f"transformer.h.{layer}.{subtype}.weight"
        if name not in weight_data[steps[0]]:
            continue
        eff_ranks = [weight_data[s][name]['effective_rank'] for s in steps]
        ax.plot(steps, eff_ranks, color=colors[layer], alpha=0.7, linewidth=1.2, label=f"L{layer}" if layer % 5 == 0 else None)

    shape = weight_data[steps[0]][name]['shape']
    min_dim = min(shape)
    ax.axhline(y=min_dim, color='red', linestyle='--', alpha=0.5, label=f'Full rank ({min_dim})')

    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_effective_rank_over_training.png"))
plt.close()
print("Saved 02_effective_rank_over_training.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Singular value spectra evolution (heatmaps)
# ─────────────────────────────────────────────────────────────────────────────

# Pick representative layers: first, middle, last
repr_layers = [0, 9, 19]
repr_subtypes = ['attn.c_q', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
repr_titles = ['Attn Q', 'Attn Out', 'MLP Up', 'MLP Down']

fig, axes = plt.subplots(len(repr_layers), len(repr_subtypes), figsize=(18, 12))
fig.suptitle("Singular Value Spectrum Evolution Over Training (top-64 SVs)\nBrighter = larger singular value", fontsize=14, fontweight='bold')

for row, layer in enumerate(repr_layers):
    for col, (subtype, stitle) in enumerate(zip(repr_subtypes, repr_titles)):
        ax = axes[row, col]
        name = f"transformer.h.{layer}.{subtype}.weight"
        if name not in weight_data[steps[0]]:
            ax.set_visible(False)
            continue

        # Build heatmap: rows=steps, cols=sv index
        spectra_matrix = np.array([weight_data[s][name]['singular_values'] for s in steps])
        im = ax.imshow(spectra_matrix, aspect='auto', cmap='hot',
                       extent=[0, spectra_matrix.shape[1], steps[-1], steps[0]])
        ax.set_title(f"Layer {layer} — {stitle}")
        ax.set_xlabel('SV Index')
        ax.set_ylabel('Step')
        plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_sv_spectrum_heatmaps.png"))
plt.close()
print("Saved 03_sv_spectrum_heatmaps.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: SV spectrum snapshots (line plots at key training stages)
# ─────────────────────────────────────────────────────────────────────────────

snapshot_steps = [steps[0], steps[len(steps)//4], steps[len(steps)//2], steps[3*len(steps)//4], steps[-1]]
colors_snap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Singular Value Spectra at Key Training Stages\n(Layer 10 — middle of network)", fontsize=14, fontweight='bold')

for ax_idx, (subtype, title) in enumerate(zip(subtypes, subtype_titles)):
    ax = axes.flat[ax_idx]
    name = f"transformer.h.10.{subtype}.weight"
    if name not in weight_data[steps[0]]:
        ax.set_visible(False)
        continue

    for ss, color in zip(snapshot_steps, colors_snap):
        svs = weight_data[ss][name]['singular_values']
        ax.plot(range(len(svs)), svs, color=color, alpha=0.8, linewidth=1.5, label=f"step {ss}")

    ax.set_title(title)
    ax.set_xlabel('SV Index')
    ax.set_ylabel('Singular Value')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_sv_snapshots_layer10.png"))
plt.close()
print("Saved 04_sv_snapshots_layer10.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Condition number (σ_max / σ_64) over training
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Condition Number (σ₁/σ₆₄) of Weight Matrices Over Training\n(Lower = more uniform spectrum, higher = more concentrated)", fontsize=14, fontweight='bold')

for ax_idx, (subtype, title) in enumerate(zip(subtypes, subtype_titles)):
    ax = axes.flat[ax_idx]
    colors = cm.viridis(np.linspace(0, 1, 20))

    for layer in range(20):
        name = f"transformer.h.{layer}.{subtype}.weight"
        if name not in weight_data[steps[0]]:
            continue
        cond_nums = []
        for s in steps:
            svs = weight_data[s][name]['singular_values']
            cond_nums.append(svs[0] / max(svs[-1], 1e-10))
        ax.plot(steps, cond_nums, color=colors[layer], alpha=0.7, linewidth=1.2, label=f"L{layer}" if layer % 5 == 0 else None)

    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel('Condition Number (σ₁/σ₆₄)')
    ax.set_yscale('log')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_condition_number.png"))
plt.close()
print("Saved 05_condition_number.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: Spectral & Frobenius norms over training
# ─────────────────────────────────────────────────────────────────────────────

repr_layers_fig6 = [0, 10, 19]
repr_colors_fig6 = {'L0': '#D4A017', 'L10': '#2E8B7A', 'L19': '#A0306A'}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Spectral Norm (σ_max) and Frobenius Norm Over Training\n(solid=spectral, dashed=frobenius/√min_dim, shaded=mean±std across all layers)", fontsize=14, fontweight='bold')

for ax_idx, (subtype, title) in enumerate(zip(subtypes, subtype_titles)):
    ax = axes.flat[ax_idx]

    # Collect all layers for mean/std band
    all_spectral = []
    all_frob = []
    for layer in range(20):
        name = f"transformer.h.{layer}.{subtype}.weight"
        if name not in weight_data[steps[0]]:
            continue
        spectral = [weight_data[s][name]['spectral_norm'] for s in steps]
        shape = weight_data[steps[0]][name]['shape']
        min_dim = min(shape)
        frob_normalized = [weight_data[s][name]['frobenius_norm'] / np.sqrt(min_dim) for s in steps]
        all_spectral.append(spectral)
        all_frob.append(frob_normalized)

    # Plot mean/std band
    if all_spectral:
        arr_s = np.array(all_spectral)
        arr_f = np.array(all_frob)
        mean_s, std_s = arr_s.mean(axis=0), arr_s.std(axis=0)
        mean_f, std_f = arr_f.mean(axis=0), arr_f.std(axis=0)
        ax.fill_between(steps, mean_s - std_s, mean_s + std_s, color='#2196F3', alpha=0.1)
        ax.fill_between(steps, mean_f - std_f, mean_f + std_f, color='#FF9800', alpha=0.1)

    # Plot representative layers
    for layer in repr_layers_fig6:
        name = f"transformer.h.{layer}.{subtype}.weight"
        if name not in weight_data[steps[0]]:
            continue
        spectral = [weight_data[s][name]['spectral_norm'] for s in steps]
        shape = weight_data[steps[0]][name]['shape']
        min_dim = min(shape)
        frob_normalized = [weight_data[s][name]['frobenius_norm'] / np.sqrt(min_dim) for s in steps]
        c = repr_colors_fig6[f'L{layer}']
        ax.plot(steps, spectral, color=c, alpha=0.9, linewidth=2.0, label=f"L{layer} spectral")
        ax.plot(steps, frob_normalized, color=c, alpha=0.6, linewidth=1.5, linestyle='--', label=f"L{layer} frob/√d")

    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel('Norm')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_norms_over_training.png"))
plt.close()
print("Saved 06_norms_over_training.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7: Gradient spectrum evolution
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Gradient Singular Value Spectra at Key Training Stages (Layer 10)\nShows what directions the loss landscape cares about", fontsize=14, fontweight='bold')

for ax_idx, (subtype, title) in enumerate(zip(subtypes, subtype_titles)):
    ax = axes.flat[ax_idx]
    name = f"transformer.h.10.{subtype}.weight"
    if name not in gradient_data[steps[0]]:
        ax.set_visible(False)
        continue

    for ss, color in zip(snapshot_steps, colors_snap):
        if ss not in gradient_data or name not in gradient_data[ss]:
            continue
        svs = gradient_data[ss][name]['singular_values']
        # Normalize by max for comparison
        svs_arr = np.array(svs)
        svs_norm = svs_arr / svs_arr[0] if svs_arr[0] > 0 else svs_arr
        ax.plot(range(len(svs_norm)), svs_norm, color=color, alpha=0.8, linewidth=1.5, label=f"step {ss}")

    ax.set_title(title)
    ax.set_xlabel('SV Index')
    ax.set_ylabel('Normalized SV (σ/σ_max)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "07_gradient_spectra.png"))
plt.close()
print("Saved 07_gradient_spectra.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8: Update spectrum evolution
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Update (ΔW) Singular Value Spectra at Key Training Stages (Layer 10)\nShows the rank structure of Muon's actual updates", fontsize=14, fontweight='bold')

update_steps_avail = sorted(update_data.keys())
update_snapshot_steps = [update_steps_avail[i] for i in [0, len(update_steps_avail)//4, len(update_steps_avail)//2, 3*len(update_steps_avail)//4, -1]]

for ax_idx, (subtype, title) in enumerate(zip(subtypes, subtype_titles)):
    ax = axes.flat[ax_idx]
    name = f"transformer.h.10.{subtype}.weight"

    for ss, color in zip(update_snapshot_steps, colors_snap):
        if ss not in update_data or name not in update_data[ss]:
            continue
        svs = update_data[ss][name]['singular_values']
        svs_arr = np.array(svs)
        svs_norm = svs_arr / svs_arr[0] if svs_arr[0] > 0 else svs_arr
        ax.plot(range(len(svs_norm)), svs_norm, color=color, alpha=0.8, linewidth=1.5, label=f"step {ss}")

    ax.set_title(title)
    ax.set_xlabel('SV Index')
    ax.set_ylabel('Normalized SV (σ/σ_max)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_update_spectra.png"))
plt.close()
print("Saved 08_update_spectra.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9: Update effective rank & stable rank over training
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Effective Rank of Updates (ΔW) Over Training\nHigh = Muon spreads updates; Low = rank-deficient updates", fontsize=14, fontweight='bold')

for ax_idx, (subtype, title) in enumerate(zip(subtypes, subtype_titles)):
    ax = axes.flat[ax_idx]
    colors = cm.viridis(np.linspace(0, 1, 20))

    for layer in range(20):
        name = f"transformer.h.{layer}.{subtype}.weight"
        eff_ranks = []
        update_steps_plot = []
        for s in update_steps_avail:
            if name in update_data[s]:
                eff_ranks.append(update_data[s][name]['effective_rank'])
                update_steps_plot.append(s)
        if eff_ranks:
            ax.plot(update_steps_plot, eff_ranks, color=colors[layer], alpha=0.7, linewidth=1.2, label=f"L{layer}" if layer % 5 == 0 else None)

    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "09_update_effective_rank.png"))
plt.close()
print("Saved 09_update_effective_rank.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10: Gradient effective rank over training
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Effective Rank of Gradients Over Training\nShows how many directions the loss cares about", fontsize=14, fontweight='bold')

grad_steps_avail = sorted(gradient_data.keys())

for ax_idx, (subtype, title) in enumerate(zip(subtypes, subtype_titles)):
    ax = axes.flat[ax_idx]
    colors = cm.viridis(np.linspace(0, 1, 20))

    for layer in range(20):
        name = f"transformer.h.{layer}.{subtype}.weight"
        eff_ranks = []
        grad_steps_plot = []
        for s in grad_steps_avail:
            if name in gradient_data[s]:
                eff_ranks.append(gradient_data[s][name]['effective_rank'])
                grad_steps_plot.append(s)
        if eff_ranks:
            ax.plot(grad_steps_plot, eff_ranks, color=colors[layer], alpha=0.7, linewidth=1.2, label=f"L{layer}" if layer % 5 == 0 else None)

    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "10_gradient_effective_rank.png"))
plt.close()
print("Saved 10_gradient_effective_rank.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 11: Layer-wise summary at end of training
# ─────────────────────────────────────────────────────────────────────────────

last_step = steps[-1]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Layer-wise Summary at End of Training (step {last_step})", fontsize=14, fontweight='bold')

metrics = [
    ('stable_rank', 'Stable Rank (||W||²_F / σ²_max)'),
    ('effective_rank', 'Effective Rank (||W||_* / σ_max)'),
    ('spectral_norm', 'Spectral Norm (σ_max)'),
    ('frobenius_norm', 'Frobenius Norm'),
]

for ax_idx, (metric, ylabel) in enumerate(metrics):
    ax = axes.flat[ax_idx]

    for subtype, marker, color in [
        ('attn.c_q', 'o', '#1f77b4'), ('attn.c_k', 's', '#ff7f0e'),
        ('attn.c_v', '^', '#2ca02c'), ('attn.c_proj', 'D', '#d62728'),
        ('mlp.c_fc', 'v', '#9467bd'), ('mlp.c_proj', 'P', '#8c564b'),
    ]:
        layers_plot = []
        vals = []
        for layer in range(20):
            name = f"transformer.h.{layer}.{subtype}.weight"
            if name in weight_data[last_step]:
                layers_plot.append(layer)
                vals.append(weight_data[last_step][name][metric])
        if vals:
            ax.plot(layers_plot, vals, marker=marker, color=color, alpha=0.8, linewidth=1.5, markersize=5, label=subtype)

    ax.set_xlabel('Layer')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "11_layerwise_summary.png"))
plt.close()
print("Saved 11_layerwise_summary.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 12: Concentration ratio (σ₁/Σσ) — how much energy in top SV
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Top-1 Concentration Ratio (σ₁ / nuclear_norm) Over Training\nHigher = spectrum concentrating into fewer directions", fontsize=14, fontweight='bold')

for ax_idx, (subtype, title) in enumerate(zip(subtypes, subtype_titles)):
    ax = axes.flat[ax_idx]
    colors = cm.viridis(np.linspace(0, 1, 20))

    for layer in range(20):
        name = f"transformer.h.{layer}.{subtype}.weight"
        if name not in weight_data[steps[0]]:
            continue
        ratios = [weight_data[s][name]['spectral_norm'] / max(weight_data[s][name]['nuclear_norm'], 1e-10) for s in steps]
        ax.plot(steps, ratios, color=colors[layer], alpha=0.7, linewidth=1.2, label=f"L{layer}" if layer % 5 == 0 else None)

    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel('σ₁ / Σσᵢ')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "12_concentration_ratio.png"))
plt.close()
print("Saved 12_concentration_ratio.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 13: Gradient vs Weight effective rank comparison
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Weight vs Gradient vs Update Effective Rank (Layer 10)\nComparing rank structure across all three", fontsize=14, fontweight='bold')

for ax_idx, (subtype, title) in enumerate(zip(subtypes, subtype_titles)):
    ax = axes.flat[ax_idx]
    name = f"transformer.h.10.{subtype}.weight"

    # Weights
    w_ranks = [weight_data[s][name]['effective_rank'] for s in steps if name in weight_data[s]]
    w_steps = [s for s in steps if name in weight_data[s]]
    ax.plot(w_steps, w_ranks, 'b-', linewidth=2, label='Weight', alpha=0.8)

    # Gradients
    g_ranks = [gradient_data[s][name]['effective_rank'] for s in grad_steps_avail if name in gradient_data[s]]
    g_steps = [s for s in grad_steps_avail if name in gradient_data[s]]
    if g_ranks:
        ax.plot(g_steps, g_ranks, 'r-', linewidth=2, label='Gradient', alpha=0.8)

    # Updates
    u_ranks = [update_data[s][name]['effective_rank'] for s in update_steps_avail if name in update_data[s]]
    u_steps = [s for s in update_steps_avail if name in update_data[s]]
    if u_ranks:
        ax.plot(u_steps, u_ranks, 'g-', linewidth=2, label='Update (ΔW)', alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel('Effective Rank')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "13_weight_vs_gradient_vs_update_rank.png"))
plt.close()
print("Saved 13_weight_vs_gradient_vs_update_rank.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 14: Frobenius norm of updates (learning signal magnitude)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Frobenius Norm of Updates (||ΔW||_F) Over Training\nShows learning signal magnitude per layer", fontsize=14, fontweight='bold')

for ax_idx, (subtype, title) in enumerate(zip(subtypes, subtype_titles)):
    ax = axes.flat[ax_idx]
    colors = cm.viridis(np.linspace(0, 1, 20))

    for layer in range(20):
        name = f"transformer.h.{layer}.{subtype}.weight"
        frob_norms = []
        u_steps_plot = []
        for s in update_steps_avail:
            if name in update_data[s]:
                frob_norms.append(update_data[s][name]['frobenius_norm'])
                u_steps_plot.append(s)
        if frob_norms:
            ax.plot(u_steps_plot, frob_norms, color=colors[layer], alpha=0.7, linewidth=1.2, label=f"L{layer}" if layer % 5 == 0 else None)

    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel('||ΔW||_F')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "14_update_frobenius_norm.png"))
plt.close()
print("Saved 14_update_frobenius_norm.png")

print(f"\nAll plots saved to {OUT_DIR}/")
