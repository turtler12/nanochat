"""Plot how singular values of weight matrices grow over training (baseline Muon run)."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

svd_dir = "cache/base_checkpoints/baseline_muon/svd_logs"

# Load all weight SVD files
steps = []
all_spectra = {}  # layer_name -> {step: singular_values}

for fname in sorted(os.listdir(svd_dir)):
    if not fname.startswith("weight_svd_step"):
        continue
    with open(os.path.join(svd_dir, fname)) as f:
        d = json.load(f)
    step = d["step"]
    steps.append(step)
    for layer_name, spec in d["spectra"].items():
        if layer_name not in all_spectra:
            all_spectra[layer_name] = {}
        all_spectra[layer_name][step] = spec

steps = sorted(set(steps))
layer_names = sorted(all_spectra.keys())

# Classify layers by type
layer_types = defaultdict(list)
for name in layer_names:
    if "c_q" in name:
        layer_types["Q projection"].append(name)
    elif "c_k" in name:
        layer_types["K projection"].append(name)
    elif "c_v" in name:
        layer_types["V projection"].append(name)
    elif "c_proj" in name:
        layer_types["Attention output"].append(name)
    elif "c_fc" in name:
        layer_types["MLP up"].append(name)
    elif "c_fc2" in name:
        layer_types["MLP gate"].append(name)
    elif "c_proj2" in name or "mlp.c_proj" in name:
        layer_types["MLP down"].append(name)
    else:
        layer_types["Other"].append(name)

# Re-classify: just use simpler grouping
layer_types = defaultdict(list)
for name in layer_names:
    if "attn" in name:
        if "c_q" in name:
            layer_types["attn.Q"].append(name)
        elif "c_k" in name:
            layer_types["attn.K"].append(name)
        elif "c_v" in name:
            layer_types["attn.V"].append(name)
        elif "c_proj" in name:
            layer_types["attn.out"].append(name)
    elif "mlp" in name:
        if "c_fc2" in name:
            layer_types["mlp.gate"].append(name)
        elif "c_fc" in name:
            layer_types["mlp.up"].append(name)
        elif "c_proj" in name:
            layer_types["mlp.down"].append(name)

# ============================================================
# Figure 1: Spectral norm & effective rank over training, averaged by layer type
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Singular Value Dynamics During Baseline Muon Training (depth=20)", fontsize=14, fontweight="bold")

colors = plt.cm.tab10(np.linspace(0, 1, len(layer_types)))

# Top-left: spectral norm (σ₁)
ax = axes[0, 0]
for (ltype, names), color in zip(layer_types.items(), colors):
    vals = []
    for step in steps:
        v = np.mean([all_spectra[n][step]["spectral_norm"] for n in names if step in all_spectra[n]])
        vals.append(v)
    ax.plot(steps, vals, label=ltype, color=color, linewidth=1.5)
ax.set_xlabel("Training step")
ax.set_ylabel("Spectral norm (σ₁)")
ax.set_title("Spectral norm growth")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Top-right: Frobenius norm
ax = axes[0, 1]
for (ltype, names), color in zip(layer_types.items(), colors):
    vals = []
    for step in steps:
        v = np.mean([all_spectra[n][step]["frobenius_norm"] for n in names if step in all_spectra[n]])
        vals.append(v)
    ax.plot(steps, vals, label=ltype, color=color, linewidth=1.5)
ax.set_xlabel("Training step")
ax.set_ylabel("Frobenius norm")
ax.set_title("Frobenius norm growth")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Bottom-left: effective rank
ax = axes[1, 0]
for (ltype, names), color in zip(layer_types.items(), colors):
    vals = []
    for step in steps:
        v = np.mean([all_spectra[n][step]["effective_rank"] for n in names if step in all_spectra[n]])
        vals.append(v)
    ax.plot(steps, vals, label=ltype, color=color, linewidth=1.5)
ax.set_xlabel("Training step")
ax.set_ylabel("Effective rank (nuclear/spectral)")
ax.set_title("Effective rank over training")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Bottom-right: stable rank
ax = axes[1, 1]
for (ltype, names), color in zip(layer_types.items(), colors):
    vals = []
    for step in steps:
        v = np.mean([all_spectra[n][step]["stable_rank"] for n in names if step in all_spectra[n]])
        vals.append(v)
    ax.plot(steps, vals, label=ltype, color=color, linewidth=1.5)
ax.set_xlabel("Training step")
ax.set_ylabel("Stable rank (||W||²_F / σ₁²)")
ax.set_title("Stable rank over training")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/baseline_muon/svd_growth_summary.png", dpi=150, bbox_inches="tight")
print("Saved plots/baseline_muon/svd_growth_summary.png")

# ============================================================
# Figure 2: Full singular value spectra at different training stages
# ============================================================
# Pick a representative layer from each type and show spectrum evolution
sample_steps = [0, 500, 1000, 2000, 3000, 4250]
sample_steps = [s for s in sample_steps if s in steps]

# Pick one layer per type (middle layer)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Singular Value Spectra at Different Training Steps", fontsize=14, fontweight="bold")

all_types = list(layer_types.keys())
for idx, ltype in enumerate(all_types[:8]):  # up to 8 types
    ax = axes.flat[idx]
    names = layer_types[ltype]
    # Pick middle layer
    rep_name = names[len(names) // 2]
    short = rep_name.replace("transformer.h.", "L").replace(".weight", "")

    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(sample_steps)))
    for step, color in zip(sample_steps, cmap):
        if step in all_spectra[rep_name]:
            svs = all_spectra[rep_name][step]["singular_values"]
            ax.plot(range(len(svs)), svs, color=color, label=f"step {step}", linewidth=1.2)

    ax.set_title(f"{ltype}\n({short})", fontsize=9)
    ax.set_xlabel("SV index")
    ax.set_ylabel("σ")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(len(all_types), 8):
    axes.flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig("plots/baseline_muon/svd_spectra_evolution.png", dpi=150, bbox_inches="tight")
print("Saved plots/baseline_muon/svd_spectra_evolution.png")

# ============================================================
# Figure 3: Heatmap of top-k SVs per layer at final step
# ============================================================
fig, ax = plt.subplots(figsize=(16, 10))
final_step = max(steps)

# Group by layer type, show all layers
matrix = []
labels = []
for ltype in all_types:
    for name in layer_types[ltype]:
        if final_step in all_spectra[name]:
            svs = all_spectra[name][final_step]["singular_values"]
            matrix.append(svs)
            short = name.replace("transformer.h.", "L").replace(".weight", "")
            labels.append(short)

matrix = np.array(matrix)
# Normalize each row for visibility
matrix_norm = matrix / matrix.max(axis=1, keepdims=True)

im = ax.imshow(matrix_norm, aspect="auto", cmap="viridis", interpolation="nearest")
ax.set_xlabel("Singular value index")
ax.set_ylabel("Layer")
ax.set_title(f"Normalized SV Spectra Across All Layers (step {final_step})")
ax.set_yticks(range(0, len(labels), max(1, len(labels)//20)))
ax.set_yticklabels([labels[i] for i in range(0, len(labels), max(1, len(labels)//20))], fontsize=6)
plt.colorbar(im, ax=ax, label="σ / σ_max")
plt.tight_layout()
plt.savefig("plots/baseline_muon/svd_heatmap_final.png", dpi=150, bbox_inches="tight")
print("Saved plots/baseline_muon/svd_heatmap_final.png")

# ============================================================
# Figure 4: σ₁/σ_k condition ratio over training (how much spectrum spreads)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for (ltype, names), color in zip(layer_types.items(), colors):
    vals = []
    for step in steps:
        ratios = []
        for n in names:
            if step in all_spectra[n]:
                svs = all_spectra[n][step]["singular_values"]
                if len(svs) > 1 and svs[-1] > 1e-10:
                    ratios.append(svs[0] / svs[-1])
        if ratios:
            vals.append(np.mean(ratios))
        else:
            vals.append(np.nan)
    ax.plot(steps, vals, label=ltype, color=color, linewidth=1.5)

ax.set_xlabel("Training step")
ax.set_ylabel("σ₁ / σ_k (condition ratio, top-k)")
ax.set_title("Condition Ratio (σ₁/σ_k) Over Training")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("plots/baseline_muon/svd_condition_ratio.png", dpi=150, bbox_inches="tight")
print("Saved plots/baseline_muon/svd_condition_ratio.png")

plt.close("all")
print("Done!")
