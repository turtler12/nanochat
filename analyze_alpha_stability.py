"""
Experiment 6: α stability over training.

Fits power-law exponent α via linear regression of log(σ_i) on log(i)
for each (step, layer) with ≥2 singular values, then plots α vs step
per layer colored by layer type, and reports CV statistics.
"""

import json
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

LOG_PATH = "verifying_results/small_model/heuristics/gradient_spectrum_log.jsonl"
OUT_PLOT  = "verifying_results/small_model/heuristics/alpha_stability.png"
OUT_STATS = "verifying_results/small_model/heuristics/alpha_stats.txt"

# ── layer type extraction ──────────────────────────────────────────────────────
_KNOWN_TYPES = ["ve_gate", "c_proj", "c_fc", "c_q", "c_k", "c_v"]

def layer_type(name: str) -> str:
    for t in _KNOWN_TYPES:
        if f".{t}." in name or name.endswith(f".{t}.weight"):
            return t
    # fallback: last component before .weight
    return name.replace(".weight", "").split(".")[-1]


# ── load & fit ─────────────────────────────────────────────────────────────────
def fit_alpha(svs: list[float]) -> float | None:
    """OLS of log(σ_i) ~ α·log(i), i=1..len(svs), returns slope magnitude."""
    svs = [s for s in svs if s > 0]
    if len(svs) < 2:
        return None
    n = len(svs)
    log_i = np.log(np.arange(1, n + 1, dtype=float))
    log_s = np.log(np.array(svs, dtype=float))
    # OLS: α = cov(log_i, log_s) / var(log_i)
    log_i_c = log_i - log_i.mean()
    slope = (log_i_c @ log_s) / (log_i_c @ log_i_c)
    return float(-slope)          # power-law decay → slope is negative; α > 0


alpha_data: dict[str, dict[int, float]] = defaultdict(dict)  # layer → step → α

with open(LOG_PATH) as f:
    for line in f:
        rec = json.loads(line)
        if "_config" in rec:
            continue
        svs = rec.get("singular_values", [])
        if len(svs) < 2:
            continue
        a = fit_alpha(svs)
        if a is not None:
            alpha_data[rec["layer_name"]][rec["step"]] = a


# ── statistics ─────────────────────────────────────────────────────────────────
layer_names = sorted(alpha_data.keys())
type_names  = sorted({layer_type(n) for n in layer_names})

# Build per-layer arrays (sorted by step)
layer_steps:  dict[str, np.ndarray] = {}
layer_alphas: dict[str, np.ndarray] = {}
for name in layer_names:
    sd = alpha_data[name]
    steps_sorted = sorted(sd)
    layer_steps[name]  = np.array(steps_sorted)
    layer_alphas[name] = np.array([sd[s] for s in steps_sorted])

# Per-layer summary
stats: list[dict] = []
for name in layer_names:
    a = layer_alphas[name]
    mean_ = float(a.mean())
    std_  = float(a.std())
    cv    = float(std_ / mean_ * 100) if mean_ != 0 else float("nan")
    stats.append(dict(layer=name, ltype=layer_type(name),
                      mean=mean_, std=std_, cv=cv,
                      n_steps=len(a)))

stats.sort(key=lambda x: x["cv"])

# Print / save stats
lines_out = []
lines_out.append(f"{'Layer':<55} {'Type':<10} {'Mean α':>8} {'Std α':>8} {'CV %':>7} {'Steps':>6}")
lines_out.append("-" * 100)
for s in stats:
    lines_out.append(
        f"{s['layer']:<55} {s['ltype']:<10} {s['mean']:>8.3f} {s['std']:>8.4f} {s['cv']:>7.2f} {s['n_steps']:>6}"
    )

lines_out.append("")
lines_out.append("=== Per layer-type summary ===")
lines_out.append(f"{'Type':<12} {'Layers':>6} {'Mean α (mean±std)':>20} {'CV mean':>10} {'CV max':>10}")
lines_out.append("-" * 65)
for t in type_names:
    sub = [s for s in stats if s["ltype"] == t]
    means  = np.array([s["mean"] for s in sub])
    cvs    = np.array([s["cv"]   for s in sub])
    lines_out.append(
        f"{t:<12} {len(sub):>6} {means.mean():>10.3f}±{means.std():>7.3f}   {cvs.mean():>10.2f} {cvs.max():>10.2f}"
    )

lines_out.append("")
lines_out.append("=== Overall ===")
all_cvs = np.array([s["cv"] for s in stats if not np.isnan(s["cv"])])
lines_out.append(f"  Mean CV across all layers: {all_cvs.mean():.2f}%")
lines_out.append(f"  Max  CV across all layers: {all_cvs.max():.2f}%")
lines_out.append(f"  Layers with CV < 10%:      {(all_cvs < 10).sum()} / {len(all_cvs)}")
lines_out.append(f"  Layers with CV < 5%:       {(all_cvs < 5).sum()} / {len(all_cvs)}")

report = "\n".join(lines_out)
print(report)
with open(OUT_STATS, "w") as f:
    f.write(report + "\n")


# ── plot ───────────────────────────────────────────────────────────────────────
type_colors = {t: c for t, c in zip(
    type_names,
    ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628"],
)}

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
axes_flat = axes.flatten()

for ax_idx, ltype in enumerate(type_names):
    ax = axes_flat[ax_idx]
    color = type_colors[ltype]
    layers_of_type = [n for n in layer_names if layer_type(n) == ltype]
    for name in layers_of_type:
        ax.plot(layer_steps[name], layer_alphas[name],
                color=color, alpha=0.55, linewidth=0.9)
    ax.set_title(ltype, fontsize=11)
    ax.set_ylabel("α (power-law exponent)")
    ax.set_xlabel("step")
    ax.grid(True, linestyle="--", alpha=0.4)

    # annotate mean CV for this type
    sub = [s for s in stats if s["ltype"] == ltype]
    cvs = np.array([s["cv"] for s in sub])
    ax.text(0.97, 0.97, f"mean CV={cvs.mean():.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

fig.suptitle("Experiment 6 — Power-law exponent α vs training step (per layer type)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=150)
print(f"\nPlot saved → {OUT_PLOT}")
print(f"Stats saved → {OUT_STATS}")
