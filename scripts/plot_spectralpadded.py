"""
Plot spectral-padded Muon experiment: SV diagnostics.

Layout:
  Top row:    Val BPB comparison  |  σ_max over training (absolute scale)
  Middle row: Full 5-quantile fan chart for each stage (large matrix)
  Bottom row: Same for small matrix — shows the contrast

Usage:
  python scripts/plot_spectralpadded.py
"""

import json
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Paths ────────────────────────────────────────────────────────────────────
SPECTRAL = "spectralpadded_results/spectralpad_gamma0.1_pi1_pb4"
BASELINE = "padded_muon_results/baseline"
OUTPUT = "spectralpadded_results/comparison.png"

# ── Style ────────────────────────────────────────────────────────────────────
STAGE_CFG = {
    "pre":     {"color": "#2563eb", "label": "Pre-ortho\n(after momentum)"},
    "polar":   {"color": "#16a34a", "label": "After Polar\nExpress"},
    "blended": {"color": "#ea580c", "label": "After spectral\nblend"},
}

# ── Data loading ─────────────────────────────────────────────────────────────
def load_val(path):
    with open(path) as f:
        return json.load(f)

def load_sv(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries

def group_sv(entries):
    groups = defaultdict(lambda: {
        "steps": [], "p05": [], "p25": [], "p50": [], "p75": [], "p95": [],
        "sigma_max": [], "sigma_hat": [],
    })
    for e in entries:
        key = (e["layer_id"], e["stage"])
        groups[key]["steps"].append(e["step"])
        for f in ["p05", "p25", "p50", "p75", "p95", "sigma_max"]:
            groups[key][f].append(e[f])
        groups[key]["sigma_hat"].append(e.get("sigma_hat", None))
    return dict(groups)

def to_abs(d):
    """Convert normalized percentiles to absolute singular values."""
    smax = np.array(d["sigma_max"])
    out = {}
    for f in ["p05", "p25", "p50", "p75", "p95"]:
        out[f] = np.array(d[f]) * smax
    out["sigma_max"] = smax
    out["steps"] = np.array(d["steps"])
    return out

# ── Load data ────────────────────────────────────────────────────────────────
spectral_val = load_val(f"{SPECTRAL}/val_loss.json")
baseline_val = load_val(f"{BASELINE}/val_loss.json")
spectral_sv = group_sv(load_sv(f"{SPECTRAL}/sv_log.jsonl"))

large_layer = small_layer = None
for (lid, _) in spectral_sv:
    if "largest" in lid:
        large_layer = lid
    if "smallest" in lid:
        small_layer = lid

gamma = spectral_val["pad_ratio"]

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30,
                      height_ratios=[1, 1.2, 1.2])
fig.suptitle(
    f"Spectral-Padded Muon  (γ = {gamma},  power_iters = {spectral_val['power_iters']},"
    f"  power_batch = {spectral_val['power_batch']})",
    fontsize=15, fontweight="bold", y=0.995,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Row 0, col 0: Val BPB
# ═══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 0])
for data, name, color, ls in [
    (baseline_val, "Baseline", "#2563eb", "-"),
    (spectral_val, "Spectral-pad γ=0.1", "#ea580c", "--"),
]:
    steps = [e["step"] for e in data["val_loss_log"]]
    bpbs = [e["val_bpb"] for e in data["val_loss_log"]]
    ax.plot(steps, bpbs, color=color, linestyle=ls,
            label=f"{name}  →  {bpbs[-1]:.4f}", linewidth=2.2,
            marker="o", markersize=2.5)
ax.set_xlabel("Step")
ax.set_ylabel("Val BPB ↓")
ax.set_title("Validation Loss", fontsize=12, fontweight="bold")
ax.legend(framealpha=0.9, fontsize=9, loc="upper right")
ax.grid(True, alpha=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# Row 0, col 1-2: σ_max (absolute) for both layers, log scale
# ═══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 1:])
for lid, marker, layer_label in [
    (large_layer, "o", "Large [5120×1280]"),
    (small_layer, "s", "Small [10×32]"),
]:
    if lid is None:
        continue
    for stage, cfg in STAGE_CFG.items():
        key = (lid, stage)
        if key not in spectral_sv:
            continue
        d = spectral_sv[key]
        steps = np.array(d["steps"][1:])
        smax = np.array(d["sigma_max"][1:])
        ax.plot(steps, smax, color=cfg["color"], marker=marker, markersize=4,
                linewidth=1.8, alpha=0.85,
                label=f"{layer_label} — {cfg['label'].replace(chr(10), ' ')}")
ax.set_yscale("log")
ax.set_xlabel("Step")
ax.set_ylabel("σ_max  (absolute)")
ax.set_title("Absolute σ_max Over Training", fontsize=12, fontweight="bold")
ax.legend(framealpha=0.9, fontsize=7.5, ncol=2, loc="upper right")
ax.grid(True, alpha=0.2, which="both")

# ═══════════════════════════════════════════════════════════════════════════════
# Rows 1-2: Fan charts  (one row per layer)
# Each row has 3 panels: pre | polar | blended
# Fan chart shows: p05-p95 outer band, p25-p75 inner band, p50 line
# All in ABSOLUTE singular value units
# ═══════════════════════════════════════════════════════════════════════════════
for row_idx, (lid, layer_label) in enumerate(
    [(large_layer, "Large Matrix [5120 × 1280]"),
     (small_layer, "Small Matrix [10 × 32]")],
    start=1
):
    if lid is None:
        continue
    for col_idx, stage in enumerate(["pre", "polar", "blended"]):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        cfg = STAGE_CFG[stage]
        key = (lid, stage)
        if key not in spectral_sv:
            continue

        d = spectral_sv[key]
        # Skip step 0
        mask = np.array(d["steps"]) > 0
        steps = np.array(d["steps"])[mask]
        smax = np.array(d["sigma_max"])[mask]
        color = cfg["color"]

        # Absolute values
        p05 = np.array(d["p05"])[mask] * smax
        p25 = np.array(d["p25"])[mask] * smax
        p50 = np.array(d["p50"])[mask] * smax
        p75 = np.array(d["p75"])[mask] * smax
        p95 = np.array(d["p95"])[mask] * smax

        # Outer band: p05-p95
        ax.fill_between(steps, p05, p95, color=color, alpha=0.12,
                        label="p05–p95")
        # Inner band: p25-p75
        ax.fill_between(steps, p25, p75, color=color, alpha=0.25,
                        label="p25–p75")
        # Median
        ax.plot(steps, p50, color=color, linewidth=2.2, label="p50 (median)")
        # σ_max line
        ax.plot(steps, smax, color="black", linewidth=1.2, linestyle="--",
                alpha=0.6, label="σ_max")

        # For blended stage, show the spectral floor γ·σ_hat
        if stage == "blended":
            sigma_hat_raw = np.array(d["sigma_hat"])[mask]
            valid = np.array([x is not None for x in sigma_hat_raw])
            if valid.any():
                sh = np.array([float(x) if x is not None else 0 for x in sigma_hat_raw])
                floor = gamma * sh
                ax.plot(steps[valid], floor[valid], color="#a855f7",
                        linewidth=1.5, linestyle=":", alpha=0.8,
                        label=f"γ·σ̂ = {gamma}·σ_max(g)")

        ax.set_xlabel("Step")
        if col_idx == 0:
            ax.set_ylabel("Singular Value (absolute)")
        ax.set_title(f"{cfg['label'].replace(chr(10), ' ')}",
                     fontsize=10, fontweight="bold", color=color)

        # Use log scale for the large matrix (huge dynamic range)
        if "largest" in lid:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.2, which="both")

        # Only put legend on first panel of each row
        if col_idx == 0:
            ax.legend(framealpha=0.9, fontsize=7, loc="upper right")

    # Row label on left side
    fig.text(0.005, 0.5 - (row_idx - 1) * 0.32,
             layer_label, fontsize=11, fontweight="bold",
             rotation=90, va="center", ha="left", color="#555")

plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT}")
