"""
Live comparison: running baseline (job 380909) vs completed spectral-padded run.
Parses val_bpb from the SLURM log for the in-progress run.

Usage:
  python scripts/testing_plot.py
"""

import json
import re
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
SPECTRAL_DIR = "spectralpadded_results/spectralpad_gamma0.1_pi1_pb4"
BASELINE_SV  = "spectralpadded_results/baseline/sv_log.jsonl"
BASELINE_LOG = "slurm_logs/spectralpad_muon_380909.log"
OUTPUT = "spectralpadded_results/comparison.png"

# ── Stage config ─────────────────────────────────────────────────────────────
STAGE_CFG = {
    "pre":     {"color": "#2563eb", "label": "Pre-ortho (after momentum)"},
    "polar":   {"color": "#16a34a", "label": "After Polar Express"},
    "blended": {"color": "#ea580c", "label": "After spectral blend"},
}

# ── Data loading ─────────────────────────────────────────────────────────────
def load_val_json(path):
    with open(path) as f:
        return json.load(f)

def parse_val_from_log(path):
    """Extract val_bpb entries from a SLURM log file."""
    pattern = re.compile(r"Step (\d+) \| val_bpb: ([\d.]+)")
    entries = []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                entries.append({"step": int(m.group(1)), "val_bpb": float(m.group(2))})
    return entries

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

# ── Load data ────────────────────────────────────────────────────────────────
# Completed spectral-padded run
spectral_val = load_val_json(f"{SPECTRAL_DIR}/val_loss.json")
spectral_sv = group_sv(load_sv(f"{SPECTRAL_DIR}/sv_log.jsonl"))

# Running baseline — parse from log + SV jsonl
baseline_val_log = parse_val_from_log(BASELINE_LOG)
baseline_sv = group_sv(load_sv(BASELINE_SV))

# Find layer ids
large_layer = small_layer = None
for (lid, _) in spectral_sv:
    if "largest" in lid:  large_layer = lid
    if "smallest" in lid: small_layer = lid

gamma = spectral_val["pad_ratio"]

# ── Figure: GridSpec so row 0 can have 2 cols (1:2 split) while rows 1-2 have 3
fig = plt.figure(figsize=(16, 14))
gs_top = fig.add_gridspec(1, 2, left=0.06, right=0.98, top=0.92, bottom=0.68,
                          wspace=0.25, width_ratios=[1, 2])
gs_bot = fig.add_gridspec(2, 3, left=0.06, right=0.98, top=0.62, bottom=0.04,
                          hspace=0.40, wspace=0.30, height_ratios=[1, 1])
fig.suptitle(
    f"Baseline (γ=0, running) vs Spectral-Padded (γ={gamma}, completed)",
    fontsize=15, fontweight="bold",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Row 0, col 0: Val BPB
# ═══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs_top[0, 0])

# Spectral-padded (completed)
steps_sp = [e["step"] for e in spectral_val["val_loss_log"]]
bpbs_sp = [e["val_bpb"] for e in spectral_val["val_loss_log"]]
ax.plot(steps_sp, bpbs_sp, color="#ea580c", linestyle="--", linewidth=2.2,
        marker="o", markersize=2.5,
        label=f"Spectral-pad γ={gamma}  →  {bpbs_sp[-1]:.4f}")

# Baseline (in progress)
steps_bl = [e["step"] for e in baseline_val_log]
bpbs_bl = [e["val_bpb"] for e in baseline_val_log]
ax.plot(steps_bl, bpbs_bl, color="#2563eb", linestyle="-", linewidth=2.2,
        marker="o", markersize=2.5,
        label=f"Baseline (running)  →  {bpbs_bl[-1]:.4f}")

ax.set_xlabel("Step")
ax.set_ylabel("Val BPB ↓")
ax.set_title("Validation Loss", fontsize=12, fontweight="bold")
ax.legend(framealpha=0.9, fontsize=9, loc="upper right")
ax.grid(True, alpha=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# Row 0, col 1: σ_max comparison (spans 2/3 of width)
# ═══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs_top[0, 1])
run_styles = [
    ("Spectral γ=0.1", spectral_sv, "--"),
    ("Baseline", baseline_sv, "-"),
]
for run_label, sv_data, ls in run_styles:
    for lid, marker, layer_lbl in [
        (large_layer, "o", "5120×1280"),
        (small_layer, "s", "10×32"),
    ]:
        if lid is None: continue
        for stage, cfg in STAGE_CFG.items():
            key = (lid, stage)
            if key not in sv_data: continue
            d = sv_data[key]
            steps = np.array(d["steps"][1:])
            smax = np.array(d["sigma_max"][1:])
            ax.plot(steps, smax, color=cfg["color"], marker=marker,
                    markersize=3.5, linewidth=1.5, alpha=0.8, linestyle=ls,
                    label=f"{run_label} {layer_lbl} {cfg['label']}")
ax.set_yscale("log")
ax.set_xlabel("Step")
ax.set_ylabel("σ_max (absolute)")
ax.set_title("σ_max Trajectories — Both Runs", fontsize=12, fontweight="bold")
ax.legend(framealpha=0.9, fontsize=7, ncol=3, loc="upper right")
ax.grid(True, alpha=0.2, which="both")

# ═══════════════════════════════════════════════════════════════════════════════
# Rows 1-2: Fan charts side by side — Baseline vs Spectral for each stage
# Each row = one layer, each col = one stage
# Two overlaid fan charts per panel (blue=baseline, orange=spectral)
# ═══════════════════════════════════════════════════════════════════════════════
for row_idx, (lid, layer_label) in enumerate(
    [(large_layer, "Large Matrix [5120 × 1280]"),
     (small_layer, "Small Matrix [10 × 32]")],
):
    if lid is None: continue

    for col_idx, stage in enumerate(["pre", "polar", "blended"]):
        ax = fig.add_subplot(gs_bot[row_idx, col_idx])
        cfg = STAGE_CFG[stage]

        for run_label, sv_data, color, alpha_mult in [
            ("Baseline", baseline_sv, "#2563eb", 1.0),
            (f"Spectral γ={gamma}", spectral_sv, "#ea580c", 0.7),
        ]:
            key = (lid, stage)
            if key not in sv_data: continue
            d = sv_data[key]
            mask = np.array(d["steps"]) > 0
            steps = np.array(d["steps"])[mask]
            smax = np.array(d["sigma_max"])[mask]

            p05 = np.array(d["p05"])[mask] * smax
            p25 = np.array(d["p25"])[mask] * smax
            p50 = np.array(d["p50"])[mask] * smax
            p75 = np.array(d["p75"])[mask] * smax
            p95 = np.array(d["p95"])[mask] * smax

            ax.fill_between(steps, p05, p95, color=color, alpha=0.08 * alpha_mult)
            ax.fill_between(steps, p25, p75, color=color, alpha=0.18 * alpha_mult)
            ax.plot(steps, p50, color=color, linewidth=2.0, label=f"{run_label} p50")
            ax.plot(steps, smax, color=color, linewidth=0.8, linestyle="--",
                    alpha=0.4, label=f"{run_label} σ_max")

        ax.set_xlabel("Step")
        if col_idx == 0:
            ax.set_ylabel(f"{layer_label}\nSingular Value (absolute)",
                          fontsize=9)
        ax.set_title(f"{cfg['label']}", fontsize=10, fontweight="bold",
                     color=cfg["color"])

        if "largest" in lid:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.2, which="both")

        if col_idx == 0:
            ax.legend(framealpha=0.9, fontsize=7, loc="upper right")

plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT}")
print(f"Baseline progress: step {steps_bl[-1]}/4357 ({100*steps_bl[-1]/4357:.0f}%)")
print(f"Baseline latest val_bpb: {bpbs_bl[-1]:.4f}")
