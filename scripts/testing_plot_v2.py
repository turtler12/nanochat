"""
Live comparison: v2 spectral-padded Muon runs.
Auto-detects latest SLURM logs and all result directories.

Usage:
  cd /data/scratch/medhaven/nanochat && .venv/bin/python scripts/testing_plot_v2.py
"""

import json
import re
import os
import glob
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Auto-detect paths ────────────────────────────────────────────────────────
RESULTS_DIR = "spectralpadded_v2_results"
LOG_DIR = "slurm_logs"
OUTPUT = f"{RESULTS_DIR}/comparison.png"

def find_latest_log(tag):
    """Find the latest SLURM log for a given run tag (e.g. 'baseline', 'spectralpad_gamma0.1')."""
    pattern = os.path.join(LOG_DIR, "spectralpad_v2_*.log")
    candidates = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            head = f.read(500)
        # Match PAD_RATIO=X followed by comma or end-of-line to avoid 0.0 matching 0.05
        if tag == "baseline":
            if re.search(r"PAD_RATIO=0\.0[,\s]", head):
                candidates.append(path)
        else:
            if re.search(rf"PAD_RATIO={re.escape(tag)}[,\s]", head):
                candidates.append(path)
    return candidates[-1] if candidates else None

def find_sv_log(run_dir):
    path = os.path.join(run_dir, "sv_log.jsonl")
    return path if os.path.exists(path) else None

# Build run configs: discover all subdirs in results
RUN_CONFIGS = []
for d in sorted(os.listdir(RESULTS_DIR)):
    full = os.path.join(RESULTS_DIR, d)
    if not os.path.isdir(full):
        continue
    if d == "baseline":
        log = find_latest_log("baseline")
        RUN_CONFIGS.append({"tag": "baseline", "gamma": 0.0, "color": "#2563eb",
                            "ls": "-", "log": log, "sv": find_sv_log(full), "dir": full})
    elif d.startswith("spectralpad_gamma"):
        # Parse gamma from dir name like spectralpad_gamma0.1_pi1_pb4
        m = re.match(r"spectralpad_gamma([\d.]+)", d)
        if m:
            gamma = float(m.group(1))
            log = find_latest_log(str(gamma))
            # Assign distinct colors for different gammas
            colors = {"0.1": "#ea580c", "0.05": "#a855f7", "0.2": "#16a34a"}
            color = colors.get(str(gamma), "#888888")
            RUN_CONFIGS.append({"tag": d, "gamma": gamma, "color": color,
                                "ls": "--", "log": log, "sv": find_sv_log(full), "dir": full})

print(f"Found {len(RUN_CONFIGS)} runs:")
for rc in RUN_CONFIGS:
    print(f"  {rc['tag']}: gamma={rc['gamma']}, log={rc['log']}, sv={rc['sv']}")

# ── Stage config ─────────────────────────────────────────────────────────────
STAGE_CFG = {
    "pre":     {"color": "#2563eb", "label": "Pre-ortho (after momentum)"},
    "polar":   {"color": "#16a34a", "label": "After Polar Express"},
    "blended": {"color": "#ea580c", "label": "After spectral blend"},
}

# ── Data loading ─────────────────────────────────────────────────────────────
def parse_val_from_log(path):
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
runs = []
for rc in RUN_CONFIGS:
    val_log = parse_val_from_log(rc["log"]) if rc["log"] else []
    sv_data = group_sv(load_sv(rc["sv"])) if rc["sv"] else {}
    if not val_log:
        print(f"  WARNING: no val data for {rc['tag']}, skipping")
        continue
    steps = [e["step"] for e in val_log]
    bpbs = [e["val_bpb"] for e in val_log]
    pct = 100 * steps[-1] / 4357
    runs.append({**rc, "val_log": val_log, "sv": sv_data,
                 "steps": steps, "bpbs": bpbs, "pct": pct})

# Find layer ids from any available SV data
large_layer = small_layer = None
for r in runs:
    for (lid, _) in r["sv"]:
        if "largest" in lid:  large_layer = lid
        if "smallest" in lid: small_layer = lid

# ── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
gs_top = fig.add_gridspec(1, 2, left=0.06, right=0.98, top=0.92, bottom=0.68,
                          wspace=0.25, width_ratios=[1, 2])
gs_bot = fig.add_gridspec(2, 3, left=0.06, right=0.98, top=0.62, bottom=0.04,
                          hspace=0.40, wspace=0.30, height_ratios=[1, 1])

subtitle_parts = []
for r in runs:
    subtitle_parts.append(f"γ={r['gamma']} ({r['pct']:.0f}%)")
fig.suptitle(
    f"Spectral-Padded Muon v2 (fixed blend)\n"
    f"{' vs '.join(subtitle_parts)}",
    fontsize=14, fontweight="bold",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Row 0, col 0: Val BPB
# ═══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs_top[0, 0])
for r in runs:
    label = f"γ={r['gamma']}"
    if r["gamma"] == 0:
        label = "Baseline γ=0"
    ax.plot(r["steps"], r["bpbs"], color=r["color"], linestyle=r["ls"],
            linewidth=2.2, marker="o", markersize=2.5,
            label=f"{label}  →  {r['bpbs'][-1]:.4f}")
ax.set_xlabel("Step")
ax.set_ylabel("Val BPB ↓")
ax.set_title("Validation Loss", fontsize=12, fontweight="bold")
ax.legend(framealpha=0.9, fontsize=9, loc="upper right")
ax.grid(True, alpha=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# Row 0, col 1: σ_max comparison
# ═══════════════════════════════════════════════════════════════════════════════
ax = fig.add_subplot(gs_top[0, 1])
for r in runs:
    label = f"γ={r['gamma']}"
    for lid, marker, layer_lbl in [
        (large_layer, "o", "5120×1280"),
        (small_layer, "s", "10×32"),
    ]:
        if lid is None: continue
        for stage, cfg in STAGE_CFG.items():
            key = (lid, stage)
            if key not in r["sv"]: continue
            d = r["sv"][key]
            steps = np.array(d["steps"][1:])
            smax = np.array(d["sigma_max"][1:])
            ax.plot(steps, smax, color=cfg["color"], marker=marker,
                    markersize=3.5, linewidth=1.5, alpha=0.8, linestyle=r["ls"],
                    label=f"{label} {layer_lbl} {cfg['label']}")
ax.set_yscale("log")
ax.set_xlabel("Step")
ax.set_ylabel("σ_max (absolute)")
ax.set_title("σ_max Trajectories", fontsize=12, fontweight="bold")
ax.legend(framealpha=0.9, fontsize=6, ncol=3, loc="upper right")
ax.grid(True, alpha=0.2, which="both")

# ═══════════════════════════════════════════════════════════════════════════════
# Rows 1-2: Fan charts — all runs overlaid for each stage
# ═══════════════════════════════════════════════════════════════════════════════
for row_idx, (lid, layer_label) in enumerate(
    [(large_layer, "Large Matrix [5120 × 1280]"),
     (small_layer, "Small Matrix [10 × 32]")],
):
    if lid is None: continue

    for col_idx, stage in enumerate(["pre", "polar", "blended"]):
        ax = fig.add_subplot(gs_bot[row_idx, col_idx])
        cfg = STAGE_CFG[stage]

        for r in runs:
            key = (lid, stage)
            if key not in r["sv"]: continue
            d = r["sv"][key]
            mask = np.array(d["steps"]) > 0
            steps = np.array(d["steps"])[mask]
            smax = np.array(d["sigma_max"])[mask]

            p05 = np.array(d["p05"])[mask] * smax
            p25 = np.array(d["p25"])[mask] * smax
            p50 = np.array(d["p50"])[mask] * smax
            p75 = np.array(d["p75"])[mask] * smax
            p95 = np.array(d["p95"])[mask] * smax

            rlabel = f"γ={r['gamma']}"
            ax.fill_between(steps, p05, p95, color=r["color"], alpha=0.08)
            ax.fill_between(steps, p25, p75, color=r["color"], alpha=0.18)
            ax.plot(steps, p50, color=r["color"], linewidth=2.0, label=f"{rlabel} p50")
            ax.plot(steps, smax, color=r["color"], linewidth=0.8, linestyle="--",
                    alpha=0.4, label=f"{rlabel} σ_max")

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
for r in runs:
    print(f"  γ={r['gamma']:4.2f}:  step {r['steps'][-1]}/4357 ({r['pct']:.0f}%), val_bpb={r['bpbs'][-1]:.4f}")
