"""
Plot spectral-padded Muon experiment: SV diagnostics.

Layout (3×2):
  Row 0: Val BPB comparison          | σ_max trajectories (abs, log)
  Row 1: Large matrix fan charts     — Pre vs Blended overlay (abs) | Polar (normalized)
  Row 2: Small matrix fan charts     — Pre vs Blended overlay (abs) | Polar (normalized)

Usage:
  python scripts/plot_spectralpadded.py
"""

import json
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
SPECTRAL = "spectralpadded_results/spectralpad_gamma0.1_pi1_pb4"
BASELINE = "padded_muon_results/baseline"
OUTPUT = "spectralpadded_results/comparison.png"

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

def get_abs(d):
    """Get absolute SV values, skipping step 0."""
    mask = np.array(d["steps"]) > 0
    smax = np.array(d["sigma_max"])[mask]
    steps = np.array(d["steps"])[mask]
    out = {"steps": steps, "sigma_max": smax}
    for f in ["p05", "p25", "p50", "p75", "p95"]:
        out[f] = np.array(d[f])[mask] * smax
    sigma_hat_raw = [d["sigma_hat"][i] for i, m in enumerate(mask) if m]
    out["sigma_hat"] = np.array([float(x) if x is not None else np.nan for x in sigma_hat_raw])
    return out

def get_norm(d):
    """Get normalized SV values (σ/σ_max), skipping step 0."""
    mask = np.array(d["steps"]) > 0
    steps = np.array(d["steps"])[mask]
    out = {"steps": steps, "sigma_max": np.array(d["sigma_max"])[mask]}
    for f in ["p05", "p25", "p50", "p75", "p95"]:
        out[f] = np.array(d[f])[mask]
    return out

# ── Load data ────────────────────────────────────────────────────────────────
spectral_val = load_val(f"{SPECTRAL}/val_loss.json")
baseline_val = load_val(f"{BASELINE}/val_loss.json")
spectral_sv = group_sv(load_sv(f"{SPECTRAL}/sv_log.jsonl"))

large_layer = small_layer = None
for (lid, _) in spectral_sv:
    if "largest" in lid:  large_layer = lid
    if "smallest" in lid: small_layer = lid

gamma = spectral_val["pad_ratio"]

# ── Helper: draw a fan chart ────────────────────────────────────────────────
def fan(ax, steps, d, color, label, draw_smax=False):
    """Draw p05-p95 outer + p25-p75 inner bands + p50 median."""
    ax.fill_between(steps, d["p05"], d["p95"], color=color, alpha=0.10)
    ax.fill_between(steps, d["p25"], d["p75"], color=color, alpha=0.22)
    ax.plot(steps, d["p50"], color=color, linewidth=2.2, label=label)
    if draw_smax:
        ax.plot(steps, d["sigma_max"], color=color, linewidth=1.0,
                linestyle="--", alpha=0.5, label=f"{label} σ_max")

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(15, 15),
                         gridspec_kw={"hspace": 0.32, "wspace": 0.25,
                                      "height_ratios": [0.85, 1, 1]})
fig.suptitle(
    f"Spectral-Padded Muon  (γ = {gamma},  power_iters = {spectral_val['power_iters']},"
    f"  power_batch = {spectral_val['power_batch']})"
    f"\nFinal BPB: spectral-pad {spectral_val['min_val_bpb']:.4f}  vs  baseline {baseline_val['min_val_bpb']:.4f}"
    f"  (Δ = +{spectral_val['min_val_bpb'] - baseline_val['min_val_bpb']:.4f})",
    fontsize=13, fontweight="bold", y=1.005,
)

# ═══════════════════════════════════════════════════════════════════════════════
# [0,0] Val BPB
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0, 0]
for data, name, color, ls in [
    (baseline_val, "Baseline", "#2563eb", "-"),
    (spectral_val, f"Spectral-pad γ={gamma}", "#ea580c", "--"),
]:
    steps = [e["step"] for e in data["val_loss_log"]]
    bpbs = [e["val_bpb"] for e in data["val_loss_log"]]
    ax.plot(steps, bpbs, color=color, linestyle=ls,
            label=f"{name}  →  {bpbs[-1]:.4f}", linewidth=2.2,
            marker="o", markersize=2.5)
ax.set_xlabel("Step")
ax.set_ylabel("Val BPB ↓")
ax.set_title("Validation Loss", fontsize=12, fontweight="bold")
ax.legend(framealpha=0.9, fontsize=9.5, loc="upper right")
ax.grid(True, alpha=0.2)

# ═══════════════════════════════════════════════════════════════════════════════
# [0,1] σ_max trajectories (absolute, log scale)
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0, 1]
stage_colors = {"pre": "#2563eb", "polar": "#16a34a", "blended": "#ea580c"}
stage_labels = {"pre": "Pre-ortho", "polar": "Polar Express", "blended": "Blended"}
for lid, mk, lbl in [(large_layer, "o", "5120×1280"), (small_layer, "s", "10×32")]:
    if lid is None: continue
    for stage in ["pre", "polar", "blended"]:
        key = (lid, stage)
        if key not in spectral_sv: continue
        d = get_abs(spectral_sv[key])
        ax.plot(d["steps"], d["sigma_max"], color=stage_colors[stage],
                marker=mk, markersize=3.5, linewidth=1.5, alpha=0.8,
                label=f"{lbl} {stage_labels[stage]}")
ax.set_yscale("log")
ax.set_xlabel("Step")
ax.set_ylabel("σ_max (absolute)")
ax.set_title("σ_max Trajectories", fontsize=12, fontweight="bold")
ax.legend(framealpha=0.9, fontsize=7.5, ncol=2, loc="center right")
ax.grid(True, alpha=0.2, which="both")

# ═══════════════════════════════════════════════════════════════════════════════
# Rows 1-2: Fan charts for large / small matrices
# Col 0: Pre vs Blended overlay (absolute SVs) — the key comparison
# Col 1: Polar stage (normalized) — shows Polar Express quality
# ═══════════════════════════════════════════════════════════════════════════════
for row, (lid, layer_name) in enumerate(
    [(large_layer, "Large Matrix [5120 × 1280]"),
     (small_layer, "Small Matrix [10 × 32]")],
    start=1
):
    if lid is None: continue

    # ── Left: Pre vs Blended (absolute) ──────────────────────────────────────
    ax = axes[row, 0]
    pre_key = (lid, "pre")
    blend_key = (lid, "blended")

    if pre_key in spectral_sv:
        d = get_abs(spectral_sv[pre_key])
        fan(ax, d["steps"], d, "#2563eb", "Pre-ortho", draw_smax=True)
    if blend_key in spectral_sv:
        d = get_abs(spectral_sv[blend_key])
        fan(ax, d["steps"], d, "#ea580c", "After blend", draw_smax=True)
        # Draw the spectral floor γ·σ̂
        valid = ~np.isnan(d["sigma_hat"])
        if valid.any():
            floor = gamma * d["sigma_hat"][valid]
            ax.plot(d["steps"][valid], floor, color="#a855f7",
                    linewidth=2, linestyle=":", label=f"γ·σ̂(g) = {gamma}·σ_max(grad)")

    ax.set_xlabel("Step")
    ax.set_ylabel("Singular Value (absolute)")
    ax.set_title(f"{layer_name} — Pre vs Blended",
                 fontsize=11, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

    # Annotate the band meanings
    if row == 1:
        ax.annotate("shaded = p25–p75 (dark) & p05–p95 (light)",
                    xy=(0.98, 0.02), xycoords="axes fraction",
                    fontsize=7, ha="right", color="#666", fontstyle="italic")

    # ── Right: Polar stage (normalized σ/σ_max) ─────────────────────────────
    ax = axes[row, 1]
    polar_key = (lid, "polar")
    if polar_key in spectral_sv:
        d = get_norm(spectral_sv[polar_key])
        fan(ax, d["steps"], d, "#16a34a", "Polar Express")
        # Reference line at 1.0
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("σ / σ_max  (normalized)")
    ax.set_title(f"{layer_name} — Polar Express Output",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(framealpha=0.9, fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.2)

    if row == 1:
        ax.annotate("Ideal: all SVs = 1 (perfect orthogonalization)",
                    xy=(0.98, 0.02), xycoords="axes fraction",
                    fontsize=7, ha="right", color="#666", fontstyle="italic")

plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT}")
