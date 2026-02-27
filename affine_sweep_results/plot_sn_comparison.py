"""
Analyze all affine sweep results: 3 intercepts x 2 SN modes = 6 runs.

The experiment replaces Muon's Polar Express orthogonalization with an SVD-based
affine mapping on singular values: f_i(sigma) = (1 - i) * sigma + i

Two variants per intercept:
  - YES_SN: sigma is first normalized by sigma_max (spectral normalization)
  - NO_SN:  sigma is used raw

Intercept semantics (with YES_SN, where sigma in [0,1]):
  i=0.0 -> f(s) = s          (preserves relative singular value magnitudes)
  i=0.5 -> f(s) = 0.5s + 0.5 (compresses toward 1, partial equalization)
  i=1.0 -> f(s) = 1           (all singular values = 1, full orthogonalization = Muon)
"""

import json
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results_dir = os.path.dirname(os.path.abspath(__file__))

# ── Load all runs ────────────────────────────────────────────────────
runs = {}
for d in sorted(glob.glob(os.path.join(results_dir, "affine_*"))):
    vl = os.path.join(d, "val_loss.json")
    if not os.path.isfile(vl):
        continue
    with open(vl) as f:
        data = json.load(f)
    runs[data["run_tag"]] = data

print(f"Loaded {len(runs)} runs: {list(runs.keys())}")

def get_curve(data):
    steps = [p["step"] for p in data["val_loss_log"]]
    bpb = [p["val_bpb"] for p in data["val_loss_log"]]
    return steps, bpb

intercepts = [0.0, 0.5, 1.0]

# ── Build figure: 2x2 layout ────────────────────────────────────────
fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

fig.suptitle(
    "Affine SVD Mapping Sweep: Dissecting Muon's Orthogonalization\n"
    r"$f_i(\sigma) = (1-i)\,\sigma + i$"
    "    |    depth=20, 4xH100 NVL, 4357 steps, ~1M batch",
    fontsize=13, fontweight="bold",
)

c_yes = {"0.0": "#2563eb", "0.5": "#7c3aed", "1.0": "#0891b2"}
c_no  = {"0.0": "#dc2626", "0.5": "#ea580c", "1.0": "#65a30d"}

# ── Panel 1 (top-left): All 6 curves ────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
for i in intercepts:
    i_key = str(i) if i == float(int(i)) else str(i)
    i_str = f"{i:.3f}"
    tag_yes = f"affine_{i_str}_YES_SN"
    tag_no = f"affine_{i_str}_NO_SN"
    ik = f"{i:.1f}"
    if tag_yes in runs:
        s, b = get_curve(runs[tag_yes])
        ax.plot(s, b, "s-", color=c_yes[ik], markersize=3, linewidth=1.5,
                label=f"i={ik} YES_SN ({runs[tag_yes]['min_val_bpb']:.4f})")
    if tag_no in runs:
        s, b = get_curve(runs[tag_no])
        ax.plot(s, b, "o--", color=c_no[ik], markersize=3, linewidth=1.5, alpha=0.8,
                label=f"i={ik} NO_SN  ({runs[tag_no]['min_val_bpb']:.4f})")
ax.set_xlabel("Training Step")
ax.set_ylabel("Validation BPB")
ax.set_title("All Runs: Validation Loss Curves")
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_ylim(0.75, 1.6)

# ── Panel 2 (top-right): Zoomed final convergence ───────────────────
ax = fig.add_subplot(gs[0, 1])
for i in intercepts:
    i_str = f"{i:.3f}"
    ik = f"{i:.1f}"
    tag_yes = f"affine_{i_str}_YES_SN"
    tag_no = f"affine_{i_str}_NO_SN"
    if tag_yes in runs:
        s, b = get_curve(runs[tag_yes])
        ax.plot(s, b, "s-", color=c_yes[ik], markersize=4, linewidth=1.8,
                label=f"i={ik} YES_SN ({runs[tag_yes]['min_val_bpb']:.4f})")
    if tag_no in runs:
        s, b = get_curve(runs[tag_no])
        ax.plot(s, b, "o--", color=c_no[ik], markersize=4, linewidth=1.8, alpha=0.8,
                label=f"i={ik} NO_SN  ({runs[tag_no]['min_val_bpb']:.4f})")
ax.set_xlabel("Training Step")
ax.set_ylabel("Validation BPB")
ax.set_title("Zoomed: Final Convergence Region")
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(2000, 4400)
all_final = [r["min_val_bpb"] for r in runs.values()]
ax.set_ylim(min(all_final) - 0.02, max(all_final) + 0.02)

# ── Panel 3 (bottom-left): Bar chart of final bpb ───────────────────
ax = fig.add_subplot(gs[1, 0])
yes_finals, no_finals = [], []
for i in intercepts:
    i_str = f"{i:.3f}"
    yes_finals.append(runs.get(f"affine_{i_str}_YES_SN", {}).get("min_val_bpb"))
    no_finals.append(runs.get(f"affine_{i_str}_NO_SN", {}).get("min_val_bpb"))

x = range(len(intercepts))
w = 0.35
bars_no = ax.bar([xi - w/2 for xi in x], no_finals, w, color="#ef4444", alpha=0.8,
                 label="NO_SN", edgecolor="white")
bars_yes = ax.bar([xi + w/2 for xi in x], yes_finals, w, color="#3b82f6", alpha=0.8,
                  label="YES_SN", edgecolor="white")
for bar, val in zip(bars_no, no_finals):
    if val:
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.4f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#b91c1c")
for bar, val in zip(bars_yes, yes_finals):
    if val:
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.4f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1d4ed8")
ax.set_xticks(list(x))
ax.set_xticklabels([f"i = {i}" for i in intercepts])
ax.set_xlabel("Affine Intercept")
ax.set_ylabel("Final Validation BPB (lower = better)")
ax.set_title("Final BPB by Configuration")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(min(all_final) - 0.05, max(all_final) + 0.05)

# ── Panel 4 (bottom-right): SN gap analysis ─────────────────────────
ax = fig.add_subplot(gs[1, 1])
gaps = []
gap_labels = []
for idx, i in enumerate(intercepts):
    i_str = f"{i:.3f}"
    no_val = no_finals[idx]
    yes_val = yes_finals[idx]
    if no_val and yes_val:
        gaps.append(no_val - yes_val)
        gap_labels.append(f"i = {i}")

bar_colors = ["#f87171", "#a78bfa", "#67e8f9"]
bars = ax.bar(range(len(gaps)), gaps, color=bar_colors, edgecolor="white", width=0.5)
for idx, (bar, gap) in enumerate(zip(bars, gaps)):
    no_val = no_finals[idx]
    pct = gap / no_val * 100 if no_val else 0
    label = f"{gap:+.4f} bpb\n({pct:+.1f}%)"
    y_pos = max(gap, 0) + 0.002
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(gap_labels)))
ax.set_xticklabels(gap_labels)
ax.set_xlabel("Affine Intercept")
ax.set_ylabel("BPB Gap (NO_SN - YES_SN)")
ax.set_title("Effect of Spectral Normalization")
ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
ax.grid(True, alpha=0.3, axis="y")

interp = (
    "Interpretation:\n"
    "  i=0: SN is everything (22.6% improvement)\n"
    "  i=0.5: SN barely matters (affine map already compresses)\n"
    "  i=1: SN is irrelevant (all s->1 regardless)"
)
ax.text(0.98, 0.95, interp, transform=ax.transAxes, fontsize=7.5,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.9))

output_path = os.path.join(results_dir, "affine_sweep_full.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to: {output_path}")

# ── Summary table ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Affine SVD Mapping Sweep - Final Results Summary")
print(f"  f_i(s) = (1-i)*s + i,  with or without s -> s/s_max normalization")
print(f"{'='*70}")
print(f"  {'Intercept':>10} {'NO_SN':>10} {'YES_SN':>10} {'Gap':>10} {'SN helps?':>12}")
print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
for idx, i in enumerate(intercepts):
    no_val = no_finals[idx]
    yes_val = yes_finals[idx]
    gap = no_val - yes_val if no_val and yes_val else 0
    pct = gap / no_val * 100 if no_val else 0
    verdict = f"YES ({pct:.1f}%)" if gap > 0.005 else "negligible"
    print(f"  i = {i:<5}  {no_val:.4f}     {yes_val:.4f}     {gap:+.4f}   {verdict:>12}")
print(f"{'='*70}")
