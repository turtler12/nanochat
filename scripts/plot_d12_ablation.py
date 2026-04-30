"""
D12 ablation: val BPB vs FTRL→Muon switch step (all 4gpu, d12, lr=0.5, same eta schedule).

Usage:
    python -m scripts.plot_d12_ablation
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

MY = Path("my_runs")
OUT = Path("plots/headline/d12_ablation.png")

# ── load val logs ──────────────────────────────────────────────────────────
def final_bpb(path):
    rows = [json.loads(l) for l in open(path) if "_config" not in l]
    return min(r["val_bpb"] for r in rows)

hybrid_switch_steps = [100, 150, 200, 250, 300, 400]
hybrid_bpbs = [
    final_bpb(MY / f"val_log_ftrl_then_muon_s{s}_d12_4gpu.jsonl")
    for s in hybrid_switch_steps
]

muon_bpb = final_bpb(MY / "val_log_muon_d12_4gpu.jsonl")

print(f"Muon (pure):  {muon_bpb:.4f}")
for s, b in zip(hybrid_switch_steps, hybrid_bpbs):
    print(f"Hybrid @{s:3d}: {b:.4f}  (Δ vs muon: {b - muon_bpb:+.4f})")

# ── plot ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(9, 5))

# hybrid curve
ax.plot(hybrid_switch_steps, hybrid_bpbs,
        color="#e377c2", linewidth=2.5, marker="o", markersize=8,
        zorder=3, label="FTRL→Muon hybrid (4gpu, d12)")

# annotate each point
_OPT_MUON = 3531.0
_OPT_FTRL = 2119.0
_STEPS = 2205
_muon_opt_total = _OPT_MUON * _STEPS

best_step = hybrid_switch_steps[int(np.argmin(hybrid_bpbs))]
for s, b in zip(hybrid_switch_steps, hybrid_bpbs):
    saved_pct = (1 - (_OPT_FTRL * s + _OPT_MUON * (_STEPS - s)) / _muon_opt_total) * 100
    is_best = s == best_step
    yoff_bpb = -16 if is_best else 10
    yoff_pct = -27 if is_best else 21
    ax.annotate(f"{b:.4f}", xy=(s, b), xytext=(0, yoff_bpb),
                textcoords="offset points", ha="center", fontsize=9,
                color="#e377c2", fontweight="bold" if is_best else "normal")
    ax.annotate(f"{saved_pct:.1f}% saved", xy=(s, b), xytext=(0, yoff_pct),
                textcoords="offset points", ha="center", fontsize=8,
                color="#b0457a", fontweight="bold" if is_best else "normal")

# muon baseline
ax.axhline(muon_bpb, color="#1f77b4", linewidth=1.8, linestyle="--",
           label=f"Pure Muon (4gpu, d12)  {muon_bpb:.4f}")


# highlight best hybrid
best_bpb = min(hybrid_bpbs)
ax.axhline(best_bpb, color="#e377c2", linewidth=0.8, linestyle=":", alpha=0.6)

# gap bracket between best hybrid and muon
gap = best_bpb - muon_bpb
x_arrow = 420
ax.annotate("", xy=(x_arrow, muon_bpb), xytext=(x_arrow, best_bpb),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
ax.text(x_arrow + 4, (muon_bpb + best_bpb) / 2,
        f"Δ{gap:+.4f}", color="red", fontsize=9, va="center")

ax.set_xlabel("FTRL phase length (steps before switching to Muon)")
ax.set_ylabel("Final Val BPB")
ax.set_title("D12 Ablation: FTRL→Muon Switch Step vs Val BPB\n(all 4gpu, lr=0.5, exp-abs η schedule, 2205 total steps)")
ax.set_xticks(hybrid_switch_steps)
ax.legend(fontsize=10, loc="upper right")

# tight y range to show the variation clearly
all_bpbs = list(hybrid_bpbs) + [muon_bpb]
ymin, ymax = min(all_bpbs), max(all_bpbs)
pad = (ymax - ymin) * 0.6
ax.set_ylim(ymin - pad, ymax + pad)

plt.tight_layout()
OUT.parent.mkdir(exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
