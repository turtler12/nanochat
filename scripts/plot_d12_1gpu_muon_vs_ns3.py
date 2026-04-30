"""
Val BPB vs step and vs wall time: Muon vs NS3 stock (d12, 1gpu).

Usage:
    python -m scripts.plot_d12_1gpu_muon_vs_ns3
"""

import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("plots/headline/d12_1gpu_muon_vs_ns3.png")

# ── loaders ────────────────────────────────────────────────────────────────

def load_val_jsonl(path):
    steps, bpbs, times = [], [], []
    for line in Path(path).read_text().splitlines():
        d = json.loads(line)
        if "_config" in d:
            continue
        steps.append(d["step"])
        bpbs.append(d["val_bpb"])
        times.append(d["total_training_time"] / 60)
    return np.array(steps), np.array(bpbs), np.array(times)

_STEP_PAT = re.compile(r"^step (\d+)/\d+.*\| dt: ([\d.]+)ms")
_VAL_PAT  = re.compile(r"^Step (\d+) \| val bpb: ([\d.]+)")

def load_val_slurm(path):
    step_dt = {}
    for line in Path(path).read_text().splitlines():
        m = _STEP_PAT.match(line)
        if m:
            s = int(m.group(1))
            if s > 0:
                step_dt[s] = float(m.group(2)) / 1000.0

    cum = 0.0
    cum_time = {0: 0.0}
    for s in sorted(step_dt):
        cum += step_dt[s]
        cum_time[s] = cum

    def lookup(vs):
        best = max((k for k in cum_time if k <= vs), default=0)
        return cum_time[best] / 60.0

    steps, bpbs, times = [], [], []
    for line in Path(path).read_text().splitlines():
        m = _VAL_PAT.match(line)
        if m:
            s = int(m.group(1))
            steps.append(s)
            bpbs.append(float(m.group(2)))
            times.append(lookup(s))
    return np.array(steps), np.array(bpbs), np.array(times)

# ── load ───────────────────────────────────────────────────────────────────
muon_steps, muon_bpbs, muon_times = load_val_slurm(
    "/Users/medha/Desktop/muon_local/nanochat/slurm_logs/abl_abl_muon_720980.log")
ns3_steps,  ns3_bpbs,  ns3_times  = load_val_slurm(
    "/Users/medha/Desktop/muon_local/nanochat/slurm_logs/abl_abl_ns3_720982.log")

for name, bpbs in [("Muon", muon_bpbs), ("NS3 stock", ns3_bpbs)]:
    print(f"{name}: final BPB={bpbs[-1]:.4f}")

# ── plot ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

MUON_STYLE = dict(color="#1f77b4", linewidth=2, marker="o", markersize=5)
NS3_STYLE  = dict(color="#4CAF50", linewidth=2, marker="s", markersize=5, linestyle="-.")

xs_m = muon_steps[1:]
xs_n = ns3_steps[1:]
ys_m = muon_bpbs[1:]
ys_n = ns3_bpbs[1:]

ax1.plot(xs_m, ys_m, label=f"Muon (1gpu)  {ys_m[-1]:.4f}", **MUON_STYLE)
ax1.plot(xs_n, ys_n, label=f"NS3 stock (1gpu)  {ys_n[-1]:.4f}", **NS3_STYLE)

ax1.annotate(f"{ys_m[-1]:.4f}", xy=(xs_m[-1], ys_m[-1]),
             xytext=(6, -12), textcoords="offset points",
             color=MUON_STYLE["color"], fontsize=9, fontweight="bold", va="center")
ax1.annotate(f"{ys_n[-1]:.4f}", xy=(xs_n[-1], ys_n[-1]),
             xytext=(6, 10), textcoords="offset points",
             color=NS3_STYLE["color"], fontsize=9, fontweight="bold", va="center")


ax1.set_xlabel("Step")
ax1.set_ylabel("Validation BPB")
ax1.set_title("Val BPB vs Step")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.4)

plt.suptitle("Muon vs NS3 stock — d12, 1×H100, 2205 steps", fontsize=13)
plt.tight_layout()
OUT.parent.mkdir(exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
