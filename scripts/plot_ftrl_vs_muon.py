"""
Plot val BPB vs step and val BPB vs wall-clock time for:
  - muon (baseline)
  - ns3_ftrl_eta0p1 (winner)

Usage:
    python -m scripts.plot_ftrl_vs_muon
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLURM_LOGS = Path("slurm_logs")
OUT = Path("plots/ftrl_vs_muon.png")

RUNS = {
    "Muon (15 matmuls)":           "abl_abl_muon_720980.log",
    "ns3+FTRL η=0.1 (9 matmuls)": "abl_ns3_ftrl_eta0p1_723737.log",
}

STYLES = {
    "Muon (15 matmuls)":           dict(color="black",   linestyle="--", linewidth=2, marker="o", markersize=4),
    "ns3+FTRL η=0.1 (9 matmuls)": dict(color="#2196F3", linestyle="-",  linewidth=2, marker="o", markersize=4),
}

LABEL_YOFFSET = {
    "Muon (15 matmuls)":           -10,
    "ns3+FTRL η=0.1 (9 matmuls)":  +10,
}


def parse_log(path: Path):
    step_pat = re.compile(r"^step (\d+)/\d+.*\| dt: ([\d.]+)ms")
    val_pat  = re.compile(r"^Step (\d+) \| val bpb: ([\d.]+)")

    steps, dts = [], []
    val_steps, val_bpb = [], []

    for line in path.read_text().splitlines():
        m = step_pat.match(line)
        if m:
            steps.append(int(m.group(1)))
            dts.append(float(m.group(2)) / 1000.0)
            continue
        m = val_pat.match(line)
        if m:
            val_steps.append(int(m.group(1)))
            val_bpb.append(float(m.group(2)))

    step_dt = {s: dt for s, dt in zip(steps, dts)}
    cum = 0.0
    cum_time = {0: 0.0}
    for s in sorted(step_dt):
        if s == 0:
            continue
        cum += step_dt[s]
        cum_time[s] = cum

    def lookup_time(vs):
        best = max((s for s in cum_time if s <= vs), default=0)
        return cum_time[best]

    val_times = [lookup_time(s) / 60.0 for s in val_steps]

    # dense step→time for the time vs step panel (all steps, skip warmup step 0)
    dense_steps = sorted(s for s in cum_time if s > 0)
    dense_times = [cum_time[s] / 60.0 for s in dense_steps]

    return val_steps, val_times, val_bpb, dense_steps, dense_times


# ── parse ──────────────────────────────────────────────────────────────────
data = {}
for label, fname in RUNS.items():
    val_steps, val_times, val_bpb, dense_steps, dense_times = parse_log(SLURM_LOGS / fname)
    data[label] = dict(steps=val_steps, times=val_times, bpb=val_bpb,
                       dense_steps=dense_steps, dense_times=dense_times)
    print(f"{label}: final BPB={val_bpb[-1]:.6f}  ({val_times[-1]:.1f} min)")

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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

for label, d in data.items():
    sty  = STYLES[label]
    yoff = LABEL_YOFFSET[label]
    xs = d["steps"][1:]
    ts = d["times"][1:]
    ys = d["bpb"][1:]

    ax1.plot(xs, ys, label=label, **sty)
    ax2.plot(ts, ys, label=label, **sty)

    # time vs step: dense line, no markers
    dense_sty = {**sty, "marker": None, "markersize": 0}
    ax3.plot(d["dense_steps"], d["dense_times"], label=label, **dense_sty)

    for ax, xval in [(ax1, xs[-1]), (ax2, ts[-1])]:
        ax.annotate(f"{ys[-1]:.4f}", xy=(xval, ys[-1]),
                    xytext=(6, yoff), textcoords="offset points",
                    color=sty["color"], fontsize=9, va="center", fontweight="bold")
    ax3.annotate(f"{d['dense_times'][-1]:.1f}m", xy=(d["dense_steps"][-1], d["dense_times"][-1]),
                 xytext=(6, yoff), textcoords="offset points",
                 color=sty["color"], fontsize=9, va="center", fontweight="bold")

# gap arrow on step plot
muon_final = data["Muon (15 matmuls)"]["bpb"][-1]
ftrl_final = data["ns3+FTRL η=0.1 (9 matmuls)"]["bpb"][-1]
gap = ftrl_final - muon_final
ax1.annotate("", xy=(2050, muon_final), xytext=(2050, ftrl_final),
             arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
ax1.text(2040, (muon_final + ftrl_final) / 2, f"Δ{gap:.4f}",
         color="red", fontsize=9, ha="right", va="center")

ax1.set_xlabel("Step")
ax1.set_ylabel("Validation BPB")
ax1.set_title("Validation BPB vs Training Step")
ax1.legend(fontsize=9)

ax2.set_xlabel("Wall Time (minutes)")
ax2.set_ylabel("Validation BPB")
ax2.set_title("Validation BPB vs Wall Time")
ax2.legend(fontsize=9)

ax3.set_xlabel("Step")
ax3.set_ylabel("Wall Time (minutes)")
ax3.set_title("Wall Time vs Step")
ax3.legend(fontsize=9)

plt.suptitle("Muon vs ns3+FTRL η=0.1  —  d12, lr=0.02, 1×H100", fontsize=12)
plt.tight_layout()
OUT.parent.mkdir(exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
