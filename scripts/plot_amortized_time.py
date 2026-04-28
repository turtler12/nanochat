"""Plot smoothed train loss vs wall time for batch-1 amortized runs vs Muon baseline."""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_FILES = {
    "Muon (15 matmuls)":               "slurm_logs/abl_abl_muon_720980.log",
    "amort_2_e0  (7.5 mat/step, η=0)": "slurm_logs/amort_amort_amort_2_e0_744666.log",
    "amort_2_e01 (7.5 mat/step, η=0.1)":"slurm_logs/amort_amort_amort_2_e01_744667.log",
    "amort_3_e0  (5.0 mat/step, η=0)": "slurm_logs/amort_amort_amort_3_e0_744668.log",
}

STYLES = {
    "Muon (15 matmuls)":               dict(color="black",   ls="--", lw=2.0, zorder=10),
    "amort_2_e0  (7.5 mat/step, η=0)": dict(color="#1f77b4", ls="-",  lw=1.5, zorder=5),
    "amort_2_e01 (7.5 mat/step, η=0.1)":dict(color="#ff7f0e", ls="-", lw=1.5, zorder=5),
    "amort_3_e0  (5.0 mat/step, η=0)": dict(color="#2ca02c", ls="-",  lw=1.5, zorder=5),
}

STEP_RE = re.compile(
    r"step\s+(\d+)/\d+.*?loss:\s+([\d.]+).*?dt:\s+([\d.]+)ms"
)

import os
BASE = os.path.join(os.path.dirname(__file__), "..")


def parse_log(path):
    steps, losses, dts = [], [], []
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                dts.append(float(m.group(3)) / 1000.0)  # seconds
    return steps, losses, dts


def build_time_series(steps, losses, dts, warmup=10):
    """Accumulate wall time, skip first `warmup` steps for compile/warmup noise."""
    times, smooth = [], []
    t = 0.0
    for i, (s, l, dt) in enumerate(zip(steps, losses, dts)):
        if i >= warmup:
            t += dt
        times.append(t / 60.0)  # minutes
        smooth.append(l)
    return times, smooth


fig, ax = plt.subplots(figsize=(9, 5.5))

for label, relpath in LOG_FILES.items():
    path = os.path.join(BASE, relpath)
    steps, losses, dts = parse_log(path)
    times, smooth = build_time_series(steps, losses, dts)
    st = STYLES[label]
    ax.plot(times, smooth, label=label,
            color=st["color"], ls=st["ls"], lw=st["lw"], zorder=st["zorder"],
            alpha=0.85)
    # Annotate final value
    ax.annotate(f'{smooth[-1]:.4f}',
                xy=(times[-1], smooth[-1]),
                xytext=(5, 0), textcoords="offset points",
                color=st["color"], fontsize=8, va="center", fontweight="bold")

ax.set_xlabel("Wall Time (minutes)", fontsize=12)
ax.set_ylabel("Smoothed Train Loss", fontsize=12)
ax.set_title("Amortized NS — Train Loss vs Wall Time (1-GPU, d12, lr=0.02)", fontsize=12)
ax.set_ylim(0.5, 4.0)
ax.set_xlim(left=0)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(BASE, "plots/amortized_batch1_time.png")
plt.savefig(out, dpi=150)
print(f"Saved {out}")
