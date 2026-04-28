"""Plot val BPB curves for batch-1 amortized runs vs 1-GPU Muon baseline."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = {
    "Muon (15 matmuls)": {
        "steps": [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2205],
        "bpb":   [3.254653, 1.130779, 1.042014, 1.006763, 0.986753, 0.965703, 0.943465, 0.924104, 0.907889, 0.900093],
        "color": "black", "ls": "--", "lw": 2.0, "zorder": 10,
    },
    "amort_2_e0  (7.5 matmuls/step, η=0)": {
        "steps": [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2205],
        "bpb":   [3.254653, 1.176259, 1.064382, 1.027211, 1.004807, 0.978762, 0.953331, 0.931352, 0.913952, 0.906036],
        "color": "#1f77b4", "ls": "-", "lw": 1.5, "zorder": 5,
    },
    "amort_2_e01 (7.5 matmuls/step, η=0.1)": {
        "steps": [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2205],
        "bpb":   [3.254653, 1.176154, 1.065882, 1.026191, 1.003323, 0.977858, 0.952071, 0.930201, 0.912803, 0.904896],
        "color": "#ff7f0e", "ls": "-", "lw": 1.5, "zorder": 5,
    },
    "amort_3_e0  (5.0 matmuls/step, η=0)": {
        "steps": [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2205],
        "bpb":   [3.254653, 1.201692, 1.084929, 1.056423, 1.008716, 0.984707, 0.964800, 0.936984, 0.920034, 0.912274],
        "color": "#2ca02c", "ls": "-", "lw": 1.5, "zorder": 5,
    },
}

fig, ax = plt.subplots(figsize=(9, 5.5))

for label, d in data.items():
    ax.plot(d["steps"], d["bpb"], label=label,
            color=d["color"], ls=d["ls"], lw=d["lw"], zorder=d["zorder"],
            marker="o", markersize=4)
    # Annotate final value
    ax.annotate(f'{d["bpb"][-1]:.4f}',
                xy=(d["steps"][-1], d["bpb"][-1]),
                xytext=(8, 0), textcoords="offset points",
                color=d["color"], fontsize=8.5, va="center", fontweight="bold")

ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Validation BPB", fontsize=12)
ax.set_title("Amortized NS — Batch 1 vs Muon Baseline (1-GPU, d12, lr=0.02)", fontsize=12)
ax.set_xlim(-50, 2400)
ax.set_ylim(0.885, 1.25)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)

# Gap annotations at final step
muon_final = 0.900093
for label, d in data.items():
    if label.startswith("Muon"):
        continue
    gap = d["bpb"][-1] - muon_final
    print(f"{label}: final={d['bpb'][-1]:.6f}, gap vs muon={gap:+.6f}")

plt.tight_layout()
out = "plots/amortized_batch1.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
