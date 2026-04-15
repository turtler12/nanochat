"""Plot LowRankSoftmaxMuon results: step vs loss and time vs loss, with baseline."""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

BASE = "cache/base_checkpoints"
OUT_DIR = "softmax_sweep_results"
os.makedirs(OUT_DIR, exist_ok=True)

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if "_config" not in obj:
                rows.append(obj)
    return rows

runs = {
    "Baseline Muon (2 GPU)": f"{BASE}/wallclock_baseline_d20",
    "Full SVD Softmax η=0.1 (fp32)": f"{BASE}/smxbf16_eta0.1_d20",
    "MedRank k=64 Softmax η=0.1": f"{BASE}/medrank_k64_eta0.1_d20",
}

colors = {
    "Baseline Muon (2 GPU)": "#888888",
    "Full SVD Softmax η=0.1 (fp32)": "#e41a1c",
    "MedRank k=64 Softmax η=0.1": "#1f77b4",
}
linewidths = {
    "Baseline Muon (2 GPU)": 2.5,
    "Full SVD Softmax η=0.1 (fp32)": 2.5,
    "MedRank k=64 Softmax η=0.1": 2.5,
}
linestyles = {
    "Baseline Muon (2 GPU)": "--",
    "Full SVD Softmax η=0.1 (fp32)": "-",
    "MedRank k=64 Softmax η=0.1": "-",
}

smooth_window = 50

# Load all train data
train_data = {}
for name, path in runs.items():
    tp = os.path.join(path, "train_log.jsonl")
    if not os.path.exists(tp):
        print(f"  Missing: {tp}")
        continue
    data = load_jsonl(tp)
    if len(data) < smooth_window + 1:
        print(f"  Skipping {name}: only {len(data)} points")
        continue
    steps = np.array([d["step"] for d in data])
    losses = np.array([d["loss"] for d in data])
    times_sec = np.array([d["dt"] for d in data])
    times_min = np.cumsum(times_sec) / 60.0

    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(losses, kernel, mode="valid")
    s_steps = steps[smooth_window - 1:]
    s_times = times_min[smooth_window - 1:]

    train_data[name] = {
        "steps": s_steps, "times": s_times, "loss": smoothed,
        "n": len(data), "final_loss": smoothed[-1],
    }
    print(f"  {name}: {len(data)} steps, final smoothed loss={smoothed[-1]:.4f}")

# =========================================================================
# Figure: Step vs Loss + Time vs Loss side by side
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

for name, d in train_data.items():
    kw = dict(
        color=colors[name], linewidth=linewidths[name],
        linestyle=linestyles[name],
        alpha=0.6 if "Baseline" in name else 1.0,
        zorder=1 if "Baseline" in name else 2,
        label=f"{name} (final={d['final_loss']:.3f})",
    )
    ax1.plot(d["steps"], d["loss"], **kw)
    ax2.plot(d["times"], d["loss"], **kw)

ax1.set_xlabel("Training Step", fontsize=12)
ax1.set_ylabel("Training Loss (smoothed)", fontsize=12)
ax1.set_title("Step vs Loss", fontsize=13)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Wall-Clock Time (minutes)", fontsize=12)
ax2.set_ylabel("Training Loss (smoothed)", fontsize=12)
ax2.set_title("Time vs Loss", fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle("SVD Softmax Muon vs Baseline (2 GPU)", fontsize=14, y=1.01)
fig.tight_layout()
out = os.path.join(OUT_DIR, "svd_softmax_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
