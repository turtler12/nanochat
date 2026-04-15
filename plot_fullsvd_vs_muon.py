"""Plot full SVD softmax vs baseline Muon — step vs loss (4 GPU runs)."""
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
    "Baseline Muon": f"{BASE}/baseline_muon",
    "Full SVD Softmax η=0.1": f"{BASE}/softmax_v2_eta0.1_d20",
    "Full SVD Softmax η=0.5": f"{BASE}/softmax_v2_eta0.5_d20",
    "Full SVD Softmax η=0.75": f"{BASE}/softmax_v2_eta0.75_d20",
    "Full SVD Softmax η=1.0": f"{BASE}/softmax_v2_eta1.0_d20",
}

colors = {
    "Baseline Muon": "#888888",
    "Full SVD Softmax η=0.1": "#e41a1c",
    "Full SVD Softmax η=0.5": "#1f77b4",
    "Full SVD Softmax η=0.75": "#2ca02c",
    "Full SVD Softmax η=1.0": "#ff7f0e",
}
linestyles = {
    "Baseline Muon": "--",
    "Full SVD Softmax η=0.1": "-",
    "Full SVD Softmax η=0.5": "-",
    "Full SVD Softmax η=0.75": "-",
    "Full SVD Softmax η=1.0": "-",
}

smooth_window = 50

# Load train data
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

    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(losses, kernel, mode="valid")
    s_steps = steps[smooth_window - 1:]

    train_data[name] = {"steps": s_steps, "loss": smoothed, "n": len(data), "final_loss": smoothed[-1]}
    print(f"  {name}: {len(data)} steps, final smoothed loss={smoothed[-1]:.4f}")

# Load val data
val_data = {}
for name, path in runs.items():
    vp = os.path.join(path, "val_log.jsonl")
    if not os.path.exists(vp):
        continue
    vd = load_jsonl(vp)
    if len(vd) >= 2:
        val_data[name] = vd

# Figure: step vs loss + val BPB
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

for name, d in train_data.items():
    kw = dict(
        color=colors[name], linewidth=2.5,
        linestyle=linestyles[name],
        alpha=0.6 if "Baseline" in name else 1.0,
        zorder=1 if "Baseline" in name else 2,
        label=f"{name} (loss={d['final_loss']:.3f} @ step {int(d['steps'][-1])})",
    )
    ax1.plot(d["steps"], d["loss"], **kw)

ax1.set_xlabel("Training Step", fontsize=12)
ax1.set_ylabel("Training Loss (smoothed)", fontsize=12)
ax1.set_title("Training Loss vs Step (zoomed)", fontsize=13)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1000, 3000)
ax1.set_ylim(2.6, 3.2)

for name, vd in val_data.items():
    vsteps = [d["step"] for d in vd]
    vbpbs = [d["val_bpb"] for d in vd]
    kw = dict(
        color=colors[name], linewidth=2.0,
        linestyle=linestyles[name], marker="o", markersize=3,
        alpha=0.6 if "Baseline" in name else 1.0,
        zorder=1 if "Baseline" in name else 2,
        label=f"{name} (bpb={vbpbs[-1]:.4f} @ step {vsteps[-1]})",
    )
    ax2.plot(vsteps, vbpbs, **kw)

ax2.set_xlabel("Training Step", fontsize=12)
ax2.set_ylabel("Validation BPB", fontsize=12)
ax2.set_title("Validation BPB vs Step (zoomed)", fontsize=13)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(500, 3000)
ax2.set_ylim(0.82, 1.05)

fig.suptitle("Full SVD Softmax vs Muon Baseline (Step-Matched)", fontsize=14, y=1.01)
fig.tight_layout()
out = os.path.join(OUT_DIR, "full_svd_vs_muon.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
