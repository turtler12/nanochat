"""
Plot verification results for the small (d12) model: Muon vs Adam vs Taylor(eta=0.5).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "cache/base_checkpoints"
OUTPUT = "verifying_results/verify_d12.png"

runs = {
    "Muon (baseline)": {"dir": f"{BASE}/verify_muon_d12", "color": "#2563eb", "ls": "-"},
    "Adam": {"dir": f"{BASE}/verify_adam_d12", "color": "#dc2626", "ls": "-"},
    "Taylor (eta=0.1)": {"dir": f"{BASE}/verify_taylor_eta0.1_d12", "color": "#16a34a", "ls": "-"},
}

def load_val_log(path):
    steps, bpb = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "val_bpb" in rec:
                steps.append(rec["step"])
                bpb.append(rec["val_bpb"])
    return steps, bpb

def load_train_log(path):
    steps, loss, dt = [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "loss" in rec and rec.get("step", -1) >= 0:
                steps.append(rec["step"])
                loss.append(rec["loss"])
                dt.append(rec.get("dt", 0))
    return steps, loss, dt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

for name, cfg in runs.items():
    val_path = os.path.join(cfg["dir"], "val_log.jsonl")
    train_path = os.path.join(cfg["dir"], "train_log.jsonl")
    if not os.path.exists(val_path):
        print(f"Missing: {val_path}")
        continue

    steps, bpb = load_val_log(val_path)
    ax1.plot(steps, bpb, label=name, color=cfg["color"], ls=cfg["ls"], linewidth=1.5)

    if os.path.exists(train_path):
        tsteps, tloss, tdt = load_train_log(train_path)
        # Subsample train loss for readability
        stride = max(1, len(tsteps) // 500)
        ax2.plot(tsteps[::stride], tloss[::stride], label=name, color=cfg["color"], ls=cfg["ls"], linewidth=1.0, alpha=0.8)

        # Wall-clock time vs loss
        times_min = np.cumsum(tdt) / 60.0
        smooth_window = 50
        if len(tloss) > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(tloss, kernel, mode="valid")
            s_times = times_min[smooth_window - 1:]
        else:
            smoothed = tloss
            s_times = times_min
        ax3.plot(s_times, smoothed, label=name, color=cfg["color"], ls=cfg["ls"], linewidth=1.5)

ax1.set_xlabel("Step")
ax1.set_ylabel("Validation BPB")
ax1.set_title("Small Model (d12) — Validation BPB")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Step")
ax2.set_ylabel("Training Loss")
ax2.set_title("Small Model (d12) — Training Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.set_xlabel("Wall-Clock Time (minutes)")
ax3.set_ylabel("Training Loss (smoothed)")
ax3.set_title("Small Model (d12) — Loss vs Time")
ax3.legend()
ax3.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=150)
print(f"Saved: {OUTPUT}")
