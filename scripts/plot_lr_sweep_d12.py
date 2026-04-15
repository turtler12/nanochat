"""
Plot LR sweep results for Hypothesis B test (d12).
Compares best-of-sweep for Muon vs Full SVD eta=0.5 vs Taylor eta=0.1.
Run after all sweep jobs complete.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "cache/base_checkpoints"
OUTPUT = "verifying_results/small_model/lr_sweep_hypothesis_b_d12.png"

# LR sweep configurations
configs = {
    "Muon": {
        0.01: f"{BASE}/verify_muon_lr0.01_d12",
        0.02: f"{BASE}/verify_muon_d12",  # existing run
        0.04: f"{BASE}/verify_muon_lr0.04_d12",
    },
    "SVD eta=0.5": {
        0.01: f"{BASE}/verify_svd_eta0.5_lr0.01_d12",
        0.02: f"{BASE}/verify_svd_eta0.5_lr0.02_d12",
        0.04: f"{BASE}/verify_svd_eta0.5_lr0.04_d12",
    },
    "Taylor eta=0.1": {
        0.01: f"{BASE}/verify_taylor_eta0.1_lr0.01_d12",
        0.02: f"{BASE}/verify_taylor_eta0.1_lr0.02_d12",
        0.04: f"{BASE}/verify_taylor_eta0.1_lr0.04_d12",
    },
}

colors = {"Muon": "#2563eb", "SVD eta=0.5": "#ff69b4", "Taylor eta=0.1": "#2ca02c"}
markers = {"Muon": "o", "SVD eta=0.5": "D", "Taylor eta=0.1": "s"}


def load_val_log(path):
    """Load val_log.jsonl and return (steps, bpbs, times_min)."""
    steps, bpbs, times = [], [], []
    vpath = os.path.join(path, "val_log.jsonl")
    if not os.path.exists(vpath):
        return np.array([]), np.array([]), np.array([])
    with open(vpath) as f:
        for line in f:
            rec = json.loads(line)
            if "step" not in rec:
                continue
            steps.append(rec["step"])
            bpbs.append(rec["val_bpb"])
            times.append(rec.get("total_training_time", 0) / 60.0)  # to minutes
    return np.array(steps), np.array(bpbs), np.array(times)


def load_train_log(path, smooth_window=50):
    """Load train_log.jsonl and return (steps, smoothed_loss, times_min)."""
    steps, losses, dts = [], [], []
    tpath = os.path.join(path, "train_log.jsonl")
    if not os.path.exists(tpath):
        return np.array([]), np.array([]), np.array([])
    with open(tpath) as f:
        for line in f:
            rec = json.loads(line)
            if "step" not in rec or "loss" not in rec:
                continue
            steps.append(rec["step"])
            losses.append(rec["loss"])
            dts.append(rec.get("dt", 0))
    steps, losses, dts = np.array(steps), np.array(losses), np.array(dts)
    times = np.cumsum(dts) / 60.0  # cumulative time in minutes
    if len(losses) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        losses = np.convolve(losses, kernel, mode="valid")
        steps = steps[smooth_window - 1:]
        times = times[smooth_window - 1:]
    return steps, losses, times


fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 5)
ax5 = fig.add_subplot(2, 3, 6)

# Left panel: LR vs final val_bpb for each config
print("=== LR Sweep Results (Hypothesis B) ===\n")
best_per_config = {}

for name, lr_runs in configs.items():
    lrs_found, bpbs_found = [], []
    for lr, run_dir in sorted(lr_runs.items()):
        steps, bpbs, _ = load_val_log(run_dir)
        if len(bpbs) == 0:
            print(f"  {name} lr={lr}: not found ({run_dir})")
            continue
        if steps[-1] < 2000:
            print(f"  {name} lr={lr}: incomplete (last step {int(steps[-1])}, skipping)")
            continue
        final_bpb = bpbs[-1]
        lrs_found.append(lr)
        bpbs_found.append(final_bpb)
        print(f"  {name} lr={lr}: final val_bpb = {final_bpb:.6f} (step {int(steps[-1])})")

    if lrs_found:
        ax1.plot(lrs_found, bpbs_found, marker=markers[name], color=colors[name],
                 label=name, linewidth=2, markersize=8)
        best_idx = np.argmin(bpbs_found)
        best_per_config[name] = (lrs_found[best_idx], bpbs_found[best_idx])
        ax1.scatter([lrs_found[best_idx]], [bpbs_found[best_idx]],
                    s=200, facecolors="none", edgecolors=colors[name], linewidths=2.5, zorder=10)

ax1.set_xscale("log")
ax1.set_xlabel("Matrix LR", fontsize=12)
ax1.set_ylabel("Final Val BPB", fontsize=12)
ax1.set_title("LR Sweep — Best-of-Sweep Comparison", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Print summary
print("\n=== Best-of-Sweep Summary ===")
for name, (lr, bpb) in sorted(best_per_config.items()):
    print(f"  {name}: best LR={lr}, val_bpb={bpb:.6f}")
if "Muon" in best_per_config:
    muon_best = best_per_config["Muon"][1]
    for name, (lr, bpb) in best_per_config.items():
        if name != "Muon":
            delta = bpb - muon_best
            print(f"  {name} vs Muon: {'+' if delta > 0 else ''}{delta:.6f} BPB")

# Right panel: training curves at best LR for each config
for name, lr_runs in configs.items():
    if name not in best_per_config:
        continue
    best_lr = best_per_config[name][0]
    run_dir = lr_runs[best_lr]
    vpath = os.path.join(run_dir, "val_log.jsonl")
    if not os.path.exists(vpath):
        continue
    steps, bpbs, times = load_val_log(run_dir)
    ax2.plot(steps, bpbs, color=colors[name], label=f"{name} (lr={best_lr})", linewidth=2)
    ax4.plot(times, bpbs, color=colors[name], label=f"{name} (lr={best_lr})", linewidth=2)

ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("Val BPB", fontsize=12)
ax2.set_title("Val Curves at Best LR per Config", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
# Zoom into the converged region (skip initial high-loss steps)
ax2.set_xlim(left=200)
ax2.set_ylim(top=1.05)

# Third panel: training loss curves at best LR for each config
for name, lr_runs in configs.items():
    if name not in best_per_config:
        continue
    best_lr = best_per_config[name][0]
    run_dir = lr_runs[best_lr]
    steps, losses, times = load_train_log(run_dir)
    if len(steps) == 0:
        continue
    ax3.plot(steps, losses, color=colors[name], label=f"{name} (lr={best_lr})", linewidth=1.5, alpha=0.9)
    ax5.plot(times, losses, color=colors[name], label=f"{name} (lr={best_lr})", linewidth=1.5, alpha=0.9)

ax3.set_xlabel("Step", fontsize=12)
ax3.set_ylabel("Train Loss (smoothed)", fontsize=12)
ax3.set_title("Train Loss at Best LR per Config", fontsize=13)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(left=200)

# Bottom-left: val BPB vs time
ax4.set_xlabel("Time (min)", fontsize=12)
ax4.set_ylabel("Val BPB", fontsize=12)
ax4.set_title("Val Curves vs Wall Time", fontsize=13)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(left=2)
ax4.set_ylim(top=1.05)

# Bottom-right: train loss vs time
ax5.set_xlabel("Time (min)", fontsize=12)
ax5.set_ylabel("Train Loss (smoothed)", fontsize=12)
ax5.set_title("Train Loss vs Wall Time", fontsize=13)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(left=2)

fig.tight_layout()
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT}")
