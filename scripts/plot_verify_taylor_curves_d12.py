"""
Plot Taylor eta sweep curves for small model (d12): loss vs step, loss vs time.
Includes baseline Muon. Caption with final loss values.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "cache/base_checkpoints"
OUTPUT = "verifying_results/small_model/taylor_sweep_curves_d12.png"
SMOOTH_WINDOW = 50

runs = {
    "Muon baseline": {"dir": f"{BASE}/verify_muon_d12", "color": "#2563eb", "ls": "-", "lw": 2.5},
    "Taylor eta=0.01": {"dir": f"{BASE}/verify_taylor_eta0.01_d12", "color": "#1f77b4", "ls": "--", "lw": 1.5},
    "Taylor eta=0.1":  {"dir": f"{BASE}/verify_taylor_eta0.1_d12",  "color": "#e6550d", "ls": "--", "lw": 1.5},
    "Taylor eta=0.3":  {"dir": f"{BASE}/verify_taylor_eta0.3_d12",  "color": "#17becf", "ls": "--", "lw": 1.5},
    "Taylor eta=0.5":  {"dir": f"{BASE}/verify_taylor0.5_d12",      "color": "#2ca02c", "ls": "--", "lw": 1.5},
    "Taylor eta=1.0":  {"dir": f"{BASE}/verify_taylor_eta1.0_d12",  "color": "#d62728", "ls": "--", "lw": 1.5},
    "Taylor eta=2.0":  {"dir": f"{BASE}/verify_taylor_eta2.0_d12",  "color": "#9467bd", "ls": "--", "lw": 1.5},
    "Taylor eta=5.0":  {"dir": f"{BASE}/verify_taylor_eta5.0_d12",  "color": "#8c564b", "ls": "--", "lw": 1.5},
    "Full SVD eta=0.01": {"dir": f"{BASE}/verify_svd_softmax_eta0.01_d12", "color": "#ff69b4", "ls": "-.", "lw": 2.0},
    "Full SVD eta=0.1":  {"dir": f"{BASE}/verify_svd_softmax_eta0.1_d12",  "color": "#e377c2", "ls": "-.", "lw": 2.0},
    "Full SVD eta=0.5":  {"dir": f"{BASE}/verify_svd_softmax_eta0.5_d12",  "color": "#c44e7b", "ls": "-.", "lw": 2.0},
    "Full SVD eta=2.0":  {"dir": f"{BASE}/verify_svd_softmax_eta2.0_d12",  "color": "#7b2c8c", "ls": "-.", "lw": 2.0},
    "Full SVD eta=5.0":  {"dir": f"{BASE}/verify_svd_softmax_eta5.0_d12",  "color": "#5a1a6b", "ls": "-.", "lw": 2.0},
    "Full SVD eta=10.0": {"dir": f"{BASE}/verify_svd_softmax_eta10.0_d12", "color": "#3d0f4a", "ls": "-.", "lw": 2.0},
}

def load_train_log(path):
    steps, losses, dt = [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "loss" in rec and rec.get("step", -1) >= 0:
                steps.append(rec["step"])
                losses.append(rec["loss"])
                dt.append(rec.get("dt", 0))
    return np.array(steps), np.array(losses), np.array(dt)

def load_val_log(path):
    steps, bpb = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "val_bpb" in rec:
                steps.append(rec["step"])
                bpb.append(rec["val_bpb"])
    return np.array(steps), np.array(bpb)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

caption_lines = []
val_caption_lines = []

for name, cfg in runs.items():
    train_path = os.path.join(cfg["dir"], "train_log.jsonl")
    val_path = os.path.join(cfg["dir"], "val_log.jsonl")
    if not os.path.exists(train_path):
        print(f"  {name}: not found yet")
        continue
    steps, losses, dt = load_train_log(train_path)
    if len(losses) < SMOOTH_WINDOW + 1:
        print(f"  {name}: only {len(steps)} steps, skipping")
        continue

    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    smoothed = np.convolve(losses, kernel, mode="valid")
    s_steps = steps[SMOOTH_WINDOW - 1:]
    times_min = np.cumsum(dt) / 60.0
    s_times = times_min[SMOOTH_WINDOW - 1:]

    final_loss = smoothed[-1]
    final_step = int(s_steps[-1])
    caption_lines.append((name, final_loss, final_step))

    is_bl = "baseline" in name
    alpha = 0.7 if is_bl else 1.0
    zorder = 1 if is_bl else 2

    ax1.plot(s_steps, smoothed, label=name, color=cfg["color"], ls=cfg["ls"], linewidth=cfg["lw"],
             alpha=alpha, zorder=zorder)
    ax2.plot(s_times, smoothed, label=name, color=cfg["color"], ls=cfg["ls"], linewidth=cfg["lw"],
             alpha=alpha, zorder=zorder)

    # Validation BPB
    if os.path.exists(val_path):
        vsteps, vbpb = load_val_log(val_path)
        if len(vsteps) > 0:
            ax3.plot(vsteps, vbpb, label=name, color=cfg["color"], ls=cfg["ls"], linewidth=cfg["lw"],
                     alpha=alpha, zorder=zorder)
            val_caption_lines.append((name, vbpb[-1], int(vsteps[-1])))

    print(f"  {name} @ step {final_step}: train={final_loss:.4f}" +
          (f"  val={vbpb[-1]:.4f}" if os.path.exists(val_path) and len(vsteps) > 0 else ""))

# Build captions sorted by loss
caption_lines.sort(key=lambda x: x[1])
caption_text = "Final train loss:\n" + "\n".join(f"{name}: {loss:.4f} (step {step})" for name, loss, step in caption_lines)

val_caption_lines.sort(key=lambda x: x[1])
val_caption_text = "Final val BPB:\n" + "\n".join(f"{name}: {bpb:.4f} (step {step})" for name, bpb, step in val_caption_lines)

for ax in (ax1, ax2):
    ax.text(0.98, 0.98, caption_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"))

ax3.text(0.98, 0.98, val_caption_text, transform=ax3.transAxes, fontsize=8,
         verticalalignment="top", horizontalalignment="right",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"))

ax1.set_xlabel("Training Step", fontsize=12)
ax1.set_ylabel("Training Loss (smoothed)", fontsize=12)
ax1.set_title("Small Model (d12) — Train Loss vs Step", fontsize=13)
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Wall-Clock Time (minutes)", fontsize=12)
ax2.set_ylabel("Training Loss (smoothed)", fontsize=12)
ax2.set_title("Small Model (d12) — Train Loss vs Time", fontsize=13)
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(True, alpha=0.3)

ax3.set_xlabel("Training Step", fontsize=12)
ax3.set_ylabel("Validation BPB", fontsize=12)
ax3.set_title("Small Model (d12) — Validation BPB", fontsize=13)
ax3.legend(fontsize=9, loc="upper left")
ax3.grid(True, alpha=0.3)

fig.suptitle("Taylor Eta Sweep — Small Model (d12)", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
