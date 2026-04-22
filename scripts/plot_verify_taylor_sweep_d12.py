"""
Plot Taylor eta sweep for small model (d12): eta vs loss at step 2000.
Rerun after all jobs complete to get the full picture.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "cache/base_checkpoints"
OUTPUT = "verifying_results/small_model/eta_vs_loss_d12.png"

# All taylor eta runs + baseline muon
taylor_runs = {
    0.01: f"{BASE}/verify_taylor_eta0.01_d12",
    0.1:  f"{BASE}/verify_taylor_eta0.1_d12",
    0.3:  f"{BASE}/verify_taylor_eta0.3_d12",
    0.5:  f"{BASE}/verify_taylor0.5_d12",
    1.0:  f"{BASE}/verify_taylor_eta1.0_d12",
    2.0:  f"{BASE}/verify_taylor_eta2.0_d12",
    5.0:  f"{BASE}/verify_taylor_eta5.0_d12",
}
baseline_dir = f"{BASE}/verify_muon_d12"

TARGET_STEP = 2000
SMOOTH_WINDOW = 50

def load_train_log(path):
    steps, losses = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "loss" in rec and rec.get("step", -1) >= 0:
                steps.append(rec["step"])
                losses.append(rec["loss"])
    return np.array(steps), np.array(losses)

def smoothed_loss_at_step(steps, losses, target_step):
    if len(losses) < SMOOTH_WINDOW + 1:
        return None, None
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    smoothed = np.convolve(losses, kernel, mode="valid")
    s_steps = steps[SMOOTH_WINDOW - 1:]
    idx = np.searchsorted(s_steps, target_step, side="right") - 1
    idx = max(0, min(idx, len(smoothed) - 1))
    return float(smoothed[idx]), int(s_steps[idx])

# Collect data
eta_points = []  # (eta, loss, step)

# Baseline
bl_path = os.path.join(baseline_dir, "train_log.jsonl")
if os.path.exists(bl_path):
    steps, losses = load_train_log(bl_path)
    loss, step = smoothed_loss_at_step(steps, losses, TARGET_STEP)
    if loss is not None:
        baseline_loss = loss
        baseline_step = step
        print(f"Baseline Muon @ step {step}: {loss:.4f}")
else:
    baseline_loss = None

# Taylor runs
for eta, run_dir in sorted(taylor_runs.items()):
    train_path = os.path.join(run_dir, "train_log.jsonl")
    if not os.path.exists(train_path):
        print(f"  eta={eta}: not found yet")
        continue
    steps, losses = load_train_log(train_path)
    if steps[-1] < TARGET_STEP * 0.9:
        print(f"  eta={eta}: only {int(steps[-1])} steps, still running")
        continue
    loss, step = smoothed_loss_at_step(steps, losses, TARGET_STEP)
    if loss is not None:
        eta_points.append((eta, loss, step))
        print(f"  eta={eta} @ step {step}: {loss:.4f}")
    else:
        print(f"  eta={eta}: not enough data yet ({len(steps)} steps)")

if not eta_points:
    print("No data available yet.")
    exit(0)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

etas = [e for e, l, s in eta_points]
losses = [l for e, l, s in eta_points]

ax.scatter(etas, losses, s=80, color="#2ca02c", zorder=5, label="Taylor")
for e, l, s in eta_points:
    ax.annotate(f"({e}, {l:.4f})", (e, l),
                textcoords="offset points", xytext=(8, 8), fontsize=9)

if baseline_loss is not None:
    ax.axhline(y=baseline_loss, color="#2563eb", linewidth=1.5, linestyle="--", zorder=1, label=f"Muon baseline ({baseline_loss:.4f})")

# Full SVD softmax runs
svd_label_added = False
for svd_eta_str, svd_eta_val in [("0.01", 0.01), ("0.1", 0.1), ("0.5", 0.5), ("2.0", 2.0), ("5.0", 5.0), ("10.0", 10.0)]:
    svd_path = os.path.join(BASE, f"verify_svd_softmax_eta{svd_eta_str}_d12", "train_log.jsonl")
    if not os.path.exists(svd_path):
        print(f"  Full SVD eta={svd_eta_str}: not found yet")
        continue
    steps, losses = load_train_log(svd_path)
    if len(steps) == 0 or steps[-1] < TARGET_STEP * 0.9:
        print(f"  Full SVD eta={svd_eta_str}: only {int(steps[-1]) if len(steps) > 0 else 0} steps, still running")
        continue
    svd_loss, svd_step = smoothed_loss_at_step(steps, losses, TARGET_STEP)
    if svd_loss is not None:
        label = "Full SVD" if not svd_label_added else None
        ax.scatter([svd_eta_val], [svd_loss], s=120, color="#ff69b4", marker="D", zorder=6, label=label)
        ax.annotate(f"SVD ({svd_loss:.4f})", (svd_eta_val, svd_loss),
                    textcoords="offset points", xytext=(8, -12), fontsize=9, color="#ff69b4")
        svd_label_added = True
        print(f"  Full SVD eta={svd_eta_str} @ step {svd_step}: {svd_loss:.4f}")

ax.set_xscale("log")
ax.set_xlabel("eta", fontsize=12)
ax.set_ylabel(f"Smoothed Training Loss @ step ~{TARGET_STEP}", fontsize=12)
ax.set_title(f"Taylor Eta Sweep — Small Model (d12)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT}")
