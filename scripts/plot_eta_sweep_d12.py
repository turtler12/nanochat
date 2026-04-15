"""
Plot eta sweep results for Taylor approximation (d12, lr=0.02 fixed).
Tests whether negative eta improves over Muon (eta=0).

Expected outcomes:
  - Symmetric V-shape around eta=0: Muon is locally optimal, project done.
  - Monotonic decrease into negative eta: real win, sweep further.
  - Asymmetric (small neg helps, large neg hurts): optimum in negative-eta space.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "cache/base_checkpoints"
OUTPUT = "verifying_results/small_model/eta_sweep_d12.png"

# All eta values and their run directories
# eta=0 is pure Muon (no Taylor correction)
eta_runs = {
    -0.5:  f"{BASE}/eta_sweep_neg0.5_lr0.02_d12",
    -0.3:  f"{BASE}/eta_sweep_neg0.3_lr0.02_d12",
    -0.1:  f"{BASE}/eta_sweep_neg0.1_lr0.02_d12",
    -0.05: f"{BASE}/eta_sweep_neg0.05_lr0.02_d12",
    0.0:   f"{BASE}/verify_muon_d12",  # Muon baseline (eta=0)
    0.05:  f"{BASE}/eta_sweep_pos0.05_lr0.02_d12",
    0.1:   f"{BASE}/eta_sweep_pos0.1_lr0.02_d12",  # new run; fallback to existing
}

# Fallback for eta=0.1 if new run not found
ETA_0_1_FALLBACK = f"{BASE}/verify_taylor_eta0.1_lr0.02_d12"


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
            times.append(rec.get("total_training_time", 0) / 60.0)
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
    times = np.cumsum(dts) / 60.0
    if len(losses) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        losses = np.convolve(losses, kernel, mode="valid")
        steps = steps[smooth_window - 1:]
        times = times[smooth_window - 1:]
    return steps, losses, times


# Collect results
print("=== Eta Sweep Results (d12, lr=0.02) ===\n")
etas_found, bpbs_found = [], []

for eta, run_dir in sorted(eta_runs.items()):
    # Fallback for eta=0.1
    if eta == 0.1 and not os.path.exists(os.path.join(run_dir, "val_log.jsonl")):
        run_dir = ETA_0_1_FALLBACK

    steps, bpbs, _ = load_val_log(run_dir)
    if len(bpbs) == 0:
        print(f"  eta={eta:+.2f}: NOT FOUND ({run_dir})")
        continue
    if steps[-1] < 2000:
        print(f"  eta={eta:+.2f}: incomplete (last step {int(steps[-1])}, skipping)")
        continue
    final_bpb = bpbs[-1]
    etas_found.append(eta)
    bpbs_found.append(final_bpb)
    print(f"  eta={eta:+.2f}: final val_bpb = {final_bpb:.6f} (step {int(steps[-1])})")

etas_found = np.array(etas_found)
bpbs_found = np.array(bpbs_found)

if len(etas_found) == 0:
    print("\nNo completed runs found. Exiting.")
    exit(0)

# Interpret the result
muon_bpb = None
if 0.0 in etas_found:
    muon_idx = np.where(etas_found == 0.0)[0][0]
    muon_bpb = bpbs_found[muon_idx]

print(f"\n=== Summary ===")
best_idx = np.argmin(bpbs_found)
print(f"  Best eta: {etas_found[best_idx]:+.2f} with val_bpb = {bpbs_found[best_idx]:.6f}")
if muon_bpb is not None:
    print(f"  Muon (eta=0): val_bpb = {muon_bpb:.6f}")
    for i, (eta, bpb) in enumerate(zip(etas_found, bpbs_found)):
        if eta != 0.0:
            delta = bpb - muon_bpb
            print(f"  eta={eta:+.2f} vs Muon: {delta:+.6f} BPB")

# Diagnose the shape
neg_etas = [(e, b) for e, b in zip(etas_found, bpbs_found) if e < 0]
pos_etas = [(e, b) for e, b in zip(etas_found, bpbs_found) if e > 0]
if muon_bpb is not None and len(neg_etas) >= 2 and len(pos_etas) >= 1:
    neg_better = any(b < muon_bpb - 0.001 for _, b in neg_etas)
    pos_better = any(b < muon_bpb - 0.001 for _, b in pos_etas)
    neg_monotonic = all(neg_etas[i][1] <= neg_etas[i+1][1] for i in range(len(neg_etas)-1))

    if not neg_better and not pos_better:
        print("\n  --> SYMMETRIC V-SHAPE: Muon (eta=0) appears locally optimal.")
        print("      Hypothesis A confirmed. Consider writing up negative result.")
    elif neg_better and neg_monotonic:
        print("\n  --> MONOTONIC DECREASE into negative eta!")
        print("      Sweep further into negative territory to find the optimum.")
    elif neg_better and not neg_monotonic:
        print("\n  --> ASYMMETRIC: small negative eta helps, large negative hurts.")
        print("      There's an optimum in negative-eta space. Find it and verify on large model.")
    else:
        print("\n  --> Unexpected pattern. Inspect the plot carefully.")


# === PLOTTING ===
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: Eta vs final val BPB (the key result)
ax1 = axes[0]
ax1.plot(etas_found, bpbs_found, "o-", color="#2563eb", linewidth=2, markersize=10, zorder=5)
ax1.scatter([etas_found[best_idx]], [bpbs_found[best_idx]],
            s=200, facecolors="none", edgecolors="red", linewidths=2.5, zorder=10,
            label=f"Best: eta={etas_found[best_idx]:+.2f}")
if muon_bpb is not None:
    ax1.axhline(muon_bpb, color="gray", linestyle="--", alpha=0.7, label=f"Muon (eta=0): {muon_bpb:.4f}")
ax1.axvline(0, color="gray", linestyle=":", alpha=0.5)
ax1.set_xlabel("eta (Taylor correction strength)", fontsize=13)
ax1.set_ylabel("Final Val BPB", fontsize=13)
ax1.set_title("Eta Sweep: Final Val BPB vs Eta", fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Val BPB curves over steps for all runs
ax2 = axes[1]
cmap = plt.cm.coolwarm
norm = plt.Normalize(vmin=min(etas_found), vmax=max(etas_found))

for eta, run_dir in sorted(eta_runs.items()):
    if eta == 0.1 and not os.path.exists(os.path.join(run_dir, "val_log.jsonl")):
        run_dir = ETA_0_1_FALLBACK
    steps, bpbs, _ = load_val_log(run_dir)
    if len(bpbs) == 0 or steps[-1] < 2000:
        continue
    color = "black" if eta == 0.0 else cmap(norm(eta))
    lw = 2.5 if eta == 0.0 else 1.5
    ax2.plot(steps, bpbs, color=color, linewidth=lw,
             label=f"eta={eta:+.2f}", alpha=0.9)

ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("Val BPB", fontsize=12)
ax2.set_title("Val BPB Curves (all etas)", fontsize=13)
ax2.legend(fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(left=200)
ax2.set_ylim(top=1.05)

# Panel 3: Train loss curves for all runs
ax3 = axes[2]
for eta, run_dir in sorted(eta_runs.items()):
    if eta == 0.1 and not os.path.exists(os.path.join(run_dir, "val_log.jsonl")):
        run_dir = ETA_0_1_FALLBACK
    steps, losses, _ = load_train_log(run_dir)
    if len(steps) == 0:
        continue
    color = "black" if eta == 0.0 else cmap(norm(eta))
    lw = 2.5 if eta == 0.0 else 1.5
    ax3.plot(steps, losses, color=color, linewidth=lw,
             label=f"eta={eta:+.2f}", alpha=0.9)

ax3.set_xlabel("Step", fontsize=12)
ax3.set_ylabel("Train Loss (smoothed)", fontsize=12)
ax3.set_title("Train Loss Curves (all etas)", fontsize=13)
ax3.legend(fontsize=8, ncol=2)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(left=200)

fig.tight_layout()
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"\nSaved: {OUTPUT}")
