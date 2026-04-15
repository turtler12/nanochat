"""
Plot Taylor(eta=0.01) vs baseline Muon (both 2 GPU, d20) to verify they match.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "cache/base_checkpoints"
OUTPUT = "verifying_results/taylor_eta0.01_vs_muon_d20.png"

runs = {
    "Muon (baseline, 2 GPU)": {"dir": f"{BASE}/wallclock_baseline_d20", "color": "#2563eb", "ls": "-"},
    "Taylor (eta=0.01)": {"dir": f"{BASE}/taylor_eta0.01_full_d20", "color": "#dc2626", "ls": "--"},
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
    steps, loss = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "loss" in rec and rec.get("step", -1) >= 0:
                steps.append(rec["step"])
                loss.append(rec["loss"])
    return steps, loss

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for name, cfg in runs.items():
    val_path = os.path.join(cfg["dir"], "val_log.jsonl")
    train_path = os.path.join(cfg["dir"], "train_log.jsonl")
    if not os.path.exists(val_path):
        print(f"Missing: {val_path}")
        continue

    steps, bpb = load_val_log(val_path)
    ax1.plot(steps, bpb, label=name, color=cfg["color"], ls=cfg["ls"], linewidth=1.5)

    if os.path.exists(train_path):
        tsteps, tloss = load_train_log(train_path)
        stride = max(1, len(tsteps) // 500)
        ax2.plot(tsteps[::stride], tloss[::stride], label=name, color=cfg["color"], ls=cfg["ls"], linewidth=1.0, alpha=0.8)

ax1.set_xlabel("Step")
ax1.set_ylabel("Validation BPB")
ax1.set_title("d20: Taylor(eta=0.01) vs Muon Baseline")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Step")
ax2.set_ylabel("Training Loss")
ax2.set_title("d20: Taylor(eta=0.01) vs Muon Baseline — Train Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=150)
print(f"Saved: {OUTPUT}")
