"""
Plot val BPB: muon vs ns3_ftrl_exp_abs_then_muon (d12, 2205 steps).
Uses val_log.jsonl from both runs (eval every 250 steps).
"""
import json
import numpy as np
import matplotlib.pyplot as plt

LOGS = {
    "muon": (
        "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_muon_d12/val_log.jsonl",
        "#1f77b4",
    ),
    "ns3+ftrl exp abs → muon (switch @200)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log.jsonl",
        "#e377c2",
    ),
}

def load_val_log(path):
    steps, bpbs, times = [], [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "_config" in d:
                continue
            steps.append(d["step"])
            bpbs.append(d["val_bpb"])
            times.append(d["total_training_time"] / 60)
    return np.array(steps), np.array(bpbs), np.array(times)

data = {name: load_val_log(path) for name, (path, _) in LOGS.items()}
colors = {name: color for name, (_, color) in LOGS.items()}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for name, (steps, bpbs, times) in data.items():
    axes[0].plot(steps, bpbs, label=name, color=colors[name], linewidth=1.8, marker='o', markersize=3)
    axes[1].plot(times, bpbs, label=name, color=colors[name], linewidth=1.8, marker='o', markersize=3)
    print(f"{name}: final val bpb = {bpbs[-1]:.4f} at step {steps[-1]}")

# Mark the switch point at step 200
for ax_idx, x_key in [(0, "step"), (1, "time")]:
    ax = axes[ax_idx]
    if x_key == "step":
        ax.axvline(200, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(210, ax.get_ylim()[1] * 0.98 if ax.get_ylim()[1] > 1 else 1.05,
                "switch\n@step 200", fontsize=7, color="gray", va="top")
    ax.set_ylabel("Val BPB")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[0].set_xlabel("Step")
axes[0].set_title("Val BPB vs Step")
axes[1].set_xlabel("Time (minutes)")
axes[1].set_title("Val BPB vs Time")

# Add switch marker on time axis
switch_time_hybrid = data["ns3+ftrl exp abs → muon (switch @200)"][2]
switch_step_idx = np.searchsorted(data["ns3+ftrl exp abs → muon (switch @200)"][0], 200)
if switch_step_idx < len(switch_time_hybrid):
    switch_t = switch_time_hybrid[switch_step_idx]
    axes[1].axvline(switch_t, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].text(switch_t + 0.3, axes[1].get_ylim()[1] if axes[1].get_ylim()[1] > 1 else 1.05,
                 f"switch\n@{switch_t:.1f}m", fontsize=7, color="gray", va="top")

plt.suptitle("Muon vs FTRL→Muon hybrid (d12, 2×H100, 2205 steps)", fontsize=12)
plt.tight_layout()
out = "/Users/medha/Desktop/muon_local/nanochat/plots/ftrl_then_muon_d12.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to {out}")
