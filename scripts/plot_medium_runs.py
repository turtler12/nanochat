"""
Plot depth-20 runs: loss vs step, loss vs time, d(loss)/dt vs time.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

LOGS = {
    "muon (d20)": (
        "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_muon_d20_train_log.jsonl",
        "#1f77b4",
    ),
    "ns3+ftrl exp η=0.3→0 relative (d20)": (
        "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_ns3_ftrl_exp_eta0p3_d20_train_log.jsonl",
        "#d62728",
    ),
    "ns3+ftrl exp η=0.3→0 absolute (d20)": (
        "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_ns3_ftrl_exp_abs_eta0p3_d20_train_log.jsonl",
        "#9467bd",
    ),
}
SMOOTH = 30

def load_log(path):
    steps, losses, times = [], [], []
    cumtime = 0.0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "_config" in d:
                continue
            steps.append(d["step"])
            losses.append(d["loss"])
            cumtime += d["dt"]
            times.append(cumtime / 60)
    return np.array(steps), np.array(losses), np.array(times)

def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode='valid')

import math
NUM_ITERATIONS = 4777
ETA0 = 0.3
_D12_REF_STEPS = 2205
_ETA_LAMBDA = math.log(3.0) / 100.0  # η(100) = 0.1

fig, axes = plt.subplots(2, 3, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1.2]})

all_data = {}
for name, (path, color) in LOGS.items():
    steps, losses, times = load_log(path)
    all_data[name] = (steps, losses, times, color)

    axes[0, 0].plot(steps, losses, label=name, color=color, linewidth=1.2)
    axes[0, 1].plot(times, losses, label=name, color=color, linewidth=1.2)

    if len(losses) > SMOOTH + 1:
        dloss = np.diff(losses)
        dt    = np.diff(times)
        deriv = dloss / dt
        deriv_s = smooth(deriv, SMOOTH)
        times_s = times[SMOOTH // 2 : SMOOTH // 2 + len(deriv_s)]
        axes[0, 2].plot(times_s, deriv_s, label=name, color=color, linewidth=1.2)

    print(f"{name}: {len(steps)} steps, {times[-1]:.1f}m elapsed, last loss {losses[-1]:.4f}")

axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("Train Loss"); axes[0, 0].set_title("Loss vs Step")
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].set_xlabel("Time (minutes)"); axes[0, 1].set_ylabel("Train Loss"); axes[0, 1].set_title("Loss vs Time")
axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].axhline(0, color="black", linewidth=0.5)
axes[0, 2].set_xlabel("Time (minutes)"); axes[0, 2].set_ylabel("d(loss)/dt (loss/min)")
axes[0, 2].set_title("Rate of Loss Decrease vs Time")
axes[0, 2].legend(fontsize=8); axes[0, 2].grid(True, alpha=0.3)

# ── Row 1: eta schedule (only steps actually run) ─────────────────────────────
rel_steps, _, rel_times, rel_color = all_data["ns3+ftrl exp η=0.3→0 relative (d20)"]
abs_steps, _, abs_times, abs_color = all_data["ns3+ftrl exp η=0.3→0 absolute (d20)"]

eta_rel = ETA0 * np.exp(-5.0 * rel_steps / NUM_ITERATIONS)         # relative (current)
eta_abs = ETA0 * np.exp(-_ETA_LAMBDA * abs_steps)                  # fast absolute: η(100)=0.1

# d12 reference — old absolute schedule
d12_steps_run = np.arange(min(_D12_REF_STEPS, int(rel_steps[-1]) + 1))
eta_d12 = ETA0 * np.exp(-5.0 * d12_steps_run / _D12_REF_STEPS)
d12_times_run = d12_steps_run * (rel_times[-1] / max(rel_steps[-1], 1))

axes[1, 0].plot(rel_steps, eta_rel, color=rel_color,   linewidth=1.5, label="exp η relative (d20)")
axes[1, 0].plot(abs_steps, eta_abs, color=abs_color,   linewidth=1.5, label="exp η absolute (d20)")
axes[1, 0].plot(d12_steps_run, eta_d12, color="#ff7f0e", linewidth=1.5, linestyle="--", label="exp η (d12 reference)")
axes[1, 0].axhline(0.1, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
axes[1, 0].text(10, 0.105, "η=0.1", fontsize=7, color="gray")
axes[1, 0].set_xlabel("Step"); axes[1, 0].set_ylabel("η (FTRL eta)")
axes[1, 0].set_title("η Schedule vs Step (actual steps run)")
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(rel_times, eta_rel, color=rel_color,  linewidth=1.5, label="exp η relative (d20)")
axes[1, 1].plot(abs_times, eta_abs, color=abs_color,  linewidth=1.5, label="exp η absolute (d20)")
axes[1, 1].plot(d12_times_run, eta_d12, color="#ff7f0e", linewidth=1.5, linestyle="--", label="exp η (d12 reference)")
axes[1, 1].axhline(0.1, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
axes[1, 1].set_xlabel("Time (minutes)"); axes[1, 1].set_ylabel("η (FTRL eta)")
axes[1, 1].set_title("η Schedule vs Time (actual steps run)")
axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].axis("off")

plt.suptitle("Depth-20 runs (2×H100, 4777 steps total) — partial results", fontsize=12)
plt.tight_layout()
plt.savefig("/Users/medha/Desktop/muon_local/nanochat/plots/cluster_muon_vs_ns3ftrl_medium.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to plots/cluster_muon_vs_ns3ftrl_medium.png")
