"""
Plot train loss vs step, loss vs time, and d(loss)/dt vs time for muon vs ns3+ftrl.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOGS = {
    "muon":              "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_muon_d12_train_log.jsonl",
    "ns3+ftrl η=0.1":   "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_ns3_ftrl_eta0p1_d12_train_log.jsonl",
    "ns3+ftrl linear η=0.3→0": "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_ns3_ftrl_linear_eta0p3_d12_train_log.jsonl",
    "ns3+ftrl exp η=0.3→0":    "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_ns3_ftrl_exp_eta0p3_d12_train_log.jsonl",
}
COLORS = {
    "muon":              "#1f77b4",
    "ns3+ftrl η=0.1":   "#ff7f0e",
    "ns3+ftrl linear η=0.3→0": "#2ca02c",
    "ns3+ftrl exp η=0.3→0":    "#d62728",
}
SMOOTH = 30  # window for derivative smoothing

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
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='valid')

data = {name: load_log(path) for name, path in LOGS.items()}

# Find crossover: first step >50 where muon <= fixed-eta ns3+ftrl
muon_loss = data["muon"][1]
ns3_loss  = data["ns3+ftrl η=0.1"][1]
muon_time = data["muon"][2]
cross_indices = np.where((muon_loss - ns3_loss) <= 0)[0]
cross_indices = cross_indices[cross_indices > 50]
crossover_time = muon_time[cross_indices[0]] if len(cross_indices) > 0 else None
crossover_step = int(data["muon"][0][cross_indices[0]]) if len(cross_indices) > 0 else None

# Compute eta schedules over time for the two decay modes
# Use muon's time axis as reference (all runs have same step count/timing)
ref_steps = data["muon"][0]
ref_times  = data["muon"][2]
T = len(ref_steps)
eta_linear = 0.3 * (1.0 - ref_steps / max(ref_steps[-1], 1))
eta_exp    = 0.3 * np.exp(-5.0 * ref_steps / max(ref_steps[-1], 1))
eta_fixed  = np.full_like(ref_steps, 0.1, dtype=float)

fig, axes = plt.subplots(2, 3, figsize=(16, 8),
                         gridspec_kw={"height_ratios": [3, 1.2]})

# ── Row 0: loss plots ──────────────────────────────────────────────────────────

# --- Panel [0,0]: Loss vs Step ---
ax = axes[0, 0]
for name, (steps, losses, times) in data.items():
    ax.plot(steps, losses, label=name, color=COLORS[name], linewidth=1.2)
if crossover_step is not None:
    ax.axvline(crossover_step, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(crossover_step + 20, 3.8, f"step {crossover_step}\nmuon catches up", fontsize=7, color="gray")
ax.set_xlabel("Step")
ax.set_ylabel("Train Loss")
ax.set_title("Loss vs Step")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel [0,1]: Loss vs Time ---
ax = axes[0, 1]
for name, (steps, losses, times) in data.items():
    ax.plot(times, losses, label=name, color=COLORS[name], linewidth=1.2)
if crossover_time is not None:
    ax.axvline(crossover_time, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(crossover_time + 0.3, 3.8, f"{crossover_time:.1f}m\nmuon catches up", fontsize=7, color="gray")
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Train Loss")
ax.set_title("Loss vs Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel [0,2]: d(loss)/dt vs Time ---
ax = axes[0, 2]
for name, (steps, losses, times) in data.items():
    dloss = np.diff(losses)
    dt    = np.diff(times)
    deriv = dloss / dt
    deriv_s = smooth(deriv, SMOOTH)
    times_s  = times[SMOOTH // 2 : SMOOTH // 2 + len(deriv_s)]
    ax.plot(times_s, deriv_s, label=name, color=COLORS[name], linewidth=1.2)
if crossover_time is not None:
    ax.axvline(crossover_time, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("d(loss)/dt  (loss/min)")
ax.set_title("Rate of Loss Decrease vs Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Row 1: eta schedules ───────────────────────────────────────────────────────

# --- Panel [1,0]: η vs Step ---
ax = axes[1, 0]
ax.plot(ref_steps, eta_fixed,  label="ns3+ftrl η=0.1",            color=COLORS["ns3+ftrl η=0.1"],            linewidth=1.5)
ax.plot(ref_steps, eta_linear, label="ns3+ftrl linear η=0.3→0",   color=COLORS["ns3+ftrl linear η=0.3→0"],   linewidth=1.5)
ax.plot(ref_steps, eta_exp,    label="ns3+ftrl exp η=0.3→0",       color=COLORS["ns3+ftrl exp η=0.3→0"],      linewidth=1.5)
if crossover_step is not None:
    ax.axvline(crossover_step, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Step")
ax.set_ylabel("η (FTRL eta)")
ax.set_title("η Schedule vs Step")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel [1,1]: η vs Time ---
ax = axes[1, 1]
ax.plot(ref_times, eta_fixed,  label="ns3+ftrl η=0.1",            color=COLORS["ns3+ftrl η=0.1"],            linewidth=1.5)
ax.plot(ref_times, eta_linear, label="ns3+ftrl linear η=0.3→0",   color=COLORS["ns3+ftrl linear η=0.3→0"],   linewidth=1.5)
ax.plot(ref_times, eta_exp,    label="ns3+ftrl exp η=0.3→0",       color=COLORS["ns3+ftrl exp η=0.3→0"],      linewidth=1.5)
if crossover_time is not None:
    ax.axvline(crossover_time, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(crossover_time + 0.3, 0.22, f"{crossover_time:.1f}m", fontsize=7, color="gray")
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("η (FTRL eta)")
ax.set_title("η Schedule vs Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel [1,2]: empty / hide ---
axes[1, 2].axis("off")

plt.suptitle("Muon vs NS3+FTRL variants (depth-12, 2×H100, 2205 steps)", fontsize=12)
plt.tight_layout()
plt.savefig("/Users/medha/Desktop/muon_local/nanochat/plots/cluster_muon_vs_ns3ftrl.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Crossover: step {crossover_step}, time {crossover_time:.1f}m")
print("Saved to plots/cluster_muon_vs_ns3ftrl.png")
