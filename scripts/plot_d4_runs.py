"""
Plot d4 runs: train loss vs step, train loss vs time, d(loss)/dt vs time, val BPB vs step.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

TRAIN_LOGS = {
    "muon (d4, 2gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_muon_d4.jsonl",
        "#1f77b4",
    ),
    "muon (d4, 4gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_muon_d4_4gpu.jsonl",
        "#17becf",
    ),
    "ns3+ftrl exp abs →muon @100 (d4, 2gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s100_d4.jsonl",
        "#e377c2",
    ),
    "ns3+ftrl exp abs →muon @200 (d4, 2gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s200_d4.jsonl",
        "#2ca02c",
    ),
    "ns3+ftrl exp abs →muon @100 (d4, 4gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s100_d4_4gpu.jsonl",
        "#ff7f0e",
    ),
    "ns3 only (d4, 4gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ns3_d4_4gpu.jsonl",
        "#7f7f7f",
    ),
}
VAL_LOGS = {
    "muon (d4, 2gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_muon_d4.jsonl",
        "#1f77b4",
    ),
    "muon (d4, 4gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_muon_d4_4gpu.jsonl",
        "#17becf",
    ),
    "ns3+ftrl exp abs →muon @100 (d4, 2gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s100_d4.jsonl",
        "#e377c2",
    ),
    "ns3+ftrl exp abs →muon @200 (d4, 2gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s200_d4.jsonl",
        "#2ca02c",
    ),
    "ns3+ftrl exp abs →muon @100 (d4, 4gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s100_d4_4gpu.jsonl",
        "#ff7f0e",
    ),
    "ns3 only (d4, 4gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ns3_d4_4gpu.jsonl",
        "#7f7f7f",
    ),
}
SMOOTH = 30

def load_train_log(path):
    steps, losses, times = [], [], []
    cumtime = 0.0
    # First pass: compute median dt to detect compile-stall outliers
    dts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "_config" in d or d["step"] == 0:
                continue
            dts.append(d["dt"])
    median_dt = float(np.median(dts)) if dts else 1.0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "_config" in d:
                continue
            # skip compile-stall steps entirely (dt > 10x median)
            if d["step"] > 0 and d["dt"] >= 10 * median_dt:
                continue
            steps.append(d["step"])
            losses.append(d["loss"])
            if d["step"] > 0:
                cumtime += d["dt"]
            times.append(cumtime / 60)
    return np.array(steps), np.array(losses), np.array(times)

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

def smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode='valid')

SWITCH_STEPS = {
    "ns3+ftrl exp abs →muon @100 (d4, 2gpu)": 100,
    "ns3+ftrl exp abs →muon @200 (d4, 2gpu)": 200,
    "ns3+ftrl exp abs →muon @100 (d4, 4gpu)": 100,
}

train_data = {name: load_train_log(path) for name, (path, _) in TRAIN_LOGS.items()}
val_data   = {name: load_val_log(path)   for name, (path, _) in VAL_LOGS.items()}
colors     = {name: color for name, (_, color) in TRAIN_LOGS.items()}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# --- [0,0]: Train Loss vs Step ---
ax = axes[0, 0]
for name, (steps, losses, times) in train_data.items():
    ax.plot(steps, losses, color=colors[name], linewidth=1.2, label=f"{name} (final: {losses[-1]:.4f})")
for name, sw in SWITCH_STEPS.items():
    ax.axvline(sw, color=colors[name], linestyle="--", linewidth=0.9, alpha=0.5)
    ax.text(sw + 10, 9.5, f"@{sw}", fontsize=7, color=colors[name])
ax.set_xlabel("Step")
ax.set_ylabel("Train Loss")
ax.set_title("Train Loss vs Step")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- [0,1]: Train Loss vs Time ---
ax = axes[0, 1]
for name, (steps, losses, times) in train_data.items():
    ax.plot(times, losses, color=colors[name], linewidth=1.2, label=f"{name} (final: {losses[-1]:.4f})")
    if name in SWITCH_STEPS:
        switch_idx = np.searchsorted(steps, SWITCH_STEPS[name])
        if switch_idx < len(times):
            ax.axvline(times[switch_idx], color=colors[name], linestyle="--", linewidth=0.9, alpha=0.5)
            ax.text(times[switch_idx] + 0.02, 9.5,
                    f"@{times[switch_idx]:.2f}m", fontsize=7, color=colors[name])
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Train Loss")
ax.set_title("Train Loss vs Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- [1,0]: d(loss)/dt vs Time ---
ax = axes[1, 0]
for name, (steps, losses, times) in train_data.items():
    if len(losses) > SMOOTH + 1:
        dloss = np.diff(losses)
        dt    = np.diff(times)
        deriv = dloss / dt
        deriv_s = smooth(deriv, SMOOTH)
        times_s = times[SMOOTH // 2: SMOOTH // 2 + len(deriv_s)]
        ax.plot(times_s, deriv_s, color=colors[name], linewidth=1.2, label=f"{name} (final: {deriv_s[-1]:.4f})")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("d(loss)/dt (loss/min)")
ax.set_title("Rate of Loss Decrease vs Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- [1,1]: Val BPB vs Step ---
ax = axes[1, 1]
for name, (steps, bpbs, times) in val_data.items():
    ax.plot(steps, bpbs, color=colors[name], linewidth=1.5, marker='o', markersize=4, label=f"{name} (final: {bpbs[-1]:.4f})")
    print(f"{name}: final val bpb = {bpbs[-1]:.4f}")
for name, sw in SWITCH_STEPS.items():
    ax.axvline(sw, color=colors[name], linestyle="--", linewidth=0.9, alpha=0.5)
ax.set_xlabel("Step")
ax.set_ylabel("Val BPB")
ax.set_title("Val BPB vs Step")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle("FTRL→Muon @100 (depth-4, 2×H100, 2205 steps)", fontsize=12)
plt.tight_layout()
out = "/Users/medha/Desktop/muon_local/nanochat/plots/d4_ftrl_then_muon.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to {out}")
