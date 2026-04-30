"""
Plot train loss vs step, loss vs time, and FLOPs table for d12 4gpu runs.
Hybrid-only variant: muon baseline + FTRL→muon switch-step ablation.
"""
import json
import math
import numpy as np
import matplotlib.pyplot as plt

LOGS = {
    "muon (4gpu)":                        "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_muon_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @100 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s100_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @150 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s150_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @200 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s200_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @250 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s250_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @300 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s300_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @400 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s400_d12_4gpu.jsonl",
}
COLORS = {
    "muon (4gpu)":                        "#17becf",
    "ns3+ftrl exp abs →muon @100 (4gpu)": "#bcbd22",
    "ns3+ftrl exp abs →muon @150 (4gpu)": "#f39c12",
    "ns3+ftrl exp abs →muon @200 (4gpu)": "#8c564b",
    "ns3+ftrl exp abs →muon @250 (4gpu)": "#9467bd",
    "ns3+ftrl exp abs →muon @300 (4gpu)": "#e6550d",
    "ns3+ftrl exp abs →muon @400 (4gpu)": "#6b4c9a",
}

VAL_LOGS = {
    "muon (4gpu)":                        "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_muon_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @100 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s100_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @150 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s150_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @200 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s200_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @250 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s250_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @300 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s300_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @400 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s400_d12_4gpu.jsonl",
}

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

data     = {name: load_log(path)     for name, path in LOGS.items()}
val_data = {name: load_val_log(path) for name, path in VAL_LOGS.items()}

_ETA_LAMBDA = math.log(3.0) / 100.0
switch_steps  = [100, 150, 200, 250, 300, 400]
switch_colors = [COLORS[f"ns3+ftrl exp abs →muon @{s} (4gpu)"] for s in switch_steps]
ref_steps = data["muon (4gpu)"][0]
ref_times = data["muon (4gpu)"][2]

def eta_schedule(steps, switch_step):
    return np.where(steps < switch_step, 0.3 * np.exp(-_ETA_LAMBDA * steps), 0.0)

fig, axes = plt.subplots(3, 3, figsize=(16, 12),
                         gridspec_kw={"height_ratios": [3, 1.2, 2]})

# ── Row 0: loss ────────────────────────────────────────────────────────────
ax = axes[0, 0]
for name, (steps, losses, times) in data.items():
    ax.plot(steps, losses, label=f"{name} ({losses[-1]:.4f})", color=COLORS[name], linewidth=1.2)
ax.set_xlabel("Step"); ax.set_ylabel("Train Loss"); ax.set_title("Loss vs Step")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for name, (steps, losses, times) in data.items():
    ax.plot(times, losses, label=f"{name} ({losses[-1]:.4f})", color=COLORS[name], linewidth=1.2)
ax.set_xlabel("Time (minutes)"); ax.set_ylabel("Train Loss"); ax.set_title("Loss vs Time")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

axes[0, 2].axis("off")

# ── Row 1: eta schedules ───────────────────────────────────────────────────
ax = axes[1, 0]
for sw, col in zip(switch_steps, switch_colors):
    ax.plot(ref_steps, eta_schedule(ref_steps, sw), label=f"hybrid →muon @{sw}", color=col, linewidth=1.5)
ax.set_xlabel("Step"); ax.set_ylabel("η (FTRL eta)"); ax.set_title("η Schedule vs Step")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for sw, col in zip(switch_steps, switch_colors):
    ax.plot(ref_times, eta_schedule(ref_steps, sw), label=f"hybrid →muon @{sw}", color=col, linewidth=1.5)
ax.set_xlabel("Time (minutes)"); ax.set_ylabel("η (FTRL eta)"); ax.set_title("η Schedule vs Time")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

axes[1, 2].axis("off")

# ── Row 2: val BPB ────────────────────────────────────────────────────────
ax = axes[2, 0]
for name, (steps, bpbs, times) in val_data.items():
    ax.plot(steps, bpbs, label=f"{name} ({bpbs[-1]:.4f})", color=COLORS[name], linewidth=1.5, marker='o', markersize=3)
ax.set_xlabel("Step"); ax.set_ylabel("Val BPB"); ax.set_title("Val BPB vs Step")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2, 1]
for name, (steps, bpbs, times) in val_data.items():
    ax.plot(times, bpbs, label=f"{name} ({bpbs[-1]:.4f})", color=COLORS[name], linewidth=1.5, marker='o', markersize=3)
ax.set_xlabel("Time (minutes)"); ax.set_ylabel("Val BPB"); ax.set_title("Val BPB vs Time")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── FLOPs table ───────────────────────────────────────────────────────────
ax = axes[2, 2]
ax.axis("off")

_FWD_BWD_PER_STEP = 267181.0
_TOTAL_STEPS = 2205
_OPT_MUON = 1630.0
_OPT_FTRL = 979.0

_ROWS = [
    ("muon (4gpu)", "muon (4gpu)", "muon (4gpu)",
     _OPT_MUON * _TOTAL_STEPS / 1e3,
     (_OPT_MUON + _FWD_BWD_PER_STEP) * _TOTAL_STEPS / 1e6),
] + [
    (f"hybrid→muon @{sw} (4gpu)",
     f"ns3+ftrl exp abs →muon @{sw} (4gpu)",
     f"ns3+ftrl exp abs →muon @{sw} (4gpu)",
     (_OPT_FTRL * sw + _OPT_MUON * (_TOTAL_STEPS - sw)) / 1e3,
     ((_OPT_FTRL * sw + _OPT_MUON * (_TOTAL_STEPS - sw)) / _TOTAL_STEPS + _FWD_BWD_PER_STEP) * _TOTAL_STEPS / 1e6)
    for sw in switch_steps
]

muon_opt_total = _ROWS[0][3]
col_labels = ["Run", "Opt FLOPs\n(TFLOPs)", "Total\n(PFLOPs)", "Opt\nSaved %", "Final\nTrain Loss", "Best\nVal BPB"]
table_data = []
for short_label, data_key, val_key, opt_tf, total_pf in _ROWS:
    saved_pct  = (1.0 - opt_tf / muon_opt_total) * 100.0
    final_train = f"{data[data_key][1][-1]:.4f}" if data_key in data else "—"
    best_val    = f"{min(val_data[val_key][1]):.4f}" if val_key in val_data else "—"
    table_data.append([short_label, f"{opt_tf:.0f}", f"{total_pf:.3f}", f"{saved_pct:.1f}%", final_train, best_val])

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               cellLoc="center", loc="center", bbox=[0.0, 0.0, 1.0, 1.0])
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#d0d0d0")
        cell.set_text_props(fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f5f5f5")
    cell.set_width({0: 0.32, 1: 0.14, 2: 0.12, 3: 0.12, 4: 0.15, 5: 0.15}.get(col, 0.14))
    cell.set_edgecolor("#cccccc")
ax.set_title("FLOPs Comparison (d12, 4gpu runs)", fontsize=9, pad=4)

plt.suptitle("Muon vs FTRL→Muon hybrid (depth-12, 4×H100, 2205 steps)", fontsize=12)
plt.tight_layout()
plt.savefig("/Users/medha/Desktop/muon_local/nanochat/plots/headline/cluster_muon_vs_ns3ftrl_4gpu_v2.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to plots/headline/cluster_muon_vs_ns3ftrl_4gpu_v2.png")
