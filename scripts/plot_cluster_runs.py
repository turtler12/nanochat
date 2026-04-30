"""
Plot train loss vs step, loss vs time, and d(loss)/dt vs time for muon vs ns3+ftrl.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOGS = {
    "muon (2gpu)":       "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_muon_d12_train_log.jsonl",
    "muon (4gpu)":       "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_muon_d12_4gpu.jsonl",
    "ns3+ftrl η=0.1 (2gpu)":          "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_ns3_ftrl_eta0p1_d12_train_log.jsonl",
    "ns3+ftrl linear η=0.3→0 (2gpu)": "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_ns3_ftrl_linear_eta0p3_d12_train_log.jsonl",
    "ns3+ftrl exp η=0.3→0 (2gpu)":    "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_ns3_ftrl_exp_eta0p3_d12_train_log.jsonl",
    "ns3+ftrl exp abs →muon @200 (2gpu, run1)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon.jsonl",
    "ns3+ftrl exp abs →muon @200 (2gpu, run2)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_run2.jsonl",
    "ns3+ftrl exp abs →muon @100 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s100_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @150 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s150_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @200 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s200_d12_4gpu.jsonl",
    "ns3 only (4gpu)":  "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ns3_d12_4gpu.jsonl",
    "ns3+ftrl exp abs η→0 @100 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ns3_ftrl_exp_abs_zero_s100_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @250 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s250_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @300 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s300_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @400 (4gpu)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s400_d12_4gpu.jsonl",
}
COLORS = {
    "muon (2gpu)":       "#1f77b4",
    "muon (4gpu)":       "#17becf",
    "ns3+ftrl η=0.1 (2gpu)":          "#ff7f0e",
    "ns3+ftrl linear η=0.3→0 (2gpu)": "#2ca02c",
    "ns3+ftrl exp η=0.3→0 (2gpu)":    "#d62728",
    "ns3+ftrl exp abs →muon @200 (2gpu, run1)": "#e377c2",
    "ns3+ftrl exp abs →muon @200 (2gpu, run2)": "#c7429a",
    "ns3+ftrl exp abs →muon @100 (4gpu)": "#bcbd22",
    "ns3+ftrl exp abs →muon @150 (4gpu)": "#f39c12",
    "ns3+ftrl exp abs →muon @200 (4gpu)": "#8c564b",
    "ns3 only (4gpu)":  "#7f7f7f",
    "ns3+ftrl exp abs η→0 @100 (4gpu)": "#17a589",
    "ns3+ftrl exp abs →muon @250 (4gpu)": "#9467bd",
    "ns3+ftrl exp abs →muon @300 (4gpu)": "#e6550d",
    "ns3+ftrl exp abs →muon @400 (4gpu)": "#6b4c9a",
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

VAL_LOGS = {
    "muon (2gpu)":                              "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_muon_d12/val_log.jsonl",
    "muon (4gpu)":                              "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_muon_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @200 (2gpu, run1)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log.jsonl",
    "ns3+ftrl exp abs →muon @200 (2gpu, run2)": "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_run2.jsonl",
    "ns3+ftrl exp abs →muon @100 (4gpu)":       "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s100_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @150 (4gpu)":       "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s150_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @200 (4gpu)":       "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s200_d12_4gpu.jsonl",
    "ns3 only (4gpu)":                          "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ns3_d12_4gpu.jsonl",
    "ns3+ftrl exp abs η→0 @100 (4gpu)":        "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ns3_ftrl_exp_abs_zero_s100_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @250 (4gpu)":      "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s250_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @300 (4gpu)":      "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s300_d12_4gpu.jsonl",
    "ns3+ftrl exp abs →muon @400 (4gpu)":      "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s400_d12_4gpu.jsonl",
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

def smooth(x, w):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='valid')

data = {name: load_log(path) for name, path in LOGS.items()}
val_data = {name: load_val_log(path) for name, path in VAL_LOGS.items()}

# Find crossover: first step >50 where muon <= fixed-eta ns3+ftrl
muon_loss = data["muon (2gpu)"][1]
ns3_loss  = data["ns3+ftrl η=0.1 (2gpu)"][1]
muon_time = data["muon (2gpu)"][2]
cross_indices = np.where((muon_loss - ns3_loss) <= 0)[0]
cross_indices = cross_indices[cross_indices > 50]
crossover_time = muon_time[cross_indices[0]] if len(cross_indices) > 0 else None
crossover_step = int(data["muon (2gpu)"][0][cross_indices[0]]) if len(cross_indices) > 0 else None

# Compute eta schedules over time for the two decay modes
# Use muon's time axis as reference (all runs have same step count/timing)
ref_steps = data["muon (2gpu)"][0]
ref_times  = data["muon (2gpu)"][2]
T = len(ref_steps)
eta_linear = 0.3 * (1.0 - ref_steps / max(ref_steps[-1], 1))
eta_exp    = 0.3 * np.exp(-5.0 * ref_steps / max(ref_steps[-1], 1))
eta_fixed  = np.full_like(ref_steps, 0.1, dtype=float)
# Pink: exp abs decay for steps 0-199, then 0 (pure Muon)
eta_hybrid = np.where(ref_steps < 200, 0.3 * np.exp(-5.0 * ref_steps / 2205.0), 0.0)

fig, axes = plt.subplots(3, 3, figsize=(16, 12),
                         gridspec_kw={"height_ratios": [3, 1.2, 2]})

# ── Row 0: loss plots ──────────────────────────────────────────────────────────

# --- Panel [0,0]: Loss vs Step ---
ax = axes[0, 0]
for name, (steps, losses, times) in data.items():
    ax.plot(steps, losses, label=f"{name} ({losses[-1]:.4f})", color=COLORS[name], linewidth=1.2)
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
    ax.plot(times, losses, label=f"{name} ({losses[-1]:.4f})", color=COLORS[name], linewidth=1.2)
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
    ax.plot(times_s, deriv_s, label=f"{name} ({deriv_s[-1]:.4f})", color=COLORS[name], linewidth=1.2)
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
ax.plot(ref_steps, eta_fixed,  label="ns3+ftrl η=0.1 (2gpu)",            color=COLORS["ns3+ftrl η=0.1 (2gpu)"],            linewidth=1.5)
ax.plot(ref_steps, eta_linear, label="ns3+ftrl linear η=0.3→0 (2gpu)",   color=COLORS["ns3+ftrl linear η=0.3→0 (2gpu)"],   linewidth=1.5)
ax.plot(ref_steps, eta_exp,    label="ns3+ftrl exp η=0.3→0 (2gpu)",       color=COLORS["ns3+ftrl exp η=0.3→0 (2gpu)"],      linewidth=1.5)
ax.plot(ref_steps, eta_hybrid, label="ns3+ftrl exp abs →muon @200 (2gpu)", color=COLORS["ns3+ftrl exp abs →muon @200 (2gpu, run1)"], linewidth=1.5)
if crossover_step is not None:
    ax.axvline(crossover_step, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Step")
ax.set_ylabel("η (FTRL eta)")
ax.set_title("η Schedule vs Step")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel [1,1]: η vs Time ---
ax = axes[1, 1]
ax.plot(ref_times, eta_fixed,  label="ns3+ftrl η=0.1 (2gpu)",            color=COLORS["ns3+ftrl η=0.1 (2gpu)"],            linewidth=1.5)
ax.plot(ref_times, eta_linear, label="ns3+ftrl linear η=0.3→0 (2gpu)",   color=COLORS["ns3+ftrl linear η=0.3→0 (2gpu)"],   linewidth=1.5)
ax.plot(ref_times, eta_exp,    label="ns3+ftrl exp η=0.3→0 (2gpu)",       color=COLORS["ns3+ftrl exp η=0.3→0 (2gpu)"],      linewidth=1.5)
ax.plot(ref_times, eta_hybrid, label="ns3+ftrl exp abs →muon @200 (2gpu)", color=COLORS["ns3+ftrl exp abs →muon @200 (2gpu, run1)"], linewidth=1.5)
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

# ── Row 2: val BPB ────────────────────────────────────────────────────────────

# --- Panel [2,0]: Val BPB vs Step ---
ax = axes[2, 0]
for name, (steps, bpbs, times) in val_data.items():
    ax.plot(steps, bpbs, label=f"{name} ({bpbs[-1]:.4f})", color=COLORS[name], linewidth=1.5, marker='o', markersize=3)
ax.set_xlabel("Step")
ax.set_ylabel("Val BPB")
ax.set_title("Val BPB vs Step")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel [2,1]: Val BPB vs Time ---
ax = axes[2, 1]
for name, (steps, bpbs, times) in val_data.items():
    ax.plot(times, bpbs, label=f"{name} ({bpbs[-1]:.4f})", color=COLORS[name], linewidth=1.5, marker='o', markersize=3)
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Val BPB")
ax.set_title("Val BPB vs Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel [2,2]: FLOPs table (4gpu runs, apples-to-apples) ---
ax = axes[2, 2]
ax.axis("off")

# FLOPs per NS iteration on (m,n) matrix: 4*s*r^2 + 2*r^3  (r=min, s=max)
# d12 model: dim=768, ffn_dim=3072, depth=12, vocab=50257
# Matrices and their (m,n): each layer has attn proj (768,768) and ffn (768,3072)+(3072,768)
# Computed analytically: muon=5 NS iters, ns3=3 NS iters, ftrl adds ~6*m*n elementwise (negligible)
# Forward+backward: 6 * non_embedding_params * tokens_per_step
_FWD_BWD_PER_STEP = 267181.0  # GFLOPs
_TOTAL_STEPS = 2205
_OPT_MUON  = 1630.0   # GFLOPs/step  (5 NS iters)
_OPT_NS3   = 978.0    # GFLOPs/step  (3 NS iters)
_OPT_FTRL  = 979.0    # GFLOPs/step  (3 NS + elementwise correction, ~negligible diff)

_4GPU_ROWS = [
    # (short label, data_key, val_key, opt_total_TF, total_run_PF)
    ("muon (4gpu)",
     "muon (4gpu)", "muon (4gpu)",
     _OPT_MUON * _TOTAL_STEPS / 1e3,
     (_OPT_MUON + _FWD_BWD_PER_STEP) * _TOTAL_STEPS / 1e6),
    ("ns3 only (4gpu)",
     "ns3 only (4gpu)", "ns3 only (4gpu)",
     _OPT_NS3 * _TOTAL_STEPS / 1e3,
     (_OPT_NS3 + _FWD_BWD_PER_STEP) * _TOTAL_STEPS / 1e6),
    ("ns3+ftrl η→0 @100 (4gpu)",
     "ns3+ftrl exp abs η→0 @100 (4gpu)", "ns3+ftrl exp abs η→0 @100 (4gpu)",
     _OPT_FTRL * _TOTAL_STEPS / 1e3,
     (_OPT_FTRL + _FWD_BWD_PER_STEP) * _TOTAL_STEPS / 1e6),
    ("hybrid→muon @100 (4gpu)",
     "ns3+ftrl exp abs →muon @100 (4gpu)", "ns3+ftrl exp abs →muon @100 (4gpu)",
     (_OPT_FTRL * 100 + _OPT_MUON * (_TOTAL_STEPS - 100)) / 1e3,
     ((_OPT_FTRL * 100 + _OPT_MUON * (_TOTAL_STEPS - 100)) / _TOTAL_STEPS + _FWD_BWD_PER_STEP) * _TOTAL_STEPS / 1e6),
    ("hybrid→muon @150 (4gpu)",
     "ns3+ftrl exp abs →muon @150 (4gpu)", "ns3+ftrl exp abs →muon @150 (4gpu)",
     (_OPT_FTRL * 150 + _OPT_MUON * (_TOTAL_STEPS - 150)) / 1e3,
     ((_OPT_FTRL * 150 + _OPT_MUON * (_TOTAL_STEPS - 150)) / _TOTAL_STEPS + _FWD_BWD_PER_STEP) * _TOTAL_STEPS / 1e6),
    ("hybrid→muon @200 (4gpu)",
     "ns3+ftrl exp abs →muon @200 (4gpu)", "ns3+ftrl exp abs →muon @200 (4gpu)",
     (_OPT_FTRL * 200 + _OPT_MUON * (_TOTAL_STEPS - 200)) / 1e3,
     ((_OPT_FTRL * 200 + _OPT_MUON * (_TOTAL_STEPS - 200)) / _TOTAL_STEPS + _FWD_BWD_PER_STEP) * _TOTAL_STEPS / 1e6),
    ("hybrid→muon @250 (4gpu)",
     "ns3+ftrl exp abs →muon @250 (4gpu)", "ns3+ftrl exp abs →muon @250 (4gpu)",
     (_OPT_FTRL * 250 + _OPT_MUON * (_TOTAL_STEPS - 250)) / 1e3,
     ((_OPT_FTRL * 250 + _OPT_MUON * (_TOTAL_STEPS - 250)) / _TOTAL_STEPS + _FWD_BWD_PER_STEP) * _TOTAL_STEPS / 1e6),
    ("hybrid→muon @300 (4gpu)",
     "ns3+ftrl exp abs →muon @300 (4gpu)", "ns3+ftrl exp abs →muon @300 (4gpu)",
     (_OPT_FTRL * 300 + _OPT_MUON * (_TOTAL_STEPS - 300)) / 1e3,
     ((_OPT_FTRL * 300 + _OPT_MUON * (_TOTAL_STEPS - 300)) / _TOTAL_STEPS + _FWD_BWD_PER_STEP) * _TOTAL_STEPS / 1e6),
    ("hybrid→muon @400 (4gpu)",
     "ns3+ftrl exp abs →muon @400 (4gpu)", "ns3+ftrl exp abs →muon @400 (4gpu)",
     (_OPT_FTRL * 400 + _OPT_MUON * (_TOTAL_STEPS - 400)) / 1e3,
     ((_OPT_FTRL * 400 + _OPT_MUON * (_TOTAL_STEPS - 400)) / _TOTAL_STEPS + _FWD_BWD_PER_STEP) * _TOTAL_STEPS / 1e6),
]

muon_opt_total = _4GPU_ROWS[0][3]
col_labels = ["Run", "Opt FLOPs\n(TFLOPs)", "Total\n(PFLOPs)", "Opt\nSaved %", "Final\nTrain Loss", "Best\nVal BPB"]
table_data = []
for short_label, data_key, val_key, opt_tf, total_pf in _4GPU_ROWS:
    saved_pct = (1.0 - opt_tf / muon_opt_total) * 100.0
    final_train = f"{data[data_key][1][-1]:.4f}" if data_key in data else "—"
    best_val = f"{min(val_data[val_key][1]):.4f}" if val_key in val_data else "—"
    table_data.append([short_label, f"{opt_tf:.0f}", f"{total_pf:.3f}", f"{saved_pct:.1f}%", final_train, best_val])

tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0.0, 0.0, 1.0, 1.0],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#d0d0d0")
        cell.set_text_props(fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f5f5f5")
    col_widths = {0: 0.32, 1: 0.14, 2: 0.12, 3: 0.12, 4: 0.15, 5: 0.15}
    cell.set_width(col_widths.get(col, 0.14))
    cell.set_edgecolor("#cccccc")
ax.set_title("FLOPs Comparison (d12, 4gpu runs)", fontsize=9, pad=4)

plt.suptitle("Muon vs NS3+FTRL variants (depth-12, 2×H100, 2205 steps)", fontsize=12)
plt.tight_layout()
plt.savefig("/Users/medha/Desktop/muon_local/nanochat/plots/headline/cluster_muon_vs_ns3ftrl.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Crossover: step {crossover_step}, time {crossover_time:.1f}m")
print("Saved to plots/headline/cluster_muon_vs_ns3ftrl.png")
