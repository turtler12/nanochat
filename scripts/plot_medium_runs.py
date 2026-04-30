"""
Plot depth-20 runs: loss vs step, loss vs time, and FLOPs table.
"""
import json
import math
import numpy as np
import matplotlib.pyplot as plt

LOGS = {
    "muon (d20, 1gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_muon_d20_train_log.jsonl",
        "#1f77b4",
    ),
    "ns3+ftrl exp η=0.3→0 relative (d20, 1gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_ns3_ftrl_exp_eta0p3_d20_train_log.jsonl",
        "#d62728",
    ),
    "ns3+ftrl exp η=0.3→0 absolute (d20, 1gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat_cache/base_checkpoints/ablation_ns3_ftrl_exp_abs_eta0p3_d20_train_log.jsonl",
        "#9467bd",
    ),
    "ns3+ftrl exp abs →muon @100 (d20, 4gpu)": (
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/train_log_ftrl_then_muon_s100_d20_4gpu.jsonl",
        "#e377c2",
    ),
}

VAL_LOGS = {
    "ns3+ftrl exp abs →muon @100 (d20, 4gpu)":
        "/Users/medha/Desktop/muon_local/nanochat/my_runs/val_log_ftrl_then_muon_s100_d20_4gpu.jsonl",
}

SMOOTH = 30

def load_log(path):
    steps, losses, times = [], [], []
    cumtime = 0.0
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

NUM_ITERATIONS = 4777
ETA0 = 0.3
_D12_REF_STEPS = 2205
_ETA_LAMBDA = math.log(3.0) / 100.0  # η(100) = 0.1

fig, axes = plt.subplots(2, 3, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1.2]})

all_data = {}
for name, (path, color) in LOGS.items():
    steps, losses, times = load_log(path)
    all_data[name] = (steps, losses, times, color)

    label = f"{name} ({losses[-1]:.4f})"
    axes[0, 0].plot(steps, losses, label=label, color=color, linewidth=1.2)
    axes[0, 1].plot(times, losses, label=label, color=color, linewidth=1.2)

    print(f"{name}: {len(steps)} steps, {times[-1]:.1f}m elapsed, last loss {losses[-1]:.4f}")

val_data = {name: load_val_log(path) for name, path in VAL_LOGS.items()}

hybrid_color = LOGS["ns3+ftrl exp abs →muon @100 (d20, 4gpu)"][1]
axes[0, 0].axvline(100, color=hybrid_color, linestyle="--", linewidth=0.9, alpha=0.5)
axes[0, 0].text(105, 9.5, "@100", fontsize=7, color=hybrid_color)
axes[0, 0].set_xlabel("Step"); axes[0, 0].set_ylabel("Train Loss"); axes[0, 0].set_title("Loss vs Step")
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

hybrid_steps, _, hybrid_times, _ = all_data["ns3+ftrl exp abs →muon @100 (d20, 4gpu)"]
switch_idx = np.searchsorted(hybrid_steps, 100)
if switch_idx < len(hybrid_times):
    axes[0, 1].axvline(hybrid_times[switch_idx], color=hybrid_color, linestyle="--", linewidth=0.9, alpha=0.5)
    axes[0, 1].text(hybrid_times[switch_idx] + 0.05, 9.5, f"@{hybrid_times[switch_idx]:.1f}m", fontsize=7, color=hybrid_color)
axes[0, 1].set_xlabel("Time (minutes)"); axes[0, 1].set_ylabel("Train Loss"); axes[0, 1].set_title("Loss vs Time")
axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

# ── Panel [0,2]: FLOPs table ───────────────────────────────────────────────
ax_tbl = axes[0, 2]
ax_tbl.axis("off")

_FWD_BWD_PER_STEP = 578844.0   # GFLOPs/step for d20 (scaled from d12)
_OPT_MUON  = 3531.0            # GFLOPs/step d20 (5 NS iters, larger matrices)
_OPT_FTRL  = 2119.0            # GFLOPs/step d20 (3 NS iters + elementwise)
_SWITCH_STEP = 100

_TABLE_ROWS = [
    ("muon (d20, 1gpu)",
     "muon (d20, 1gpu)", None,
     _OPT_MUON * NUM_ITERATIONS / 1e3,
     (_OPT_MUON + _FWD_BWD_PER_STEP) * NUM_ITERATIONS / 1e6),
    ("ns3+ftrl exp rel (d20, 1gpu)",
     "ns3+ftrl exp η=0.3→0 relative (d20, 1gpu)", None,
     _OPT_FTRL * NUM_ITERATIONS / 1e3,
     (_OPT_FTRL + _FWD_BWD_PER_STEP) * NUM_ITERATIONS / 1e6),
    ("ns3+ftrl exp abs (d20, 1gpu)",
     "ns3+ftrl exp η=0.3→0 absolute (d20, 1gpu)", None,
     _OPT_FTRL * NUM_ITERATIONS / 1e3,
     (_OPT_FTRL + _FWD_BWD_PER_STEP) * NUM_ITERATIONS / 1e6),
    ("hybrid→muon @100 (d20, 4gpu)",
     "ns3+ftrl exp abs →muon @100 (d20, 4gpu)",
     "ns3+ftrl exp abs →muon @100 (d20, 4gpu)",
     (_OPT_FTRL * _SWITCH_STEP + _OPT_MUON * (NUM_ITERATIONS - _SWITCH_STEP)) / 1e3,
     ((_OPT_FTRL * _SWITCH_STEP + _OPT_MUON * (NUM_ITERATIONS - _SWITCH_STEP)) / NUM_ITERATIONS
      + _FWD_BWD_PER_STEP) * NUM_ITERATIONS / 1e6),
]

muon_opt_total = _TABLE_ROWS[0][3]
col_labels = ["Run", "Opt FLOPs\n(TFLOPs)", "Total\n(PFLOPs)", "Opt\nSaved %", "Final\nTrain Loss", "Best\nVal BPB"]
table_data = []
for short_label, data_key, val_key, opt_tf, total_pf in _TABLE_ROWS:
    saved_pct = (1.0 - opt_tf / muon_opt_total) * 100.0
    final_train = f"{all_data[data_key][1][-1]:.4f}" if data_key in all_data else "—"
    best_val = f"{min(val_data[val_key][1]):.4f}" if val_key and val_key in val_data else "—"
    table_data.append([short_label, f"{opt_tf:.0f}", f"{total_pf:.3f}", f"{saved_pct:.1f}%", final_train, best_val])

tbl = ax_tbl.table(
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
ax_tbl.set_title("FLOPs Comparison (d20, 4777 steps)", fontsize=9, pad=4)

# ── Row 1: eta schedule ────────────────────────────────────────────────────
rel_steps, _, rel_times, rel_color = all_data["ns3+ftrl exp η=0.3→0 relative (d20, 1gpu)"]
abs_steps, _, abs_times, abs_color = all_data["ns3+ftrl exp η=0.3→0 absolute (d20, 1gpu)"]

eta_rel    = ETA0 * np.exp(-5.0 * rel_steps / NUM_ITERATIONS)
eta_abs    = ETA0 * np.exp(-_ETA_LAMBDA * abs_steps)
eta_hybrid = np.where(hybrid_steps < 100, ETA0 * np.exp(-_ETA_LAMBDA * hybrid_steps), 0.0)

d12_steps_run = np.arange(min(_D12_REF_STEPS, int(rel_steps[-1]) + 1))
eta_d12       = ETA0 * np.exp(-5.0 * d12_steps_run / _D12_REF_STEPS)
d12_times_run = d12_steps_run * (rel_times[-1] / max(rel_steps[-1], 1))

axes[1, 0].plot(rel_steps, eta_rel,    color=rel_color,    linewidth=1.5, label="exp η relative (d20, 1gpu)")
axes[1, 0].plot(abs_steps, eta_abs,    color=abs_color,    linewidth=1.5, label="exp η absolute (d20, 1gpu)")
axes[1, 0].plot(d12_steps_run, eta_d12, color="#ff7f0e",   linewidth=1.5, linestyle="--", label="exp η (d12 reference)")
axes[1, 0].plot(hybrid_steps, eta_hybrid, color=hybrid_color, linewidth=1.5, label="exp abs →muon @100 (d20, 4gpu)")
axes[1, 0].axhline(0.1, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
axes[1, 0].text(10, 0.105, "η=0.1", fontsize=7, color="gray")
axes[1, 0].set_xlabel("Step"); axes[1, 0].set_ylabel("η (FTRL eta)")
axes[1, 0].set_title("η Schedule vs Step (actual steps run)")
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(rel_times, eta_rel,     color=rel_color,   linewidth=1.5, label="exp η relative (d20, 1gpu)")
axes[1, 1].plot(abs_times, eta_abs,     color=abs_color,   linewidth=1.5, label="exp η absolute (d20, 1gpu)")
axes[1, 1].plot(d12_times_run, eta_d12, color="#ff7f0e",   linewidth=1.5, linestyle="--", label="exp η (d12 reference)")
axes[1, 1].plot(hybrid_times, eta_hybrid, color=hybrid_color, linewidth=1.5, label="exp abs →muon @100 (d20, 4gpu)")
axes[1, 1].axhline(0.1, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
axes[1, 1].set_xlabel("Time (minutes)"); axes[1, 1].set_ylabel("η (FTRL eta)")
axes[1, 1].set_title("η Schedule vs Time (actual steps run)")
axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].axis("off")

plt.suptitle("Depth-20 runs (4777 steps total) — partial + hybrid results", fontsize=12)
plt.tight_layout()
plt.savefig("/Users/medha/Desktop/muon_local/nanochat/plots/headline/cluster_muon_vs_ns3ftrl_medium.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to plots/headline/cluster_muon_vs_ns3ftrl_medium.png")
