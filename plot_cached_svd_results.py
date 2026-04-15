"""Plot cached-SVD SoftmaxMuon sweep + weight-softmax results."""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

BASE = "cache/base_checkpoints"
OUT_DIR = "softmax_sweep_results"
os.makedirs(OUT_DIR, exist_ok=True)

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if "_config" not in obj:
                rows.append(obj)
    return rows

smooth_window = 50

# =========================================================================
# Figure 1: Cached-SVD sweep — loss vs wall-clock and loss vs step
# Use 2-GPU baseline from the same sweep for fair comparison
# =========================================================================
csvd_runs = {}

# Reference runs (2-GPU baseline from same sweep, + full SVD softmax)
for label, dirname in [
    ("Baseline Muon (2 GPU)", "cached_svd_baseline_d20"),
    ("SVD SoftmaxMuon eta=0.5", "softmax_v2_eta0.5_d20"),
]:
    p = os.path.join(BASE, dirname, "train_log.jsonl")
    if os.path.exists(p):
        csvd_runs[label] = p

# Cached-SVD runs
for freq in [1, 5, 10, 20]:
    p = os.path.join(BASE, f"cached_svd_freq{freq}_d20", "train_log.jsonl")
    if os.path.exists(p):
        data = load_jsonl(p)
        if len(data) > smooth_window + 1:
            csvd_runs[f"Cached SVD freq={freq}"] = p
        else:
            print(f"  Skipping cached_svd_freq{freq}: only {len(data)} steps")

csvd_colors = {
    "Baseline Muon (2 GPU)": "#555555",
    "SVD SoftmaxMuon eta=0.5": "#9467bd",
    "Cached SVD freq=1": "#d62728",
    "Cached SVD freq=5": "#2ca02c",
    "Cached SVD freq=10": "#1f77b4",
    "Cached SVD freq=20": "#ff7f00",
}
csvd_linestyles = {
    "Baseline Muon (2 GPU)": "-",
    "SVD SoftmaxMuon eta=0.5": "--",
    "Cached SVD freq=1": "--",
    "Cached SVD freq=5": "-.",
    "Cached SVD freq=10": "-",
    "Cached SVD freq=20": ":",
}
csvd_linewidths = {
    "Baseline Muon (2 GPU)": 2.5,
    "SVD SoftmaxMuon eta=0.5": 1.8,
    "Cached SVD freq=1": 1.8,
    "Cached SVD freq=5": 1.8,
    "Cached SVD freq=10": 2.0,
    "Cached SVD freq=20": 1.8,
}

print(f"Cached-SVD runs: {list(csvd_runs.keys())}")

if csvd_runs:
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(16, 6))

    caption_lines = []
    for name, path in csvd_runs.items():
        data = load_jsonl(path)
        if len(data) < smooth_window + 1:
            continue

        steps = np.array([d["step"] for d in data])
        losses = np.array([d["loss"] for d in data])
        dts = np.array([d["dt"] for d in data])
        times_min = np.cumsum(dts) / 60.0

        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(losses, kernel, mode="valid")
        s_steps = steps[smooth_window - 1:]
        s_times = times_min[smooth_window - 1:]

        avg_dt_ms = np.mean(dts[min(20, len(dts)):]) * 1000

        c = csvd_colors.get(name, "#333")
        ls = csvd_linestyles.get(name, "-")
        lw = csvd_linewidths.get(name, 1.5)
        is_ref = "Baseline" in name or "SVD Softmax" in name
        alpha = 0.6 if is_ref else 1.0
        zorder = 1 if is_ref else 2

        label = f"{name} ({avg_dt_ms:.0f}ms/step, {len(data)} steps)"

        ax1a.plot(s_times, smoothed, label=label, color=c, linewidth=lw,
                  linestyle=ls, alpha=alpha, zorder=zorder)
        ax1b.plot(s_steps, smoothed, label=label, color=c, linewidth=lw,
                  linestyle=ls, alpha=alpha, zorder=zorder)

        final_loss = smoothed[-1]
        caption_lines.append((avg_dt_ms, f"{name}: {avg_dt_ms:.0f}ms/step, loss={final_loss:.3f} @ step {int(s_steps[-1])}"))

    caption_lines.sort(key=lambda x: x[0])
    caption = "Per-step time & final loss:\n" + "\n".join(line for _, line in caption_lines)
    ax1b.text(0.97, 0.97, caption, transform=ax1b.transAxes, fontsize=7.5,
              verticalalignment="top", horizontalalignment="right", family="monospace",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#cccccc"))

    ax1a.set_xlabel("Wall-Clock Time (minutes)", fontsize=12)
    ax1a.set_ylabel("Training Loss (smoothed)", fontsize=12)
    ax1a.set_title("Loss vs Wall-Clock Time", fontsize=13)
    ax1a.legend(fontsize=7.5, loc="upper right")
    ax1a.grid(True, alpha=0.3)

    ax1b.set_xlabel("Training Step", fontsize=12)
    ax1b.set_ylabel("Training Loss (smoothed)", fontsize=12)
    ax1b.set_title("Loss vs Step", fontsize=13)
    ax1b.legend(fontsize=7.5, loc="lower left")
    ax1b.grid(True, alpha=0.3)

    fig1.suptitle("Cached-SVD SoftmaxMuon (eta=0.5) — SVD Frequency Sweep", fontsize=14, y=1.01)
    fig1.tight_layout()
    out1 = os.path.join(OUT_DIR, "cached_svd_loss_vs_wallclock.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")

# =========================================================================
# Figure 2: Per-step wall time bar chart (use 2-GPU baseline)
# =========================================================================
timing_data = {}
for label, dirname in [
    ("Baseline Muon\n(2 GPU)", "cached_svd_baseline_d20"),
    ("SVD Softmax\n(every step)", "softmax_v2_eta0.5_d20"),
    ("Cached SVD\nfreq=1", "cached_svd_freq1_d20"),
    ("Cached SVD\nfreq=5", "cached_svd_freq5_d20"),
    ("Cached SVD\nfreq=10", "cached_svd_freq10_d20"),
    ("Cached SVD\nfreq=20", "cached_svd_freq20_d20"),
]:
    p = os.path.join(BASE, dirname, "train_log.jsonl")
    if os.path.exists(p):
        data = load_jsonl(p)
        if len(data) > 20:
            dts = [d["dt"] for d in data[11:]]
            timing_data[label] = np.mean(dts) * 1000

if timing_data:
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    bar_colors = {
        "Baseline Muon\n(2 GPU)": "#555555",
        "SVD Softmax\n(every step)": "#9467bd",
        "Cached SVD\nfreq=1": "#d62728",
        "Cached SVD\nfreq=5": "#2ca02c",
        "Cached SVD\nfreq=10": "#1f77b4",
        "Cached SVD\nfreq=20": "#ff7f00",
    }

    names = list(timing_data.keys())
    times = list(timing_data.values())
    cols = [bar_colors.get(n, "#333") for n in names]

    bars = ax2.bar(names, times, color=cols, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f"{t:.0f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Add speedup annotation relative to full SVD
    if "SVD Softmax\n(every step)" in timing_data:
        svd_time = timing_data["SVD Softmax\n(every step)"]
        for bar, (n, t) in zip(bars, zip(names, times)):
            if "Cached" in n and t < svd_time * 0.9:
                speedup = svd_time / t
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                         f"{speedup:.1f}x faster\nvs full SVD", ha="center", va="center",
                         fontsize=8, color="white", fontweight="bold")

    ax2.set_ylabel("Average Step Time (ms)", fontsize=12)
    ax2.set_title("Per-Step Wall Time — Cached SVD vs Baselines", fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    fig2.tight_layout()
    out2 = os.path.join(OUT_DIR, "cached_svd_step_time.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")

# =========================================================================
# Figure 3: Validation BPB comparison (use 2-GPU baseline)
# =========================================================================
val_runs = {}
for label, dirname in [
    ("Baseline Muon (2 GPU)", "cached_svd_baseline_d20"),
    ("SVD SoftmaxMuon eta=0.5", "softmax_v2_eta0.5_d20"),
    ("Cached SVD freq=1", "cached_svd_freq1_d20"),
    ("Cached SVD freq=5", "cached_svd_freq5_d20"),
    ("Cached SVD freq=10", "cached_svd_freq10_d20"),
    ("Cached SVD freq=20", "cached_svd_freq20_d20"),
]:
    vp = os.path.join(BASE, dirname, "val_log.jsonl")
    tp = os.path.join(BASE, dirname, "train_log.jsonl")
    if os.path.exists(vp):
        vdata = load_jsonl(vp)
        tdata = load_jsonl(tp) if os.path.exists(tp) else []
        if len(vdata) >= 2:
            val_runs[label] = dict(val=vdata, train=tdata)

val_colors = {
    "Baseline Muon (2 GPU)": "#555555",
    "SVD SoftmaxMuon eta=0.5": "#9467bd",
    "Cached SVD freq=1": "#d62728",
    "Cached SVD freq=5": "#2ca02c",
    "Cached SVD freq=10": "#1f77b4",
    "Cached SVD freq=20": "#ff7f00",
}

if val_runs:
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

    for name, rd in val_runs.items():
        vd = rd["val"]
        vsteps = [d["step"] for d in vd]
        vbpbs = [d["val_bpb"] for d in vd]
        vtimes = [d["total_training_time"] / 60.0 for d in vd]

        c = val_colors.get(name, "#333")
        is_ref = "Baseline" in name or "SVD Softmax" in name
        lw = 2.5 if "Baseline" in name else 1.8
        alpha = 0.6 if is_ref else 1.0

        final_bpb = vbpbs[-1]
        label = f"{name} (bpb={final_bpb:.4f} @ step {vsteps[-1]})"

        ax3a.plot(vsteps, vbpbs, marker="o", markersize=4,
                  label=label, color=c, linewidth=lw, alpha=alpha)
        ax3b.plot(vtimes, vbpbs, marker="o", markersize=4,
                  label=label, color=c, linewidth=lw, alpha=alpha)

    ax3a.set_xlabel("Training Step", fontsize=12)
    ax3a.set_ylabel("Validation BPB", fontsize=12)
    ax3a.set_title("Validation BPB vs Step", fontsize=13)
    ax3a.legend(fontsize=7.5, loc="upper right")
    ax3a.grid(True, alpha=0.3)

    ax3b.set_xlabel("Wall-Clock Time (minutes)", fontsize=12)
    ax3b.set_ylabel("Validation BPB", fontsize=12)
    ax3b.set_title("Validation BPB vs Wall-Clock Time", fontsize=13)
    ax3b.legend(fontsize=7.5, loc="upper right")
    ax3b.grid(True, alpha=0.3)

    fig3.suptitle("Validation BPB — Cached SVD vs Baselines (2 GPU)", fontsize=14, y=1.01)
    fig3.tight_layout()
    out3 = os.path.join(OUT_DIR, "cached_svd_val_bpb.png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved {out3}")

# =========================================================================
# Figure 4: Weight-Softmax results (use 2-GPU baseline)
# =========================================================================
wsm_runs = {}
for label, dirname in [
    ("Baseline Muon (2 GPU)", "cached_svd_baseline_d20"),
    ("SVD SoftmaxMuon eta=0.5", "softmax_v2_eta0.5_d20"),
    ("W-SoftmaxMuon eta=0.5", "wsoftmax_eta0.5_d20"),
]:
    p = os.path.join(BASE, dirname, "train_log.jsonl")
    if os.path.exists(p):
        data = load_jsonl(p)
        if len(data) > smooth_window + 1:
            wsm_runs[label] = data
        else:
            print(f"  W-Softmax: {dirname} has only {len(data)} steps (need >{smooth_window})")

wsm_colors = {
    "Baseline Muon (2 GPU)": "#555555",
    "SVD SoftmaxMuon eta=0.5": "#1b9e77",
    "W-SoftmaxMuon eta=0.5": "#9467bd",
}

if wsm_runs:
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(16, 6))

    caption_lines_4 = []
    for name, data in wsm_runs.items():
        steps = np.array([d["step"] for d in data])
        losses = np.array([d["loss"] for d in data])
        dts = np.array([d["dt"] for d in data])
        times_min = np.cumsum(dts) / 60.0

        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(losses, kernel, mode="valid")
        s_steps = steps[smooth_window - 1:]
        s_times = times_min[smooth_window - 1:]
        avg_dt_ms = np.mean(dts[min(20, len(dts)):]) * 1000

        c = wsm_colors.get(name, "#333")
        is_baseline = "Baseline" in name
        lw = 2.5 if is_baseline else 1.8
        alpha = 0.6 if is_baseline else 1.0
        zorder = 1 if is_baseline else 2

        label = f"{name} ({avg_dt_ms:.0f}ms/step)"
        ax4a.plot(s_times, smoothed, label=label, color=c, linewidth=lw,
                  alpha=alpha, zorder=zorder)
        ax4b.plot(s_steps, smoothed, label=label, color=c, linewidth=lw,
                  alpha=alpha, zorder=zorder)

        caption_lines_4.append((avg_dt_ms, f"{name}: {avg_dt_ms:.0f}ms/step, loss={smoothed[-1]:.3f} @ step {int(s_steps[-1])}"))

    caption_lines_4.sort(key=lambda x: x[0])
    caption_4 = "\n".join(line for _, line in caption_lines_4)
    ax4b.text(0.97, 0.97, caption_4, transform=ax4b.transAxes, fontsize=8.5,
              verticalalignment="top", horizontalalignment="right", family="monospace",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#cccccc"))

    ax4a.set_xlabel("Wall-Clock Time (minutes)", fontsize=12)
    ax4a.set_ylabel("Training Loss (smoothed)", fontsize=12)
    ax4a.set_title("Loss vs Wall-Clock Time", fontsize=13)
    ax4a.legend(fontsize=9, loc="upper right")
    ax4a.grid(True, alpha=0.3)

    ax4b.set_xlabel("Training Step", fontsize=12)
    ax4b.set_ylabel("Training Loss (smoothed)", fontsize=12)
    ax4b.set_title("Loss vs Step", fontsize=13)
    ax4b.legend(fontsize=9, loc="lower left")
    ax4b.grid(True, alpha=0.3)

    fig4.suptitle("Weight-SoftmaxMuon vs SVD SoftmaxMuon vs Baseline (2 GPU)", fontsize=14, y=1.01)
    fig4.tight_layout()
    out4 = os.path.join(OUT_DIR, "wsoftmax_comparison.png")
    fig4.savefig(out4, dpi=150, bbox_inches="tight")
    print(f"Saved {out4}")
else:
    print("Not enough W-Softmax data for comparison plot")

# =========================================================================
# Summary table
# =========================================================================
print("\n" + "=" * 90)
print("SUMMARY: All Experiments (2 GPU)")
print("=" * 90)
print(f"{'Run':<30} {'Steps':>6} {'Avg dt (ms)':>12} {'Final Loss':>12} {'Final Val BPB':>15}")
print("-" * 77)

all_summaries = [
    ("Baseline Muon (2 GPU)", "cached_svd_baseline_d20"),
    ("SVD SoftmaxMuon eta=0.5", "softmax_v2_eta0.5_d20"),
    ("Cached SVD freq=1", "cached_svd_freq1_d20"),
    ("Cached SVD freq=5", "cached_svd_freq5_d20"),
    ("Cached SVD freq=10", "cached_svd_freq10_d20"),
    ("Cached SVD freq=20", "cached_svd_freq20_d20"),
    ("W-SoftmaxMuon eta=0.5", "wsoftmax_eta0.5_d20"),
]
for label, dirname in all_summaries:
    tp = os.path.join(BASE, dirname, "train_log.jsonl")
    vp = os.path.join(BASE, dirname, "val_log.jsonl")
    if not os.path.exists(tp):
        continue
    tdata = load_jsonl(tp)
    if len(tdata) < 2:
        continue
    dts = [d["dt"] for d in tdata[min(11, len(tdata)):]]
    avg_dt = np.mean(dts) * 1000 if dts else 0
    final_loss = tdata[-1]["loss"]
    vdata = load_jsonl(vp) if os.path.exists(vp) else []
    val_str = f"{vdata[-1]['val_bpb']:.4f}" if vdata else "N/A"
    print(f"{label:<30} {len(tdata):>6} {avg_dt:>12.0f} {final_loss:>12.4f} {val_str:>15}")

print("=" * 90)
