"""Plot randomized SVD SoftmaxMuon sweep results."""
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
# Figure 1: Loss vs wall-clock and loss vs step
# =========================================================================
runs = {}

# Reference runs
for label, dirname in [
    ("Baseline Muon (2 GPU)", "cached_svd_baseline_d20"),
    ("Full SVD SoftmaxMuon eta=0.5", "softmax_v2_eta0.5_d20"),
]:
    p = os.path.join(BASE, dirname, "train_log.jsonl")
    if os.path.exists(p):
        data = load_jsonl(p)
        if len(data) > smooth_window + 1:
            runs[label] = p

# Randomized SVD runs
for k in [16, 32, 64]:
    p = os.path.join(BASE, f"rsvd_k{k}_d20", "train_log.jsonl")
    if os.path.exists(p):
        data = load_jsonl(p)
        if len(data) > smooth_window + 1:
            runs[f"Rand SVD k={k}"] = p
        else:
            print(f"  Skipping rsvd_k{k}: only {len(data)} steps")

colors = {
    "Baseline Muon (2 GPU)": "#555555",
    "Full SVD SoftmaxMuon eta=0.5": "#9467bd",
    "Rand SVD k=16": "#d62728",
    "Rand SVD k=32": "#2ca02c",
    "Rand SVD k=64": "#1f77b4",
}
linestyles = {
    "Baseline Muon (2 GPU)": "-",
    "Full SVD SoftmaxMuon eta=0.5": "--",
    "Rand SVD k=16": "-.",
    "Rand SVD k=32": "-",
    "Rand SVD k=64": "--",
}
linewidths = {
    "Baseline Muon (2 GPU)": 2.5,
    "Full SVD SoftmaxMuon eta=0.5": 1.8,
    "Rand SVD k=16": 1.8,
    "Rand SVD k=32": 2.0,
    "Rand SVD k=64": 1.8,
}

print(f"Randomized SVD runs: {list(runs.keys())}")

if runs:
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(16, 6))

    caption_lines = []
    for name, path in runs.items():
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

        c = colors.get(name, "#333")
        ls = linestyles.get(name, "-")
        lw = linewidths.get(name, 1.5)
        is_ref = "Baseline" in name or "Full SVD" in name
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

    fig1.suptitle("Randomized SVD SoftmaxMuon (eta=0.5) — k Sweep", fontsize=14, y=1.01)
    fig1.tight_layout()
    out1 = os.path.join(OUT_DIR, "rsvd_loss_comparison.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")

# =========================================================================
# Figure 2: Per-step wall time bar chart
# =========================================================================
timing_data = {}
for label, dirname in [
    ("Baseline Muon\n(2 GPU)", "cached_svd_baseline_d20"),
    ("Full SVD\nSoftmax", "softmax_v2_eta0.5_d20"),
    ("Rand SVD\nk=16", "rsvd_k16_d20"),
    ("Rand SVD\nk=32", "rsvd_k32_d20"),
    ("Rand SVD\nk=64", "rsvd_k64_d20"),
]:
    p = os.path.join(BASE, dirname, "train_log.jsonl")
    if os.path.exists(p):
        data = load_jsonl(p)
        if len(data) > 20:
            dts = [d["dt"] for d in data[11:]]
            timing_data[label] = np.mean(dts) * 1000

if timing_data:
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    bar_colors = {
        "Baseline Muon\n(2 GPU)": "#555555",
        "Full SVD\nSoftmax": "#9467bd",
        "Rand SVD\nk=16": "#d62728",
        "Rand SVD\nk=32": "#2ca02c",
        "Rand SVD\nk=64": "#1f77b4",
    }

    names = list(timing_data.keys())
    times = list(timing_data.values())
    cols = [bar_colors.get(n, "#333") for n in names]

    bars = ax2.bar(names, times, color=cols, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f"{t:.0f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Add speedup annotation relative to full SVD
    if "Full SVD\nSoftmax" in timing_data:
        svd_time = timing_data["Full SVD\nSoftmax"]
        for bar, (n, t) in zip(bars, zip(names, times)):
            if "Rand" in n:
                speedup = svd_time / t
                color = "white" if t < svd_time * 0.9 else "black"
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                         f"{speedup:.2f}x vs\nfull SVD", ha="center", va="center",
                         fontsize=8, color=color, fontweight="bold")

    # Mark 20% overhead threshold relative to baseline
    if "Baseline Muon\n(2 GPU)" in timing_data:
        baseline_time = timing_data["Baseline Muon\n(2 GPU)"]
        threshold = baseline_time * 1.2
        ax2.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
                     label=f"20% overhead threshold ({threshold:.0f}ms)")
        ax2.legend(fontsize=9)

    ax2.set_ylabel("Average Step Time (ms)", fontsize=12)
    ax2.set_title("Per-Step Wall Time — Randomized SVD vs Baselines", fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    fig2.tight_layout()
    out2 = os.path.join(OUT_DIR, "rsvd_step_time.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")

# =========================================================================
# Figure 3: Validation BPB comparison
# =========================================================================
val_runs = {}
for label, dirname in [
    ("Baseline Muon (2 GPU)", "cached_svd_baseline_d20"),
    ("Full SVD SoftmaxMuon eta=0.5", "softmax_v2_eta0.5_d20"),
    ("Rand SVD k=16", "rsvd_k16_d20"),
    ("Rand SVD k=32", "rsvd_k32_d20"),
    ("Rand SVD k=64", "rsvd_k64_d20"),
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
    "Full SVD SoftmaxMuon eta=0.5": "#9467bd",
    "Rand SVD k=16": "#d62728",
    "Rand SVD k=32": "#2ca02c",
    "Rand SVD k=64": "#1f77b4",
}

if val_runs:
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

    for name, rd in val_runs.items():
        vd = rd["val"]
        vsteps = [d["step"] for d in vd]
        vbpbs = [d["val_bpb"] for d in vd]
        vtimes = [d["total_training_time"] / 60.0 for d in vd]

        c = val_colors.get(name, "#333")
        is_ref = "Baseline" in name or "Full SVD" in name
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

    fig3.suptitle("Validation BPB — Randomized SVD vs Baselines (2 GPU)", fontsize=14, y=1.01)
    fig3.tight_layout()
    out3 = os.path.join(OUT_DIR, "rsvd_val_bpb.png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved {out3}")

# =========================================================================
# Summary table
# =========================================================================
print("\n" + "=" * 90)
print("SUMMARY: Randomized SVD Experiments (2 GPU)")
print("=" * 90)
print(f"{'Run':<35} {'Steps':>6} {'Avg dt (ms)':>12} {'Final Loss':>12} {'Final Val BPB':>15}")
print("-" * 82)

all_summaries = [
    ("Baseline Muon (2 GPU)", "cached_svd_baseline_d20"),
    ("Full SVD SoftmaxMuon eta=0.5", "softmax_v2_eta0.5_d20"),
    ("Rand SVD k=16", "rsvd_k16_d20"),
    ("Rand SVD k=32", "rsvd_k32_d20"),
    ("Rand SVD k=64", "rsvd_k64_d20"),
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
    print(f"{label:<35} {len(tdata):>6} {avg_dt:>12.0f} {final_loss:>12.4f} {val_str:>15}")

print("=" * 90)
