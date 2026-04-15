"""Plot softmax v2 sweep results: training loss + spectral diagnostics."""
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

# Runs to plot — v2 (tau = min(m,n) fix)
# Use 2-GPU baseline to match softmax v2 / Taylor runs (all 2-GPU)
runs = {"Baseline Muon": f"{BASE}/wallclock_baseline_d20"}
for eta in ["0.01", "0.1", "0.3", "0.5", "0.75", "1.0", "5.0", "20.0"]:
    d = f"{BASE}/softmax_v2_eta{eta}_d20"
    if os.path.isdir(d):
        runs[f"eta={eta}"] = d

# Taylor eta=0.01 full run (ran to completion, complements incomplete softmax_v2 eta=0.01)
taylor001_dir = f"{BASE}/taylor_eta0.01_full_d20"
if os.path.isdir(taylor001_dir):
    runs["Taylor eta=0.01"] = taylor001_dir

colors = {
    "Baseline Muon": "#bbbbbb",
    "eta=0.01": "#1f77b4",
    "eta=0.1": "#e6550d",
    "eta=0.3": "#17becf",
    "eta=0.5": "#d4a017",
    "eta=0.75": "#8c564b",
    "eta=1.0": "#2ca02c",
    "eta=5.0": "#d62728",
    "eta=20.0": "#7b2d8e",
    "Taylor eta=0.01": "#ff69b4",
}
linewidths = {
    "Baseline Muon": 3.0,
    "eta=0.01": 1.8,
    "eta=0.1": 1.8,
    "eta=0.3": 1.8,
    "eta=0.5": 1.8,
    "eta=0.75": 1.8,
    "eta=1.0": 1.8,
    "eta=5.0": 1.8,
    "eta=20.0": 1.8,
    "Taylor eta=0.01": 2.0,
}
linestyles = {
    "Baseline Muon": "-",
    "eta=0.01": "-",
    "eta=0.1": "--",
    "eta=0.3": (0, (1, 1)),
    "eta=0.5": (0, (5, 2)),
    "eta=0.75": (0, (3, 2, 1, 2)),
    "eta=1.0": "-.",
    "eta=5.0": (0, (3, 1, 1, 1)),
    "eta=20.0": ":",
    "Taylor eta=0.01": (0, (5, 1)),
}

print(f"Found runs: {list(runs.keys())}")

# =========================================================================
# Figure 1: Training loss vs step
# =========================================================================
fig1, (ax1_log, ax1_lin) = plt.subplots(1, 2, figsize=(18, 6))
smooth_window = 50

# First pass: load all smoothed data, keep baseline for step-matching
all_smoothed = {}  # name -> (s_steps, smoothed, data_len)
for name, path in runs.items():
    train_path = os.path.join(path, "train_log.jsonl")
    if not os.path.exists(train_path):
        continue
    data = load_jsonl(train_path)
    if len(data) < 10:
        print(f"  Skipping {name}: only {len(data)} train points")
        continue
    steps = np.array([d["step"] for d in data])
    losses = np.array([d["loss"] for d in data])
    if len(losses) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(losses, kernel, mode="valid")
        s_steps = steps[smooth_window - 1:]
    else:
        smoothed = losses
        s_steps = steps
    all_smoothed[name] = (s_steps, smoothed, len(data))

# Find the max step among softmax v2 eta runs (exclude baseline & Taylor eta=0.01 which run longer)
non_baseline_max_step = 0
for name, (s_steps, smoothed, _) in all_smoothed.items():
    if "Baseline" not in name and "Taylor" not in name:
        non_baseline_max_step = max(non_baseline_max_step, int(s_steps[-1]))

# Plot and collect end-of-run losses for caption
caption_lines = []
for name, (s_steps, smoothed, data_len) in all_smoothed.items():
    lw = linewidths.get(name, 1.5)
    ls = linestyles.get(name, "-")
    label = f"{name} ({data_len} steps)"
    color = colors.get(name)
    is_baseline = "Baseline" in name
    zorder = 1 if is_baseline else 2
    for ax in (ax1_log, ax1_lin):
        ax.plot(s_steps, smoothed, label=label, color=color, linewidth=lw, linestyle=ls, alpha=0.7 if is_baseline else 1.0, zorder=zorder)

    runs_longer = is_baseline or "Taylor" in name
    if runs_longer and non_baseline_max_step > 0:
        # Report at the step where the shorter softmax v2 runs end
        idx = np.searchsorted(s_steps, non_baseline_max_step, side="right") - 1
        idx = max(0, min(idx, len(smoothed) - 1))
        caption_lines.append(f"{name} @ step {int(s_steps[idx])}: {smoothed[idx]:.3f}")
    else:
        caption_lines.append(f"{name} @ step {int(s_steps[-1])}: {smoothed[-1]:.3f}")

# Sort caption lines by loss value (lowest first)
caption_lines.sort(key=lambda s: float(s.split(": ")[-1]))
caption = "Loss at latest step:\n" + "\n".join(caption_lines)
for ax in (ax1_log, ax1_lin):
    ax.text(0.98, 0.98, caption, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"))

ax1_log.set_xscale("log")
ax1_log.set_yscale("log")
ax1_log.set_xlabel("Training Step", fontsize=12)
ax1_log.set_ylabel("Training Loss (smoothed)", fontsize=12)
ax1_log.set_title("Log-Log Scale", fontsize=13)
ax1_log.legend(fontsize=9)
ax1_log.grid(True, alpha=0.3, which="both")

ax1_lin.set_xlabel("Training Step", fontsize=12)
ax1_lin.set_ylabel("Training Loss (smoothed)", fontsize=12)
ax1_lin.set_title("Linear Scale", fontsize=13)
ax1_lin.legend(fontsize=9)
ax1_lin.grid(True, alpha=0.3)

fig1.suptitle("SoftmaxMuon v2 Eta Sweep — Training Loss", fontsize=14, y=1.01)

fig1.tight_layout()
out1 = os.path.join(OUT_DIR, "softmax_v2_train_loss.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved {out1}")

# =========================================================================
# Figure 2: Spectral diagnostics (entropy + nuclear norm) vs step
# =========================================================================
# Collect runs that have spectral logs
spectral_runs = {}
for name, path in runs.items():
    sp = os.path.join(path, "spectral_log.jsonl")
    if os.path.exists(sp):
        data = load_jsonl(sp)
        if len(data) >= 2:
            spectral_runs[name] = data

# Baseline doesn't have spectral_log; derive from weight SVD logs
baseline_svd_dir = os.path.join(BASE, "baseline_muon", "svd_logs")
if "Baseline Muon" not in spectral_runs and os.path.isdir(baseline_svd_dir):
    baseline_spectral = []
    for fname in sorted(os.listdir(baseline_svd_dir)):
        if not fname.startswith("weight_svd") or not fname.endswith(".json"):
            continue
        step = int(fname.split("step")[1].split(".")[0])
        with open(os.path.join(baseline_svd_dir, fname)) as f:
            svd_data = json.load(f)
        spectra = svd_data.get("spectra", {})
        for layer_name, layer_data in spectra.items():
            if "h.0.mlp.c_fc" in layer_name:
                sv = np.array(layer_data.get("singular_values", []))
                nuclear = layer_data.get("nuclear_norm", sv.sum())
                if len(sv) > 0:
                    p_sv = sv / (sv.sum() + 1e-8)
                    entropy = -(p_sv * np.log(p_sv + 1e-8)).sum()
                    baseline_spectral.append({"step": step, "sv_entropy": float(entropy), "nuclear_norm": float(nuclear)})
                break
    if baseline_spectral:
        spectral_runs["Baseline Muon"] = baseline_spectral
        print(f"  Baseline spectral: {len(baseline_spectral)} points (entropy from top-64 SVs only)")

if spectral_runs:
    fig2, (ax_ent, ax_nn) = plt.subplots(1, 2, figsize=(14, 5))

    for name, data in spectral_runs.items():
        steps = [d["step"] for d in data]
        c = colors.get(name)
        is_baseline = "Baseline" in name
        lw = linewidths.get(name, 1.5)
        ls = linestyles.get(name, "-")

        if "sv_entropy" in data[0] and "Baseline" not in name:
            ent = [d["sv_entropy"] for d in data]
            ax_ent.plot(steps, ent, label=name, color=c, linewidth=lw, linestyle=ls)

        if "nuclear_norm" in data[0]:
            nn = [d["nuclear_norm"] for d in data]
            alpha = 0.7 if is_baseline else 1.0
            zorder = 1 if is_baseline else 2
            ax_nn.plot(steps, nn, label=name, color=c, linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder)

    ax_ent.set_xlabel("Training Step")
    ax_ent.set_ylabel("SV Entropy")
    ax_ent.set_title("Singular Value Entropy of W\n(lower = more concentrated spectrum)")
    ax_ent.legend(fontsize=9)
    ax_ent.grid(True, alpha=0.3)

    ax_nn.set_xlabel("Training Step")
    ax_nn.set_ylabel("Nuclear Norm")
    ax_nn.set_title("Nuclear Norm of W\n(MLP layer 0 up-proj)")
    ax_nn.legend(fontsize=9)
    ax_nn.grid(True, alpha=0.3)

    fig2.suptitle("SoftmaxMuon v2 — Spectral Diagnostics", fontsize=14, y=1.02)
    fig2.tight_layout()
    out2 = os.path.join(OUT_DIR, "softmax_v2_spectral.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")
else:
    print("No spectral logs found, skipping figure 2")

# =========================================================================
# Figure 3: Loss vs eta scatter (log-log)
# =========================================================================
smooth_window_eta = 50
eta_points = []

# Find max step among softmax runs
max_softmax_step_eta = 0
for eta_str in ["0.1", "0.3", "0.5", "0.75", "1.0", "5.0", "20.0"]:
    p = os.path.join(BASE, f"softmax_v2_eta{eta_str}_d20", "train_log.jsonl")
    if os.path.exists(p):
        d = load_jsonl(p)
        if d:
            max_softmax_step_eta = max(max_softmax_step_eta, d[-1]["step"])

# Baseline = eta=0 (2-GPU to match softmax v2 / Taylor runs)
baseline_path = os.path.join(BASE, "wallclock_baseline_d20", "train_log.jsonl")
if os.path.exists(baseline_path) and max_softmax_step_eta > 0:
    data = load_jsonl(baseline_path)
    losses_arr = np.array([d["loss"] for d in data])
    steps_arr = np.array([d["step"] for d in data])
    if len(losses_arr) > smooth_window_eta:
        kernel = np.ones(smooth_window_eta) / smooth_window_eta
        smoothed = np.convolve(losses_arr, kernel, mode="valid")
        s_steps = steps_arr[smooth_window_eta - 1:]
        idx = np.searchsorted(s_steps, max_softmax_step_eta, side="right") - 1
        idx = max(0, min(idx, len(smoothed) - 1))
        eta_points.append((0, smoothed[idx], int(s_steps[idx])))

for eta_str in ["0.1", "0.3", "0.5", "0.75", "1.0", "5.0", "20.0"]:
    p = os.path.join(BASE, f"softmax_v2_eta{eta_str}_d20", "train_log.jsonl")
    if not os.path.exists(p):
        continue
    data = load_jsonl(p)
    if len(data) < smooth_window_eta + 1:
        continue
    losses_arr = np.array([d["loss"] for d in data])
    steps_arr = np.array([d["step"] for d in data])
    kernel = np.ones(smooth_window_eta) / smooth_window_eta
    smoothed = np.convolve(losses_arr, kernel, mode="valid")
    eta_points.append((float(eta_str), smoothed[-1], int(steps_arr[-1])))

# Taylor eta=0.01 full run — evaluate at same step as other runs
taylor001_train = os.path.join(BASE, "taylor_eta0.01_full_d20", "train_log.jsonl")
if os.path.exists(taylor001_train) and max_softmax_step_eta > 0:
    data = load_jsonl(taylor001_train)
    if len(data) > smooth_window_eta + 1:
        losses_arr = np.array([d["loss"] for d in data])
        steps_arr = np.array([d["step"] for d in data])
        kernel = np.ones(smooth_window_eta) / smooth_window_eta
        smoothed = np.convolve(losses_arr, kernel, mode="valid")
        s_steps = steps_arr[smooth_window_eta - 1:]
        idx = np.searchsorted(s_steps, max_softmax_step_eta, side="right") - 1
        idx = max(0, min(idx, len(smoothed) - 1))
        eta_points.append((0.01, smoothed[idx], int(s_steps[idx]), "Taylor eta=0.01"))

if eta_points:
    eta_points.sort()
    print("Eta vs loss:", [(f"eta={p[0]}", f"loss={p[1]:.4f}", f"step={p[2]}") for p in eta_points])

    fig3, ax3 = plt.subplots(figsize=(8, 5))

    # Separate baseline (eta=0) from the rest
    baseline_loss = None
    for p in eta_points:
        e, l, s = p[0], p[1], p[2]
        label = p[3] if len(p) > 3 else None
        if e == 0:
            baseline_loss = l
        else:
            color = "#ff69b4" if label and "Taylor" in label else None
            marker = "D" if label and "Taylor" in label else "o"
            ax3.scatter(e, l, s=80, zorder=5, color=color, marker=marker)
            tag = f"Taylor ({e}, {l:.3f})" if label and "Taylor" in label else f"({e}, {l:.3f})"
            ax3.annotate(tag, (e, l),
                         textcoords="offset points", xytext=(8, 8), fontsize=10)

    # Plot baseline on the y-axis
    if baseline_loss is not None:
        ax3.axhline(y=baseline_loss, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)
        # Place dot at left edge of plot
        ax3.scatter([ax3.get_xlim()[0] if ax3.get_xlim()[0] > 0 else 0.005], [baseline_loss],
                    s=100, color="#1f77b4", zorder=6, clip_on=False, marker="o")
        ax3.annotate(f"Baseline ({baseline_loss:.3f})", xy=(0.005, baseline_loss),
                     textcoords="offset points", xytext=(8, -15), fontsize=10)

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlim(0.005, 50)
    ax3.set_xlabel("eta", fontsize=12)
    ax3.set_ylabel("Smoothed Training Loss", fontsize=12)
    ax3.set_title("SoftmaxMuon v2 — Loss vs eta", fontsize=13)
    ax3.grid(True, alpha=0.3)

    fig3.tight_layout()
    out3 = os.path.join(OUT_DIR, "eta_vs_loss.png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved {out3}")

# =========================================================================
# Figure 4: Loss vs wall-clock time (baseline, SVD softmax, Taylor)
# =========================================================================
wallclock_runs = {}

# Baseline (2-GPU to match other runs)
bl_path = os.path.join(BASE, "wallclock_baseline_d20", "train_log.jsonl")
if os.path.exists(bl_path):
    wallclock_runs["Baseline Muon"] = bl_path

# SVD SoftmaxMuon v2
for eta_str in ["0.1", "0.3", "0.5", "1.0", "20.0"]:
    p = os.path.join(BASE, f"softmax_v2_eta{eta_str}_d20", "train_log.jsonl")
    if os.path.exists(p):
        wallclock_runs[f"SVD eta={eta_str}"] = p

# Taylor approximation
for eta_str in ["0.3", "0.5"]:
    p = os.path.join(BASE, f"taylor_eta{eta_str}_d20", "train_log.jsonl")
    if os.path.exists(p):
        wallclock_runs[f"Taylor eta={eta_str}"] = p

# WeightSoftmaxMuon
for eta_str in ["0.5"]:
    p = os.path.join(BASE, f"wsoftmax_eta{eta_str}_d20", "train_log.jsonl")
    if os.path.exists(p):
        wallclock_runs[f"W-Softmax eta={eta_str}"] = p

wc_colors = {
    "Baseline Muon": "#bbbbbb",
    "SVD eta=0.1": "#e6550d",
    "SVD eta=0.3": "#17becf",
    "SVD eta=0.5": "#d4a017",
    "SVD eta=1.0": "#2ca02c",
    "SVD eta=20.0": "#7b2d8e",
    "Taylor eta=0.3": "#e41a1c",
    "Taylor eta=0.5": "#ff7f00",
    "W-Softmax eta=0.5": "#9467bd",
}
wc_linestyles = {
    "Baseline Muon": "-",
    "SVD eta=0.1": "--",
    "SVD eta=0.3": "--",
    "SVD eta=0.5": "--",
    "SVD eta=1.0": "--",
    "SVD eta=20.0": "--",
    "Taylor eta=0.3": "-",
    "Taylor eta=0.5": "-",
    "W-Softmax eta=0.5": "-",
}

if wallclock_runs:
    fig4, ax4 = plt.subplots(figsize=(12, 6))

    for name, path in wallclock_runs.items():
        data = load_jsonl(path)
        if len(data) < smooth_window + 1:
            print(f"  Skipping {name} wallclock: only {len(data)} points")
            continue

        losses = np.array([d["loss"] for d in data])
        # Compute cumulative wall-clock time from dt (seconds -> minutes)
        times_min = np.cumsum([d["dt"] for d in data]) / 60.0

        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(losses, kernel, mode="valid")
        s_times = times_min[smooth_window - 1:]

        is_baseline = "Baseline" in name
        c = wc_colors.get(name, "#333333")
        ls = wc_linestyles.get(name, "-")
        lw = 3.0 if is_baseline else 1.8
        alpha = 0.7 if is_baseline else 1.0
        zorder = 1 if is_baseline else 2

        ax4.plot(s_times, smoothed, label=name, color=c, linewidth=lw,
                 linestyle=ls, alpha=alpha, zorder=zorder)

    ax4.set_xlabel("Wall-Clock Time (minutes)", fontsize=12)
    ax4.set_ylabel("Training Loss (smoothed)", fontsize=12)
    ax4.set_title("Loss vs Wall-Clock Time — Baseline / SVD SoftmaxMuon / Taylor Approx", fontsize=13)
    ax4.legend(fontsize=9, loc="upper right")
    ax4.grid(True, alpha=0.3)

    fig4.tight_layout()
    out4 = os.path.join(OUT_DIR, "loss_vs_wallclock.png")
    fig4.savefig(out4, dpi=150, bbox_inches="tight")
    print(f"Saved {out4}")

# =========================================================================
# Figure 5: Loss vs wall-clock — same-conditions comparison
# (wallclock_baseline_d20 vs wallclock_taylor_eta0.5_d20, same node/GPUs)
# =========================================================================
same_cond_runs = {}
bl_wc = os.path.join(BASE, "wallclock_baseline_d20", "train_log.jsonl")
if os.path.exists(bl_wc):
    same_cond_runs["Baseline Muon (2 GPU)"] = bl_wc
tay_wc = os.path.join(BASE, "wallclock_taylor_eta0.5_d20", "train_log.jsonl")
if os.path.exists(tay_wc):
    same_cond_runs["TaylorMuon eta=0.5 (2 GPU)"] = tay_wc
wsm_wc = os.path.join(BASE, "wsoftmax_eta0.5_d20", "train_log.jsonl")
if os.path.exists(wsm_wc):
    same_cond_runs["W-SoftmaxMuon eta=0.5 (2 GPU)"] = wsm_wc
wsm005_wc = os.path.join(BASE, "wsoftmax_eta0.05_d20", "train_log.jsonl")
if os.path.exists(wsm005_wc):
    same_cond_runs["W-SoftmaxMuon eta=0.05 (2 GPU)"] = wsm005_wc
# Original baseline (4 GPU, different conditions) for reference
bl_4gpu = os.path.join(BASE, "baseline_muon", "train_log.jsonl")
if os.path.exists(bl_4gpu):
    same_cond_runs["Baseline Muon (4 GPU, ref)"] = bl_4gpu

sc_colors = {
    "Baseline Muon (2 GPU)": "#555555",
    "TaylorMuon eta=0.5 (2 GPU)": "#e6550d",
    "W-SoftmaxMuon eta=0.5 (2 GPU)": "#9467bd",
    "W-SoftmaxMuon eta=0.05 (2 GPU)": "#d62728",
    "Baseline Muon (4 GPU, ref)": "#bbbbbb",
}
sc_linestyles = {
    "Baseline Muon (2 GPU)": "-",
    "TaylorMuon eta=0.5 (2 GPU)": "-",
    "W-SoftmaxMuon eta=0.5 (2 GPU)": "-",
    "W-SoftmaxMuon eta=0.05 (2 GPU)": "-.",
    "Baseline Muon (4 GPU, ref)": "-",
}
sc_linewidths = {
    "Baseline Muon (2 GPU)": 2.0,
    "TaylorMuon eta=0.5 (2 GPU)": 2.0,
    "W-SoftmaxMuon eta=0.5 (2 GPU)": 2.0,
    "W-SoftmaxMuon eta=0.05 (2 GPU)": 2.0,
    "Baseline Muon (4 GPU, ref)": 1.5,
}
sc_alphas = {
    "Baseline Muon (2 GPU)": 1.0,
    "TaylorMuon eta=0.5 (2 GPU)": 1.0,
    "W-SoftmaxMuon eta=0.5 (2 GPU)": 1.0,
    "W-SoftmaxMuon eta=0.05 (2 GPU)": 1.0,
    "Baseline Muon (4 GPU, ref)": 0.5,
}
sc_zorders = {
    "Baseline Muon (2 GPU)": 3,
    "TaylorMuon eta=0.5 (2 GPU)": 2,
    "W-SoftmaxMuon eta=0.5 (2 GPU)": 2,
    "W-SoftmaxMuon eta=0.05 (2 GPU)": 2,
    "Baseline Muon (4 GPU, ref)": 1,
}

if same_cond_runs:
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, 6))

    # First pass: load and plot all runs, store smoothed data for cross-referencing
    sc_data = {}  # name -> (s_times, s_steps, smoothed, avg_dt, data_len)
    for name, path in same_cond_runs.items():
        data = load_jsonl(path)
        if len(data) < smooth_window + 1:
            print(f"  Skipping {name} same-cond: only {len(data)} points")
            continue

        losses = np.array([d["loss"] for d in data])
        steps = np.array([d["step"] for d in data])
        times_min = np.cumsum([d["dt"] for d in data]) / 60.0

        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(losses, kernel, mode="valid")
        s_times = times_min[smooth_window - 1:]
        s_steps = steps[smooth_window - 1:]
        avg_dt = times_min[-1] / len(data)

        sc_data[name] = (s_times, s_steps, smoothed, avg_dt, len(data))

        c = sc_colors.get(name, "#333")
        ls = sc_linestyles.get(name, "-")
        lw = sc_linewidths.get(name, 2.0)
        alpha = sc_alphas.get(name, 1.0)
        zorder = sc_zorders.get(name, 2)

        # Left: loss vs wall-clock time
        ax5a.plot(s_times, smoothed, label=name, color=c, linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder)
        # Right: loss vs step
        ax5b.plot(s_steps, smoothed, label=name, color=c, linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder)

    # Second pass: annotate
    bl_key = "Baseline Muon (2 GPU)"
    tay_key = "TaylorMuon eta=0.5 (2 GPU)"
    bl_end_time = None
    bl_end_step = None

    # Build caption text block and place in upper-right empty space
    caption_lines_5 = []

    if bl_key in sc_data:
        s_times_bl, s_steps_bl, sm_bl, avg_dt_bl, n_bl = sc_data[bl_key]
        bl_end_time = s_times_bl[-1]
        bl_end_step = int(s_steps_bl[-1])
        caption_lines_5.append(f"Baseline (2 GPU) @ {bl_end_time:.0f}m:")
        caption_lines_5.append(f"  loss={sm_bl[-1]:.3f}, step {bl_end_step}, {avg_dt_bl*60:.2f}s/step")

    if tay_key in sc_data:
        s_times_tay, s_steps_tay, sm_tay, avg_dt_tay, n_tay = sc_data[tay_key]

        # Taylor at baseline's wall-clock time
        if bl_end_time is not None and bl_end_time <= s_times_tay[-1]:
            idx_at_bl_time = np.searchsorted(s_times_tay, bl_end_time, side="right") - 1
            idx_at_bl_time = max(0, min(idx_at_bl_time, len(sm_tay) - 1))
            tay_loss_at_bl = sm_tay[idx_at_bl_time]
            tay_step_at_bl = int(s_steps_tay[idx_at_bl_time])
            caption_lines_5.append(f"Taylor (2 GPU) @ {bl_end_time:.0f}m:")
            caption_lines_5.append(f"  loss={tay_loss_at_bl:.3f}, step {tay_step_at_bl}, {avg_dt_tay*60:.2f}s/step")

        # Taylor at its own endpoint
        caption_lines_5.append(f"Taylor final @ {s_times_tay[-1]:.0f}m:")
        caption_lines_5.append(f"  loss={sm_tay[-1]:.3f}, step {int(s_steps_tay[-1])}, {avg_dt_tay*60:.2f}s/step")

        # Baseline at Taylor's final wall-clock time
        if bl_key in sc_data:
            tay_end_time = s_times_tay[-1]
            if tay_end_time <= s_times_bl[-1]:
                idx_bl_at_tay = np.searchsorted(s_times_bl, tay_end_time, side="right") - 1
                idx_bl_at_tay = max(0, min(idx_bl_at_tay, len(sm_bl) - 1))
                caption_lines_5.append(f"Baseline (2 GPU) @ {tay_end_time:.0f}m:")
                caption_lines_5.append(f"  loss={sm_bl[idx_bl_at_tay]:.3f}, step {int(s_steps_bl[idx_bl_at_tay])}")

    caption_5 = "\n".join(caption_lines_5)
    ax5a.text(0.97, 0.97, caption_5, transform=ax5a.transAxes, fontsize=8.5,
              verticalalignment="top", horizontalalignment="right", family="monospace",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#cccccc"))

    ax5a.set_xlabel("Wall-Clock Time (minutes)", fontsize=12)
    ax5a.set_ylabel("Training Loss (smoothed)", fontsize=12)
    ax5a.set_title("Loss vs Wall-Clock Time\n(same node, same GPUs, sequential)", fontsize=13)
    ax5a.legend(fontsize=10)
    ax5a.grid(True, alpha=0.3)

    ax5b.set_xlabel("Training Step", fontsize=12)
    ax5b.set_ylabel("Training Loss (smoothed)", fontsize=12)
    ax5b.set_title("Loss vs Step\n(same conditions)", fontsize=13)
    ax5b.legend(fontsize=10)
    ax5b.grid(True, alpha=0.3)

    fig5.suptitle("Baseline Muon vs TaylorMuon — Same-Conditions Comparison", fontsize=14, y=1.01)
    fig5.tight_layout()
    out5 = os.path.join(OUT_DIR, "loss_vs_time_same_conditions.png")
    fig5.savefig(out5, dpi=150, bbox_inches="tight")
    print(f"Saved {out5}")
else:
    print("No wallclock comparison data found yet (waiting for job 561072)")

# =========================================================================
# Figure 6: Validation BPB comparison
# (taylor_eta0.5_d20 vs wallclock_baseline_d20 vs wallclock_taylor_eta0.5_d20)
# =========================================================================
val_runs = {
    "Baseline Muon (2 GPU)": f"{BASE}/wallclock_baseline_d20",
    "TaylorMuon eta=0.5 (2 GPU)": f"{BASE}/taylor_eta0.5_d20",
    "SoftmaxMuon eta=0.5 (2 GPU)": f"{BASE}/softmax_v2_eta0.5_d20",
    "W-SoftmaxMuon eta=0.5 (2 GPU)": f"{BASE}/wsoftmax_eta0.5_d20",
    "W-SoftmaxMuon eta=0.05 (2 GPU)": f"{BASE}/wsoftmax_eta0.05_d20",
}

val_colors = {
    "Baseline Muon (2 GPU)": "#555555",
    "TaylorMuon eta=0.5 (2 GPU)": "#e6550d",
    "SoftmaxMuon eta=0.5 (2 GPU)": "#1b9e77",
    "W-SoftmaxMuon eta=0.5 (2 GPU)": "#9467bd",
    "W-SoftmaxMuon eta=0.05 (2 GPU)": "#d62728",
}
val_linestyles = {
    "Baseline Muon (2 GPU)": "-",
    "TaylorMuon eta=0.5 (2 GPU)": "-",
    "SoftmaxMuon eta=0.5 (2 GPU)": "-",
    "W-SoftmaxMuon eta=0.5 (2 GPU)": "-",
    "W-SoftmaxMuon eta=0.05 (2 GPU)": "-.",
}
val_linewidths = {
    "Baseline Muon (2 GPU)": 2.5,
    "TaylorMuon eta=0.5 (2 GPU)": 2.0,
    "SoftmaxMuon eta=0.5 (2 GPU)": 2.0,
    "W-SoftmaxMuon eta=0.5 (2 GPU)": 2.0,
    "W-SoftmaxMuon eta=0.05 (2 GPU)": 2.0,
}

# Load val and train data for these runs
val_log_data = {}
train_log_data = {}
for vname, vpath in val_runs.items():
    vp = os.path.join(vpath, "val_log.jsonl")
    tp = os.path.join(vpath, "train_log.jsonl")
    if os.path.exists(vp):
        val_log_data[vname] = load_jsonl(vp)
        print(f"{vname}: {len(val_log_data[vname])} val points")
    if os.path.exists(tp):
        train_log_data[vname] = load_jsonl(tp)
        print(f"{vname}: {len(train_log_data[vname])} train points")

if val_log_data:
    fig6, axes6 = plt.subplots(2, 2, figsize=(16, 12))
    ax_vs, ax_vt = axes6[0]  # val vs step, val vs time
    ax_ts, ax_tt = axes6[1]  # train vs step, train vs time

    for vname in val_runs:
        c = val_colors[vname]
        ls = val_linestyles[vname]
        lw = val_linewidths[vname]

        # Val bpb
        if vname in val_log_data and len(val_log_data[vname]) >= 2:
            vd = val_log_data[vname]
            vsteps = [d["step"] for d in vd]
            vbpbs = [d["val_bpb"] for d in vd]
            vtimes = [d["total_training_time"] / 60.0 for d in vd]

            ax_vs.plot(vsteps, vbpbs, marker="o", markersize=4,
                       label=f"{vname} (final={vbpbs[-1]:.4f})",
                       color=c, linestyle=ls, linewidth=lw)
            ax_vt.plot(vtimes, vbpbs, marker="o", markersize=4,
                       label=f"{vname} (final={vbpbs[-1]:.4f})",
                       color=c, linestyle=ls, linewidth=lw)

        # Train loss (smoothed)
        if vname in train_log_data and len(train_log_data[vname]) > smooth_window:
            td = train_log_data[vname]
            tsteps = np.array([d["step"] for d in td])
            tlosses = np.array([d["loss"] for d in td])
            ttimes = np.cumsum([d["dt"] for d in td]) / 60.0

            kernel = np.ones(smooth_window) / smooth_window
            tsmoothed = np.convolve(tlosses, kernel, mode="valid")
            ts_steps = tsteps[smooth_window - 1:]
            ts_times = ttimes[smooth_window - 1:]

            ax_ts.plot(ts_steps, tsmoothed, label=f"{vname} (final={tsmoothed[-1]:.4f})",
                       color=c, linestyle=ls, linewidth=lw)
            ax_tt.plot(ts_times, tsmoothed, label=f"{vname} (final={tsmoothed[-1]:.4f})",
                       color=c, linestyle=ls, linewidth=lw)

    # Annotate final val bpb for each run + baseline at step 2500
    annot_lines_6 = []
    for vname in val_runs:
        if vname in val_log_data and val_log_data[vname]:
            vd = val_log_data[vname]
            annot_lines_6.append(f"{vname}: {vd[-1]['val_bpb']:.4f} @ step {vd[-1]['step']}")
            # Also show val bpb at step 2500 for fair comparison
            for d in vd:
                if d["step"] == 2500:
                    annot_lines_6.append(f"  (@ step 2500: {d['val_bpb']:.4f})")
                    break
    if annot_lines_6:
        caption_6 = "Final val bpb:\n" + "\n".join(annot_lines_6)
        ax_vs.text(0.97, 0.97, caption_6, transform=ax_vs.transAxes, fontsize=9,
                   verticalalignment="top", horizontalalignment="right", family="monospace",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#cccccc"))

    for ax, xlabel, title in [
        (ax_vs, "Training Step", "Validation BPB vs Step"),
        (ax_vt, "Wall-Clock Time (minutes)", "Validation BPB vs Wall-Clock Time"),
        (ax_ts, "Training Step", "Training Loss vs Step (smoothed)"),
        (ax_tt, "Wall-Clock Time (minutes)", "Training Loss vs Wall-Clock Time (smoothed)"),
    ]:
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("BPB" if "Val" in title else "Loss", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig6.suptitle("Validation & Training Loss — Taylor eta=0.5 vs Baseline Muon", fontsize=14, y=1.01)
    fig6.tight_layout()
    out6 = os.path.join(OUT_DIR, "val_bpb_comparison.png")
    fig6.savefig(out6, dpi=150, bbox_inches="tight")
    print(f"Saved {out6}")
else:
    print("No validation log data found for comparison plot")
