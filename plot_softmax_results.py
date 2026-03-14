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
runs = {"Baseline Muon": f"{BASE}/baseline_muon"}
for eta in ["0.01", "0.1", "1.0", "5.0", "20.0"]:
    d = f"{BASE}/softmax_v2_eta{eta}_d20"
    if os.path.isdir(d):
        runs[f"eta={eta}"] = d

colors = {
    "Baseline Muon": "#bbbbbb",
    "eta=0.01": "#1f77b4",
    "eta=0.1": "#e6550d",
    "eta=1.0": "#2ca02c",
    "eta=5.0": "#d62728",
    "eta=20.0": "#7b2d8e",
}
linewidths = {
    "Baseline Muon": 3.0,
    "eta=0.01": 1.8,
    "eta=0.1": 1.8,
    "eta=1.0": 1.8,
    "eta=5.0": 1.8,
    "eta=20.0": 1.8,
}
linestyles = {
    "Baseline Muon": "-",
    "eta=0.01": "-",
    "eta=0.1": "--",
    "eta=1.0": "-.",
    "eta=5.0": (0, (3, 1, 1, 1)),
    "eta=20.0": ":",
}

print(f"Found runs: {list(runs.keys())}")

# =========================================================================
# Figure 1: Training loss vs step
# =========================================================================
fig1, (ax1_log, ax1_lin) = plt.subplots(1, 2, figsize=(18, 6))
smooth_window = 50

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
    # Smooth with rolling mean
    if len(losses) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(losses, kernel, mode="valid")
        s_steps = steps[smooth_window - 1:]
    else:
        smoothed = losses
        s_steps = steps
    lw = linewidths.get(name, 1.5)
    ls = linestyles.get(name, "-")
    label = f"{name} ({len(data)} steps)"
    color = colors.get(name)
    is_baseline = "Baseline" in name
    zorder = 1 if is_baseline else 2
    for ax in (ax1_log, ax1_lin):
        ax.plot(s_steps, smoothed, label=label, color=color, linewidth=lw, linestyle=ls, alpha=0.7 if is_baseline else 1.0, zorder=zorder)

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
            ax_nn.plot(steps, nn, label=name, color=c, linewidth=lw, linestyle=ls)

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
