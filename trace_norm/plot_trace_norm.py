"""
Plot comparison of trace/norm-constrained Muon variants.

4 subplots:
1. Training loss vs step
2. Validation loss vs step
3. Spectral norm over training for one representative layer
4. SV entropy over training

Includes old baseline Muon from cache/base_checkpoints/baseline_muon/
and new variants from cache/trace_norm/tn_{fro,nuclear,fixed-fro}_d20/
"""

import os
import json
import glob
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
TRACE_BASE = os.path.join(ROOT, "cache", "trace_norm")
OLD_BASELINE = os.path.join(ROOT, "cache", "base_checkpoints", "baseline_muon")

def _tn(dirname):
    """Helper to build paths for a trace_norm variant."""
    d = os.path.join(TRACE_BASE, dirname)
    return {
        "train_log": os.path.join(d, "train_log.jsonl"),
        "val_log": os.path.join(d, "val_log.jsonl"),
        "spectral_source": "spectral_jsonl",
        "spectral_log": os.path.join(d, "spectral_log.jsonl"),
    }

VARIANTS = [
    {"key": "baseline", "label": "Baseline Muon", "color": "#1f77b4",
     "train_log": os.path.join(OLD_BASELINE, "train_log.jsonl"),
     "val_log": os.path.join(OLD_BASELINE, "val_log.jsonl"),
     "spectral_source": "svd_dir",
     "svd_dir": os.path.join(OLD_BASELINE, "svd_logs")},
    {"key": "nuclear_every", "label": "8: Nuclear Every Step", "color": "#d62728",
     **_tn("tn_nuclear-every_d20")},
]

# Representative layer
REPR_LAYER = "transformer.h.10.mlp.c_fc.weight"
REPR_LAYER_SHORT = "L10.mlp.c_fc"


def load_jsonl(path):
    """Load JSONL file, skipping config lines."""
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_config" in obj:
                continue
            entries.append(obj)
    return entries


def load_svd_dir_spectral(svd_dir, layer_name):
    """Load spectral norm and SV entropy from old-format SVD log directory."""
    steps, spectral_norms, entropies = [], [], []
    for path in sorted(glob.glob(os.path.join(svd_dir, "weight_svd_step*.json"))):
        with open(path) as f:
            data = json.load(f)
        step = data["step"]
        spectra = data.get("spectra", {})
        if layer_name not in spectra:
            continue
        info = spectra[layer_name]
        steps.append(step)
        spectral_norms.append(info["spectral_norm"])
        # Compute entropy from singular values
        svs = np.array(info["singular_values"], dtype=np.float64)
        if svs.sum() > 0:
            p = svs / svs.sum()
            p = np.clip(p, 1e-10, None)
            entropy = -np.sum(p * np.log(p))
        else:
            entropy = 0.0
        entropies.append(entropy)
    return steps, spectral_norms, entropies


def load_spectral_jsonl(path, layer_name):
    """Load spectral norm and SV entropy from new-format spectral_log.jsonl."""
    entries = load_jsonl(path)
    steps, spectral_norms, entropies = [], [], []
    for entry in entries:
        norm_stats = entry.get("norm_stats", {})
        sv_entropy = entry.get("sv_entropy", {})
        if layer_name in norm_stats:
            steps.append(entry["step"])
            spectral_norms.append(norm_stats[layer_name]["spectral_norm"])
            entropies.append(sv_entropy.get(layer_name, float("nan")))
    return steps, spectral_norms, entropies


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for info in VARIANTS:
        key = info["key"]
        label = info["label"]
        color = info["color"]

        # ---------- Subplot 1: Training loss ----------
        train_data = load_jsonl(info["train_log"])
        if train_data:
            steps = [d["step"] for d in train_data]
            losses = [d["loss"] for d in train_data]
            axes[0, 0].plot(steps, losses, label=label, color=color, alpha=0.8, linewidth=1.0)
        else:
            print(f"  No train log for {key}")

        # ---------- Subplot 2: Validation loss ----------
        val_data = load_jsonl(info["val_log"])
        if val_data:
            steps = [d["step"] for d in val_data]
            bpbs = [d["val_bpb"] for d in val_data]
            axes[0, 1].plot(steps, bpbs, label=label, color=color, marker='o', markersize=3, alpha=0.8)
        else:
            print(f"  No val log for {key}")

        # ---------- Subplots 3 & 4: Spectral norm and SV entropy ----------
        if info["spectral_source"] == "svd_dir":
            s_steps, s_norms, s_ents = load_svd_dir_spectral(info["svd_dir"], REPR_LAYER)
        else:
            s_steps, s_norms, s_ents = load_spectral_jsonl(info["spectral_log"], REPR_LAYER)

        if s_steps:
            axes[1, 0].plot(s_steps, s_norms, label=label, color=color, marker='o', markersize=3, alpha=0.8)
            axes[1, 1].plot(s_steps, s_ents, label=label, color=color, marker='o', markersize=3, alpha=0.8)
        else:
            print(f"  No spectral data for {key}")

    # Annotate loss difference at last step of Option 8
    # Load baseline and nuclear_every training data
    baseline_train = load_jsonl(VARIANTS[0]["train_log"])
    nuc_train = load_jsonl(VARIANTS[1]["train_log"])
    if baseline_train and nuc_train:
        last_nuc_step = nuc_train[-1]["step"]
        last_nuc_loss = nuc_train[-1]["loss"]
        # Find closest baseline step (exact match or closest <=)
        baseline_at_step = None
        for d in baseline_train:
            if d["step"] <= last_nuc_step:
                baseline_at_step = d
        if baseline_at_step:
            bl_loss = baseline_at_step["loss"]
            diff = bl_loss - last_nuc_loss
            sign = "+" if diff > 0 else ""
            axes[0, 0].annotate(
                f"@ step {last_nuc_step}\n"
                f"Baseline: {bl_loss:.4f}\n"
                f"Nuclear:  {last_nuc_loss:.4f}\n"
                f"Δ = {sign}{diff:.4f}",
                xy=(last_nuc_step, last_nuc_loss),
                xytext=(last_nuc_step + 300, last_nuc_loss + 0.8),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
                fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
            )

    # Formatting
    for ax, title, ylabel in [
        (axes[0, 0], "Training Loss", "Training Loss (smoothed)"),
        (axes[0, 1], "Validation Loss", "Validation BPB"),
        (axes[1, 0], f"Spectral Norm ({REPR_LAYER_SHORT})", "Spectral Norm"),
        (axes[1, 1], f"Singular Value Entropy ({REPR_LAYER_SHORT})", "SV Entropy"),
    ]:
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

    fig.suptitle("Trace/Norm-Constrained Muon Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "trace_norm_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
