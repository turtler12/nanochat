"""
Compare SoftmaxMuon eta sweep results against baseline Muon.
Generates two figures:
  1. Validation BPB vs training step for all runs
  2. Spectral diagnostics (SV entropy, nuclear norm, effective eta) for representative layer

Usage:
    python -m scripts.compare_softmax_runs
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.environ.get("NANOCHAT_BASE_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache"))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "base_checkpoints")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "softmax_sweep_results")

RUNS = {
    "baseline":   ("baseline_muon",         "Baseline Muon",         "#1f77b4"),
    "eta_0.01":   ("softmax_eta0.01_d20",   "SoftmaxMuon eta=0.01",  "#ff7f0e"),
    "eta_0.1":    ("softmax_eta0.1_d20",    "SoftmaxMuon eta=0.1",   "#2ca02c"),
    "eta_1.0":    ("softmax_eta1.0_d20",    "SoftmaxMuon eta=1.0",   "#d62728"),
    "eta_5.0":    ("softmax_eta5.0_d20",    "SoftmaxMuon eta=5.0",   "#9467bd"),
    "eta_20.0":   ("softmax_eta20.0_d20",   "SoftmaxMuon eta=20.0",  "#8c564b"),
}


def load_jsonl(path):
    """Load a JSONL file, skipping the config header line."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "_config" in d:
                continue
            records.append(d)
    return records


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all data
    train_data = {}
    val_data = {}
    spectral_data = {}
    found_runs = []

    for key, (tag, label, color) in RUNS.items():
        run_dir = os.path.join(CHECKPOINT_DIR, tag)
        train_records = load_jsonl(os.path.join(run_dir, "train_log.jsonl"))
        val_records = load_jsonl(os.path.join(run_dir, "val_log.jsonl"))
        spectral_records = load_jsonl(os.path.join(run_dir, "spectral_log.jsonl"))

        if not train_records:
            print(f"  [SKIP] {key}: no train log found")
            continue

        train_data[key] = train_records
        val_data[key] = val_records
        spectral_data[key] = spectral_records
        found_runs.append(key)
        print(f"  [OK] {key}: {len(train_records)} train, {len(val_records)} val, {len(spectral_records)} spectral")

    if not found_runs:
        print("No runs found! Make sure experiments have completed.")
        return

    # =========================================================================
    # Figure 1: Validation BPB vs training step
    # =========================================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 7))
    fig1.suptitle("SoftmaxMuon Eta Sweep: Validation BPB vs Step", fontsize=14, fontweight="bold")

    for key in found_runs:
        tag, label, color = RUNS[key]
        records = val_data[key]
        if not records:
            continue
        steps = [r["step"] for r in records]
        bpbs = [r["val_bpb"] for r in records]
        linewidth = 2.5 if key == "baseline" else 1.5
        linestyle = "--" if key == "baseline" else "-"
        ax1.plot(steps, bpbs, label=label, color=color, alpha=0.9, linewidth=linewidth,
                 linestyle=linestyle, marker="o", markersize=3)

    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Validation BPB", fontsize=12)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plot1_path = os.path.join(OUTPUT_DIR, "val_bpb_comparison.png")
    fig1.savefig(plot1_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure 1 saved to {plot1_path}")
    plt.close(fig1)

    # =========================================================================
    # Figure 2: Spectral diagnostics (SV entropy, nuclear norm, effective eta)
    # =========================================================================
    runs_with_spectral = [k for k in found_runs if spectral_data.get(k)]

    if runs_with_spectral:
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
        fig2.suptitle("SoftmaxMuon Spectral Diagnostics (MLP up-proj layer 0)", fontsize=14, fontweight="bold")

        # Subplot 1: SV entropy vs step
        ax = axes2[0]
        for key in runs_with_spectral:
            tag, label, color = RUNS[key]
            records = spectral_data[key]
            steps = [r["step"] for r in records if "sv_entropy" in r]
            entropy = [r["sv_entropy"] for r in records if "sv_entropy" in r]
            if steps:
                linewidth = 2.5 if key == "baseline" else 1.5
                linestyle = "--" if key == "baseline" else "-"
                ax.plot(steps, entropy, label=label, color=color, alpha=0.9,
                        linewidth=linewidth, linestyle=linestyle)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("SV Entropy of W", fontsize=11)
        ax.set_title("Singular Value Entropy")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        # Subplot 2: Nuclear norm vs step
        ax = axes2[1]
        for key in runs_with_spectral:
            tag, label, color = RUNS[key]
            records = spectral_data[key]
            steps = [r["step"] for r in records if "nuclear_norm" in r]
            nuc = [r["nuclear_norm"] for r in records if "nuclear_norm" in r]
            if steps:
                linewidth = 2.5 if key == "baseline" else 1.5
                linestyle = "--" if key == "baseline" else "-"
                ax.plot(steps, nuc, label=label, color=color, alpha=0.9,
                        linewidth=linewidth, linestyle=linestyle)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Nuclear Norm of W", fontsize=11)
        ax.set_title("Nuclear Norm")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        # Subplot 3: Effective eta (constant per run, but shown for clarity)
        ax = axes2[2]
        eta_vals = []
        eta_labels = []
        eta_colors = []
        final_entropy = []
        for key in runs_with_spectral:
            tag, label, color = RUNS[key]
            records = spectral_data[key]
            etas = [r["eta"] for r in records if "eta" in r]
            entropies = [r["sv_entropy"] for r in records if "sv_entropy" in r]
            if etas and entropies:
                eta_vals.append(etas[0])
                eta_labels.append(label.replace(" ", "\n"))
                eta_colors.append(color)
                final_entropy.append(entropies[-1])

        if eta_vals:
            bars = ax.bar(range(len(eta_vals)), final_entropy, color=eta_colors, alpha=0.8, edgecolor="black")
            ax.set_xticks(range(len(eta_vals)))
            ax.set_xticklabels(eta_labels, fontsize=7)
            ax.set_ylabel("Final SV Entropy", fontsize=11)
            ax.set_title("Final SV Entropy by Eta")
            for bar, val in zip(bars, final_entropy):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plot2_path = os.path.join(OUTPUT_DIR, "spectral_diagnostics.png")
        fig2.savefig(plot2_path, dpi=150, bbox_inches="tight")
        print(f"Figure 2 saved to {plot2_path}")
        plt.close(fig2)
    else:
        print("No spectral data found, skipping Figure 2")

    # =========================================================================
    # Summary table
    # =========================================================================
    print("\n" + "=" * 120)
    print(f"{'Run':<30s} {'Final Train':<14s} {'Final Val BPB':<14s} {'Total Time':<14s} {'Avg Step (ms)':<14s} {'Overhead':<10s}")
    print("=" * 120)

    baseline_avg_dt = None
    summary = {}
    for key in found_runs:
        tag, label, color = RUNS[key]
        train_records = train_data[key]
        val_records = val_data[key]

        final_train = train_records[-1]["loss"] if train_records else float("nan")
        final_val = val_records[-1]["val_bpb"] if val_records else float("nan")
        total_time = train_records[-1]["total_training_time"] if train_records else 0

        dts = [r["dt"] for r in train_records if r["step"] > 10]
        avg_dt = np.mean(dts) * 1000 if dts else 0

        if key == "baseline":
            baseline_avg_dt = avg_dt
            overhead_str = "---"
        elif baseline_avg_dt and baseline_avg_dt > 0:
            overhead_pct = (avg_dt - baseline_avg_dt) / baseline_avg_dt * 100
            overhead_str = f"{overhead_pct:+.1f}%"
        else:
            overhead_str = "N/A"

        print(f"{label:<30s} {final_train:<14.6f} {final_val:<14.6f} {total_time/60:<14.2f} {avg_dt:<14.2f} {overhead_str:<10s}")

        summary[key] = {
            "label": label,
            "final_train_loss": final_train,
            "final_val_bpb": final_val if not np.isnan(final_val) else None,
            "total_time_minutes": total_time / 60,
            "avg_step_ms": avg_dt,
            "num_steps": len(train_records),
        }

    print("=" * 120)

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
