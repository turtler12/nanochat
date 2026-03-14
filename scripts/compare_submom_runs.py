"""
Compare subspace momentum experiment results.
Reads JSONL logs from cache/base_checkpoints/submom_*/train_log.jsonl and val_log.jsonl.
Generates a 4-subplot comparison figure and a summary table.

Usage:
    python -m scripts.compare_submom_runs
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = os.environ.get("NANOCHAT_BASE_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache"))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "base_checkpoints")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "subspace_momentum")

RUNS = {
    "baseline":      ("submom_baseline_d20", "Baseline (standard Muon)", "#1f77b4"),
    "submom_double": ("submom_double_d20",   "SubMom Double (m=0.95)",   "#d62728"),
    "submom_single": ("submom_single_d20",   "SubMom Single (m=0.95)",   "#2ca02c"),
    "submom_low":    ("submom_low_d20",      "SubMom Double (m=0.85)",   "#9467bd"),
    "submom_mid":    ("submom_mid_d20",      "SubMom Double (m=0.90)",   "#ff7f0e"),
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
    found_runs = []

    for key, (tag, label, color) in RUNS.items():
        run_dir = os.path.join(CHECKPOINT_DIR, tag)
        train_log = os.path.join(run_dir, "train_log.jsonl")
        val_log = os.path.join(run_dir, "val_log.jsonl")

        train_records = load_jsonl(train_log)
        val_records = load_jsonl(val_log)

        if not train_records:
            print(f"  [SKIP] {key}: no train log found at {train_log}")
            continue

        train_data[key] = train_records
        val_data[key] = val_records
        found_runs.append(key)
        print(f"  [OK] {key}: {len(train_records)} train steps, {len(val_records)} val evals")

    if not found_runs:
        print("No runs found! Make sure experiments have completed.")
        return

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Subspace Momentum Experiments", fontsize=16, fontweight="bold")

    # Subplot 1: Training loss vs step
    ax = axes[0, 0]
    for key in found_runs:
        tag, label, color = RUNS[key]
        records = train_data[key]
        steps = [r["step"] for r in records]
        losses = [r["loss"] for r in records]
        ax.plot(steps, losses, label=label, color=color, alpha=0.8, linewidth=1.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss (EMA)")
    ax.set_title("Training Loss vs Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 2: Validation loss vs step
    ax = axes[0, 1]
    for key in found_runs:
        tag, label, color = RUNS[key]
        records = val_data[key]
        if not records:
            continue
        steps = [r["step"] for r in records]
        bpbs = [r["val_bpb"] for r in records]
        ax.plot(steps, bpbs, label=label, color=color, alpha=0.8, linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation BPB")
    ax.set_title("Validation Loss vs Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 3: Training loss vs wall-clock time
    ax = axes[1, 0]
    for key in found_runs:
        tag, label, color = RUNS[key]
        records = train_data[key]
        times = [r["total_training_time"] / 60 for r in records]  # minutes
        losses = [r["loss"] for r in records]
        ax.plot(times, losses, label=label, color=color, alpha=0.8, linewidth=1.0)
    ax.set_xlabel("Wall-Clock Time (minutes)")
    ax.set_ylabel("Training Loss (EMA)")
    ax.set_title("Training Loss vs Wall-Clock Time (key plot)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 4: Bar chart of final validation loss
    ax = axes[1, 1]
    bar_labels = []
    bar_vals = []
    bar_colors = []
    for key in found_runs:
        tag, label, color = RUNS[key]
        records = val_data[key]
        if records:
            final_bpb = records[-1]["val_bpb"]
            bar_labels.append(label.replace(" ", "\n"))
            bar_vals.append(final_bpb)
            bar_colors.append(color)
    if bar_vals:
        bars = ax.bar(range(len(bar_vals)), bar_vals, color=bar_colors, alpha=0.8, edgecolor="black")
        ax.set_xticks(range(len(bar_vals)))
        ax.set_xticklabels(bar_labels, fontsize=8)
        ax.set_ylabel("Final Validation BPB")
        ax.set_title("Final Validation Loss Comparison")
        # Add value labels on bars
        for bar, val in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "submom_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {plot_path}")
    plt.close()

    # Summary table
    print("\n" + "=" * 120)
    print(f"{'Run':<30s} {'Final Train':<14s} {'Final Val BPB':<14s} {'Total Time':<14s} {'Avg Step (ms)':<14s} {'Overhead':<10s}")
    print("=" * 120)

    baseline_avg_dt = None
    for key in found_runs:
        tag, label, color = RUNS[key]
        train_records = train_data[key]
        val_records = val_data[key]

        final_train = train_records[-1]["loss"] if train_records else float("nan")
        final_val = val_records[-1]["val_bpb"] if val_records else float("nan")
        total_time = train_records[-1]["total_training_time"] if train_records else 0

        # Average step time (skip first 10 steps which are warmup)
        dts = [r["dt"] for r in train_records if r["step"] > 10]
        avg_dt = np.mean(dts) * 1000 if dts else 0  # ms

        if key == "baseline":
            baseline_avg_dt = avg_dt
            overhead_str = "---"
        elif baseline_avg_dt and baseline_avg_dt > 0:
            overhead_pct = (avg_dt - baseline_avg_dt) / baseline_avg_dt * 100
            overhead_str = f"{overhead_pct:+.1f}%"
        else:
            overhead_str = "N/A"

        print(f"{label:<30s} {final_train:<14.6f} {final_val:<14.6f} {total_time/60:<14.2f} {avg_dt:<14.2f} {overhead_str:<10s}")

    print("=" * 120)

    # Save summary as JSON
    summary = {}
    for key in found_runs:
        tag, label, color = RUNS[key]
        train_records = train_data[key]
        val_records = val_data[key]
        dts = [r["dt"] for r in train_records if r["step"] > 10]
        summary[key] = {
            "label": label,
            "final_train_loss": train_records[-1]["loss"] if train_records else None,
            "final_val_bpb": val_records[-1]["val_bpb"] if val_records else None,
            "total_time_minutes": train_records[-1]["total_training_time"] / 60 if train_records else None,
            "avg_step_ms": np.mean(dts) * 1000 if dts else None,
            "num_steps": len(train_records),
        }

    summary_path = os.path.join(OUTPUT_DIR, "submom_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
