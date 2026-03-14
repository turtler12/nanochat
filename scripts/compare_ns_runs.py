#!/usr/bin/env python3
"""Compare Newton-Schulz baseline vs fast4 training runs.

Usage:
    python -m scripts.compare_ns_runs [--base-dir CACHE_DIR]

Reads train_log.jsonl and val_log.jsonl from:
    {base_dir}/base_checkpoints/ns_baseline_d20/
    {base_dir}/base_checkpoints/ns_fast4_d20/

Produces: training_comparison.png
"""

import json
import os
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load_jsonl(path):
    """Load JSONL file, skipping config header lines."""
    records = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if "_config" not in obj:
                records.append(obj)
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=os.environ.get("NANOCHAT_BASE_DIR", "cache"))
    parser.add_argument("--output", type=str, default="training_comparison.png")
    args = parser.parse_args()

    base = Path(args.base_dir) / "base_checkpoints"
    baseline_dir = base / "ns_baseline_d20"
    fast4_dir = base / "ns_fast4_d20"

    # Load training logs
    print(f"Loading baseline from {baseline_dir}")
    print(f"Loading fast4 from {fast4_dir}")

    bl_train = load_jsonl(baseline_dir / "train_log.jsonl")
    f4_train = load_jsonl(fast4_dir / "train_log.jsonl")

    bl_steps = [r["step"] for r in bl_train]
    bl_loss = [r["loss"] for r in bl_train]
    bl_dt = [r["dt"] for r in bl_train]
    bl_time = [r["total_training_time"] for r in bl_train]

    f4_steps = [r["step"] for r in f4_train]
    f4_loss = [r["loss"] for r in f4_train]
    f4_dt = [r["dt"] for r in f4_train]
    f4_time = [r["total_training_time"] for r in f4_train]

    # Load val logs
    has_val = True
    try:
        bl_val = load_jsonl(baseline_dir / "val_log.jsonl")
        f4_val = load_jsonl(fast4_dir / "val_log.jsonl")
        bl_val_steps = [r["step"] for r in bl_val]
        bl_val_bpb = [r["val_bpb"] for r in bl_val]
        f4_val_steps = [r["step"] for r in f4_val]
        f4_val_bpb = [r["val_bpb"] for r in f4_val]
    except (FileNotFoundError, KeyError):
        has_val = False

    # ── Colors — high contrast ──
    c_baseline = '#D62728'  # red
    c_fast4 = '#1F77B4'     # blue

    # ── Figure ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Train loss vs step
    ax = axes[0, 0]
    ax.plot(bl_steps, bl_loss, color=c_baseline, linewidth=1.2, alpha=0.85, label='Baseline 5×quintic (15 matmuls)')
    ax.plot(f4_steps, f4_loss, color=c_fast4, linewidth=1.2, alpha=0.85, linestyle='--', label='Fast4 4×quintic (12 matmuls)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Training loss (EMA)')
    ax.set_title('A. Training Loss vs Step', fontweight='bold', loc='left')
    ax.legend(fontsize=9)
    # Annotate final losses
    ax.annotate(f'{bl_loss[-1]:.4f}', xy=(bl_steps[-1], bl_loss[-1]),
                xytext=(-60, -18), textcoords='offset points', fontsize=9, color=c_baseline, fontweight='bold')
    ax.annotate(f'{f4_loss[-1]:.4f}', xy=(f4_steps[-1], f4_loss[-1]),
                xytext=(-60, 12), textcoords='offset points', fontsize=9, color=c_fast4, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: Val loss vs step
    ax = axes[0, 1]
    if has_val:
        ax.plot(bl_val_steps, bl_val_bpb, 'o-', color=c_baseline, linewidth=1.5, markersize=5, label='Baseline 5×quintic (15 matmuls)')
        ax.plot(f4_val_steps, f4_val_bpb, 's--', color=c_fast4, linewidth=1.5, markersize=5, label='Fast4 4×quintic (12 matmuls)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Validation BPB')
        ax.set_title('B. Validation Loss vs Step', fontweight='bold', loc='left')
        ax.legend(fontsize=9)
        # Annotate final val losses
        ax.annotate(f'{bl_val_bpb[-1]:.4f}', xy=(bl_val_steps[-1], bl_val_bpb[-1]),
                    xytext=(-60, -18), textcoords='offset points', fontsize=9, color=c_baseline, fontweight='bold')
        ax.annotate(f'{f4_val_bpb[-1]:.4f}', xy=(f4_val_steps[-1], f4_val_bpb[-1]),
                    xytext=(-60, 12), textcoords='offset points', fontsize=9, color=c_fast4, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No validation data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('B. Validation Loss vs Step', fontweight='bold', loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 3: Train loss vs wall-clock time (THE KEY PLOT)
    ax = axes[1, 0]
    ax.plot([t/60 for t in bl_time], bl_loss, color=c_baseline, linewidth=1.2, alpha=0.85, label='Baseline 5×quintic (15 matmuls)')
    ax.plot([t/60 for t in f4_time], f4_loss, color=c_fast4, linewidth=1.2, alpha=0.85, linestyle='--', label='Fast4 4×quintic (12 matmuls)')
    ax.set_xlabel('Wall-clock time (minutes)')
    ax.set_ylabel('Training loss (EMA)')
    ax.set_title('C. Training Loss vs Wall-Clock Time', fontweight='bold', loc='left')
    ax.legend(fontsize=9)
    # Annotate final losses with time
    ax.annotate(f'{bl_loss[-1]:.4f} @ {bl_time[-1]/60:.1f}m', xy=(bl_time[-1]/60, bl_loss[-1]),
                xytext=(-120, -18), textcoords='offset points', fontsize=9, color=c_baseline, fontweight='bold')
    ax.annotate(f'{f4_loss[-1]:.4f} @ {f4_time[-1]/60:.1f}m', xy=(f4_time[-1]/60, f4_loss[-1]),
                xytext=(-120, 12), textcoords='offset points', fontsize=9, color=c_fast4, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 4: Average step time bar chart
    ax = axes[1, 1]
    # Exclude first 10 warmup steps
    bl_dt_steady = [r["dt"] for r in bl_train if r["step"] > 10]
    f4_dt_steady = [r["dt"] for r in f4_train if r["step"] > 10]
    bl_avg_dt = np.mean(bl_dt_steady) * 1000  # ms
    f4_avg_dt = np.mean(f4_dt_steady) * 1000  # ms
    speedup = bl_avg_dt / f4_avg_dt

    bars = ax.bar([0, 1], [bl_avg_dt, f4_avg_dt],
                  color=[c_baseline, c_fast4], width=0.55, edgecolor='white', linewidth=1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline\n(15 matmuls)', 'Fast4\n(12 matmuls)'], fontsize=10)
    ax.set_ylabel('Average step time (ms)')
    ax.set_title('D. Step Time Comparison', fontweight='bold', loc='left')
    for bar, t in zip(bars, [bl_avg_dt, f4_avg_dt]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{t:.1f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    speedup_label = f'{speedup:.2f}× faster' if speedup > 1.005 else f'{speedup:.2f}× (no speedup)'
    speedup_color = '#27AE60' if speedup > 1.005 else '#888888'
    ax.text(0.5, max(bl_avg_dt, f4_avg_dt) * 0.5,
            speedup_label, fontsize=14, fontweight='bold', color=speedup_color, ha='center')
    ax.set_ylim(0, max(bl_avg_dt, f4_avg_dt) * 1.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Newton-Schulz Optimization: Baseline (5 iter, 15mm) vs Fast4 (4 iter, 12mm)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(args.output, dpi=180, bbox_inches='tight', facecolor='white')
    print(f"Saved {args.output}")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print(f"{'Metric':<30} {'Baseline':>18} {'Fast4':>18}")
    print("=" * 70)
    print(f"{'NS iterations':<30} {'5':>18} {'4':>18}")
    print(f"{'Matmuls per step':<30} {'15':>18} {'12':>18}")
    print(f"{'Final train loss':<30} {bl_loss[-1]:>18.6f} {f4_loss[-1]:>18.6f}")
    if has_val:
        print(f"{'Final val BPB':<30} {bl_val_bpb[-1]:>18.6f} {f4_val_bpb[-1]:>18.6f}")
        print(f"{'Best val BPB':<30} {min(bl_val_bpb):>18.6f} {min(f4_val_bpb):>18.6f}")
    bl_total_time = bl_time[-1] / 60
    f4_total_time = f4_time[-1] / 60
    print(f"{'Total training time (min)':<30} {bl_total_time:>18.2f} {f4_total_time:>18.2f}")
    print(f"{'Avg step time (ms)':<30} {bl_avg_dt:>18.2f} {f4_avg_dt:>18.2f}")
    print(f"{'Speedup':<30} {'':>18} {speedup:>17.2f}×")
    print("=" * 70)

if __name__ == "__main__":
    main()
