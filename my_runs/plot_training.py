"""
Parse nanochat SLURM training logs and plot key metrics.
Usage: python my_runs/plot_training.py [--log slurm_logs/job_312238.log]
"""

import re
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def parse_log(log_path):
    """Parse the training log and extract metrics per step."""
    steps, losses, mfus, tok_secs, times_min = [], [], [], [], []
    core_steps, core_values = [], []
    lr_mults = []

    # Pattern for training lines:
    # step 00123/07226 (1.70%) | loss: 5.432 | lrm: 1.00 | dt: 4000ms | tok/sec: 262,000 | bf16_mfu: 48.5 | epoch: 1 | total time: 1.23m
    step_pattern = re.compile(
        r"step (\d+)/(\d+).*?"
        r"loss: ([\d.]+).*?"
        r"lrm: ([\d.]+).*?"
        r"dt: ([\d.]+)ms.*?"
        r"tok/sec: ([\d,]+).*?"
        r"bf16_mfu: ([\d.]+).*?"
        r"total time: ([\d.]+)m"
    )

    # Pattern for CORE metric:
    # Step 02000 | CORE metric: 0.1840
    core_pattern = re.compile(r"Step (\d+) \| CORE metric: ([\d.]+)")

    with open(log_path) as f:
        for line in f:
            m = step_pattern.search(line)
            if m:
                step = int(m.group(1))
                total_steps = int(m.group(2))
                loss = float(m.group(3))
                lrm = float(m.group(4))
                dt_ms = float(m.group(5))
                tok_sec = int(m.group(6).replace(",", ""))
                mfu = float(m.group(7))
                time_min = float(m.group(8))

                steps.append(step)
                losses.append(loss)
                lr_mults.append(lrm)
                tok_secs.append(tok_sec)
                mfus.append(mfu)
                times_min.append(time_min)
                continue

            m = core_pattern.search(line)
            if m:
                core_steps.append(int(m.group(1)))
                core_values.append(float(m.group(2)))

    return {
        "steps": steps, "total_steps": total_steps if steps else 0,
        "losses": losses, "lr_mults": lr_mults,
        "tok_secs": tok_secs, "mfus": mfus, "times_min": times_min,
        "core_steps": core_steps, "core_values": core_values,
    }


def plot_metrics(data, output_path):
    """Create a 2x2 figure with training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"nanochat Training Run — 4× H100 NVL — depth 26\n"
        f"{len(data['steps'])} steps logged / {data['total_steps']} total",
        fontsize=14, fontweight="bold",
    )

    # Use a nicer color palette
    c1, c2, c3, c4 = "#2563eb", "#dc2626", "#16a34a", "#9333ea"

    # ── 1. Loss vs Step ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(data["steps"], data["losses"], linewidth=0.6, color=c1, alpha=0.4, label="raw")
    # Smoothed loss (running average, window 50)
    window = 50
    if len(data["losses"]) > window:
        smoothed = []
        for i in range(len(data["losses"])):
            start = max(0, i - window)
            smoothed.append(sum(data["losses"][start:i+1]) / (i - start + 1))
        ax.plot(data["steps"], smoothed, linewidth=1.5, color=c1, label=f"smoothed (w={window})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss vs Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 2. Loss vs Wall-Clock Time ───────────────────────────────────
    ax = axes[0, 1]
    hours = [t / 60 for t in data["times_min"]]
    ax.plot(hours, data["losses"], linewidth=0.6, color=c2, alpha=0.4, label="raw")
    if len(data["losses"]) > window:
        ax.plot(hours, smoothed, linewidth=1.5, color=c2, label=f"smoothed (w={window})")
    ax.set_xlabel("Wall-Clock Time (hours)")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss vs Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 3. Throughput: MFU and tok/sec ───────────────────────────────
    ax = axes[1, 0]
    # Skip step 0 (compilation step with inflated dt)
    s = 1 if len(data["steps"]) > 1 else 0
    ax.plot(data["steps"][s:], data["mfus"][s:], linewidth=0.8, color=c3, alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("bf16 MFU (%)", color=c3)
    ax.tick_params(axis="y", labelcolor=c3)
    ax.set_ylim(0, max(data["mfus"][s:]) * 1.15 if data["mfus"][s:] else 100)
    ax.set_title("GPU Utilization (MFU) & Throughput")
    ax.grid(True, alpha=0.3)

    # Second y-axis for tok/sec
    ax2 = ax.twinx()
    ax2.plot(data["steps"][s:], [t / 1000 for t in data["tok_secs"][s:]], linewidth=0.8, color=c4, alpha=0.7)
    ax2.set_ylabel("Tokens/sec (×1000)", color=c4)
    ax2.tick_params(axis="y", labelcolor=c4)

    # ── 4. CORE Metric (if available) ────────────────────────────────
    ax = axes[1, 1]
    if data["core_steps"]:
        ax.plot(data["core_steps"], data["core_values"], "o-", color=c1, markersize=8, linewidth=2)
        for x, y in zip(data["core_steps"], data["core_values"]):
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")
        # GPT-2 baseline
        ax.axhline(y=0.2565, color=c2, linestyle="--", linewidth=1.5, label="GPT-2 baseline (0.2565)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("CORE Score")
        ax.set_title("DCLM CORE Metric (higher = better)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(min(data["core_values"]) * 0.9, max(max(data["core_values"]), 0.2565) * 1.05)
    else:
        # If no CORE data, show LR multiplier schedule
        ax.plot(data["steps"], data["lr_mults"], linewidth=1.2, color=c4)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("LR Multiplier")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot nanochat training metrics")
    parser.add_argument("--log", default="slurm_logs/job_312238.log", help="Path to SLURM log file")
    parser.add_argument("--output", default=None, help="Output image path (default: my_runs/training_plot.png)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(__file__), "training_plot.png")

    print(f"Parsing log: {args.log}")
    data = parse_log(args.log)
    print(f"  Found {len(data['steps'])} training steps")
    print(f"  Loss range: {min(data['losses']):.3f} → {max(data['losses']):.3f}" if data["losses"] else "  No training data found!")
    print(f"  CORE evaluations: {len(data['core_steps'])}")
    if data["core_values"]:
        print(f"  CORE scores: {data['core_values']}")
    if data["times_min"]:
        print(f"  Total training time: {max(data['times_min']):.1f} min ({max(data['times_min'])/60:.1f} hours)")

    plot_metrics(data, args.output)


if __name__ == "__main__":
    main()
