"""
Plot Polar Interpolation vs Baseline (Muon) training curves.

Parses SLURM log files and produces a comparison plot.

Usage:
  python polar_interp_plot.py
"""

import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────

BASELINE_LOG = "slurm_logs/pi_baseline_394090.log"
PI_V1_LOG = "slurm_logs/pi_beta03_399897.log"
# v2 log will be auto-detected below
PI_LOG = None  # set in main

SAVE_PATH = "polar_interpolation_benchmark.png"

# ── Log Parsing ──────────────────────────────────────────────────────────

def parse_log(path):
    """Parse a nanochat training log into structured data."""
    data = {
        "steps": [], "loss": [], "dt": [], "mfu": [], "tok_per_sec": [],
        "wall_time": [],
        "val_steps": [], "val_bpb": [],
        "core_steps": [], "core_metric": [],
        "total_time": None, "min_val_bpb": None, "peak_mem": None,
    }
    if not os.path.exists(path):
        return None

    with open(path) as f:
        for line in f:
            m = re.match(
                r'step (\d+)/(\d+).*\| loss: ([\d.]+).*\| dt: ([\d.]+)ms.*\| tok/sec: ([\d,]+).*\| bf16_mfu: ([\d.]+).*\| total time: ([\d.]+)m',
                line
            )
            if m:
                data["steps"].append(int(m.group(1)))
                data["loss"].append(float(m.group(3)))
                data["dt"].append(float(m.group(4)))
                data["tok_per_sec"].append(int(m.group(5).replace(',', '')))
                data["mfu"].append(float(m.group(6)))
                data["wall_time"].append(float(m.group(7)))
                continue

            m = re.match(r'Step (\d+) \| Validation bpb: ([\d.]+)', line)
            if m:
                data["val_steps"].append(int(m.group(1)))
                data["val_bpb"].append(float(m.group(2)))
                continue

            m = re.match(r'Step (\d+) \| CORE metric: ([\d.]+)', line)
            if m:
                data["core_steps"].append(int(m.group(1)))
                data["core_metric"].append(float(m.group(2)))
                continue

            m = re.match(r'Total training time: ([\d.]+)m', line)
            if m:
                data["total_time"] = float(m.group(1))
            m = re.match(r'Minimum validation bpb: ([\d.]+)', line)
            if m:
                data["min_val_bpb"] = float(m.group(1))
            m = re.match(r'Peak memory usage: ([\d.]+)MiB', line)
            if m:
                data["peak_mem"] = float(m.group(1))

    for k in ["steps", "loss", "dt", "mfu", "tok_per_sec", "wall_time", "val_steps", "val_bpb", "core_steps", "core_metric"]:
        data[k] = np.array(data[k])
    return data


# ── Smoothing ────────────────────────────────────────────────────────────

def ema_smooth(values, alpha=0.95):
    """Exponential moving average without debiasing (simple EMA)."""
    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * values[i]
    return smoothed


def get_val_wall_times(data):
    """Map validation step numbers to wall-clock times."""
    if len(data["steps"]) == 0 or len(data["wall_time"]) == 0:
        return [], []
    times, bpbs = [], []
    for vs, vb in zip(data["val_steps"], data["val_bpb"]):
        mask = data["steps"] <= vs
        if mask.any():
            idx = np.where(mask)[0][-1]
            times.append(data["wall_time"][idx])
            bpbs.append(vb)
    return times, bpbs


# ── Plotting ─────────────────────────────────────────────────────────────

def make_plot(baseline, pi=None):
    """Generate the comparison plot."""
    has_pi = pi is not None and len(pi["steps"]) > 0
    skip = 10

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    fig.suptitle(
        "Polar Interpolation vs Muon Baseline\n"
        "depth=20, 4xH100 NVL, FP8, ~900M params"
        + (", PI v2: beta=0.3, 5 NS steps" if has_pi else ""),
        fontsize=14, fontweight='bold'
    )

    lbl_b = 'Muon (baseline)'
    lbl_p = 'Polar Interp v2 (beta=0.3)'

    # ── 1. Training Loss vs Step ──
    ax = axes[0, 0]
    ax.set_title('Training Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    if len(baseline["steps"]) > skip:
        s, l = baseline["steps"][skip:], baseline["loss"][skip:]
        ax.plot(s, ema_smooth(l), lw=1.5, label=lbl_b)
    if has_pi and len(pi["steps"]) > skip:
        s, l = pi["steps"][skip:], pi["loss"][skip:]
        ax.plot(s, ema_smooth(l), lw=1.5, label=lbl_p)
    ax.legend(fontsize=8)

    # ── 2. Training Loss vs Wall Time ──
    ax = axes[0, 1]
    ax.set_title('Training Loss vs Wall Time')
    ax.set_xlabel('Wall Time (min)')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    if len(baseline["wall_time"]) > skip:
        t, l = baseline["wall_time"][skip:], baseline["loss"][skip:]
        ax.plot(t, ema_smooth(l), lw=1.5, label=lbl_b)
    if has_pi and len(pi["wall_time"]) > skip:
        t, l = pi["wall_time"][skip:], pi["loss"][skip:]
        ax.plot(t, ema_smooth(l), lw=1.5, label=lbl_p)
    ax.legend(fontsize=8)

    # ── 3. Validation BPB vs Step ──
    ax = axes[0, 2]
    ax.set_title('Validation BPB (lower = better)')
    ax.set_xlabel('Step')
    ax.set_ylabel('BPB')
    ax.grid(True, alpha=0.3)
    if len(baseline["val_steps"]) > 0:
        ax.plot(baseline["val_steps"], baseline["val_bpb"], 'o-',
                lw=1.5, ms=3, label=lbl_b)
    if has_pi and len(pi["val_steps"]) > 0:
        ax.plot(pi["val_steps"], pi["val_bpb"], 's-',
                lw=1.5, ms=3, label=lbl_p)
    ax.legend(fontsize=8)

    # ── 4. Validation BPB vs Wall Time ──
    ax = axes[0, 3]
    ax.set_title('Validation BPB vs Wall Time')
    ax.set_xlabel('Wall Time (min)')
    ax.set_ylabel('BPB')
    ax.grid(True, alpha=0.3)
    bt, bb = get_val_wall_times(baseline)
    if bt:
        ax.plot(bt, bb, 'o-', lw=1.5, ms=3, label=lbl_b)
    if has_pi:
        pt, pb = get_val_wall_times(pi)
        if pt:
            ax.plot(pt, pb, 's-', lw=1.5, ms=3, label=lbl_p)
    ax.legend(fontsize=8)

    # ── 5. CORE Metric ──
    ax = axes[1, 0]
    ax.set_title('CORE Benchmark (higher = better)')
    ax.set_xlabel('Step')
    ax.set_ylabel('CORE Score')
    ax.grid(True, alpha=0.3)
    if len(baseline["core_steps"]) > 0:
        ax.plot(baseline["core_steps"], baseline["core_metric"], 'o-',
                lw=2, ms=6, label=lbl_b)
        for s, c in zip(baseline["core_steps"], baseline["core_metric"]):
            ax.annotate(f'{c:.3f}', xy=(s, c), fontsize=8,
                        textcoords="offset points", xytext=(8, 5))
    if has_pi and len(pi["core_steps"]) > 0:
        ax.plot(pi["core_steps"], pi["core_metric"], 's-',
                lw=2, ms=6, label=lbl_p)
        for s, c in zip(pi["core_steps"], pi["core_metric"]):
            ax.annotate(f'{c:.3f}', xy=(s, c), fontsize=8,
                        textcoords="offset points", xytext=(8, -12))
    ax.legend(fontsize=8)

    # ── 6. Step Latency ──
    ax = axes[1, 1]
    ax.set_title('Step Latency')
    ax.set_xlabel('Step')
    ax.set_ylabel('Time per step (ms)')
    ax.grid(True, alpha=0.3)
    if len(baseline["steps"]) > skip:
        s, dt = baseline["steps"][skip:], baseline["dt"][skip:]
        ax.plot(s, ema_smooth(dt, 0.99), lw=1.5, label=f'{lbl_b} (avg {np.mean(dt):.0f}ms)')
    if has_pi and len(pi["steps"]) > skip:
        s, dt = pi["steps"][skip:], pi["dt"][skip:]
        ax.plot(s, ema_smooth(dt, 0.99), lw=1.5, label=f'{lbl_p} (avg {np.mean(dt):.0f}ms)')
    ax.legend(fontsize=8)

    # ── 7. MFU ──
    ax = axes[1, 2]
    ax.set_title('Model FLOP Utilization')
    ax.set_xlabel('Step')
    ax.set_ylabel('BF16 MFU (%)')
    ax.grid(True, alpha=0.3)
    if len(baseline["steps"]) > skip:
        s, mfu = baseline["steps"][skip:], baseline["mfu"][skip:]
        ax.plot(s, ema_smooth(mfu, 0.99), lw=1.5, label=f'{lbl_b} (avg {np.mean(mfu):.1f}%)')
    if has_pi and len(pi["steps"]) > skip:
        s, mfu = pi["steps"][skip:], pi["mfu"][skip:]
        ax.plot(s, ema_smooth(mfu, 0.99), lw=1.5, label=f'{lbl_p} (avg {np.mean(mfu):.1f}%)')
    ax.legend(fontsize=8)

    # ── 8. Summary Table ──
    ax = axes[1, 3]
    ax.axis('off')
    ax.set_title('Summary')

    def fmt(v, fmt_str='.4f'):
        return f'{v:{fmt_str}}' if v is not None else '-'

    headers = ['', 'Baseline']
    if has_pi:
        headers.append('Polar Interp')

    rows = []
    def add_row(label, b_val, p_val=None):
        row = [label, b_val]
        if has_pi:
            row.append(p_val if p_val is not None else '-')
        rows.append(row)

    # Best val BPB so far
    b_best = fmt(baseline['min_val_bpb']) if baseline['min_val_bpb'] else fmt(baseline['val_bpb'].min()) if len(baseline['val_bpb']) > 0 else '-'
    p_best = '-'
    if has_pi:
        p_best = fmt(pi['min_val_bpb']) if pi['min_val_bpb'] else fmt(pi['val_bpb'].min()) if len(pi['val_bpb']) > 0 else '-'
    add_row('Best Val BPB', b_best, p_best if has_pi else None)

    b_core = fmt(baseline['core_metric'][-1], '.3f') if len(baseline['core_metric']) > 0 else '-'
    p_core = fmt(pi['core_metric'][-1], '.3f') if has_pi and len(pi['core_metric']) > 0 else '-'
    add_row('Final CORE', b_core, p_core if has_pi else None)

    add_row('Wall Time', f"{baseline['total_time']:.1f} min" if baseline['total_time'] else '-',
            f"{pi['total_time']:.1f} min" if has_pi and pi['total_time'] else '-')

    add_row('Peak Memory', f"{baseline['peak_mem']:.0f} MiB" if baseline['peak_mem'] else '-',
            f"{pi['peak_mem']:.0f} MiB" if has_pi and pi['peak_mem'] else '-')

    avg_mfu_b = f"{np.mean(baseline['mfu'][skip:]):.1f}%" if len(baseline['mfu']) > skip else '-'
    avg_mfu_p = f"{np.mean(pi['mfu'][skip:]):.1f}%" if has_pi and len(pi['mfu']) > skip else '-'
    add_row('Avg MFU', avg_mfu_b, avg_mfu_p if has_pi else None)

    avg_dt_b = f"{np.mean(baseline['dt'][skip:]):.0f} ms" if len(baseline['dt']) > skip else '-'
    avg_dt_p = f"{np.mean(pi['dt'][skip:]):.0f} ms" if has_pi and len(pi['dt']) > skip else '-'
    add_row('Avg Step Time', avg_dt_b, avg_dt_p if has_pi else None)

    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
    print(f"Saved {SAVE_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Parsing baseline log...")
    baseline = parse_log(BASELINE_LOG)
    if baseline is None:
        print(f"ERROR: Baseline log not found at {BASELINE_LOG}")
        exit(1)
    print(f"  {len(baseline['steps'])} training steps, {len(baseline['val_steps'])} val evals, {len(baseline['core_steps'])} CORE evals")

    # Auto-detect PI log: prefer v2, fall back to v1
    import glob
    v2_logs = sorted(glob.glob("slurm_logs/pi_v2_b03_*.log"))
    if v2_logs:
        PI_LOG = v2_logs[-1]  # most recent v2
        print(f"Found PI v2 log: {PI_LOG}")
    else:
        PI_LOG = PI_V1_LOG
        print(f"No v2 log found, using v1: {PI_LOG}")

    pi = None
    if os.path.exists(PI_LOG):
        print("Parsing Polar Interpolation log...")
        pi = parse_log(PI_LOG)
        if pi and len(pi["steps"]) > 0:
            print(f"  {len(pi['steps'])} training steps, {len(pi['val_steps'])} val evals, {len(pi['core_steps'])} CORE evals")
        else:
            print("  Polar Interp log exists but has no training steps yet")
            pi = None
    else:
        print(f"  Polar Interp log not found ({PI_LOG}), plotting baseline only")

    make_plot(baseline, pi)
