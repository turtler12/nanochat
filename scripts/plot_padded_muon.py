"""
Plot padded Muon experiment results (3-way comparison).
Clean 2x2 layout focused on readability.

Usage:
  python -m scripts.plot_padded_muon \
    --runs baseline=padded_muon_results/baseline/val_loss.json \
    --runs padded_polar=padded_muon_results/padded_polar_alpha0.03/val_loss.json \
    --runs padded_only=padded_muon_results/padded_only_alpha0.03/val_loss.json \
    --sv-logs baseline=padded_muon_results/baseline/sv_log.jsonl \
    --sv-logs padded_polar=padded_muon_results/padded_polar_alpha0.03/sv_log.jsonl \
    --sv-logs padded_only=padded_muon_results/padded_only_alpha0.03/sv_log.jsonl \
    --output padded_muon_results/comparison.png
"""

import json
import argparse
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Style ────────────────────────────────────────────────────────────────────
VARIANT_STYLE = {
    # name -> (color, display label)
    "baseline":     ("#2563eb", "Baseline (Polar only)"),
    "padded_polar": ("#ea580c", "Pad + Polar"),
    "padded_only":  ("#16a34a", "Pad only (no Polar)"),
}

def _style(name):
    """Return (color, label) for a variant name, with fallback."""
    if name in VARIANT_STYLE:
        return VARIANT_STYLE[name]
    return ("#888888", name)


# ── Data loading ─────────────────────────────────────────────────────────────
def load_results(path):
    with open(path) as f:
        return json.load(f)

def load_sv_log(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def group_sv_entries(entries):
    """Group SV entries by (layer_id, stage). Handles old + new formats."""
    groups = defaultdict(lambda: {"steps": [], "p05": [], "p50": [], "p95": [], "sigma_max": []})
    for e in entries:
        if "layer_id" in e:
            key = (e["layer_id"], e["stage"])
            groups[key]["steps"].append(e["step"])
            groups[key]["p05"].append(e["p05"])
            groups[key]["p50"].append(e["p50"])
            groups[key]["p95"].append(e["p95"])
            groups[key]["sigma_max"].append(e["sigma_max"])
        elif "layers" in e:
            # Old format: nested {step, layers: {label: stats}}
            for label, stats in e["layers"].items():
                shape = stats.get("shape", [])
                layer_id = f"largest_{shape}" if "last" in label else f"smallest_{shape}"
                key = (layer_id, "pre")
                groups[key]["steps"].append(e["step"])
                groups[key]["p05"].append(stats["p05"])
                groups[key]["p50"].append(stats["p50"])
                groups[key]["p95"].append(stats["p95"])
                groups[key]["sigma_max"].append(stats["sigma_max"])
    return dict(groups)


def parse_kv(items):
    result = {}
    for item in items:
        name, _, path = item.partition("=")
        result[name] = path
    return result


def pick_large_layer(sv_data):
    """Find the layer_id containing 'largest' (the interesting big matrix)."""
    for groups in sv_data.values():
        for (layer_id, _stage) in groups:
            if "largest" in layer_id:
                return layer_id
    return None


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", action="append", required=True)
    parser.add_argument("--sv-logs", action="append", default=[])
    parser.add_argument("--output", default="padded_muon_comparison.png")
    args = parser.parse_args()

    run_paths = parse_kv(args.runs)
    sv_paths = parse_kv(args.sv_logs)
    runs = {name: load_results(path) for name, path in run_paths.items()}
    sv_data = {name: group_sv_entries(load_sv_log(path)) for name, path in sv_paths.items()}

    layer_id = pick_large_layer(sv_data)
    has_sv = layer_id is not None and len(sv_data) > 0

    # ── Layout: 3 rows × 1 col ──────────────────────────────────────────────
    #   Row 0: Val BPB comparison
    #   Row 1: SV median (p50) across variants, one line per (variant, stage)
    #   Row 2: SV spread — p05 & p95 as shaded bands around p50
    nrows = 3 if has_sv else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.5 * nrows))
    if nrows == 1:
        axes = [axes]
    fig.suptitle("Padded Muon Experiment — 3-Way Comparison", fontsize=14, fontweight="bold", y=0.995)

    # ── Panel 1: Val BPB ─────────────────────────────────────────────────────
    ax = axes[0]
    for name, data in runs.items():
        color, label = _style(name)
        steps = [e["step"] for e in data["val_loss_log"]]
        bpbs = [e["val_bpb"] for e in data["val_loss_log"]]
        ax.plot(steps, bpbs, color=color, label=f"{label}  (final {bpbs[-1]:.4f})",
                linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Val BPB (lower is better)")
    ax.set_title("Validation BPB")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.2)

    if not has_sv:
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"Saved plot to {args.output}")
        return

    # ── Panel 2: SV median (p50) — one line per variant×stage ────────────────
    ax = axes[1]
    stage_ls = {"pre": "-", "pad": "--", "post": ":"}
    stage_label = {"pre": "after momentum", "pad": "after padding", "post": "after Polar Express"}

    for name, groups in sv_data.items():
        color, vlabel = _style(name)
        for stage in ["pre", "pad", "post"]:
            key = (layer_id, stage)
            if key not in groups:
                continue
            d = groups[key]
            # skip step 0 (all zeros before any gradient)
            steps = d["steps"][1:]
            p50 = d["p50"][1:]
            if not steps:
                continue
            ax.plot(steps, p50, color=color, linestyle=stage_ls[stage], linewidth=2,
                    label=f"{vlabel} — {stage_label[stage]}", alpha=0.9)

    ax.set_xlabel("Step")
    ax.set_ylabel("Median SV / max SV")
    ax.set_title(f"SV Spectrum Health — Median (p50)  [{layer_id}]")
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.9, fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.2)

    # ── Panel 3: SV spread — p05-p95 bands, pre stage only ──────────────────
    ax = axes[2]
    for name, groups in sv_data.items():
        color, vlabel = _style(name)
        # Show "pre" stage for all variants (apples-to-apples: the raw update direction)
        key = (layer_id, "pre")
        if key not in groups:
            continue
        d = groups[key]
        steps = d["steps"][1:]
        p05 = d["p05"][1:]
        p50 = d["p50"][1:]
        p95 = d["p95"][1:]
        if not steps:
            continue
        ax.plot(steps, p50, color=color, linewidth=2, label=f"{vlabel} median")
        ax.fill_between(steps, p05, p95, color=color, alpha=0.15,
                        label=f"{vlabel} p05–p95")

    ax.set_xlabel("Step")
    ax.set_ylabel("Normalized SV (sigma / sigma_max)")
    ax.set_title(f"SV Spread Before Orth (p05–p95 band)  [{layer_id}]")
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.9, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
