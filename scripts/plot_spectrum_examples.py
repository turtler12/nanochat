"""
Plot actual singular value spectra for representative layers,
with power law fits overlaid, to make the decay visible.
"""

import os
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if "_config" in obj:
                continue
            records.append(obj)
    return records


def power_law(i, a, alpha):
    return a * np.power(i, -alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = load_jsonl(args.input)

    # Pick a mid-training step
    steps = sorted(set(r["step"] for r in records))
    mid_step = steps[len(steps) // 2]

    # Representative layers: one from each type
    targets = {
        "ve_gate (6x32)":       "transformer.h.5.attn.ve_gate.weight",
        "attn V (768x768)":     "transformer.h.3.attn.c_v.weight",
        "attn Q (768x768)":     "transformer.h.5.attn.c_q.weight",
        "attn K (768x768)":     "transformer.h.5.attn.c_k.weight",
        "MLP up (3072x768)":    "transformer.h.5.mlp.c_fc.weight",
        "MLP down (768x3072)":  "transformer.h.5.mlp.c_proj.weight",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, (label, layer_name) in zip(axes, targets.items()):
        r = next((r for r in records if r["layer_name"] == layer_name and r["step"] == mid_step), None)
        if r is None:
            ax.set_title(f"{label}\n(not found at step {mid_step})")
            continue

        svs = np.array(r["singular_values"])
        n = len(svs)
        indices = np.arange(1, n + 1, dtype=float)

        # Plot actual SVs
        ax.bar(indices, svs, color="C0", alpha=0.7, width=0.8, label="Actual SVs")

        # Fit and plot power law
        try:
            popt, _ = curve_fit(power_law, indices, svs, p0=[svs[0], 1.0], maxfev=5000)
            a_fit, alpha_fit = popt
            fit_curve = power_law(indices, *popt)
            ss_res = np.sum((svs - fit_curve) ** 2)
            ss_tot = np.sum((svs - svs.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.plot(indices, fit_curve, "r-", linewidth=2,
                    label=f"Fit: s_i ~ i^(-{alpha_fit:.2f})  R²={r2:.3f}")
        except (RuntimeError, ValueError):
            alpha_fit = None

        # Annotate with key stats
        nn = r["nuclear_norm"]
        top1_pct = r["sv_sum_top1"] / nn * 100 if nn > 0 else 0
        top4_pct = r["sv_sum_top4"] / nn * 100 if nn > 0 else 0
        ax.set_title(f"{label}", fontsize=11, fontweight="bold")
        stats_text = f"L1/L2={r['l1_l2_ratio']:.1f}  effRank={r['effective_rank']:.1f}\ntop1={top1_pct:.0f}%  top4={top4_pct:.0f}%"
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=8,
                va="top", ha="right", bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
        ax.set_xlabel("SV index i")
        ax.set_ylabel("Singular value")
        ax.legend(fontsize=8, loc="upper center")

    plt.suptitle(f"Gradient singular value spectra at step {mid_step}\n"
                 f"Bar = actual top-16 SVs, red line = power law fit (s_i = a · i^(-α))",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "tier3_spectrum_examples.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved tier3_spectrum_examples.png (step {mid_step})")


if __name__ == "__main__":
    main()
