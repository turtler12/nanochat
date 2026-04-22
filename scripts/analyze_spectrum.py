"""
Analyze gradient spectrum from gradient_spectrum_log.jsonl.
Produces summary statistics and plots for heuristic nuclear norm approximation design.

Usage:
    python -m scripts.analyze_spectrum --input path/to/gradient_spectrum_log.jsonl --output-dir path/to/spectrum_analysis/
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict

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
    """s_i = a * i^(-alpha), where i is 1-indexed."""
    return a * np.power(i, -alpha)


def main():
    parser = argparse.ArgumentParser(description="Analyze gradient spectrum logs")
    parser.add_argument("--input", type=str, required=True, help="Path to gradient_spectrum_log.jsonl")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    steps = sorted(set(r["step"] for r in records))
    layer_names = sorted(set(r["layer_name"] for r in records))
    print(f"Steps: {len(steps)} ({min(steps)}..{max(steps)})")
    print(f"Layers: {len(layer_names)}")

    # =========================================================================
    # 1. Spectrum concentration analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. SPECTRUM CONCENTRATION ANALYSIS")
    print("=" * 80)

    ks = [1, 2, 4, 8, 16]
    top_k_fractions = {k: [] for k in ks}

    for r in records:
        nn = r["nuclear_norm"]
        if nn <= 0:
            continue
        for k in ks:
            key = f"sv_sum_top{k}"
            frac = r[key] / nn
            top_k_fractions[k].append(frac)

    print(f"\n{'k':>4} | {'mean':>8} | {'median':>8} | {'min':>8} | {'max':>8} | {'std':>8}")
    print("-" * 60)
    for k in ks:
        vals = np.array(top_k_fractions[k])
        print(f"{k:>4} | {vals.mean():>8.4f} | {np.median(vals):>8.4f} | {vals.min():>8.4f} | {vals.max():>8.4f} | {vals.std():>8.4f}")

    # Plot: side-by-side histograms
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4), sharey=True)
    for ax, k in zip(axes, ks):
        vals = np.array(top_k_fractions[k])
        ax.hist(vals, bins=40, edgecolor="black", alpha=0.7, color=f"C{ks.index(k)}")
        ax.set_title(f"top-{k} fraction")
        ax.set_xlabel(f"sv_sum_top{k} / nuclear_norm")
        ax.axvline(vals.mean(), color="red", linestyle="--", label=f"mean={vals.mean():.3f}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Count")
    plt.suptitle("Spectrum Concentration: Top-k SV Fraction of Nuclear Norm", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "spectrum_concentration.png"), dpi=150)
    plt.close()
    print(f"  Saved spectrum_concentration.png")

    # =========================================================================
    # 2. Morris's heuristic test: c = nuclear_norm / sv_sum_topk
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. MORRIS'S HEURISTIC TEST: c = nuclear_norm / sv_sum_topk")
    print("=" * 80)

    heuristic_ks = [1, 2, 4, 8]
    c_values = {k: [] for k in heuristic_ks}
    c_by_step_layer = {k: defaultdict(dict) for k in heuristic_ks}

    for r in records:
        for k in heuristic_ks:
            topk = r[f"sv_sum_top{k}"]
            if topk > 0:
                c = r["nuclear_norm"] / topk
                c_values[k].append(c)
                c_by_step_layer[k][r["step"]][r["layer_name"]] = c

    print(f"\n{'k':>4} | {'mean':>8} | {'median':>8} | {'std':>8} | {'min':>8} | {'max':>8} | {'CV':>8}")
    print("-" * 72)
    for k in heuristic_ks:
        vals = np.array(c_values[k])
        cv = vals.std() / vals.mean() if vals.mean() > 0 else float("inf")
        print(f"{k:>4} | {vals.mean():>8.3f} | {np.median(vals):>8.3f} | {vals.std():>8.3f} | {vals.min():>8.3f} | {vals.max():>8.3f} | {cv:>8.4f}")

    # Plot: c vs step, one line per layer (or heatmap if too many)
    for k in heuristic_ks:
        fig, ax = plt.subplots(figsize=(12, 6))
        step_layer_data = c_by_step_layer[k]
        if len(layer_names) <= 30:
            for ln in layer_names:
                xs = sorted(step_layer_data.keys())
                ys = [step_layer_data[s].get(ln, float("nan")) for s in xs]
                short_name = ln.split(".")[-2] + "." + ln.split(".")[-1] if "." in ln else ln
                ax.plot(xs, ys, alpha=0.5, linewidth=0.8, label=short_name)
            ax.legend(fontsize=5, ncol=3, loc="upper right")
        else:
            # Heatmap
            step_list = sorted(step_layer_data.keys())
            data_matrix = np.full((len(layer_names), len(step_list)), np.nan)
            for si, s in enumerate(step_list):
                for li, ln in enumerate(layer_names):
                    data_matrix[li, si] = step_layer_data[s].get(ln, np.nan)
            im = ax.imshow(data_matrix, aspect="auto", interpolation="nearest",
                           extent=[step_list[0], step_list[-1], len(layer_names), 0])
            plt.colorbar(im, ax=ax, label="c value")
            ax.set_ylabel("Layer index")
        ax.set_xlabel("Step")
        ax.set_title(f"Morris's heuristic constant c = nuclear_norm / sv_sum_top{k}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"morris_heuristic_k{k}.png"), dpi=150)
        plt.close()
    print(f"  Saved morris_heuristic_k*.png")

    # =========================================================================
    # 3. Per-layer breakdown
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. PER-LAYER BREAKDOWN")
    print("=" * 80)

    layer_stats = {}
    for ln in layer_names:
        lr = [r for r in records if r["layer_name"] == ln]
        l1l2 = [r["l1_l2_ratio"] for r in lr]
        eff_rank = [r["effective_rank"] for r in lr]
        top4_frac = [r["sv_sum_top4"] / r["nuclear_norm"] for r in lr if r["nuclear_norm"] > 0]
        layer_stats[ln] = {
            "l1_l2_mean": np.mean(l1l2),
            "l1_l2_std": np.std(l1l2),
            "effective_rank_mean": np.mean(eff_rank),
            "top4_fraction_mean": np.mean(top4_frac) if top4_frac else 0,
            "shape": lr[0]["shape"] if lr else None,
        }

    # Sort by l1_l2_ratio ascending
    sorted_layers = sorted(layer_stats.items(), key=lambda x: x[1]["l1_l2_mean"])

    print(f"\n{'Layer':<55} | {'Shape':>12} | {'L1/L2 mean':>10} | {'L1/L2 std':>10} | {'EffRank':>8} | {'Top4 frac':>10}")
    print("-" * 120)
    for ln, st in sorted_layers:
        short = ln if len(ln) < 55 else "..." + ln[-(55-3):]
        shape_str = f"{st['shape'][0]}x{st['shape'][1]}" if st['shape'] else "?"
        print(f"{short:<55} | {shape_str:>12} | {st['l1_l2_mean']:>10.4f} | {st['l1_l2_std']:>10.4f} | {st['effective_rank_mean']:>8.2f} | {st['top4_fraction_mean']:>10.4f}")

    # Categorize layers
    attn_stats = [(ln, st) for ln, st in sorted_layers if any(k in ln for k in ["c_q", "c_k", "c_v", "c_proj"]) and "mlp" not in ln]
    mlp_stats = [(ln, st) for ln, st in sorted_layers if "mlp" in ln]
    other_stats = [(ln, st) for ln, st in sorted_layers if (ln, st) not in attn_stats and (ln, st) not in mlp_stats]

    for cat, items in [("Attention", attn_stats), ("MLP", mlp_stats), ("Other", other_stats)]:
        if items:
            l1l2s = [st["l1_l2_mean"] for _, st in items]
            print(f"\n  {cat} layers ({len(items)}): L1/L2 range [{min(l1l2s):.4f}, {max(l1l2s):.4f}], mean {np.mean(l1l2s):.4f}")

    # Plot per-layer breakdown
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    names_short = [ln.replace("transformer.h.", "").replace(".weight", "") for ln, _ in sorted_layers]
    x_pos = np.arange(len(sorted_layers))

    l1l2_means = [st["l1_l2_mean"] for _, st in sorted_layers]
    l1l2_stds = [st["l1_l2_std"] for _, st in sorted_layers]
    colors = []
    for ln, _ in sorted_layers:
        if any(k in ln for k in ["c_q", "c_k", "c_v", "c_proj"]) and "mlp" not in ln:
            colors.append("C0")
        elif "mlp" in ln:
            colors.append("C1")
        else:
            colors.append("C2")

    ax1.bar(x_pos, l1l2_means, yerr=l1l2_stds, color=colors, alpha=0.7, capsize=2)
    ax1.set_ylabel("L1/L2 ratio")
    ax1.set_title("Per-layer L1/L2 ratio (sorted ascending)")
    # Legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color="C0", label="Attention"), Patch(color="C1", label="MLP"), Patch(color="C2", label="Other")], fontsize=8)

    eff_ranks = [st["effective_rank_mean"] for _, st in sorted_layers]
    ax2.bar(x_pos, eff_ranks, color=colors, alpha=0.7)
    ax2.set_ylabel("Effective rank")
    ax2.set_title("Per-layer effective rank")

    top4_fracs = [st["top4_fraction_mean"] for _, st in sorted_layers]
    ax3.bar(x_pos, top4_fracs, color=colors, alpha=0.7)
    ax3.set_ylabel("Top-4 fraction")
    ax3.set_title("Per-layer top-4 SV fraction of nuclear norm")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names_short, rotation=90, fontsize=5)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "per_layer_breakdown.png"), dpi=150)
    plt.close()
    print(f"  Saved per_layer_breakdown.png")

    # =========================================================================
    # 4. Temporal stability
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. TEMPORAL STABILITY")
    print("=" * 80)

    # Pick a few representative layers: first attention, first MLP, last attention, last MLP
    representative = []
    attn_layers = [ln for ln in layer_names if ("c_q" in ln or "c_proj" in ln) and "mlp" not in ln]
    mlp_layers = [ln for ln in layer_names if "mlp" in ln and "c_fc" in ln]
    if attn_layers:
        representative.append(attn_layers[0])
        representative.append(attn_layers[-1])
    if mlp_layers:
        representative.append(mlp_layers[0])
        representative.append(mlp_layers[-1])
    # Deduplicate
    representative = list(dict.fromkeys(representative))[:6]

    # Compute step-to-step changes
    print(f"\n  Representative layers: {representative}")
    for ln in representative:
        lr = sorted([r for r in records if r["layer_name"] == ln], key=lambda r: r["step"])
        if len(lr) < 2:
            continue
        sigma_maxes = [r["sigma_max"] for r in lr]
        pct_changes = [abs(sigma_maxes[i] - sigma_maxes[i-1]) / max(abs(sigma_maxes[i-1]), 1e-10) * 100 for i in range(1, len(sigma_maxes))]
        print(f"  {ln}: sigma_max changes {np.mean(pct_changes):.2f}% per step on avg (max {np.max(pct_changes):.2f}%)")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    metrics = [("sigma_max", "Sigma max"), ("nuclear_norm", "Nuclear norm"), ("l1_l2_ratio", "L1/L2 ratio")]
    for ax, (metric, title) in zip(axes, metrics):
        for ln in representative:
            lr = sorted([r for r in records if r["layer_name"] == ln], key=lambda r: r["step"])
            xs = [r["step"] for r in lr]
            ys = [r[metric] for r in lr]
            short = ln.replace("transformer.h.", "").replace(".weight", "")
            ax.plot(xs, ys, label=short, alpha=0.7, linewidth=1.2)
        ax.set_ylabel(title)
        ax.set_title(f"{title} over training")
        ax.legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "temporal_stability.png"), dpi=150)
    plt.close()
    print(f"  Saved temporal_stability.png")

    # =========================================================================
    # 5. Power law fit
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. POWER LAW FIT: s_i = a * i^(-alpha)")
    print("=" * 80)

    alphas = []
    fit_r2s = []
    for r in records:
        svs = np.array(r["singular_values"])
        n_svs = len(svs)
        if n_svs < 4:
            continue
        indices = np.arange(1, n_svs + 1, dtype=float)
        try:
            popt, _ = curve_fit(power_law, indices, svs, p0=[svs[0], 1.0], maxfev=5000)
            alpha = popt[1]
            # R^2
            predicted = power_law(indices, *popt)
            ss_res = np.sum((svs - predicted) ** 2)
            ss_tot = np.sum((svs - svs.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            alphas.append(alpha)
            fit_r2s.append(r2)
        except (RuntimeError, ValueError):
            continue

    alphas = np.array(alphas)
    fit_r2s = np.array(fit_r2s)

    if len(alphas) > 0:
        print(f"\n  Power law exponent alpha:")
        print(f"    Mean:   {alphas.mean():.4f}")
        print(f"    Median: {np.median(alphas):.4f}")
        print(f"    Std:    {alphas.std():.4f}")
        print(f"    Min:    {alphas.min():.4f}")
        print(f"    Max:    {alphas.max():.4f}")
        print(f"  Fit R^2: mean={fit_r2s.mean():.4f}, median={np.median(fit_r2s):.4f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(alphas, bins=40, edgecolor="black", alpha=0.7)
        ax1.axvline(np.median(alphas), color="red", linestyle="--", label=f"median={np.median(alphas):.3f}")
        ax1.set_xlabel("Power law exponent alpha")
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution of power law exponent alpha")
        ax1.legend()

        ax2.hist(fit_r2s, bins=40, edgecolor="black", alpha=0.7, color="C1")
        ax2.axvline(np.median(fit_r2s), color="red", linestyle="--", label=f"median R^2={np.median(fit_r2s):.3f}")
        ax2.set_xlabel("R^2")
        ax2.set_ylabel("Count")
        ax2.set_title("Power law fit quality (R^2)")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "power_law_fit.png"), dpi=150)
        plt.close()
        print(f"  Saved power_law_fit.png")
    else:
        print("  WARNING: No successful power law fits")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    # Top-1
    top1 = np.array(top_k_fractions[1]) * 100
    print(f"\n  * Top 1 SV captures {top1.mean():.1f}% of nuclear norm on average (range: {top1.min():.1f}%-{top1.max():.1f}%)")

    # Top-4
    top4 = np.array(top_k_fractions[4]) * 100
    print(f"  * Top 4 SVs capture {top4.mean():.1f}% of nuclear norm on average (range: {top4.min():.1f}%-{top4.max():.1f}%)")

    # Top-8
    top8 = np.array(top_k_fractions[8]) * 100
    print(f"  * Top 8 SVs capture {top8.mean():.1f}% of nuclear norm on average")

    # Top-16
    top16 = np.array(top_k_fractions[16]) * 100
    print(f"  * Top 16 SVs capture {top16.mean():.1f}% of nuclear norm on average")

    # Morris constant for k=1 (sigma_1)
    c1_vals = np.array(c_values[1])
    c1_cv = c1_vals.std() / c1_vals.mean() if c1_vals.mean() > 0 else float("inf")
    stability_1 = "stable" if c1_cv < 0.15 else "not stable"
    print(f"  * The constant c in ||G||* ~ c * sigma_1 has mean {c1_vals.mean():.2f} and CV {c1_cv:.4f} -- {stability_1} enough to use as a fixed constant")

    # Morris constant for k=4
    c4_vals = np.array(c_values[4])
    c4_cv = c4_vals.std() / c4_vals.mean() if c4_vals.mean() > 0 else float("inf")
    stability_4 = "stable" if c4_cv < 0.10 else "not stable"
    print(f"  * The constant c in ||G||* ~ c * sv_sum_top4 has mean {c4_vals.mean():.3f} and CV {c4_cv:.4f} -- {stability_4} enough to use as a fixed constant")

    # Per-layer L1/L2 variation
    all_l1l2_means = [st["l1_l2_mean"] for _, st in sorted_layers]
    min_l1l2_layer = sorted_layers[0]
    max_l1l2_layer = sorted_layers[-1]

    # Determine which category has most concentrated spectra (lowest L1/L2)
    cat_means = {}
    for cat_name, items in [("attention", attn_stats), ("MLP", mlp_stats), ("other", other_stats)]:
        if items:
            cat_means[cat_name] = np.mean([st["l1_l2_mean"] for _, st in items])
    most_concentrated = min(cat_means, key=cat_means.get) if cat_means else "unknown"
    print(f"  * Per-layer L1/L2 ratio varies from {min(all_l1l2_means):.4f} to {max(all_l1l2_means):.4f}, with {most_concentrated} layers being most concentrated")

    # Temporal stability of sigma_max
    all_pct_changes = []
    for ln in layer_names:
        lr = sorted([r for r in records if r["layer_name"] == ln], key=lambda r: r["step"])
        if len(lr) < 2:
            continue
        sigma_maxes = [r["sigma_max"] for r in lr]
        for i in range(1, len(sigma_maxes)):
            pct = abs(sigma_maxes[i] - sigma_maxes[i-1]) / max(abs(sigma_maxes[i-1]), 1e-10) * 100
            all_pct_changes.append(pct)

    if all_pct_changes:
        mean_pct = np.mean(all_pct_changes)
        cacheable = "cacheable" if mean_pct < 10 else "not cacheable"
        print(f"  * Sigma_max changes by {mean_pct:.1f}% per logging interval on average -- {cacheable}")

    # Power law
    if len(alphas) > 0:
        decay = "steep" if np.median(alphas) > 0.5 else "shallow"
        print(f"  * Power law exponent alpha has median {np.median(alphas):.3f}, suggesting {decay} decay (R^2={np.median(fit_r2s):.3f})")

    print("\n" + "=" * 80)
    print("Analysis complete. Plots saved to:", args.output_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
