"""
Tiered gradient spectrum analysis.
Produces:
  - Tier 1: tables written to a text file
  - Tier 2: decision-critical plots
  - Tier 3: supplementary plots
"""

import os
import json
import argparse
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
    parser.add_argument("--table-file", type=str, required=True, help="Path for tier-1 data tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = load_jsonl(args.input)
    steps = sorted(set(r["step"] for r in records))
    layer_names = sorted(set(r["layer_name"] for r in records))

    # =====================================================================
    # Helpers
    # =====================================================================
    def layer_type(name):
        if "ve_gate" in name:
            return "ve_gate"
        if "mlp" in name:
            if "c_fc" in name:
                return "MLP up"
            elif "c_proj" in name:
                return "MLP down"
            return "MLP other"
        if "c_q" in name:
            return "attn Q"
        if "c_k" in name:
            return "attn K"
        if "c_v" in name:
            return "attn V"
        if "c_proj" in name:
            return "attn proj"
        return "other"

    def short_name(name):
        return name.replace("transformer.h.", "").replace(".weight", "")

    out_lines = []
    def tprint(s=""):
        print(s)
        out_lines.append(s)

    # =====================================================================
    # TIER 1
    # =====================================================================
    tprint("=" * 90)
    tprint("TIER 1: DECISION-CRITICAL DATA")
    tprint("=" * 90)

    # --- 1a. Top-k fraction statistics ---
    tprint()
    tprint("1a. TOP-K FRACTION: sv_sum_topk / nuclear_norm")
    tprint("    (How concentrated is the gradient spectrum?)")
    tprint()
    ks = [1, 2, 4, 8, 16]
    top_k_fracs = {k: [] for k in ks}
    for r in records:
        nn = r["nuclear_norm"]
        if nn <= 0:
            continue
        for k in ks:
            top_k_fracs[k].append(r[f"sv_sum_top{k}"] / nn)

    tprint(f"  {'k':>4}  {'mean':>8}  {'median':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    tprint("  " + "-" * 52)
    for k in ks:
        v = np.array(top_k_fracs[k])
        tprint(f"  {k:>4}  {v.mean():>8.4f}  {np.median(v):>8.4f}  {v.std():>8.4f}  {v.min():>8.4f}  {v.max():>8.4f}")

    # --- 1b. Morris's constant c ---
    tprint()
    tprint("1b. MORRIS'S CONSTANT: c = nuclear_norm / sv_sum_topk")
    tprint("    (Is c stable enough to use as a fixed multiplier?)")
    tprint("    CV < 10% => works as-is | 10-30% => per-layer calibration | >30% => heuristic doesn't hold")
    tprint()
    for k in [1, 4]:
        c_all = []
        c_by_type = defaultdict(list)
        for r in records:
            topk = r[f"sv_sum_top{k}"]
            if topk > 0:
                c = r["nuclear_norm"] / topk
                c_all.append(c)
                c_by_type[layer_type(r["layer_name"])].append(c)
        c_all = np.array(c_all)
        cv = c_all.std() / c_all.mean()
        tprint(f"  k={k} (global, all layers pooled):")
        tprint(f"    mean={c_all.mean():.3f}  median={np.median(c_all):.3f}  std={c_all.std():.3f}  "
               f"min={c_all.min():.3f}  max={c_all.max():.3f}  CV={cv:.4f} ({cv*100:.1f}%)")
        tprint()
        tprint(f"  k={k} (broken out by layer type):")
        tprint(f"    {'type':<14}  {'mean':>8}  {'median':>8}  {'std':>8}  {'CV':>8}  {'n':>5}")
        tprint("    " + "-" * 54)
        for lt in sorted(c_by_type.keys()):
            vals = np.array(c_by_type[lt])
            cv_lt = vals.std() / vals.mean()
            tprint(f"    {lt:<14}  {vals.mean():>8.3f}  {np.median(vals):>8.3f}  {vals.std():>8.3f}  "
                   f"{cv_lt:>8.4f}  {len(vals):>5}")
        tprint()

    # --- 1c. Per-layer table ---
    tprint()
    tprint("1c. PER-LAYER TABLE (sorted by L1/L2 ascending)")
    tprint("    L1/L2 = nuclear_norm / frobenius_norm.  Low = concentrated spectrum.  High = flat/diffuse.")
    tprint()

    layer_stats = {}
    for ln in layer_names:
        lr = [r for r in records if r["layer_name"] == ln]
        l1l2 = [r["l1_l2_ratio"] for r in lr]
        eff_rank = [r["effective_rank"] for r in lr]
        top4_frac = [r["sv_sum_top4"] / r["nuclear_norm"] for r in lr if r["nuclear_norm"] > 0]
        layer_stats[ln] = {
            "l1_l2_mean": np.mean(l1l2),
            "l1_l2_std": np.std(l1l2),
            "eff_rank_mean": np.mean(eff_rank),
            "top4_frac_mean": np.mean(top4_frac) if top4_frac else 0,
            "shape": lr[0]["shape"] if lr else None,
            "type": layer_type(ln),
        }

    sorted_layers = sorted(layer_stats.items(), key=lambda x: x[1]["l1_l2_mean"])

    tprint(f"  {'layer':<50} {'shape':>10} {'type':<12} {'L1/L2':>7} {'effRank':>8} {'top4%':>7}")
    tprint("  " + "-" * 100)
    for ln, st in sorted_layers:
        sh = f"{st['shape'][0]}x{st['shape'][1]}" if st['shape'] else "?"
        tprint(f"  {short_name(ln):<50} {sh:>10} {st['type']:<12} "
               f"{st['l1_l2_mean']:>7.2f} {st['eff_rank_mean']:>8.2f} {st['top4_frac_mean']*100:>6.1f}%")

    # Summary by type
    tprint()
    tprint("  Summary by layer type:")
    type_groups = defaultdict(list)
    for ln, st in sorted_layers:
        type_groups[st["type"]].append(st)
    for lt in ["ve_gate", "attn V", "attn proj", "attn Q", "attn K", "MLP up", "MLP down"]:
        if lt not in type_groups:
            continue
        items = type_groups[lt]
        l1l2s = [s["l1_l2_mean"] for s in items]
        top4s = [s["top4_frac_mean"] for s in items]
        tprint(f"    {lt:<14}  n={len(items):>2}  L1/L2: {np.mean(l1l2s):>6.2f} [{min(l1l2s):.2f}, {max(l1l2s):.2f}]"
               f"  top4: {np.mean(top4s)*100:>5.1f}% [{min(top4s)*100:.1f}%, {max(top4s)*100:.1f}%]")

    # Write table file
    with open(args.table_file, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\n  => Tier 1 tables written to {args.table_file}")

    # =====================================================================
    # TIER 2: DECISION-CRITICAL PLOTS
    # =====================================================================
    print()
    print("=" * 90)
    print("TIER 2: DECISION-CRITICAL PLOTS")
    print("=" * 90)

    # --- 2a. Histogram of top-k fractions ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, k in zip(axes, [1, 2, 4, 8]):
        vals = np.array(top_k_fracs[k])
        ax.hist(vals, bins=50, edgecolor="black", linewidth=0.3, alpha=0.8, color=f"C{[1,2,4,8].index(k)}")
        ax.axvline(vals.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={vals.mean():.3f}")
        ax.axvline(np.median(vals), color="black", linestyle=":", linewidth=1.5, label=f"med={np.median(vals):.3f}")
        ax.set_title(f"top-{k} fraction", fontsize=12)
        ax.set_xlabel(f"sv_sum_top{k} / nuclear_norm")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Count")
    plt.suptitle("Spectrum Concentration: Distribution of top-k SV fraction across all (step, layer) pairs", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "tier2_topk_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved tier2_topk_histogram.png")

    # --- 2b. c vs step, one line per layer, for k=1 and k=4 ---
    for k in [1, 4]:
        fig, ax = plt.subplots(figsize=(14, 7))
        # Group by layer type for coloring
        type_colors = {"ve_gate": "C5", "attn V": "C0", "attn Q": "C1", "attn K": "C2",
                       "attn proj": "C3", "MLP up": "C4", "MLP down": "C6", "other": "C7"}
        plotted_types = set()
        for ln in layer_names:
            lr = sorted([r for r in records if r["layer_name"] == ln], key=lambda r: r["step"])
            xs = [r["step"] for r in lr]
            ys = [r["nuclear_norm"] / r[f"sv_sum_top{k}"] if r[f"sv_sum_top{k}"] > 0 else 0 for r in lr]
            lt = layer_type(ln)
            color = type_colors.get(lt, "gray")
            label = lt if lt not in plotted_types else None
            plotted_types.add(lt)
            ax.plot(xs, ys, alpha=0.4, linewidth=0.9, color=color, label=label)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel(f"c = nuclear_norm / sv_sum_top{k}", fontsize=12)
        ax.set_title(f"Morris's constant c vs training step  (k={k})\nEach line = one layer, colored by type", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"tier2_morris_c_vs_step_k{k}.png"), dpi=150)
        plt.close()
    print("  Saved tier2_morris_c_vs_step_k1.png, tier2_morris_c_vs_step_k4.png")

    # --- 2c. Sigma_max temporal stability ---
    # Pick: early attn (layer 0 c_proj), late attn (layer 11 c_q), early MLP (layer 0 c_fc),
    #        late MLP (layer 11 c_fc), and layer 9 MLP down-proj
    representative = [
        "transformer.h.0.attn.c_proj.weight",
        "transformer.h.11.attn.c_q.weight",
        "transformer.h.0.mlp.c_fc.weight",
        "transformer.h.11.mlp.c_fc.weight",
        "transformer.h.11.mlp.c_proj.weight",
    ]
    # Filter to those that exist
    representative = [ln for ln in representative if ln in layer_names]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ln in representative:
        lr = sorted([r for r in records if r["layer_name"] == ln], key=lambda r: r["step"])
        xs = [r["step"] for r in lr]
        ys = [r["sigma_max"] for r in lr]
        label = short_name(ln)
        ax1.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, label=label)

        # Pct change
        pct = [abs(ys[i] - ys[i-1]) / max(abs(ys[i-1]), 1e-15) * 100 for i in range(1, len(ys))]
        ax2.plot(xs[1:], pct, marker="o", markersize=3, linewidth=1.2, label=f"{label} (avg={np.mean(pct):.1f}%)")

    ax1.set_ylabel("sigma_max")
    ax1.set_title("Sigma_max over training (representative layers)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_ylabel("% change between logging intervals")
    ax2.set_xlabel("Step")
    ax2.set_title("Step-to-step % change in sigma_max (< 5% = cacheable)")
    ax2.axhline(5, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label="5% threshold")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "tier2_sigma_max_stability.png"), dpi=150)
    plt.close()
    print("  Saved tier2_sigma_max_stability.png")

    # =====================================================================
    # TIER 3: SUPPLEMENTARY PLOTS
    # =====================================================================
    print()
    print("=" * 90)
    print("TIER 3: SUPPLEMENTARY PLOTS")
    print("=" * 90)

    # --- 3a. Power law exponent alpha distribution ---
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(alphas, bins=50, edgecolor="black", linewidth=0.3, alpha=0.8)
        ax1.axvline(np.median(alphas), color="red", linestyle="--", label=f"median={np.median(alphas):.3f}")
        ax1.set_xlabel("Power law exponent α")
        ax1.set_ylabel("Count")
        ax1.set_title("Distribution of α  (s_i = a · i^(-α))")
        ax1.legend()

        ax2.hist(fit_r2s, bins=50, edgecolor="black", linewidth=0.3, alpha=0.8, color="C1")
        ax2.axvline(np.median(fit_r2s), color="red", linestyle="--", label=f"median R²={np.median(fit_r2s):.3f}")
        ax2.set_xlabel("R²")
        ax2.set_ylabel("Count")
        ax2.set_title("Power law fit quality")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "tier3_power_law_alpha.png"), dpi=150)
        plt.close()
        print(f"  Saved tier3_power_law_alpha.png  (median α={np.median(alphas):.3f}, median R²={np.median(fit_r2s):.3f})")

    # --- 3b. L1/L2 ratio vs step, one line per layer ---
    fig, ax = plt.subplots(figsize=(14, 7))
    type_colors = {"ve_gate": "C5", "attn V": "C0", "attn Q": "C1", "attn K": "C2",
                   "attn proj": "C3", "MLP up": "C4", "MLP down": "C6", "other": "C7"}
    plotted_types = set()
    for ln in layer_names:
        lr = sorted([r for r in records if r["layer_name"] == ln], key=lambda r: r["step"])
        xs = [r["step"] for r in lr]
        ys = [r["l1_l2_ratio"] for r in lr]
        lt = layer_type(ln)
        color = type_colors.get(lt, "gray")
        label = lt if lt not in plotted_types else None
        plotted_types.add(lt)
        ax.plot(xs, ys, alpha=0.4, linewidth=0.9, color=color, label=label)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("L1/L2 ratio (nuclear_norm / frobenius_norm)", fontsize=12)
    ax.set_title("L1/L2 ratio over training\nLow = concentrated spectrum, High = diffuse spectrum", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "tier3_l1l2_vs_step.png"), dpi=150)
    plt.close()
    print("  Saved tier3_l1l2_vs_step.png")

    print()
    print("=" * 90)
    print("ALL DONE. Files:")
    print(f"  Tables:  {args.table_file}")
    print(f"  Plots:   {args.output_dir}/")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".png"):
            print(f"           - {f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
