"""
Plot results from the affine SVD mapping sweep experiment.

Generates:
  1. val_loss_without_sn.png - All intercept curves WITHOUT spectral normalization
  2. val_loss_with_sn.png   - All intercept curves WITH spectral normalization
  3. val_loss_combined.png   - Side-by-side comparison
  4. final_bpb_summary.png  - Bar chart of final val bpb per intercept, grouped by SN mode

Usage:
  python -m scripts.affine_sweep_plot --results-dir affine_sweep_results
"""

import os
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

parser = argparse.ArgumentParser(description="Plot affine sweep results")
parser.add_argument("--results-dir", type=str, default="affine_sweep_results", help="Directory with sweep results")
args = parser.parse_args()

results_dir = args.results_dir

# =============================================================================
# Load all results
# =============================================================================

results = {"yes": {}, "no": {}}  # sn_mode -> {intercept: data}

for entry in sorted(os.listdir(results_dir)):
    val_loss_file = os.path.join(results_dir, entry, "val_loss.json")
    if not os.path.isfile(val_loss_file):
        continue
    with open(val_loss_file) as f:
        data = json.load(f)
    sn_mode = data["sn_mode"]
    intercept = data["affine_intercept"]
    results[sn_mode][intercept] = data

print(f"Loaded results:")
print(f"  YES_SN: {sorted(results['yes'].keys())} ({len(results['yes'])} runs)")
print(f"  NO_SN:  {sorted(results['no'].keys())} ({len(results['no'])} runs)")

if not results["yes"] and not results["no"]:
    print("No results found. Exiting.")
    exit(0)

# =============================================================================
# Color map: each intercept gets a consistent color across plots
# =============================================================================

all_intercepts = sorted(set(list(results["yes"].keys()) + list(results["no"].keys())))
cmap = cm.get_cmap("tab10" if len(all_intercepts) <= 10 else "tab20", len(all_intercepts))
color_map = {intercept: cmap(i) for i, intercept in enumerate(all_intercepts)}


def plot_val_loss_curves(sn_mode, title, filename):
    """Plot val_loss vs step for all intercepts in one SN mode."""
    data_dict = results[sn_mode]
    if not data_dict:
        print(f"  No data for {sn_mode}, skipping {filename}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for intercept in sorted(data_dict.keys()):
        data = data_dict[intercept]
        log = data["val_loss_log"]
        steps = [entry["step"] for entry in log]
        bpbs = [entry["val_bpb"] for entry in log]
        label = f"i={intercept}"
        if intercept == 0.0:
            label += " (identity)"
        ax.plot(steps, bpbs, label=label, color=color_map[intercept], linewidth=2, alpha=0.85)

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Val BPB (bits per byte)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    filepath = os.path.join(results_dir, filename)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filepath}")


# =============================================================================
# Plot 1: Without SN
# =============================================================================
print("\nGenerating plots...")
plot_val_loss_curves("no", "Val Loss vs Step — Without Spectral Normalization (NO_SN)", "val_loss_without_sn.png")

# =============================================================================
# Plot 2: With SN
# =============================================================================
plot_val_loss_curves("yes", "Val Loss vs Step — With Spectral Normalization (YES_SN)", "val_loss_with_sn.png")

# =============================================================================
# Plot 3: Combined side-by-side
# =============================================================================

if results["no"] and results["yes"]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7), sharey=True)

    for intercept in sorted(results["no"].keys()):
        data = results["no"][intercept]
        log = data["val_loss_log"]
        steps = [e["step"] for e in log]
        bpbs = [e["val_bpb"] for e in log]
        label = f"i={intercept}" + (" (identity)" if intercept == 0.0 else "")
        ax1.plot(steps, bpbs, label=label, color=color_map[intercept], linewidth=2, alpha=0.85)

    ax1.set_xlabel("Step", fontsize=13)
    ax1.set_ylabel("Val BPB", fontsize=13)
    ax1.set_title("Without SN", fontsize=15, fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    for intercept in sorted(results["yes"].keys()):
        data = results["yes"][intercept]
        log = data["val_loss_log"]
        steps = [e["step"] for e in log]
        bpbs = [e["val_bpb"] for e in log]
        label = f"i={intercept}" + (" (identity)" if intercept == 0.0 else "")
        ax2.plot(steps, bpbs, label=label, color=color_map[intercept], linewidth=2, alpha=0.85)

    ax2.set_xlabel("Step", fontsize=13)
    ax2.set_title("With SN", fontsize=15, fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Affine SVD Mapping Sweep: Effect of Spectral Normalization", fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    filepath = os.path.join(results_dir, "val_loss_combined.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filepath}")

# =============================================================================
# Plot 4: Final BPB summary bar chart
# =============================================================================

if results["no"] or results["yes"]:
    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.35
    x_indices = np.arange(len(all_intercepts))

    # Get final BPB for each intercept and SN mode
    no_sn_finals = []
    yes_sn_finals = []
    for intercept in all_intercepts:
        if intercept in results["no"]:
            log = results["no"][intercept]["val_loss_log"]
            no_sn_finals.append(log[-1]["val_bpb"] if log else None)
        else:
            no_sn_finals.append(None)
        if intercept in results["yes"]:
            log = results["yes"][intercept]["val_loss_log"]
            yes_sn_finals.append(log[-1]["val_bpb"] if log else None)
        else:
            yes_sn_finals.append(None)

    # Plot bars
    for idx, (intercept, no_val, yes_val) in enumerate(zip(all_intercepts, no_sn_finals, yes_sn_finals)):
        if no_val is not None:
            ax.bar(idx - bar_width/2, no_val, bar_width, color="salmon", edgecolor="darkred", linewidth=0.5,
                   label="NO_SN" if idx == 0 else "")
        if yes_val is not None:
            ax.bar(idx + bar_width/2, yes_val, bar_width, color="steelblue", edgecolor="navy", linewidth=0.5,
                   label="YES_SN" if idx == 0 else "")

    ax.set_xlabel("Affine Intercept (i)", fontsize=13)
    ax.set_ylabel("Final Val BPB", fontsize=13)
    ax.set_title("Final Validation BPB by Intercept and SN Mode", fontsize=15, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{i:.3f}" for i in all_intercepts], fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    filepath = os.path.join(results_dir, "final_bpb_summary.png")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filepath}")

# =============================================================================
# Print summary table
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Final Val BPB")
print("=" * 70)
print(f"{'Intercept':>12} {'NO_SN':>12} {'YES_SN':>12} {'Diff':>12}")
print("-" * 70)

for intercept in all_intercepts:
    no_val = None
    yes_val = None
    if intercept in results["no"]:
        log = results["no"][intercept]["val_loss_log"]
        if log:
            no_val = log[-1]["val_bpb"]
    if intercept in results["yes"]:
        log = results["yes"][intercept]["val_loss_log"]
        if log:
            yes_val = log[-1]["val_bpb"]

    no_str = f"{no_val:.6f}" if no_val else "N/A"
    yes_str = f"{yes_val:.6f}" if yes_val else "N/A"
    diff_str = f"{no_val - yes_val:.6f}" if (no_val and yes_val) else "N/A"
    print(f"{intercept:>12.3f} {no_str:>12} {yes_str:>12} {diff_str:>12}")

print("=" * 70)

# Spread analysis
if results["no"] and len(results["no"]) > 1:
    no_finals = [results["no"][i]["val_loss_log"][-1]["val_bpb"]
                 for i in sorted(results["no"].keys()) if results["no"][i]["val_loss_log"]]
    no_spread = max(no_finals) - min(no_finals)
    print(f"\nNO_SN  spread (max - min): {no_spread:.6f}")

if results["yes"] and len(results["yes"]) > 1:
    yes_finals = [results["yes"][i]["val_loss_log"][-1]["val_bpb"]
                  for i in sorted(results["yes"].keys()) if results["yes"][i]["val_loss_log"]]
    yes_spread = max(yes_finals) - min(yes_finals)
    print(f"YES_SN spread (max - min): {yes_spread:.6f}")

if results["no"] and results["yes"] and len(results["no"]) > 1 and len(results["yes"]) > 1:
    no_finals = [results["no"][i]["val_loss_log"][-1]["val_bpb"]
                 for i in sorted(results["no"].keys()) if results["no"][i]["val_loss_log"]]
    yes_finals = [results["yes"][i]["val_loss_log"][-1]["val_bpb"]
                  for i in sorted(results["yes"].keys()) if results["yes"][i]["val_loss_log"]]
    no_spread = max(no_finals) - min(no_finals)
    yes_spread = max(yes_finals) - min(yes_finals)

    print(f"\nConclusion:")
    if yes_spread < no_spread * 0.5:
        print(f"  YES_SN curves COLLAPSE together (spread {yes_spread:.6f}) while")
        print(f"  NO_SN curves DIVERGE (spread {no_spread:.6f}).")
        print(f"  => SN plays a dominant role, consistent with modded-nanogpt findings.")
    elif yes_spread < no_spread:
        print(f"  YES_SN has less spread ({yes_spread:.6f}) than NO_SN ({no_spread:.6f}).")
        print(f"  => SN reduces sensitivity to eigenvalue mapping, partially consistent.")
    else:
        print(f"  YES_SN spread ({yes_spread:.6f}) >= NO_SN spread ({no_spread:.6f}).")
        print(f"  => Different pattern from modded-nanogpt. Further investigation needed.")

print("\nDone!")
