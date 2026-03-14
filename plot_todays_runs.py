"""Plot today's runs: diag_adam, diag_muon, sam_muon vs baseline."""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

def load_train_log(path, smooth_window=20):
    steps, losses, times = [], [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if '_config' in d:
                continue
            steps.append(d['step'])
            losses.append(d['loss'])
            times.append(d.get('total_training_time', 0))
    # Smooth losses
    if smooth_window > 1 and len(losses) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(losses, kernel, mode='valid')
        offset = smooth_window - 1
        return steps[offset:], smoothed.tolist(), times[offset:]
    return steps, losses, times

def load_val_log(path):
    steps, bpbs, times = [], [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if '_config' in d:
                continue
            steps.append(d['step'])
            bpbs.append(d['val_bpb'])
            times.append(d.get('total_training_time', 0))
    return steps, bpbs, times

runs = {
    'Baseline Muon': ('cache/base_checkpoints/baseline_muon', 'k', '--', 2.0),
    'Diag Adam': ('cache/base_checkpoints/diag_adam_d20', 'tab:blue', '-', 1.5),
    'Diag Muon': ('cache/base_checkpoints/diag_muon_d20', 'tab:orange', '-', 1.5),
    'SAM Muon (ρ=0.05)': ('cache/base_checkpoints/sam_muon_rho005', 'tab:red', '-', 1.5),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: smoothed train loss vs step
ax1 = axes[0]
for name, (path, color, ls, lw) in runs.items():
    s, l, t = load_train_log(f'{path}/train_log.jsonl', smooth_window=20)
    ax1.plot(s, l, color=color, linestyle=ls, linewidth=lw, label=name, alpha=0.85)
    # Annotate final value
    if l:
        ax1.annotate(f'{l[-1]:.3f}', xy=(s[-1], l[-1]), fontsize=7, color=color,
                     xytext=(5, 5), textcoords='offset points')

ax1.set_xlabel('Step')
ax1.set_ylabel('Train Loss (smoothed)')
ax1.set_title('Train Loss vs Step')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Right: val bpb vs step (where available)
ax2 = axes[1]
for name, (path, color, ls, lw) in runs.items():
    s, b, t = load_val_log(f'{path}/val_log.jsonl')
    if len(s) > 1:  # need at least 2 points
        ax2.plot(s, b, color=color, linestyle=ls, linewidth=lw, label=name,
                 marker='o', markersize=4, alpha=0.85)
        ax2.annotate(f'{b[-1]:.4f}', xy=(s[-1], b[-1]), fontsize=7, color=color,
                     xytext=(5, 5), textcoords='offset points')

ax2.set_xlabel('Step')
ax2.set_ylabel('Validation BPB')
ax2.set_title('Validation BPB vs Step')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.suptitle("Today's Runs: Diag Adam, Diag Muon, SAM Muon vs Baseline (896M, d20, FP8)", fontsize=11)
plt.tight_layout()
plt.savefig('plots/todays_runs.png', dpi=150, bbox_inches='tight')
print("Saved to plots/todays_runs.png")
