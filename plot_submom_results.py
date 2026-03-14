"""Plot subspace momentum experiment results"""
import matplotlib.pyplot as plt
import re

def parse_val(filepath):
    val_steps, val_bpb = [], []
    with open(filepath) as f:
        for line in f:
            m = re.match(r'Step (\d+) \| Validation bpb: ([\d.]+)', line)
            if m:
                val_steps.append(int(m.group(1)))
                val_bpb.append(float(m.group(2)))
    return val_steps, val_bpb

def parse_loss(filepath):
    steps, losses = [], []
    with open(filepath) as f:
        for line in f:
            m = re.match(r'step (\d+)/\d+.*loss: ([\d.]+)', line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return steps, losses

experiments = {
    'Baseline (mom=0.95)': 'slurm_logs/submom_baseline_507839.log',
    'Single ortho (mom=0.95)': 'slurm_logs/submom_single_507841.log',
    'Double ortho / mid (mom=0.90)': 'slurm_logs/submom_mid_507843.log',
    'Double ortho / low (mom=0.85)': 'slurm_logs/submom_low_507842.log',
}
# Double (mom=0.95) crashed immediately, skip it

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot 1: Validation BPB
ax = axes[0]
for (name, path), color in zip(experiments.items(), colors):
    vs, vb = parse_val(path)
    if vs:
        marker = 'o' if 'Baseline' in name else ('x' if 'Single' in name else 's')
        ax.plot(vs, vb, marker=marker, label=name, markersize=6, color=color, linewidth=1.5)
ax.set_xlabel('Step')
ax.set_ylabel('Validation BPB')
ax.set_title('Subspace Momentum - Validation BPB')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.75, 1.5)

# Plot 2: Training loss (subsampled)
ax = axes[1]
skip = 20
for (name, path), color in zip(experiments.items(), colors):
    steps, losses = parse_loss(path)
    if steps:
        ax.plot(steps[::skip], losses[::skip], alpha=0.7, label=name, linewidth=1, color=color)
ax.set_xlabel('Step')
ax.set_ylabel('Training Loss')
ax.set_title('Subspace Momentum - Training Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/submom_comparison.png', dpi=150, bbox_inches='tight')
print("Saved plots/submom_comparison.png")
