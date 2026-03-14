"""Plot nuclear norm experiment results: tn_nuclear vs tn_nuc_every"""
import matplotlib.pyplot as plt
import re
import numpy as np

def parse_log(filepath):
    steps, losses, val_steps, val_bpb = [], [], [], []
    with open(filepath) as f:
        for line in f:
            m = re.match(r'step (\d+)/\d+.*loss: ([\d.]+)', line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
            m = re.match(r'Step (\d+) \| Validation bpb: ([\d.]+)', line)
            if m:
                val_steps.append(int(m.group(1)))
                val_bpb.append(float(m.group(2)))
    return steps, losses, val_steps, val_bpb

# Parse logs
nuc_steps, nuc_loss, nuc_vs, nuc_vb = parse_log('slurm_logs/tn_nuclear_503812.log')
eve_steps, eve_loss, eve_vs, eve_vb = parse_log('slurm_logs/tn_nuc_every_507112.log')
bas_steps, bas_loss, bas_vs, bas_vb = parse_log('slurm_logs/submom_baseline_507839.log')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Training loss vs step
ax = axes[0]
# Subsample for clarity
skip = 5
ax.plot(bas_steps[::skip], bas_loss[::skip], alpha=0.5, label='Baseline (standard Muon)', linewidth=1.5, linestyle='--', color='gray')
ax.plot(nuc_steps[::skip], nuc_loss[::skip], alpha=0.7, label=f'Nuclear every 10 steps (step {nuc_steps[-1]})', linewidth=1)
ax.plot(eve_steps[::skip], eve_loss[::skip], alpha=0.7, label=f'Nuclear every step (step {eve_steps[-1]})', linewidth=1)
ax.set_xlabel('Step')
ax.set_ylabel('Training Loss')
ax.set_title('Nuclear Norm Experiments - Training Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Validation BPB
ax = axes[1]
ax.plot(bas_vs, bas_vb, 'd--', label='Baseline (standard Muon)', markersize=5, color='gray', alpha=0.7)
ax.plot(nuc_vs, nuc_vb, 'o-', label='Nuclear every 10 steps', markersize=8)
ax.plot(eve_vs, eve_vb, 's-', label='Nuclear every step', markersize=8)
ax.set_xlabel('Step')
ax.set_ylabel('Validation BPB')
ax.set_title('Nuclear Norm Experiments - Validation BPB')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/nuclear_norm_comparison.png', dpi=150, bbox_inches='tight')
print("Saved plots/nuclear_norm_comparison.png")
