"""
Baseline (original Polar Express Muon) vs Affine 1.0 YES SN — both depth 20.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import re

# --- Affine 1.0 YES SN (from val_loss.json) ---
with open('/data/scratch/medhaven/nanochat/affine_sweep_results/affine_1.000_YES_SN/val_loss.json') as f:
    affine_data = json.load(f)
affine_steps = [e['step'] for e in affine_data['val_loss_log']]
affine_bpb = [e['val_bpb'] for e in affine_data['val_loss_log']]

# --- Baseline d20 (from slurm log) ---
baseline_steps = []
baseline_bpb = []
with open('/data/scratch/medhaven/nanochat/slurm_logs/baseline_d20_366604.log') as f:
    for line in f:
        m = re.search(r'Step (\d+) \| Validation bpb: ([\d.]+)', line)
        if m:
            baseline_steps.append(int(m.group(1)))
            baseline_bpb.append(float(m.group(2)))

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(baseline_steps, baseline_bpb, 'o-', color='#1565c0', lw=2.5, markersize=5,
        label=f'Baseline: Polar Express (min {min(baseline_bpb):.4f})')
ax.plot(affine_steps, affine_bpb, 's-', color='#d32f2f', lw=2.5, markersize=5,
        label=f'Affine 1.0 YES SN (min {min(affine_bpb):.4f})')

ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Validation BPB', fontsize=12)
ax.set_title('Baseline Polar Express vs Affine 1.0 (YES SN)\nBoth depth 20, same hyperparameters',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/data/scratch/medhaven/nanochat/affine_sweep_results/testing_plot.png', dpi=150)
print("Saved testing_plot.png")
