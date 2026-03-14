"""Plot latest experiment results: gradnovelty + spectral sharpen vs baseline."""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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

# Load data
base_s, base_b, base_t = load_val_log('cache/base_checkpoints/baseline_muon/val_log.jsonl')
gn_s, gn_b, gn_t = load_val_log('cache/base_checkpoints/gradnovelty_d20/val_log.jsonl')
sh_s, sh_b, sh_t = load_val_log('cache/trace_norm/tn_sharpen_d20/val_log.jsonl')
se_s, se_b, se_t = load_val_log('cache/trace_norm/tn_sharp_ext_d20/val_log.jsonl')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: val bpb vs step
ax1.plot(base_s, base_b, 'k--', label='Baseline Muon', linewidth=2, alpha=0.7)
ax1.plot(gn_s, gn_b, 'r-o', label='Gradient Novelty', markersize=3, linewidth=1.5)
ax1.plot(sh_s, sh_b, 'b-s', label='Spectral Sharpen (α=1.05, K=50)', markersize=3, linewidth=1.5)
ax1.plot(se_s, se_b, 'g-^', label='Spectral Sharpen Ext (α=1.2, K=10)', markersize=3, linewidth=1.5)
ax1.set_xlabel('Step')
ax1.set_ylabel('Validation BPB')
ax1.set_title('Validation BPB vs Training Step')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.75, 1.15)

# Plot 2: val bpb vs wall time (minutes)
base_t_min = [t / 60 for t in base_t]
gn_t_min = [t / 60 for t in gn_t]
sh_t_min = [t / 60 for t in sh_t]

se_t_min = [t / 60 for t in se_t]

ax2.plot(base_t_min, base_b, 'k--', label='Baseline Muon', linewidth=2, alpha=0.7)
ax2.plot(gn_t_min, gn_b, 'r-o', label='Gradient Novelty', markersize=3, linewidth=1.5)
ax2.plot(sh_t_min, sh_b, 'b-s', label='Spectral Sharpen (α=1.05, K=50)', markersize=3, linewidth=1.5)
ax2.plot(se_t_min, se_b, 'g-^', label='Spectral Sharpen Ext (α=1.2, K=10)', markersize=3, linewidth=1.5)
ax2.set_xlabel('Wall Time (minutes)')
ax2.set_ylabel('Validation BPB')
ax2.set_title('Validation BPB vs Wall Time')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.75, 1.15)

# Add final values as annotations
ax1.annotate(f'{base_b[-1]:.4f}', xy=(base_s[-1], base_b[-1]), fontsize=8, color='black',
             xytext=(-50, 10), textcoords='offset points')
ax1.annotate(f'{gn_b[-1]:.4f}', xy=(gn_s[-1], gn_b[-1]), fontsize=8, color='red',
             xytext=(-50, 10), textcoords='offset points')
ax1.annotate(f'{sh_b[-1]:.4f}', xy=(sh_s[-1], sh_b[-1]), fontsize=8, color='blue',
             xytext=(-50, 10), textcoords='offset points')
ax1.annotate(f'{se_b[-1]:.4f}', xy=(se_s[-1], se_b[-1]), fontsize=8, color='green',
             xytext=(-50, 10), textcoords='offset points')

plt.suptitle('Latest Experiment Results (896M, 20-layer, FP8, 10.5x data ratio)', fontsize=12)
plt.tight_layout()
plt.savefig('plots/latest_results.png', dpi=150, bbox_inches='tight')
print("Saved to plots/latest_results.png")
