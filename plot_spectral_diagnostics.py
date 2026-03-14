"""Plot spectral diagnostics: Muon vs Adam overlaid."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_diag_log(path):
    steps, nuclear, entropy, frobenius, spectral = [], [], [], [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if '_config' in d:
                continue
            steps.append(d['step'])
            nuclear.append(d['nuclear_norm'])
            entropy.append(d['sv_entropy'])
            frobenius.append(d['frobenius_norm'])
            spectral.append(d['spectral_norm'])
    return steps, nuclear, entropy, frobenius, spectral

muon_path = 'cache/base_checkpoints/diag_muon_d20/spectral_diag.jsonl'
adam_path = 'cache/base_checkpoints/diag_adam_d20/spectral_diag.jsonl'

# Try to load both, handle missing gracefully
datasets = {}
for name, path, color in [('Muon', muon_path, '#e74c3c'), ('Adam', adam_path, '#3498db')]:
    try:
        datasets[name] = (load_diag_log(path), color)
        print(f"Loaded {name}: {len(datasets[name][0][0])} data points")
    except FileNotFoundError:
        print(f"Warning: {path} not found, skipping {name}")

if not datasets:
    print("No data found. Run the diagnostic training scripts first.")
    exit(1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics = [
    (0, 0, 'nuclear', 'Nuclear Norm (sum of SVs)', 'Nuclear Norm'),
    (0, 1, 'entropy', 'SV Shannon Entropy', 'Entropy'),
    (1, 0, 'frobenius', 'Frobenius Norm', 'Frobenius Norm'),
    (1, 1, 'spectral', 'Spectral Norm (largest SV)', 'Spectral Norm'),
]

for row, col, metric_key, title, ylabel in metrics:
    ax = axes[row][col]
    for name, (data, color) in datasets.items():
        steps, nuclear, entropy, frobenius, spectral = data
        values = {'nuclear': nuclear, 'entropy': entropy, 'frobenius': frobenius, 'spectral': spectral}[metric_key]
        ax.plot(steps, values, '-', color=color, label=name, linewidth=1.5, alpha=0.9)
    ax.set_xlabel('Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Spectral Diagnostics: Muon vs Adam\n(Layer 5 MLP up-projection, 896M 20-layer FP8)', fontsize=13)
plt.tight_layout()
plt.savefig('plots/spectral_diagnostics.png', dpi=150, bbox_inches='tight')
print("Saved to plots/spectral_diagnostics.png")
