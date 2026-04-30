# Optimizer FLOPs Comparison: Muon vs NS3 vs Hybrid@100

**Setup:** 2205 steps, total batch size 524,288 tokens/step, 4×H100  
**Forward+backward FLOPs** estimated via standard `6 × params × tokens` formula.  
**Optimizer FLOPs** counted analytically from Newton-Schulz kernel: `4·s·r²·+ 2·r³` per iteration per matrix (r=min(m,n), s=max(m,n)).  
**FTRL correction** is elementwise only (~6·m·n FLOPs per matrix — negligible).

---

## d4 (model\_dim=256, 3.1M transformer params)

| Optimizer | Opt FLOPs / step | Fwd+Bwd / step | Total / step | Opt FLOPs (full run) | Total FLOPs (full run) | Wall time (4gpu) | Total saved vs Muon | Optimizer saved vs Muon |
|---|---|---|---|---|---|---|---|---|
| **muon** (5 NS steps) | 20.13 GFLOPs | 9,896 GFLOPs | 9,916 GFLOPs | 44.4 TFLOPs | 21,864 TFLOPs | 2.52 min | — | — |
| **ns3 only** (3 NS steps) | 12.08 GFLOPs | 9,896 GFLOPs | 9,908 GFLOPs | 26.6 TFLOPs | 21,846 TFLOPs | — | 17.8 TFLOPs (0.08%) | 17.8 TFLOPs (40.0%) |
| **hybrid@100** (ns3+ftrl → muon) | 12.10 → 20.13 GFLOPs | 9,896 GFLOPs | 9,908 GFLOPs | 43.6 TFLOPs | 21,863 TFLOPs | 2.49 min | 0.8 TFLOPs (0.004%) | 0.8 TFLOPs (1.8%) |

---

## d12 (model\_dim=768, 84.9M transformer params)

| Optimizer | Opt FLOPs / step | Fwd+Bwd / step | Total / step | Opt FLOPs (full run) | Total FLOPs (full run) | Wall time (4gpu) | Total saved vs Muon | Optimizer saved vs Muon |
|---|---|---|---|---|---|---|---|---|
| **muon** (5 NS steps) | 1,630 GFLOPs | 267,181 GFLOPs | 268,812 GFLOPs | 3,596 TFLOPs | 592,731 TFLOPs | 17.46 min | — | — |
| **ns3 only** (3 NS steps) | 978 GFLOPs | 267,181 GFLOPs | 268,160 GFLOPs | 2,157 TFLOPs | 591,292 TFLOPs | 17.41 min | 1,438 TFLOPs (0.24%) | 1,438 TFLOPs (40.0%) |
| **ns3+ftrl** (3 NS + correction) | 979 GFLOPs | 267,181 GFLOPs | 268,160 GFLOPs | 2,159 TFLOPs | 591,293 TFLOPs | — | 1,437 TFLOPs (0.24%) | 1,437 TFLOPs (40.0%) |
| **hybrid@100** (ns3+ftrl → muon) | 979 → 1,630 GFLOPs | 267,181 GFLOPs | 268,160 GFLOPs | 3,531 TFLOPs | 592,665 TFLOPs | 17.43 min | 65 TFLOPs (0.011%) | 65 TFLOPs (1.8%) |

---

## Key Takeaways

- **The optimizer is a tiny fraction of total compute** — fwd+bwd dominates completely (~0.6% optimizer share at d12, ~0.2% at d4)
- **ns3 vs muon saves ~40% of optimizer FLOPs**, but only ~0.24% of total run FLOPs at d12 (wall-clock savings are negligible: 17.41 vs 17.46 min)
- **hybrid@100 saves only 1.8% of optimizer FLOPs** — most steps still run muon (5 NS steps), so the total savings shrink to ~0.011% of the full run
- **Wall-clock times are nearly identical** across all three optimizers — the optimizer step is not the compute bottleneck; memory bandwidth and fwd+bwd are

## Notes on FLOPs formulas

**Per NS iteration on an (m, n) matrix** (r = min(m,n), s = max(m,n)):
```
FLOPs = 4·s·r² + 2·r³
```
- `X.T @ X` or `X @ X.T`: `2·s·r²`
- `A @ A`:                 `2·r³`
- `X @ B`:                 `2·s·r²`

**FTRL correction** (elementwise only, after 3 NS steps):
```
FLOPs ≈ 6·m·n  per matrix
```
Dot product ⟨Q, G⟩ + scale + blend — negligible vs matmuls.

**Forward + backward**:
```
FLOPs ≈ 6 × (non-embedding params) × (tokens per step)
```
