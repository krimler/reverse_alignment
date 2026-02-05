# Evaluation Code

Code to reproduce Figure 2 and Table 1 from the paper.

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

Install:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

```bash
python reproduce_figure.py
```

## Output

| File | Description |
|------|-------------|
| `figure2.pdf` | Main evaluation figure |
| `figure2.png` | PNG preview |
| `results.csv` | Numerical results (FAR, FRR, Acc, ECE per model) |
| `far_by_type.csv` | FAR broken down by attack type per model |

## Data

The script generates synthetic request data with:
- 100,000 requests (60% benign, 13% single-dim, 13% cross-dim, 14% evasive)
- 22 features across 3 dimensions (identity, action, behavior)
- 7 attack subtypes across 4 categories:

| Category | Subtypes | Description |
|----------|----------|-------------|
| Benign | — | Normal user requests |
| Single-dim | A1 (role), A2 (payload), A3 (rate) | One feature clearly anomalous |
| Cross-dim | A4 (priv+loc), A5 (payload+time), A6 (id+behav) | Each feature in normal range; combination is harmful |
| Evasive | A7 | All features near decision boundary |

## Models

All models are evaluated at a calibrated **1% FRR** threshold.

| Model | Description |
|-------|-------------|
| Rules | RBAC + payload filter + rate limiter (threshold-per-feature) |
| Factored | Logistic regression per dimension, combined with MAX |
| Monotone | MLP (32-16) with feature orientation for monotonicity |
| Full MLP | Unconstrained MLP (64-32-16) |

## Expected Results

Overall metrics at 1% FRR:

| Model | FAR | FRR | Accuracy | ECE |
|-------|-----|-----|----------|-----|
| Rules | ~0.66 | ~0.00 | ~0.67 | ~0.27 |
| Factored | ~0.02 | ~0.01 | ~0.99 | ~0.01 |
| Monotone | ~0.00 | ~0.01 | ~0.99 | ~0.01 |
| Full MLP | ~0.00 | ~0.01 | ~0.99 | ~0.01 |

FAR by attack type:

| Model | Single-dim | Cross-dim | Evasive |
|-------|-----------|-----------|---------|
| Rules | Low | ~97% | ~97% |
| Factored | <1% | Low | <1% |
| Monotone | <1% | <1% | <1% |
| Full MLP | <1% | <1% | <1% |

Key takeaway: Rules miss cross-dimensional and evasive attacks entirely. The Factored model (MAX aggregation) catches single-dim signals but can still miss subtle cross-dim patterns where no individual dimension is anomalous.

## License

MIT
