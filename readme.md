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
| `results.csv` | Numerical results |

## Data

The script generates synthetic request data with:
- 100,000 requests
- 22 features across 3 dimensions (identity, action, behavior)
- 4 attack categories: benign, single-dim, cross-dim, evasive

## Models

| Model | Description |
|-------|-------------|
| Rules | RBAC + payload filter + rate limiter |
| Factored | Logistic regression per dimension |
| Monotone | MLP with monotonicity structure |
| Full MLP | Unconstrained MLP |

## Expected Results

| Model | FAR | FRR | Accuracy |
|-------|-----|-----|----------|
| Rules | 0.66 | 0.00 | 0.67 |
| Factored | 0.02 | 0.26 | 0.86 |
| Monotone | 0.00 | 0.01 | 0.99 |
| Full MLP | 0.00 | 0.01 | 0.99 |

Cross-dimensional attack FAR:
- Rules: 97%
- Learned models: <1%

## License

MIT
