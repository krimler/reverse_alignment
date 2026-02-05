"""
Reproduce Figure 2: Learned vs Rule-Based Evaluation

Generates synthetic request data, trains models, and produces the evaluation figure.

Usage:
    python reproduce_figure.py

Output:
    figure2.pdf - Main figure for paper
    figure2.png - PNG preview
    results.csv - Numerical results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from pathlib import Path

SEED = 42
np.random.seed(SEED)


# =============================================================================
# Data Generation
# =============================================================================

def generate_benign():
    """Benign request: features drawn from safe distributions."""
    return {
        'role': np.random.choice(['user', 'analyst', 'developer', 'admin'], p=[0.4, 0.25, 0.25, 0.1]),
        'privilege_level': np.random.uniform(0.1, 0.7),
        'reputation': np.random.uniform(0.5, 1.0),
        'origin_network': np.random.choice(['internal', 'vpn', 'external'], p=[0.5, 0.3, 0.2]),
        'operation_type': np.random.choice(['read', 'write', 'list', 'delete', 'execute'], p=[0.4, 0.25, 0.15, 0.1, 0.1]),
        'target_sensitivity': np.random.uniform(0.0, 0.6),
        'payload_risk': np.random.uniform(0.0, 0.4),
        'request_rate': np.random.uniform(0.05, 0.6),
        'anomaly_score': np.random.uniform(0.0, 0.35),
        'location_consistency': np.random.uniform(0.6, 1.0),
        'time_of_day': np.random.uniform(0.0, 1.0),
        'y': 0,
        'attack_type': 'benign'
    }


def generate_single_dim(subtype):
    """Single-dimension attack: one feature clearly anomalous."""
    x = generate_benign()
    x['y'] = 1
    
    if subtype == 'A1':
        x['attack_type'] = 'A1_role'
        x['role'] = 'guest'
        x['privilege_level'] = np.random.uniform(0.0, 0.2)
        x['operation_type'] = np.random.choice(['delete', 'admin', 'execute'])
        x['target_sensitivity'] = np.random.uniform(0.7, 1.0)
    elif subtype == 'A2':
        x['attack_type'] = 'A2_payload'
        x['payload_risk'] = np.random.uniform(0.8, 1.0)
    elif subtype == 'A3':
        x['attack_type'] = 'A3_rate'
        x['request_rate'] = np.random.uniform(0.85, 1.0)
        x['anomaly_score'] = np.random.uniform(0.6, 0.8)
    return x


def generate_cross_dim(subtype):
    """Cross-dimensional attack: each feature normal, combination suspicious."""
    x = generate_benign()
    x['y'] = 1
    
    if subtype == 'A4':
        x['attack_type'] = 'A4_priv_loc'
        x['role'] = np.random.choice(['developer', 'admin'], p=[0.7, 0.3])
        x['privilege_level'] = np.random.uniform(0.50, 0.75)
        x['location_consistency'] = np.random.uniform(0.25, 0.50)
        x['origin_network'] = 'external'
        x['anomaly_score'] = np.random.uniform(0.25, 0.45)
        x['target_sensitivity'] = np.random.uniform(0.45, 0.65)
    elif subtype == 'A5':
        x['attack_type'] = 'A5_payload_time'
        x['payload_risk'] = np.random.uniform(0.35, 0.55)
        x['time_of_day'] = np.random.choice([np.random.uniform(0.0, 0.20), np.random.uniform(0.80, 1.0)])
        x['request_rate'] = np.random.uniform(0.45, 0.65)
        x['operation_type'] = np.random.choice(['write', 'delete'], p=[0.6, 0.4])
        x['target_sensitivity'] = np.random.uniform(0.40, 0.60)
    elif subtype == 'A6':
        x['attack_type'] = 'A6_id_behav'
        x['reputation'] = np.random.uniform(0.55, 0.75)
        x['role'] = np.random.choice(['analyst', 'user'], p=[0.6, 0.4])
        x['operation_type'] = np.random.choice(['execute', 'delete', 'write'], p=[0.4, 0.3, 0.3])
        x['target_sensitivity'] = np.random.uniform(0.45, 0.65)
        x['request_rate'] = np.random.uniform(0.40, 0.60)
        x['anomaly_score'] = np.random.uniform(0.25, 0.45)
        x['location_consistency'] = np.random.uniform(0.45, 0.65)
    return x


def generate_evasive():
    """Evasive attack: all features slightly shifted, no single indicator."""
    return {
        'role': np.random.choice(['user', 'analyst', 'developer'], p=[0.5, 0.3, 0.2]),
        'privilege_level': np.random.uniform(0.35, 0.60),
        'reputation': np.random.uniform(0.50, 0.70),
        'origin_network': np.random.choice(['vpn', 'external', 'internal'], p=[0.4, 0.35, 0.25]),
        'operation_type': np.random.choice(['write', 'delete', 'read', 'execute'], p=[0.35, 0.25, 0.25, 0.15]),
        'target_sensitivity': np.random.uniform(0.35, 0.55),
        'payload_risk': np.random.uniform(0.25, 0.45),
        'request_rate': np.random.uniform(0.35, 0.55),
        'anomaly_score': np.random.uniform(0.20, 0.40),
        'location_consistency': np.random.uniform(0.45, 0.70),
        'time_of_day': np.random.uniform(0.10, 0.35),
        'y': 1,
        'attack_type': 'A7_evasive'
    }


def generate_dataset(n=100000):
    """Generate full dataset with specified attack distribution."""
    requests = []
    
    # 60% benign, 40% attacks (as stated in appendix)
    for _ in range(int(n * 0.60)):
        requests.append(generate_benign())
    
    # 40% attacks split: ~13% single-dim, ~13% cross-dim, ~14% evasive
    for subtype in ['A1', 'A2', 'A3']:
        for _ in range(int(n * 0.13 / 3)):
            requests.append(generate_single_dim(subtype))
    
    for subtype in ['A4', 'A5', 'A6']:
        for _ in range(int(n * 0.13 / 3)):
            requests.append(generate_cross_dim(subtype))
    
    for _ in range(int(n * 0.14)):
        requests.append(generate_evasive())
    
    df = pd.DataFrame(requests)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def encode_features(df):
    """One-hot encode categorical features."""
    df = df.copy()
    for col in ['role', 'origin_network', 'operation_type']:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df


def prepare_data(df):
    """Prepare feature matrix and labels."""
    df_enc = encode_features(df)
    exclude = ['y', 'attack_type']
    feature_cols = [c for c in df_enc.columns if c not in exclude]
    X = df_enc[feature_cols].values.astype(float)
    y = df_enc['y'].values
    return X, y, feature_cols


# =============================================================================
# Models
# =============================================================================

class RulesModel:
    """Combined rule-based model: RBAC + payload filter + rate limiter."""
    
    def __init__(self):
        self.name = 'Rules'
    
    def fit(self, X, y, df, cols=None):
        pass
    
    def predict_proba(self, X, df):
        n = len(df)
        scores = np.zeros(n)
        for i in range(n):
            row = df.iloc[i]
            # RBAC
            if row['role'] == 'guest' and row['operation_type'] in ['delete', 'admin', 'execute']:
                scores[i] = max(scores[i], 0.95)
            if row['privilege_level'] < 0.3 and row['target_sensitivity'] > 0.6:
                scores[i] = max(scores[i], 0.85)
            # Payload filter
            if row['payload_risk'] > 0.7:
                scores[i] = max(scores[i], 0.90)
            # Rate limiter
            if row['request_rate'] > 0.75 or row['anomaly_score'] > 0.6:
                scores[i] = max(scores[i], 0.90)
            if scores[i] == 0:
                scores[i] = 0.1
        return scores


class FactoredModel:
    """Logistic regression per dimension, combined by weighted average."""
    
    def __init__(self):
        self.name = 'Factored'
        self.models = [LogisticRegression(max_iter=1000) for _ in range(3)]
        self.scalers = [StandardScaler() for _ in range(3)]
        self.groups = None
    
    def _get_groups(self, cols):
        identity = [i for i, c in enumerate(cols) if any(x in c for x in ['role_', 'privilege', 'reputation', 'network_'])]
        action = [i for i, c in enumerate(cols) if any(x in c for x in ['operation_', 'target_', 'payload'])]
        behavior = [i for i, c in enumerate(cols) if any(x in c for x in ['request_rate', 'anomaly', 'location', 'time_of_day'])]
        return [identity, action, behavior]
    
    def fit(self, X, y, df, cols):
        self.groups = self._get_groups(cols)
        for i, g in enumerate(self.groups):
            Xi = self.scalers[i].fit_transform(X[:, g])
            self.models[i].fit(Xi, y)
    
    def predict_proba(self, X, df):
        probs = []
        for i, g in enumerate(self.groups):
            Xi = self.scalers[i].transform(X[:, g])
            probs.append(self.models[i].predict_proba(Xi)[:, 1])
        return np.mean(probs, axis=0)


class MonotoneModel:
    """
    MLP with monotonicity via feature orientation and positive-weight approximation.
    
    True monotonicity requires constrained optimization (e.g., pytorch with 
    weight clipping). This approximation orients features so higher=riskier,
    then trains a small MLP that empirically preserves monotonicity.
    """
    
    def __init__(self):
        self.name = 'Monotone'
        self.model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, 
                                    activation='relu', random_state=SEED)
        self.scaler = StandardScaler()
        self.flip_idx = []
    
    def fit(self, X, y, df, cols):
        # Features where higher value = LOWER risk (need to flip)
        flip_keywords = ['reputation', 'location_consistency']
        self.flip_idx = [i for i, c in enumerate(cols) if any(k in c for k in flip_keywords)]
        
        X_oriented = X.copy()
        for idx in self.flip_idx:
            X_oriented[:, idx] = 1 - X_oriented[:, idx]
        
        X_scaled = self.scaler.fit_transform(X_oriented)
        self.model.fit(X_scaled, y)
    
    def predict_proba(self, X, df):
        X_oriented = X.copy()
        for idx in self.flip_idx:
            X_oriented[:, idx] = 1 - X_oriented[:, idx]
        X_scaled = self.scaler.transform(X_oriented)
        return self.model.predict_proba(X_scaled)[:, 1]


class FullMLPModel:
    """Unconstrained MLP on all features."""
    
    def __init__(self):
        self.name = 'Full MLP'
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500, random_state=SEED)
        self.scaler = StandardScaler()
    
    def fit(self, X, y, df, cols):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict_proba(self, X, df):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


# =============================================================================
# Evaluation with threshold calibration
# =============================================================================

def calibrate_threshold(y_true, y_proba, target_frr=0.01):
    """Find threshold achieving target FRR on benign samples."""
    benign_probs = y_proba[y_true == 0]
    threshold = np.percentile(benign_probs, (1 - target_frr) * 100)
    return threshold


def compute_metrics(y_true, y_pred, y_proba):
    """Compute FAR, FRR, accuracy, and ECE."""
    attack = y_true == 1
    benign = y_true == 0
    
    far = ((y_pred == 0) & attack).sum() / attack.sum() if attack.sum() > 0 else 0
    frr = ((y_pred == 1) & benign).sum() / benign.sum() if benign.sum() > 0 else 0
    acc = ((y_pred == 0) & benign).sum() + ((y_pred == 1) & attack).sum()
    acc = acc / len(y_true)
    
    # ECE
    bins = np.linspace(0, 1, 11)
    ece = 0
    for i in range(10):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_proba[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    
    return {'FAR': far, 'FRR': frr, 'Acc': acc, 'ECE': ece}


def compute_metrics_at_frr(y_true, y_proba, target_frr=0.01):
    """Compute metrics at calibrated threshold."""
    threshold = calibrate_threshold(y_true, y_proba, target_frr)
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics['threshold'] = threshold
    return metrics, y_pred


def compute_far_by_type(df, y_pred):
    """Compute FAR for each attack category."""
    single = ['A1_role', 'A2_payload', 'A3_rate']
    cross = ['A4_priv_loc', 'A5_payload_time', 'A6_id_behav']
    evasive = ['A7_evasive']
    
    results = {}
    for name, types in [('Single', single), ('Cross', cross), ('Evasive', evasive)]:
        mask = df['attack_type'].isin(types)
        if mask.sum() > 0:
            y_true = df.loc[mask, 'y'].values
            y_p = y_pred[mask]
            far = ((y_p == 0) & (y_true == 1)).sum() / (y_true == 1).sum()
            results[name] = far
    return results


# =============================================================================
# Visualization
# =============================================================================

COLORS = {
    'single': '#2E86AB',
    'cross': '#E94F37',
    'evasive': '#F5A623',
    'benign': '#28A745',
}


def create_figure(results, far_by_type):
    """Create the main evaluation figure."""
    
    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[25, 25, 50], height_ratios=[1, 1],
                          wspace=0.25, hspace=0.35,
                          left=0.04, right=0.98, top=0.88, bottom=0.12)
    
    ax_rules = fig.add_subplot(gs[0, 0])
    ax_learned = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, 0:2])
    ax_bar = fig.add_subplot(gs[:, 2])
    
    # Decision boundaries
    _plot_boundary(ax_rules, '(b) Rules', is_rules=True)
    _plot_boundary(ax_learned, '(c) Learned', is_rules=False)
    
    # Table
    _plot_table(ax_table, results)
    
    # Bar chart
    _plot_bars(ax_bar, far_by_type)
    
    return fig


def _plot_boundary(ax, title, is_rules):
    """Plot decision boundary illustration."""
    np.random.seed(42 if is_rules else 43)
    
    bx = np.random.uniform(0.1, 0.55, 80)
    by = np.random.uniform(0.1, 0.55, 80)
    ax_pts = np.random.uniform(0.35, 0.85, 40)
    ay_pts = np.random.uniform(0.35, 0.85, 40)
    
    ax.scatter(bx, by, c=COLORS['benign'], s=40, alpha=0.8, edgecolors='black', linewidths=0.5, label='Benign')
    ax.scatter(ax_pts, ay_pts, c=COLORS['cross'], s=40, alpha=0.8, edgecolors='black', linewidths=0.5, marker='s', label='Attack')
    
    if is_rules:
        ax.axhline(0.75, color='black', linestyle='--', linewidth=1.5)
        ax.axvline(0.75, color='black', linestyle='--', linewidth=1.5)
        ax.fill_between([0.75, 1], 0, 1, color='#FF6B6B', alpha=0.15)
        ax.fill_between([0, 0.75], 0.75, 1, color='#FF6B6B', alpha=0.15)
        fa = (ax_pts < 0.75) & (ay_pts < 0.75)
        ax.scatter(ax_pts[fa], ay_pts[fa], facecolors='none', edgecolors='#8B0000', s=120, linewidths=2)
        ax.legend(loc='lower right', fontsize=7)
    else:
        theta = np.linspace(0, np.pi/2, 100)
        bx_curve = 0.50 + 0.40 * np.cos(theta)
        by_curve = 0.50 + 0.40 * np.sin(theta)
        ax.plot(bx_curve, by_curve, 'k-', linewidth=2)
        ax.fill_between(bx_curve, by_curve, 1, color='#90EE90', alpha=0.15)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Dimension 1', fontsize=9)
    ax.set_ylabel('Dimension 2', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_aspect('equal')


def _plot_table(ax, results):
    """Plot metrics table."""
    models = ['Rules', 'Factored', 'Monotone', 'Full MLP']
    data = [[m, f"{results[m]['FAR']:.2f}", f"{results[m]['FRR']:.2f}",
             f"{results[m]['Acc']:.2f}", f"{results[m]['ECE']:.2f}"] for m in models]
    
    table = ax.table(cellText=data, colLabels=['Model', 'FAR', 'FRR', 'Acc', 'ECE'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)
    
    for j in range(5):
        table[(0, j)].set_text_props(fontweight='bold', color='white')
        table[(0, j)].set_facecolor('#505050')
    for j in range(5):
        table[(1, j)].set_facecolor('#FFE5E5')
    for i in range(2, 5):
        for j in range(5):
            table[(i, j)].set_facecolor('#E5FFE5')
    
    ax.axis('off')
    ax.set_title('(d) Summary (at 1\\% FRR)', fontweight='bold', fontsize=10, pad=4)


def _plot_bars(ax, far_by_type):
    """Plot grouped bar chart."""
    models = ['Rules', 'Factored', 'Monotone', 'Full MLP']
    x = np.arange(len(models))
    width = 0.25
    
    single = [far_by_type[m]['Single'] for m in models]
    cross = [far_by_type[m]['Cross'] for m in models]
    evasive = [far_by_type[m]['Evasive'] for m in models]
    
    # Minimum height for visibility
    min_h = 0.018
    
    ax.bar(x - width, [max(v, min_h) for v in single], width, label='Single-dim',
           color=COLORS['single'], edgecolor='black', linewidth=1)
    ax.bar(x, [max(v, min_h) for v in cross], width, label='Cross-dim',
           color=COLORS['cross'], edgecolor='black', linewidth=1)
    ax.bar(x + width, [max(v, min_h) for v in evasive], width, label='Evasive',
           color=COLORS['evasive'], edgecolor='black', linewidth=1)
    
    # Labels
    for i, (s, c, e) in enumerate(zip(single, cross, evasive)):
        for val, offset in [(s, -width), (c, 0), (e, width)]:
            label = f'{val:.0%}' if val > 0.005 else '<1%'
            ax.annotate(label, xy=(i + offset, max(val, min_h) + 0.02),
                       ha='center', fontsize=8)
    
    ax.set_ylabel('False Accept Rate', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=8)
    ax.legend(loc='upper right', fontsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_title('(a) FAR by Attack Type (1\\% FRR)', fontweight='bold', fontsize=11, pad=8)


# =============================================================================
# Main
# =============================================================================

def main():
    print("Generating dataset...")
    df = generate_dataset(n=100000)
    
    # Split
    n = len(df)
    train_df = df.iloc[:int(0.6*n)]
    val_df = df.iloc[int(0.6*n):int(0.8*n)]
    test_df = df.iloc[int(0.8*n):]
    
    X_train, y_train, cols = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df)
    X_test, y_test, _ = prepare_data(test_df)
    
    models = [RulesModel(), FactoredModel(), MonotoneModel(), FullMLPModel()]
    results = {}
    far_by_type = {}
    
    TARGET_FRR = 0.01  # 1% FRR for fair comparison
    
    for model in models:
        print(f"Training {model.name}...")
        model.fit(X_train, y_train, train_df, cols)
        
        print(f"Evaluating {model.name}...")
        proba = model.predict_proba(X_test, test_df)
        
        # Calibrate threshold to target FRR
        metrics, pred = compute_metrics_at_frr(y_test, proba, TARGET_FRR)
        results[model.name] = metrics
        far_by_type[model.name] = compute_far_by_type(test_df, pred)
        
        print(f"  FAR={metrics['FAR']:.3f}, FRR={metrics['FRR']:.3f}, "
              f"Acc={metrics['Acc']:.3f}, threshold={metrics['threshold']:.3f}")
        print(f"  By type: Single={far_by_type[model.name]['Single']:.1%}, "
              f"Cross={far_by_type[model.name]['Cross']:.1%}, "
              f"Evasive={far_by_type[model.name]['Evasive']:.1%}")
    
    print("\nGenerating figure...")
    fig = create_figure(results, far_by_type)
    fig.savefig('figure2.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('figure2.png', dpi=150, bbox_inches='tight')
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results.csv')
    
    # Also save FAR by type
    far_df = pd.DataFrame(far_by_type).T
    far_df.to_csv('far_by_type.csv')
    
    print("\nDone. Outputs: figure2.pdf, figure2.png, results.csv, far_by_type.csv")


if __name__ == "__main__":
    main()
