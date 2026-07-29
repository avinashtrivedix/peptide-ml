import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report

print("📊 Running 5-Fold Cross-Validation & Generating ROC Curve Plot...\n")

# 1. Load Dataset (27 Features)
df = pd.read_csv("data/real_peptides.csv")
ignore_cols = ['sequence', 'is_bioactive']
feature_cols = [c for c in df.columns if c not in ignore_cols]

X = df[feature_cols].values
y = df['is_bioactive'].values

# 2. Setup 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tprs = []
aucs = []
accuracies = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

print("Fold Performance Breakdown:")
print("-" * 35)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]
    preds = model.predict(X_val)
    
    acc = accuracy_score(y_val, preds)
    fpr, tpr, _ = roc_curve(y_val, probs)
    roc_auc = auc(fpr, tpr)
    
    aucs.append(roc_auc)
    accuracies.append(acc)

    # Interpolate TPR for mean curve calculation
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

    ax.plot(fpr, tpr, lw=1, alpha=0.35, label=f'Fold {fold} (AUC = {roc_auc:.4f})')
    print(f"Fold {fold}: Accuracy = {acc * 100:.2f}% | ROC-AUC = {roc_auc:.4f}")

print("-" * 35)
print(f"Mean Accuracy: {np.mean(accuracies) * 100:.2f}% ± {np.std(accuracies) * 100:.2f}%")
print(f"Mean ROC-AUC:  {np.mean(aucs):.4f} ± {np.std(aucs):.4f}\n")

# 3. Compute and Plot Mean ROC Curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

ax.plot(
    mean_fpr,
    mean_tpr,
    color='#1f77b4',
    label=rf'Mean ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})',
    lw=2.5,
    alpha=0.9
)

# Fill Standard Deviation Envelope
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color='grey',
    alpha=0.2,
    label=r'$\pm$ 1 Std. Dev.'
)

# Reference Diagonal Line (Random Guess)
ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='red', label='Random Chance (AUC = 0.50)', alpha=0.7)

# Formatting for Publication Quality
ax.set(
    xlim=[-0.02, 1.02],
    ylim=[-0.02, 1.02],
    xlabel='False Positive Rate (1 - Specificity)',
    ylabel='True Positive Rate (Sensitivity)',
    title='5-Fold Cross-Validation ROC Curve\nTier 1 Baseline: XGBoost + RDKit & AAC Features'
)
ax.legend(loc="lower right", fontsize=9, frameon=True)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
print("✅ Saved figure to 'roc_curve.png'")