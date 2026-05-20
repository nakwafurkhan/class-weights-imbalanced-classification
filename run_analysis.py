"""
run_analysis.py
--------------------------------------------------------------
End-to-end analysis script for the class-weights assignment.

Runs the full pipeline, saves all figures to assets/, and prints
each section heading so we can capture the outputs into a Jupyter
notebook later.

Dataset: synthetic customer-churn prediction (92/8 split) generated
via sklearn.make_classification. We use a synthetic dataset because:

  1. It's fully reproducible (random_state=42)
  2. We control the imbalance ratio exactly
  3. No external download or data-cleaning is needed
  4. The class imbalance is realistic — 8% positive matches typical
     customer churn rates in subscription businesses
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS, exist_ok=True)

# Match the notebook's aesthetic
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 140

RANDOM_STATE = 42


# ----------------------------------------------------------
# 1. DATA — generate a realistic 92/8 imbalanced dataset
# ----------------------------------------------------------
print("=" * 70)
print("PART 1: DATA")
print("=" * 70)

X, y = make_classification(
    n_samples=5000,
    n_features=12,
    n_informative=6,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    weights=[0.92, 0.08],  # 92/8 imbalance — realistic churn rate
    flip_y=0.02,            # small amount of label noise
    class_sep=1.1,
    random_state=RANDOM_STATE,
)

feature_names = [f"feat_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["churned"] = y

class_counts = df["churned"].value_counts().sort_index()
class_pcts = (class_counts / len(df) * 100).round(2)

print(f"\nTotal samples:      {len(df)}")
print(f"Features:           {X.shape[1]}")
print(f"\nClass distribution:")
print(f"  Class 0 (stayed):   {class_counts[0]:>5}  ({class_pcts[0]:>5.2f}%)")
print(f"  Class 1 (churned):  {class_counts[1]:>5}  ({class_pcts[1]:>5.2f}%)")
print(f"\nImbalance ratio:    {class_counts[0] / class_counts[1]:.1f}:1")


# Class distribution plot
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#6366f1", "#f43f5e"]  # iris vs coral
bars = ax.bar(["Stayed (0)", "Churned (1)"], class_counts.values, color=colors)
for bar, count, pct in zip(bars, class_counts.values, class_pcts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 30,
        f"{count}\n({pct}%)",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
ax.set_title("Class distribution — synthetic customer churn (92 / 8)", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of samples")
ax.set_ylim(0, max(class_counts.values) * 1.18)
sns.despine(left=True, bottom=False)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS, "01_class_distribution.png"))
plt.close()
print(f"\nSaved: assets/01_class_distribution.png")


# ----------------------------------------------------------
# 2. STRATIFIED SPLIT
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

train_pos = int(y_train.sum())
test_pos = int(y_test.sum())
print(f"\nTrain set:  {len(X_train)} samples ({train_pos} positives, {train_pos/len(X_train)*100:.2f}%)")
print(f"Test set:   {len(X_test)} samples ({test_pos} positives, {test_pos/len(X_test)*100:.2f}%)")


# ----------------------------------------------------------
# 3. BASELINE: DummyClassifier (must-beat floor)
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("PART 2: BASELINE — DummyClassifier (most_frequent)")
print("=" * 70)

dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred_dummy):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_dummy):.4f}  (always predicts majority)")
print(f"F1:        {f1_score(y_test, y_pred_dummy):.4f}")
print("\nThis is the floor that any real model must clearly beat on minority recall.")


# ----------------------------------------------------------
# 4. UNWEIGHTED MODEL
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("PART 3: UNWEIGHTED RandomForestClassifier")
print("=" * 70)

rf_unweighted = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_unweighted.fit(X_train, y_train)
y_pred_unw = rf_unweighted.predict(X_test)
y_prob_unw = rf_unweighted.predict_proba(X_test)[:, 1]

unw_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_unw),
    "precision": precision_score(y_test, y_pred_unw, zero_division=0),
    "recall": recall_score(y_test, y_pred_unw, zero_division=0),
    "f1": f1_score(y_test, y_pred_unw, zero_division=0),
    "pr_auc": average_precision_score(y_test, y_prob_unw),
}

print(f"\nAccuracy:       {unw_metrics['accuracy']:.4f}")
print(f"Precision (1):  {unw_metrics['precision']:.4f}")
print(f"Recall (1):     {unw_metrics['recall']:.4f}")
print(f"F1 (1):         {unw_metrics['f1']:.4f}")
print(f"PR-AUC:         {unw_metrics['pr_auc']:.4f}")

print("\nFull classification report (unweighted):")
print(classification_report(y_test, y_pred_unw, target_names=["Stayed", "Churned"], digits=3))

cm_unw = confusion_matrix(y_test, y_pred_unw)
print(f"Confusion matrix (unweighted):")
print(f"  TN = {cm_unw[0,0]:>4}    FP = {cm_unw[0,1]:>4}")
print(f"  FN = {cm_unw[1,0]:>4}    TP = {cm_unw[1,1]:>4}")


# ----------------------------------------------------------
# 5. CLASS WEIGHTS — display the computed weights
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("PART 4: COMPUTED CLASS WEIGHTS")
print("=" * 70)

classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
weight_dict = dict(zip(classes, weights))

print(f"\nclass_weight='balanced' computes:")
print(f"  w_0 = n / (k * n_0) = {len(y_train)} / (2 * {int((y_train==0).sum())}) = {weight_dict[0]:.4f}")
print(f"  w_1 = n / (k * n_1) = {len(y_train)} / (2 * {int((y_train==1).sum())}) = {weight_dict[1]:.4f}")
print(f"\nRatio (minority / majority): {weight_dict[1] / weight_dict[0]:.2f}x")


# ----------------------------------------------------------
# 6. WEIGHTED MODEL
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("PART 5: WEIGHTED RandomForestClassifier (class_weight='balanced')")
print("=" * 70)

rf_weighted = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_weighted.fit(X_train, y_train)
y_pred_w = rf_weighted.predict(X_test)
y_prob_w = rf_weighted.predict_proba(X_test)[:, 1]

w_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_w),
    "precision": precision_score(y_test, y_pred_w, zero_division=0),
    "recall": recall_score(y_test, y_pred_w, zero_division=0),
    "f1": f1_score(y_test, y_pred_w, zero_division=0),
    "pr_auc": average_precision_score(y_test, y_prob_w),
}

print(f"\nAccuracy:       {w_metrics['accuracy']:.4f}")
print(f"Precision (1):  {w_metrics['precision']:.4f}")
print(f"Recall (1):     {w_metrics['recall']:.4f}")
print(f"F1 (1):         {w_metrics['f1']:.4f}")
print(f"PR-AUC:         {w_metrics['pr_auc']:.4f}")

print("\nFull classification report (weighted):")
print(classification_report(y_test, y_pred_w, target_names=["Stayed", "Churned"], digits=3))

cm_w = confusion_matrix(y_test, y_pred_w)
print(f"Confusion matrix (weighted):")
print(f"  TN = {cm_w[0,0]:>4}    FP = {cm_w[0,1]:>4}")
print(f"  FN = {cm_w[1,0]:>4}    TP = {cm_w[1,1]:>4}")


# ----------------------------------------------------------
# 7. CONFUSION MATRICES SIDE-BY-SIDE
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, model, title, cmap in zip(
    axes,
    [rf_unweighted, rf_weighted],
    ["Without class weights", "With class weights (balanced)"],
    ["Blues", "Purples"],
):
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Stayed", "Churned"],
        cmap=cmap,
        ax=ax,
        values_format="d",
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS, "02_confusion_matrices.png"))
plt.close()
print(f"\nSaved: assets/02_confusion_matrices.png")


# ----------------------------------------------------------
# 8. COMPARISON TABLE
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("PART 6: COMPARISON TABLE")
print("=" * 70)

comparison = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Precision (minority)", "Recall (minority)", "F1 (minority)", "PR-AUC"],
        "Without weights": [
            unw_metrics["accuracy"],
            unw_metrics["precision"],
            unw_metrics["recall"],
            unw_metrics["f1"],
            unw_metrics["pr_auc"],
        ],
        "With weights": [
            w_metrics["accuracy"],
            w_metrics["precision"],
            w_metrics["recall"],
            w_metrics["f1"],
            w_metrics["pr_auc"],
        ],
    }
)
comparison["Δ (w − unw)"] = comparison["With weights"] - comparison["Without weights"]
comparison["Direction"] = comparison["Δ (w − unw)"].apply(
    lambda d: "↑" if d > 0.001 else ("↓" if d < -0.001 else "≈")
)

# Format numeric cols
formatted = comparison.copy()
for col in ["Without weights", "With weights", "Δ (w − unw)"]:
    formatted[col] = formatted[col].apply(lambda v: f"{v:.4f}")
print("\n" + formatted.to_string(index=False))


# ----------------------------------------------------------
# 9. CROSS-VALIDATION (sanity check)
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("PART 7: 5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in [
    ("Unweighted", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)),
    ("Weighted  ", RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
]:
    cv_recall = cross_val_score(model, X_train, y_train, cv=skf, scoring="recall", n_jobs=-1)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1", n_jobs=-1)
    cv_pr = cross_val_score(model, X_train, y_train, cv=skf, scoring="average_precision", n_jobs=-1)
    print(f"\n{name}")
    print(f"  Recall    : {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
    print(f"  F1        : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"  PR-AUC    : {cv_pr.mean():.4f} ± {cv_pr.std():.4f}")


# ----------------------------------------------------------
# 10. PR CURVE — overlay both
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for prob, label, color in [
    (y_prob_unw, "Without weights", "#6366f1"),
    (y_prob_w, "With weights", "#f43f5e"),
]:
    precision, recall, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    ax.plot(recall, precision, label=f"{label} (AP = {ap:.3f})", linewidth=2, color=color)
# Baseline = prevalence
prevalence = y_test.mean()
ax.axhline(prevalence, linestyle="--", color="#94a3b8", label=f"Baseline (prevalence = {prevalence:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall curve — unweighted vs weighted", fontweight="bold")
ax.legend(loc="upper right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS, "03_pr_curve.png"))
plt.close()
print(f"\nSaved: assets/03_pr_curve.png")


# ----------------------------------------------------------
# 11. GridSearchCV over manual weight ratios
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("PART 8: GridSearchCV over manual weight ratios")
print("=" * 70)

param_grid = {
    "class_weight": [
        None,  # for comparison
        {0: 1, 1: 5},
        {0: 1, 1: 10},
        {0: 1, 1: 15},
        {0: 1, 1: 20},
        "balanced",
        "balanced_subsample",
    ]
}

grid = GridSearchCV(
    RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1),
    param_grid,
    cv=skf,
    scoring="f1",
    n_jobs=-1,
    refit=True,
)
grid.fit(X_train, y_train)

print(f"\nBest class_weight:  {grid.best_params_}")
print(f"Best CV F1:         {grid.best_score_:.4f}")

# show every row
grid_results = pd.DataFrame(grid.cv_results_)[
    ["params", "mean_test_score", "std_test_score"]
].sort_values("mean_test_score", ascending=False)
print("\nFull grid (sorted by F1):")
for _, row in grid_results.iterrows():
    print(f"  {str(row['params']):<55} F1 = {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")


# ----------------------------------------------------------
# 12. Save the comparison table as CSV for the README
# ----------------------------------------------------------
comparison.to_csv(os.path.join(ASSETS, "comparison_table.csv"), index=False)
print(f"\nSaved: assets/comparison_table.csv")

# Save a JSON of the key metrics so the README can reference exact numbers
import json
metrics_json = {
    "imbalance_ratio": float(class_counts[0] / class_counts[1]),
    "minority_pct": float(class_pcts[1]),
    "computed_weights": {
        "majority": float(weight_dict[0]),
        "minority": float(weight_dict[1]),
        "ratio": float(weight_dict[1] / weight_dict[0]),
    },
    "unweighted": unw_metrics,
    "weighted": w_metrics,
    "best_grid_config": str(grid.best_params_),
    "best_grid_f1": float(grid.best_score_),
    "confusion_matrix": {
        "unweighted": {"TN": int(cm_unw[0,0]), "FP": int(cm_unw[0,1]), "FN": int(cm_unw[1,0]), "TP": int(cm_unw[1,1])},
        "weighted":   {"TN": int(cm_w[0,0]),   "FP": int(cm_w[0,1]),   "FN": int(cm_w[1,0]),   "TP": int(cm_w[1,1])},
    },
}
with open(os.path.join(ASSETS, "metrics.json"), "w") as f:
    json.dump(metrics_json, f, indent=2)
print(f"Saved: assets/metrics.json")

print("\n" + "=" * 70)
print("DONE. All outputs in assets/.")
print("=" * 70)
