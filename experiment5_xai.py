"""
Experiment 5: Explainable AI (XAI) Methods - SHAP, LIME & Fairlearn Audit
==========================================================================
Aim    : Apply SHAP and LIME to interpret model predictions; audit fairness
         using Fairlearn across sensitive features (customer_country).
Models : baseline_xgboost.pkl (XGBoost)
Dataset: shopify_sales_final_cleaned_modified.csv

Deliverables:
  - SHAP global summary plot (bar)
  - SHAP global summary plot (beeswarm)
  - SHAP dependence plots (top-3 features)
  - LIME local explanations (3 sample predictions)
  - Fairlearn fairness audit report (demographic parity, equalized odds)
  - Bias mitigation: threshold-adjustment post-processing

Author : Experiment Team
Date   : September 2026
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
import shap
import lime
import lime.lime_tabular
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    false_negative_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer

# Configuration
RANDOM_STATE     = 42
np.random.seed(RANDOM_STATE)
CSV_PATH         = "shopify_sales_final_cleaned_modified.csv"
SAVE_DIR         = "./experiment5_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

CATEGORICAL_COLS = [
    "product_category", "customer_country",
    "traffic_source", "payment_method", "discount_tier"
]
LEAKAGE_COLS = [
    "order_id", "customer_id", "product_id",
    "discounted_price", "revenue", "profit", "order_date"
]
SENSITIVE_COL = "customer_country"


# 1. Data Loading
print("[1/7] Loading and preprocessing data...")
df = pd.read_csv(CSV_PATH)

df["order_amount_pre_discount"] = df["product_price"] * df["quantity"]
df["discount_amount"] = (
    df["product_price"] * df["quantity"] * (df["discount_percent"] / 100.0)
)
df["shipping_ratio"] = df["shipping_cost"] / (df["order_amount_pre_discount"] + 1e-5)
df["price_per_rating"] = df["product_price"] / (df["rating"] + 1e-5)
if "order_date" in df.columns:
    ds = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_dayofweek"] = ds.dt.dayofweek.fillna(0).astype(int)

feature_cols = [
    c for c in df.columns if c not in LEAKAGE_COLS and c != "is_returned"
]
X_raw = df[feature_cols].copy()
y     = df["is_returned"].copy()
sens  = df[SENSITIVE_COL].copy()

if "discount_tier" in X_raw.columns:
    X_raw["discount_tier"] = X_raw["discount_tier"].fillna("Unknown")

numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
sens_test = sens.loc[X_test_raw.index]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_COLS),
    ],
    remainder="drop",
)
preprocessor.fit(X_train_raw)
cat_names     = preprocessor.named_transformers_["cat"].get_feature_names_out(
    CATEGORICAL_COLS
).tolist()
feature_names = numeric_cols + cat_names

X_train_t = preprocessor.transform(X_train_raw)
X_test_t  = preprocessor.transform(X_test_raw)
print(f"    Features: {len(feature_names)} | Train: {X_train_t.shape[0]} | Test: {X_test_t.shape[0]}")


# 2. Load Models
print("[2/7] Loading trained XGBoost model...")
xgb_pipeline = joblib.load("saved_models/baseline_xgboost.pkl")
xgb_clf      = xgb_pipeline.named_steps["classifier"]
xgb_clf.fit(X_train_t, y_train)
print("    XGBoost classifier ready.")


# 3. SHAP Global Feature Importance
print("[3/7] Computing SHAP values (TreeExplainer on XGBoost)...")
explainer_shap = shap.TreeExplainer(xgb_clf)
n_shap = min(2000, len(X_test_t))
idx_s  = np.random.choice(len(X_test_t), size=n_shap, replace=False)
X_shap = X_test_t[idx_s]
shap_vals = explainer_shap.shap_values(X_shap)

# 3a. Summary bar plot
print("    Saving SHAP summary bar plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals, X_shap, feature_names=feature_names,
    plot_type="bar", show=False, max_display=20
)
plt.title(
    "SHAP Global Feature Importance\n(Mean |SHAP Value| - XGBoost on Shopify Returns)",
    fontsize=13, pad=12
)
plt.tight_layout()
shap_bar_path = os.path.join(SAVE_DIR, "shap_summary_bar.png")
plt.savefig(shap_bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved -> {shap_bar_path}")

# 3b. Beeswarm plot
print("    Saving SHAP beeswarm plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals, X_shap, feature_names=feature_names,
    show=False, max_display=20
)
plt.title(
    "SHAP Beeswarm Plot - Feature Impact Direction on Return Prediction",
    fontsize=12, pad=12
)
plt.tight_layout()
shap_bee_path = os.path.join(SAVE_DIR, "shap_summary_beeswarm.png")
plt.savefig(shap_bee_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved -> {shap_bee_path}")

# 3c. Identify top-3 features
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
top3_idx      = np.argsort(mean_abs_shap)[::-1][:3]
top3_features = [feature_names[i] for i in top3_idx]
top3_vals     = [float(mean_abs_shap[i]) for i in top3_idx]
print(f"    Top-3 SHAP features: {top3_features}")

# 3d. Dependence plots for top-3
print("    Saving SHAP dependence plots...")
dep_paths = []
for feat in top3_features:
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(
        feat, shap_vals, X_shap,
        feature_names=feature_names, show=False,
        interaction_index="auto"
    )
    plt.title(f"SHAP Dependence Plot: {feat}", fontsize=12, pad=10)
    plt.tight_layout()
    safe     = feat.replace("/", "_").replace(" ", "_")
    dep_path = os.path.join(SAVE_DIR, f"shap_dependence_{safe}.png")
    plt.savefig(dep_path, dpi=150, bbox_inches="tight")
    plt.close()
    dep_paths.append(dep_path)
    print(f"    Saved -> {dep_path}")


# 4. LIME Local Explanations
print("[4/7] Computing LIME local explanations...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data        = X_train_t,
    feature_names        = feature_names,
    class_names          = ["Not Returned", "Returned"],
    mode                 = "classification",
    discretize_continuous= True,
    random_state         = RANDOM_STATE,
)

def lime_predict_fn(data):
    return xgb_clf.predict_proba(data)

y_prob_test = xgb_clf.predict_proba(X_test_t)[:, 1]

mask_hi_ret   = (y_prob_test > 0.75) & (y_test.values == 1)
mask_hi_noret = (y_prob_test < 0.25) & (y_test.values == 0)
mask_border   = (y_prob_test > 0.45) & (y_prob_test < 0.55)

lime_samples = {}
if mask_hi_ret.any():
    lime_samples["high_confidence_return"]     = int(np.where(mask_hi_ret)[0][0])
if mask_hi_noret.any():
    lime_samples["high_confidence_non_return"]  = int(np.where(mask_hi_noret)[0][0])
if mask_border.any():
    lime_samples["borderline_prediction"]       = int(np.where(mask_border)[0][0])

lime_paths = {}
for label, idx in lime_samples.items():
    exp = lime_explainer.explain_instance(
        X_test_t[idx], lime_predict_fn, num_features=12, top_labels=2
    )
    fig = exp.as_pyplot_figure(label=1)
    fig.set_size_inches(10, 6)
    title_label = label.replace("_", " ").title()
    fig.suptitle(
        f"LIME Local Explanation - {title_label}\n"
        f"P(Return)={y_prob_test[idx]:.3f}  |  Actual={int(y_test.values[idx])}",
        fontsize=11,
    )
    fig.tight_layout()
    lime_path = os.path.join(SAVE_DIR, f"lime_{label}.png")
    fig.savefig(lime_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved -> {lime_path}")
    lime_paths[label] = lime_path


# 5. Fairlearn Fairness Audit
print("[5/7] Running Fairlearn fairness audit...")
y_pred_bin = xgb_clf.predict(X_test_t)

metrics_dict = {
    "accuracy":            lambda yt, yp: float((yt == yp).mean()),
    "recall":              lambda yt, yp: float(yp[yt == 1].mean()) if (yt == 1).any() else 0.0,
    "false_positive_rate": false_positive_rate,
    "false_negative_rate": false_negative_rate,
}

mf = MetricFrame(
    metrics            = metrics_dict,
    y_true             = y_test.values,
    y_pred             = y_pred_bin,
    sensitive_features = sens_test.values,
)

dpd = demographic_parity_difference(
    y_test.values, y_pred_bin, sensitive_features=sens_test.values
)
dpr = demographic_parity_ratio(
    y_test.values, y_pred_bin, sensitive_features=sens_test.values
)
eod = equalized_odds_difference(
    y_test.values, y_pred_bin, sensitive_features=sens_test.values
)

print(f"    Demographic Parity Difference  : {dpd:.4f}")
print(f"    Demographic Parity Ratio       : {dpr:.4f}")
print(f"    Equalized Odds Difference      : {eod:.4f}")

# 5a. Per-group metrics bar chart
by_group = mf.by_group
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
for ax, metric in zip(axes, ["accuracy", "recall", "false_positive_rate"]):
    vals = by_group[metric]
    bars = ax.bar(vals.index, vals.values, color=colors[:len(vals)], alpha=0.85)
    ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_xlabel("Customer Country")
    ax.set_ylabel(metric)
    ax.set_xticklabels(vals.index, rotation=30, ha="right", fontsize=9)
    ax.axhline(mf.overall[metric], color="red", linestyle="--", lw=1.5, label="Overall avg")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, vals.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8
        )
fig.suptitle(
    "Fairlearn Audit: Per-Group Performance Metrics by Customer Country",
    fontsize=13, fontweight="bold", y=1.02
)
plt.tight_layout()
fairness_bar_path = os.path.join(SAVE_DIR, "fairlearn_group_metrics.png")
plt.savefig(fairness_bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved -> {fairness_bar_path}")

# 5b. Equalized odds heatmap
fpr_vals  = by_group["false_positive_rate"].values
fnr_vals  = by_group["false_negative_rate"].values
groups    = by_group["false_positive_rate"].index.tolist()
heat_data = pd.DataFrame({"FPR": fpr_vals, "FNR": fnr_vals}, index=groups)
fig, ax   = plt.subplots(figsize=(9, 3))
sns.heatmap(
    heat_data.T, annot=True, fmt=".3f", cmap="RdYlGn_r",
    linewidths=0.5, ax=ax, vmin=0.1, vmax=0.6
)
ax.set_title(
    "Equalized Odds Heatmap: FPR and FNR by Customer Country",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
fairness_heat_path = os.path.join(SAVE_DIR, "fairlearn_equalized_odds_heatmap.png")
plt.savefig(fairness_heat_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved -> {fairness_heat_path}")


# 6. Bias Mitigation - Manual Per-Group Threshold Adjustment
# (Replaces ThresholdOptimizer due to Fairlearn/pandas 2.x dtype incompatibility)
# Conceptually equivalent: post-processing thresholds per sensitive group to
# equalise positive prediction rates (demographic parity).
print("[6/7] Applying bias mitigation (post-processing threshold adjustment)...")

# Get predicted probabilities from XGBoost pipeline
y_prob_full = xgb_pipeline.predict_proba(X_test_raw)[:, 1]
sens_test_arr = sens_test.values

# Compute the overall positive prediction rate at default threshold 0.5
default_preds  = (y_prob_full >= 0.5).astype(int)
overall_rate   = default_preds.mean()

# Per-group: find threshold that brings each group's positive rate
# as close as possible to the overall rate (demographic parity objective)
groups_unique  = np.unique(sens_test_arr)
y_pred_mit_arr = np.zeros(len(y_prob_full), dtype=int)
thresholds_used = {}

for grp in groups_unique:
    mask       = (sens_test_arr == grp)
    probs_grp  = y_prob_full[mask]
    # Binary search for threshold that matches overall_rate
    best_thresh = 0.5
    best_diff   = np.inf
    for thresh in np.linspace(0.1, 0.9, 161):
        rate = (probs_grp >= thresh).mean()
        diff = abs(rate - overall_rate)
        if diff < best_diff:
            best_diff  = diff
            best_thresh = thresh
    y_pred_mit_arr[mask] = (probs_grp >= best_thresh).astype(int)
    thresholds_used[grp] = round(float(best_thresh), 3)

print(f"    Per-group thresholds used: {thresholds_used}")

dpd_mit = demographic_parity_difference(
    y_test.values, y_pred_mit_arr, sensitive_features=sens_test_arr
)
eod_mit = equalized_odds_difference(
    y_test.values, y_pred_mit_arr, sensitive_features=sens_test_arr
)
f1_mit  = f1_score(y_test.values, y_pred_mit_arr)
y_pred_mitigated = y_pred_mit_arr

print(f"    DPD: {dpd:.4f} -> {dpd_mit:.4f}")
print(f"    EOD: {eod:.4f} -> {eod_mit:.4f}")
print(f"    F1 post-mitigation: {f1_mit:.4f}")

# 6a. Before vs After comparison chart
fig, ax = plt.subplots(figsize=(7, 4))
metrics_compare = ["Demographic Parity Diff", "Equalized Odds Diff"]
before_vals = [abs(dpd), abs(eod)]
after_vals  = [abs(dpd_mit), abs(eod_mit)]
x = np.arange(len(metrics_compare))
w = 0.35
b1 = ax.bar(x - w / 2, before_vals, w, label="Before Mitigation",
            color="#E74C3C", alpha=0.85)
b2 = ax.bar(x + w / 2, after_vals, w, label="After Mitigation",
            color="#27AE60", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metrics_compare, fontsize=10)
ax.set_ylabel("Metric Value (lower = fairer)", fontsize=10)
ax.set_title(
    "Bias Mitigation: ThresholdOptimizer (Demographic Parity)\nBefore vs After Comparison",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=9)
ax.set_ylim(0, max(before_vals + after_vals) * 1.45)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha="center", fontsize=9)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{bar.get_height():.3f}", ha="center", fontsize=9)
plt.tight_layout()
mitigation_path = os.path.join(SAVE_DIR, "fairlearn_mitigation_comparison.png")
plt.savefig(mitigation_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved -> {mitigation_path}")


# 7. JSON Report
print("[7/7] Writing full experiment report...")
report = {
    "experiment":            "Experiment 5 - XAI and Fairness Audit",
    "model_used":            "XGBoost (baseline_xgboost.pkl)",
    "sensitive_attribute":   SENSITIVE_COL,
    "shap": {
        "top_3_features":    top3_features,
        "mean_abs_shap_top3": top3_vals,
        "all_feature_importance": {
            feature_names[i]: float(mean_abs_shap[i])
            for i in np.argsort(mean_abs_shap)[::-1]
        },
        "plot_bar":          shap_bar_path,
        "plot_beeswarm":     shap_bee_path,
        "dependence_plots":  dep_paths,
    },
    "lime": {
        "samples_explained": list(lime_samples.keys()),
        "plots":             lime_paths,
    },
    "fairness_before_mitigation": {
        "demographic_parity_difference": float(dpd),
        "demographic_parity_ratio":      float(dpr),
        "equalized_odds_difference":     float(eod),
    },
    "fairness_after_mitigation": {
        "demographic_parity_difference": float(dpd_mit),
        "equalized_odds_difference":     float(eod_mit),
        "f1_score_post_mitigation":      float(f1_mit),
    },
    "per_group_metrics": mf.by_group.to_dict(),
}
report_path = os.path.join(SAVE_DIR, "experiment5_report.json")
with open(report_path, "w") as fh:
    json.dump(report, fh, indent=4, default=str)
print(f"    Saved -> {report_path}")

print()
print("=" * 60)
print("EXPERIMENT 5 COMPLETE")
print(f"All outputs saved to: {SAVE_DIR}/")
print("=" * 60)
print(f"  SHAP top feature  : {top3_features[0]}")
print(f"  Mean |SHAP|       : {top3_vals[0]:.4f}")
print(f"  DPD before/after  : {dpd:.4f} / {dpd_mit:.4f}")
print(f"  EOD before/after  : {eod:.4f} / {eod_mit:.4f}")
print(f"  F1 (post-mitig.)  : {f1_mit:.4f}")
