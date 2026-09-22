"""
Experiment 4: ML Modeling & Experiment Tracking Pipeline
=========================================================
Target: is_returned (Binary Classification ~85% non-returned / 15% returned)
Dataset: shopify_sales_final_cleaned_modified.csv (60,000 Shopify orders)

Author: Antigravity AI Engineer
Date: September 2026
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import xgboost as xgb
import mlflow
import mlflow.sklearn

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Setup directories
OUTPUT_DIR = "./saved_models"
ARTIFACTS_DIR = "./evaluation_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Configure MLflow
TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)
EXPERIMENT_NAME = "Shopify_Return_Prediction_Experiment_4_v2"
mlflow.set_experiment(EXPERIMENT_NAME)


def load_and_preprocess_data(csv_path="shopify_sales_final_cleaned_modified.csv"):
    """
    Loads dataset, performs leakage prevention, feature engineering,
    and returns train/test splits.
    """
    print(f"[*] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[*] Raw dataset shape: {df.shape}")

    # Inspect class balance
    return_counts = df['is_returned'].value_counts()
    pos_rate = df['is_returned'].mean()
    print(f"[*] Target distribution: 0={return_counts[0]} ({1-pos_rate:.2%}), 1={return_counts[1]} ({pos_rate:.2%})")

    # Feature Engineering (pre-purchase features only, no leakage)
    print("[*] Performing feature engineering...")
    df['order_amount_pre_discount'] = df['product_price'] * df['quantity']
    df['discount_amount'] = df['product_price'] * df['quantity'] * (df['discount_percent'] / 100.0)
    df['shipping_ratio'] = df['shipping_cost'] / (df['order_amount_pre_discount'] + 1e-5)
    df['price_per_rating'] = df['product_price'] / (df['rating'] + 1e-5)
    
    # Optional calendar features from order_date
    if 'order_date' in df.columns:
        date_series = pd.to_datetime(df['order_date'], errors='coerce')
        df['order_month'] = date_series.dt.month.fillna(1).astype(int)
        df['order_dayofweek'] = date_series.dt.dayofweek.fillna(0).astype(int)

    # Exclude leakage-prone and pure ID columns
    leakage_cols = ['order_id', 'customer_id', 'product_id', 'discounted_price', 'revenue', 'profit', 'order_date']
    feature_cols = [col for col in df.columns if col not in leakage_cols and col != 'is_returned']
    
    print(f"[*] Selected {len(feature_cols)} predictors:")
    print("    ", feature_cols)

    X = df[feature_cols].copy()
    y = df['is_returned'].copy()

    # Fill nulls in discount_tier (7,567 missing) with 'Unknown' category
    if 'discount_tier' in X.columns:
        X['discount_tier'] = X['discount_tier'].fillna('Unknown')
        print(f"[*] Filled discount_tier nulls with 'Unknown'")


    # Identify numeric and categorical columns
    categorical_cols = ['product_category', 'customer_country', 'traffic_source', 'payment_method', 'discount_tier']
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    print(f"[*] Numerical features ({len(numeric_cols)}): {numeric_cols}")
    print(f"[*] Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[*] Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )

    return X_train, X_test, y_train, y_test, preprocessor, feature_cols, numeric_cols, categorical_cols


def evaluate_and_plot(model_pipeline, X_test, y_test, model_name, plot_prefix):
    """
    Computes comprehensive evaluation metrics and saves diagnostic plots.
    """
    y_pred = model_pipeline.predict(X_test)
    
    if hasattr(model_pipeline, "predict_proba"):
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]
    elif hasattr(model_pipeline, "decision_function"):
        d = model_pipeline.decision_function(X_test)
        y_prob = (d - d.min()) / (d.max() - d.min() + 1e-9)
    else:
        y_prob = y_pred

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc)
    }

    # Generate Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Returned', 'Returned'],
                yticklabels=['Not Returned', 'Returned'])
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    cm_path = os.path.join(ARTIFACTS_DIR, f"{plot_prefix}_cm.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Generate ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(ARTIFACTS_DIR, f"{plot_prefix}_roc.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # Text report
    cr = classification_report(y_test, y_pred, target_names=['Not Returned (0)', 'Returned (1)'], output_dict=True)
    
    return metrics, cm_path, roc_path, cr


def log_run_to_mlflow(run_name, model_pipeline, params, metrics, cm_path, roc_path, model_type="baseline", algo="sklearn"):
    """
    Tracks complete experiment details to MLflow.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("algorithm", algo)
        mlflow.set_tag("dataset_size", "60000")
        mlflow.set_tag("target", "is_returned")
        
        # Log params & metrics
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        # Log artifacts
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path, artifact_path="plots")
        if os.path.exists(roc_path):
            mlflow.log_artifact(roc_path, artifact_path="plots")
            
        # Log model pipeline
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        run_id = mlflow.active_run().info.run_id
        print(f"    [MLflow] Logged run '{run_name}' (ID: {run_id})")
        return run_id


def train_baseline_models(X_train, y_train, X_test, y_test, preprocessor):
    """
    Trains 5 baseline classifiers with default/simple balanced configurations.
    """
    print("\n" + "="*70)
    print("STEP 2: BASELINE MODEL TRAINING (5 Classifiers)")
    print("="*70)

    # Imbalance ratio for scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / pos_count

    baseline_definitions = {
        "Logistic Regression": {
            "key": "baseline_logistic_regression",
            "algo": "LogisticRegression",
            "estimator": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
            "params": {"penalty": "l2", "C": 1.0, "class_weight": "balanced", "solver": "lbfgs"}
        },
        "Random Forest": {
            "key": "baseline_random_forest",
            "algo": "RandomForestClassifier",
            "estimator": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
            "params": {"n_estimators": 100, "max_depth": 12, "class_weight": "balanced", "n_jobs": -1}
        },
        "HistGradientBoosting": {
            "key": "baseline_hist_gradient_boosting",
            "algo": "HistGradientBoostingClassifier",
            "estimator": HistGradientBoostingClassifier(class_weight='balanced', max_iter=100, random_state=RANDOM_STATE),
            "params": {"max_iter": 100, "class_weight": "balanced", "learning_rate": 0.1}
        },
        "XGBoost": {
            "key": "baseline_xgboost",
            "algo": "XGBClassifier",
            "estimator": xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos,
                eval_metric='logloss',
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "scale_pos_weight": float(scale_pos)}
        },
        "Linear SVM (Calibrated)": {
            "key": "baseline_calibrated_linearsvc",
            "algo": "LinearSVC_Calibrated",
            "estimator": CalibratedClassifierCV(
                estimator=LinearSVC(class_weight='balanced', dual=False, random_state=RANDOM_STATE)
            ),
            "params": {"base_estimator": "LinearSVC", "class_weight": "balanced", "calibration": "sigmoid"}
        }
    }

    baseline_results = {}
    trained_pipelines = {}

    for name, config in baseline_definitions.items():
        print(f"\n[*] Training {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', config['estimator'])
        ])

        pipeline.fit(X_train, y_train)
        trained_pipelines[config['key']] = pipeline

        # Evaluate
        metrics, cm_path, roc_path, _ = evaluate_and_plot(
            pipeline, X_test, y_test, name, config['key']
        )
        
        # Log to MLflow
        run_id = log_run_to_mlflow(
            run_name=f"Baseline_{name.replace(' ', '_')}",
            model_pipeline=pipeline,
            params=config['params'],
            metrics=metrics,
            cm_path=cm_path,
            roc_path=roc_path,
            model_type="baseline",
            algo=config['algo']
        )

        # Save individual pickle model
        pkl_path = os.path.join(OUTPUT_DIR, f"{config['key']}.pkl")
        joblib.dump(pipeline, pkl_path)
        print(f"    [Pickle Saved] {pkl_path}")

        baseline_results[name] = {
            "key": config['key'],
            "metrics": metrics,
            "run_id": run_id,
            "pkl_path": pkl_path
        }
        print(f"    Accuracy: {metrics['accuracy']:.4f} | Recall: {metrics['recall']:.4f} | Precision: {metrics['precision']:.4f} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

    return baseline_results, trained_pipelines


def tune_top_models(X_train, y_train, X_test, y_test, preprocessor):
    """
    Performs hyperparameter tuning on top baseline models (Random Forest and XGBoost)
    using Stratified K-Fold Cross Validation.
    """
    print("\n" + "="*70)
    print("STEP 3: HYPERPARAMETER TUNING (Random Forest & XGBoost)")
    print("="*70)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = neg_count / pos_count

    tuned_results = {}
    tuned_pipelines = {}

    # 1. Tuning Random Forest
    print("\n[*] Hyperparameter Tuning: Random Forest Classifier...")
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1))
    ])

    rf_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [8, 12, 16],
        'classifier__min_samples_split': [5, 10],
        'classifier__min_samples_leaf': [2, 4]
    }

    rf_search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=rf_param_grid,
        n_iter=6,
        scoring='roc_auc',
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    rf_search.fit(X_train, y_train)
    best_rf_pipeline = rf_search.best_estimator_
    tuned_pipelines['tuned_random_forest'] = best_rf_pipeline

    rf_metrics, rf_cm, rf_roc, _ = evaluate_and_plot(
        best_rf_pipeline, X_test, y_test, "Tuned Random Forest", "tuned_random_forest"
    )

    rf_cleaned_params = {k.replace('classifier__', ''): str(v) for k, v in rf_search.best_params_.items()}
    rf_cleaned_params['best_cv_roc_auc'] = f"{rf_search.best_score_:.4f}"

    rf_run_id = log_run_to_mlflow(
        run_name="Tuned_Random_Forest",
        model_pipeline=best_rf_pipeline,
        params=rf_cleaned_params,
        metrics=rf_metrics,
        cm_path=rf_cm,
        roc_path=rf_roc,
        model_type="tuned",
        algo="RandomForestClassifier"
    )

    rf_pkl = os.path.join(OUTPUT_DIR, "tuned_random_forest.pkl")
    joblib.dump(best_rf_pipeline, rf_pkl)
    print(f"    [Pickle Saved] {rf_pkl}")
    tuned_results["Tuned Random Forest"] = {
        "key": "tuned_random_forest",
        "metrics": rf_metrics,
        "best_params": rf_cleaned_params,
        "run_id": rf_run_id,
        "pkl_path": rf_pkl
    }

    # 2. Tuning XGBoost
    print("\n[*] Hyperparameter Tuning: XGBoost Classifier...")
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            scale_pos_weight=scale_pos,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    xgb_param_grid = {
        'classifier__n_estimators': [100, 150],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.03, 0.1],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    }

    xgb_search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=xgb_param_grid,
        n_iter=6,
        scoring='roc_auc',
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    xgb_search.fit(X_train, y_train)
    best_xgb_pipeline = xgb_search.best_estimator_
    tuned_pipelines['tuned_xgboost'] = best_xgb_pipeline

    xgb_metrics, xgb_cm, xgb_roc, _ = evaluate_and_plot(
        best_xgb_pipeline, X_test, y_test, "Tuned XGBoost", "tuned_xgboost"
    )

    xgb_cleaned_params = {k.replace('classifier__', ''): str(v) for k, v in xgb_search.best_params_.items()}
    xgb_cleaned_params['best_cv_roc_auc'] = f"{xgb_search.best_score_:.4f}"

    xgb_run_id = log_run_to_mlflow(
        run_name="Tuned_XGBoost",
        model_pipeline=best_xgb_pipeline,
        params=xgb_cleaned_params,
        metrics=xgb_metrics,
        cm_path=xgb_cm,
        roc_path=xgb_roc,
        model_type="tuned",
        algo="XGBClassifier"
    )

    xgb_pkl = os.path.join(OUTPUT_DIR, "tuned_xgboost.pkl")
    joblib.dump(best_xgb_pipeline, xgb_pkl)
    print(f"    [Pickle Saved] {xgb_pkl}")
    tuned_results["Tuned XGBoost"] = {
        "key": "tuned_xgboost",
        "metrics": xgb_metrics,
        "best_params": xgb_cleaned_params,
        "run_id": xgb_run_id,
        "pkl_path": xgb_pkl
    }

    return tuned_results, tuned_pipelines


def generate_comparison_table(baseline_results, tuned_results):
    """
    Builds and displays comparative evaluation tables.
    """
    print("\n" + "="*70)
    print("STEP 4: MODEL PERFORMANCE COMPARISON TABLE")
    print("="*70)

    records = []
    for model_name, res in baseline_results.items():
        m = res['metrics']
        records.append({
            "Model Stage": "Baseline",
            "Model Name": model_name,
            "Accuracy": round(m['accuracy'], 4),
            "Precision": round(m['precision'], 4),
            "Recall": round(m['recall'], 4),
            "F1-Score": round(m['f1'], 4),
            "ROC-AUC": round(m['roc_auc'], 4),
            "PR-AUC": round(m['pr_auc'], 4),
            "Pickle File": os.path.basename(res['pkl_path'])
        })

    for model_name, res in tuned_results.items():
        m = res['metrics']
        records.append({
            "Model Stage": "Tuned",
            "Model Name": model_name,
            "Accuracy": round(m['accuracy'], 4),
            "Precision": round(m['precision'], 4),
            "Recall": round(m['recall'], 4),
            "F1-Score": round(m['f1'], 4),
            "ROC-AUC": round(m['roc_auc'], 4),
            "PR-AUC": round(m['pr_auc'], 4),
            "Pickle File": os.path.basename(res['pkl_path'])
        })

    comparison_df = pd.DataFrame(records)
    print("\n", comparison_df.to_string(index=False))

    # Save comparison to CSV
    csv_path = "./model_comparison_metrics.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n[*] Saved comparison table to {csv_path}")

    return comparison_df


def select_and_save_best_model(baseline_results, tuned_results, all_pipelines):
    """
    Selects best model based on F1 and ROC-AUC trade-off and registers it.
    """
    print("\n" + "="*70)
    print("STEP 5: MODEL SELECTION & FINAL SAVING")
    print("="*70)

    # Combine all results
    all_models = {**baseline_results, **tuned_results}
    
    # Selection criteria: Harmonic balance between ROC-AUC and F1-Score
    # Returns in e-commerce: Catching returns (Recall) without excessive false alarms (Precision)
    best_name = max(all_models.keys(), key=lambda k: all_models[k]['metrics']['roc_auc'] + all_models[k]['metrics']['f1'])
    best_info = all_models[best_name]
    best_key = best_info['key']
    best_pipeline = all_pipelines[best_key]

    print(f"[*] Best Model Selected: '{best_name}'")
    print(f"    ROC-AUC:   {best_info['metrics']['roc_auc']:.4f}")
    print(f"    F1-Score:  {best_info['metrics']['f1']:.4f}")
    print(f"    Recall:    {best_info['metrics']['recall']:.4f}")
    print(f"    Precision: {best_info['metrics']['precision']:.4f}")

    # Save as best_model.pkl
    best_pkl_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
    joblib.dump(best_pipeline, best_pkl_path)
    print(f"[*] Final Production Model saved to: {best_pkl_path}")

    # Save comprehensive registry manifest
    metadata = {
        "selected_model": best_name,
        "selected_model_key": best_key,
        "metrics": best_info['metrics'],
        "mlflow_run_id": best_info['run_id'],
        "all_saved_pickles": {
            k: os.path.basename(v['pkl_path']) for k, v in all_models.items()
        }
    }
    with open(os.path.join(OUTPUT_DIR, "model_registry_summary.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[*] Metadata registry written to {os.path.join(OUTPUT_DIR, 'model_registry_summary.json')}")

    # Verification load test
    loaded_model = joblib.load(best_pkl_path)
    print("[*] Successfully verified loading 'best_model.pkl' from disk!")
    
    return best_name, best_info


def main():
    print("="*70)
    print("SHOPIFY SALES RETURN PREDICTION - END-TO-END ML & TRACKING PIPELINE")
    print("="*70)

    # 1. Dataset Preparation
    X_train, X_test, y_train, y_test, preprocessor, feature_cols, num_cols, cat_cols = load_and_preprocess_data()

    # 2. Baseline Model Training (5 Models)
    baseline_results, baseline_pipelines = train_baseline_models(
        X_train, y_train, X_test, y_test, preprocessor
    )

    # 3. Hyperparameter Tuning (Top 2 Models)
    tuned_results, tuned_pipelines = tune_top_models(
        X_train, y_train, X_test, y_test, preprocessor
    )

    # Combine all pipelines for selection
    all_pipelines = {**baseline_pipelines, **tuned_pipelines}

    # 4. Comparison Table
    comparison_df = generate_comparison_table(baseline_results, tuned_results)

    # 5. Model Selection & Final Serialization
    best_name, best_info = select_and_save_best_model(baseline_results, tuned_results, all_pipelines)

    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE!")
    print(f"All 7 model pickle files + best_model.pkl saved to: '{OUTPUT_DIR}'")
    print(f"MLflow runs and metrics logged under experiment: '{EXPERIMENT_NAME}'")
    print("To launch MLflow UI, run:")
    print("    python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")
    print("="*70)


if __name__ == "__main__":
    main()
