"""
tests/test_pipeline.py
======================
Unit tests for the Shopify Returns ML pipeline (Experiment 4/7).
Validates model artifacts, pipeline structure, and performance gates.

Run with: pytest tests/ -v
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

MODELS_DIR = "saved_models"
CSV_PATH   = "shopify_sales_final_cleaned_modified.csv"

EXPECTED_MODEL_FILES = [
    "best_model.pkl",
    "baseline_logistic_regression.pkl",
    "baseline_random_forest.pkl",
    "baseline_hist_gradient_boosting.pkl",
    "baseline_xgboost.pkl",
    "baseline_calibrated_linearsvc.pkl",
    "tuned_random_forest.pkl",
    "tuned_xgboost.pkl",
    "model_registry_summary.json",
]


@pytest.fixture(scope="module")
def best_model():
    """Load the champion model once for the test session."""
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    return joblib.load(path)


@pytest.fixture(scope="module")
def registry():
    """Load model registry JSON."""
    path = os.path.join(MODELS_DIR, "model_registry_summary.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sample_data():
    """Load a 200-row sample of the dataset with feature engineering applied."""
    df = pd.read_csv(CSV_PATH, nrows=200)

    # Feature engineering — mirrors ml_pipeline_v2.py
    df["order_amount_pre_discount"] = df["product_price"] * df["quantity"]
    df["discount_amount"] = (
        df["product_price"] * df["quantity"] * (df["discount_percent"] / 100.0)
    )
    df["shipping_ratio"] = df["shipping_cost"] / (df["order_amount_pre_discount"] + 1e-5)
    df["price_per_rating"] = df["product_price"] / (df["rating"] + 1e-5)
    if "order_date" in df.columns:
        ds = pd.to_datetime(df["order_date"], errors="coerce")
        df["order_dayofweek"] = ds.dt.dayofweek.fillna(0).astype(int)

    leakage_cols = [
        "order_id", "customer_id", "product_id",
        "discounted_price", "revenue", "profit", "order_date", "is_returned"
    ]
    feature_cols = [c for c in df.columns if c not in leakage_cols]
    X = df[feature_cols].copy()
    if "discount_tier" in X.columns:
        X["discount_tier"] = X["discount_tier"].fillna("Unknown")
    return X


# ── Tests: Artifact Existence ─────────────────────────────────────────────────

class TestArtifactsExist:
    """Verify all expected model artifacts are present on disk."""

    @pytest.mark.parametrize("filename", EXPECTED_MODEL_FILES)
    def test_model_file_exists(self, filename):
        path = os.path.join(MODELS_DIR, filename)
        assert os.path.exists(path), f"Missing artifact: {path}"

    def test_models_dir_exists(self):
        assert os.path.isdir(MODELS_DIR), f"saved_models/ directory not found"


# ── Tests: Pipeline Structure ─────────────────────────────────────────────────

class TestPipelineStructure:
    """Validate that the best model is a well-formed sklearn Pipeline."""

    def test_best_model_is_pipeline(self, best_model):
        assert isinstance(best_model, Pipeline), \
            f"Expected Pipeline, got {type(best_model)}"

    def test_pipeline_has_preprocessor(self, best_model):
        assert "preprocessor" in best_model.named_steps, \
            "Pipeline missing 'preprocessor' step"

    def test_pipeline_has_classifier(self, best_model):
        assert "classifier" in best_model.named_steps, \
            "Pipeline missing 'classifier' step"

    def test_model_has_predict(self, best_model):
        assert hasattr(best_model, "predict"), "Model missing predict() method"

    def test_model_has_predict_proba(self, best_model):
        assert hasattr(best_model, "predict_proba"), \
            "Model missing predict_proba() method"


# ── Tests: Inference Validity ─────────────────────────────────────────────────

class TestInference:
    """Validate model predictions are correctly shaped and within valid ranges."""

    def test_predict_output_shape(self, best_model, sample_data):
        preds = best_model.predict(sample_data)
        assert preds.shape == (len(sample_data),), \
            f"Expected shape ({len(sample_data)},), got {preds.shape}"

    def test_predict_binary_labels(self, best_model, sample_data):
        preds = best_model.predict(sample_data)
        unique_labels = set(preds)
        assert unique_labels.issubset({0, 1}), \
            f"Predictions contain non-binary values: {unique_labels}"

    def test_predict_proba_shape(self, best_model, sample_data):
        proba = best_model.predict_proba(sample_data)
        assert proba.shape == (len(sample_data), 2), \
            f"Expected proba shape ({len(sample_data)}, 2), got {proba.shape}"

    def test_predict_proba_sums_to_one(self, best_model, sample_data):
        proba = best_model.predict_proba(sample_data)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            "predict_proba rows do not sum to 1.0"

    def test_predict_proba_range(self, best_model, sample_data):
        proba = best_model.predict_proba(sample_data)
        assert proba.min() >= 0.0, "Probabilities contain negative values"
        assert proba.max() <= 1.0, "Probabilities exceed 1.0"

    def test_model_handles_unknown_categories(self, best_model, sample_data):
        """Model should not crash on unseen categories (handle_unknown='ignore')."""
        X_mod = sample_data.copy()
        if "product_category" in X_mod.columns:
            X_mod["product_category"] = "UnknownCategory_XYZ"
        # Should not raise
        preds = best_model.predict(X_mod)
        assert len(preds) == len(X_mod)


# ── Tests: Registry & Performance Gates ──────────────────────────────────────

class TestModelRegistry:
    """Validate model registry JSON and enforce minimum performance thresholds."""

    def test_registry_has_required_keys(self, registry):
        required = ["selected_model", "selected_model_key", "metrics",
                    "mlflow_run_id", "all_saved_pickles"]
        for key in required:
            assert key in registry, f"Registry missing key: '{key}'"

    def test_registry_metrics_present(self, registry):
        required_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        for m in required_metrics:
            assert m in registry["metrics"], f"Registry metrics missing: '{m}'"

    def test_minimum_f1_score(self, registry):
        f1 = registry["metrics"]["f1"]
        assert f1 >= 0.40, f"F1-Score {f1:.4f} below minimum threshold of 0.40"

    def test_minimum_roc_auc(self, registry):
        roc_auc = registry["metrics"]["roc_auc"]
        assert roc_auc >= 0.65, f"ROC-AUC {roc_auc:.4f} below minimum threshold of 0.65"

    def test_minimum_recall(self, registry):
        recall = registry["metrics"]["recall"]
        assert recall >= 0.40, f"Recall {recall:.4f} below minimum threshold of 0.40"

    def test_all_pickles_registered(self, registry):
        """Every pickle listed in registry must exist on disk."""
        for name, filename in registry["all_saved_pickles"].items():
            path = os.path.join(MODELS_DIR, filename)
            assert os.path.exists(path), \
                f"Registry references '{filename}' for '{name}' but file not found"
