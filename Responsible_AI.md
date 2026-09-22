# Responsible AI Audit Report: Shopify Returns Prediction Pipeline
**Experiment 8: Explainability, Fairness, and Responsible AI Governance**  
**Date:** September 2026  
**Auditor:** ML Quality & Responsible AI Engineering Team  
**System Evaluated:** XGBoost Production Model (`saved_models/best_model.pkl`)  
**Sensitive Attribute Evaluated:** `customer_country` (7 geographic cohorts: Australia, Canada, Germany, India, UAE, UK, USA)

---

## 1. Executive Summary

This Responsible AI Audit evaluates the production Machine Learning pipeline for Shopify Product Return Prediction. As algorithmic return predictions can directly influence merchant inventory management, customer return policies, and checkout experiences, ensuring **fairness**, **transparency**, and **accountability** is essential.

### Key Audit Findings:
1. **Explainability Validation**: SHAP analysis definitively demonstrates that `discount_percent` is the single most dominant driver of return probability (mean $|SHAP| = 0.6300$), followed by `rating` ($0.4248$) and `shipping_cost` ($0.0803$). This aligns with project design specifications where heavy discounting skews consumer impulse purchase behavior and subsequent return rates.
2. **Fairness Assessment**: Evaluating predictions across 7 customer countries revealed strong baseline demographic fairness:
   - **Demographic Parity Difference (DPD)**: $0.0298$ (well below the $0.05$ threshold for notable disparity).
   - **Demographic Parity Ratio (DPR)**: $0.9315$ (surpassing the standard 80% Four-Fifths rule for equal impact).
   - **Equalized Odds Difference (EOD)**: $0.0730$.
3. **Mitigation Performance**: Applying Fairlearn's `ThresholdOptimizer` with Demographic Parity constraint further compressed the Demographic Parity Difference to **$0.0034$** (an **$88.6\%$** reduction in cross-group disparity).

---

## 2. Model Overview & Intended Use

| Attribute | Specification |
|---|---|
| **Model Architecture** | Tuned XGBoost Classifier (`saved_models/best_model.pkl`) |
| **Target Variable** | `return_status` (Binary: 0 = Retained, 1 = Returned) |
| **Primary Domain** | E-commerce transaction risk scoring and proactive return management |
| **Intended Users** | E-commerce merchants, logistics planners, and customer retention systems |
| **Unintended Uses** | Punitive customer blacklisting, discriminatory pricing, or regional order cancellation |

---

## 3. Explainability Analysis (XAI)

### 3.1 Global Feature Importance (SHAP)
SHAP (SHapley Additive exPlanations) values computed on the test cohort revealed clear feature attribution rankings:

| Rank | Feature | Mean \|SHAP Value\| | Relative Impact | Interpretation |
|:---:|---|:---:|:---:|---|
| 1 | `discount_percent` | **0.6300** | Primary Driver | Higher discounts induce impulse purchases with higher likelihood of return |
| 2 | `rating` | **0.4248** | Secondary Driver | Lower product ratings strongly push probability toward return |
| 3 | `shipping_cost` | **0.0803** | Moderate Driver | Orders with high shipping friction have elevated return sensitivity |
| 4 | `discount_amount` | **0.0668** | Minor Driver | Corroborates dollar value of discount |
| 5 | `price_per_unit` | **0.0537** | Minor Driver | Unit price context relative to category |
| 6 | `price_per_rating` | **0.0466** | Minor Driver | Interaction feature: perceived value for money |
| 7 | `shipping_ratio` | **0.0410** | Minor Driver | Ratio of shipping fee to gross order total |

Geographic one-hot indicators (`customer_country_*`) each contributed $< 0.007$ mean $|SHAP|$, demonstrating that geographic origin does not exert disproportionate algorithmic leverage on individual decisions.

### 3.2 Local Explanations (LIME)
LIME (Local Interpretable Model-agnostic Explanations) validates decision fidelity at the single-transaction level:
- **High-Risk Case (Prediction = 93% Return)**: Heavily driven by `discount_percent > 25%` and `rating < 3.5`.
- **Low-Risk Case (Prediction = 8% Return)**: Driven by `discount_percent < 5%`, `rating >= 4.5`, and zero shipping friction.
- **Borderline Case (Prediction = 49% Return)**: Competing vectors where high discount is counteracted by a 5.0 product rating.

---

## 4. Algorithmic Fairness & Bias Audit

The fairness evaluation assessed whether the classifier exhibits bias across customer countries under equal opportunity and demographic parity frameworks.

### 4.1 Per-Group Performance Breakdown

| Customer Country | Accuracy | Recall (Sensitivity) | False Positive Rate (FPR) | False Negative Rate (FNR) |
|---|:---:|:---:|:---:|:---:|
| **Australia** | 69.08% | 70.62% | 31.51% | 29.38% |
| **Canada** | 66.39% | 66.43% | 33.62% | 33.57% |
| **Germany** | 64.81% | 63.33% | 34.58% | 36.67% |
| **India** | 67.56% | 64.15% | 31.07% | 35.85% |
| **UAE** | 67.64% | 65.61% | 31.54% | 34.39% |
| **UK** | 69.01% | 67.67% | 30.46% | 32.33% |
| **USA** | 65.79% | 66.88% | 34.62% | 33.12% |

### 4.2 Group Fairness Metrics Summary

| Metric | Measured Baseline | Regulatory / Industry Standard | Status |
|---|:---:|:---:|:---:|
| **Demographic Parity Difference** | `0.0298` | $< 0.10$ acceptable, $< 0.05$ optimal | **PASSED** (Low Disparity) |
| **Demographic Parity Ratio** | `0.9315` | $\ge 0.80$ (EEOC 80% Rule) | **PASSED** (Compliant) |
| **Equalized Odds Difference** | `0.0730` | $< 0.10$ acceptable | **PASSED** (Acceptable) |

---

## 5. Fairness Mitigation

To achieve equitable acceptance across geographic regions without degrading predictive capability, Fairlearn's `ThresholdOptimizer` was trained using Demographic Parity constraints:

| Metric | Baseline Model | Mitigated Model | Improvement |
|---|:---:|:---:|:---:|
| **Demographic Parity Difference** | $0.0298$ | **$0.0034$** | **$-88.6\%$** disparity reduction |
| **Equalized Odds Difference** | $0.0730$ | $0.0790$ | Stable ($+0.006$) |
| **Mitigated Model F1-Score** | 0.5410 | 0.5282 | Minimal utility loss ($-0.013$) |

The optimization equalizes the positive prediction rate across all regional cohorts with virtually zero loss of predictive utility.

---

## 6. Ethical Risk Matrix & Safeguards

| Risk Category | Potential Harms | Mitigation Implemented | Ongoing Safeguard |
|---|---|---|---|
| **Disparate Impact** | High return predictions for specific countries causing restrictive shipping terms | Fairlearn ThresholdOptimizer; Country features have low SHAP importance ($< 0.007$) | Periodic automated bias auditing via CI/CD |
| **Explainability Gap** | Merchants distrusting automated return risk tags | Global SHAP summaries and local LIME waterfall plots embedded in Streamlit Dashboard | Mandatory human-in-the-loop review for high-value orders |
| **Data Drift** | Shifting holiday discount policies invalidating discount thresholds | Automated Evidently AI drift monitoring and CI unit tests (`test_pipeline.py`) | Retraining trigger when Wasserstein distance exceeds threshold |

---

## 7. Operational Deployment & Dashboard Integration

The Responsible AI insights are accessible to non-technical stakeholders via the **Streamlit Dashboard** (`dashboard.py`):
1. **Live Prediction Tab**: Enables merchants to test hypothetical orders and view real-time risk scores with explainability factors.
2. **XAI Interpretability Tab**: Displays global SHAP feature distributions and dependence interactions.
3. **Fairness Audit Tab**: Visualizes Demographic Parity across nations and toggles pre- and post-mitigation metrics.
4. **FastAPI Integration**: Microservice exposes `/predict` with standardized input validation (Pydantic schema).

---
*Report certified by ML Governance Team — Experiments 5, 6, 7, and 8 validated.*
