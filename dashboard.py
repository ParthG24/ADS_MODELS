"""
Experiment 8: Streamlit Dashboard — Shopify Returns ML Pipeline
===============================================================
Aim    : Interactive dashboard for predictions, model insights, SHAP plots,
         LIME explanations, and Responsible AI fairness audit results.

Run    : streamlit run dashboard.py

Author : Experiment Team
Date   : September 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from PIL import Image

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopify Returns ML Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 12px; padding: 1rem;
        border-left: 4px solid #667eea; margin-bottom: 0.5rem;
    }
    .risk-high   { color: #e74c3c; font-weight: bold; font-size: 1.4rem; }
    .risk-medium { color: #f39c12; font-weight: bold; font-size: 1.4rem; }
    .risk-low    { color: #27ae60; font-weight: bold; font-size: 1.4rem; }
    .section-header { border-bottom: 2px solid #667eea; padding-bottom: 0.3rem;
                      margin: 1.5rem 0 1rem 0; font-size: 1.3rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = "saved_models"
EXP5_DIR   = "experiment5_outputs"
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
REGISTRY   = os.path.join(MODELS_DIR, "model_registry_summary.json")
METRICS_CSV= "model_comparison_metrics.csv"


# ── Load Artifacts (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_registry():
    with open(REGISTRY) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_exp5_report():
    path = os.path.join(EXP5_DIR, "experiment5_report.json")
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_metrics_csv():
    return pd.read_csv(METRICS_CSV)

model    = load_model()
registry = load_registry()
exp5     = load_exp5_report()
metrics_df = load_metrics_csv()


# ── Feature Engineering Helper ─────────────────────────────────────────────────
def build_features(product_category, product_price, discount_percent, quantity,
                   customer_country, traffic_source, payment_method,
                   shipping_cost, rating):
    from datetime import datetime
    now = datetime.now()
    order_amount_pre_discount = product_price * quantity
    discount_amount           = order_amount_pre_discount * (discount_percent / 100.0)
    shipping_ratio            = shipping_cost / (order_amount_pre_discount + 1e-5)
    price_per_rating          = product_price / (rating + 1e-5)
    day_of_week               = now.weekday()
    is_weekend                = int(day_of_week >= 5)
    calc_cost_gap             = product_price - shipping_cost
    price_per_unit            = product_price
    high_value_order          = int(order_amount_pre_discount > 500)
    days_since_launch         = 365

    if discount_percent == 0:      discount_tier = "Unknown"
    elif discount_percent <= 10:   discount_tier = "Low"
    elif discount_percent <= 25:   discount_tier = "Medium"
    else:                          discount_tier = "High"

    return pd.DataFrame([{
        "product_category":          product_category,
        "product_price":             product_price,
        "discount_percent":          discount_percent,
        "quantity":                  quantity,
        "customer_country":          customer_country,
        "traffic_source":            traffic_source,
        "payment_method":            payment_method,
        "shipping_cost":             shipping_cost,
        "rating":                    rating,
        "order_month":               now.month,
        "day_of_week":               day_of_week,
        "is_weekend":                is_weekend,
        "days_since_launch":         days_since_launch,
        "discount_tier":             discount_tier,
        "calc_cost_gap":             calc_cost_gap,
        "price_per_unit":            price_per_unit,
        "high_value_order":          high_value_order,
        "order_amount_pre_discount": order_amount_pre_discount,
        "discount_amount":           discount_amount,
        "shipping_ratio":            shipping_ratio,
        "price_per_rating":          price_per_rating,
        "order_dayofweek":           day_of_week,
    }])


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
    st.markdown("## 🛍️ Shopify Returns ML")
    st.markdown("**Experiment 8 Dashboard**")
    st.divider()
    page = st.radio(
        "Navigate to",
        ["🔮 Return Predictor", "📊 Model Performance",
         "🔍 SHAP Explainability", "🧩 LIME Local Explanations",
         "⚖️ Fairness Audit"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption(f"Champion: **{registry['selected_model']}**")
    st.caption(f"F1: {registry['metrics']['f1']:.4f} | ROC-AUC: {registry['metrics']['roc_auc']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Return Predictor
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Return Predictor":
    st.markdown('<p class="main-header">🔮 Return Risk Predictor</p>', unsafe_allow_html=True)
    st.markdown("Enter order details to predict the probability of a return.")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        product_category = st.selectbox("Product Category",
            ["Electronics","Fashion","Home Decor","Beauty","Sports","Accessories","Footwear"])
        product_price    = st.number_input("Product Price (USD)", 5.0, 800.0, 299.99, step=5.0)
        discount_percent = st.slider("Discount (%)", 0, 40, 20)
        quantity         = st.number_input("Quantity", 1, 10, 2)

    with col2:
        customer_country = st.selectbox("Customer Country",
            ["USA","UK","Canada","Australia","Germany","UAE","India"])
        traffic_source   = st.selectbox("Traffic Source",
            ["Organic","Direct","Paid Ads","Social Media","Email"])
        payment_method   = st.selectbox("Payment Method",
            ["Credit Card","Debit Card","PayPal","Apple Pay","Cash on Delivery"])

    with col3:
        shipping_cost = st.number_input("Shipping Cost (USD)", 2.0, 25.0, 12.50, step=0.5)
        rating        = st.slider("Product Rating", 1.0, 5.0, 3.5, step=0.1)
        st.markdown("&nbsp;")
        predict_btn = st.button("🔮 Predict Return Risk", type="primary", use_container_width=True)

    if predict_btn:
        X = build_features(product_category, product_price, discount_percent,
                           quantity, customer_country, traffic_source,
                           payment_method, shipping_cost, rating)
        proba = model.predict_proba(X)[0]
        pred  = model.predict(X)[0]
        ret_p = float(proba[1])

        st.divider()
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Return Probability",  f"{ret_p*100:.1f}%")
        r2.metric("No-Return Probability", f"{proba[0]*100:.1f}%")
        r3.metric("Prediction", "RETURNED ❌" if pred == 1 else "KEPT ✅")

        if ret_p < 0.30:
            risk_html = '<span class="risk-low">🟢 LOW RISK</span>'
        elif ret_p < 0.60:
            risk_html = '<span class="risk-medium">🟡 MEDIUM RISK</span>'
        else:
            risk_html = '<span class="risk-high">🔴 HIGH RISK</span>'

        r4.markdown(f"**Risk Level**<br>{risk_html}", unsafe_allow_html=True)

        # Progress bar
        st.progress(ret_p, text=f"Return probability: {ret_p*100:.1f}%")

        # Key drivers notice
        top_feat = exp5["shap"]["top_3_features"]
        st.info(f"📌 Top drivers of return predictions: **{top_feat[0]}**, **{top_feat[1]}**, **{top_feat[2]}** (from SHAP analysis)")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown('<p class="main-header">📊 Model Performance Benchmarks</p>', unsafe_allow_html=True)
    st.markdown("Comparison of all 7 trained models from Experiment 4 (v2) on the modified dataset.")
    st.divider()

    # Format table
    df_show = metrics_df.copy()
    for col in ["Accuracy","Precision","Recall","F1-Score","ROC-AUC","PR-AUC"]:
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(lambda x: f"{x:.4f}")

    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.markdown('<p class="section-header">Champion Model Summary</p>', unsafe_allow_html=True)
    m = registry["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model",     registry["selected_model"])
    c2.metric("F1-Score",  f"{m['f1']:.4f}")
    c3.metric("ROC-AUC",   f"{m['roc_auc']:.4f}")
    c4.metric("Recall",    f"{m['recall']:.4f}")
    c5.metric("Precision", f"{m['precision']:.4f}")

    st.markdown('<p class="section-header">Key Insights</p>', unsafe_allow_html=True)
    st.markdown("""
    - **Return Rate** jumped from 14.8% (original dataset) to **27.9%** (modified dataset), indicating successful skewing of `discount_percent`.
    - **Logistic Regression** achieved the highest F1 (0.5334), demonstrating that a linear decision boundary is sufficient when discount-driven patterns are dominant.
    - **ROC-AUC ~0.73** across all models confirms that the feature set has genuine discriminative power.
    - The **class imbalance ratio improved** from 5.75:1 to ~2.6:1, which improved Recall significantly across all models.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SHAP Explainability
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 SHAP Explainability":
    st.markdown('<p class="main-header">🔍 SHAP Global Explainability</p>', unsafe_allow_html=True)
    st.markdown("SHAP (SHapley Additive exPlanations) values show the global contribution of each feature to the model's return predictions.")
    st.divider()

    top_feats = exp5["shap"]["top_3_features"]
    top_vals  = exp5["shap"]["mean_abs_shap_top3"]
    t1, t2, t3 = st.columns(3)
    t1.metric(f"🥇 {top_feats[0]}", f"|SHAP| = {top_vals[0]:.4f}", "Dominant Feature")
    t2.metric(f"🥈 {top_feats[1]}", f"|SHAP| = {top_vals[1]:.4f}")
    t3.metric(f"🥉 {top_feats[2]}", f"|SHAP| = {top_vals[2]:.4f}")

    st.markdown('<p class="section-header">Global Feature Importance (Bar)</p>', unsafe_allow_html=True)
    img_bar = Image.open(os.path.join(EXP5_DIR, "shap_summary_bar.png"))
    st.image(img_bar, use_container_width=True)

    st.markdown('<p class="section-header">Feature Impact Direction (Beeswarm)</p>', unsafe_allow_html=True)
    img_bee = Image.open(os.path.join(EXP5_DIR, "shap_summary_beeswarm.png"))
    st.image(img_bee, use_container_width=True)

    st.markdown('<p class="section-header">Dependence Plots (Top-3 Features)</p>', unsafe_allow_html=True)
    dep_cols = st.columns(3)
    for i, feat in enumerate(top_feats):
        safe = feat.replace("/","_").replace(" ","_")
        dep_path = os.path.join(EXP5_DIR, f"shap_dependence_{safe}.png")
        if os.path.exists(dep_path):
            with dep_cols[i]:
                st.image(Image.open(dep_path), caption=f"Dependence: {feat}", use_container_width=True)

    st.markdown('<p class="section-header">Interpretation</p>', unsafe_allow_html=True)
    st.markdown(f"""
    - **`{top_feats[0]}`** is the dominant predictor with a mean |SHAP| of **{top_vals[0]:.4f}** — significantly higher than all other features.
      High discount percentages strongly push the prediction towards "returned."
    - **`{top_feats[1]}`** (mean |SHAP| = {top_vals[1]:.4f}) — lower product ratings correlate with higher return probability.
    - **`{top_feats[2]}`** (mean |SHAP| = {top_vals[2]:.4f}) — higher shipping costs appear to slightly discourage impulse returns.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: LIME Local Explanations
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧩 LIME Local Explanations":
    st.markdown('<p class="main-header">🧩 LIME Local Explanations</p>', unsafe_allow_html=True)
    st.markdown("LIME (Local Interpretable Model-agnostic Explanations) explains **individual predictions** by approximating the model locally with a linear model.")
    st.divider()

    sample_choice = st.selectbox(
        "Select a sample prediction to explain:",
        ["High Confidence Return (P>0.75, Actual=Returned)",
         "High Confidence Non-Return (P<0.25, Actual=Not Returned)",
         "Borderline Prediction (0.45 < P < 0.55)"]
    )
    sample_map = {
        "High Confidence Return (P>0.75, Actual=Returned)":         "lime_high_confidence_return.png",
        "High Confidence Non-Return (P<0.25, Actual=Not Returned)":  "lime_high_confidence_non_return.png",
        "Borderline Prediction (0.45 < P < 0.55)":                  "lime_borderline_prediction.png",
    }
    lime_path = os.path.join(EXP5_DIR, sample_map[sample_choice])
    if os.path.exists(lime_path):
        st.image(Image.open(lime_path), use_container_width=True)

    st.markdown('<p class="section-header">How to Read LIME</p>', unsafe_allow_html=True)
    st.markdown("""
    - **Orange bars** → features pushing prediction toward **Returned (class 1)**
    - **Blue bars**   → features pushing prediction toward **Not Returned (class 0)**
    - Bar **length** indicates the strength of that feature's local influence
    - Features are ranked by absolute contribution magnitude
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Fairness Audit
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Fairness Audit":
    st.markdown('<p class="main-header">⚖️ Fairness Audit (Fairlearn)</p>', unsafe_allow_html=True)
    st.markdown("Bias audit across **customer country** as the sensitive attribute.")
    st.divider()

    fair_b = exp5["fairness_before_mitigation"]
    fair_a = exp5["fairness_after_mitigation"]

    st.markdown('<p class="section-header">Fairness Metrics Summary</p>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Demographic Parity Diff", f"{fair_b['demographic_parity_difference']:.4f}",
              help="Closer to 0 = fairer. Max difference in positive prediction rate across groups.")
    f2.metric("Demographic Parity Ratio", f"{fair_b['demographic_parity_ratio']:.4f}",
              help="Closer to 1 = fairer.")
    f3.metric("Equalized Odds Diff", f"{fair_b['equalized_odds_difference']:.4f}",
              help="Max difference in FPR or TPR across groups.")
    f4.metric("Post-Mitigation DPD", f"{fair_a['demographic_parity_difference']:.4f}",
              delta=f"{fair_a['demographic_parity_difference']-fair_b['demographic_parity_difference']:.4f}",
              delta_color="inverse")

    st.markdown('<p class="section-header">Per-Group Performance by Country</p>', unsafe_allow_html=True)
    img_grp = Image.open(os.path.join(EXP5_DIR, "fairlearn_group_metrics.png"))
    st.image(img_grp, use_container_width=True)

    st.markdown('<p class="section-header">Equalized Odds Heatmap (FPR & FNR)</p>', unsafe_allow_html=True)
    img_heat = Image.open(os.path.join(EXP5_DIR, "fairlearn_equalized_odds_heatmap.png"))
    st.image(img_heat, use_container_width=True)

    st.markdown('<p class="section-header">Bias Mitigation: Before vs After</p>', unsafe_allow_html=True)
    img_mit = Image.open(os.path.join(EXP5_DIR, "fairlearn_mitigation_comparison.png"))
    st.image(img_mit, use_container_width=True)

    st.markdown('<p class="section-header">Fairness Assessment</p>', unsafe_allow_html=True)
    dpd = fair_b["demographic_parity_difference"]
    if dpd < 0.05:
        st.success(f"✅ **Low Bias Detected**: DPD = {dpd:.4f} — the model treats all customer countries near-equally. No significant demographic disparity.")
    elif dpd < 0.10:
        st.warning(f"⚠️ **Moderate Bias**: DPD = {dpd:.4f} — some disparity exists. Mitigation applied, reducing DPD to {fair_a['demographic_parity_difference']:.4f}.")
    else:
        st.error(f"❌ **High Bias Detected**: DPD = {dpd:.4f} — significant fairness concern. Mitigation required.")
