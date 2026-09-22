"""
Experiment 6: FastAPI Prediction API
=====================================
Aim    : Package the trained Shopify Returns ML model into a REST API
         for real-time return-probability predictions.
Endpoint: POST /predict
Model   : saved_models/best_model.pkl (Logistic Regression pipeline)

Run locally  : uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Run in Docker: docker build -t shopify-returns-api . && docker run -p 8000:8000 shopify-returns-api

Author : Experiment Team
Date   : September 2026
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
import joblib
import numpy as np
import pandas as pd
import os
import json
import uvicorn
from datetime import datetime

# ── App Initialisation ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Shopify Returns Prediction API",
    description=(
        "Predicts the probability that a Shopify order will be returned. "
        "Powered by a scikit-learn Logistic Regression pipeline trained on "
        "60,000 orders with Experiment-4 feature engineering."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Loading ──────────────────────────────────────────────────────────────
MODEL_PATH    = "saved_models/best_model.pkl"
REGISTRY_PATH = "saved_models/model_registry_summary.json"

_model    = None
_registry = None


def get_model():
    global _model, _registry
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    if _registry is None and os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            _registry = json.load(f)
    return _model, _registry


@app.on_event("startup")
async def load_on_startup():
    get_model()
    print("[API] Model loaded successfully on startup.")


# ── Request / Response Schemas ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Input features for a single Shopify order.
    All fields correspond to pre-purchase information available at checkout.
    """
    product_category: Literal[
        "Electronics", "Fashion", "Home Decor", "Beauty",
        "Sports", "Accessories", "Footwear"
    ] = Field(..., example="Electronics")

    product_price: float = Field(..., gt=0, example=299.99,
                                 description="Base retail price (USD)")
    discount_percent: float = Field(..., ge=0, le=40, example=20.0,
                                    description="Discount applied (0-40%)")
    quantity: int = Field(..., ge=1, le=10, example=2)
    customer_country: Literal[
        "USA", "UK", "Canada", "Australia", "Germany", "UAE", "India"
    ] = Field(..., example="USA")

    traffic_source: Literal[
        "Organic", "Direct", "Paid Ads", "Social Media", "Email"
    ] = Field(..., example="Paid Ads")

    payment_method: Literal[
        "Credit Card", "Debit Card", "PayPal", "Apple Pay", "Cash on Delivery"
    ] = Field(..., example="Credit Card")

    shipping_cost: float = Field(..., ge=0, example=12.50,
                                 description="Shipping cost (USD)")
    rating: float = Field(..., ge=1.0, le=5.0, example=3.8,
                          description="Historical product rating (1-5)")

    # Pre-computed enriched fields (optional — will be derived if not provided)
    order_month: Optional[int] = Field(None, ge=1, le=12, example=6)
    day_of_week: Optional[int] = Field(None, ge=0, le=6, example=2)
    is_weekend: Optional[int] = Field(None, ge=0, le=1, example=0)
    days_since_launch: Optional[int] = Field(None, ge=0, example=180)
    discount_tier: Optional[Literal["Low", "Medium", "High", "Unknown"]] = Field(
        None, example="High"
    )
    calc_cost_gap: Optional[float] = Field(None, example=15.0)
    price_per_unit: Optional[float] = Field(None, example=149.99)
    high_value_order: Optional[int] = Field(None, ge=0, le=1, example=1)


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="0 = Not Returned, 1 = Returned")
    return_probability: float = Field(..., description="Probability of return (0-1)")
    not_return_probability: float = Field(..., description="Probability of NOT returning (0-1)")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH return risk")
    model_used: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    f1_score: float
    roc_auc: float
    recall: float
    model_file: str
    api_version: str


# ── Feature Engineering Helper ─────────────────────────────────────────────────

def build_feature_row(req: PredictRequest) -> pd.DataFrame:
    """
    Constructs a single-row DataFrame with all features expected by the pipeline.
    Derives engineered features from raw inputs where not provided.
    """
    order_amount_pre_discount = req.product_price * req.quantity
    discount_amount           = order_amount_pre_discount * (req.discount_percent / 100.0)
    shipping_ratio            = req.shipping_cost / (order_amount_pre_discount + 1e-5)
    price_per_rating          = req.product_price / (req.rating + 1e-5)

    # Infer optional fields if not provided
    order_month       = req.order_month       or datetime.now().month
    day_of_week       = req.day_of_week       if req.day_of_week is not None else datetime.now().weekday()
    is_weekend        = req.is_weekend        if req.is_weekend is not None else int(day_of_week >= 5)
    days_since_launch = req.days_since_launch if req.days_since_launch is not None else 365
    calc_cost_gap     = req.calc_cost_gap     if req.calc_cost_gap is not None else (req.product_price - req.shipping_cost)
    price_per_unit    = req.price_per_unit    if req.price_per_unit is not None else req.product_price
    high_value_order  = req.high_value_order  if req.high_value_order is not None else int(order_amount_pre_discount > 500)

    # Derive discount_tier from discount_percent if not provided
    if req.discount_tier is None:
        if req.discount_percent == 0:
            discount_tier = "Unknown"
        elif req.discount_percent <= 10:
            discount_tier = "Low"
        elif req.discount_percent <= 25:
            discount_tier = "Medium"
        else:
            discount_tier = "High"
    else:
        discount_tier = req.discount_tier

    row = {
        "product_category":          req.product_category,
        "product_price":             req.product_price,
        "discount_percent":          req.discount_percent,
        "quantity":                  req.quantity,
        "customer_country":          req.customer_country,
        "traffic_source":            req.traffic_source,
        "payment_method":            req.payment_method,
        "shipping_cost":             req.shipping_cost,
        "rating":                    req.rating,
        "order_month":               order_month,
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
        "order_dayofweek":           day_of_week,  # alias used by pipeline
    }
    return pd.DataFrame([row])


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Shopify Returns Prediction API",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe — returns 200 if model is loaded and API is healthy."""
    try:
        model, _ = get_model()
        return {
            "status":      "healthy",
            "model_loaded": model is not None,
            "timestamp":   datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Returns metadata about the currently loaded champion model."""
    _, registry = get_model()
    if registry is None:
        raise HTTPException(status_code=404, detail="Model registry not found")
    return ModelInfoResponse(
        model_name  = registry.get("selected_model", "Unknown"),
        f1_score    = registry["metrics"]["f1"],
        roc_auc     = registry["metrics"]["roc_auc"],
        recall      = registry["metrics"]["recall"],
        model_file  = MODEL_PATH,
        api_version = "1.0.0",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predicts whether a given Shopify order will be returned.

    - **prediction**: 0 (Not Returned) or 1 (Returned)
    - **return_probability**: Confidence score for the return class
    - **risk_level**: LOW (<30%), MEDIUM (30-60%), HIGH (>60%)
    """
    try:
        model, registry = get_model()
        X = build_feature_row(request)
        proba      = model.predict_proba(X)[0]
        pred       = int(model.predict(X)[0])
        ret_prob   = float(proba[1])
        no_ret_prob= float(proba[0])

        if ret_prob < 0.30:
            risk = "LOW"
        elif ret_prob < 0.60:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        return PredictResponse(
            prediction              = pred,
            return_probability      = round(ret_prob, 4),
            not_return_probability  = round(no_ret_prob, 4),
            risk_level              = risk,
            model_used              = registry.get("selected_model", "Unknown") if registry else "Unknown",
            timestamp               = datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── Dev server entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
