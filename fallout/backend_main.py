# =============================================================================
# Backend: FastAPI - Retail Sales Dashboard + ML Inference  v3.1
#
# Run:
#   cd C:\Users\thiti\OneDrive\Desktop\fallout
#   python -m uvicorn backend_main:app --reload --host 0.0.0.0 --port 8000
#
# Models folder (next to this file):
#   models/revenue_model.pkl
#   models/scaler.pkl
#   models/encoder.pkl
#   models/model_metrics.json
#
# Copy from WSL:
#   \\wsl.localhost\Ubuntu-22.04\home\thiti\airflow\models\
# =============================================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
import calendar

app = FastAPI(title="Retail Sales Dashboard API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# PATHS  — __file__ gives the absolute path of this script
# =============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# =============================================================================
# LOAD ML MODELS AT STARTUP
# =============================================================================
model        = None
scaler       = None
encoders     = None
metrics      = {}
feature_cols = []
cat_features = []
num_features = []
target_col   = "total_amount"

def load_models():
    global model, scaler, encoders, metrics, feature_cols, cat_features, num_features, target_col
    try:
        print(f"Loading models from: {MODELS_DIR}")
        model    = joblib.load(os.path.join(MODELS_DIR, "revenue_model.pkl"))
        scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        encoders = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
        mpath    = os.path.join(MODELS_DIR, "model_metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                metrics = json.load(f)
        feature_cols = metrics.get("feature_columns", [])
        cat_features = metrics.get("cat_features",    [])
        num_features = metrics.get("num_features",    [])
        target_col   = metrics.get("target_column",   "total_amount")
        print("✅ Models loaded successfully!")
        print(f"   R2={metrics.get('r2')}  MAE={metrics.get('mae')}  RMSE={metrics.get('rmse')}")
        print(f"   Features: {feature_cols}")
    except Exception as e:
        print(f"❌ Could not load models: {e}")
        print(f"   Expected path: {MODELS_DIR}")
        print(f"   Files in folder: {os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else 'FOLDER NOT FOUND'}")

load_models()

# =============================================================================
# HELPERS
# =============================================================================
def detect_col(df: pd.DataFrame, *keywords) -> Optional[str]:
    for kw in keywords:
        match = next((c for c in df.columns if kw.lower() in c.lower()), None)
        if match:
            return match
    return None

def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except Exception:
            continue
    raise ValueError("Could not parse CSV with any encoding")

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\s/\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df

def encode_feature(col: str, val: Any) -> float:
    if col in cat_features and encoders and col in encoders:
        try:
            return float(encoders[col].transform([str(val)])[0])
        except ValueError:
            return 0.0
    try:
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0

# =============================================================================
# ROUTES
# =============================================================================
@app.get("/")
def root():
    return {
        "message":      "Retail Sales Dashboard API v3.1",
        "status":       "running",
        "model_loaded": model is not None,
    }

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "models_dir":   MODELS_DIR,
        "r2":           metrics.get("r2",   "N/A"),
        "mae":          metrics.get("mae",  "N/A"),
        "rmse":         metrics.get("rmse", "N/A"),
        "features":     feature_cols,
        "target":       target_col,
    }

# =============================================================================
# UPLOAD & ANALYZE
# =============================================================================
@app.post("/upload/analyze")
async def analyze_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file provided")

    raw   = await file.read()
    fname = file.filename.lower()
    print(f"Received: {file.filename} ({len(raw)} bytes)")

    try:
        if fname.endswith(".csv"):
            df = parse_csv(raw)
        else:
            raise HTTPException(400, "Please upload a CSV file (retail_sales_dataset.csv or cleaned_data.csv)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Parse error: {e}")

    df = clean_columns(df)
    print(f"Parsed: {df.shape}  columns: {list(df.columns)}")

    # Detect columns
    date_col  = detect_col(df, "date")
    amt_col   = detect_col(df, "total_amount", "amount", "total")
    cat_col   = detect_col(df, "product_category", "categ", "category")
    gen_col   = detect_col(df, "gender")
    age_col   = detect_col(df, "age")
    qty_col   = detect_col(df, "quantity", "quant")
    price_col = detect_col(df, "price_per_unit", "price")

    print(f"Detected: date={date_col} amount={amt_col} category={cat_col} gender={gen_col} age={age_col} qty={qty_col}")

    # Parse types
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    for c in [amt_col, qty_col, age_col, price_col]:
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    result: Dict[str, Any] = {}

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kpis = {"total_rows": len(df)}
    if amt_col:
        kpis["total_revenue"]   = round(float(df[amt_col].sum()),  2)
        kpis["avg_order_value"] = round(float(df[amt_col].mean()), 2)
        kpis["max_order"]       = round(float(df[amt_col].max()),  2)
        kpis["min_order"]       = round(float(df[amt_col].min()),  2)
    if qty_col:
        kpis["total_units_sold"] = int(df[qty_col].sum())
    if cat_col:
        kpis["num_categories"] = int(df[cat_col].nunique())
    result["kpis"] = kpis

    # ── Monthly trend ─────────────────────────────────────────────────────────
    if date_col and amt_col:
        tmp = df.dropna(subset=[date_col]).copy()
        tmp["_period"] = tmp[date_col].dt.to_period("M")
        monthly = tmp.groupby("_period")[amt_col].sum().reset_index()
        monthly.columns = ["month", "revenue"]
        monthly["month"] = monthly["month"].astype(str)
        result["monthly_trend"] = monthly.to_dict(orient="records")

    # ── Category breakdown ─────────────────────────────────────────────────────
    if cat_col and amt_col:
        cat_data = df.groupby(cat_col)[amt_col].agg(
            revenue="sum", orders="count", avg_order="mean"
        ).round(2).reset_index()
        cat_data.columns = ["category", "revenue", "orders", "avg_order"]
        cat_data = cat_data.sort_values("revenue", ascending=False)
        result["category_breakdown"] = cat_data.to_dict(orient="records")

    # ── Gender breakdown ───────────────────────────────────────────────────────
    if gen_col and amt_col:
        gen_data = df.groupby(gen_col)[amt_col].agg(
            revenue="sum", orders="count", avg_order="mean"
        ).round(2).reset_index()
        gen_data.columns = ["gender", "revenue", "orders", "avg_order"]
        result["gender_breakdown"] = gen_data.to_dict(orient="records")

    # ── Age distribution ───────────────────────────────────────────────────────
    if age_col and amt_col:
        df["_age_group"] = pd.cut(
            df[age_col],
            bins=[0, 25, 35, 50, 65, 120],
            labels=["18-25", "26-35", "36-50", "51-65", "65+"]
        )
        age_cnt = df.groupby("_age_group", observed=True).agg(
            count=(age_col, "count"),
            revenue=(amt_col, "sum")
        ).reset_index()
        age_cnt.columns = ["age_group", "count", "revenue"]
        age_cnt["age_group"] = age_cnt["age_group"].astype(str)
        result["age_distribution"] = age_cnt.to_dict(orient="records")

    # ── Weekday trend ──────────────────────────────────────────────────────────
    if date_col and amt_col:
        tmp2 = df.dropna(subset=[date_col]).copy()
        tmp2["_weekday"] = tmp2[date_col].dt.day_name()
        wd_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wd = (tmp2.groupby("_weekday")[amt_col]
                  .agg(revenue="sum", orders="count")
                  .reindex(wd_order, fill_value=0)
                  .reset_index())
        wd.columns = ["weekday", "revenue", "orders"]
        result["weekday_trend"] = wd.to_dict(orient="records")

    # ── Quarterly summary ──────────────────────────────────────────────────────
    if date_col and amt_col:
        tmp3 = df.dropna(subset=[date_col]).copy()
        tmp3["_quarter"] = "Q" + tmp3[date_col].dt.quarter.astype(str)
        qtr = tmp3.groupby("_quarter")[amt_col].agg(revenue="sum", orders="count").reset_index()
        qtr.columns = ["quarter", "revenue", "orders"]
        result["quarterly_summary"] = qtr.to_dict(orient="records")

    # ── Category + Gender heatmap ─────────────────────────────────────────────
    if cat_col and gen_col and amt_col:
        heat = df.groupby([cat_col, gen_col])[amt_col].sum().reset_index()
        heat.columns = ["category", "gender", "revenue"]
        result["category_gender_heatmap"] = heat.to_dict(orient="records")

    # ── Model info ────────────────────────────────────────────────────────────
    result["model_info"] = {
        "loaded":             model is not None,
        "r2":                 metrics.get("r2",   "N/A"),
        "mae":                metrics.get("mae",  "N/A"),
        "rmse":               metrics.get("rmse", "N/A"),
        "algorithm":          "Gradient Boosting Regressor",
        "training_rows":      metrics.get("training_rows", "N/A"),
        "test_rows":          metrics.get("test_rows",     "N/A"),
        "target_column":      target_col,
        "feature_columns":    feature_cols,
        "feature_importance": metrics.get("feature_importance", {}),
    }

    result["columns_detected"] = {
        "date": date_col, "amount": amt_col, "category": cat_col,
        "gender": gen_col, "age": age_col, "quantity": qty_col, "price": price_col,
    }

    return JSONResponse(content=result)

# =============================================================================
# ML INFERENCE
# =============================================================================
class PredictRequest(BaseModel):
    product_category: Optional[str]   = None
    gender:           Optional[str]   = None
    age:              Optional[float] = None
    quantity:         Optional[float] = None
    price_per_unit:   Optional[float] = None
    month:            Optional[int]   = None
    day_of_week:      Optional[int]   = None
    quarter:          Optional[int]   = None

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded. Check models/ folder.")

    data = req.model_dump()

    # Auto-derive features
    if data.get("quantity") and data.get("price_per_unit"):
        data["revenue_per_item"] = round(float(data["quantity"]) * float(data["price_per_unit"]), 2)

    if data.get("month"):
        data["month_name"] = calendar.month_abbr[int(data["month"])]

    if data.get("age") and not data.get("age_group"):
        a = float(data["age"])
        if   a <= 25: data["age_group"] = "18-25"
        elif a <= 35: data["age_group"] = "26-35"
        elif a <= 50: data["age_group"] = "36-50"
        elif a <= 65: data["age_group"] = "51-65"
        else:         data["age_group"] = "65+"

    if data.get("day_of_week") is not None:
        data["is_weekend"] = 1 if int(data["day_of_week"]) >= 5 else 0

    if data.get("month") and not data.get("quarter"):
        data["quarter"] = (int(data["month"]) - 1) // 3 + 1

    # Build feature vector in exact trained order
    row      = [encode_feature(col, data.get(col, 0)) for col in feature_cols]
    X        = np.array(row).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred     = float(model.predict(X_scaled)[0])

    mae_val  = float(metrics.get("mae", 0))
    return {
        "predicted_total_amount": round(pred, 2),
        "confidence_low":         round(pred - mae_val, 2),
        "confidence_high":        round(pred + mae_val, 2),
        "model_r2":               metrics.get("r2",   "N/A"),
        "model_mae":              metrics.get("mae",  "N/A"),
        "model_rmse":             metrics.get("rmse", "N/A"),
        "features_used":          {k: data.get(k) for k in feature_cols if data.get(k) is not None},
    }

# =============================================================================
# MODEL INFO & OPTIONS
# =============================================================================
@app.get("/model/info")
def model_info():
    return {
        "loaded":             model is not None,
        "algorithm":          "Gradient Boosting Regressor",
        "target_column":      target_col,
        "feature_columns":    feature_cols,
        "cat_features":       cat_features,
        "num_features":       num_features,
        "metrics": {
            "r2":            metrics.get("r2",            "N/A"),
            "mae":           metrics.get("mae",           "N/A"),
            "rmse":          metrics.get("rmse",          "N/A"),
            "training_rows": metrics.get("training_rows", "N/A"),
            "test_rows":     metrics.get("test_rows",     "N/A"),
        },
        "feature_importance": metrics.get("feature_importance", {}),
    }

@app.get("/model/options")
def model_options():
    options = {}
    if encoders:
        for col, enc in encoders.items():
            options[col] = sorted(list(enc.classes_))
    return options

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_main:app", host="0.0.0.0", port=8000, reload=True)
