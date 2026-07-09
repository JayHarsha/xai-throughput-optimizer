from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from collections import deque
import time
import pandas as pd
import joblib
import os
import numpy as np

from src.worker.celery_app import celery_app
from src.worker.tasks import process_async_xai
from src.ml.explainer import compute_main_effects, compute_shap_interactions, get_counterfactuals

app = FastAPI(title="Hybrid Credit Risk Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

request_log = {
    "synch": deque(maxlen=200),
    "asynch": deque(maxlen=200),
    "baseline": deque(maxlen=200)
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
preprocessor = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'realtime_preprocessor.joblib'))
model = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'lightgbm_credit_model.joblib'))

@app.post("/predict/synch")
async def predict_risk_sync(payload: dict):
    """
    BASELINE: Fully synchronous endpoint - everything blocks.
    Used for dissertation comparison against hybrid async pipeline.
    """
    start_time = time.time()

    # Same preprocessing as hybrid
    df = pd.DataFrame([payload])
    X_processed = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()

    # Tier 1: Predict + Basic SHAP (same as hybrid)
    probability = model.predict_proba(X_processed)[0][1]
    base_explanation = compute_main_effects(X_processed, feature_names)
    tier = "Low" if probability < 0.40 else "Medium" if probability <= 0.60 else "High"

    # Tier 2: Complex SHAP + Counterfactuals — BUT BLOCKING (no Celery, no Redis)
    complex_interactions = None
    actionable_recourse = None

    if tier in ["Medium", "High"]:
        complex_interactions = compute_shap_interactions(X_processed, feature_names)
        actionable_recourse = get_counterfactuals(payload, preprocessor, model, probability)

    latency_ms = (time.time() - start_time) * 1000
    request_log["synch"].append(latency_ms)

    return {
        "default_probability": round(float(probability), 4),
        "risk_tier": tier,
        "immediate_primary_factors": base_explanation,
        "complex_interactions": complex_interactions,
        "actionable_recourse": actionable_recourse,
        "api_latency_ms": round(latency_ms, 2),
        "mode": "synchronous_xai"
    }

@app.post("/predict/asynch")
async def predict_risk_asynch(payload: dict):
    start_time = time.time()

    # 1. TIER 1: Synchronous Base Prediction
    df = pd.DataFrame([payload])
    X_processed = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()

    probability = model.predict_proba(X_processed)[0][1]
    base_explanation = compute_main_effects(X_processed, feature_names)

    tier = "Low" if probability < 0.40 else "Medium" if probability <= 0.60 else "High"

    # 2. TIER 2: Asynchronous Deep Analysis
    task = None
    if tier in ["Medium", "High"]:
        task = process_async_xai.delay(payload, probability)

    latency_ms = (time.time() - start_time) * 1000
    request_log["asynch"].append(latency_ms)

    return {
        "default_probability": round(float(probability), 4),
        "risk_tier": tier,
        "immediate_primary_factors": base_explanation,
        "api_latency_ms": round(latency_ms, 2),
        "deep_analysis": {
            "status": "Processing in background..." if task else "Not required for Low Risk",
            "task_id": task.id if task else None,
            "status_url": f"/result/{task.id}" if task else None
        }
    }

@app.post("/predict/baseline")
async def predict_risk_baseline(payload: dict):
    """
    PURE INFERENCE: No XAI computation. Isolates ML prediction overhead
    from SHAP/counterfactual costs. Used as control group in dissertation benchmarks.
    """
    start_time = time.time()
    df = pd.DataFrame([payload])
    X_processed = preprocessor.transform(df)
    probability = model.predict_proba(X_processed)[0][1]
    tier = "Low" if probability < 0.40 else "Medium" if probability <= 0.60 else "High"
    latency_ms = (time.time() - start_time) * 1000
    request_log["baseline"].append(latency_ms)
    return {
        "default_probability": round(float(probability), 4),
        "risk_tier": tier,
        "api_latency_ms": round(latency_ms, 2),
        "mode": "baseline_no_xai"
    }

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    # 3. Check Redis for the task status
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.state == 'PENDING':
        return {"task_id": task_id, "status": "Processing..."}
    elif task_result.state == 'SUCCESS':
        return {"task_id": task_id, "status": "Completed", "data": task_result.result}
    elif task_result.state == 'FAILURE':
        return {"task_id": task_id, "status": "Failed", "error": str(task_result.info)}
    else:
        return {"task_id": task_id, "status": task_result.state}

@app.get("/metrics/live")
async def live_metrics():
    def calc(times):
        if not times:
            return {"p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "request_count": 0}
        arr = list(times)
        return {
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "request_count": len(arr)
        }
    return {
        "synch": calc(request_log["synch"]),
        "asynch": calc(request_log["asynch"]),
        "baseline": calc(request_log["baseline"])
    }

@app.get("/")
async def serve_dashboard():
    return FileResponse("/app/dashboard.html")
