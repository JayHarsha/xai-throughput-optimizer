from src.worker.celery_app import celery_app
import joblib
import pandas as pd
import time
import os

# Import your existing, correctly optimized explainer logic
from src.ml.explainer import compute_shap_interactions, get_counterfactuals

# Load ML Artifacts once when the Celery worker boots up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'artifacts', 'realtime_preprocessor.joblib')
MODEL_PATH = os.path.join(BASE_DIR, 'artifacts', 'lightgbm_credit_model.joblib')

print("Worker initializing: Loading ML Artifacts into memory...")
preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)
print("Worker ready to process tasks.")

@celery_app.task(name='process_async_xai')
def process_async_xai(payload: dict, probability: float):
    start_time = time.time()
    df = pd.DataFrame([payload])
    X_processed = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()
    
    complex_interactions = compute_shap_interactions(X_processed, feature_names)
    actionable_recourse = get_counterfactuals(payload, preprocessor, model, probability)
    
    latency_ms = (time.time() - start_time) * 1000

    return {
        "complex_interactions": complex_interactions,
        "actionable_recourse": actionable_recourse,
        "background_processing_time_ms": round(latency_ms, 2)
    }