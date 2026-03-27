"""
ml_engine.py
NagarMitra Phase 2 — ML Inference Engine
Loads trained models and provides prediction functions.
"""

import os
import joblib
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

# Lazy loaded artifacts
_xgb_forecaster = None
_forecaster_scaler = None
_forecaster_features = None

_source_classifier = None
_source_encoder = None
_classifier_features = None

def get_forecaster():
    global _xgb_forecaster, _forecaster_scaler, _forecaster_features
    if _xgb_forecaster is None:
        try:
            _xgb_forecaster = joblib.load(os.path.join(MODEL_DIR, "xgb_forecaster.pkl"))
            _forecaster_scaler = joblib.load(os.path.join(MODEL_DIR, "aqi_scaler.pkl"))
            _forecaster_features = joblib.load(os.path.join(MODEL_DIR, "forecaster_features.pkl"))
        except Exception as e:
            print(f"Error loading forecaster models: {e}")
    return _xgb_forecaster, _forecaster_scaler, _forecaster_features

def get_classifier():
    global _source_classifier, _source_encoder, _classifier_features
    if _source_classifier is None:
        try:
            _source_classifier = joblib.load(os.path.join(MODEL_DIR, "source_classifier.pkl"))
            _source_encoder = joblib.load(os.path.join(MODEL_DIR, "source_encoder.pkl"))
            _classifier_features = joblib.load(os.path.join(MODEL_DIR, "classifier_features.pkl"))
        except Exception as e:
            print(f"Error loading classifier models: {e}")
    return _source_classifier, _source_encoder, _classifier_features

def predict_forecast_72h(input_df: pd.DataFrame) -> list:
    """
    Predicts the next 72 hours of AQI given exactly 72 hours of historical features.
    """
    model, scaler, features = get_forecaster()
    if not model:
        raise ValueError("Forecaster models not loaded")
        
    df = input_df[features].copy()
    
    # Scale input
    scaled_input = scaler.transform(df)
    
    # Flatten the 72h window for XGBoost
    X_pred = np.array([scaled_input.flatten()])
    
    # Get predictions (shape: [1, 72])
    y_pred_scaled = model.predict(X_pred)
    
    target_idx = features.index('aqi')
    dummy = np.zeros((72, len(features)))
    dummy[:, target_idx] = y_pred_scaled.flatten()
    
    aqi_predictions = scaler.inverse_transform(dummy)[:, target_idx]
    return np.round(np.clip(aqi_predictions, 0, 500), 1).tolist()

def predict_source(features_dict: dict) -> str:
    """
    Classifies the main pollution source based on current metrics.
    """
    model, encoder, features = get_classifier()
    if not model:
        raise ValueError("Classifier models not loaded")
        
    # Ensure features are in the exact order required by the model
    input_array = np.array([[features_dict.get(f, 0.0) for f in features]])
    
    pred_idx = model.predict(input_array)[0]
    return encoder.inverse_transform([pred_idx])[0]
