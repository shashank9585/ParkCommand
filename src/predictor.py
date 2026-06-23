import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'severity_model.pkl')

def load_model():
    if os.path.exists(MODEL_PATH): return joblib.load(MODEL_PATH)
    return None

def predict_severity(hour, day_of_week, is_junction, is_sensitive):
    model = load_model()
    if not model:
        score = 40
        if 8 <= hour <= 10 or 17 <= hour <= 20: score += 25
        if is_junction: score += 15
        if is_sensitive: score += 20
        return min(score, 100)

    is_weekend = 1 if day_of_week >= 5 else 0
    is_peak_hour = 1 if 8 <= hour <= 10 or 17 <= hour <= 20 else 0
    
    features = pd.DataFrame([{
        'hour': hour, 'day_of_week': day_of_week, 'is_weekend': is_weekend,
        'is_peak_hour': is_peak_hour, 'is_junction': is_junction, 'is_sensitive': is_sensitive
    }])
    return round(float(model.predict(features)[0]), 2)