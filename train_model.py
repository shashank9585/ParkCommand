# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

def train_and_save_model():
    print("🚀 Loading data and training ML model...")
    try:
        df = pd.read_csv('parking_violations.csv')
    except:
        print(" parking_violations.csv not found. Please place it in the root directory.")
        return

    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    df = df.dropna(subset=['created_datetime'])
    
    df['hour'] = df['created_datetime'].dt.hour
    df['day_of_week'] = df['created_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if 8 <= x <= 10 or 17 <= x <= 20 else 0)
    df['is_junction'] = df['junction_name'].apply(lambda x: 0 if x == 'No Junction' else 1)
    df['is_sensitive'] = df['location'].str.contains('School|Hospital|Metro|College', case=False, na=False).astype(int)
    
    df['target_severity'] = (df['is_peak_hour'] * 40 + df['is_junction'] * 30 + df['is_sensitive'] * 20 + np.random.randint(0, 10, len(df)))

    features = ['hour', 'day_of_week', 'is_weekend', 'is_peak_hour', 'is_junction', 'is_sensitive']
    X = df[features]
    y = df['target_severity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print(f"✅ Model trained! Test R² Score: {model.score(X_test, y_test):.2f}")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/severity_model.pkl')
    print("💾 Model saved to models/severity_model.pkl")

if __name__ == "__main__":
    train_and_save_model()