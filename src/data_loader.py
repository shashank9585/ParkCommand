import streamlit as st
import pandas as pd
import numpy as np

POI_DATABASE = [
    {"name": "MG Road Metro", "lat": 12.9756, "lon": 77.6069, "type": "Metro Station"},
    {"name": "Mantri Square Mall", "lat": 12.9895, "lon": 77.5712, "type": "Shopping Mall"},
    {"name": "Phoenix Marketcity", "lat": 12.9917, "lon": 77.7015, "type": "Shopping Mall"},
    {"name": "Orion Mall", "lat": 12.9911, "lon": 77.5601, "type": "Shopping Mall"},
    {"name": "Yeshwanthpur Metro", "lat": 13.0211, "lon": 77.5400, "type": "Metro Station"},
    {"name": "Jayanagar 4th Block", "lat": 12.9250, "lon": 77.5838, "type": "Commercial Market"},
    {"name": "Koramangala 80ft Road", "lat": 12.9352, "lon": 77.6245, "type": "Commercial Hub"}
]

@st.cache_data(ttl=3600)
def load_data():
    try: df = pd.read_csv('parking_violations.csv')
    except: df = create_sample_data()
    
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    df = df.dropna(subset=['created_datetime']) 
    df['hour'] = df['created_datetime'].dt.hour
    df['date'] = df['created_datetime'].dt.date
    df['day_of_week'] = df['created_datetime'].dt.day_name()
    df['junction_name'] = df['junction_name'].fillna('No Junction')
    df['vehicle_type'] = df['vehicle_type'].fillna('UNKNOWN')
    df['police_station'] = df['police_station'].fillna('Unknown')
    df['violation_type'] = df['violation_type'].fillna('["NO PARKING"]')
    df['context_tags'] = df['location'].apply(lambda x: "Near School/Hospital" if any(w in str(x) for w in ['School', 'Hospital', 'College']) else "")
    df['lat_round'] = df['latitude'].round(4)
    df['lon_round'] = df['longitude'].round(4)
    df['grid_id'] = df['lat_round'].astype(str) + "_" + df['lon_round'].astype(str)
    return df

def create_sample_data():
    np.random.seed(42); n = 500
    return pd.DataFrame({
        'id': [f'FKID{i:06d}' for i in range(n)],
        'latitude': np.random.uniform(12.90, 13.05, n), 'longitude': np.random.uniform(77.50, 77.70, n),
        'location': [f'Location {i} near School, Bengaluru' if i%5==0 else f'Location {i}, Bengaluru' for i in range(n)],
        'vehicle_type': np.random.choice(['CAR', 'SCOOTER', 'TANKER', 'AUTO'], n),
        'violation_type': np.random.choice(['["WRONG PARKING"]', '["NO PARKING"]'], n),
        'created_datetime': pd.date_range(start='2023-11-01', periods=n, freq='H'),
        'police_station': np.random.choice(['Madiwala', 'Bellandur', 'Byatarayanapura'], n),
        'junction_name': np.random.choice(['No Junction', 'BTP044 - Sagar Theatre'], n),
    })

@st.cache_data(ttl=3600)
def calculate_severity_score(df):
    location_stats = df.groupby(['grid_id', 'lat_round', 'lon_round', 'police_station']).agg({
        'id': 'count', 'junction_name': lambda x: (x != 'No Junction').sum(), 'hour': 'mean',
        'vehicle_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'UNKNOWN',
        'location': 'first', 'latitude': 'mean', 'longitude': 'mean', 'context_tags': 'first'
    }).reset_index()
    location_stats.columns = ['grid_id', 'lat_round', 'lon_round', 'police_station', 'violation_count', 'junction_violations', 'avg_hour', 'dominant_vehicle', 'location', 'latitude', 'longitude', 'context_tags']
    
    def get_poi_context(lat, lon):
        for poi in POI_DATABASE:
            dist = ((lat - poi['lat'])**2 + (lon - poi['lon'])**2)**0.5 * 111
            if dist < 1.5: return f"Near {poi['type']} ({poi['name']})"
        return ""
    location_stats['poi_context'] = location_stats.apply(lambda x: get_poi_context(x['latitude'], x['longitude']), axis=1)
    
    max_v = location_stats['violation_count'].max() or 1
    max_j = location_stats['junction_violations'].max() or 1
    location_stats['severity_score'] = (
        (location_stats['violation_count'] / max_v) * 40 + (location_stats['junction_violations'] / max_j) * 30 +
        (1 - abs(location_stats['avg_hour'] - 18) / 12) * 20 + (location_stats['poi_context'] != "").astype(int) * 10
    )
    location_stats['severity_level'] = pd.cut(location_stats['severity_score'], bins=[-1, 25, 50, 75, 100], labels=['Low', 'Medium', 'High', 'Critical'])
    return location_stats.sort_values('severity_score', ascending=False)