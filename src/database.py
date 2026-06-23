import os
import json
import numpy as np
from datetime import datetime

AUDIT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'audit_log.json')

def load_audit_log():
    if os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, 'r') as f: return json.load(f)
    return []

def save_audit_log(log):
    with open(AUDIT_FILE, 'w') as f: json.dump(log, f, indent=2)

def clear_zone_with_geofence(grid_id, location, cop_lat, cop_lon, hotspot_lat, hotspot_lon):
    R = 6371
    dlat = np.radians(hotspot_lat - cop_lat)
    dlon = np.radians(hotspot_lon - cop_lon)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(cop_lat)) * np.cos(np.radians(hotspot_lat)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance_m = R * c * 1000
    
    if distance_m > 100:
        return False, f"❌ Geofence Failed: You are {distance_m:.0f}m away. Must be within 100m."
    
    log = load_audit_log()
    log.append({"grid_id": grid_id, "location": location, "cleared_at": datetime.now().isoformat(), "distance_m": round(distance_m, 2)})
    save_audit_log(log)
    return True, f"✅ Cleared Successfully. Verified at {distance_m:.0f}m distance."

def reset_audit_log():
    save_audit_log([])