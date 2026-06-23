import json
import urllib.request
import numpy as np

def get_real_driving_distance(lat1, lon1, lat2, lon2):
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=5).read()
        data = json.loads(response)
        if data['code'] == 'Ok': return data['routes'][0]['distance'] / 1000 
    except: pass
    return ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5 * 111

def generate_real_patrol_route(location_stats):
    if len(location_stats) < 2: return [], 0
    top_zones = location_stats.head(5)[['latitude', 'longitude']].copy()
    top_zones['visited'] = False
    current_idx = 0; route = [0]; top_zones.iloc[0, 2] = True; total_dist = 0
    for _ in range(len(top_zones) - 1):
        curr_lat, curr_lon = top_zones.iloc[current_idx][['latitude', 'longitude']]
        min_dist, next_idx = float('inf'), -1
        for i in range(len(top_zones)):
            if not top_zones.iloc[i]['visited']:
                lat, lon = top_zones.iloc[i][['latitude', 'longitude']]
                dist = get_real_driving_distance(curr_lat, curr_lon, lat, lon)
                if dist < min_dist: min_dist, next_idx = dist, i
        top_zones.iloc[next_idx, 2] = True; route.append(next_idx); total_dist += min_dist; current_idx = next_idx
    route_coords = top_zones.iloc[route][['latitude', 'longitude']].values.tolist()
    route_coords.append(route_coords[0])
    return route_coords, round(total_dist, 2)