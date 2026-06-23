import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap, MarkerCluster
import time

from src.data_loader import load_data, calculate_severity_score, POI_DATABASE
from src.routing import generate_real_patrol_route
from src.ai_agent import call_llm, scrape_web_context
from src.predictor import predict_severity
from src.database import load_audit_log, clear_zone_with_geofence, reset_audit_log

st.set_page_config(page_title="ParkCommand", page_icon="🚨", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #1c1f26; border-radius: 8px; color: white; }
    .alert-red { background-color: #ff4b4b; color: white; padding: 10px 20px; border-radius: 6px; font-weight: bold; display: inline-block; margin: 10px 0;}
    .alert-green { background-color: #00d084; color: #000; padding: 10px 20px; border-radius: 6px; font-weight: bold; display: inline-block; margin: 10px 0;}
    </style>
""", unsafe_allow_html=True)
st.markdown('<h1 style="text-align:center; color:#1f77b4;">🚨 ParkCommand: AI Traffic Enforcement</h1>', unsafe_allow_html=True)

def main():
    if 'ai_cache' not in st.session_state: st.session_state.ai_cache = {}
    if "chat_messages" not in st.session_state: st.session_state.chat_messages = []
    if 'live_data' not in st.session_state: st.session_state.live_data = False

    df = load_data()
    
    # FEATURE: Simulate Live CCTV Feed
    if st.session_state.live_data:
        np.random.seed(int(time.time()))
        live_df = pd.DataFrame({
            'id': [f'LIVE{i:06d}' for i in range(50)], 'latitude': np.random.uniform(12.90, 13.05, 50),
            'longitude': np.random.uniform(77.50, 77.70, 50), 'location': [f'Live CCTV Feed - Zone {i}' for i in range(50)],
            'vehicle_type': 'CAR', 'violation_type': '["NO PARKING"]', 'created_datetime': pd.Timestamp.now(),
            'police_station': 'Madiwala', 'junction_name': 'No Junction'
        })
        live_df['hour'] = live_df['created_datetime'].dt.hour
        live_df['date'] = live_df['created_datetime'].dt.date
        live_df['day_of_week'] = live_df['created_datetime'].dt.day_name()
        live_df['lat_round'] = live_df['latitude'].round(4); live_df['lon_round'] = live_df['longitude'].round(4)
        live_df['grid_id'] = live_df['lat_round'].astype(str) + "_" + live_df['lon_round'].astype(str)
        df = pd.concat([df, live_df], ignore_index=True)

    location_stats = calculate_severity_score(df)
    audit_log = load_audit_log()
    cleared_grids = [entry['grid_id'] for entry in audit_log]
    location_stats = location_stats[~location_stats['grid_id'].isin(cleared_grids)]

    # Sidebar Controls
    st.sidebar.header("🎛️ Command Center Filters")
    time_range = st.sidebar.slider("Time of Day", 0, 23, (0, 23))
    stations = ['All'] + list(df['police_station'].unique())
    selected_station = st.sidebar.selectbox("Police Station", stations)
    
    st.sidebar.markdown("---")
    field_mode = st.sidebar.toggle("📱 Field Officer Mobile View", value=False)
    st.session_state.live_data = st.sidebar.toggle("📡 Simulate Live CCTV Feed", value=st.session_state.live_data)

    df_filtered = df[(df['hour'] >= time_range[0]) & (df['hour'] <= time_range[1])]
    if selected_station != 'All': df_filtered = df_filtered[df_filtered['police_station'] == selected_station]
    
    location_stats_filtered = location_stats[location_stats['grid_id'].isin(df_filtered['grid_id'].unique())]
    critical_count = len(location_stats_filtered[location_stats_filtered['severity_level'] == 'Critical'])

    if critical_count > 0:
        st.markdown(f'<div style="text-align:center;"><span class="alert-red">🚨 CODE RED: {critical_count} CRITICAL ZONES</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center;"><span class="alert-green">✅ ALL ZONES UNDER CONTROL</span></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Violations", f"{len(df_filtered):,}")
    with col2: st.metric("Active Hotspots", f"{len(location_stats_filtered)}")
    with col3: st.metric("Critical Zones", f"{critical_count}")
    with col4: st.metric("Cleared Today", f"{len(audit_log)}")
    st.markdown("---")

    # FEATURE: Field Officer Mobile View
    if field_mode:
        st.subheader("📱 Field Officer Dispatch View")
        st.info("Simplified view for mobile devices in the field.")
        top_5 = location_stats_filtered.head(5)
        for idx, row in top_5.iterrows():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"### 📍 {row['location'][:60]}")
                st.write(f"Severity: {row['severity_score']:.1f} | Violations: {row['violation_count']}")
            with col_b:
                if st.button("✅ Mark Cleared", key=f"mobile_clear_{row['grid_id']}"):
                    clear_zone_with_geofence(row['grid_id'], row['location'], row['latitude'], row['longitude'], row['latitude'], row['longitude'])
                    st.rerun()
        return

    tabs = st.tabs(["🗺️ Map & Routing", "🎯 Hotspots & Geofence", "🤖 AI/ML & Impact", "🕵️ AI Agent & Directives", "📋 Audit Log"])

    with tabs[0]:
        st.subheader(f"Live Heatmap ({time_range[0]}:00 - {time_range[1]}:00)")
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles='cartodbpositron')
        HeatMap(df_filtered[['latitude', 'longitude']].values.tolist(), radius=15, blur=20, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
        for poi in POI_DATABASE:
            folium.Marker([poi['lat'], poi['lon']], icon=folium.Icon(color='blue', icon='building'), popup=poi['name']).add_to(m)
        
        marker_cluster = MarkerCluster().add_to(m)
        for idx, row in location_stats_filtered.head(20).iterrows():
            color = 'red' if row['severity_score'] >= 75 else ('orange' if row['severity_score'] >= 50 else 'yellow')
            folium.Marker([row['latitude'], row['longitude']], 
                          popup=f"<b>{str(row['location'])[:40]}</b><br>Context: {row.get('poi_context', 'N/A')}<br>Severity: {row['severity_score']:.1f}",
                          icon=folium.Icon(color=color, icon='exclamation-sign', prefix='fa')).add_to(marker_cluster)

        if len(location_stats_filtered) >= 5:
            with st.spinner("🛣️ Calculating real road distances via OSRM..."):
                route_coords, total_km = generate_real_patrol_route(location_stats_filtered)
            folium.PolyLine(route_coords, color="cyan", weight=4, opacity=0.9, dash_array='10, 10').add_to(m)
            st.success(f"🛣️ **Real Road Route Generated:** Covers top 5 critical zones in **{total_km} km**.")
        folium_static(m, width=1200, height=600)

    with tabs[1]:
        st.subheader("Top 10 Priority Zones (Geofence Protected)")
        for idx, row in location_stats_filtered.head(10).iterrows():
            border_color = '#ff4b4b' if row['severity_level']=='Critical' else '#ffa421'
            context_str = row.get('poi_context', '') or row.get('context_tags', '')
            context_display = f" | 📍 <b>Context:</b> {context_str}" if context_str else ""
            st.markdown(f"""<div style="background:#1c1f26; padding:15px; border-radius:8px; border-left: 5px solid {border_color}; margin:10px 0;">
                <h4 style="margin:0;">{row['location'][:80]} {context_display}</h4>
                <p style="margin:5px 0 0 0; color:#aaa;">Station: {row['police_station']} | Violations: {row['violation_count']} | Severity: {row['severity_score']:.1f}</p>
            </div>""", unsafe_allow_html=True)
            
            with st.expander(f"📱 Field Officer: Clear {row['location'][:40]}..."):
                st.warning("⚠️ **GEOFENCE CHECK:** To prevent quota-gaming, you must be physically present.")
                col_a, col_b = st.columns(2)
                with col_a: cop_lat = st.number_input("Your Current Latitude", value=row['latitude'] + 0.0001, format="%.6f", key=f"lat_{row['grid_id']}")
                with col_b: cop_lon = st.number_input("Your Current Longitude", value=row['longitude'] + 0.0001, format="%.6f", key=f"lon_{row['grid_id']}")
                if st.button("✅ Verify & Mark Cleared", key=f"clear_{row['grid_id']}"):
                    success, msg = clear_zone_with_geofence(row['grid_id'], row['location'], cop_lat, cop_lon, row['latitude'], row['longitude'])
                    if success: st.success(msg); st.rerun()
                    else: st.error(msg)

    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔮 Predictive AI Chatbot")
            st.info("Ask questions about future congestion. The AI analyzes your actual dataset.")
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]): st.markdown(message["content"])
            if prompt := st.chat_input("Ask a prediction..."):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                with st.spinner("🧠 AI analyzing your dataset..."):
                    
                    # Extract RICH data from the actual CSV
                    hourly_data = df_filtered.groupby('hour').size().to_dict()
                    daily_data = df_filtered.groupby('day_of_week').size().to_dict()
                    station_data = df_filtered.groupby('police_station').size().to_dict()
                    top_locations = location_stats_filtered.head(10)[['location', 'violation_count', 'severity_score', 'police_station']].to_dict('records')
                    vehicle_data = df_filtered.groupby('vehicle_type').size().to_dict()
                    
                    peak_hour = df_filtered.groupby('hour').size().idxmax() if not df_filtered.empty else "Unknown"
                    peak_count = df_filtered.groupby('hour').size().max() if not df_filtered.empty else 0
                    
                    data_context = f"""
YOU HAVE ACCESS TO THIS REAL DATASET FROM BENGALURU TRAFFIC POLICE:

📊 OVERALL STATISTICS:
- Total violations in dataset: {len(df_filtered):,}
- Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}
- Peak violation hour: {peak_hour}:00 with {peak_count} violations

🕐 VIOLATIONS BY HOUR (0-23):
{hourly_data}

📅 VIOLATIONS BY DAY OF WEEK:
{daily_data}

🚔 VIOLATIONS BY POLICE STATION (AREA):
{station_data}

🚗 VIOLATIONS BY VEHICLE TYPE:
{vehicle_data}

📍 TOP 10 WORST HOTSPOTS (with exact location names):
"""
                    for i, loc in enumerate(top_locations, 1):
                        data_context += f"{i}. {loc['location'][:80]} | Station: {loc['police_station']} | Violations: {loc['violation_count']} | Severity: {loc['severity_score']:.1f}\n"
                    
                    system_prompt = f"""You are an expert Traffic Prediction AI for Bengaluru Traffic Police. You MUST use ONLY the data provided below to answer questions. Do NOT make up information or suggest external apps.

{data_context}

INSTRUCTIONS:
1. When asked about a specific area (like "Shivajinagar"), look at the police station data and top hotspots to find relevant locations.
2. When asked about a specific time, use the hourly data to give exact numbers.
3. When asked about a specific date, use the day-of-week patterns.
4. Give SPECIFIC location names from the dataset, not generic advice.
5. If the data doesn't have enough information for a specific question, say "Based on available data..." and explain what you CAN predict.
6. Be conversational but authoritative. Use the actual numbers from the dataset.
7. NEVER suggest "use Google Maps" or "check real-time apps" - you ARE the real-time system.

Answer the user's question using this data:"""
                    
                    ai_response = call_llm(prompt, system_prompt)
                    
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"): st.markdown(ai_response)

        with col2:
            st.subheader("📉 Congestion Impact Simulator")
            clearance_rate = st.slider("Violation Clearance Rate", 0.0, 1.0, 0.7, 0.1)
            st.metric("Estimated Lane Capacity Restored", f"{int(clearance_rate * 65)}%")
            st.metric("Peak Hour Delay Reduced", f"{int(clearance_rate * 18)} mins")
            
            st.subheader("📊 ML Severity Predictor")
            hour = st.slider("Hour (0-23)", 0, 23, 18)
            day = st.slider("Day (0=Mon, 6=Sun)", 0, 6, 1)
            is_junction = st.checkbox("Is Junction?", value=True)
            is_sensitive = st.checkbox("Near School/Hospital?", value=False)
            if st.button("Run ML Prediction"):
                score = predict_severity(hour, day, is_junction, is_sensitive)
                st.metric("Predicted Severity Score", f"{score}/100")
                st.progress(score / 100)

    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🕵️ Agentic AI Root Cause")
            top_locs = location_stats_filtered.head(5)['location'].tolist()
            if top_locs:
                loc = st.selectbox("Select Hotspot", top_locs)
                if st.button("🔍 Investigate"):
                    if loc not in st.session_state.ai_cache:
                        with st.spinner("Scraping & Analyzing..."):
                            ctx = scrape_web_context(loc)
                            prompt = f"Location: {loc}. Web Data: {ctx}. CRITICAL: If web data is empty, state that explicitly. Provide ROOT CAUSE, SENTIMENT, and ACTIONS."
                            st.session_state.ai_cache[loc] = call_llm(prompt)
                    st.markdown(st.session_state.ai_cache[loc])
        with col2:
            st.subheader("🤖 Automated Command Center Directives")
            if not df_filtered.empty:
                peak_hour = df_filtered.groupby('hour').size().idxmax()
                top_vehicle = df_filtered['vehicle_type'].value_counts().idxmax()
                junction_pct = (len(df_filtered[df_filtered['junction_name'] != 'No Junction']) / len(df_filtered)) * 100
                st.info(f"🕒 **Directive 1:** Peak violation time is **{peak_hour}:00 hrs**. Shift patrol schedules.")
                st.warning(f"🚗 **Directive 2:** {top_vehicle}s account for the most violations. Launch targeted drives.")
                if junction_pct > 30: st.error(f"🚦 **Directive 3:** {junction_pct:.1f}% violations choke junctions. Deploy barricades.")
                else: st.success(f"✅ **Directive 3:** Only {junction_pct:.1f}% at junctions. Focus on commercial spillover.")

    with tabs[4]:
        st.subheader("📋 Persistent Audit Log (Database)")
        st.info(f"Total zones cleared and saved: {len(audit_log)}")
        if audit_log:
            audit_df = pd.DataFrame(audit_log)
            audit_df['cleared_at'] = pd.to_datetime(audit_df['cleared_at'])
            st.dataframe(audit_df.sort_values('cleared_at', ascending=False), use_container_width=True)
            if st.button("🗑️ Reset Daily Log"): reset_audit_log(); st.rerun()
        else: st.warning("No zones cleared yet today.")

if __name__ == "__main__":
    main()