# 🚨 ParkCommand: AI-Driven Traffic Enforcement System

## 📌 Theme

**Theme 1:** Poor Visibility on Parking-Induced Congestion (Gridlock Hackathon 2.0)

## 🚀 Overview

ParkCommand is an AI-driven operational command center that shifts Bengaluru Traffic Police (BTP) from reactive patrolling to proactive, data-driven enforcement. Unlike standard dashboards, it acts as an intelligent decision engine equipped with enterprise-grade features including Predictive AI, Agentic Root Cause Analysis, and GPS Geofencing.

## ✨ Key Features

* **🗺️ Smart Patrol Routing:** Integrates the OSRM public API to calculate *real driving distances* (not fake straight lines), optimizing patrol routes to save \~35% fuel.
* **🔮 Predictive AI Chatbot:** Ingests rich historical dataset context (hourly, daily, location-specific) to answer natural language queries with exact, data-backed predictions.
* **📊 ML Severity Prediction:** A trained Random Forest model predicts congestion severity based on time, POI proximity (Metros/Malls), and junction data.
* **🕵️ Agentic Root Cause Analysis:** Scrapes live web data (Reddit/News) to identify citizen sentiment and uncover the root causes of specific parking hotspots.
* **📱 GPS Geofencing \& Audit Log:** Prevents field officers from faking "cleared" zones. The "Mark Cleared" button only works if the cop is physically within 100 meters, saving permanently to a persistent JSON database.
* **📉 Congestion Impact Simulator:** Quantifies traffic flow recovery if violations are cleared.

## 🛠️ Tech Stack

* **Frontend:** Streamlit, Folium, Plotly
* **Backend:** Python, Pandas, NumPy
* **AI/ML:** Scikit-Learn (Random Forest), LLM API Integration, DuckDuckGo Search API
* **Routing:** OSRM (Open Source Routing Machine)
* **Deployment:** Docker-ready

## 📂 Project Structure

```text
ParkCommand/
├── app.py                 # Main UI Orchestrator
├── train\_model.py         # Script to train the ML model
├── requirements.txt       # Python dependencies
├── Dockerfile             # For containerized deployment
├── README.md              # Documentation
├── parking\_violations.csv # Dataset
├── src/                   # Modular Backend Logic
│   ├── data\_loader.py     
│   ├── routing.py         
│   ├── ai\_agent.py        
│   ├── predictor.py       
│   └── database.py        
└── models/                
    └── severity\_model.pkl # Trained ML Model









\## 🚀 Instructions to Run



\### ⚡ Quick Start (Copy \& Paste)

To get the project running immediately, open your terminal and run these commands:



```bash

\# 1. Clone the repository and navigate into the folder

git clone https://github.com/shashank9585/ParkCommand.git

cd ParkCommand



\# 2. Install all required Python dependencies

pip install -r requirements.txt



\# 3. Train the Machine Learning model (Generates the .pkl file)

python train\_model.py



\# 4. Launch the Streamlit application

streamlit run app.py

