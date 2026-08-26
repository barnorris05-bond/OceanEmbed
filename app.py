import datetime
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="OceanEmbed Prototype", layout="wide")

# Custom Styles
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; }
    .sub-title { font-size: 1.0rem; color: #475569; margin-bottom: 20px; }
    .badge-green { background-color: #DCFCE7; color: #15803D; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
    .badge-yellow { background-color: #FEF9C3; color: #A16207; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
    .badge-red { background-color: #FEE2E2; color: #B91C1C; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌊 OCEANEMBED AI</div>', unsafe_allow_html=True)
st.markdown('🟢 **Model Loaded:** LightGBM Multi-Output Regression (`model.pkl`) | **Target Scope:** Arabian Sea (8°N–24°N, 60°E–77°E)', unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('test_sample.csv')
    except FileNotFoundError:
        return pd.read_csv('ocean_data.csv')

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    df = load_dataset()
    artifact = load_model()
    model = artifact['model']
    real_metrics_df = artifact['metrics']
except Exception as e:
    st.error("Missing dataset or trained model artifact. Please run `py train_model.py` first!")
    st.stop()

# Sidebar: Select Observation Point
st.sidebar.header("📍 Select Observation Point")
unique_locs = df[['lat', 'lon']].drop_duplicates().head(30)
options = [f"Lat: {row['lat']}°N, Lon: {row['lon']}°E" for _, row in unique_locs.iterrows()]

selected_str = st.sidebar.selectbox("Arabian Sea Coordinates:", options)
selected_idx = options.index(selected_str)
matched_row = unique_locs.iloc[selected_idx]

row_data = df[(df['lat'] == matched_row['lat']) & (df['lon'] == matched_row['lon'])].iloc[0]

lat = float(row_data['lat'])
lon = float(row_data['lon'])
day_of_year = int(row_data['day_of_year'])
sst = float(row_data['sst'])
ssh = float(row_data['ssh'])
sss = float(row_data['sss'])

# Calculate readable calendar date from Day of Year
formatted_date = (datetime.datetime(2026, 1, 1) + datetime.timedelta(days=day_of_year - 1)).strftime("%d %b %Y")

# Actual ground truth profile from Argo dataset
actual_profile = [sst, float(row_data['temp_50m']), float(row_data['temp_100m']), float(row_data['temp_200m']), float(row_data['temp_500m'])]

# Model Inference via model.pkl
input_features = pd.DataFrame([[lat, lon, day_of_year, sst, ssh, sss]], columns=['lat', 'lon', 'day_of_year', 'sst', 'ssh', 'sss'])
predicted_depths = model.predict(input_features)[0]
predicted_profile = [sst, round(float(predicted_depths[0]), 2), round(float(predicted_depths[1]), 2), round(float(predicted_depths[2]), 2), round(float(predicted_depths[3]), 2)]

# Calculate absolute errors
errors = [abs(p - a) for p, a in zip(predicted_profile[1:], actual_profile[1:])]

# Helper for color-coded error status
def get_error_badge(err):
    if err <= 0.25:
        return f'<span class="badge-green">🟢 Error: {err:.2f} °C</span>'
    elif err <= 0.50:
        return f'<span class="badge-yellow">🟡 Error: {err:.2f} °C</span>'
    else:
        return f'<span class="badge-red">🔴 Error: {err:.2f} °C</span>'

# --- 1. Surface Ocean Conditions ---
st.subheader("🛰️ Surface Ocean Conditions (Model Inputs)")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Latitude", f"{lat:.2f} °N")
c2.metric("Longitude", f"{lon:.2f} °E")
c3.metric("Observation Date", formatted_date, help=f"Day of Year: {day_of_year}")
c4.metric("SST (Surface Temp)", f"{sst:.2f} °C")
c5.metric("SSH (Surface Height)", f"{ssh:.3f} m")
c6.metric("SSS (Salinity)", f"{sss:.2f} PSU")

st.markdown("---")

# --- 2. AI Subsurface Predictions ---
st.subheader("🤖 AI Subsurface Predictions (`model.pkl` Inference)")
p1, p2, p3, p4 = st.columns(4)

with p1:
    st.metric("Predicted 50m", f"{predicted_profile[1]:.2f} °C")
    st.markdown(get_error_badge(errors[0]), unsafe_allow_html=True)

with p2:
    st.metric("Predicted 100m", f"{predicted_profile[2]:.2f} °C")
    st.markdown(get_error_badge(errors[1]), unsafe_allow_html=True)

with p3:
    st.metric("Predicted 200m", f"{predicted_profile[3]:.2f} °C")
    st.markdown(get_error_badge(errors[2]), unsafe_allow_html=True)

with p4:
    st.metric("Predicted 500m", f"{predicted_profile[4]:.2f} °C")
    st.markdown(get_error_badge(errors[3]), unsafe_allow_html=True)

st.markdown("---")

# --- 3. Profile Plot & Performance Metrics ---
col_graph, col_stats = st.columns([2, 1])

with col_graph:
    st.subheader("📊 Subsurface Temperature Profile")
    fig, ax = plt.subplots(figsize=(7, 4))
    depth_levels = [0, 50, 100, 200, 500]

    ax.plot(predicted_profile, depth_levels, 'o-', color='#0066CC', label='OceanEmbed AI Prediction', linewidth=2.5)
    ax.plot(actual_profile, depth_levels, 's--', color='#FF5500', label='Argo Actual Observation', linewidth=2.0)

    # Invert y-axis directly on the Axes object
    ax.invert_yaxis()
    
    ax.set_xlabel("Temperature (°C)", fontsize=11)
    ax.set_ylabel("Depth (m)", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left')
    st.pyplot(fig)

with col_stats:
    st.subheader("🎯 Model Performance Metrics")
    st.caption("Evaluated across independent Argo test split:")
    st.table(real_metrics_df)