import datetime
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# Page Config & Custom Styling
st.set_page_config(page_title="OceanEmbed AI", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; }
    .badge-green { background-color: #DCFCE7; color: #15803D; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
    .badge-yellow { background-color: #FEF9C3; color: #A16207; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
    .badge-red { background-color: #FEE2E2; color: #B91C1C; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cloud API: NVIDIA Nemotron 3 Ultra Integration
# ---------------------------------------------------------
# Safely fetch key from Streamlit secrets, then system environment variables
NEMOTRON_API_KEY = st.secrets.get("NEMOTRON_API_KEY", os.getenv("NEMOTRON_API_KEY"))

# Instantiate OpenAI client globally if key is available
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NEMOTRON_API_KEY
) if NEMOTRON_API_KEY else None


def query_nemotron(prompt):
    """Fast, low-latency API call to Nemotron-3-Ultra."""
    if not NEMOTRON_API_KEY or not client:
        return "⚠️ Please set a valid NVIDIA API key in .streamlit/secrets.toml to enable AI reasoning."

    try:
        completion = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[
                {"role": "system", "content": "You are OceanEmbed AI's reasoning engine. Keep answers under 3 concise sentences or short bullet points."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            top_p=0.95,
            max_tokens=256,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Nemotron API Error: {str(e)}"

# ---------------------------------------------------------
# Load Dataset & Pre-trained Artifacts
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Header & Input Controls
# ---------------------------------------------------------
st.markdown('<div class="main-title">🌊 OCEANEMBED AI</div>', unsafe_allow_html=True)
st.markdown('🟢 **Model Loaded:** LightGBM Multi-Output (`model.pkl`) | **Target Scope:** Arabian Sea (8°N–24°N, 60°E–77°E)')
st.markdown("---")

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

formatted_date = (datetime.datetime(2026, 1, 1) + datetime.timedelta(days=day_of_year - 1)).strftime("%d %b %Y")
actual_profile = [sst, float(row_data['temp_50m']), float(row_data['temp_100m']), float(row_data['temp_200m']), float(row_data['temp_500m'])]

# Model Prediction
input_features = pd.DataFrame([[lat, lon, day_of_year, sst, ssh, sss]], columns=['lat', 'lon', 'day_of_year', 'sst', 'ssh', 'sss'])
predicted_depths = model.predict(input_features)[0]
predicted_profile = [sst, round(float(predicted_depths[0]), 2), round(float(predicted_depths[1]), 2), round(float(predicted_depths[2]), 2), round(float(predicted_depths[3]), 2)]

errors = [abs(p - a) for p, a in zip(predicted_profile[1:], actual_profile[1:])]

if "baseline_loc" not in st.session_state:
    st.session_state["baseline_loc"] = {"name": selected_str, "sst": sst, "pred": predicted_profile[1:]}

def get_error_badge(err):
    if err <= 0.25:
        return f'<span class="badge-green">🟢 Error: {err:.2f} °C</span>'
    elif err <= 0.50:
        return f'<span class="badge-yellow">🟡 Error: {err:.2f} °C</span>'
    else:
        return f'<span class="badge-red">🔴 Error: {err:.2f} °C</span>'

# ---------------------------------------------------------
# Streamlit Dashboard UI
# ---------------------------------------------------------
st.subheader("🛰️ Surface Ocean Conditions (Model Inputs)")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Latitude", f"{lat:.2f} °N")
c2.metric("Longitude", f"{lon:.2f} °E")
c3.metric("Observation Date", formatted_date)
c4.metric("SST (Surface Temp)", f"{sst:.2f} °C")
c5.metric("SSH (Surface Height)", f"{ssh:.3f} m")
c6.metric("SSS (Salinity)", f"{sss:.2f} PSU")

st.markdown("---")

st.subheader("🤖 AI Subsurface Predictions (`model.pkl` Inference)")
p1, p2, p3, p4 = st.columns(4)
for col, d, pred, err in zip([p1, p2, p3, p4], ["50m", "100m", "200m", "500m"], predicted_profile[1:], errors):
    with col:
        st.metric(f"Predicted {d}", f"{pred:.2f} °C")
        st.markdown(get_error_badge(err), unsafe_allow_html=True)

st.markdown("---")

col_graph, col_stats = st.columns([2, 1])
with col_graph:
    st.subheader("📊 Subsurface Temperature Profile")
    fig, ax = plt.subplots(figsize=(7, 4))
    depth_levels = [0, 50, 100, 200, 500]
    ax.plot(predicted_profile, depth_levels, 'o-', color='#0066CC', label='OceanEmbed AI Prediction', linewidth=2.5)
    ax.plot(actual_profile, depth_levels, 's--', color='#FF5500', label='Argo Actual Observation', linewidth=2.0)
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

# ---------------------------------------------------------
# Nemotron AI Reasoning Layer (API Version)
# ---------------------------------------------------------
st.markdown("---")
st.header("🧠 Nemotron Intelligence & Reasoning Layer")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Analyze Observation", 
    "📊 Prediction Reliability", 
    "🔬 Compare Observations", 
    "🤖 OceanEmbed Copilot"
])

with tab1:
    st.write("Generate an automated oceanographic reasoning report powered by Nemotron 3 Ultra.")
    if st.button("Run Observation Analysis"):
        with st.spinner("Nemotron 3 Ultra analyzing profile..."):
            prompt = f"""
            Analyze this ocean observation:
            Location: {lat}°N, {lon}°E on {formatted_date}
            Surface Inputs: SST={sst}°C, SSH={ssh}m, SSS={sss}PSU
            Predicted Profiles: 50m={predicted_profile[1]}°C, 100m={predicted_profile[2]}°C, 200m={predicted_profile[3]}°C, 500m={predicted_profile[4]}°C
            Errors vs Argo: 50m={errors[0]:.2f}°C, 100m={errors[1]:.2f}°C, 200m={errors[2]:.2f}°C, 500m={errors[3]:.2f}°C
            
            Interpret only the supplied observations, predictions, errors, and model metrics. Do not invent measurements or causal explanations. Clearly distinguish predictions from observations. Discuss prediction reliability using the supplied errors rather than claiming statistical confidence.
            """
            st.info(query_nemotron(prompt))

with tab2:
    st.write("Evaluate dynamic confidence levels across vertical ocean depth boundaries.")
    if st.button("Generate Reliability Breakdown"):
        with st.spinner("Nemotron evaluating error metrics..."):
            prompt = f"""
            Analyze these depth metrics:
            {real_metrics_df.to_string(index=False)}
            
            Interpret only the supplied observations, predictions, errors, and model metrics. Do not invent measurements or causal explanations. Clearly distinguish predictions from observations. Discuss prediction reliability using the supplied errors rather than claiming statistical confidence.
            
            Explain where error is largest and where user confidence is highest based solely on these metrics.
            """
            st.warning(query_nemotron(prompt))

with tab3:
    st.write("Compare the current profile with a saved baseline point.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Baseline Location A:** {st.session_state['baseline_loc']['name']}")
        if st.button("Save Current as Baseline"):
            st.session_state["baseline_loc"] = {"name": selected_str, "sst": sst, "pred": predicted_profile[1:]}
            st.rerun()
    with col_b:
        st.markdown(f"**Comparison Location B:** {selected_str}")

    if st.button("Compare Profiles with Nemotron"):
        b_data = st.session_state["baseline_loc"]
        with st.spinner("Comparing thermal gradients..."):
            prompt = f"""
            Compare two ocean locations:
            Location A ({b_data['name']}): SST={b_data['sst']}°C, 50m={b_data['pred'][0]}°C, 500m={b_data['pred'][3]}°C
            Location B ({selected_str}): SST={sst}°C, 50m={predicted_profile[1]}°C, 500m={predicted_profile[4]}°C
            
            Interpret only the supplied observations, predictions, errors, and model metrics. Do not invent measurements or causal explanations. Clearly distinguish predictions from observations. Discuss prediction reliability using the supplied errors rather than claiming statistical confidence.
            
            Contrast thermal gradients in 2-3 concise bullet points based solely on the data provided.
            """
            st.success(query_nemotron(prompt))

with tab4:
    st.write("Ask natural language questions regarding the predictions, model performance, or ocean dynamics.")
    query = st.text_input("Ask OceanEmbed Copilot:", placeholder="Why is accuracy higher at 50m than 500m?")
    if query:
        with st.spinner("Copilot generating response..."):
            prompt = f"""
            Context: OceanEmbed AI is predicting ocean subsurface temperatures for location {lat}°N, {lon}°E (SST: {sst}°C, 500m Predicted: {predicted_profile[4]}°C).
            
            User Question: {query}
            
            Interpret only the supplied observations, predictions, errors, and model metrics. Do not invent measurements or causal explanations. Clearly distinguish predictions from observations. Discuss prediction reliability using the supplied errors rather than claiming statistical confidence.
            """
            st.markdown(f"**Copilot Response:**\n\n{query_nemotron(prompt)}")