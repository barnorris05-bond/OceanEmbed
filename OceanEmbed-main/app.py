import datetime
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI

# Page Config & Custom Styling
st.set_page_config(page_title="OceanEmbed AI", layout="wide")

st.markdown("""
    <style>
body {
    background-color: #F4F8FB;
}

[data-testid="stAppViewContainer"] {
    background-color: #F4F8FB;
}

[data-testid="stHeader"] {
    background-color: #F4F8FB;
}
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; }
    .badge-green { background-color: #DCFCE7; color: #15803D; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
    .badge-yellow { background-color: #FEF9C3; color: #A16207; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
    .badge-red { background-color: #FEE2E2; color: #B91C1C; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
    .ocean-hero {
    padding: 1.2rem 1.8rem 1.3rem 1.8rem;
    border-radius: 18px;
    margin: 1rem 0 1.5rem 0;
    border: 1px solid rgba(100, 150, 180, 0.25);
    background: linear-gradient(
    135deg,
    #E8F3F8 0%,
    #F4F8FB 55%,
    #FFFFFF 100%
);
}
.ocean-title {
    font-size: 2.6rem;
    font-weight: 750;
    margin-bottom: 0.2rem;
}

.ocean-subtitle {
    font-size: 1.25rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

.ocean-description {
    font-size: 0.95rem;
    opacity: 0.75;
    margin-bottom: 1rem;
}

.ocean-meta {
    display: flex;
    gap: 0.7rem;
    flex-wrap: wrap;
}

.ocean-system-status {
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px solid rgba(100, 150, 180, 0.2);
    font-size: 0.78rem;
    opacity: 0.7;
}
.ocean-status {
    float: right;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.8rem;
    border: 1px solid rgba(80, 180, 120, 0.35);
}
.observation-section {
    padding: 1rem 1.2rem;
    border-radius: 14px;
    border: 1px solid rgba(100, 150, 180, 0.22);
    margin-bottom: 1rem;
}

.section-label {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
    opacity: 0.75;
}
    </style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="ocean-hero">

<div class="ocean-status">🟢 MODEL ONLINE</div>

<div class="ocean-title">🌊 OceanEmbed</div>

<div class="ocean-subtitle">
AI-Powered Subsurface Ocean Intelligence
</div>

<div class="ocean-description">
From observable surface conditions to interpretable subsurface predictions.
</div>

<div class="ocean-meta">
<span class="ocean-badge">🤖 LightGBM Multi-Output</span>
<span class="ocean-badge">🌊 Arabian Sea</span>
<span class="ocean-badge">🧠 AI-Assisted Analysis</span>
<div class="ocean-system-status">
    ✓ Model loaded: <strong>model.pkl</strong>
    &nbsp;•&nbsp;
    4 depth outputs
    &nbsp;•&nbsp;
    Arabian Sea: 8°N–24°N, 60°E–77°E
</div>
</div>

</div>
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
            model="nvidia/nemotron-3-super-120b-a12b",
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
# ---------------------------------------------------------
# Ocean Insight Engine
# ---------------------------------------------------------
depths = [0, 50, 100, 200, 500]
profile_temps = predicted_profile

# Temperature gradient between consecutive depth levels
gradients = []

for i in range(len(depths) - 1):
    depth_change = depths[i + 1] - depths[i]
    temp_change = profile_temps[i + 1] - profile_temps[i]
    gradients.append(abs(temp_change / depth_change))

max_gradient_index = int(np.argmax(gradients))
max_gradient = gradients[max_gradient_index]

gradient_start = depths[max_gradient_index]
gradient_end = depths[max_gradient_index + 1]

# Prototype threshold for identifying a strong thermal gradient
if max_gradient >= 0.05:
    insight_title = "Strong thermal gradient detected"
    insight_indication = "Enhanced stratification"
    insight_level = "Strong"
elif max_gradient >= 0.025:
    insight_title = "Moderate thermal gradient detected"
    insight_indication = "Possible stratification"
    insight_level = "Moderate"
else:
    insight_title = "Weak thermal gradient detected"
    insight_indication = "Relatively mixed water column"
    insight_level = "Weak"

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
# ---------------------------------------------------------
# Observation Point
# ---------------------------------------------------------
st.subheader("📍 Observation Point")

c1, c2, c3 = st.columns(3)

c1.metric("📍Latitude", f"{lat:.2f} °N")
c2.metric("📍Longitude", f"{lon:.2f} °E")
c3.metric("📅Observation Date", formatted_date)


# ---------------------------------------------------------
# Surface Ocean Conditions
# ---------------------------------------------------------
st.subheader("🌊 Surface Ocean Conditions")
c4, c5, c6 = st.columns(3)

c4.metric("🌡️ SST", f"{sst:.2f} °C")
c5.metric("🌊 SSH", f"{ssh:.3f} m")
c6.metric("🧂 Salinity", f"{sss:.2f} PSU")

st.markdown("---")

# ---------------------------------------------------------
# Arabian Sea Observation Map
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🗺️ Arabian Sea Observation Location")
st.caption(
    "Geographic position of the selected ocean observation."
)

map_col, location_col = st.columns([2.2, 1])

with map_col:
    

    fig_map = go.Figure()

    fig_map.add_trace(
        go.Scattermap(
            lon=[lon],
            lat=[lat],
            mode="markers",
            marker=dict(
                size=16
            ),
            text=[
                f"Observation Point<br>"
                f"Latitude: {lat:.2f}°N<br>"
                f"Longitude: {lon:.2f}°E"
            ],
            hoverinfo="text"
        )
    )

    fig_map.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(
                lat=lat,
                lon=lon
            ),
            zoom=4.2
        ),
        height=430,
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "showTips": False,
            "staticPlot": True,
            "responsive": True
        }
    )

with location_col:
    with st.container(border=True):
        st.markdown("### 📍 Observation Point")

        st.metric(
            "Latitude",
            f"{lat:.2f}°N"
        )

        st.metric(
            "Longitude",
            f"{lon:.2f}°E"
        )

        st.markdown("---")

        st.markdown("**Region**")
        st.write("Arabian Sea")

        st.markdown("**Observation Date**")
        st.write(formatted_date)

        st.caption(
            "This location is used as the surface input "
            "for subsurface temperature prediction."
        )

st.markdown("---")


st.subheader("🤖 AI Subsurface Prediction")

st.caption(
    "Predicted ocean temperature profile derived from observed surface conditions."
)

p1, p2, p3, p4 = st.columns(4)

for col, d, pred, err in zip(
    [p1, p2, p3, p4],
    ["50m", "100m", "200m", "500m"],
    predicted_profile[1:],
    errors
):
    with col:
        with st.container(border=True):
            st.markdown(
                f"<div style='text-align:center; font-size:0.85rem; "
                f"opacity:0.7; font-weight:600;'>DEPTH</div>",
                unsafe_allow_html=True
            )

            st.markdown(
                f"<div style='text-align:center; font-size:1.4rem; "
                f"font-weight:700; margin-bottom:0.4rem;'>{d}</div>",
                unsafe_allow_html=True
            )

            st.metric(
                "Predicted Temperature",
                f"{pred:.2f} °C"
            )

            st.markdown(
                get_error_badge(err),
                unsafe_allow_html=True
            )
# ---------------------------------------------------------
# Ocean Insight
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🌊 Ocean Insight")
st.caption(
    "Scientific interpretation derived directly from the predicted temperature profile."
)

insight_col1, insight_col2 = st.columns([2, 1])

with insight_col1:
    with st.container(border=True):
        st.markdown(f"### {insight_title}")
        st.markdown(
            f"**Depth range:** {gradient_start}–{gradient_end} m"
        )
        st.markdown(
            f"**Potential indication:** {insight_indication}"
        )
        st.caption(
            f"Temperature gradient: {max_gradient:.3f} °C/m"
        )

with insight_col2:
    with st.container(border=True):
        st.markdown("### 🔬 Signal Strength")
        st.metric("Gradient", insight_level)
        st.caption(
            "Based on the steepest predicted temperature change "
            "between adjacent depth levels."
        )
# ---------------------------------------------------------
# Scientific Consistency & Subsurface Deviation
# ---------------------------------------------------------

depths_check = [0, 50, 100, 200, 500]
predicted_check = predicted_profile

# Check whether temperature generally decreases with depth
temperature_changes = [
    predicted_check[i + 1] - predicted_check[i]
    for i in range(len(predicted_check) - 1)
]

inversions = [
    i for i, change in enumerate(temperature_changes)
    if change > 0
]

# Check for physically unusual temperature jumps
max_jump = max(abs(change) for change in temperature_changes)

if len(inversions) == 0:
    consistency_status = "🟢 Consistent"
    consistency_message = "Temperature decreases continuously with depth."
else:
    consistency_status = "🟡 Review"
    consistency_message = (
        f"Temperature inversion detected between "
        f"{depths_check[inversions[0]]}m and "
        f"{depths_check[inversions[0] + 1]}m."
    )

# Compare prediction against ARGO observations
argo_errors = np.array(errors)
largest_error_index = int(np.argmax(argo_errors))
largest_error = float(argo_errors[largest_error_index])

comparison_depths = [50, 100, 200, 500]
largest_error_depth = comparison_depths[largest_error_index]

# Prototype threshold for highlighting substantial model disagreement
deviation_threshold = 0.75

if largest_error >= deviation_threshold:
    deviation_status = "🔴 Significant deviation"
    deviation_message = (
        "Large disagreement between OceanEmbed prediction "
        "and the available ARGO observation."
    )
else:
    deviation_status = "🟢 Good agreement"
    deviation_message = (
        "OceanEmbed predictions remain reasonably close "
        "to the available ARGO observations."
    )

st.markdown("---")
st.subheader("🔬 Scientific Validation")
st.caption(
    "Automated checks applied to the predicted subsurface temperature profile."
)

check_col1, check_col2 = st.columns(2)

with check_col1:
    with st.container(border=True):
        st.markdown("### 🧪 Scientific Consistency")

        st.markdown(
            f"## {consistency_status}"
        )

        st.write(consistency_message)

        st.caption(
            f"Maximum temperature change between adjacent depths: "
            f"{max_jump:.2f} °C"
        )

with check_col2:
    with st.container(border=True):
        st.markdown("### ⚠️ Subsurface Deviation")

        st.markdown(
            f"## {deviation_status}"
        )

        st.write(deviation_message)

        st.metric(
            f"Largest difference at {largest_error_depth}m",
            f"{largest_error:.2f} °C"
        )

        st.caption(
            "Comparison against the available ARGO reference profile."
        )
st.markdown("---")
# Prediction Drivers / Feature Importance
feature_names = ['lat', 'lon', 'day_of_year', 'sst', 'ssh', 'sss']

try:
    importances = []

    for estimator in model.estimators_:
        importances.append(estimator.feature_importances_)

    avg_importance = np.mean(importances, axis=0)
    importance_pct = (avg_importance / avg_importance.sum()) * 100

    feature_labels = {
        'lat': '📍 Latitude',
        'lon': '📍 Longitude',
        'day_of_year': '📅 Day of Year',
        'sst': '🌡️ SST',
        'ssh': '🌊 SSH',
        'sss': '🧂 Salinity'
    }

    driver_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_pct
    }).sort_values('Importance', ascending=True)

    st.subheader("🧠 Why did the model predict this?")
    st.caption(
        "Relative feature importance across the four depth prediction models."
    )

    chart_df = driver_df.copy()
    chart_df['Feature'] = chart_df['Feature'].map(feature_labels)

    top_feature = driver_df.iloc[-1]['Feature']
    top_importance = driver_df.iloc[-1]['Importance']

    col_graph, col_stats = st.columns([2.4, 1])

    with col_graph:
        

        fig = px.bar(
            chart_df,
            x='Importance',
            y='Feature',
            orientation='h',
            text='Importance'
        )

        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            width=0.55,
            hoverinfo='skip'
        )

        fig.update_layout(
            height=380,
            xaxis_title="Relative Importance (%)",
            yaxis_title="",
            xaxis=dict(
                range=[0, max(chart_df['Importance']) * 1.2],
                fixedrange=True
            ),
            yaxis=dict(
                fixedrange=True
            ),
            margin=dict(l=20, r=60, t=20, b=45),
            showlegend=False,
            dragmode=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "scrollZoom": False,
                "doubleClick": False,
                "showTips": False,
                "responsive": True
            }
        )

    with col_stats:
        with st.container(border=True):
            st.markdown("### 🏆 Top Driver")

            st.markdown(
                f"## {feature_labels[top_feature]}"
            )

            st.metric(
                "Relative Importance",
                f"{top_importance:.1f}%"
            )

            st.caption(
                "Highest feature importance across the four "
                "depth predictions."
            )

    st.info(
        f"🧠 **KEY INSIGHT**\n\n"
        f"{feature_labels[top_feature]} has the highest relative feature "
        f"importance at {top_importance:.1f}% across the four depth predictions."
    )
except Exception as e:
    st.warning(f"Prediction driver analysis unavailable: {e}")
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
            Provide a 3-bullet physical summary of thermocline decay and local confidence.
            """
            st.info(query_nemotron(prompt))

with tab2:
    st.write("Evaluate dynamic confidence levels across vertical ocean depth boundaries.")
    if st.button("Generate Reliability Breakdown"):
        with st.spinner("Nemotron evaluating error metrics..."):
            prompt = f"""
            System: You are an ML diagnostic expert. Analyze these depth metrics:
            {real_metrics_df.to_string(index=False)}
            Explain why error accumulates at 500m and where user confidence is highest.
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
            Contrast thermal gradients and thermocline decay in 3 concise bullet points.
            """
            st.success(query_nemotron(prompt))

with tab4:
    st.write("Ask natural language questions regarding the predictions, model performance, or ocean dynamics.")
    query = st.text_input("Ask OceanEmbed Copilot:", placeholder="Why is accuracy higher at 50m than 500m?")
    if query:
        with st.spinner("Copilot generating response..."):
            prompt = f"""
            Answer as OceanEmbed AI Assistant for location {lat}°N, {lon}°E (SST: {sst}°C, 500m Pred: {predicted_profile[4]}°C).
            User Question: {query}
            """
            st.markdown(f"**Copilot Response:**\n\n{query_nemotron(prompt)}")