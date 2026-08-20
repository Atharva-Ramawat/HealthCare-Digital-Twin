"""
Streamlit + Plotly Clinical Digital Twin Dashboard - Main Application
Integrates real-time data streaming, CNN-BiLSTM predictions, 15-min vital forecasts,
XAI Integrated Gradients feature attributions, and interactive What-If counterfactual simulations.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from synthetic_generator import SyntheticClinicalDataGenerator
from data_streamer import ClinicalDataStreamer
from model import DigitalTwinPredictor
from explainability import ClinicalXAIExplainer

# Streamlit Page Config
st.set_page_config(
    page_title="AI Clinical Digital Twin",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Clinical UI Aesthetics
st.markdown("""
<style>
    /* Dark Clinical Theme Styling */
    .stApp {
        background-color: #0E1117;
        color: #E2E8F0;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .badge-high {
        background-color: #EF4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: #F59E0B;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .alert-banner {
        background-color: rgba(239, 68, 68, 0.2);
        border-left: 4px solid #EF4444;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State Singletons
@st.cache_resource
def load_core_engine():
    """Load model predictor and streamer once into memory."""
    predictor = DigitalTwinPredictor()
    streamer = ClinicalDataStreamer()
    explainer = ClinicalXAIExplainer(predictor.model)
    return predictor, streamer, explainer


predictor, streamer, explainer = load_core_engine()

if 'alerts_history' not in st.session_state:
    st.session_state.alerts_history = []


# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/000000/heart-health.png", width=60)
st.sidebar.title("Digital Twin Engine")
st.sidebar.markdown("---")

# Navigation View Selection
view_mode = st.sidebar.radio(
    "Select Navigation View",
    ["Ward Overview", "Patient Digital Twin", "What-If Counterfactual", "Early Warning Alerts"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Controls")

# Stream Playback Controls
col_p1, col_p2 = st.sidebar.columns(2)
if col_p1.button("▶ Play", use_container_width=True):
    streamer.play()
if col_p2.button("⏸ Pause", use_container_width=True):
    streamer.pause()

playback_speed = st.sidebar.select_slider(
    "Playback Speed",
    options=["1x", "2x", "5x", "10x"],
    value="1x"
)
speed_mult = float(playback_speed.replace("x", ""))
streamer.set_speed(speed_mult)

if st.sidebar.button("⏭ Advance Step", use_container_width=True):
    streamer.step_single()

st.sidebar.markdown("---")
st.sidebar.subheader("Active Patient Switcher")
patient_options = list(streamer.patient_profiles.keys())
selected_pid = st.sidebar.selectbox(
    "Select Monitored Patient",
    options=patient_options,
    format_func=lambda pid: f"{pid} - {streamer.patient_profiles[pid]['name']}"
)
streamer.set_active_patient(selected_pid)

st.sidebar.markdown("---")
st.sidebar.subheader("Inject Anomaly / Shock")
col_s1, col_s2 = st.sidebar.columns(2)
if col_s1.button("🚨 Septic Shock", use_container_width=True):
    streamer.inject_anomaly("septic_spike", selected_pid)
    st.sidebar.warning(f"Injected Septic Shock into {selected_pid}!")

if col_s2.button("🫁 ARDS / Hypoxia", use_container_width=True):
    streamer.inject_anomaly("hypoxia_drop", selected_pid)
    st.sidebar.warning(f"Injected Hypoxia into {selected_pid}!")

if st.sidebar.button("🔄 Clear Active Shock", use_container_width=True):
    streamer.inject_anomaly("reset", selected_pid)
    st.sidebar.success("Cleared active shock anomalies!")


# Core Inference Calculation for Ward Patients
ward_inference_results = {}
for p_id in patient_options:
    window = streamer.get_sliding_window_matrix(p_id)
    pred = predictor.predict(window)
    ward_inference_results[p_id] = pred
    
    # Alert logging trigger
    if pred['risk_score'] > config.RISK_THRESHOLD_HIGH:
        timestamp_str = time.strftime("%H:%M:%S")
        latest_vital = streamer.buffers[p_id][-1]
        alert_entry = {
            "timestamp": timestamp_str,
            "patient_id": p_id,
            "name": streamer.patient_profiles[p_id]['name'],
            "risk_score": pred['risk_score'],
            "news2_tier": pred['news2_tier'],
            "trigger_vital": f"HR: {latest_vital['heart_rate']} | SpO2: {latest_vital['spo2']}% | BP: {latest_vital['systolic_bp']}"
        }
        if len(st.session_state.alerts_history) == 0 or st.session_state.alerts_history[-1]['patient_id'] != p_id or st.session_state.alerts_history[-1]['timestamp'] != timestamp_str:
            st.session_state.alerts_history.append(alert_entry)


# Top Banner Notification for Critical Alerts
critical_active = [p_id for p_id, pred in ward_inference_results.items() if pred['risk_score'] > config.RISK_THRESHOLD_HIGH]
if critical_active:
    st.markdown(
        f"""
        <div class="alert-banner">
            ⚠️ <strong>CRITICAL CLINICAL ALERT:</strong> High risk threshold breached for patient(s): 
            <strong>{', '.join([f"{pid} ({streamer.patient_profiles[pid]['name']})" for pid in critical_active])}</strong>. 
            Immediate clinical review required!
        </div>
        """,
        unsafe_allow_html=True
    )


# VIEW 1: WARD OVERVIEW
if view_mode == "Ward Overview":
    st.title("🏥 ICU Ward Overview - Live Patient Twin Grid")
    st.markdown("Real-time clinical monitoring across active virtual ICU beds.")

    cols = st.columns(2)
    for idx, p_id in enumerate(patient_options):
        profile = streamer.patient_profiles[p_id]
        pred = ward_inference_results[p_id]
        latest = streamer.buffers[p_id][-1]
        risk = pred['risk_score']

        badge_class = "badge-high" if risk > config.RISK_THRESHOLD_HIGH else ("badge-medium" if risk > config.RISK_THRESHOLD_MEDIUM else "badge-low")

        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h3 style="margin:0; color:#F8FAFC;">{profile['bed']}: {profile['name']}</h3>
                        <span class="{badge_class}">{pred['news2_tier'].upper()} ({risk:.2f})</span>
                    </div>
                    <p style="color:#94A3B8; margin-top:4px;">ID: {p_id} | Age: {profile['age']} | Trajectory: {profile['trajectory'].upper()}</p>
                    <hr style="border-color: rgba(255,255,255,0.1); margin: 8px 0;">
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 0.9rem;">
                        <div>❤️ HR: <strong>{latest['heart_rate']}</strong> bpm</div>
                        <div>🫁 SpO2: <strong>{latest['spo2']}</strong> %</div>
                        <div>🩸 SBP: <strong>{latest['systolic_bp']}</strong> mmHg</div>
                        <div>🫁 RR: <strong>{latest['respiratory_rate']}</strong> /min</div>
                        <div>🌡️ Temp: <strong>{latest['body_temperature']}</strong> °C</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# VIEW 2: PATIENT DIGITAL TWIN
elif view_mode == "Patient Digital Twin":
    st.title(f"👤 Digital Twin: {selected_pid} ({streamer.patient_profiles[selected_pid]['name']})")
    
    current_pred = ward_inference_results[selected_pid]
    history_df = streamer.get_patient_history_df(selected_pid)
    forecast_matrix = current_pred['forecast_matrix']  # Shape (15, 5)

    # Top Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Health Risk Score", f"{current_pred['risk_score']:.3f}", delta=None)
    m_col2.metric("NEWS 2 Risk Tier", current_pred['news2_tier'])
    m_col3.metric("Current Heart Rate", f"{history_df['heart_rate'].iloc[-1]} bpm")
    m_col4.metric("Current SpO2", f"{history_df['spo2'].iloc[-1]} %")

    # Risk Gauge & XAI Side-by-Side
    g_col1, g_col2 = st.columns([1, 1])

    with g_col1:
        st.subheader("Risk Score Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_pred['risk_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Digital Twin Health Risk Index"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "#EF4444" if current_pred['risk_score'] > 0.65 else "#3B82F6"},
                'steps': [
                    {'range': [0, 0.35], 'color': "rgba(16, 185, 129, 0.3)"},
                    {'range': [0.35, 0.65], 'color': "rgba(245, 158, 11, 0.3)"},
                    {'range': [0.65, 1.0], 'color': "rgba(239, 68, 68, 0.3)"}
                ]
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font_color="#E2E8F0")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with g_col2:
        st.subheader("XAI Integrated Gradients Attribution")
        window_mat = streamer.get_sliding_window_matrix(selected_pid)
        xai_res = explainer.compute_integrated_gradients(window_mat)
        
        params = [info['display_name'] for info in xai_res.values()]
        impacts = [info['impact_percentage'] for info in xai_res.values()]

        fig_xai = go.Figure(go.Bar(
            x=impacts,
            y=params,
            orientation='h',
            marker=dict(color=['#FF4B4B', '#00D26A', '#3B82F6', '#F59E0B', '#8B5CF6'])
        ))
        fig_xai.update_layout(
            title="Parameter Contribution to Risk Score (%)",
            xaxis_title="Impact %",
            height=280,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E2E8F0"
        )
        st.plotly_chart(fig_xai, use_container_width=True)

    # Plotly Continuous Vitals + 15-Min Trajectory Forecast Subplots
    st.subheader("📈 Live Continuous Vitals & 15-Minute Future Forecast Horizon")

    fig_vitals = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[config.VITALS_META[k]['display_name'] for k in config.VITALS_KEYS]
    )

    t_hist = np.arange(len(history_df))
    t_fore = np.arange(len(history_df) - 1, len(history_df) + config.FORECAST_HORIZON)

    for i, key in enumerate(config.VITALS_KEYS):
        # Historical actual values
        fig_vitals.add_trace(
            go.Scatter(
                x=t_hist,
                y=history_df[key],
                mode='lines+markers',
                name=f"{config.VITALS_META[key]['display_name']} (Live)",
                line=dict(color=config.VITALS_META[key]['color'], width=2)
            ),
            row=i+1, col=1
        )
        # Forecasted values trajectory
        forecast_series = np.concatenate([[history_df[key].iloc[-1]], forecast_matrix[:, i]])
        fig_vitals.add_trace(
            go.Scatter(
                x=t_fore,
                y=forecast_series,
                mode='lines',
                name=f"{config.VITALS_META[key]['display_name']} (15m Forecast)",
                line=dict(color=config.VITALS_META[key]['color'], width=2, dash='dash')
            ),
            row=i+1, col=1
        )

    fig_vitals.update_layout(
        height=700,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font_color="#E2E8F0"
    )
    st.plotly_chart(fig_vitals, use_container_width=True)


# VIEW 3: WHAT-IF COUNTERFACTUAL SIMULATION
elif view_mode == "What-If Counterfactual":
    st.title("🧪 'What-If' Counterfactual Simulation Panel")
    st.markdown("Interactively adjust vitals using sliders to observe real-time Digital Twin recalibration.")

    latest_frame = dict(streamer.buffers[selected_pid][-1])
    orig_pred = ward_inference_results[selected_pid]

    col_sim_controls, col_sim_view = st.columns([1, 2])

    with col_sim_controls:
        st.subheader("Counterfactual Vital Sliders")
        sim_hr = st.slider("Heart Rate (bpm)", 40.0, 180.0, float(latest_frame['heart_rate']))
        sim_spo2 = st.slider("Oxygen Saturation SpO2 (%)", 70.0, 100.0, float(latest_frame['spo2']))
        sim_sbp = st.slider("Systolic BP (mmHg)", 60.0, 200.0, float(latest_frame['systolic_bp']))
        sim_rr = st.slider("Respiratory Rate (/min)", 8.0, 45.0, float(latest_frame['respiratory_rate']))
        sim_temp = st.slider("Body Temperature (°C)", 34.0, 41.5, float(latest_frame['body_temperature']))

        overrides = {
            'heart_rate': sim_hr,
            'spo2': sim_spo2,
            'systolic_bp': sim_sbp,
            'respiratory_rate': sim_rr,
            'body_temperature': sim_temp
        }

    # Compute Recalibrated Digital Twin Risk
    sim_window = streamer.get_sliding_window_matrix(selected_pid)
    sim_window[-1] = np.array([sim_hr, sim_spo2, sim_sbp, sim_rr, sim_temp])
    recalibrated_pred = predictor.predict(sim_window)

    with col_sim_view:
        st.subheader("Live Recalibration Comparison")
        
        rc1, rc2 = st.columns(2)
        rc1.metric("Original Risk Score", f"{orig_pred['risk_score']:.3f}", delta=None)
        
        delta_risk = recalibrated_pred['risk_score'] - orig_pred['risk_score']
        rc2.metric("Recalibrated Risk Score", f"{recalibrated_pred['risk_score']:.3f}", delta=f"{delta_risk:+.3f}")

        st.markdown(f"**Original NEWS 2 Tier:** {orig_pred['news2_tier']}  ➡️  **Simulated Tier:** {recalibrated_pred['news2_tier']}")

        # Comparison Bar Chart
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='Original Baseline',
            x=['Risk Score'],
            y=[orig_pred['risk_score']],
            marker_color='#3B82F6'
        ))
        fig_comp.add_trace(go.Bar(
            name='Counterfactual Simulation',
            x=['Risk Score'],
            y=[recalibrated_pred['risk_score']],
            marker_color='#EF4444' if recalibrated_pred['risk_score'] > 0.65 else '#10B981'
        ))
        fig_comp.update_layout(
            title="Baseline vs Recalibrated Digital Twin Risk Index",
            barmode='group',
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E2E8F0"
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# VIEW 4: EARLY WARNING ALERTS LOG
elif view_mode == "Early Warning Alerts":
    st.title("🚨 Clinical Early Warning System (NEWS 2 Alert Feed)")
    st.markdown("Automated risk threshold breach logging and clinical alert audit trail.")

    if st.session_state.alerts_history:
        alerts_df = pd.DataFrame(st.session_state.alerts_history)
        st.dataframe(alerts_df, use_container_width=True)
    else:
        st.info("No critical clinical risk breaches recorded in current session.")


# Auto-refresh loop when playing in Streamlit
if streamer.is_playing:
    time.sleep(1.0 / speed_mult)
    streamer.step_single()
    st.rerun()
