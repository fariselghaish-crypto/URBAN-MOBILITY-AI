# =========================================================
# FINAL URBAN MOBILITY DIGITAL TWIN DECISION DASHBOARD
# Final stable version: full-page layout, 0-100 scores, status banner,
# sustained alert logic, event differentiation, predictive forecast,
# anomaly monitoring, environmental intelligence and export outputs.
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Urban Mobility Digital Twin Decision Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1580px;
}
.note-box {
    background-color: #eef8ff;
    border-left: 6px solid #0284c7;
    padding: 0.9rem 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    font-size: 0.96rem;
}
.status-stable {
    background-color: #dcfce7;
    border-left: 7px solid #16a34a;
    padding: 0.85rem 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    font-size: 1rem;
}
.status-elevated {
    background-color: #fef9c3;
    border-left: 7px solid #ca8a04;
    padding: 0.85rem 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    font-size: 1rem;
}
.status-critical {
    background-color: #fee2e2;
    border-left: 7px solid #dc2626;
    padding: 0.85rem 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    font-size: 1rem;
}
.warning-box {
    background-color: #fff7ed;
    border-left: 6px solid #f97316;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    font-size: 0.94rem;
}
.small-note {
    background-color: #f8fafc;
    border: 1px solid #e5e7eb;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    margin-top: 0.5rem;
    font-size: 0.90rem;
}
div[data-testid="stMetricValue"] {
    font-size: 1.55rem;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.82rem;
    white-space: normal;
}
</style>
""", unsafe_allow_html=True)

st.title("Urban Mobility Digital Twin Decision Intelligence Dashboard")

st.markdown("""
<div class="note-box">
<b>Digital Twin Simulation:</b> This dashboard uses behaviourally constrained synthetic real-time sensor data
calibrated from historical ONS traffic camera activity patterns. Values are generated for decision-intelligence
demonstration and should not be interpreted as live measured city data.
</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

# For Streamlit Cloud, keep the Excel file in the same GitHub repository folder as this .py file.
DATA_FILE = Path("synthetic_realtime_urban_sensor_data.xlsx")

@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path, sheet_name="5min_Realtime_Sensors")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

try:
    realtime_df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(
        f"Could not find: {DATA_FILE}. Upload synthetic_realtime_urban_sensor_data.xlsx to the same GitHub folder as this app."
    )
    st.stop()

# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.title("Dashboard Controls")

regions = sorted(realtime_df["region"].dropna().unique())
modes = sorted(realtime_df["mode"].dropna().unique())
events = sorted(realtime_df["event_type"].dropna().unique())

selected_regions = st.sidebar.multiselect("Regions", regions, default=regions)
selected_modes = st.sidebar.multiselect("Mobility modes", modes, default=modes)
selected_events = st.sidebar.multiselect("Event types", events, default=events)

date_min = realtime_df["timestamp"].min()
date_max = realtime_df["timestamp"].max()

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(date_min.date(), date_max.date()),
    min_value=date_min.date(),
    max_value=date_max.date()
)

risk_threshold = st.sidebar.slider(
    "Operational risk threshold",
    min_value=0,
    max_value=100,
    value=60,
    step=5
)

environment_threshold = st.sidebar.slider(
    "Environmental pressure threshold",
    min_value=0,
    max_value=100,
    value=50,
    step=5
)

top_n_alerts = st.sidebar.slider(
    "Live alerts to show",
    min_value=25,
    max_value=300,
    value=100,
    step=25
)

filtered = realtime_df[
    (realtime_df["region"].isin(selected_regions)) &
    (realtime_df["mode"].isin(selected_modes)) &
    (realtime_df["event_type"].isin(selected_events)) &
    (realtime_df["timestamp"].dt.date >= selected_dates[0]) &
    (realtime_df["timestamp"].dt.date <= selected_dates[1])
].copy()

if filtered.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

if len(selected_regions) < 3:
    st.markdown("""
    <div class="warning-box">
    <b>Scope note:</b> The current filtered view includes fewer than three regions. For a stronger city-scale
    digital-twin demonstration, generate or select additional regions in the synthetic data file.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# DERIVED METRICS
# =========================================================

mode_environment_factor = {
    "Cars": 1.10,
    "Buses": 1.25,
    "Trucks": 1.45,
    "Vans": 1.20,
    "Motorbikes": 0.85,
    "Pedestrians & cyclists": 0.35
}

mode_congestion_factor = {
    "Cars": 1.15,
    "Buses": 1.10,
    "Trucks": 1.25,
    "Vans": 1.12,
    "Motorbikes": 0.75,
    "Pedestrians & cyclists": 0.40
}

event_factor = {
    "Heavy Rain": 1.25,
    "Roadworks": 1.35,
    "Public Transport Disruption": 1.40,
    "Sport Event": 1.15,
    "Weekend Event": 0.90,
    "Normal": 1.00
}

filtered["Mode Environmental Factor"] = filtered["mode"].map(mode_environment_factor).fillna(1.0)
filtered["Mode Congestion Factor"] = filtered["mode"].map(mode_congestion_factor).fillna(1.0)
filtered["Event Factor"] = filtered["event_type"].map(event_factor).fillna(1.0)

filtered["Operational Risk Score"] = np.clip(
    (
        filtered["disruption_risk"] * 100
        + 0.18 * filtered["congestion_index"]
        + 0.12 * filtered["bus_delay_minutes"] * 5
    ) * filtered["Event Factor"],
    0,
    100
)

filtered["CO2 Proxy Index"] = (
    filtered["congestion_index"] * 1.8 * filtered["Mode Congestion Factor"] +
    filtered["NO2_ugm3"] * 0.7 * filtered["Mode Environmental Factor"] +
    filtered["PM25_ugm3"] * 0.5 * filtered["Mode Environmental Factor"]
)

filtered["Environmental Pressure Score"] = (
    0.35 * (filtered["NO2_ugm3"] / 120) * filtered["Mode Environmental Factor"] +
    0.25 * (filtered["PM25_ugm3"] / 60) * filtered["Mode Environmental Factor"] +
    0.25 * (filtered["congestion_index"] / 100) * filtered["Mode Congestion Factor"] +
    0.10 * (filtered["rainfall_mm"] / 20) +
    0.05 * (filtered["CO2 Proxy Index"] / 250)
)

filtered["Environmental Pressure Score"] = np.clip(filtered["Environmental Pressure Score"] * 100, 0, 100)

filtered["Decision Required"] = np.where(
    filtered["Operational Risk Score"] >= risk_threshold,
    "Yes",
    "No"
)

# Sustained rolling logic to prevent single-sample environmental spikes from inflating alert counts.
filtered["Environmental Rolling Score"] = (
    filtered
    .sort_values("timestamp")
    .groupby(["region", "mode"])["Environmental Pressure Score"]
    .transform(lambda x: x.rolling(3, min_periods=1).mean())
)

filtered["Environmental Alert"] = np.where(
    filtered["Environmental Rolling Score"] >= environment_threshold,
    "Yes",
    "No"
)

filtered["Operational Risk Level"] = pd.cut(
    filtered["Operational Risk Score"],
    bins=[-0.01, 30, 50, 70, 100],
    labels=["Low", "Moderate", "High", "Critical"]
)

filtered["Environmental Risk Level"] = pd.cut(
    filtered["Environmental Pressure Score"],
    bins=[-0.01, 25, 50, 75, 100],
    labels=["Low", "Moderate", "High", "Critical"]
)

# =========================================================
# HOURLY DIGITAL TWIN STATE
# =========================================================

filtered["hour"] = filtered["timestamp"].dt.floor("h")

hourly_state = (
    filtered
    .groupby(["hour", "region", "mode"])
    .agg(
        Activity_Index=("synthetic_activity_index", "mean"),
        Congestion_Index=("congestion_index", "mean"),
        Speed_kmh=("average_speed_kmh", "mean"),
        Bus_Delay_min=("bus_delay_minutes", "mean"),
        NO2_ugm3=("NO2_ugm3", "mean"),
        PM25_ugm3=("PM25_ugm3", "mean"),
        Rainfall_mm=("rainfall_mm", "mean"),
        Parking_pct=("parking_occupancy_pct", "mean"),
        Anomaly_Probability=("anomaly_probability", "mean"),
        Operational_Risk_Score=("Operational Risk Score", "mean"),
        Environmental_Pressure_Score=("Environmental Pressure Score", "mean"),
        CO2_Proxy_Index=("CO2 Proxy Index", "mean")
    )
    .reset_index()
)

# =========================================================
# REGION-MODE SUMMARY
# =========================================================

summary = (
    filtered
    .groupby(["region", "mode"])
    .agg(
        Mean_Activity_Index=("synthetic_activity_index", "mean"),
        Mean_Congestion_Index=("congestion_index", "mean"),
        Mean_Speed_kmh=("average_speed_kmh", "mean"),
        Mean_Rainfall_mm=("rainfall_mm", "mean"),
        Mean_Bus_Delay_min=("bus_delay_minutes", "mean"),
        Mean_NO2_ugm3=("NO2_ugm3", "mean"),
        Mean_PM25_ugm3=("PM25_ugm3", "mean"),
        Mean_CO2_Proxy_Index=("CO2 Proxy Index", "mean"),
        Mean_Parking_pct=("parking_occupancy_pct", "mean"),
        Mean_Anomaly_Probability=("anomaly_probability", "mean"),
        Mean_Operational_Risk_Score=("Operational Risk Score", "mean"),
        Max_Operational_Risk_Score=("Operational Risk Score", "max"),
        Mean_Environmental_Pressure_Score=("Environmental Pressure Score", "mean"),
        Operational_Alerts=("Decision Required", lambda x: (x == "Yes").sum()),
        Environmental_Alerts=("Environmental Alert", lambda x: (x == "Yes").sum())
    )
    .reset_index()
)

summary["Operational Pressure Score"] = (
    0.30 * summary["Mean_Congestion_Index"] +
    0.20 * summary["Mean_Parking_pct"] +
    0.20 * np.clip(summary["Mean_Bus_Delay_min"] / 30 * 100, 0, 100) +
    0.15 * np.clip(summary["Mean_NO2_ugm3"] / 120 * 100, 0, 100) +
    0.15 * np.clip(summary["Mean_Anomaly_Probability"] * 100, 0, 100)
)

summary["Decision Priority Score"] = (
    0.40 * summary["Mean_Operational_Risk_Score"] +
    0.25 * summary["Operational Pressure Score"] +
    0.20 * summary["Mean_Environmental_Pressure_Score"] +
    0.15 * np.clip(summary["Operational_Alerts"] / max(summary["Operational_Alerts"].max(), 1) * 100, 0, 100)
)

summary["Urban Resilience Score"] = (
    100
    - 0.35 * summary["Mean_Operational_Risk_Score"]
    - 0.25 * summary["Mean_Environmental_Pressure_Score"]
    - 0.20 * summary["Mean_Congestion_Index"]
    - 0.10 * np.clip(summary["Mean_Bus_Delay_min"] / 30 * 100, 0, 100)
    - 0.10 * np.clip(summary["Mean_Anomaly_Probability"] * 100, 0, 100)
)
summary["Urban Resilience Score"] = np.clip(summary["Urban Resilience Score"], 0, 100)

def score_band(x):
    if x < 30:
        return "Low"
    if x < 50:
        return "Moderate"
    if x < 70:
        return "High"
    return "Critical"

def resilience_band(x):
    if x >= 75:
        return "Strong"
    if x >= 55:
        return "Moderate"
    if x >= 35:
        return "Weak"
    return "Critical"

summary["Decision Priority Level"] = summary["Decision Priority Score"].apply(score_band)
summary["Environmental Risk Level"] = summary["Mean_Environmental_Pressure_Score"].apply(score_band)
summary["Resilience Level"] = summary["Urban Resilience Score"].apply(resilience_band)

def final_recommendation(row):
    if row["Mean_Environmental_Pressure_Score"] >= 75:
        return "Critical environmental response: low-emission routing, reduced idling, public transport priority and air-quality advisory."
    if row["Decision Priority Score"] >= 70:
        return "Immediate intervention: adaptive signals, diversion routing and live public alert."
    if row["Mean_Operational_Risk_Score"] >= 60:
        return "High monitoring: prioritise public transport and prepare congestion mitigation."
    if row["Mean_Congestion_Index"] >= 70:
        return "Congestion response: optimise junction timing and manage parking inflow."
    if row["Mean_Bus_Delay_min"] >= 10:
        return "Public transport response: apply bus priority strategy."
    if row["Mean_NO2_ugm3"] >= 60:
        return "Environmental response: promote low-emission routing and reduce vehicle idling."
    return "Normal operation: continue passive monitoring."

summary["Recommended Action"] = summary.apply(final_recommendation, axis=1)
summary = summary.sort_values("Decision Priority Score", ascending=False)

# =========================================================
# SHORT-HORIZON FORECAST WITH CONFIDENCE BAND
# =========================================================

forecast_base = (
    hourly_state
    .groupby(["hour", "region"])
    .agg(
        Forecast_Risk=("Operational_Risk_Score", "mean"),
        Forecast_Environmental=("Environmental_Pressure_Score", "mean"),
        Forecast_Congestion=("Congestion_Index", "mean")
    )
    .reset_index()
)

forecast_outputs = []
for region in forecast_base["region"].unique():
    temp = forecast_base[forecast_base["region"] == region].sort_values("hour").copy()
    if len(temp) < 12:
        continue

    last_window = temp.tail(12)
    recent_risk_trend = last_window["Forecast_Risk"].diff().mean()
    recent_env_trend = last_window["Forecast_Environmental"].diff().mean()
    recent_cong_trend = last_window["Forecast_Congestion"].diff().mean()

    risk_sigma = max(last_window["Forecast_Risk"].std(), 1.5)
    env_sigma = max(last_window["Forecast_Environmental"].std(), 1.5)
    cong_sigma = max(last_window["Forecast_Congestion"].std(), 1.5)

    last_time = temp["hour"].max()
    last_risk = temp["Forecast_Risk"].iloc[-1]
    last_env = temp["Forecast_Environmental"].iloc[-1]
    last_cong = temp["Forecast_Congestion"].iloc[-1]

    for h in range(1, 7):
        diurnal = 4.0 * np.sin((h / 6) * np.pi)
        event_pulse = 3.0 if h in [2, 3] else 0.0

        risk_pred = np.clip(last_risk + h * recent_risk_trend + diurnal + event_pulse, 0, 100)
        env_pred = np.clip(last_env + h * recent_env_trend + 0.55 * diurnal + 0.5 * event_pulse, 0, 100)
        cong_pred = np.clip(last_cong + h * recent_cong_trend + 1.1 * diurnal + event_pulse, 0, 100)

        forecast_outputs.append({
            "Forecast Hour": last_time + pd.Timedelta(hours=h),
            "Region": region,
            "Predicted Operational Risk": risk_pred,
            "Risk Lower": np.clip(risk_pred - 1.75 * risk_sigma, 0, 100),
            "Risk Upper": np.clip(risk_pred + 1.75 * risk_sigma, 0, 100),
            "Predicted Environmental Pressure": env_pred,
            "Environmental Lower": np.clip(env_pred - 1.75 * env_sigma, 0, 100),
            "Environmental Upper": np.clip(env_pred + 1.75 * env_sigma, 0, 100),
            "Predicted Congestion Index": cong_pred,
            "Congestion Lower": np.clip(cong_pred - 1.75 * cong_sigma, 0, 100),
            "Congestion Upper": np.clip(cong_pred + 1.75 * cong_sigma, 0, 100)
        })

forecast_df = pd.DataFrame(forecast_outputs)

# =========================================================
# SYSTEM STATUS
# =========================================================

mean_risk = filtered["Operational Risk Score"].mean()
mean_env = filtered["Environmental Pressure Score"].mean()
mean_resilience = summary["Urban Resilience Score"].mean()
max_risk = filtered["Operational Risk Score"].max()

# System status is based on average operating conditions.
# Localised spikes are handled separately in the live alert layer.
if mean_risk >= 70 or mean_env >= 75:
    system_status = "CRITICAL"
    status_class = "status-critical"
    status_text = "Average system conditions require immediate operational attention."
elif mean_risk >= 50 or mean_env >= 60:
    system_status = "ELEVATED"
    status_class = "status-elevated"
    status_text = "Average risk is elevated. Active monitoring and local mitigation should be maintained."
else:
    system_status = "STABLE"
    status_class = "status-stable"
    status_text = "Average system conditions are stable. Localised alerts are handled through the live alert layer."

st.markdown(
    f"""
    <div class="{status_class}">
    <b>SYSTEM STATUS: {system_status}</b> - {status_text}
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# KPI OVERVIEW
# =========================================================

st.subheader("1. System State Overview")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Risk Score", f"{mean_risk:.1f}/100")
k2.metric("Congestion", f"{filtered['congestion_index'].mean():.1f}/100")
k3.metric("Speed", f"{filtered['average_speed_kmh'].mean():.1f} km/h")
k4.metric("Op. Alerts", int((filtered["Decision Required"] == "Yes").sum()))

e1, e2, e3, e4 = st.columns(4)
e1.metric("NO2", f"{filtered['NO2_ugm3'].mean():.1f} ug/m3")
e2.metric("PM2.5", f"{filtered['PM25_ugm3'].mean():.1f} ug/m3")
e3.metric("Env. Score", f"{mean_env:.1f}/100")
e4.metric("Env. Alerts", int((filtered["Environmental Alert"] == "Yes").sum()))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rainfall", f"{filtered['rainfall_mm'].mean():.1f} mm")
m2.metric("Bus Delay", f"{filtered['bus_delay_minutes'].mean():.1f} min")
m3.metric("Parking", f"{filtered['parking_occupancy_pct'].mean():.1f}%")
m4.metric("CO2 Proxy Index", f"{filtered['CO2 Proxy Index'].mean():.1f}")

g1, g2, g3 = st.columns(3)

with g1:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=mean_risk,
        title={"text": "Operational Risk"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#ef4444"},
               "steps": [
                   {"range": [0, 30], "color": "#dcfce7"},
                   {"range": [30, 60], "color": "#fef9c3"},
                   {"range": [60, 100], "color": "#fee2e2"}
               ]}
    ))
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with g2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=mean_env,
        title={"text": "Environmental Pressure"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#0ea5e9"},
               "steps": [
                   {"range": [0, 30], "color": "#dcfce7"},
                   {"range": [30, 60], "color": "#fef9c3"},
                   {"range": [60, 100], "color": "#fee2e2"}
               ]}
    ))
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with g3:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=mean_resilience,
        title={"text": "Urban Resilience"},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "#22c55e"},
               "steps": [
                   {"range": [0, 35], "color": "#fee2e2"},
                   {"range": [35, 75], "color": "#fef9c3"},
                   {"range": [75, 100], "color": "#dcfce7"}
               ]}
    ))
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="small-note">
<b>Alert interpretation:</b> System status is based on average operating conditions, while alert counts capture
localised exceedances across region, mode and time. A stable average system can therefore still generate localised
operational or environmental alerts.
</div>
""", unsafe_allow_html=True)

# =========================================================
# DECISION RANKING
# =========================================================

st.subheader("2. Decision Priority Ranking")

summary_display = summary.rename(columns={
    "region": "Region",
    "mode": "Mobility Mode",
    "Mean_Operational_Risk_Score": "Risk Score",
    "Mean_Congestion_Index": "Congestion",
    "Mean_Speed_kmh": "Speed km/h",
    "Mean_Bus_Delay_min": "Bus Delay min",
    "Mean_NO2_ugm3": "NO2 ug/m3",
    "Mean_PM25_ugm3": "PM2.5 ug/m3",
    "Mean_CO2_Proxy_Index": "CO2 Proxy Index",
    "Mean_Environmental_Pressure_Score": "Env. Score",
    "Operational_Alerts": "Op. Alerts",
    "Environmental_Alerts": "Env. Alerts"
})

ranking_cols = [
    "Region", "Mobility Mode", "Decision Priority Score", "Decision Priority Level",
    "Risk Score", "Operational Pressure Score", "Env. Score", "Environmental Risk Level",
    "Urban Resilience Score", "Resilience Level", "Congestion", "Speed km/h",
    "Bus Delay min", "NO2 ug/m3", "PM2.5 ug/m3", "CO2 Proxy Index",
    "Op. Alerts", "Env. Alerts", "Recommended Action"
]

st.dataframe(
    summary_display[ranking_cols].round(2),
    use_container_width=True,
    hide_index=True
)

color_sequence = px.colors.qualitative.Safe

c1, c2 = st.columns(2)

with c1:
    fig = px.bar(
        summary_display,
        x="Decision Priority Score",
        y="Region",
        color="Mobility Mode",
        orientation="h",
        title="Stacked Decision Priority Contribution by Region and Mobility Mode",
        template="plotly_white",
        color_discrete_sequence=color_sequence,
        height=470,
        hover_data=["Mobility Mode", "Decision Priority Level", "Recommended Action"]
    )
    fig.update_layout(xaxis_title="Stacked decision contribution by mode", yaxis_title="Region", legend_title="Mode")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.scatter(
        hourly_state,
        x="Congestion_Index",
        y="Operational_Risk_Score",
        size="Anomaly_Probability",
        color="mode",
        facet_col="region" if len(selected_regions) <= 3 else None,
        title="Hourly Congestion-Risk Decision Space",
        labels={
            "Congestion_Index": "Congestion Index",
            "Operational_Risk_Score": "Operational Risk Score",
            "mode": "Mobility Mode",
            "region": "Region"
        },
        template="plotly_white",
        color_discrete_sequence=color_sequence,
        height=470,
        opacity=0.65,
        hover_data=["hour", "Speed_kmh", "NO2_ugm3"]
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# REAL-TIME MONITORING
# =========================================================

st.subheader("3. Real-Time Operational Monitoring")

region_ts = (
    filtered
    .groupby(["timestamp", "region"])
    .agg(
        Risk_Score=("Operational Risk Score", "mean"),
        Congestion=("congestion_index", "mean"),
        Speed_kmh=("average_speed_kmh", "mean")
    )
    .reset_index()
    .rename(columns={"timestamp": "Timestamp", "region": "Region"})
)

r1, r2 = st.columns(2)

with r1:
    fig = px.line(
        region_ts,
        x="Timestamp",
        y="Risk_Score",
        color="Region",
        title="Real-Time Operational Risk Score",
        template="plotly_white",
        height=430
    )
    fig.add_hline(y=risk_threshold, line_dash="dash", annotation_text="Risk Threshold")
    fig.update_layout(yaxis_title="Risk Score (0-100)")
    st.plotly_chart(fig, use_container_width=True)

with r2:
    fig = px.line(
        region_ts,
        x="Timestamp",
        y="Congestion",
        color="Region",
        title="Real-Time Congestion Index",
        template="plotly_white",
        height=430
    )
    fig.update_layout(yaxis_title="Congestion Index (0-100)")
    st.plotly_chart(fig, use_container_width=True)

h1, h2 = st.columns(2)

with h1:
    risk_heatmap = summary.pivot(index="region", columns="mode", values="Mean_Operational_Risk_Score")
    fig = px.imshow(
        risk_heatmap,
        text_auto=".1f",
        aspect="auto",
        title="Operational Risk Heatmap",
        labels=dict(x="Mobility Mode", y="Region", color="Risk Score"),
        template="plotly_white",
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

with h2:
    priority_heatmap = summary.pivot(index="region", columns="mode", values="Decision Priority Score")
    fig = px.imshow(
        priority_heatmap,
        text_auto=".1f",
        aspect="auto",
        title="Decision Priority Heatmap",
        labels=dict(x="Mobility Mode", y="Region", color="Priority Score"),
        template="plotly_white",
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FORECAST AND ANOMALY LAYER
# =========================================================

st.subheader("4. Predictive and Anomaly Intelligence")

p1, p2 = st.columns(2)

with p1:
    if not forecast_df.empty:
        fig = go.Figure()
        for region in forecast_df["Region"].unique():
            temp = forecast_df[forecast_df["Region"] == region]
            fig.add_trace(go.Scatter(
                x=temp["Forecast Hour"],
                y=temp["Risk Upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=temp["Forecast Hour"],
                y=temp["Risk Lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name=f"{region} confidence band",
                opacity=0.18
            ))
            fig.add_trace(go.Scatter(
                x=temp["Forecast Hour"],
                y=temp["Predicted Operational Risk"],
                mode="lines+markers",
                name=region
            ))
        fig.add_hline(y=risk_threshold, line_dash="dash", annotation_text="Risk Threshold")
        fig.update_layout(
            title="Next 6-Hour Operational Risk Forecast with Confidence Band",
            xaxis_title="Forecast Hour",
            yaxis_title="Predicted Risk Score (0-100)",
            template="plotly_white",
            height=430
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough hourly data to generate the short-horizon forecast.")

with p2:
    anomaly_summary = (
        hourly_state
        .assign(Anomaly_Score=lambda d: d["Anomaly_Probability"] * 100)
        .groupby(["region", "mode"])
        .agg(Mean_Anomaly_Score=("Anomaly_Score", "mean"))
        .reset_index()
    )
    anomaly_heatmap = anomaly_summary.pivot(index="region", columns="mode", values="Mean_Anomaly_Score")
    fig = px.imshow(
        anomaly_heatmap,
        text_auto=".1f",
        aspect="auto",
        title="Anomaly Probability Heatmap",
        labels=dict(x="Mobility Mode", y="Region", color="Anomaly Score"),
        template="plotly_white",
        height=430
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# ENVIRONMENTAL INTELLIGENCE
# =========================================================

st.subheader("5. Environmental Intelligence and Mobility Trade-Offs")

env1, env2 = st.columns(2)

with env1:
    fig = px.scatter(
        hourly_state,
        x="Congestion_Index",
        y="NO2_ugm3",
        size="PM25_ugm3",
        color="mode",
        hover_data=["region", "hour"],
        title="Hourly Congestion and Air-Quality Relationship",
        labels={
            "Congestion_Index": "Congestion Index",
            "NO2_ugm3": "NO2 (ug/m3)",
            "PM25_ugm3": "PM2.5 (ug/m3)",
            "mode": "Mobility Mode"
        },
        template="plotly_white",
        color_discrete_sequence=color_sequence,
        height=430,
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

with env2:
    fig = px.scatter(
        hourly_state,
        x="Speed_kmh",
        y="Environmental_Pressure_Score",
        size="CO2_Proxy_Index",
        color="mode",
        hover_data=["region", "hour"],
        title="Speed, Environmental Pressure and CO2 Proxy Index",
        labels={
            "Speed_kmh": "Average Speed (km/h)",
            "Environmental_Pressure_Score": "Environmental Pressure Score",
            "CO2_Proxy_Index": "CO2 Proxy Index",
            "mode": "Mobility Mode"
        },
        template="plotly_white",
        color_discrete_sequence=color_sequence,
        height=430,
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

env3, env4 = st.columns(2)

with env3:
    env_heatmap = summary.pivot(index="region", columns="mode", values="Mean_Environmental_Pressure_Score")
    fig = px.imshow(
        env_heatmap,
        text_auto=".1f",
        aspect="auto",
        title="Mode-Adjusted Environmental Pressure Heatmap",
        labels=dict(x="Mobility Mode", y="Region", color="Env. Score"),
        template="plotly_white",
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)

with env4:
    env_ts = (
        filtered
        .groupby(["timestamp", "region"])
        .agg(Env_Score=("Environmental Pressure Score", "mean"))
        .reset_index()
        .rename(columns={"timestamp": "Timestamp", "region": "Region"})
    )
    fig = px.line(
        env_ts,
        x="Timestamp",
        y="Env_Score",
        color="Region",
        title="Real-Time Environmental Pressure Score",
        template="plotly_white",
        height=420
    )
    fig.add_hline(y=environment_threshold, line_dash="dash", annotation_text="Environmental Threshold")
    fig.update_layout(yaxis_title="Environmental Score (0-100)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="small-note">
<b>Environmental score note:</b> The environmental layer uses mode-adjusted pressure factors so that heavy vehicles,
public transport and private vehicles are differentiated more realistically from pedestrian and cycling activity.
</div>
""", unsafe_allow_html=True)

# =========================================================
# EVENT-BASED ANALYSIS
# =========================================================

st.subheader("6. Event-Based Decision and Environmental Analysis")

event_summary = (
    filtered
    .groupby("event_type")
    .agg(
        Risk_Score=("Operational Risk Score", "mean"),
        Env_Score=("Environmental Pressure Score", "mean"),
        Congestion=("congestion_index", "mean"),
        Bus_Delay_min=("bus_delay_minutes", "mean"),
        NO2_ugm3=("NO2_ugm3", "mean"),
        PM25_ugm3=("PM25_ugm3", "mean"),
        CO2_Proxy_Index=("CO2 Proxy Index", "mean"),
        Operational_Alerts=("Decision Required", lambda x: (x == "Yes").sum()),
        Environmental_Alerts=("Environmental Alert", lambda x: (x == "Yes").sum())
    )
    .reset_index()
    .rename(columns={"event_type": "Event Type"})
    .sort_values("Risk_Score", ascending=False)
)

ev1, ev2 = st.columns(2)

with ev1:
    fig = px.bar(
        event_summary,
        x="Event Type",
        y="Risk_Score",
        color="Env_Score",
        title="Event-Based Risk and Environmental Pressure",
        labels={"Risk_Score": "Risk Score", "Env_Score": "Environmental Score"},
        template="plotly_white",
        height=430
    )
    st.plotly_chart(fig, use_container_width=True)

with ev2:
    fig = px.bar(
        event_summary,
        x="Event Type",
        y="Operational_Alerts",
        color="Environmental_Alerts",
        title="Operational and Environmental Alerts by Event Type",
        labels={
            "Operational_Alerts": "Operational Alerts",
            "Environmental_Alerts": "Environmental Alerts"
        },
        template="plotly_white",
        height=430
    )
    st.plotly_chart(fig, use_container_width=True)

st.dataframe(event_summary.round(2), use_container_width=True, hide_index=True)

# =========================================================
# LIVE ALERTS
# =========================================================

st.subheader("7. Live Operational and Environmental Alerts")

alerts = filtered[
    (filtered["Operational Risk Score"] >= risk_threshold) |
    (filtered["Environmental Rolling Score"] >= environment_threshold)
].sort_values("timestamp", ascending=False)

if alerts.empty:
    st.success("No operational or environmental alerts within the selected period.")
else:
    alert_view = alerts[
        [
            "timestamp", "region", "mode", "event_type",
            "congestion_index", "average_speed_kmh", "NO2_ugm3",
            "PM25_ugm3", "rainfall_mm", "Environmental Pressure Score",
            "Environmental Rolling Score", "CO2 Proxy Index", "Operational Risk Score",
            "Operational Risk Level", "Environmental Risk Level", "ai_recommendation"
        ]
    ].rename(columns={
        "timestamp": "Timestamp",
        "region": "Region",
        "mode": "Mobility Mode",
        "event_type": "Event Type",
        "congestion_index": "Congestion",
        "average_speed_kmh": "Speed km/h",
        "NO2_ugm3": "NO2 ug/m3",
        "PM25_ugm3": "PM2.5 ug/m3",
        "rainfall_mm": "Rainfall mm",
        "ai_recommendation": "AI Recommendation"
    })
    st.dataframe(alert_view.head(top_n_alerts).round(2), use_container_width=True, hide_index=True)

# =========================================================
# EXPORTS
# =========================================================

st.subheader("8. Export Decision Outputs")

csv_summary = summary_display.to_csv(index=False).encode("utf-8")
csv_alerts = alerts.to_csv(index=False).encode("utf-8")
csv_hourly = hourly_state.to_csv(index=False).encode("utf-8")
csv_forecast = forecast_df.to_csv(index=False).encode("utf-8")

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.download_button(
        "Download Decision Summary",
        data=csv_summary,
        file_name="urban_mobility_decision_summary.csv",
        mime="text/csv"
    )

with d2:
    st.download_button(
        "Download Live Alerts",
        data=csv_alerts,
        file_name="urban_mobility_live_alerts.csv",
        mime="text/csv"
    )

with d3:
    st.download_button(
        "Download Hourly Twin State",
        data=csv_hourly,
        file_name="urban_mobility_hourly_digital_twin_state.csv",
        mime="text/csv"
    )

with d4:
    st.download_button(
        "Download Forecast Output",
        data=csv_forecast,
        file_name="urban_mobility_forecast_output.csv",
        mime="text/csv"
    )

st.markdown("""
<div class="small-note">
<b>Methodological note:</b> This dashboard demonstrates an explainable AI-enabled urban mobility digital twin.
Operational and environmental thresholds are adjustable for scenario testing. The CO2 Proxy Index is not a measured
emissions inventory.
</div>
""", unsafe_allow_html=True)
