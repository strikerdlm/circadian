import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from circadian.lights import LightSchedule
from circadian.models import Forger99, Jewett99, Hannay19, Hannay19TP
from circadian.plots import Actogram
from circadian.readers import load_csv, load_json

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Circadian Explorer",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal theming helpers
PRIMARY_COLOR = "#6C63FF"
ACCENT_COLOR = "#20C997"

# -----------------------------
# Sidebar - App Header
# -----------------------------
st.sidebar.markdown(
    """
    <div style='display:flex; align-items:center; gap:10px;'>
      <span style='font-size:1.4rem'>🌙</span>
      <span style='font-weight:700; font-size:1.1rem'>Circadian Explorer</span>
    </div>
    <div style='font-size:0.9rem; color:#777; margin-top:4px;'>Simulate and visualize circadian rhythms</div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

# -----------------------------
# Utility functions
# -----------------------------
MODEL_OPTIONS = {
    "Forger99": Forger99,
    "Jewett99": Jewett99,
    "Hannay19": Hannay19,
    "Hannay19TP": Hannay19TP,
}

SCHEDULE_OPTIONS = [
    "Regular",
    "ShiftWork",
    "SlamShift",
    "SocialJetlag",
    "Custom Pulse",
]

@st.cache_data(show_spinner=False)
def generate_time_series(total_days: int, step_hours: float) -> np.ndarray:
    return np.arange(0, 24 * total_days, step_hours)

@st.cache_data(show_spinner=False)
def compute_light(schedule_name: str, time: np.ndarray, params: dict) -> np.ndarray:
    if schedule_name == "Regular":
        sched = LightSchedule.Regular(
            lux=params.get("lux", 150.0),
            lights_on=params.get("lights_on", 7.0),
            lights_off=params.get("lights_off", 23.0),
        )
    elif schedule_name == "ShiftWork":
        sched = LightSchedule.ShiftWork(
            lux=params.get("lux", 150.0),
            days_on=int(params.get("days_on", 5)),
            days_off=int(params.get("days_off", 2)),
        )
    elif schedule_name == "SlamShift":
        sched = LightSchedule.SlamShift(
            lux=params.get("lux", 150.0),
            shift=params.get("shift", 8.0),
        )
    elif schedule_name == "SocialJetlag":
        sched = LightSchedule.SocialJetlag(
            lux=params.get("lux", 150.0),
            num_regular_days=int(params.get("num_regular_days", 5)),
            late_bedtime=params.get("late_bedtime", 1.0),
            late_waketime=params.get("late_waketime", 2.0),
        )
    elif schedule_name == "Custom Pulse":
        sched = LightSchedule.from_pulse(
            lux=params.get("lux", 500.0),
            start=params.get("start", 8.0),
            duration=params.get("duration", 2.0),
            period=params.get("period", 24.0),
            baseline=params.get("baseline", 0.0),
        )
    else:
        raise ValueError("Unknown schedule")
    return sched(time)

@st.cache_data(show_spinner=False)
def equilibrate_model(model_name: str, time: np.ndarray, light: np.ndarray, equilibration_reps: int):
    model_cls = MODEL_OPTIONS[model_name]
    model = model_cls()
    return model, model.equilibrate(time, light, equilibration_reps)

@st.cache_data(show_spinner=False)
def integrate_model(model_name: str, time: np.ndarray, light: np.ndarray, x0: np.ndarray):
    model_cls = MODEL_OPTIONS[model_name]
    model = model_cls()
    traj = model(time, x0, light)
    return model, traj

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "Simulation", 
    "Wearable Data", 
])

# -----------------------------
# Tab 1: Simulation
# -----------------------------
with tabs[0]:
    left, right = st.columns([0.45, 0.55], gap="large")

    with left:
        st.subheader("Light Schedule")
        schedule_name = st.selectbox("Schedule", SCHEDULE_OPTIONS, index=0)
        total_days = st.slider("Total days", min_value=5, max_value=120, value=30, step=1)
        step_hours = st.select_slider("Time step (hours)", options=[0.05, 0.1, 0.25, 0.5, 1.0], value=0.1)

        sched_params = {}
        if schedule_name == "Regular":
            sched_params["lux"] = st.number_input("Lux", min_value=0.0, value=150.0, step=10.0)
            sched_params["lights_on"] = st.number_input("Lights on (h)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            sched_params["lights_off"] = st.number_input("Lights off (h)", min_value=0.0, max_value=24.0, value=23.0, step=0.5)
        elif schedule_name == "ShiftWork":
            sched_params["lux"] = st.number_input("Lux", min_value=0.0, value=300.0, step=10.0)
            sched_params["days_on"] = st.number_input("Night days on", min_value=1, max_value=14, value=3, step=1)
            sched_params["days_off"] = st.number_input("Days off", min_value=1, max_value=14, value=2, step=1)
        elif schedule_name == "SlamShift":
            sched_params["lux"] = st.number_input("Lux", min_value=0.0, value=300.0, step=10.0)
            sched_params["shift"] = st.number_input("Shift (h)", min_value=-12.0, max_value=12.0, value=8.0, step=0.5)
        elif schedule_name == "SocialJetlag":
            sched_params["lux"] = st.number_input("Lux", min_value=0.0, value=150.0, step=10.0)
            sched_params["num_regular_days"] = st.number_input("Regular days", min_value=1, max_value=6, value=5, step=1)
            sched_params["late_bedtime"] = st.number_input("Weekend bedtime delay (h)", min_value=0.0, max_value=12.0, value=1.0, step=0.5)
            sched_params["late_waketime"] = st.number_input("Weekend wake delay (h)", min_value=0.0, max_value=12.0, value=2.0, step=0.5)
        elif schedule_name == "Custom Pulse":
            sched_params["lux"] = st.number_input("Pulse Lux", min_value=0.0, value=500.0, step=10.0)
            sched_params["start"] = st.number_input("Pulse start (h)", min_value=0.0, max_value=24.0, value=8.0, step=0.25)
            sched_params["duration"] = st.number_input("Pulse duration (h)", min_value=0.1, max_value=24.0, value=2.0, step=0.25)
            sched_params["period"] = st.number_input("Repeat every (h)", min_value=0.0, max_value=168.0, value=24.0, step=1.0)
            sched_params["baseline"] = st.number_input("Baseline Lux", min_value=0.0, value=0.0, step=10.0)

        with st.expander("Advanced"):
            equilibration_reps = st.number_input("Equilibration repetitions", min_value=0, max_value=20, value=2, step=1)
            show_dlmo = st.checkbox("Show DLMO", value=True)
            show_cbt = st.checkbox("Show CBTmin", value=False)
            threshold = st.number_input("Actogram threshold (Lux)", min_value=0.0, value=10.0, step=1.0)
            smooth_sigma = st.number_input("Smooth sigma", min_value=0.0, value=2.0, step=0.5)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        st.caption("Tip: Use the examples on the right to quickly get started.")

    with right:
        st.subheader("Models and Examples")
        chosen_models = st.multiselect("Models", list(MODEL_OPTIONS.keys()), default=["Forger99", "Hannay19"])
        example = st.selectbox(
            "Examples",
            [
                "None",
                "Shift worker (3 nights on / 2 off)",
                "Social jetlag (5+2)",
                "Single pulse at 20:00",
            ],
            index=0,
        )

        if example != "None":
            if example == "Shift worker (3 nights on / 2 off)":
                schedule_name = "ShiftWork"
                sched_params = {"lux": 300.0, "days_on": 3, "days_off": 2}
            elif example == "Social jetlag (5+2)":
                schedule_name = "SocialJetlag"
                sched_params = {"lux": 150.0, "num_regular_days": 5, "late_bedtime": 1.0, "late_waketime": 2.0}
            elif example == "Single pulse at 20:00":
                schedule_name = "Custom Pulse"
                sched_params = {"lux": 800.0, "start": 20.0, "duration": 2.0, "period": 24.0, "baseline": 0.0}

        # Compute
        time_arr = generate_time_series(total_days=total_days, step_hours=float(step_hours))
        light_arr = compute_light(schedule_name, time_arr, sched_params)

        # Equilibrate and integrate
        plots = []
        for model_name in chosen_models:
            model, x0 = equilibrate_model(model_name, time_arr, light_arr, equilibration_reps)
            model, traj = integrate_model(model_name, time_arr, light_arr, x0)

            fig, ax = plt.subplots(figsize=(10, 5))
            acto = Actogram(time_arr, light_vals=light_arr, ax=ax, threshold=threshold, opacity=0.9, smooth=True, sigma=[smooth_sigma, smooth_sigma])
            if show_dlmo:
                dlmo = model.dlmos(traj)
                acto.plot_phasemarker(dlmo, color=PRIMARY_COLOR)
            if show_cbt:
                cbt = model.cbt(traj)
                acto.plot_phasemarker(cbt, color=ACCENT_COLOR)
            ax.set_title(f"{model_name}")
            st.pyplot(fig, use_container_width=True)

# -----------------------------
# Tab 2: Wearable Data
# -----------------------------
with tabs[1]:
    st.subheader("Wearable Data Actogram")
    st.caption("Upload CSV or JSON that matches the package reader formats. Example files are in circadian/sample_data.")

    col_u1, col_u2 = st.columns([0.6, 0.4])
    with col_u1:
        upload_type = st.radio("File type", ["CSV", "JSON"], horizontal=True)
        file = st.file_uploader("Upload file", type=["csv", "json"]) 
        threshold_u = st.number_input("Actogram threshold (Lux or proxy)", min_value=0.0, value=1.0, step=0.5)
        smooth_sigma_u = st.number_input("Smooth sigma", min_value=0.0, value=0.5, step=0.5)
    with col_u2:
        demo_choice = st.selectbox("Or load an example", ["None", "sample_actiwatch.csv", "steps_data.csv", "hr_data.csv"], index=0)
        run_plot = st.button("Plot Actogram", type="primary", use_container_width=True)

    def _load_demo(path_name: str):
        base = os.path.join("circadian", "sample_data")
        path = os.path.join(base, path_name)
        if path_name.endswith(".json"):
            return load_json(path)
        else:
            return load_csv(path)

    df = None
    if file is not None:
        try:
            if upload_type == "JSON" or file.name.endswith(".json"):
                tmp_path = os.path.join("/tmp", file.name)
                with open(tmp_path, "wb") as f:
                    f.write(file.read())
                df_dict = load_json(tmp_path)
                # Prefer 'activity' or 'steps' stream
                df = df_dict.get("activity", df_dict.get("steps"))
            else:
                tmp_path = os.path.join("/tmp", file.name)
                with open(tmp_path, "wb") as f:
                    f.write(file.read())
                df = load_csv(tmp_path)
        except Exception as e:
            st.error(f"Failed to load file: {e}")

    if demo_choice != "None":
        try:
            demo_df = _load_demo(demo_choice)
            if isinstance(demo_df, dict):
                df = demo_df.get("activity", demo_df.get("steps"))
            else:
                df = demo_df
        except Exception as e:
            st.error(f"Failed to load demo file: {e}")

    if run_plot and df is not None:
        try:
            # Prepare vectors
            if "datetime" in df.columns:
                df = df.sort_values("datetime")
                t0 = df["datetime"].iloc[0]
                t_hours = (df["datetime"] - t0).dt.total_seconds() / 3600.0
            elif "start" in df.columns and "end" in df.columns:
                df = df.sort_values("start")
                t0 = df["start"].iloc[0]
                t_hours = (df["start"] - t0).dt.total_seconds() / 3600.0
            else:
                st.error("Could not find time columns 'datetime' or 'start'/'end'.")
                st.stop()

            # Pick a signal to plot
            signal_col = None
            for c in ["light_estimate", "activity", "steps", "wake"]:
                if c in df.columns:
                    signal_col = c
                    break
            if signal_col is None:
                st.error("No usable signal column found: expected one of light_estimate, activity, steps, wake")
                st.stop()

            vals = df[signal_col].astype(float).to_numpy()

            fig, ax = plt.subplots(figsize=(10, 5))
            acto = Actogram(t_hours.to_numpy(), vals, ax=ax, threshold=threshold_u, opacity=0.9, smooth=True, sigma=[smooth_sigma_u, smooth_sigma_u])
            ax.set_title(f"Wearable Actogram ({signal_col})")
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to plot actogram: {e}")
    elif run_plot and df is None:
        st.info("Please upload a file or select an example.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr/>
<div style='text-align:center; font-size:0.9rem; color:#666;'>
  Built with <b>circadian</b> and <b>Streamlit</b>
</div>
""", unsafe_allow_html=True)