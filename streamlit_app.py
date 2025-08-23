import os
import json
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit.components.v1 import html as st_html

from circadian.lights import LightSchedule
from circadian.models import Forger99, Jewett99, Hannay19, Hannay19TP
from circadian.plots import Actogram
from circadian.metrics import esri
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

# Make the main content container span the full width
st.markdown(
    """
    <style>
    div.block-container {max-width: 100% !important; padding-left: 0.5rem; padding-right: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
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
def compute_light(
    schedule_name: str, time: np.ndarray, params: dict
) -> np.ndarray:
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
def equilibrate_model(
    model_name: str,
    time: np.ndarray,
    light: np.ndarray,
    equilibration_reps: int,
):
    model_cls = MODEL_OPTIONS[model_name]
    model = model_cls()
    return model, model.equilibrate(time, light, equilibration_reps)

@st.cache_data(show_spinner=False)
def integrate_model(
    model_name: str, time: np.ndarray, light: np.ndarray, x0: np.ndarray
):
    model_cls = MODEL_OPTIONS[model_name]
    model = model_cls()
    traj = model(time, x0, light)
    return model, traj

# -----------------------------
# ECharts helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def _get_echarts_js() -> str:
    """Load ECharts from local node_modules; fallback handled at render time."""
    try:
        base_dir = os.path.dirname(__file__)
        js_path = os.path.join(base_dir, "node_modules", "echarts", "dist", "echarts.min.js")
        with open(js_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _render_echarts(option: dict, height: int = 420, key: str = None, width: int = 0, full_bleed: bool = False):
    """Render an ECharts chart via a lightweight HTML component."""
    container_id = f"echarts-container-{int(time.time()*1000)}"
    option_json = json.dumps(option)
    echarts_js = _get_echarts_js()
    if full_bleed:
        container_style = (
            "position:relative;left:50%;right:50%;"
            "width:100vw;margin-left:-50vw;margin-right:-50vw;"
        )
    else:
        container_style = "width:100%;"
    html_str = f"""
    <div id='{container_id}' style='{container_style}height:{height}px;'></div>
    <script>
    {echarts_js}
    (function(){{
        function init(){{
            var el = document.getElementById('{container_id}');
            var chart = echarts.init(el, null, {{renderer: 'canvas'}});
            var option = {option_json};
            chart.setOption(option);
            window.addEventListener('resize', function(){{ chart.resize(); }});
        }}
        if (typeof echarts === 'undefined' || !echarts.init) {{
            var script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js';
            script.onload = init;
            document.head.appendChild(script);
        }} else {{
            init();
        }}
    }})();
    </script>
    """
    st_html(html_str, height=height+10, width=int(width))

def _build_line_option(
    title: str,
    x_label: str,
    y_label: str,
    series: list,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    y_type: str = "value",
) -> dict:
    """Build a configurable multi-series line option for ECharts.
    series: list of dicts with keys: name, data (list of [x,y]), type (default 'line').
    """
    return {
        "title": {"text": title},
        "tooltip": {"trigger": "axis"},
        "legend": {"type": "scroll"},
        "grid": {"left": 40, "right": 20, "top": 40, "bottom": 60},
        "dataZoom": [
            {"type": "inside"},
            {"type": "slider"}
        ],
        "xAxis": {
            "type": "value",
            "name": x_label,
            "min": x_min,
            "max": x_max,
            "axisPointer": {"label": {"show": True}},
        },
        "yAxis": {
            "type": y_type,
            "name": y_label,
            "min": y_min,
            "max": y_max,
            "axisPointer": {"label": {"show": True}},
        },
        "series": [{
            "name": s.get("name", "Series"),
            "type": s.get("type", "line"),
            "showSymbol": s.get("showSymbol", False),
            "smooth": s.get("smooth", True),
            "areaStyle": s.get("areaStyle"),
            "data": s.get("data", []),
            "yAxisIndex": s.get("yAxisIndex", 0),
            "xAxisIndex": s.get("xAxisIndex", 0),
            "markLine": s.get("markLine"),
        } for s in series]
    }

def _build_multi_axis_line_option(
    title: str,
    x_label: str,
    y_axes: list,
    series: list,
    x_min=None,
    x_max=None,
) -> dict:
    """Build a multi y-axis line chart. y_axes is a list of axis dicts."""
    return {
        "title": {"text": title},
        "tooltip": {"trigger": "axis"},
        "legend": {"type": "scroll"},
        "grid": {"left": 60, "right": 70, "top": 50, "bottom": 80},
        "dataZoom": [
            {"type": "inside"},
            {"type": "slider"}
        ],
        "xAxis": {
            "type": "value",
            "name": x_label,
            "min": x_min,
            "max": x_max,
            "axisPointer": {"label": {"show": True}},
        },
        "yAxis": y_axes,
        "series": [{
            "name": s.get("name", "Series"),
            "type": s.get("type", "line"),
            "showSymbol": s.get("showSymbol", False),
            "smooth": s.get("smooth", True),
            "areaStyle": s.get("areaStyle"),
            "data": s.get("data", []),
            "yAxisIndex": s.get("yAxisIndex", 0),
            "xAxisIndex": s.get("xAxisIndex", 0),
            "lineStyle": s.get("lineStyle"),
            "markLine": s.get("markLine"),
            "step": s.get("step"),
        } for s in series]
    }

def _build_actogram_heatmap_option(
    time_hours: np.ndarray,
    light_vals: np.ndarray,
    threshold: float = 10.0,
    bin_hours: float = 0.5,
    title: str = "Actogram (Heatmap)",
    overlay_events: dict = None,
) -> dict:
    """Build an actogram-like heatmap with X = days and Y = Zeitgeber Time.
    overlay_events: mapping name -> list of event times (in hours absolute).
    """
    if len(time_hours) == 0:
        return {}
    t0 = time_hours[0]
    rel_t = time_hours - t0
    total_hours = rel_t[-1]
    num_days = int(np.ceil(total_hours / 24.0))
    num_bins = int(np.round(24.0 / bin_hours))

    # prepare categories (X = Days, Y = Zeitgeber Time)
    x_cats = [f"Day {d+1}" for d in range(num_days)]
    y_cats = [f"{(i*bin_hours):.1f}" for i in range(num_bins)]

    # normalize light (log-scale-like) for coloring
    lv = np.asarray(light_vals).astype(float)
    lv_norm = np.log10(1.0 + lv)
    vmax = float(np.nanmax(lv_norm)) if np.isfinite(lv_norm).any() else 1.0
    if vmax <= 0:
        vmax = 1.0

    # matrix indexed as [bin_idx, day_idx] so that Y corresponds to ZT bins
    heat = np.zeros((num_bins, num_days), dtype=float)
    counts = np.zeros((num_bins, num_days), dtype=float)
    for t, v in zip(rel_t, lv_norm):
        day_idx = int(np.floor(t / 24.0))
        if day_idx < 0 or day_idx >= num_days:
            continue
        tod = t % 24.0
        bin_idx = int(np.floor(tod / bin_hours))
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        heat[bin_idx, day_idx] += v
        counts[bin_idx, day_idx] += 1.0
    with np.errstate(invalid='ignore'):
        heat = np.divide(
            heat,
            counts,
            out=np.zeros_like(heat),
            where=counts > 0,
        )
    # scale to 0..1
    heat_scaled = (heat / vmax).tolist()

    data = []
    for yi in range(num_bins):
        for xi in range(num_days):
            data.append([xi, yi, round(heat_scaled[yi][xi], 4)])

    series = [{
        "name": "Light",
        "type": "heatmap",
        "data": data,
        "progressive": 0,
    }]

    # overlay events (e.g., CBT/DLMO)
    if overlay_events:
        for name, times in overlay_events.items():
            scat = []
            for t_abs in times:
                t = float(t_abs - t0)
                if t < 0 or t > total_hours:
                    continue
                day_idx = int(np.floor(t / 24.0))
                tod = (t % 24.0)
                yi = int(np.floor(tod / bin_hours))
                if yi >= num_bins:
                    yi = num_bins - 1
                scat.append([day_idx, yi])
            series.append({
                "name": name,
                "type": "scatter",
                "symbolSize": 8,
                "itemStyle": {"borderWidth": 0.5},
                "data": scat,
                "tooltip": {"valueFormatter": None},
            })

    option = {
        "title": {"text": title},
        "tooltip": {"position": "top"},
        "grid": {"left": 60, "right": 20, "top": 40, "bottom": 40},
        "xAxis": {
            "type": "category",
            "name": "Days",
            "data": x_cats,
        },
        "yAxis": {
            "type": "category",
            "name": "Zeitgeber Time (h)",
            "data": y_cats,
        },
        "visualMap": {
            "min": 0,
            "max": 1,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 0,
        },
        "series": series,
        "legend": {"type": "scroll"},
        "animation": False,
    }
    return option

def _build_two_row_line_option(
    title_top: str,
    title_bottom: str,
    x_label: str,
    y_label_top: str,
    y_label_bottom: str,
    series_top: list,
    series_bottom: list,
    x_min=None,
    x_max=None,
):
    """Two vertically stacked line charts sharing the x domain."""
    option = {
        "title": [
            {"text": title_top, "left": 10, "top": 5},
            {"text": title_bottom, "left": 10, "top": "52%"},
        ],
        "tooltip": {"trigger": "axis"},
        "legend": {"type": "scroll", "top": 28},
        "grid": [
            {"left": 60, "right": 30, "top": 50, "height": "35%"},
            {"left": 60, "right": 30, "top": "58%", "height": "32%"},
        ],
        "dataZoom": [
            {"type": "inside", "xAxisIndex": [0, 1]},
            {"type": "slider", "xAxisIndex": [0, 1]},
        ],
        "xAxis": [
            {"type": "value", "name": x_label, "min": x_min, "max": x_max},
            {"type": "value", "name": x_label, "min": x_min, "max": x_max, "gridIndex": 1},
        ],
        "yAxis": [
            {"type": "value", "name": y_label_top},
            {"type": "value", "name": y_label_bottom, "gridIndex": 1},
        ],
        "series": [],
    }
    for s in series_top:
        s_new = dict(s)
        s_new["xAxisIndex"] = 0
        s_new["yAxisIndex"] = 0
        option["series"].append(s_new)
    for s in series_bottom:
        s_new = dict(s)
        s_new["xAxisIndex"] = 1
        s_new["yAxisIndex"] = 1
        option["series"].append(s_new)
    return option

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "Simulation", 
    "Wearable Data", 
    "Interactive Charts",
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
# Tab 3: Interactive Charts (ECharts)
# -----------------------------
with tabs[2]:
    st.subheader("Interactive Charts (ECharts)")
    st.caption("Interactive, publication-aligned visualizations with ECharts. Uses your Simulation settings.")

    # Recompute core arrays based on current selections
    time_arr = generate_time_series(total_days=total_days, step_hours=float(step_hours))
    light_arr = compute_light(schedule_name, time_arr, sched_params)

    col_ec1, col_ec2 = st.columns([0.6, 0.4], gap="large")
    with col_ec1:
        chart_type = st.selectbox(
            "Chart type",
            ["Amplitude & Phase", "Actogram (Heatmap)", "ESRI"],
            index=0,
        )
    with col_ec2:
        smooth_lines = st.checkbox("Smooth lines", value=True)
        show_light_overlay = st.checkbox("Show light overlay", value=True)
        show_dlmo_ec = st.checkbox("Overlay DLMO", value=True)
        show_cbt_ec = st.checkbox("Overlay CBTmin", value=False)
    plot_height = st.slider("Plot height (px)", min_value=360, max_value=900, value=560, step=20)

    if chart_type == "Actogram (Heatmap)":
        bin_hours = st.select_slider("Bin size (hours)", options=[0.25, 0.5, 1.0, 2.0], value=0.5)
        # Compute events from first selected model for overlay clarity
        overlay = {}
        if len(chosen_models) > 0:
            model_name = chosen_models[0]
            model, x0 = equilibrate_model(model_name, time_arr, light_arr, equilibration_reps)
            _, traj = integrate_model(model_name, time_arr, light_arr, x0)
            if show_dlmo_ec:
                overlay["DLMO"] = list(model.dlmos(traj))
            if show_cbt_ec:
                overlay["CBTmin"] = list(model.cbt(traj))
        option = _build_actogram_heatmap_option(
            time_arr,
            light_arr,
            threshold=threshold,
            bin_hours=float(bin_hours),
            title="Actogram (Heatmap)",
            overlay_events=overlay,
        )
        _render_echarts(option, height=plot_height)

    elif chart_type == "ESRI":
        col_ea, col_eb, col_ec = st.columns(3)
        with col_ea:
            esri_days = st.number_input("Analysis days", min_value=2, max_value=10, value=4, step=1)
        with col_eb:
            esri_dt = st.select_slider("ESRI dt (h)", options=[0.5, 1.0, 2.0], value=1.0)
        with col_ec:
            init_amp = st.number_input("Initial amplitude", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        try:
            esri_t, esri_vals = esri(time_arr, light_arr, analysis_days=int(esri_days), esri_dt=float(esri_dt), initial_amplitude=float(init_amp))
            series = [{
                "name": "ESRI",
                "data": [[float(t), float(v) if np.isfinite(v) else None] for t, v in zip(esri_t, esri_vals)],
                "type": "line",
                "showSymbol": False,
                "smooth": smooth_lines,
            }]
            # convert to days on x-axis
            if len(esri_t):
                t0 = float(esri_t[0])
            else:
                t0 = 0.0
            series[0]["data"] = [[(float(t)-t0)/24.0, float(v) if np.isfinite(v) else None] for t, v in zip(esri_t, esri_vals)]
            option = _build_line_option("ESRI over time", "Time (days)", "ESRI (a.u.)", series)
            _render_echarts(option, height=plot_height)
        except Exception as e:
            st.error(f"Failed to compute ESRI: {e}")

    else:
        # Amplitude & Phase view
        # Compute for each selected model
        amp_series = []
        phase_series = []
        cbt_times_all = []
        dlmo_times_all = []
        for model_name in chosen_models:
            try:
                model, x0 = equilibrate_model(model_name, time_arr, light_arr, equilibration_reps)
                _, traj = integrate_model(model_name, time_arr, light_arr, x0)
                amp_vals = model.amplitude(traj)
                phi = model.phase(traj)
                # convert phase (rad) to hours in [0,24)
                phi = np.mod(phi, 2.0*np.pi)
                phi_hours = (phi * 12.0 / np.pi)
                amp_series.append({
                    "name": f"{model_name} Amplitude",
                    "data": [[float(t), float(a)] for t, a in zip(time_arr, amp_vals)],
                    "type": "line",
                    "yAxisIndex": 0,
                    "showSymbol": False,
                    "smooth": smooth_lines,
                })
                phase_series.append({
                    "name": f"{model_name} Phase (h)",
                    "data": [[float(t), float(ph)] for t, ph in zip(time_arr, phi_hours)],
                    "type": "line",
                    "yAxisIndex": 1,
                    "showSymbol": False,
                    "smooth": smooth_lines,
                    "lineStyle": {"type": "dashed"},
                })
                if show_cbt_ec:
                    cbt_times = list(model.cbt(traj))
                    cbt_times_all.extend(cbt_times)
                if show_dlmo_ec:
                    dlmo_times = list(model.dlmos(traj))
                    dlmo_times_all.extend(dlmo_times)
            except Exception as e:
                st.warning(f"{model_name} failed: {e}")

        # Build light overlay series (secondary axis, log-like visually)
        light_series = []
        if show_light_overlay:
            light_plot_vals = np.log10(1.0 + np.asarray(light_arr, dtype=float))
            light_series.append({
                "name": "Light (log10(1+lux))",
                "data": [[float(t), float(v)] for t, v in zip(time_arr, light_plot_vals)],
                "type": "line",
                "yAxisIndex": 2,
                "showSymbol": False,
                "smooth": False,
                "areaStyle": {"opacity": 0.25},
                "lineStyle": {"width": 1},
            })

        # Vertical markers for phase markers (markLine on first amplitude series if exists)
        def _mk_markline(times: list, name: str, color: str):
            if not times:
                return None
            return {
                "symbol": ["none", "none"],
                "label": {"show": False},
                "lineStyle": {"type": "dotted", "color": color, "width": 1},
                "data": [{"xAxis": float(t), "name": name} for t in times],
            }

        if amp_series:
            amp_series[0]["markLine"] = _mk_markline(cbt_times_all, "CBTmin", "#20C997") if show_cbt_ec else None
            if amp_series[0].get("markLine") and show_dlmo_ec:
                # ECharts markLine data can be combined
                dl = _mk_markline(dlmo_times_all, "DLMO", "#6C63FF")
                if dl:
                    amp_series[0]["markLine"]["data"].extend(dl["data"])
            elif show_dlmo_ec:
                amp_series[0]["markLine"] = _mk_markline(dlmo_times_all, "DLMO", "#6C63FF")

        y_axes = [
            {"type": "value", "name": "Amplitude (a.u.)"},
            {"type": "value", "name": "Phase (h)", "min": 0, "max": 24},
            {"type": "value", "name": "Light (log10(1+lux))"},
        ]

        # Convert x-axis to days for readability
        time_days = (time_arr - float(time_arr[0])) / 24.0 if len(time_arr) else time_arr
        for s in amp_series + phase_series + light_series:
            s["data"] = [[float(td), y] for (td, y) in zip(time_days, [p[1] for p in s["data"]])]

        # Two-row layout: top amplitude, bottom phase + light
        option = _build_two_row_line_option(
            title_top="Amplitude",
            title_bottom="Phase and Light",
            x_label="Time (days)",
            y_label_top="Amplitude (a.u.)",
            y_label_bottom="Phase (h) / Light",
            series_top=amp_series,
            series_bottom=phase_series + light_series,
            x_min=float(time_days[0]) if len(time_arr) else None,
            x_max=float(time_days[-1]) if len(time_arr) else None,
        )
        _render_echarts(option, height=plot_height)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr/>
<div style='text-align:center; font-size:0.9rem; color:#666;'>
  Built with <b>circadian</b> and <b>Streamlit</b>
</div>
""", unsafe_allow_html=True)