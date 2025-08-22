# Circadian Explorer (Streamlit UI)

A modern Streamlit app to simulate and visualize circadian rhythms using the `circadian` Python library.

## Quickstart

1) Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) Run the app
```bash
streamlit run streamlit_app.py
```

3) Open the browser URL that Streamlit prints (usually http://localhost:8501).

## Features

- Simulation tab:
  - Select from light schedules: Regular, ShiftWork, SlamShift, SocialJetlag, or a Custom Pulse
  - Choose one or more circadian models (Forger99, Hannay19, Hannay19TP, Jewett99)
  - Configure total days, time step, and advanced options (threshold, smoothing, DLMO/CBT markers)
  - Interactive actograms with phase markers

- Wearable Data tab:
  - Upload CSV/JSON or load example wearable files from `circadian/sample_data`
  - Auto-detects time columns and plots Actogram of `light_estimate`, `activity`, `steps`, or `wake`

## Examples

- Shift worker: 3 nights on / 2 off
- Social jetlag: 5 weekdays + 2 weekend days
- Single light pulse at 20:00 repeated daily

## Screenshots

- Simulation actogram with DLMO markers
- Wearable actogram from `sample_actiwatch.csv`

(Add images to `index_files/` and reference here when available.)

## Notes

- This repository already contains the `circadian` library. The UI installs it in editable mode with `-e .`.
- Optional GPU/ML models used in the CLI are not required for the Streamlit app.
- ESRI plotting used by the CLI is not currently surfaced; feel free to extend the app.