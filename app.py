import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from src.predict import zscore_transform, zscore_inverse, predict_next

st.set_page_config(page_title="ICU Vitals Forecasting", layout="centered")

st.title("ICU Patient Vitals Forecasting (LSTM)")
st.caption("Educational demo only — not for clinical use. No real patient data is included.")

MODEL_PATH = "artifacts/lstm_model.h5"
SCALER_PATH = "artifacts/scaler.json"
WINDOW_SIZE_DEFAULT = 24

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        scaler = json.load(f)
    return model, scaler

def make_window_from_last_rows(df: pd.DataFrame, feature_cols, window_size: int):
    # expects df has feature_cols in original scale
    df = df[feature_cols].copy()
    if len(df) < window_size:
        # pad by repeating first row
        pad = pd.concat([df.iloc[[0]]] * (window_size - len(df)), ignore_index=True)
        df = pd.concat([pad, df], ignore_index=True)
    else:
        df = df.tail(window_size)
    return df.to_numpy(dtype=np.float32)

model, scaler = load_artifacts()
feature_cols = scaler["feature_cols"]

st.subheader("Input options")

tab1, tab2 = st.tabs(["Upload CSV (recommended)", "Quick sliders (demo)"])

with tab1:
    st.write("Upload a CSV with columns: **HR, MAP, SpO2** (at least 24 rows recommended).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    window_size = st.number_input("Window size (T)", min_value=12, max_value=120, value=WINDOW_SIZE_DEFAULT, step=1)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")
        else:
            window_raw = make_window_from_last_rows(df, feature_cols, int(window_size))
            window_scaled = zscore_transform(window_raw, scaler)

            if st.button("Predict next vitals"):
                y_scaled = predict_next(model, window_scaled)
                y_raw = zscore_inverse(y_scaled, scaler)

                st.success("Prediction complete")
                out = {f"Predicted {c}": float(v) for c, v in zip(feature_cols, y_raw)}
                st.json(out)

                # Plot history + predicted point (original scale)
                fig = plt.figure()
                t = np.arange(window_raw.shape[0])
                for j, c in enumerate(feature_cols):
                    plt.plot(t, window_raw[:, j], label=f"{c} history")
                    plt.plot(window_raw.shape[0], y_raw[j], marker="o", linestyle="None", label=f"{c} predicted")
                plt.title("Past Window + Next-step Prediction")
                plt.xlabel("Time index")
                plt.ylabel("Value")
                plt.legend()
                st.pyplot(fig)

with tab2:
    st.write("This creates a constant window from one set of vitals (for quick demo only).")
    window_size = st.number_input("Window size (T) ", min_value=12, max_value=120, value=WINDOW_SIZE_DEFAULT, step=1, key="win2")

    hr = st.slider("Heart Rate (bpm)", 40, 160, 80)
    map_ = st.slider("Mean Arterial Pressure (mmHg)", 45, 120, 75)
    spo2 = st.slider("SpO₂ (%)", 80, 100, 96)

    if st.button("Predict next vitals (demo)"):
        window_raw = np.repeat(np.array([[hr, map_, spo2]], dtype=np.float32), int(window_size), axis=0)
        window_scaled = zscore_transform(window_raw, scaler)

        y_scaled = predict_next(model, window_scaled)
        y_raw = zscore_inverse(y_scaled, scaler)

        st.success("Prediction complete")
        out = {f"Predicted {c}": float(v) for c, v in zip(feature_cols, y_raw)}
        st.json(out)

st.markdown("---")
st.caption("Disclaimer: This app is for learning and portfolio demonstration. It is not validated for clinical decision-making.")
