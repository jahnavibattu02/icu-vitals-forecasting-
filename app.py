# ------------- Import libraries ----------------------
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from src.risk import trend_risk_score
from src.predict import zscore_transform, zscore_inverse, predict_multi_step


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="ICU Vitals Forecasting",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- LIGHT CSS (UI only) --------------------
st.markdown(
    """
    <style>
      /* ---------- Animations ---------- */
      @keyframes fadeUp {
        from {
          opacity: 0;
          transform: translateY(12px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      @keyframes softPulse {
        0% { box-shadow: 0 0 0 rgba(255, 0, 80, 0.0); }
        50% { box-shadow: 0 0 12px rgba(255, 0, 80, 0.25); }
        100% { box-shadow: 0 0 0 rgba(255, 0, 80, 0.0); }
      }

      /* ---------- Layout ---------- */
      .hero {
        padding: 1.1rem 1.2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(255,0,80,0.08), rgba(0,180,255,0.08));
        border: 1px solid rgba(255,255,255,0.10);
        animation: fadeUp 0.6s ease-out both;
      }

      .subtle {
        color: rgba(255,255,255,0.70);
        font-size: 0.95rem;
      }

      .chip {
        display:inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.12);
        margin-right: 0.35rem;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.85);
        background: rgba(255,255,255,0.03);
        transition: all 0.25s ease;
      }

      .chip:hover {
        transform: translateY(-2px);
        background: rgba(255,255,255,0.07);
        box-shadow: 0 4px 12px rgba(0, 180, 255, 0.25);
      }

      .card {
        padding: 0.9rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        animation: fadeUp 0.7s ease-out both;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
      }

      .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.35);
      }

      .risk-high {
        animation: softPulse 2s ease-in-out infinite;
        border-color: rgba(255, 80, 80, 0.6);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- CONSTANTS --------------------
MODEL_PATH = "artifacts/lstm_model.h5"
SCALER_PATH = "artifacts/scaler.json"
WINDOW_SIZE_DEFAULT = 24

# -------------------- LOAD ARTIFACTS --------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        scaler = json.load(f)
    return model, scaler


def make_window_from_last_rows(df: pd.DataFrame, feature_cols, window_size: int):
    df = df[feature_cols].copy()
    if len(df) == 0:
        raise ValueError("Uploaded CSV has 0 rows.")
    if len(df) < window_size:
        pad = pd.concat([df.iloc[[0]]] * (window_size - len(df)), ignore_index=True)
        df = pd.concat([pad, df], ignore_index=True)
    else:
        df = df.tail(window_size)
    return df.to_numpy(dtype=np.float32)  # (T, F)


def safe_numeric_check(df: pd.DataFrame, cols):
    bad = []
    for c in cols:
        if c not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            bad.append(c)
    return bad


model, scaler = load_artifacts()
feature_cols = scaler["feature_cols"]

# -------------------- HEADER --------------------
h1, h2 = st.columns([0.72, 0.28], gap="large")
with h1:
    st.markdown(
        f"""
        <div class="hero">
          <h1 style="margin:0;">🫀 ICU Patient Vitals Forecasting</h1>
          <div class="subtle">LSTM multi-step forecasting + trend-based risk alerts</div>
          <div style="margin-top:0.6rem;">
            <span class="chip">Features: {", ".join(feature_cols)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with h2:
    with st.container(border=True):
        st.markdown("### 🧠 Model Information")
        st.write("**Model Type:** LSTM-based Time-Series Forecasting")
        st.write("**Forecast Horizon:** Next 6 time steps")
        st.write("**Input Signals:** Heart Rate (HR), Mean Arterial Pressure (MAP), SpO₂")


st.divider()

# -------------------- SIDEBAR CONTROLS --------------------
st.sidebar.header("⚙️ Controls")
window_size = st.sidebar.slider("Window size (T)", 12, 120, WINDOW_SIZE_DEFAULT, 1)
show_csv_preview = st.sidebar.toggle("Show uploaded CSV preview", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: If you want smoother UI interactions, keep the plot as a single chart (recommended).")

# -------------------- INPUT TABS --------------------
st.subheader("Input options")
tab1, tab2 = st.tabs(["📤 Upload CSV (recommended)", "🎚️ Quick sliders (demo)"])

# -------------------- TAB 1: CSV UPLOAD --------------------
with tab1:
    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="card">
              <b>Expected columns:</b> {", ".join([f"<code>{c}</code>" for c in feature_cols])}<br/>
              <span class="subtle">Tip: At least {WINDOW_SIZE_DEFAULT} rows recommended.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is None:
            st.info("Upload a CSV to generate forecasts.")
        else:
            df = pd.read_csv(uploaded)

            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                st.error(f"Missing columns in CSV: {missing}. Found: {list(df.columns)}")
                st.stop()

            bad_cols = safe_numeric_check(df, feature_cols)
            if bad_cols:
                st.error(
                    f"These columns must be numeric: {bad_cols}. "
                    f"Tip: remove symbols like %, bpm, mmHg, or convert to numbers."
                )
                st.stop()

            window_raw = make_window_from_last_rows(df, feature_cols, int(window_size))
            window_scaled = zscore_transform(window_raw, scaler)

            if st.button("🚀 Predict next vitals", type="primary", use_container_width=True):
                y_scaled = predict_multi_step(model, window_scaled)
                y_raw = zscore_inverse(y_scaled, scaler)

                H = y_raw.shape[0]

                st.success("Prediction complete")

                # --- Next-step metrics ---
                st.markdown("### Next-step predictions")
                mcols = st.columns(len(feature_cols))
                for j, c in enumerate(feature_cols):
                    current = float(window_raw[-1, j])
                    nextv = float(y_raw[0, j])
                    delta = nextv - current
                    mcols[j].metric(c, f"{nextv:.2f}", f"{delta:+.2f}")

                # --- Single chart (all features) ---
                st.markdown("### Forecast chart (all features)")
                fig, ax = plt.subplots()
                T = window_raw.shape[0]
                past_t = np.arange(T)
                fut_t = np.arange(T, T + H)

                for j, c in enumerate(feature_cols):
                    ax.plot(past_t, window_raw[:, j], label=f"{c} past", linewidth=2)
                    ax.plot(fut_t, y_raw[:, j], label=f"{c} predicted", linewidth=2)

                ax.axvline(T - 1, linestyle="--", linewidth=1)
                ax.set_title(f"Multi-step Forecast (next {H} steps)")
                ax.set_xlabel("Time index")
                ax.set_ylabel("Value")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)

                # --- Risk scoring ---
                risk = trend_risk_score(window_raw, y_raw, feature_cols)
                st.markdown("### Trend-based Risk Alert")
                st.write(f"**Risk Level:** {risk['risk_level']} (score={risk['score']})")
                st.write("**Explanation:**")
                for r in risk["reasons"]:
                    st.write(f"- {r}")

    with right:
        st.markdown("### CSV guidance")
        st.code(
            "HR,MAP,SpO2\n"
            "82.1,78.0,96.4\n"
            "81.8,77.6,96.3\n"
            "...",
            language="text",
        )

        if uploaded is not None and show_csv_preview:
            st.markdown("### Uploaded CSV preview")
            st.dataframe(df.tail(12), use_container_width=True)

# -------------------- TAB 2: SLIDERS DEMO --------------------
with tab2:
    st.write("This creates a constant window from one set of vitals (for quick demo only).")

    c1, c2, c3 = st.columns(3)
    hr = c1.slider("Heart Rate (bpm)", 40, 160, 80)
    map_ = c2.slider("Mean Arterial Pressure (mmHg)", 45, 120, 75)
    spo2 = c3.slider("SpO₂ (%)", 80, 100, 96)

    if st.button("🚀 Predict next vitals (demo)", type="primary"):
        window_raw = np.repeat(np.array([[hr, map_, spo2]], dtype=np.float32), int(window_size), axis=0)
        window_scaled = zscore_transform(window_raw, scaler)

        y_scaled = predict_multi_step(model, window_scaled)
        y_raw = zscore_inverse(y_scaled, scaler)

        H = y_raw.shape[0]
        st.success("Prediction complete")

        st.markdown("### Next-step predictions")
        mcols = st.columns(len(feature_cols))
        for j, c in enumerate(feature_cols):
            mcols[j].metric(c, f"{y_raw[0, j]:.2f}", f"{(y_raw[0, j]-window_raw[-1, j]):+.2f}")

        st.markdown("### Forecast chart (all features)")
        fig, ax = plt.subplots()
        T = window_raw.shape[0]
        past_t = np.arange(T)
        fut_t = np.arange(T, T + H)

        for j, c in enumerate(feature_cols):
            ax.plot(past_t, window_raw[:, j], label=f"{c} past", linewidth=2)
            ax.plot(fut_t, y_raw[:, j], label=f"{c} predicted", linewidth=2)

        ax.axvline(T - 1, linestyle="--", linewidth=1)
        ax.set_title(f"Multi-step Forecast (next {H} steps)")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        risk = trend_risk_score(window_raw, y_raw, feature_cols)
        st.markdown("### Trend-based Risk Alert")
        st.write(f"**Risk Level:** {risk['risk_level']} (score={risk['score']})")
        st.write("**Explanation:**")
        for r in risk["reasons"]:
            st.write(f"- {r}")

st.markdown("---")
st.caption("Disclaimer: For educational use only. Not validated for clinical decision-making.")
