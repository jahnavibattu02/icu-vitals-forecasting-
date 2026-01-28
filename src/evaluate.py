import argparse
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.config import Config
from src.data_prep import (
    load_csv, standardize_columns, sort_by_patient_time,
    impute_missing_per_patient, fill_remaining_with_medians,
    apply_zscore
)
from src.dataset import make_windows_next_step, naive_last_value_baseline, mae_rmse


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=Config().model_path)
    parser.add_argument("--scaler_path", type=str, default=Config().scaler_path)
    parser.add_argument("--window", type=int, default=Config().window_size)
    parser.add_argument("--horizon", type=int, default=Config().horizon)
    parser.add_argument("--save_plot", type=str, default="results/plots/sample_forecast.png")
    args = parser.parse_args()

    cfg = Config(window_size=args.window, horizon=args.horizon)

    scaler_payload = load_json(args.scaler_path)
    stats = scaler_payload["feature_stats"]
    medians = scaler_payload["train_medians"]
    feature_cols = scaler_payload["feature_cols"]

    df = load_csv(args.data_csv)
    df = standardize_columns(df, cfg.patient_id_col, cfg.time_col, feature_cols)
    df = sort_by_patient_time(df, cfg.patient_id_col, cfg.time_col)
    df = impute_missing_per_patient(df, cfg.patient_id_col, feature_cols)
    df = fill_remaining_with_medians(df, feature_cols, medians)
    df = apply_zscore(df, feature_cols, stats)

    X, y = make_windows_next_step(df, cfg.patient_id_col, cfg.time_col, feature_cols, cfg.window_size, cfg.horizon)

    model = tf.keras.models.load_model(args.model_path)
    y_pred = model.predict(X, verbose=0)

    base_pred = naive_last_value_baseline(X)
    m_base = mae_rmse(y, base_pred)
    m_lstm = mae_rmse(y, y_pred)

    print("Baseline MAE mean:", m_base["mae_mean"], "RMSE mean:", m_base["rmse_mean"])
    print("LSTM MAE mean:", m_lstm["mae_mean"], "RMSE mean:", m_lstm["rmse_mean"])

    # Plot one example window
    os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
    idx = 0
    x0 = X[idx]
    yt = y[idx]
    yp = y_pred[idx]

    plt.figure()
    for fi, name in enumerate(feature_cols):
        # plot the window history and then next true/pred point
        t = np.arange(cfg.window_size + 1)
        series = np.concatenate([x0[:, fi], [yt[fi]]])
        plt.plot(t, series, label=f"{name} (true next)")
        plt.plot(cfg.window_size, yp[fi], marker="o", linestyle="None", label=f"{name} (pred next)")

    plt.title("Sample: Past Window + Next-step Forecast (scaled)")
    plt.xlabel("Time index")
    plt.ylabel("Scaled value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_plot, dpi=160)
    print(f"Saved plot to: {args.save_plot}")

if __name__ == "__main__":
    main()
