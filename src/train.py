import argparse
import json
import os
from typing import List

import numpy as np
import tensorflow as tf

from config import Config
from data_prep import (
    load_csv, standardize_columns, sort_by_patient_time, patient_level_split,
    impute_missing_per_patient, compute_train_medians, fill_remaining_with_medians,
    compute_scaler_stats, apply_zscore, save_json, ensure_dir
)
from dataset import make_windows_next_step, naive_last_value_baseline, mae_rmse

def build_lstm_model(window_size: int, num_features: int, lr: float) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(window_size, num_features))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(num_features)(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True, help="Path to CSV with patient_id,timestamp,HR,MAP,SpO2")
    parser.add_argument("--window", type=int, default=Config().window_size)
    parser.add_argument("--horizon", type=int, default=Config().horizon)
    parser.add_argument("--epochs", type=int, default=Config().epochs)
    parser.add_argument("--batch", type=int, default=Config().batch_size)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(window_size=args.window, horizon=args.horizon, epochs=args.epochs, batch_size=args.batch)

    # Repro
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    ensure_dir(cfg.artifacts_dir)

    # 1) Load + standardize
    df = load_csv(args.data_csv)
    df = standardize_columns(df, cfg.patient_id_col, cfg.time_col, list(cfg.feature_cols))
    df = sort_by_patient_time(df, cfg.patient_id_col, cfg.time_col)

    # 2) Split by patient
    train_df, val_df, test_df, splits = patient_level_split(
        df, cfg.patient_id_col, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, seed=args.seed
    )
    save_json(splits, cfg.splits_path)

    # 3) Impute within patient (ffill/bfill)
    train_df = impute_missing_per_patient(train_df, cfg.patient_id_col, list(cfg.feature_cols))
    val_df = impute_missing_per_patient(val_df, cfg.patient_id_col, list(cfg.feature_cols))
    test_df = impute_missing_per_patient(test_df, cfg.patient_id_col, list(cfg.feature_cols))

    # 4) Fill remaining NaNs with train medians (no leakage)
    medians = compute_train_medians(train_df, list(cfg.feature_cols))
    train_df = fill_remaining_with_medians(train_df, list(cfg.feature_cols), medians)
    val_df = fill_remaining_with_medians(val_df, list(cfg.feature_cols), medians)
    test_df = fill_remaining_with_medians(test_df, list(cfg.feature_cols), medians)

    # 5) Scale with train-only stats
    scaler = compute_scaler_stats(train_df, list(cfg.feature_cols))
    scaler_payload = {"type": "zscore", "feature_stats": scaler, "train_medians": medians, "feature_cols": list(cfg.feature_cols)}
    save_json(scaler_payload, cfg.scaler_path)

    train_df = apply_zscore(train_df, list(cfg.feature_cols), scaler)
    val_df = apply_zscore(val_df, list(cfg.feature_cols), scaler)
    test_df = apply_zscore(test_df, list(cfg.feature_cols), scaler)

    # 6) Windowing
    X_train, y_train = make_windows_next_step(train_df, cfg.patient_id_col, cfg.time_col, list(cfg.feature_cols), cfg.window_size, cfg.horizon)
    X_val, y_val = make_windows_next_step(val_df, cfg.patient_id_col, cfg.time_col, list(cfg.feature_cols), cfg.window_size, cfg.horizon)
    X_test, y_test = make_windows_next_step(test_df, cfg.patient_id_col, cfg.time_col, list(cfg.feature_cols), cfg.window_size, cfg.horizon)

    # 7) Baseline metrics
    y_pred_base = naive_last_value_baseline(X_test)
    baseline_metrics = mae_rmse(y_test, y_pred_base)

    # 8) Model
    model = build_lstm_model(cfg.window_size, len(cfg.feature_cols), cfg.learning_rate)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(cfg.model_path, monitor="val_loss", save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 9) Evaluate LSTM
    y_pred = model.predict(X_test, batch_size=cfg.batch_size, verbose=0)
    lstm_metrics = mae_rmse(y_test, y_pred)

    metrics = {
        "window_size": cfg.window_size,
        "horizon": cfg.horizon,
        "features": list(cfg.feature_cols),
        "baseline_last_value": baseline_metrics,
        "lstm": lstm_metrics,
        "history_last": {k: float(v[-1]) for k, v in history.history.items() if len(v) > 0},
        "n_windows": {"train": int(X_train.shape[0]), "val": int(X_val.shape[0]), "test": int(X_test.shape[0])},
    }
    save_json(metrics, cfg.metrics_path)

    print("\n✅ Training complete")
    print(f"Model saved to: {cfg.model_path}")
    print(f"Scaler saved to: {cfg.scaler_path}")
    print(f"Metrics saved to: {cfg.metrics_path}")
    print("\nTest baseline MAE(mean):", metrics["baseline_last_value"]["mae_mean"])
    print("Test LSTM MAE(mean):", metrics["lstm"]["mae_mean"])

if __name__ == "__main__":
    main()
