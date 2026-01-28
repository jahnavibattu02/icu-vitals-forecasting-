import argparse
import numpy as np
import tensorflow as tf

from src.config import Config
from src.data_prep import (
    load_csv, standardize_columns, sort_by_patient_time, patient_level_split,
    impute_missing_per_patient, compute_train_medians, fill_remaining_with_medians,
    compute_scaler_stats, apply_zscore, save_json, ensure_dir
)
from src.dataset import (
    make_windows_multi_step,
    naive_last_value_baseline_multi,
    mae_rmse_multi
)

import tensorflow as tf

def build_lstm_multi_step(window_size: int, num_features: int, forecast_steps: int, lr: float) -> tf.keras.Model:
    l2 = tf.keras.regularizers.l2(1e-4)

    inp = tf.keras.Input(shape=(window_size, num_features))

    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu", kernel_regularizer=l2)(inp)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="causal", activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.20, recurrent_dropout=0.10, kernel_regularizer=l2)
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=False, dropout=0.15, recurrent_dropout=0.05, kernel_regularizer=l2)
    )(x)

    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(forecast_steps * num_features)(x)
    out = tf.keras.layers.Reshape((forecast_steps, num_features))(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--window", type=int, default=Config().window_size)
    parser.add_argument("--horizon", type=int, default=Config().horizon)
    parser.add_argument("--steps", type=int, default=Config().forecast_steps, help="forecast steps H (e.g., 6 or 12)")
    parser.add_argument("--epochs", type=int, default=Config().epochs)
    parser.add_argument("--batch", type=int, default=Config().batch_size)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(window_size=args.window, horizon=args.horizon, forecast_steps=args.steps, epochs=args.epochs, batch_size=args.batch)

    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    ensure_dir(cfg.artifacts_dir)

    # Load & standardize
    df = load_csv(args.data_csv)
    df = standardize_columns(df, cfg.patient_id_col, cfg.time_col, list(cfg.feature_cols))
    df = sort_by_patient_time(df, cfg.patient_id_col, cfg.time_col)

    # Patient split
    train_df, val_df, test_df, splits = patient_level_split(
        df, cfg.patient_id_col, cfg.train_ratio, cfg.val_ratio, cfg.test_ratio, seed=args.seed
    )
    save_json(splits, cfg.splits_path)

    # Impute per patient
    train_df = impute_missing_per_patient(train_df, cfg.patient_id_col, list(cfg.feature_cols))
    val_df   = impute_missing_per_patient(val_df, cfg.patient_id_col, list(cfg.feature_cols))
    test_df  = impute_missing_per_patient(test_df, cfg.patient_id_col, list(cfg.feature_cols))

    # Fill remaining with train medians
    medians = compute_train_medians(train_df, list(cfg.feature_cols))
    train_df = fill_remaining_with_medians(train_df, list(cfg.feature_cols), medians)
    val_df   = fill_remaining_with_medians(val_df, list(cfg.feature_cols), medians)
    test_df  = fill_remaining_with_medians(test_df, list(cfg.feature_cols), medians)

    # Scale train-only
    scaler = compute_scaler_stats(train_df, list(cfg.feature_cols))
    scaler_payload = {"type": "zscore", "feature_stats": scaler, "train_medians": medians, "feature_cols": list(cfg.feature_cols)}
    save_json(scaler_payload, cfg.scaler_path)

    train_df = apply_zscore(train_df, list(cfg.feature_cols), scaler)
    val_df   = apply_zscore(val_df, list(cfg.feature_cols), scaler)
    test_df  = apply_zscore(test_df, list(cfg.feature_cols), scaler)

    # Multi-step windowing
    X_train, y_train = make_windows_multi_step(train_df, cfg.patient_id_col, cfg.time_col, list(cfg.feature_cols), cfg.window_size, cfg.horizon, cfg.forecast_steps)
    X_val,   y_val   = make_windows_multi_step(val_df,   cfg.patient_id_col, cfg.time_col, list(cfg.feature_cols), cfg.window_size, cfg.horizon, cfg.forecast_steps)
    X_test,  y_test  = make_windows_multi_step(test_df,  cfg.patient_id_col, cfg.time_col, list(cfg.feature_cols), cfg.window_size, cfg.horizon, cfg.forecast_steps)

    # Baseline
    y_pred_base = naive_last_value_baseline_multi(X_test, cfg.forecast_steps)
    baseline_metrics = mae_rmse_multi(y_test, y_pred_base)

    # Model
    model = build_lstm_multi_step(cfg.window_size, len(cfg.feature_cols), cfg.forecast_steps, cfg.learning_rate)

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

    # Evaluate
    y_pred = model.predict(X_test, batch_size=cfg.batch_size, verbose=0)
    lstm_metrics = mae_rmse_multi(y_test, y_pred)

    metrics = {
        "window_size": cfg.window_size,
        "horizon": cfg.horizon,
        "forecast_steps": cfg.forecast_steps,
        "features": list(cfg.feature_cols),
        "baseline_last_value": baseline_metrics,
        "lstm": lstm_metrics,
        "history_last": {k: float(v[-1]) for k, v in history.history.items() if len(v) > 0},
        "n_windows": {"train": int(X_train.shape[0]), "val": int(X_val.shape[0]), "test": int(X_test.shape[0])},
    }
    save_json(metrics, cfg.metrics_path)

    print("\n✅ Multi-step training complete")
    print(f"Model saved to: {cfg.model_path}")
    print(f"Metrics saved to: {cfg.metrics_path}")
    print("Baseline MAE(mean):", baseline_metrics["mae_mean"])
    print("LSTM MAE(mean):", lstm_metrics["mae_mean"])

if __name__ == "__main__":
    main()
