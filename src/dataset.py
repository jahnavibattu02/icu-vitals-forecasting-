from typing import Tuple, List
import numpy as np
import pandas as pd

def make_windows_multi_step(
    df: pd.DataFrame,
    patient_id_col: str,
    time_col: str,
    feature_cols: List[str],
    window_size: int,
    horizon: int,
    forecast_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-step forecasting:
      X: (N, T, F)
      y: (N, H, F)  where H = forecast_steps
    Target starts at (t + horizon) and spans H steps.

    Example:
      window covers [i, i+T-1]
      y covers [i+T+horizon-1, i+T+horizon-1 + (H-1)]
    """
    X_list, y_list = [], []

    for _, g in df.groupby(patient_id_col, sort=False):
        g = g.sort_values(time_col)
        arr = g[list(feature_cols)].to_numpy(dtype=np.float32)

        min_len = window_size + horizon + forecast_steps - 1
        if arr.shape[0] < min_len:
            continue

        last_start = arr.shape[0] - min_len
        for i in range(0, last_start + 1):
            x = arr[i : i + window_size]
            y_start = i + window_size + horizon - 1
            y = arr[y_start : y_start + forecast_steps]  # (H, F)
            X_list.append(x)
            y_list.append(y)

    if not X_list:
        raise ValueError("No windows created. Reduce window_size/forecast_steps or check data length per patient.")

    X = np.stack(X_list, axis=0)          # (N, T, F)
    y = np.stack(y_list, axis=0)          # (N, H, F)
    return X, y

def naive_last_value_baseline_multi(X: np.ndarray, forecast_steps: int) -> np.ndarray:
    """
    Repeat last observed value for all H future steps:
      X: (N, T, F) -> y_pred: (N, H, F)
    """
    last = X[:, -1, :]                    # (N, F)
    return np.repeat(last[:, None, :], forecast_steps, axis=1)

def mae_rmse_multi(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true/y_pred: (N, H, F)
    Returns MAE/RMSE:
      - per horizon step (H,)
      - per feature (F,)
      - overall mean
    """
    err = y_true - y_pred
    mae_per_step = np.mean(np.abs(err), axis=(0, 2))     # (H,)
    rmse_per_step = np.sqrt(np.mean(err**2, axis=(0, 2)))# (H,)

    mae_per_feature = np.mean(np.abs(err), axis=(0, 1))  # (F,)
    rmse_per_feature = np.sqrt(np.mean(err**2, axis=(0, 1)))

    return {
        "mae_per_step": mae_per_step.tolist(),
        "rmse_per_step": rmse_per_step.tolist(),
        "mae_per_feature": mae_per_feature.tolist(),
        "rmse_per_feature": rmse_per_feature.tolist(),
        "mae_mean": float(np.mean(mae_per_step)),
        "rmse_mean": float(np.mean(rmse_per_step)),
    }
