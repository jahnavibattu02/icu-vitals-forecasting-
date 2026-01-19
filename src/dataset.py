from typing import Tuple, List
import numpy as np
import pandas as pd

def make_windows_next_step(
    df: pd.DataFrame,
    patient_id_col: str,
    time_col: str,
    feature_cols: List[str],
    window_size: int,
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each patient sequence, create sliding windows of length `window_size`
    and target = next-step (t + horizon) values of all features.
    Output:
      X: (N, window_size, num_features)
      y: (N, num_features)
    """
    X_list, y_list = [], []

    for pid, g in df.groupby(patient_id_col, sort=False):
        g = g.sort_values(time_col)
        arr = g[list(feature_cols)].to_numpy(dtype=np.float32)

        # Need at least window_size + horizon points
        if arr.shape[0] < window_size + horizon:
            continue

        for i in range(0, arr.shape[0] - window_size - horizon + 1):
            x = arr[i : i + window_size]
            y = arr[i + window_size + horizon - 1]
            X_list.append(x)
            y_list.append(y)

    if not X_list:
        raise ValueError("No windows were created. Check window_size/horizon and data length per patient.")

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y

def naive_last_value_baseline(X: np.ndarray) -> np.ndarray:
    """
    Predict y as the last value in the window for each feature.
    X: (N, T, F) -> y_pred: (N, F)
    """
    return X[:, -1, :]

def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    return {
        "mae_per_feature": mae.tolist(),
        "rmse_per_feature": rmse.tolist(),
        "mae_mean": float(np.mean(mae)),
        "rmse_mean": float(np.mean(rmse)),
    }
