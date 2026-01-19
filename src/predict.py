import json
import numpy as np
import tensorflow as tf

def load_scaler(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def zscore_transform(x: np.ndarray, scaler: dict) -> np.ndarray:
    # x shape (..., F) with feature order scaler["feature_cols"]
    stats = scaler["feature_stats"]
    cols = scaler["feature_cols"]
    out = x.copy().astype(np.float32)
    for j, c in enumerate(cols):
        mean = stats[c]["mean"]
        std = stats[c]["std"]
        out[..., j] = (out[..., j] - mean) / std
    return out

def zscore_inverse(x: np.ndarray, scaler: dict) -> np.ndarray:
    stats = scaler["feature_stats"]
    cols = scaler["feature_cols"]
    out = x.copy().astype(np.float32)
    for j, c in enumerate(cols):
        mean = stats[c]["mean"]
        std = stats[c]["std"]
        out[..., j] = out[..., j] * std + mean
    return out

def load_model(model_path: str):
    return tf.keras.models.load_model(model_path)

def predict_next(model, window: np.ndarray) -> np.ndarray:
    """
    window: (T, F) scaled
    returns: (F,) scaled
    """
    x = window[np.newaxis, :, :].astype(np.float32)
    y = model.predict(x, verbose=0)[0]
    return y
