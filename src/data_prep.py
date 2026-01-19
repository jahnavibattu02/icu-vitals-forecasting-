import json
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def standardize_columns(df: pd.DataFrame, patient_id_col: str, time_col: str, feature_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in [patient_id_col, time_col, *feature_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")
    return df[[patient_id_col, time_col, *feature_cols]].copy()

def sort_by_patient_time(df: pd.DataFrame, patient_id_col: str, time_col: str) -> pd.DataFrame:
    return df.sort_values([patient_id_col, time_col]).reset_index(drop=True)

def patient_level_split(
    df: pd.DataFrame,
    patient_id_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    patient_ids = df[patient_id_col].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train + n_val]
    test_ids = patient_ids[n_train + n_val:]

    train_df = df[df[patient_id_col].isin(train_ids)].copy()
    val_df = df[df[patient_id_col].isin(val_ids)].copy()
    test_df = df[df[patient_id_col].isin(test_ids)].copy()

    splits = {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}
    return train_df, val_df, test_df, splits

def impute_missing_per_patient(df: pd.DataFrame, patient_id_col: str, feature_cols: List[str]) -> pd.DataFrame:
    # Forward-fill within each patient; then backward-fill; remaining NaNs will be handled later with global medians (train-only)
    df = df.copy()
    df[feature_cols] = (
        df.groupby(patient_id_col, sort=False)[feature_cols]
          .apply(lambda g: g.ffill().bfill())
          .reset_index(level=0, drop=True)
    )
    return df

def compute_train_medians(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    med = {}
    for c in feature_cols:
        med[c] = float(train_df[c].median(skipna=True))
    return med

def fill_remaining_with_medians(df: pd.DataFrame, feature_cols: List[str], medians: Dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    for c in feature_cols:
        df[c] = df[c].fillna(medians[c])
    return df

def compute_scaler_stats(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for c in feature_cols:
        mean = float(train_df[c].mean())
        std = float(train_df[c].std(ddof=0))
        if std == 0.0:
            std = 1.0
        stats[c] = {"mean": mean, "std": std}
    return stats

def apply_zscore(df: pd.DataFrame, feature_cols: List[str], stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = df.copy()
    for c in feature_cols:
        df[c] = (df[c] - stats[c]["mean"]) / stats[c]["std"]
    return df

def save_json(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
