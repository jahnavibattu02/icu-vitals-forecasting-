from dataclasses import dataclass

@dataclass
class Config:
    # Data columns
    patient_id_col: str = "patient_id"
    time_col: str = "timestamp"
    feature_cols: tuple = ("HR", "MAP", "SpO2")

    # Windowing
    window_size: int = 24      # T
    horizon: int = 1           # next-step forecasting

    # Splits (by patient_id)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Training
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    patience: int = 6

    # Paths
    artifacts_dir: str = "artifacts"
    model_path: str = "artifacts/lstm_model.h5"
    scaler_path: str = "artifacts/scaler.json"
    splits_path: str = "artifacts/splits.json"
    metrics_path: str = "artifacts/metrics.json"
