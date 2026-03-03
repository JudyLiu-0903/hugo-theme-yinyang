"""Colab-friendly data pipeline for energy + weather modeling with PyTorch tensors.

This script covers:
1) GPU availability check and `device` setup.
2) Conditional XGBoost installation.
3) CSV loading from /content.
4) Weather aggregation and inner merge with energy data.
5) Linear interpolation for missing values.
6) Lag feature engineering for `price actual`.
7) Feature selection, train/test split (80/20), MinMax scaling.
8) Conversion to PyTorch tensors (kept on CPU; move to GPU in training loop).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


# =============================
# 1) Environment setup (Colab)
# =============================
has_gpu = torch.cuda.is_available()
device = torch.device("cuda" if has_gpu else "cpu")
print(f"GPU available: {has_gpu} | device = {device}")


# Install xgboost if missing
if importlib.util.find_spec("xgboost") is None:
    print("xgboost not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
else:
    print("xgboost already installed.")


@dataclass
class TensorDatasetBundle:
    X_train: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor
    x_scaler: MinMaxScaler
    y_scaler: MinMaxScaler
    processed_df: pd.DataFrame


def _find_time_column(df: pd.DataFrame, preferred: str = "dt_iso") -> str:
    if preferred in df.columns:
        return preferred
    for candidate in ["time", "datetime", "timestamp", "date"]:
        if candidate in df.columns:
            return candidate
    raise KeyError("No valid time column found. Expected one of: dt_iso/time/datetime/timestamp/date")


def build_tensors(
    energy_path: str = "/content/energy_dataset.csv",
    weather_path: str = "/content/weather_features.csv",
) -> TensorDatasetBundle:
    """Load, clean, engineer features, and return train/test tensors (on CPU)."""
    # =============================
    # 2) Data loading
    # =============================
    energy_df = pd.read_csv(energy_path)
    weather_df = pd.read_csv(weather_path)

    energy_time_col = _find_time_column(energy_df)
    weather_time_col = _find_time_column(weather_df)

    energy_df[energy_time_col] = pd.to_datetime(energy_df[energy_time_col], errors="coerce")
    weather_df[weather_time_col] = pd.to_datetime(weather_df[weather_time_col], errors="coerce")

    energy_df = energy_df.dropna(subset=[energy_time_col]).copy()
    weather_df = weather_df.dropna(subset=[weather_time_col]).copy()

    if energy_time_col != "dt_iso":
        energy_df = energy_df.rename(columns={energy_time_col: "dt_iso"})
    if weather_time_col != "dt_iso":
        weather_df = weather_df.rename(columns={weather_time_col: "dt_iso"})

    # =============================
    # 3) Cleaning and aggregation
    # =============================
    weather_agg = (
        weather_df.groupby("dt_iso", as_index=False)
        .agg(
            {
                "temp": "mean",
                "humidity": "mean",
                "pressure": "mean",
                "wind_speed": "max",
            }
        )
        .rename(columns={"wind_speed": "wind_speed_max"})
    )

    merged_df = pd.merge(energy_df, weather_agg, on="dt_iso", how="inner")
    merged_df = merged_df.sort_values("dt_iso").reset_index(drop=True)

    # Linear interpolation for missing values (numeric columns)
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].interpolate(method="linear", limit_direction="both")

    # =============================
    # 4) Feature engineering
    # =============================
    merged_df["price_lag_1"] = merged_df["price actual"].shift(1)
    merged_df["price_lag_24"] = merged_df["price actual"].shift(24)

    feature_cols = [
        "total load forecast",
        "generation wind onshore",
        "generation solar",
        "temp",
        "wind_speed_max",
        "price_lag_1",
        "price_lag_24",
    ]
    target_col = "price actual"

    model_df = merged_df.dropna(subset=feature_cols + [target_col]).copy()

    X = model_df[feature_cols].values
    y = model_df[[target_col]].values

    # =============================
    # 5) Train/test split (80/20)
    # =============================
    split_idx = int(len(model_df) * 0.8)
    X_train_np, X_test_np = X[:split_idx], X[split_idx:]
    y_train_np, y_test_np = y[:split_idx], y[split_idx:]

    # =============================
    # 6) MinMax scaling
    # =============================
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_scaled = x_scaler.fit_transform(X_train_np)
    X_test_scaled = x_scaler.transform(X_test_np)
    y_train_scaled = y_scaler.fit_transform(y_train_np)
    y_test_scaled = y_scaler.transform(y_test_np)

    # =============================
    # 7) Tensor conversion (CPU only)
    # =============================
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    # Intentionally do NOT call .to(device) here.
    return TensorDatasetBundle(
        X_train=X_train_tensor,
        X_test=X_test_tensor,
        y_train=y_train_tensor,
        y_test=y_test_tensor,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        processed_df=model_df,
    )


if __name__ == "__main__":
    print("Run build_tensors() after uploading CSV files to /content in Google Colab.")
