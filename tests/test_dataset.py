"""
test_dataset.py
───────────────
Unit tests for data generation, window extraction, and PyTorch dataset/dataloaders.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import SensorWindowDataset, build_dataloader
from src.utils.preprocessing import extract_windows, fit_scaler, add_rolling_features, SENSOR_COLS

N_FEATURES = 14


def create_dummy_df(n_units: int = 5, cycles_per_unit: int = 50) -> pd.DataFrame:
    """Create a minimal synthetic sensor DataFrame for testing."""
    rows = []
    cols = [
        "vibration_x", "vibration_y", "vibration_z",
        "temperature_bearing", "temperature_ambient",
        "pressure_in", "pressure_out", "rpm", "current_draw",
        "oil_viscosity", "acoustic_emission", "torque", "humidity", "voltage",
    ]
    for uid in range(1, n_units + 1):
        for c in range(cycles_per_unit):
            rul = cycles_per_unit - c - 1
            row = {
                "unit_id": uid,
                "cycle": c,
                "operating_regime": 0,
                "RUL": rul,
                "failure_imminent": int(rul <= 30),
            }
            for col in cols:
                row[col] = float(np.random.randn())
            rows.append(row)
    return pd.DataFrame(rows)


class TestSensorDataset:
    def setup_method(self):
        df = create_dummy_df()
        self.feature_cols = [
            c for c in df.columns
            if c not in {"unit_id", "cycle", "operating_regime", "RUL", "failure_imminent"}
        ]
        self.X, self.y_cls, self.y_rul = extract_windows(
            df, self.feature_cols, window_size=20, stride=5, rul_clip=125.0
        )

    def test_window_shapes_consistent(self):
        assert self.X.shape[0] == self.y_cls.shape[0] == self.y_rul.shape[0]
        assert self.X.shape[1] == 20
        assert self.X.shape[2] == N_FEATURES

    def test_labels_binary_classification(self):
        unique_labels = set(np.unique(self.y_cls))
        assert unique_labels.issubset({0.0, 1.0})

    def test_rul_non_negative(self):
        assert (self.y_rul >= 0.0).all()

    def test_dataloader_batch_output(self):
        loader = build_dataloader(self.X, self.y_cls, self.y_rul, batch_size=4, shuffle=False)
        X_batch, y_cls_batch, y_rul_batch = next(iter(loader))
        assert X_batch.shape == (4, 20, N_FEATURES)
        assert isinstance(X_batch, torch.Tensor)
        assert isinstance(y_cls_batch, torch.Tensor)
        assert isinstance(y_rul_batch, torch.Tensor)


class TestPreprocessingPipeline:
    def test_scaler_fitting_and_transform(self):
        df = create_dummy_df()
        scaler = fit_scaler(df)
        transformed = scaler.transform(df[SENSOR_COLS])
        assert transformed.shape == (len(df), len(SENSOR_COLS))

    def test_rolling_features_expansion(self):
        df = create_dummy_df(n_units=2, cycles_per_unit=30)
        col_count_before = len(df.columns)
        df_roll = add_rolling_features(df, windows=[5, 10])
        assert len(df_roll.columns) > col_count_before

    def test_rolling_no_unit_leakage(self):
        """Verify rolling calculations do not cross machine unit boundaries."""
        df = create_dummy_df(n_units=2, cycles_per_unit=30)
        df_roll = add_rolling_features(df, windows=[5])
        for uid in [1, 2]:
            unit_start = df_roll[df_roll["unit_id"] == uid].iloc[0]
            assert not unit_start.isnull().any(), f"Unexpected NaN in unit {uid} baseline"
