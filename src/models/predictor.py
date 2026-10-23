"""Validated inference using the feature and category schema saved with the model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from src.config import ENCODER_PATH, MODEL_PATH


@lru_cache(maxsize=2)
def _load_artifacts(
    model_path: str,
    encoder_path: str,
    model_mtime: float,
    encoder_mtime: float,
) -> tuple[Any, dict]:
    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("The configured model does not support probability estimates.")
    return model, encoders


class CreditRiskPredictor:
    """Load artifacts once and validate records before inference."""

    def __init__(self, model_path=MODEL_PATH, encoder_path=ENCODER_PATH):
        self.model_path = str(model_path)
        self.encoder_path = str(encoder_path)
        model_mtime = Path(self.model_path).stat().st_mtime
        encoder_mtime = Path(self.encoder_path).stat().st_mtime
        self.model, self.encoders = _load_artifacts(
            self.model_path, self.encoder_path, model_mtime, encoder_mtime
        )
        self.feature_names = list(getattr(self.model, "feature_name_", []))
        if not self.feature_names:
            raise RuntimeError("The model artifact does not expose its expected feature names.")

    def transform(self, customer: Mapping[str, Any]) -> pd.DataFrame:
        if not isinstance(customer, Mapping):
            raise TypeError("An application must be provided as a mapping of feature names to values.")
        missing = [name for name in self.feature_names if name not in customer]
        extra = [name for name in customer if name not in self.feature_names]
        if missing or extra:
            parts = []
            if missing:
                parts.append("missing fields: " + ", ".join(missing))
            if extra:
                parts.append("unexpected fields: " + ", ".join(extra))
            raise ValueError("Invalid application schema (" + "; ".join(parts) + ").")

        frame = pd.DataFrame([{name: customer[name] for name in self.feature_names}])
        for column, encoder in self.encoders.items():
            if column not in frame:
                continue
            value = frame.at[0, column]
            if pd.isna(value):
                raise ValueError(f"{column} is required.")
            if value not in encoder.classes_:
                choices = ", ".join(map(str, encoder.classes_))
                raise ValueError(f"Unsupported {column} value {value!r}. Choose one of: {choices}.")
            frame[column] = encoder.transform(frame[column].astype(str))

        if frame.isna().any().any():
            fields = frame.columns[frame.isna().any()].tolist()
            raise ValueError("Missing values in: " + ", ".join(fields))
        try:
            frame = frame[self.feature_names].apply(pd.to_numeric)
        except (TypeError, ValueError) as exc:
            raise ValueError("Numeric model fields must contain valid numbers.") from exc
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Numeric model fields must be finite numbers.")
        return frame

    def predict(self, customer):
        frame = self.transform(customer)
        prediction = int(self.model.predict(frame)[0])
        classes = list(self.model.classes_)
        probability = float(self.model.predict_proba(frame)[0][classes.index(1)])
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError("The model returned an invalid default probability.")
        return prediction, probability
