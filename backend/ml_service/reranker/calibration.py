"""Score calibration via Platt scaling (logistic regression).

Maps raw scores to calibrated probabilities so that:
- score 0.7 ≈ 70% probability of being a good match
- threshold becomes meaningful
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PlattCalibrator:
    """Platt scaling: fit sigmoid to map raw scores → calibrated probabilities."""

    def __init__(self) -> None:
        self._a: float = 1.0  # sigmoid slope
        self._b: float = 0.0  # sigmoid offset
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit Platt scaling parameters on validation data.

        scores: raw model scores (e.g., stage1 hybrid scores)
        labels: binary labels (0/1)
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        if len(scores) < 10 or len(np.unique(labels)) < 2:
            logger.warning("Not enough data for calibration (%d samples)", len(scores))
            return

        # Platt scaling: fit a, b such that P(y=1|s) = sigmoid(a*s + b)
        # Using simple gradient descent
        a, b = 1.0, 0.0
        lr = 0.01

        for _ in range(1000):
            z = a * scores + b
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-7, 1 - 1e-7)

            # Cross-entropy gradient
            grad_a = np.mean((p - labels) * scores)
            grad_b = np.mean(p - labels)

            a -= lr * grad_a
            b -= lr * grad_b

        self._a = float(a)
        self._b = float(b)
        self._fitted = True

        # Log calibration quality
        calibrated = self.transform(scores)
        pred = (calibrated >= 0.5).astype(int)
        accuracy = float((pred == labels).mean())
        logger.info("Platt calibration fitted: a=%.3f, b=%.3f, accuracy=%.3f", self._a, self._b, accuracy)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply calibration to raw scores → calibrated probabilities."""
        if not self._fitted:
            return scores
        z = self._a * np.asarray(scores, dtype=np.float64) + self._b
        return 1.0 / (1.0 + np.exp(-z))

    def transform_single(self, score: float) -> float:
        """Calibrate a single score."""
        if not self._fitted:
            return score
        z = self._a * score + self._b
        return float(1.0 / (1.0 + np.exp(-z)))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        with open(path / "calibration.json", "w") as f:
            json.dump({"a": self._a, "b": self._b, "fitted": self._fitted}, f)

    def load(self, path: Path | str) -> None:
        path = Path(path)
        cal_path = path / "calibration.json"
        if cal_path.exists():
            with open(cal_path) as f:
                data = json.load(f)
            self._a = data["a"]
            self._b = data["b"]
            self._fitted = data["fitted"]
            logger.info("Calibration loaded: a=%.3f, b=%.3f", self._a, self._b)
