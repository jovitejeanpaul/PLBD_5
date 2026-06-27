"""
threshold_classifier.py
========================
Wrapper sklearn léger : modèle + seuil de décision intégré.

Séparé de train_model.py pour que sensor_inference.py (Raspberry Pi)
puisse désérialiser les .joblib SANS importer les bibliothèques
d'entraînement (catboost, xgboost, lightgbm, shap…).
"""

from __future__ import annotations

import numpy as np


class ThresholdClassifier:
    """
    Encapsule un estimateur sklearn et son seuil de décision optimal.

    Le seuil est sérialisé avec le modèle (joblib) — plus besoin de
    le recharger séparément en production.
    """

    def __init__(self, estimator, threshold: float = 0.5) -> None:
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        y_proba = self.predict_proba(X)[:, 1]
        return (y_proba >= self.threshold).astype(int)

    def set_params(self, **params):
        for k, v in params.items():
            if k == "threshold":
                self.threshold = v
            else:
                self.estimator.set_params(**{k: v})
        return self

    def get_params(self, deep: bool = True) -> dict:
        params = {"estimator": self.estimator, "threshold": self.threshold}
        if deep and hasattr(self.estimator, "get_params"):
            for k, v in self.estimator.get_params(deep=True).items():
                params[f"estimator__{k}"] = v
        return params

    @property
    def feature_importances_(self):
        return getattr(self.estimator, "feature_importances_", None)

    @property
    def coef_(self):
        return getattr(self.estimator, "coef_", None)

    def __repr__(self) -> str:
        return (f"ThresholdClassifier("
                f"estimator={self.estimator.__class__.__name__}, "
                f"threshold={self.threshold:.3f})")
