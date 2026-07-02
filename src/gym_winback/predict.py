"""Inference service: the single entry point every consumer scores through.

``WinbackScorer`` wraps the persisted model artifact + metadata and gives the
Streamlit app, batch jobs and tests one identical code path:

* input rows are validated against :class:`~gym_winback.schemas.ScoringRequest`,
* probabilities come from the calibrated model,
* the deployed profit-optimal threshold and propensity tiers ride along,
* per-contact SHAP explanations and per-member **best-offer selection**
  (expected-value ranking across all offers) are available on demand.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gym_winback.business import best_offer_per_member
from gym_winback.config import BusinessConfig
from gym_winback.explain import ShapExplainer
from gym_winback.features import FEATURE_COLUMNS
from gym_winback.logging_utils import get_logger
from gym_winback.models import CalibratedModel
from gym_winback.schemas import ScoringRequest

log = get_logger(module="predict")


class WinbackScorer:
    def __init__(self, model: CalibratedModel, metadata: dict):
        self.model = model
        self.metadata = metadata
        self.threshold = float(metadata["decision_threshold"])
        self._explainer: ShapExplainer | None = None

    @classmethod
    def load(cls, models_dir: str | Path) -> "WinbackScorer":
        models_dir = Path(models_dir)
        model_path = models_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found — train a model first "
                "(python -m gym_winback.cli train)"
            )
        model = joblib.load(model_path)
        metadata = json.loads((models_dir / "metadata.json").read_text())
        return cls(model, metadata)

    # ------------------------------------------------------------------ #

    def validate(self, frame: pd.DataFrame) -> None:
        missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Scoring input is missing features: {missing}")
        for record in frame[FEATURE_COLUMNS].to_dict(orient="records"):
            ScoringRequest(**record)  # raises pydantic.ValidationError on bad rows

    def propensity_tier(self, probability: np.ndarray) -> np.ndarray:
        return np.where(
            probability >= self.threshold,
            "hot",
            np.where(probability >= self.threshold / 2, "warm", "cold"),
        )

    def score_frame(self, frame: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """Score contacts; returns the input plus probability / flag / tier."""
        if validate:
            self.validate(frame)
        out = frame.copy()
        prob = self.model.predict_proba(frame[FEATURE_COLUMNS])
        out["winback_probability"] = prob
        out["recommended_contact"] = (prob >= self.threshold).astype(int)
        out["propensity_tier"] = self.propensity_tier(prob)
        return out

    def best_offer(self, row: pd.DataFrame, business: BusinessConfig) -> pd.DataFrame:
        """Expected-value ranking of every offer for one contact row."""
        return best_offer_per_member(self, row[FEATURE_COLUMNS], business)

    # ------------------------------------------------------------------ #

    def _get_explainer(self, background: pd.DataFrame) -> ShapExplainer:
        if self._explainer is None:
            self._explainer = ShapExplainer(self.model, background)
        return self._explainer

    def explain_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-contact SHAP contributions on the original business features."""
        explainer = self._get_explainer(frame)
        contributions, _ = explainer.aggregated(frame)
        return contributions
