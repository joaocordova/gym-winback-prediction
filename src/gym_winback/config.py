"""Typed configuration for the winback system.

The single YAML file at ``configs/config.yaml`` is parsed into the Pydantic
models below. Anything invalid (a negative cost, a train cutoff outside the
simulation window, an unknown offer) fails at load time — before any compute
is spent.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFIG_ENV_VAR = "GYM_WINBACK_CONFIG"
_DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"

KNOWN_OFFERS = {"none", "discount_20", "free_month", "personal_trainer"}


class _StrictModel(BaseModel):
    """Reject unknown keys so config typos surface immediately."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_StrictModel):
    name: str
    random_seed: int = Field(ge=0)


class PathsConfig(_StrictModel):
    """All paths are declared relative to the repository root."""

    raw_dir: Path
    processed_dir: Path
    sample_dir: Path
    models_dir: Path
    assets_dir: Path
    img_dir: Path
    experiments_dir: Path
    logs_dir: Path

    def resolved(self, root: Path) -> "PathsConfig":
        return PathsConfig(**{name: root / value for name, value in self})


class SimulationConfig(_StrictModel):
    n_former_members: int = Field(ge=100, le=1_000_000)
    window_start: date
    window_end: date

    @model_validator(mode="after")
    def _window_ordered(self) -> "SimulationConfig":
        if self.window_end <= self.window_start:
            raise ValueError("simulation.window_end must be after window_start")
        return self


class SplitConfig(_StrictModel):
    train_end: date
    label_horizon_days: int = Field(ge=14, le=180)


class TrainingConfig(_StrictModel):
    cv_folds: int = Field(ge=2, le=10)
    n_search_iter: int = Field(ge=1, le=500)
    scoring: str
    validation_fraction: float = Field(gt=0.0, lt=0.5)
    calibration: str

    @field_validator("calibration")
    @classmethod
    def _calibration_known(cls, value: str) -> str:
        if value not in {"auto", "none"}:
            raise ValueError("training.calibration must be 'auto' or 'none'")
        return value


class BusinessConfig(_StrictModel):
    """Financial levers that translate probabilities into dollars."""

    currency: str
    currency_symbol: str
    avg_monthly_fee: float = Field(gt=0)
    contact_cost: float = Field(ge=0)
    expected_stay_months: float = Field(gt=0)
    offer_costs: dict[str, float]
    top_fraction: float = Field(gt=0, le=1)

    @field_validator("offer_costs")
    @classmethod
    def _offers_known(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = set(value) - KNOWN_OFFERS
        if unknown:
            raise ValueError(f"business.offer_costs has unknown offers: {unknown}")
        missing = KNOWN_OFFERS - set(value)
        if missing:
            raise ValueError(f"business.offer_costs is missing offers: {missing}")
        if any(cost < 0 for cost in value.values()):
            raise ValueError("business.offer_costs must be non-negative")
        return value

    def value_per_reactivation(self, monthly_fee: float | None = None) -> float:
        """Revenue gained when one former member reactivates."""
        fee = monthly_fee if monthly_fee is not None else self.avg_monthly_fee
        return fee * self.expected_stay_months


class AppConfig(_StrictModel):
    project: ProjectConfig
    paths: PathsConfig
    simulation: SimulationConfig
    split: SplitConfig
    training: TrainingConfig
    business: BusinessConfig

    @model_validator(mode="after")
    def _split_inside_window(self) -> "AppConfig":
        if not (
            self.simulation.window_start
            < self.split.train_end
            < self.simulation.window_end
        ):
            raise ValueError(
                "split.train_end must fall strictly inside the simulation window"
            )
        return self

    @property
    def root(self) -> Path:
        return REPO_ROOT


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load, validate and path-resolve the system configuration.

    Resolution order: explicit ``path`` argument, the ``GYM_WINBACK_CONFIG``
    environment variable, then the repository default.
    """
    config_path = Path(path or os.environ.get(_CONFIG_ENV_VAR) or _DEFAULT_CONFIG)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    config = AppConfig(**payload)
    return config.model_copy(update={"paths": config.paths.resolved(REPO_ROOT)})


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Cached accessor for the default configuration."""
    return load_config()
