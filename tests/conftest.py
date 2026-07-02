"""Shared fixtures.

``trained_pipeline`` runs the real pipeline (simulate → features → train) on a
small, fast configuration in a temp directory once per test session, so the
integration tests exercise exactly the code paths production uses.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from gym_winback.config import AppConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fast_config(root: Path) -> Path:
    """Write a small/fast variant of the default config with temp paths."""
    payload = yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())
    payload["simulation"]["n_former_members"] = 1500
    payload["training"]["n_search_iter"] = 4
    payload["training"]["cv_folds"] = 2
    for key in payload["paths"]:
        payload["paths"][key] = str(root / Path(payload["paths"][key]).name)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


@pytest.fixture(scope="session")
def fast_config(tmp_path_factory) -> AppConfig:
    root = tmp_path_factory.mktemp("gym_winback_pipeline")
    return load_config(_fast_config(root))


@pytest.fixture(scope="session")
def raw_tables(fast_config):
    from gym_winback.simulation import run_simulation

    return run_simulation(fast_config)


@pytest.fixture(scope="session")
def dataset(fast_config, raw_tables):
    from gym_winback.features import build_dataset

    return build_dataset(fast_config, raw_tables)


@pytest.fixture(scope="session")
def trained_pipeline(fast_config, dataset) -> AppConfig:
    from gym_winback.train import run_training

    run_training(fast_config)
    return fast_config


@pytest.fixture()
def micro_tables():
    """Hand-built raw tables with known, hand-checkable values."""
    from gym_winback.simulation import SimulatedTables

    former_members = pd.DataFrame(
        {
            "member_id": [1, 2, 3],
            "join_date": pd.to_datetime(["2025-01-01", "2025-10-01", "2024-06-15"]),
            "cancel_date": pd.to_datetime(["2026-01-15", "2026-02-01", "2026-03-01"]),
            "cancel_reason": ["price", "relocation", "low_usage"],
            "plan_type": ["monthly", "annual", "monthly"],
            "monthly_fee": [60.0, 45.0, 70.0],
            "age": [30, 45, 22],
            "gender": ["female", "male", "other"],
            "referral_source": ["friend", "website", "instagram"],
        }
    )
    checkins = pd.DataFrame(
        {
            "member_id": [1, 1, 1, 1, 1, 2],
            "checkin_date": pd.to_datetime(
                [
                    "2026-01-05",  # 10 days before m1's cancel  (30d window)
                    "2025-12-20",  # 26 days before              (30d window)
                    "2025-11-20",  # 56 days before              (31–90d bucket)
                    "2025-11-01",  # 75 days before              (31–90d bucket)
                    "2025-09-01",  # 136 days before             (outside 90d)
                    "2025-12-15",  # m2: 48 days before cancel
                ]
            ),
            "hour": [18, 7, 19, 12, 9, 18],
            "is_class": [True, False, False, False, True, False],
        }
    )
    contacts = pd.DataFrame(
        {
            "contact_id": [1, 2, 3, 4],
            "member_id": [1, 2, 3, 3],
            "contact_date": pd.to_datetime(
                ["2026-02-14", "2026-02-21", "2026-03-21", "2026-04-15"]
            ),
            "channel": ["phone", "email", "sms", "email"],
            "offer": ["discount_20", "none", "personal_trainer", "free_month"],
            "reactivated_within_horizon": [1, 0, 0, 0],
        }
    )
    return SimulatedTables(former_members, checkins, contacts)
