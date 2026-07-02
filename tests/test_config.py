"""The config layer must fail fast on anything invalid."""

import pytest
import yaml

from gym_winback.config import REPO_ROOT, load_config


def _default_payload() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())


def _write(tmp_path, payload) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return str(path)


def test_default_config_loads():
    config = load_config()
    assert config.project.name == "gym-winback-prediction"
    assert config.paths.raw_dir.is_absolute()
    assert (
        config.simulation.window_start
        < config.split.train_end
        < config.simulation.window_end
    )


def test_value_per_reactivation_math():
    business = load_config().business
    assert business.value_per_reactivation() == pytest.approx(
        business.avg_monthly_fee * business.expected_stay_months
    )
    assert business.value_per_reactivation(100.0) == pytest.approx(
        100.0 * business.expected_stay_months
    )


def test_rejects_train_end_outside_window(tmp_path):
    payload = _default_payload()
    payload["split"]["train_end"] = "2030-01-01"
    with pytest.raises(ValueError, match="train_end"):
        load_config(_write(tmp_path, payload))


def test_rejects_unknown_offer(tmp_path):
    payload = _default_payload()
    payload["business"]["offer_costs"]["gold_plated_dumbbell"] = 999.0
    with pytest.raises(ValueError, match="unknown offers"):
        load_config(_write(tmp_path, payload))


def test_rejects_missing_offer(tmp_path):
    payload = _default_payload()
    del payload["business"]["offer_costs"]["free_month"]
    with pytest.raises(ValueError, match="missing offers"):
        load_config(_write(tmp_path, payload))


def test_rejects_negative_offer_cost(tmp_path):
    payload = _default_payload()
    payload["business"]["offer_costs"]["free_month"] = -5.0
    with pytest.raises(ValueError, match="non-negative"):
        load_config(_write(tmp_path, payload))


def test_rejects_unknown_keys(tmp_path):
    payload = _default_payload()
    payload["project"]["typo_key"] = 1
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, payload))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")
