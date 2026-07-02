"""Correctness of the contact-level feature pipeline, on hand-checkable data.

Every expected value below can be verified by reading the micro fixture in
conftest — including the check-in outside the 90-day pre-cancel window,
which must be excluded from the windowed counts.
"""

import pandas as pd
import pytest

from gym_winback.config import load_config
from gym_winback.features import (
    FEATURE_COLUMNS,
    TARGET,
    build_contact_frame,
    validate_feature_frame,
)


@pytest.fixture()
def frame(micro_tables):
    config = load_config()
    return build_contact_frame(micro_tables, config).set_index("contact_id")


def test_one_row_per_contact(frame, micro_tables):
    assert len(frame) == len(micro_tables.contacts)
    assert set(frame.reset_index()["contact_id"]) == {1, 2, 3, 4}


def test_days_since_cancel(frame):
    # m1 cancelled 2026-01-15, contacted 2026-02-14 -> 30 days.
    assert frame.loc[1, "days_since_cancel"] == 30
    # m2 cancelled 2026-02-01, contacted 2026-02-21 -> 20 days.
    assert frame.loc[2, "days_since_cancel"] == 20


def test_n_prior_contacts_is_cumulative(frame):
    # Member 3 was contacted twice: second contact has one prior touch.
    assert frame.loc[3, "n_prior_contacts"] == 0
    assert frame.loc[4, "n_prior_contacts"] == 1


def test_precancel_visit_windows(frame):
    # Member 1: 2 visits in final 30d, 4 in final 90d (the 136-day-old
    # check-in is outside the window), prior-60d rate = 2/2 = 1.0/30d
    # -> decline ratio = 2 / 1 = 2.0
    assert frame.loc[1, "visits_30d_precancel"] == 2
    assert frame.loc[1, "visits_90d_precancel"] == 4
    assert frame.loc[1, "visit_decline_ratio"] == pytest.approx(2.0)


def test_class_ratio_uses_all_history(frame):
    # Member 1: 5 lifetime check-ins, 2 are classes.
    assert frame.loc[1, "class_ratio_precancel"] == pytest.approx(2 / 5)


def test_member_with_no_checkins_gets_zeros(frame):
    # Member 3 has no check-ins at all.
    assert frame.loc[3, "visits_90d_precancel"] == 0
    assert frame.loc[3, "active_weeks_ratio_12w"] == 0.0
    assert frame.loc[3, "visit_decline_ratio"] == 1.0  # neutral default


def test_tenure_days(frame):
    # m1: 2025-01-01 -> 2026-01-15 = 379 days.
    assert frame.loc[1, "tenure_days"] == 379


def test_labels_carried_through(frame):
    assert frame.loc[1, TARGET] == 1
    assert frame.loc[2, TARGET] == 0


def test_no_nulls_and_schema(frame):
    validate_feature_frame(frame.reset_index())
    assert list(frame.reset_index()[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS


def test_validate_feature_frame_catches_nulls(frame):
    bad = frame.reset_index().copy()
    bad.loc[0, "days_since_cancel"] = None
    with pytest.raises(ValueError, match="integrity"):
        validate_feature_frame(bad)


def test_dataset_split_is_temporal(dataset, fast_config):
    train, test = dataset["train"], dataset["test"]
    cutoff = pd.Timestamp(fast_config.split.train_end)
    assert train["contact_date"].max() <= cutoff
    assert test["contact_date"].min() > cutoff
    for split in (train, test):
        rate = split[TARGET].mean()
        assert 0.02 < rate < 0.40
