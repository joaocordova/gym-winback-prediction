"""Data contracts must accept clean data and reject every corruption mode."""

import pandas as pd
import pytest

from gym_winback.schemas import (
    CONTACTS_CONTRACT,
    DataContractError,
    FORMER_MEMBERS_CONTRACT,
    ScoringRequest,
    validate_dataset,
)


def test_micro_tables_pass_contracts(micro_tables):
    validate_dataset(
        micro_tables.former_members, micro_tables.checkins, micro_tables.contacts
    )


def test_rejects_null_values(micro_tables):
    members = micro_tables.former_members.copy()
    members.loc[0, "monthly_fee"] = None
    with pytest.raises(DataContractError, match="null"):
        FORMER_MEMBERS_CONTRACT.validate(members)


def test_rejects_duplicate_member_ids(micro_tables):
    members = pd.concat(
        [micro_tables.former_members, micro_tables.former_members.iloc[[0]]]
    )
    with pytest.raises(DataContractError, match="duplicate"):
        FORMER_MEMBERS_CONTRACT.validate(members)


def test_rejects_cancel_before_join(micro_tables):
    members = micro_tables.former_members.copy()
    members.loc[0, "cancel_date"] = pd.Timestamp("2020-01-01")
    with pytest.raises(DataContractError, match="cancel_date"):
        FORMER_MEMBERS_CONTRACT.validate(members)


def test_rejects_unknown_offer(micro_tables):
    contacts = micro_tables.contacts.copy()
    contacts.loc[0, "offer"] = "yacht_membership"
    with pytest.raises(DataContractError, match="offer"):
        CONTACTS_CONTRACT.validate(contacts)


def test_rejects_non_binary_label(micro_tables):
    contacts = micro_tables.contacts.copy()
    contacts.loc[0, "reactivated_within_horizon"] = 2
    with pytest.raises(DataContractError):
        CONTACTS_CONTRACT.validate(contacts)


def test_rejects_orphan_contacts(micro_tables):
    contacts = micro_tables.contacts.copy()
    contacts.loc[0, "member_id"] = 999
    with pytest.raises(DataContractError, match="unknown member_ids"):
        validate_dataset(
            micro_tables.former_members, micro_tables.checkins, contacts
        )


def test_rejects_checkins_after_cancel(micro_tables):
    checkins = micro_tables.checkins.copy()
    checkins.loc[0, "checkin_date"] = pd.Timestamp("2026-06-01")
    with pytest.raises(DataContractError, match="after cancellation"):
        validate_dataset(
            micro_tables.former_members, checkins, micro_tables.contacts
        )


def test_rejects_contact_before_cancel(micro_tables):
    contacts = micro_tables.contacts.copy()
    contacts.loc[0, "contact_date"] = pd.Timestamp("2025-01-01")
    with pytest.raises(DataContractError, match="on/before cancel_date"):
        validate_dataset(
            micro_tables.former_members, micro_tables.checkins, contacts
        )


VALID_REQUEST = dict(
    tenure_days=300, days_since_cancel=30, n_prior_contacts=0,
    visits_30d_precancel=3, visits_90d_precancel=18, visits_per_week_90d=1.4,
    visit_decline_ratio=0.5, active_weeks_ratio_12w=0.5,
    class_ratio_precancel=0.2, monthly_fee=62.0, age=31,
    cancel_reason="low_usage", plan_type="monthly", gender="female",
    referral_source="friend", channel="phone", offer="discount_20",
)


def test_scoring_request_accepts_valid_payload():
    ScoringRequest(**VALID_REQUEST)


@pytest.mark.parametrize(
    "field,value",
    [
        ("days_since_cancel", 0),
        ("visit_decline_ratio", -0.1),
        ("cancel_reason", "alien_abduction"),
        ("offer", "yacht_membership"),
        ("channel", "carrier_pigeon"),
        ("monthly_fee", 0.0),
    ],
)
def test_scoring_request_rejects_bad_values(field, value):
    payload = dict(VALID_REQUEST)
    payload[field] = value
    with pytest.raises(Exception):
        ScoringRequest(**payload)
