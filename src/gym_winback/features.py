"""Contact-level feature engineering (the "feature store" layer).

The prediction unit is a **winback contact**: one member, contacted on one
date, through one channel, with one offer. Every feature is known at the
moment the contact list is drawn:

* pre-cancellation behaviour is frozen at the cancellation date (it cannot
  change afterwards — the member is gone),
* contact context (days since cancel, prior touches, channel, offer) is known
  by construction.

The label is a reactivation within ``label_horizon_days`` of the contact.
The split is temporal on the contact date: older campaign waves train, the
most recent waves test.

Feature dictionary (see also ``docs/feature_dictionary.md``):

===========================  ==================================================
Feature                      Why it predicts winback
===========================  ==================================================
days_since_cancel            Winback propensity decays fast; a 3-week-old
                             cancellation is warm, a 6-month one is cold.
n_prior_contacts             Diminishing returns on repeat outreach.
tenure_days                  Long-tenure members have gym habits that die
                             hard — they come back.
visits_30d/90d_precancel     How engaged the member still was at the end.
visit_decline_ratio          Final-month usage vs their own prior baseline —
                             distinguishes "faded out" from "left abruptly".
active_weeks_ratio_12w       Consistency of the pre-cancel routine.
class_ratio_precancel        Group classes = social ties = a reason to return.
cancel_reason                The dominant driver: price-churners respond to
                             discounts; relocated members are simply gone.
offer / channel              The campaign levers (randomized in the pilot).
monthly_fee / plan_type      Price sensitivity and commitment context.
age / gender / referral      Demographic and acquisition-channel context.
===========================  ==================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gym_winback.config import AppConfig
from gym_winback.logging_utils import get_logger
from gym_winback.simulation import SimulatedTables, load_raw_tables

log = get_logger(module="features")

TARGET = "reactivated_within_horizon"

NUMERIC_FEATURES = [
    "tenure_days",
    "days_since_cancel",
    "n_prior_contacts",
    "visits_30d_precancel",
    "visits_90d_precancel",
    "visits_per_week_90d",
    "visit_decline_ratio",
    "active_weeks_ratio_12w",
    "class_ratio_precancel",
    "monthly_fee",
    "age",
]

CATEGORICAL_FEATURES = [
    "cancel_reason",
    "plan_type",
    "gender",
    "referral_source",
    "channel",
    "offer",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ID_COLUMNS = ["contact_id", "member_id", "contact_date"]


def _precancel_features(tables: SimulatedTables) -> pd.DataFrame:
    """Per-member behavioural features frozen at the cancellation date."""
    members = tables.former_members.set_index("member_id")
    checkins = tables.checkins.merge(
        members[["cancel_date"]], left_on="member_id", right_index=True
    )
    days_before = (checkins["cancel_date"] - checkins["checkin_date"]).dt.days

    grp = checkins.groupby("member_id")
    out = pd.DataFrame(index=members.index)

    in_30 = checkins[days_before <= 30].groupby("member_id").size()
    in_90 = checkins[days_before <= 90].groupby("member_id").size()
    out["visits_30d_precancel"] = in_30.reindex(out.index).fillna(0).astype(int)
    out["visits_90d_precancel"] = in_90.reindex(out.index).fillna(0).astype(int)
    out["visits_per_week_90d"] = out["visits_90d_precancel"] / (90.0 / 7.0)

    # Final-month usage vs the member's own prior two months (per-30d rate).
    prior_60 = (
        checkins[(days_before > 30) & (days_before <= 90)].groupby("member_id").size()
    )
    prior_rate = (prior_60.reindex(out.index).fillna(0) / 2.0).astype(float)
    out["visit_decline_ratio"] = (
        out["visits_30d_precancel"] / prior_rate.replace(0, np.nan)
    ).fillna(1.0).clip(0, 10)

    # Weekly consistency over the final 12 pre-cancel weeks.
    recent = checkins[days_before <= 84].copy()
    recent["week_bucket"] = (
        (recent["cancel_date"] - recent["checkin_date"]).dt.days // 7
    ).clip(0, 11)
    weekly = recent.groupby(["member_id", "week_bucket"]).size().unstack(fill_value=0)
    out["active_weeks_ratio_12w"] = (
        (weekly > 0).sum(axis=1).reindex(out.index).fillna(0) / 12.0
    )

    total = grp.size()
    classes = grp["is_class"].sum()
    out["class_ratio_precancel"] = (
        (classes / total).reindex(out.index).fillna(0.0)
    )
    return out


def build_contact_frame(tables: SimulatedTables, config: AppConfig) -> pd.DataFrame:
    """Assemble the full contact-level design matrix + label."""
    members = tables.former_members.set_index("member_id")
    behaviour = _precancel_features(tables)

    contacts = tables.contacts.copy()
    contacts["days_since_cancel"] = (
        contacts["contact_date"] - contacts["member_id"].map(members["cancel_date"])
    ).dt.days
    contacts["n_prior_contacts"] = contacts.groupby("member_id").cumcount()

    frame = (
        contacts.merge(
            members[
                ["join_date", "cancel_date", "cancel_reason", "plan_type",
                 "monthly_fee", "age", "gender", "referral_source"]
            ],
            left_on="member_id", right_index=True,
        )
        .merge(behaviour, left_on="member_id", right_index=True)
    )
    frame["tenure_days"] = (frame["cancel_date"] - frame["join_date"]).dt.days

    result = frame[ID_COLUMNS + FEATURE_COLUMNS + [TARGET]].copy()
    log.info(
        "Contact frame: {rows} contacts, reactivation rate {rate:.2%}",
        rows=len(result), rate=result[TARGET].mean(),
    )
    return result


def build_dataset(
    config: AppConfig, tables: SimulatedTables | None = None
) -> dict[str, pd.DataFrame]:
    """Build train/test frames with a temporal split on the contact date."""
    tables = tables or load_raw_tables(config)
    frame = build_contact_frame(tables, config)

    cutoff = pd.Timestamp(config.split.train_end)
    train = frame[frame["contact_date"] <= cutoff].reset_index(drop=True)
    test = frame[frame["contact_date"] > cutoff].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(
            "Temporal split produced an empty train or test set — check "
            "split.train_end against the simulation window"
        )

    processed = Path(config.paths.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed / "train.csv", index=False)
    test.to_csv(processed / "test.csv", index=False)

    sample_dir = Path(config.paths.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    test.sample(
        n=min(200, len(test)), random_state=config.project.random_seed
    ).to_csv(sample_dir / "scoring_sample.csv", index=False)

    log.info(
        "Dataset written: train={tr} contacts (≤ {cut}) / test={te} contacts -> {dir}",
        tr=len(train), cut=str(config.split.train_end), te=len(test),
        dir=str(processed),
    )
    return {"train": train, "test": test}


def load_processed(config: AppConfig) -> dict[str, pd.DataFrame]:
    processed = Path(config.paths.processed_dir)
    out: dict[str, pd.DataFrame] = {}
    for split in ("train", "test"):
        path = processed / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run the features pipeline first "
                "(python -m gym_winback.cli features)"
            )
        out[split] = pd.read_csv(path, parse_dates=["contact_date"])
    return out


def validate_feature_frame(frame: pd.DataFrame, require_target: bool = True) -> None:
    """Integrity gate for the processed feature matrix."""
    problems: list[str] = []
    expected = FEATURE_COLUMNS + ([TARGET] if require_target else [])
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    else:
        if frame[FEATURE_COLUMNS].isna().any().any():
            nulls = frame[FEATURE_COLUMNS].isna().sum()
            problems.append(f"nulls present: {nulls[nulls > 0].to_dict()}")
        if (frame["days_since_cancel"] <= 0).any():
            problems.append("non-positive days_since_cancel")
        if require_target and not set(frame[TARGET].unique()) <= {0, 1}:
            problems.append("target is not binary")
    if problems:
        raise ValueError("Feature frame failed integrity checks: " + "; ".join(problems))
