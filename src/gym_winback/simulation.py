"""Synthetic winback-pilot simulator for a multi-location gym chain.

The scenario: over ten months the gym ran a **randomized winback pilot** —
every cancelled member was contacted one or more times with a random channel
(email / phone / sms) and a random offer (none / 20% discount / free month /
personal-trainer session), and reactivations within 60 days were recorded.
Random assignment is what makes offer effects in this data readable — the
model learns response propensity without confounded offer targeting.

Generative story (documented in ``docs/data_generation.md``):

* Reactivation propensity is driven by **why the member left** (price
  churners come back for discounts, relocated members are gone), **how
  engaged they were** before cancelling, **tenure** (habits die hard),
  and it **decays with days since cancellation**.
* Offers interact with the cancellation reason (discount × price-churner,
  personal-trainer × low-usage churner) — the interaction structure a
  gradient-boosted model should exploit and SHAP should reveal.
* Repeat contacts face diminishing returns.

Outputs three raw CSVs (former_members, checkins, contacts) plus a
``manifest.json`` with row counts and generation parameters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gym_winback.config import AppConfig
from gym_winback.logging_utils import get_logger
from gym_winback.schemas import validate_dataset

log = get_logger(module="simulation")

REFERRALS = ["instagram", "friend", "walk_in", "corporate", "website"]
REFERRAL_P = [0.28, 0.30, 0.16, 0.11, 0.15]

PLANS = ["monthly", "quarterly", "annual"]
PLAN_P = [0.60, 0.24, 0.16]
PLAN_FEE_MEAN = {"monthly": 66.0, "quarterly": 58.0, "annual": 49.0}

REASONS = ["low_usage", "price", "relocation", "service", "health", "other"]

# Base 60-day reactivation propensity per cancellation reason (no offer,
# email, ~30 days after cancel, average member).
REASON_BASE_P = {
    "low_usage": 0.14,
    "price": 0.19,
    "service": 0.11,
    "other": 0.09,
    "health": 0.05,
    "relocation": 0.012,
}

CHANNELS = ["email", "phone", "sms"]
CHANNEL_P = [0.5, 0.3, 0.2]
CHANNEL_MULT = {"email": 1.0, "phone": 1.35, "sms": 0.85}

OFFERS = ["none", "discount_20", "free_month", "personal_trainer"]
OFFER_P = [0.25, 0.25, 0.25, 0.25]  # randomized pilot — uniform assignment

REPEAT_CONTACT_DECAY = 0.65   # multiplier per prior contact
RECENCY_DECAY_DAYS = 60.0     # e-folding time of winback propensity


def _offer_multiplier(offer: np.ndarray, reason: np.ndarray) -> np.ndarray:
    """Offer effects, interacting with the cancellation reason."""
    mult = np.ones(len(offer))
    mult[offer == "discount_20"] = 1.4
    mult[(offer == "discount_20") & (reason == "price")] = 2.3
    mult[offer == "free_month"] = 1.6
    mult[(offer == "free_month") & (reason == "price")] = 1.9
    mult[offer == "personal_trainer"] = 1.2
    mult[(offer == "personal_trainer") & (reason == "low_usage")] = 2.2
    return mult


@dataclass(frozen=True)
class SimulatedTables:
    former_members: pd.DataFrame
    checkins: pd.DataFrame
    contacts: pd.DataFrame


def simulate(config: AppConfig) -> SimulatedTables:
    sim = config.simulation
    rng = np.random.default_rng(config.project.random_seed)
    n = sim.n_former_members

    window_start = pd.Timestamp(sim.window_start)
    window_end = pd.Timestamp(sim.window_end)
    window_days = (window_end - window_start).days

    # ------------------------------------------------------------------ #
    # Former members: latent traits, then observables.                    #
    # ------------------------------------------------------------------ #
    engagement = rng.beta(2.0, 2.4, n)          # pre-cancel visit intensity
    plan = rng.choice(PLANS, size=n, p=PLAN_P)
    fee = np.round(
        np.array([PLAN_FEE_MEAN[p] for p in plan]) + rng.normal(0, 6.0, n), 2
    ).clip(25.0, 180.0)

    # Cancellations spread over the window (contacts need outcome room, so
    # cancellations stop 30 days before window_end).
    cancel_offsets = rng.integers(0, max(window_days - 30, 1), n)
    cancel_dates = window_start + pd.to_timedelta(cancel_offsets, unit="D")

    tenure_days = np.round(rng.lognormal(mean=5.3, sigma=0.7, size=n)).astype(int)
    tenure_days = tenure_days.clip(30, 1500)
    join_dates = cancel_dates - pd.to_timedelta(tenure_days, unit="D")

    # Cancellation reason correlates with engagement: disengaged members
    # leave for usage reasons, engaged leavers skew to price/relocation.
    reasons = np.empty(n, dtype=object)
    u = rng.random(n)
    low_engaged = engagement < 0.45
    reasons[low_engaged] = np.where(
        u[low_engaged] < 0.55, "low_usage",
        np.where(u[low_engaged] < 0.75, "price",
                 np.where(u[low_engaged] < 0.85, "service",
                          np.where(u[low_engaged] < 0.93, "other",
                                   np.where(u[low_engaged] < 0.97, "health", "relocation")))),
    )
    high = ~low_engaged
    reasons[high] = np.where(
        u[high] < 0.30, "price",
        np.where(u[high] < 0.48, "relocation",
                 np.where(u[high] < 0.63, "health",
                          np.where(u[high] < 0.78, "service",
                                   np.where(u[high] < 0.90, "other", "low_usage")))),
    )

    former_members = pd.DataFrame(
        {
            "member_id": np.arange(1, n + 1),
            "join_date": pd.to_datetime(join_dates).normalize(),
            "cancel_date": pd.to_datetime(cancel_dates).normalize(),
            "cancel_reason": reasons,
            "plan_type": plan,
            "monthly_fee": fee,
            "age": rng.normal(34, 11, n).clip(16, 75).astype(int),
            "gender": rng.choice(["female", "male", "other"], size=n, p=[0.47, 0.50, 0.03]),
            "referral_source": rng.choice(REFERRALS, size=n, p=REFERRAL_P),
        }
    )

    # ------------------------------------------------------------------ #
    # Pre-cancel check-ins: 26 weekly Poisson counts before the cancel,   #
    # with usage-decay for members who left for usage-driven reasons.     #
    # ------------------------------------------------------------------ #
    n_weeks = 26
    weeks_before = np.arange(n_weeks, 0, -1)  # 26 ... 1 weeks before cancel
    base_rate = 0.3 + 4.6 * engagement
    rate = np.tile(base_rate[:, None], (1, n_weeks)).astype(float)

    usage_driven = np.isin(reasons, ["low_usage", "price", "service", "other"])
    decay_weeks = rng.integers(8, 15, n)
    ratio = np.clip(weeks_before[None, :] / decay_weeks[:, None], 0.0, 1.0)
    decay = np.where(
        usage_driven[:, None] & (weeks_before[None, :] <= decay_weeks[:, None]),
        0.03 + 0.97 * ratio**1.5,
        1.0,
    )
    rate *= decay
    # Only weeks after the member joined produce visits.
    weeks_of_tenure = (tenure_days // 7).clip(0, n_weeks)
    rate *= (weeks_before[None, :] <= weeks_of_tenure[:, None]).astype(float)

    counts = rng.poisson(rate.clip(0, 20))
    member_idx, week_idx = np.nonzero(counts)
    reps = counts[member_idx, week_idx]
    flat_member = np.repeat(former_members["member_id"].values[member_idx], reps)
    week_start_dates = (
        former_members["cancel_date"].values[member_idx]
        - (weeks_before[week_idx]) * np.timedelta64(7, "D")
    )
    flat_dates = np.repeat(week_start_dates, reps) + rng.integers(
        0, 7, reps.sum()
    ) * np.timedelta64(1, "D")

    class_propensity = rng.beta(2.0, 5.0, n)
    flat_class_p = np.repeat(class_propensity[member_idx], reps)
    evening = rng.random(reps.sum()) < 0.55
    checkins = pd.DataFrame(
        {
            "member_id": flat_member,
            "checkin_date": pd.to_datetime(flat_dates).normalize(),
            "hour": np.where(
                evening, rng.normal(18.5, 1.5, reps.sum()), rng.normal(7.5, 1.8, reps.sum())
            ).round().clip(5, 23).astype(int),
            "is_class": rng.random(reps.sum()) < flat_class_p,
        }
    )
    # Clip the few same-week check-ins that would land on/after the cancel.
    cancel_by_member = former_members.set_index("member_id")["cancel_date"]
    checkins = checkins[
        checkins["checkin_date"].values
        < cancel_by_member.loc[checkins["member_id"]].values
    ].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Contacts: 1–3 randomized touches per member, sequential in time.    #
    # A reactivated member receives no further contacts.                  #
    # ------------------------------------------------------------------ #
    tenure_factor = 0.70 + 0.45 * np.log1p(tenure_days / 180.0)
    engage_factor = 0.50 + 1.10 * engagement
    base_p = np.array([REASON_BASE_P[r] for r in reasons])

    rows: list[dict] = []
    contact_id = 1
    n_contacts_per_member = rng.integers(1, 4, n)
    for i in range(n):
        member_cancel = former_members["cancel_date"].iloc[i]
        day = 0
        for k in range(n_contacts_per_member[i]):
            day += int(rng.integers(7, 45))  # waves ~1–6 weeks apart
            contact_date = member_cancel + pd.Timedelta(days=day)
            if contact_date > window_end:
                break
            channel = CHANNELS[rng.choice(len(CHANNELS), p=CHANNEL_P)]
            offer = OFFERS[rng.choice(len(OFFERS), p=OFFER_P)]
            p = (
                base_p[i]
                * tenure_factor[i]
                * engage_factor[i]
                * np.exp(-day / RECENCY_DECAY_DAYS)
                * CHANNEL_MULT[channel]
                * _offer_multiplier(np.array([offer]), np.array([reasons[i]]))[0]
                * (REPEAT_CONTACT_DECAY**k)
            )
            p = min(float(p), 0.75)
            reactivated = int(rng.random() < p)
            rows.append(
                {
                    "contact_id": contact_id,
                    "member_id": int(former_members["member_id"].iloc[i]),
                    "contact_date": contact_date,
                    "channel": channel,
                    "offer": offer,
                    "reactivated_within_horizon": reactivated,
                }
            )
            contact_id += 1
            if reactivated:
                break

    contacts = pd.DataFrame(rows)

    log.info(
        "Simulated pilot: {m} former members, {c} pre-cancel check-ins, "
        "{k} contacts ({r:.1%} reactivation rate)",
        m=len(former_members), c=len(checkins), k=len(contacts),
        r=contacts["reactivated_within_horizon"].mean(),
    )
    return SimulatedTables(former_members, checkins, contacts)


def run_simulation(config: AppConfig) -> SimulatedTables:
    """Simulate, validate against the data contracts, and persist raw CSVs."""
    tables = simulate(config)

    validate_dataset(tables.former_members, tables.checkins, tables.contacts)
    log.info("All raw tables passed their data contracts")

    raw_dir = Path(config.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ("former_members", "checkins", "contacts"):
        frame: pd.DataFrame = getattr(tables, name)
        frame.to_csv(raw_dir / f"{name}.csv", index=False)

    manifest = {
        "generator": "gym_winback.simulation",
        "random_seed": config.project.random_seed,
        "n_former_members": config.simulation.n_former_members,
        "window": [str(config.simulation.window_start), str(config.simulation.window_end)],
        "row_counts": {
            "former_members": len(tables.former_members),
            "checkins": len(tables.checkins),
            "contacts": len(tables.contacts),
        },
        "reactivation_rate": round(
            float(tables.contacts["reactivated_within_horizon"].mean()), 4
        ),
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Raw data written to {dir}", dir=str(raw_dir))
    return tables


def load_raw_tables(config: AppConfig) -> SimulatedTables:
    """Load previously simulated raw tables, re-validating the contracts."""
    raw_dir = Path(config.paths.raw_dir)
    parse = {
        "former_members": ["join_date", "cancel_date"],
        "checkins": ["checkin_date"],
        "contacts": ["contact_date"],
    }
    frames = {
        name: pd.read_csv(raw_dir / f"{name}.csv", parse_dates=dates)
        for name, dates in parse.items()
    }
    validate_dataset(**frames)
    return SimulatedTables(**frames)
