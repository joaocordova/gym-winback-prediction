"""Data contracts for every table that enters or leaves the winback system.

Two layers of defence (same design as the sibling churn project):

1. **Row contracts** — Pydantic models applied to a deterministic sample of
   each frame.
2. **Frame contracts** — vectorised whole-frame checks: required columns,
   null policy, uniqueness, ranges, and cross-table referential integrity
   (every contact/check-in must reference a known former member; no check-in
   may fall after its member's cancellation).
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Callable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

ROW_SAMPLE_SIZE = 2_000


class DataContractError(ValueError):
    """Raised when a frame violates its declared contract."""

    def __init__(self, table: str, failures: list[str]):
        self.table = table
        self.failures = failures
        bullet_list = "\n".join(f"  - {failure}" for failure in failures)
        super().__init__(f"Data contract violated for '{table}':\n{bullet_list}")


# ---------------------------------------------------------------------------
# Enumerations shared across the system
# ---------------------------------------------------------------------------


class PlanType(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class Gender(str, Enum):
    FEMALE = "female"
    MALE = "male"
    OTHER = "other"


class ReferralSource(str, Enum):
    INSTAGRAM = "instagram"
    FRIEND = "friend"
    WALK_IN = "walk_in"
    CORPORATE = "corporate"
    WEBSITE = "website"


class CancellationReason(str, Enum):
    LOW_USAGE = "low_usage"
    PRICE = "price"
    RELOCATION = "relocation"
    SERVICE = "service"
    HEALTH = "health"
    OTHER = "other"


class Channel(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    SMS = "sms"


class Offer(str, Enum):
    NONE = "none"
    DISCOUNT_20 = "discount_20"
    FREE_MONTH = "free_month"
    PERSONAL_TRAINER = "personal_trainer"


# ---------------------------------------------------------------------------
# Row contracts (one per raw table)
# ---------------------------------------------------------------------------


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FormerMemberRecord(_Row):
    member_id: int = Field(ge=1)
    join_date: date
    cancel_date: date
    cancel_reason: CancellationReason
    plan_type: PlanType
    monthly_fee: float = Field(gt=0, lt=500)
    age: int = Field(ge=14, le=90)
    gender: Gender
    referral_source: ReferralSource


class CheckinRecord(_Row):
    member_id: int = Field(ge=1)
    checkin_date: date
    hour: int = Field(ge=5, le=23)
    is_class: bool


class ContactRecord(_Row):
    contact_id: int = Field(ge=1)
    member_id: int = Field(ge=1)
    contact_date: date
    channel: Channel
    offer: Offer
    reactivated_within_horizon: int = Field(ge=0, le=1)


# ---------------------------------------------------------------------------
# Inference contract (what the scoring API / dashboard accepts)
# ---------------------------------------------------------------------------


class ScoringRequest(_Row):
    """A single winback contact submitted for scoring.

    Ranges bound physically possible values, not the training distribution —
    drift detection is a monitoring concern, not a schema one.
    """

    tenure_days: int = Field(ge=14, le=10_000)
    days_since_cancel: int = Field(ge=1, le=730)
    n_prior_contacts: int = Field(ge=0, le=10)
    visits_30d_precancel: int = Field(ge=0, le=150)
    visits_90d_precancel: int = Field(ge=0, le=450)
    visits_per_week_90d: float = Field(ge=0, le=35)
    visit_decline_ratio: float = Field(ge=0, le=10)
    active_weeks_ratio_12w: float = Field(ge=0, le=1)
    class_ratio_precancel: float = Field(ge=0, le=1)
    monthly_fee: float = Field(gt=0, lt=500)
    age: int = Field(ge=14, le=90)
    cancel_reason: CancellationReason
    plan_type: PlanType
    gender: Gender
    referral_source: ReferralSource
    channel: Channel
    offer: Offer


# ---------------------------------------------------------------------------
# Frame contracts
# ---------------------------------------------------------------------------


class FrameContract:
    """Vectorised, whole-frame validation for one raw table."""

    def __init__(
        self,
        name: str,
        row_model: type[_Row],
        required_columns: list[str],
        unique_key: list[str] | None = None,
        checks: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] | None = None,
    ):
        self.name = name
        self.row_model = row_model
        self.required_columns = required_columns
        self.unique_key = unique_key
        self.checks = checks or []

    def validate(self, frame: pd.DataFrame) -> None:
        failures: list[str] = []

        missing = [c for c in self.required_columns if c not in frame.columns]
        if missing:
            raise DataContractError(self.name, [f"missing columns: {missing}"])

        if frame.empty:
            raise DataContractError(self.name, ["frame is empty"])

        nulls = frame[self.required_columns].isna().sum()
        for column, count in nulls[nulls > 0].items():
            failures.append(f"column '{column}' has {count} null values")

        if self.unique_key is not None:
            dupes = int(frame.duplicated(subset=self.unique_key).sum())
            if dupes:
                failures.append(f"{dupes} duplicate rows on key {self.unique_key}")

        for description, predicate in self.checks:
            bad = int((~predicate(frame)).sum())
            if bad:
                failures.append(f"{bad} rows violate: {description}")

        failures.extend(self._spot_check_rows(frame))

        if failures:
            raise DataContractError(self.name, failures)

    def _spot_check_rows(self, frame: pd.DataFrame) -> list[str]:
        """Validate a deterministic sample of rows against the Pydantic model."""
        sample = frame.head(ROW_SAMPLE_SIZE // 2)
        if len(frame) > len(sample):
            tail = frame.tail(ROW_SAMPLE_SIZE - len(sample))
            sample = pd.concat([sample, tail])
        failures: list[str] = []
        for record in sample.to_dict(orient="records"):
            try:
                self.row_model(**record)
            except Exception as exc:  # noqa: BLE001 — collect, then fail loudly
                failures.append(f"row contract: {exc}")
                if len(failures) >= 5:
                    failures.append("... (further row failures suppressed)")
                    break
        return failures


FORMER_MEMBERS_CONTRACT = FrameContract(
    name="former_members",
    row_model=FormerMemberRecord,
    required_columns=[
        "member_id", "join_date", "cancel_date", "cancel_reason", "plan_type",
        "monthly_fee", "age", "gender", "referral_source",
    ],
    unique_key=["member_id"],
    checks=[
        ("cancel_date after join_date", lambda f: f["cancel_date"] > f["join_date"]),
        ("monthly_fee in (0, 500)", lambda f: (f["monthly_fee"] > 0) & (f["monthly_fee"] < 500)),
        ("cancel_reason is known", lambda f: f["cancel_reason"].isin([r.value for r in CancellationReason])),
    ],
)

CHECKINS_CONTRACT = FrameContract(
    name="checkins",
    row_model=CheckinRecord,
    required_columns=["member_id", "checkin_date", "hour", "is_class"],
    checks=[
        ("hour within operating hours [5, 23]", lambda f: f["hour"].between(5, 23)),
    ],
)

CONTACTS_CONTRACT = FrameContract(
    name="contacts",
    row_model=ContactRecord,
    required_columns=[
        "contact_id", "member_id", "contact_date", "channel", "offer",
        "reactivated_within_horizon",
    ],
    unique_key=["contact_id"],
    checks=[
        ("channel is known", lambda f: f["channel"].isin([c.value for c in Channel])),
        ("offer is known", lambda f: f["offer"].isin([o.value for o in Offer])),
        ("label is binary", lambda f: f["reactivated_within_horizon"].isin([0, 1])),
    ],
)


def validate_dataset(
    former_members: pd.DataFrame,
    checkins: pd.DataFrame,
    contacts: pd.DataFrame,
) -> None:
    """Validate all raw tables plus cross-table referential integrity."""
    FORMER_MEMBERS_CONTRACT.validate(former_members)
    CHECKINS_CONTRACT.validate(checkins)
    CONTACTS_CONTRACT.validate(contacts)

    known = set(former_members["member_id"])
    failures: list[str] = []
    for name, frame in (("checkins", checkins), ("contacts", contacts)):
        orphans = int((~frame["member_id"].isin(known)).sum())
        if orphans:
            failures.append(f"{name}: {orphans} rows reference unknown member_ids")

    cancel = former_members.set_index("member_id")["cancel_date"]
    late_checkins = int(
        (
            pd.to_datetime(checkins["checkin_date"]).values
            > pd.to_datetime(checkins["member_id"].map(cancel)).values
        ).sum()
    )
    if late_checkins:
        failures.append(f"checkins: {late_checkins} check-ins after cancellation")

    early_contacts = int(
        (
            pd.to_datetime(contacts["contact_date"]).values
            <= pd.to_datetime(contacts["member_id"].map(cancel)).values
        ).sum()
    )
    if early_contacts:
        failures.append(f"contacts: {early_contacts} contacts on/before cancel_date")

    if failures:
        raise DataContractError("dataset", failures)
