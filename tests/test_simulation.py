"""Properties the winback-pilot simulator must guarantee."""

import pandas as pd

from gym_winback.simulation import simulate


def test_simulation_is_deterministic(fast_config):
    a = simulate(fast_config)
    b = simulate(fast_config)
    pd.testing.assert_frame_equal(a.former_members, b.former_members)
    pd.testing.assert_frame_equal(a.contacts, b.contacts)
    assert len(a.checkins) == len(b.checkins)


def test_no_checkins_after_cancellation(raw_tables):
    cancel = raw_tables.former_members.set_index("member_id")["cancel_date"]
    joined = raw_tables.checkins.assign(
        cancel_date=raw_tables.checkins["member_id"].map(cancel)
    )
    assert (joined["checkin_date"] < joined["cancel_date"]).all()


def test_contacts_only_after_cancellation(raw_tables):
    cancel = raw_tables.former_members.set_index("member_id")["cancel_date"]
    joined = raw_tables.contacts.assign(
        cancel_date=raw_tables.contacts["member_id"].map(cancel)
    )
    assert (joined["contact_date"] > joined["cancel_date"]).all()


def test_no_contact_after_reactivation(raw_tables):
    """Once a member reactivates, the campaign stops touching them."""
    contacts = raw_tables.contacts.sort_values(["member_id", "contact_date"])
    reactivated_before = (
        contacts.groupby("member_id")["reactivated_within_horizon"]
        .transform(lambda s: s.shift(1).cumsum().fillna(0))
    )
    assert (reactivated_before == 0).all()


def test_reactivation_rate_is_realistic(raw_tables):
    rate = raw_tables.contacts["reactivated_within_horizon"].mean()
    assert 0.05 < rate < 0.30


def test_relocated_members_rarely_come_back(raw_tables):
    contacts = raw_tables.contacts.merge(
        raw_tables.former_members[["member_id", "cancel_reason"]], on="member_id"
    )
    by_reason = contacts.groupby("cancel_reason")["reactivated_within_horizon"].mean()
    assert by_reason["relocation"] < 0.5 * by_reason["price"]


def test_discount_moves_price_churners(raw_tables):
    """The offer × reason interaction the model is supposed to learn."""
    contacts = raw_tables.contacts.merge(
        raw_tables.former_members[["member_id", "cancel_reason"]], on="member_id"
    )
    price = contacts[contacts["cancel_reason"] == "price"]
    with_discount = price.loc[
        price["offer"] == "discount_20", "reactivated_within_horizon"
    ].mean()
    without = price.loc[price["offer"] == "none", "reactivated_within_horizon"].mean()
    assert with_discount > without


def test_manifest_written(fast_config, raw_tables):
    assert (fast_config.paths.raw_dir / "manifest.json").exists()
