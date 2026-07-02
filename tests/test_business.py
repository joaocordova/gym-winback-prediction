"""The financial math must be exactly right — it is the number executives see."""

import numpy as np
import pytest

from gym_winback.business import campaign_outcome, optimal_threshold, profit_curve
from gym_winback.config import load_config


@pytest.fixture(scope="module")
def business():
    return load_config().business


def test_campaign_outcome_hand_computed(business):
    # 4 contacts: two reactivations at the top scores; threshold 0.5 flags 2.
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.1])
    fees = np.array([60.0, 80.0, 50.0, 50.0])
    offer_costs = np.array([0.0, 37.0, 0.0, 0.0])
    out = campaign_outcome(
        y_true, y_prob, business, threshold=0.5, fees=fees, offer_costs=offer_costs
    )

    assert out["n_flagged"] == 2
    assert out["reactivations_captured"] == 2
    expected_revenue = (
        60.0 * business.expected_stay_months - 0.0
    ) + (80.0 * business.expected_stay_months - 37.0)
    assert out["recovered_revenue"] == pytest.approx(expected_revenue, abs=0.01)
    assert out["campaign_cost"] == pytest.approx(2 * business.contact_cost)
    assert out["expected_profit"] == pytest.approx(
        expected_revenue - 2 * business.contact_cost, abs=0.01
    )


def test_offer_cost_only_paid_on_success(business):
    # The flagged non-reactivator's offer cost must NOT be charged.
    y_true = np.array([0, 1])
    y_prob = np.array([0.9, 0.8])
    fees = np.array([60.0, 60.0])
    offer_costs = np.array([500.0 - 1, 0.0])  # huge offer on the failure
    out = campaign_outcome(
        y_true, y_prob, business, threshold=0.5, fees=fees, offer_costs=offer_costs
    )
    assert out["recovered_revenue"] == pytest.approx(
        60.0 * business.expected_stay_months
    )


def test_top_fraction_targeting(business):
    y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.linspace(0.95, 0.05, 10)
    out = campaign_outcome(y_true, y_prob, business, top_fraction=0.1)
    assert out["n_flagged"] == 1
    assert out["reactivations_captured"] == 1


def test_requires_exactly_one_targeting_mode(business):
    y = np.array([0, 1])
    p = np.array([0.2, 0.8])
    with pytest.raises(ValueError):
        campaign_outcome(y, p, business)
    with pytest.raises(ValueError):
        campaign_outcome(y, p, business, threshold=0.5, top_fraction=0.1)


def test_optimal_threshold_maximises_profit(business):
    rng = np.random.default_rng(0)
    y_prob = rng.beta(1.5, 7, 3000)
    y_true = (rng.random(3000) < y_prob).astype(int)
    threshold, best = optimal_threshold(y_true, y_prob, business)
    curve = profit_curve(y_true, y_prob, business)
    assert best["expected_profit"] == pytest.approx(curve["expected_profit"].max())
    assert 0.0 <= threshold <= 1.0


def test_flagging_nobody_costs_nothing(business):
    y = np.array([0, 1, 0])
    p = np.array([0.1, 0.2, 0.3])
    out = campaign_outcome(y, p, business, threshold=0.99)
    assert out["n_flagged"] == 0
    assert out["expected_profit"] == 0.0
