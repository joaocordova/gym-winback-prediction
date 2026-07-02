"""Financial translation layer — reactivation probabilities in, dollars out.

Winback campaign economics (all levers in ``configs/config.yaml``):

* Contacting a former member costs ``contact_cost`` (staff time / messaging).
* If the member reactivates, the incentive is redeemed — the offer's cost
  (``offer_costs``) is paid *only on success*.
* A reactivated member pays their fee for ``expected_stay_months`` months.

Profit = Σ over reactivated contacts (fee × stay − offer cost) − outreach
cost. The decision threshold and the contact ranking are optimised against
this quantity — the gym buys recovered recurring revenue, not PR-AUC.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gym_winback.config import AppConfig, BusinessConfig
from gym_winback.logging_utils import get_logger

log = get_logger(module="business")


def _mean_offer_cost(business: BusinessConfig) -> float:
    return float(np.mean(list(business.offer_costs.values())))


def campaign_outcome(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    business: BusinessConfig,
    threshold: float | None = None,
    top_fraction: float | None = None,
    fees: np.ndarray | None = None,
    offer_costs: np.ndarray | None = None,
) -> dict:
    """Economics of contacting members above a threshold or in the top-k%.

    Exactly one of ``threshold`` / ``top_fraction`` must be provided.
    ``fees`` and ``offer_costs`` are optional per-row arrays; the configured
    averages are used when absent.
    """
    if (threshold is None) == (top_fraction is None):
        raise ValueError("provide exactly one of threshold or top_fraction")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if top_fraction is not None:
        n_flagged = max(1, int(round(len(y_prob) * top_fraction)))
        flagged = np.zeros(len(y_prob), dtype=bool)
        flagged[np.argsort(-y_prob)[:n_flagged]] = True
        threshold = float(np.sort(y_prob)[-n_flagged])
    else:
        flagged = y_prob >= threshold

    n_flagged = int(flagged.sum())
    fees = (
        np.asarray(fees)
        if fees is not None
        else np.full(len(y_true), business.avg_monthly_fee)
    )
    offer_costs = (
        np.asarray(offer_costs)
        if offer_costs is not None
        else np.full(len(y_true), _mean_offer_cost(business))
    )

    success = flagged & (y_true == 1)
    n_success = int(success.sum())
    recovered = float(
        (fees[success] * business.expected_stay_months - offer_costs[success]).sum()
    )
    cost = float(n_flagged * business.contact_cost)

    return {
        "threshold": float(threshold),
        "n_scored": int(len(y_true)),
        "n_flagged": n_flagged,
        "flagged_fraction": round(n_flagged / max(len(y_true), 1), 4),
        "reactivations_captured": n_success,
        "precision": round(n_success / max(n_flagged, 1), 4),
        "recall": round(n_success / max(int(y_true.sum()), 1), 4),
        "recovered_revenue": round(recovered, 2),
        "campaign_cost": round(cost, 2),
        "expected_profit": round(recovered - cost, 2),
        "roi": round((recovered - cost) / cost, 2) if cost > 0 else float("inf"),
    }


def profit_curve(
    y_true: np.ndarray, y_prob: np.ndarray, business: BusinessConfig
) -> pd.DataFrame:
    """Expected profit as a function of the decision threshold."""
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.0, 0.995, 200)))
    rows = [
        campaign_outcome(y_true, y_prob, business, threshold=float(t))
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, business: BusinessConfig
) -> tuple[float, dict]:
    """Threshold that maximises expected campaign profit."""
    curve = profit_curve(y_true, y_prob, business)
    best = curve.loc[curve["expected_profit"].idxmax()]
    return float(best["threshold"]), best.to_dict()


def build_business_report(
    test: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    config: AppConfig,
) -> dict:
    """Full financial narrative for the out-of-time contact waves, persisted
    to ``assets/business_impact.json``."""
    business = config.business
    y_true = test["reactivated_within_horizon"].to_numpy()
    fees = test["monthly_fee"].to_numpy()
    offer_costs = test["offer"].map(business.offer_costs).to_numpy(dtype=float)

    at_threshold = campaign_outcome(
        y_true, y_prob, business, threshold=threshold,
        fees=fees, offer_costs=offer_costs,
    )
    top_k = campaign_outcome(
        y_true, y_prob, business, top_fraction=business.top_fraction,
        fees=fees, offer_costs=offer_costs,
    )

    # Baseline: the same number of contacts chosen at random — expected hit
    # rate equals the base reactivation rate.
    base_rate = float(y_true.mean())
    n_random = top_k["n_flagged"]
    random_successes = base_rate * n_random
    random_revenue = random_successes * (
        float(fees.mean()) * business.expected_stay_months
        - float(offer_costs.mean())
    )
    random_profit = random_revenue - n_random * business.contact_cost

    # Blanket baseline: contact everyone (what most gyms actually do).
    blanket = campaign_outcome(
        y_true, y_prob, business, threshold=0.0, fees=fees, offer_costs=offer_costs
    )

    report = {
        "test_window_start": str(config.split.train_end),
        "currency": business.currency,
        "portfolio": {
            "contacts_scored": int(len(test)),
            "former_members": int(test["member_id"].nunique()),
            "observed_reactivation_rate": round(base_rate, 4),
            "avg_monthly_fee": round(float(fees.mean()), 2),
        },
        "campaign_at_optimal_threshold": at_threshold,
        "campaign_top_quintile": top_k,
        "random_targeting_baseline": {
            "n_flagged": n_random,
            "expected_profit": round(random_profit, 2),
        },
        "blanket_campaign_baseline": blanket,
        "model_uplift_vs_random": round(top_k["expected_profit"] - random_profit, 2),
        "assumptions": {
            "expected_stay_months": business.expected_stay_months,
            "contact_cost": business.contact_cost,
            "offer_costs": business.offer_costs,
        },
    }

    assets = Path(config.paths.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "business_impact.json").write_text(json.dumps(report, indent=2))
    log.info(
        "Business report: top-quintile campaign profit {p} {c} "
        "({u} uplift vs random targeting)",
        p=report["campaign_top_quintile"]["expected_profit"],
        c=business.currency,
        u=report["model_uplift_vs_random"],
    )
    return report


def best_offer_per_member(
    scorer,
    base_row: pd.DataFrame,
    business: BusinessConfig,
) -> pd.DataFrame:
    """Score one contact under every offer and rank by expected value.

    EV(offer) = p(reactivate | offer) × (fee × stay − offer cost) − contact
    cost. Returns one row per offer, sorted by EV — the dashboard's
    "which offer should I send?" view.
    """
    rows = []
    for offer, offer_cost in business.offer_costs.items():
        candidate = base_row.copy()
        candidate["offer"] = offer
        p = float(scorer.model.predict_proba(candidate)[0])
        fee = float(candidate["monthly_fee"].iloc[0])
        value = fee * business.expected_stay_months - offer_cost
        rows.append(
            {
                "offer": offer,
                "winback_probability": p,
                "expected_value": p * value - business.contact_cost,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("expected_value", ascending=False)
        .reset_index(drop=True)
    )
