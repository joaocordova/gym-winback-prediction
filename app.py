"""Gym Winback Studio — interactive win-back scoring & offer optimization.

Run with::

    streamlit run app.py

Three views:

1. **Score a former member** — enter a cancelled member's history and a
   planned contact; get a calibrated winback probability, a SHAP explanation,
   and an **expected-value ranking of every offer** for that member.
2. **Campaign planner** — the scored out-of-time contact list: propensity
   tiers, the ranked outreach queue and expected campaign economics.
3. **Model report** — headline metrics and the interactive evaluation plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from gym_winback.config import get_config
from gym_winback.explain import member_waterfall_figure
from gym_winback.features import FEATURE_COLUMNS
from gym_winback.plotting import SERIES, apply_theme
from gym_winback.predict import WinbackScorer

st.set_page_config(page_title="Gym Winback Studio", layout="wide")

TIER_COLORS = {"cold": SERIES[0], "warm": SERIES[2], "hot": SERIES[1]}

REASONS = ["low_usage", "price", "relocation", "service", "health", "other"]
CHANNELS = ["email", "phone", "sms"]
OFFERS = ["none", "discount_20", "free_month", "personal_trainer"]


@st.cache_resource
def load_scorer() -> WinbackScorer:
    return WinbackScorer.load(get_config().paths.models_dir)


@st.cache_data
def load_sample() -> pd.DataFrame:
    config = get_config()
    return pd.read_csv(Path(config.paths.sample_dir) / "scoring_sample.csv")


@st.cache_data
def load_json(name: str) -> dict:
    config = get_config()
    path = Path(config.paths.assets_dir) / name
    return json.loads(path.read_text()) if path.exists() else {}


def probability_gauge(probability: float, threshold: float) -> go.Figure:
    color = SERIES[1] if probability >= threshold else (
        SERIES[2] if probability >= threshold / 2 else SERIES[0]
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 44}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": color, "thickness": 0.35},
                "threshold": {
                    "line": {"color": "#0b0b0b", "width": 2},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
            title={"text": "60-day winback probability"},
        )
    )
    return apply_theme(fig, height=300, margin=dict(l=30, r=30, t=60, b=10))


def offer_ev_figure(ev: pd.DataFrame, symbol: str) -> go.Figure:
    ordered = ev.sort_values("expected_value")
    fig = go.Figure(
        go.Bar(
            x=ordered["expected_value"],
            y=[o.replace("_", " ") for o in ordered["offer"]],
            orientation="h",
            marker_color=[
                SERIES[1] if v >= 0 else SERIES[5] for v in ordered["expected_value"]
            ],
            customdata=ordered[["winback_probability"]].to_numpy(),
            hovertemplate=(
                "%{y}: EV " + symbol + "%{x:,.2f}"
                "<br>p(winback) %{customdata[0]:.1%}<extra></extra>"
            ),
        )
    )
    return apply_theme(
        fig,
        title="Which offer maximises expected value for this member?",
        xaxis_title=f"Expected value per contact ({symbol})",
        height=320,
        showlegend=False,
    )


def main() -> None:
    config = get_config()
    try:
        scorer = load_scorer()
    except FileNotFoundError:
        st.error(
            "No trained model found. Run the pipeline first:\n\n"
            "```\npython -m gym_winback.cli all\n```"
        )
        st.stop()

    st.title("Gym Winback Studio")
    st.caption(
        f"Model: **{scorer.metadata['model_name']}** · trained "
        f"{scorer.metadata['trained_at'][:10]} · profit-optimal threshold "
        f"**{scorer.threshold:.2f}** · calibrated: "
        f"**{'yes' if scorer.metadata['calibrated'] else 'no'}**"
    )

    tab_score, tab_campaign, tab_model = st.tabs(
        ["Score a former member", "Campaign planner", "Model report"]
    )

    sample = load_sample()

    # ------------------------------------------------------------------ #
    with tab_score:
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.subheader("Former member & planned contact")
            preset = st.selectbox(
                "Start from a real contact (test waves)",
                ["— manual entry —"]
                + [f"contact #{i}" for i in sample["contact_id"].head(30)],
            )
            row = (
                sample[sample["contact_id"] == int(preset.split("#")[1])].iloc[0]
                if preset != "— manual entry —"
                else None
            )

            def default(col: str, fallback):
                return type(fallback)(row[col]) if row is not None else fallback

            cancel_reason = st.selectbox(
                "Cancellation reason", REASONS,
                index=REASONS.index(default("cancel_reason", "low_usage")),
            )
            days_since_cancel = st.slider(
                "Days since cancellation", 1, 300, default("days_since_cancel", 30)
            )
            n_prior_contacts = st.slider(
                "Prior winback contacts", 0, 5, default("n_prior_contacts", 0)
            )
            tenure_days = st.slider(
                "Tenure before cancelling (days)", 30, 1500, default("tenure_days", 300)
            )
            visits_30d = st.slider(
                "Visits — final 30 days before cancel", 0, 40,
                default("visits_30d_precancel", 3),
            )
            visits_90d = st.slider(
                "Visits — final 90 days before cancel", 0, 120,
                default("visits_90d_precancel", 18),
            )
            active_weeks = st.slider(
                "Active weeks — final 12 (share)", 0.0, 1.0,
                float(default("active_weeks_ratio_12w", 0.5)),
            )
            class_ratio = st.slider(
                "Share of visits that were classes", 0.0, 1.0,
                float(default("class_ratio_precancel", 0.2)),
            )
            monthly_fee = st.number_input(
                "Monthly fee", 25.0, 180.0, float(default("monthly_fee", 62.0))
            )
            age = st.slider("Age", 16, 75, default("age", 34))
            plan_type = st.selectbox(
                "Former plan", ["monthly", "quarterly", "annual"],
                index=["monthly", "quarterly", "annual"].index(
                    default("plan_type", "monthly")
                ),
            )
            gender = st.selectbox(
                "Gender", ["female", "male", "other"],
                index=["female", "male", "other"].index(default("gender", "female")),
            )
            referral = st.selectbox(
                "Original acquisition channel",
                ["instagram", "friend", "walk_in", "corporate", "website"],
                index=["instagram", "friend", "walk_in", "corporate", "website"].index(
                    default("referral_source", "friend")
                ),
            )
            channel = st.selectbox(
                "Contact channel", CHANNELS,
                index=CHANNELS.index(default("channel", "phone")),
            )
            offer = st.selectbox(
                "Offer", OFFERS, index=OFFERS.index(default("offer", "discount_20")),
            )

        prior_60_rate = max((visits_90d - visits_30d) / 2.0, 0.0)
        contact = pd.DataFrame(
            [
                {
                    "tenure_days": tenure_days,
                    "days_since_cancel": days_since_cancel,
                    "n_prior_contacts": n_prior_contacts,
                    "visits_30d_precancel": visits_30d,
                    "visits_90d_precancel": visits_90d,
                    "visits_per_week_90d": visits_90d / (90 / 7),
                    "visit_decline_ratio": (
                        min(visits_30d / prior_60_rate, 10.0) if prior_60_rate > 0 else 1.0
                    ),
                    "active_weeks_ratio_12w": active_weeks,
                    "class_ratio_precancel": class_ratio,
                    "monthly_fee": monthly_fee,
                    "age": age,
                    "cancel_reason": cancel_reason,
                    "plan_type": plan_type,
                    "gender": gender,
                    "referral_source": referral,
                    "channel": channel,
                    "offer": offer,
                }
            ]
        )

        with right:
            scored = scorer.score_frame(contact)
            probability = float(scored["winback_probability"].iloc[0])
            tier = scored["propensity_tier"].iloc[0]

            st.plotly_chart(
                probability_gauge(probability, scorer.threshold),
                use_container_width=True,
            )
            badge = {"cold": "COLD", "warm": "WARM", "hot": "HOT"}[tier]
            action = (
                "**Action:** contact this member — expected recovered revenue "
                "exceeds outreach cost at this score."
                if probability >= scorer.threshold
                else "**Action:** deprioritise — outreach cost exceeds expected value."
            )
            st.markdown(f"### {badge} — winback propensity")
            st.markdown(action)

            ev = scorer.best_offer(contact, config.business)
            st.plotly_chart(
                offer_ev_figure(ev, config.business.currency_symbol),
                use_container_width=True,
            )
            best = ev.iloc[0]
            st.markdown(
                f"**Best offer: `{best['offer']}`** — "
                f"p(winback) {best['winback_probability']:.1%}, expected value "
                f"{config.business.currency_symbol}{best['expected_value']:,.2f} per contact."
            )

            contributions = scorer.explain_frame(contact).iloc[0]
            st.plotly_chart(
                member_waterfall_figure(
                    contributions, contact.iloc[0][FEATURE_COLUMNS], probability
                ),
                use_container_width=True,
            )

    # ------------------------------------------------------------------ #
    with tab_campaign:
        scored_all = scorer.score_frame(sample[FEATURE_COLUMNS], validate=False).assign(
            contact_id=sample["contact_id"].values,
            member_id=sample["member_id"].values,
        )
        business = load_json("business_impact.json")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Contacts scored", f"{len(scored_all):,}")
        c2.metric("Recommended contacts", int(scored_all["recommended_contact"].sum()))
        c3.metric(
            "Mean winback probability", f"{scored_all['winback_probability'].mean():.1%}"
        )
        if business:
            c4.metric(
                "Top-quintile campaign profit",
                f"${business['campaign_top_quintile']['expected_profit']:,.0f}",
                help="Expected profit of contacting the top 20% of former members "
                "(see assets/business_impact.json for assumptions).",
            )

        tier_counts = (
            scored_all["propensity_tier"]
            .value_counts()
            .reindex(["cold", "warm", "hot"])
            .fillna(0)
        )
        fig = go.Figure(
            go.Bar(
                x=tier_counts.index.str.title(),
                y=tier_counts.values,
                marker_color=[TIER_COLORS[t] for t in tier_counts.index],
                hovertemplate="%{x}: %{y} contacts<extra></extra>",
            )
        )
        st.plotly_chart(
            apply_theme(fig, title="Propensity tier mix (sample of test waves)", height=340),
            use_container_width=True,
        )

        st.subheader("Ranked winback queue")
        top = scored_all.sort_values("winback_probability", ascending=False).head(15)
        st.dataframe(
            top[
                ["member_id", "winback_probability", "propensity_tier",
                 "cancel_reason", "days_since_cancel", "tenure_days",
                 "offer", "channel", "monthly_fee"]
            ].style.format({"winback_probability": "{:.1%}", "monthly_fee": "${:.0f}"}),
            use_container_width=True,
        )

    # ------------------------------------------------------------------ #
    with tab_model:
        performance = load_json("model_performance.json")
        if performance:
            metrics = performance["metrics"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            c2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
            c3.metric("F1 @ threshold", f"{metrics['f1']:.3f}")
            c4.metric("Recall @ threshold", f"{metrics['recall']:.3f}")
            c5.metric("Lift @ top 10%", f"{metrics['lift_at_10pct']:.1f}×")
            st.caption(
                "All metrics on out-of-time contact waves (after "
                f"{performance['test_window_start']}) — campaigns the model never saw."
            )

        assets_dir = Path(config.paths.assets_dir)
        plot_names = [
            "pr_curve", "roc_curve", "calibration_curve", "confusion_matrix",
            "gains_lift", "profit_curve", "score_distribution", "cohort_winback",
            "offer_effectiveness", "shap_importance", "shap_beeswarm",
        ]
        available = [n for n in plot_names if (assets_dir / f"{n}.html").exists()]
        choice = st.selectbox("Interactive evaluation plots", available)
        if choice:
            components.html(
                (assets_dir / f"{choice}.html").read_text(encoding="utf-8"),
                height=620, scrolling=False,
            )


if __name__ == "__main__":
    main()
