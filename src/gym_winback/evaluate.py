"""Model evaluation: honest metrics on the out-of-time contact waves + the
full gallery of interactive Plotly reports saved under ``assets/``.

Every figure is generated from contacts made *after* the training cutoff —
campaign waves the model never saw. Charts produced:

* ``roc_curve`` / ``pr_curve`` — discrimination (PR is the headline for a
  ~13% positive class).
* ``calibration_curve`` — do predicted probabilities match reality?
* ``confusion_matrix`` — at the profit-optimal threshold, not 0.5.
* ``gains_lift`` — cumulative gains: "contact the top X%, capture Y% of
  reactivations".
* ``profit_curve`` — expected campaign profit vs threshold (business view).
* ``score_distribution`` — reactivated vs lost score separation.
* ``cohort_winback`` — reactivation rate by cancellation reason × recency.
* ``offer_effectiveness`` — observed reactivation rate by offer × reason
  (readable because the pilot randomized offers).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from gym_winback.business import build_business_report, profit_curve
from gym_winback.config import AppConfig
from gym_winback.features import TARGET, load_processed, validate_feature_frame
from gym_winback.logging_utils import get_logger
from gym_winback.plotting import (
    GRID,
    INK_SECONDARY,
    MUTED,
    SEQUENTIAL_BLUES,
    SERIES,
    save_figure,
    themed_figure,
)

log = get_logger(module="evaluate")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    order = np.argsort(-y_prob)
    top10 = order[: max(1, int(0.10 * len(y_prob)))]
    base_rate = float(np.mean(y_true))
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "precision_at_10pct": float(np.mean(np.asarray(y_true)[top10])),
        "lift_at_10pct": float(np.mean(np.asarray(y_true)[top10]) / base_rate)
        if base_rate > 0
        else 0.0,
        "base_reactivation_rate": base_rate,
        "threshold": float(threshold),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def roc_figure(y_true, y_prob) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig = themed_figure(
        title=f"ROC curve — AUC {auc:.3f} (out-of-time contact waves)",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
    )
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr, mode="lines", name="Model",
            line=dict(color=SERIES[0], width=2),
            hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Chance",
            line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip",
        )
    )
    return fig


def pr_figure(y_true, y_prob, threshold: float) -> go.Figure:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    base = float(np.mean(y_true))
    fig = themed_figure(
        title=f"Precision–recall tradeoff — PR-AUC {ap:.3f}",
        xaxis_title="Recall (reactivations captured)",
        yaxis_title="Precision (contacts that convert)",
    )
    fig.add_trace(
        go.Scatter(
            x=recall, y=precision, mode="lines", name="Model",
            line=dict(color=SERIES[0], width=2),
            hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=base, line=dict(color=MUTED, width=1, dash="dot"),
        annotation_text=f"Base rate {base:.1%}", annotation_font_color=MUTED,
    )
    idx = int(np.argmin(np.abs(thresholds - threshold))) if len(thresholds) else 0
    fig.add_trace(
        go.Scatter(
            x=[recall[idx]], y=[precision[idx]], mode="markers+text",
            name="Deployed threshold", text=[f"t = {threshold:.2f}"],
            textposition="top right", textfont=dict(color=INK_SECONDARY),
            marker=dict(color=SERIES[5], size=11, line=dict(color="#fcfcfb", width=2)),
            hovertemplate=(
                f"Threshold {threshold:.3f}<br>Recall %{{x:.3f}}"
                "<br>Precision %{y:.3f}<extra></extra>"
            ),
        )
    )
    return fig


def calibration_figure(y_true, y_prob, n_bins: int = 10) -> go.Figure:
    bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = 0.0, 1.0
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    frame = pd.DataFrame({"bin": idx, "prob": y_prob, "y": y_true})
    grouped = frame.groupby("bin").agg(
        mean_prob=("prob", "mean"), rate=("y", "mean"), n=("y", "size")
    )
    fig = themed_figure(
        title="Calibration — predicted probability vs observed reactivation rate",
        xaxis_title="Mean predicted probability (per decile bin)",
        yaxis_title="Observed reactivation rate",
    )
    fig.add_trace(
        go.Scatter(
            x=[0, grouped["mean_prob"].max() * 1.05],
            y=[0, grouped["mean_prob"].max() * 1.05],
            mode="lines", name="Perfect calibration",
            line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["mean_prob"], y=grouped["rate"], mode="lines+markers",
            name="Model", line=dict(color=SERIES[0], width=2),
            marker=dict(size=9),
            customdata=grouped["n"].to_numpy(),
            hovertemplate=(
                "Predicted %{x:.3f}<br>Observed %{y:.3f}"
                "<br>%{customdata} contacts<extra></extra>"
            ),
        )
    )
    return fig


def confusion_figure(y_true, y_prob, threshold: float) -> go.Figure:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred)
    labels = ["Stays lost", "Reactivates"]
    fig = themed_figure(
        title=f"Confusion matrix at profit-optimal threshold ({threshold:.2f})",
        xaxis_title="Predicted", yaxis_title="Actual",
    )
    fig.add_trace(
        go.Heatmap(
            z=matrix, x=labels, y=labels,
            colorscale=[[0, SEQUENTIAL_BLUES[0]], [1, SEQUENTIAL_BLUES[-1]]],
            showscale=False,
            text=[[f"{v:,}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=18),
            hovertemplate="Actual %{y} / Predicted %{x}: %{z:,}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def gains_lift_figure(y_true, y_prob) -> go.Figure:
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_prob))
    sorted_y = y_true[order]
    n = len(sorted_y)
    fractions = np.arange(1, 21) / 20.0
    gains, lifts = [], []
    total_pos = sorted_y.sum()
    for f in fractions:
        k = max(1, int(round(f * n)))
        captured = sorted_y[:k].sum()
        gains.append(captured / max(total_pos, 1))
        lifts.append((captured / k) / max(y_true.mean(), 1e-9))

    fig = themed_figure(
        title="Cumulative gains — reactivations captured by contacting the top X%",
        xaxis_title="Share of former members contacted (ranked by model score)",
        yaxis_title="Share of possible reactivations captured",
    )
    fig.add_trace(
        go.Scatter(
            x=fractions, y=gains, mode="lines+markers", name="Model",
            line=dict(color=SERIES[0], width=2), marker=dict(size=7),
            customdata=np.array(lifts).reshape(-1, 1),
            hovertemplate=(
                "Contact top %{x:.0%}<br>Capture %{y:.1%} of reactivations"
                "<br>Lift %{customdata[0]:.2f}×<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random targeting",
            line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip",
        )
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


def profit_figure(y_true, y_prob, config: AppConfig) -> go.Figure:
    curve = profit_curve(np.asarray(y_true), np.asarray(y_prob), config.business)
    symbol = config.business.currency_symbol
    fig = themed_figure(
        title="Expected campaign profit vs decision threshold",
        xaxis_title="Decision threshold (contact members above this probability)",
        yaxis_title=f"Expected profit ({config.business.currency})",
    )
    fig.add_trace(
        go.Scatter(
            x=curve["threshold"], y=curve["expected_profit"], mode="lines",
            name="Expected profit", line=dict(color=SERIES[1], width=2),
            customdata=curve[["n_flagged", "precision"]].to_numpy(),
            hovertemplate=(
                "Threshold %{x:.3f}<br>Profit " + symbol + "%{y:,.0f}"
                "<br>%{customdata[0]} contacts"
                "<br>Precision %{customdata[1]:.1%}<extra></extra>"
            ),
        )
    )
    best = curve.loc[curve["expected_profit"].idxmax()]
    fig.add_trace(
        go.Scatter(
            x=[best["threshold"]], y=[best["expected_profit"]],
            mode="markers+text", name="Optimal threshold",
            text=[f"{symbol}{best['expected_profit']:,.0f}"],
            textposition="top center", textfont=dict(color=INK_SECONDARY),
            marker=dict(color=SERIES[5], size=11, line=dict(color="#fcfcfb", width=2)),
            hovertemplate="Optimal threshold %{x:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=GRID, width=1))
    return fig


def score_distribution_figure(y_true, y_prob) -> go.Figure:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fig = themed_figure(
        title="Score separation — reactivated vs lost contacts",
        xaxis_title="Predicted winback probability",
        yaxis_title="Share of group",
        barmode="overlay",
    )
    for label, mask, color in (
        ("Stayed lost", y_true == 0, SERIES[0]),
        ("Reactivated", y_true == 1, SERIES[1]),
    ):
        fig.add_trace(
            go.Histogram(
                x=y_prob[mask], name=label, histnorm="probability",
                nbinsx=40, opacity=0.62, marker_color=color,
                hovertemplate=label + ": %{y:.1%} at score %{x:.2f}<extra></extra>",
            )
        )
    return fig


RECENCY_BINS = [0, 21, 42, 70, 105, 150, 10_000]
RECENCY_LABELS = ["≤3 wk", "3–6 wk", "6–10 wk", "10–15 wk", "15–21 wk", "21+ wk"]

REASON_ORDER = ["price", "low_usage", "service", "other", "health", "relocation"]


def cohort_figure(full: pd.DataFrame) -> go.Figure:
    frame = full.copy()
    frame["recency_cohort"] = pd.cut(
        frame["days_since_cancel"], bins=RECENCY_BINS, labels=RECENCY_LABELS, right=True
    )
    pivot = (
        frame.pivot_table(
            index="cancel_reason", columns="recency_cohort",
            values=TARGET, aggfunc="mean", observed=True,
        )
        .reindex(index=REASON_ORDER)
        .reindex(columns=RECENCY_LABELS)
    )
    counts = frame.pivot_table(
        index="cancel_reason", columns="recency_cohort",
        values=TARGET, aggfunc="size", observed=True,
    ).reindex(index=pivot.index, columns=pivot.columns)
    fig = themed_figure(
        title="Cohort analysis — reactivation rate by cancellation reason × recency",
        xaxis_title="Time since cancellation at contact", yaxis_title="Cancellation reason",
        height=520,
    )
    fig.add_trace(
        go.Heatmap(
            z=pivot.values, x=RECENCY_LABELS, y=pivot.index.tolist(),
            colorscale=[[0, SEQUENTIAL_BLUES[0]], [1, SEQUENTIAL_BLUES[-1]]],
            colorbar=dict(title="Reactivation", tickformat=".0%"),
            text=[[f"{v:.1%}" if pd.notna(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            customdata=counts.values,
            hovertemplate=(
                "Reason %{y}, contacted %{x} after cancel"
                "<br>Reactivation rate %{z:.1%}"
                "<br>%{customdata:,} contacts<extra></extra>"
            ),
        )
    )
    return fig


def offer_effectiveness_figure(full: pd.DataFrame) -> go.Figure:
    """Observed reactivation rate by offer, split by cancellation reason.
    Readable causally because the pilot randomized offer assignment."""
    top_reasons = ["price", "low_usage", "service", "relocation"]
    offers = ["none", "discount_20", "free_month", "personal_trainer"]
    fig = themed_figure(
        title="Offer effectiveness by cancellation reason (randomized pilot)",
        xaxis_title="Cancellation reason",
        yaxis_title="Observed reactivation rate",
        barmode="group",
    )
    for i, offer in enumerate(offers):
        subset = full[full["offer"] == offer]
        rates = [
            subset.loc[subset["cancel_reason"] == r, TARGET].mean()
            for r in top_reasons
        ]
        counts = [
            int((subset["cancel_reason"] == r).sum()) for r in top_reasons
        ]
        fig.add_trace(
            go.Bar(
                x=top_reasons, y=rates, name=offer.replace("_", " "),
                marker_color=SERIES[i],
                customdata=np.array(counts).reshape(-1, 1),
                hovertemplate=(
                    offer + " × %{x}<br>Reactivation %{y:.1%}"
                    "<br>%{customdata[0]:,} contacts<extra></extra>"
                ),
            )
        )
    fig.update_yaxes(tickformat=".0%")
    return fig


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def run_evaluation(config: AppConfig) -> dict[str, float]:
    """Score the out-of-time contacts, persist metrics, figures and the
    business-impact report."""
    models_dir = Path(config.paths.models_dir)
    model = joblib.load(models_dir / "model.joblib")
    metadata = json.loads((models_dir / "metadata.json").read_text())
    threshold = metadata["decision_threshold"]

    data = load_processed(config)
    test, train = data["test"], data["train"]
    validate_feature_frame(test)

    y_true = test[TARGET].to_numpy()
    y_prob = model.predict_proba(test)

    metrics = compute_metrics(y_true, y_prob, threshold)
    log.info("Test metrics: {m}", m={k: round(v, 4) for k, v in metrics.items()})

    full = pd.concat([train, test], ignore_index=True)
    assets, img = config.paths.assets_dir, config.paths.img_dir
    figures = {
        "roc_curve": roc_figure(y_true, y_prob),
        "pr_curve": pr_figure(y_true, y_prob, threshold),
        "calibration_curve": calibration_figure(y_true, y_prob),
        "confusion_matrix": confusion_figure(y_true, y_prob, threshold),
        "gains_lift": gains_lift_figure(y_true, y_prob),
        "profit_curve": profit_figure(y_true, y_prob, config),
        "score_distribution": score_distribution_figure(y_true, y_prob),
        "cohort_winback": cohort_figure(full),
        "offer_effectiveness": offer_effectiveness_figure(full),
    }
    for name, fig in figures.items():
        save_figure(fig, name, assets, img)

    report = {
        "model": metadata["model_name"],
        "test_window_start": str(config.split.train_end),
        "metrics": metrics,
    }
    Path(assets, "model_performance.json").write_text(json.dumps(report, indent=2))

    build_business_report(test, y_prob, threshold, config)
    return metrics
