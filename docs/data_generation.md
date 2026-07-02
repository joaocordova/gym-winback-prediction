# Synthetic data: generative assumptions

Real campaign data is proprietary, so the project ships a simulator
(`gym_winback/simulation.py`) that reproduces the *shape and failure modes*
of a real winback pilot. Everything below is encoded in code and reproducible
under `project.random_seed`.

## The scenario

Over ten months (Jul 2025 – Apr 2026) the gym ran a **randomized winback
pilot**: every cancelled member received 1–3 outreach touches with a random
channel (email 50% / phone 30% / sms 20%) and a random offer (uniform over
none / 20% discount / free month / personal-trainer session). Reactivations
within 60 days were recorded; reactivated members received no further
contacts.

**Why randomization matters:** because offers were assigned independently of
member characteristics, the offer × reason effects in this data are
identified — a model (and SHAP) can read them causally instead of echoing an
old targeting policy. This mirrors how a real winback program should be
piloted before a model takes over targeting.

## Reactivation mechanics

Per-contact reactivation probability is multiplicative:

```
p = base(reason) × tenure_factor × engagement_factor
    × exp(−days_since_cancel / 60) × channel_mult × offer_mult(offer, reason)
    × 0.65^(prior contacts)                              (capped at 0.75)
```

| Component | Values |
|---|---|
| `base(reason)` | price 0.19 · low_usage 0.14 · service 0.11 · other 0.09 · health 0.05 · **relocation 0.012** |
| `tenure_factor` | 0.70 + 0.45·log1p(tenure/180) — habits die hard |
| `engagement_factor` | 0.50 + 1.10·engagement — engaged leavers come back |
| recency decay | e-folding of 60 days — strike while warm |
| `channel_mult` | phone 1.35 · email 1.0 · sms 0.85 |
| `offer_mult` | discount_20: **2.3 × for price churners**, 1.4 otherwise · free_month: 1.6–1.9 · personal_trainer: **2.2 × for low-usage churners**, 1.2 otherwise |

The two starred interactions are the structure a gradient-boosted model
should exploit and SHAP should reveal — and they encode real retention
playbooks (price objections → discount; habit collapse → guided restart).

## Pre-cancel behaviour

Each former member has a latent engagement drawn Beta(2.0, 2.4) driving
weekly Poisson check-ins over their final 26 weeks. Members who left for
usage-driven reasons (low_usage / price / service / other) decay convexly
over their final 8–14 weeks; relocation/health leavers stop abruptly — so
`visit_decline_ratio` genuinely separates the two archetypes.

Cancellation reasons correlate with engagement (disengaged members leave for
usage reasons; engaged leavers skew price/relocation/health), reproducing the
confounding a real model must untangle.

## What the simulator guarantees (tested in `tests/test_simulation.py`)

- Deterministic under the seed.
- No check-ins on/after the cancellation; no contacts on/before it.
- No contacts after a successful reactivation.
- Relocated members reactivate at < 50% the rate of price churners.
- Discounts measurably move price churners (the learnable interaction).
- Overall reactivation rate lands in the realistic 5–30% band.
