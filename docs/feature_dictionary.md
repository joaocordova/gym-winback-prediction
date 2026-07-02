# Feature dictionary

The prediction unit is a **winback contact**: one former member, contacted on
one date, through one channel, with one offer. Pre-cancellation behaviour is
frozen at the cancellation date (it can never change afterwards); contact
context is known when the contact list is drawn. The label
(`reactivated_within_horizon`) is a reactivation within 60 days of the
contact.

## Contact context (the campaign levers)

| Feature | Definition | Business logic |
|---|---|---|
| `days_since_cancel` | Days between cancellation and contact | Winback propensity decays fast (e-folding ≈ 2 months). A 3-week-old cancellation is warm; a 6-month one is cold. |
| `n_prior_contacts` | Winback touches this member already received | Diminishing returns on repeat outreach. |
| `channel` | email / phone / sms | Phone converts best, costs most staff time. |
| `offer` | none / discount_20 / free_month / personal_trainer | The lever the best-offer engine optimises. Randomized in the pilot, so its effect is identified. |

## Pre-cancellation behaviour (frozen at cancel)

| Feature | Window | Definition | Business logic |
|---|---|---|---|
| `visits_30d_precancel` | final 30d | Check-ins in the last month of membership | How alive the habit still was at the end. |
| `visits_90d_precancel` | final 90d | Check-ins in the final quarter | Baseline engagement. |
| `visits_per_week_90d` | final 90d | Weekly rate | Normalised version of the above. |
| `visit_decline_ratio` | final 90d | Final-30d visits ÷ prior-60d monthly rate | Distinguishes "faded out" (ratio ≪ 1 — habit already dead, harder to revive) from "left abruptly while active" (ratio ≈ 1 — the habit is dormant, not dead). |
| `active_weeks_ratio_12w` | final 12w | Share of weeks with ≥ 1 visit | Routine consistency before the end. |
| `class_ratio_precancel` | lifetime | Share of visits that were group classes | Social ties are a reason to come back. |

## Membership & demographics

| Feature | Definition | Business logic |
|---|---|---|
| `cancel_reason` | low_usage / price / relocation / service / health / other | **The dominant driver.** Price churners respond to discounts; relocated members are gone at any price. |
| `tenure_days` | Membership length before cancelling | Long-tenure habits die hard — they come back. |
| `monthly_fee`, `plan_type` | Former price point and commitment | Price sensitivity + what "come back" costs them. |
| `age`, `gender`, `referral_source` | Demographics, acquisition channel | Context; weak but non-zero signal. |

## Label

| Column | Definition |
|---|---|
| `reactivated_within_horizon` | 1 if the member re-joined within 60 days of this contact, else 0. |

## Split design

```
contacts ≤ 2026-02-28  ──►  train  (older campaign waves)
contacts  > 2026-02-28  ──►  test   (Mar–Apr 2026 waves, out-of-time)
```

The same member can appear in several contact rows, so **every** split — CV
folds, validation holdout — is grouped by `member_id` to prevent identity
leakage.
