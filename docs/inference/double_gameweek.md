# Double Gameweeks (DGW)

A Double Gameweek occurs when a Premier League team plays **two fixtures** in the
same FPL gameweek round. Blank Gameweeks (BGW) are the mirror: zero fixtures.

Without explicit handling, DGW creates two distinct failure modes:

1. **Historical timeseries corruption.** The vaastav dataset stores each fixture
   as a separate row. A DGW player has two rows with the same `gameweek` value.
   If both are fed to the HMM, a large DGW total (e.g. 14 pts from two games)
   is misread as a single "Star" observation when it is actually two "Average"
   games — inflating state estimates and biasing transition learning.

2. **Forward prediction mis-scaling.** When the upcoming gameweek is a DGW,
   using a single-game expected points figure as the ILP objective coefficient
   understates a player's value by ~2×, biasing selection away from the best picks.

---

## Design: One Entry Point, Inference-Agnostic

All DGW handling is concentrated in the **data layer**. Every consumer of player
data — inference pipeline, MV-HMM, enriched predictor, Kalman Filter — always
receives exactly one row per FPL decision period and never needs to be
DGW-aware. Only the ILP objective coefficient needs the `× n_fixtures`
multiplier.

```
Data layer                   Inference              Optimizer
──────────────────────────   ────────────────────   ──────────────────────
build_player_objects()        HMM, KF, enriched      scale_predictions_for_dgw()
  └── aggregate_dgw_ts()   → points_norm (always  →  E[P] × n_fixtures
        one row / GW          single-game equiv.)     Var  × n_fixtures
                                                      DR   × √n_fixtures
```

---

## Timeseries Aggregation

`aggregate_dgw_timeseries` is called automatically inside
`VaastavLoader.build_player_objects` and inside `build_timeseries` in the live
deployment script. **You never need to call it manually.**

For a DGW gameweek, the resulting single row contains:

| Column | Value | Used for |
|--------|-------|----------|
| `points` | sum of both fixtures | Oracle scoring, `get_actual_points` |
| `points_norm` | `points / n_fixtures` | Inference (HMM emissions, enriched predictor) |
| `minutes` | sum of both | Availability proxy |
| `xG`, `xA` | mean of both | Rate features (already per-match) |
| `n_fixtures` | 2 | ILP scaling |

For a single-fixture gameweek `points_norm == points` and `n_fixtures == 1`.

```python
# For inspection / debugging only — not needed in normal usage:
from fplx.data.double_gameweek import aggregate_dgw_timeseries, detect_dgw_gameweeks

counts  = detect_dgw_gameweeks(player.timeseries)   # {gw: n_fixtures}
dgw_gws = [gw for gw, n in counts.items() if n > 1]

ts_agg  = aggregate_dgw_timeseries(player.timeseries)
# ts_agg always has one row per GW; after build_player_objects it is already applied
```

---

## Why Per-Fixture Normalisation?

The MV-HMM emission distributions are calibrated from historical data:

| State | Emission μ (pts/game) |
|-------|:---------------------:|
| Injured | 0.5 |
| Slump | 2.0 |
| Average | 4.0 |
| Good | 6.0 |
| Star | 8.5 |

A DGW player scoring 10 total points (`points_norm = 5`) should be classified as
"Average" — not as "Good" (6.0) or "Star" (8.5). Without normalisation, the raw
total of 10 would be an outlier to the emission model. With `points_norm`, the
observation is 5.0 and the HMM state assignment is correct.

---

## ILP Scaling

After inference produces per-fixture predictions, `scale_predictions_for_dgw`
applies the `n_fixtures` multiplier before the optimizer call:

```python
from fplx.data.double_gameweek import (
    get_fixture_counts_from_bootstrap,   # live API
    get_fixture_counts_from_vaastav,     # historical backtest
    scale_predictions_for_dgw,
)

# Fixture counts for the upcoming GW
fixture_counts = get_fixture_counts_from_bootstrap(bootstrap, target_gw)

# Scale: E[P]×n, Var×n, DR×√n; BGW players get E[P]=0
ep_s, var_s, dr_s = scale_predictions_for_dgw(
    expected_points, variances, downside_risks, fixture_counts
)
```

!!! note "Conservative variance mode for correlated DGW fixtures"
    Pass `variance_mode="conservative"` to multiply variance by `n²` instead
    of `n` — appropriate when both fixtures are against the same strong opponent.

---

## Effect on Results

The pre-integration backtest results (reported in the experiments page) were
collected without DGW handling. Re-running with the fix is expected to change:

- **Oracle scores** — will increase in DGW gameweeks (actual points now
  correctly summed via the patched `get_actual_points`)
- **Strategy scores** — will increase for DGW-heavy weeks (ILP correctly
  values DGW players at 2× single-game EP)
- **Inference MSE** — will decrease slightly (HMM no longer misclassifies
  DGW totals as extreme single-game events)

The experiments page flags these as pre-fix baseline numbers.

---

## `get_actual_points` Fix

The original implementation used `dict(zip(element, points))` which silently
kept only the second fixture row for DGW players. The fix uses
`groupby("element")[pts_col].sum()` which correctly sums both rows.

---

## API Reference

- `detect_dgw_gameweeks(timeseries)` → `{gw: n_fixtures}`
- `aggregate_dgw_timeseries(timeseries)` → one row/GW with `points_norm`
- `scale_predictions_for_dgw(ep, var, dr, counts)` → scaled prediction dicts
- `get_fixture_counts_from_bootstrap(bootstrap, gw)` → live detection
- `get_fixture_counts_from_vaastav(loader, gw)` → historical detection
