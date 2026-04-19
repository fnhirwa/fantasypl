# Numerical Experiments

## Backtest Protocol

All experiments use a strict walk-forward protocol: at gameweek $t$, only
GW1–$(t-1)$ data is available to the inference pipeline. No future information
leaks into predictions.

- **Dataset:** `vaastav/Fantasy-Premier-League`
- **Seasons:** 2023–24 and 2024–25
- **Evaluation window:** GW6–38 (33 gameweeks; 5-GW burn-in)
- **Player pool:** ~600 players per gameweek

## Oracle Computation

The oracle score at each gameweek is the solution to:

$$\text{Oracle}(t) = \max_{s,x} \sum_i P^{\text{actual}}_i(t)\, x_i \quad \text{s.t.\ all squad/lineup constraints}$$

solved using `TwoLevelILPOptimizer` with actual realized points. Oracle totals:
4560 pts (2023–24) and 4369 pts (2024–25) over 33 GWs.

## Results Summary

!!! warning "Pre-DGW-fix baseline numbers"
    These results were produced before the DGW integration described in
    [Double Gameweeks](../inference/double-gameweek.md). Three changes affect
    the numbers when re-run:

    - **Oracle scores will increase** in DGW gameweeks — `get_actual_points`
      previously dropped the first fixture row, understating actual points.
    - **Strategy scores will increase** in DGW-heavy weeks — the ILP now
      correctly values DGW players at `n_fixtures × E[P]`.
    - **Inference MSE will decrease slightly** — the HMM no longer
      misclassifies DGW totals as extreme single-game events; `points_norm`
      keeps emissions on a consistent scale.

    The qualitative ranking of strategies is not expected to change.
    Updated numbers will be added after the next full backtest run.

| Strategy | 2023–24 pts | % Oracle | 2024–25 pts | % Oracle |
|----------|:-----------:|:--------:|:-----------:|:--------:|
| Greedy (rolling avg) | 1272 | 27.9% | 1216 | 27.8% |
| ILP + EWMA | 1511 | 33.1% | 1612 | 36.9% |
| ILP + MV-HMM | 1325 | 29.1% | 1380 | 31.6% |
| **ILP + Enriched** | **1949** | **42.7%** | **1791** | **41.0%** |
| ILP + Semivar (λ=0.5) | 1881 | 41.3% | 1757 | 40.2% |
| ILP + Blend (λ=0) | 1919 | 42.1% | 1774 | 40.6% |
| ILP + TFT | 1334 | 29.3% | 1352 | 30.9% |
| Lagrangian + Enriched | 1912 | 41.9% | 1749 | 40.0% |

## Key Findings

**Forecast quality determines squad quality.** ILP + Enriched outperforms
ILP + EWMA by 29.1% (2023–24) and 11.1% (2024–25). This is the primary
empirical confirmation that the inference investment pays off in selection.

**TFT paradox.** TFT achieves the lowest MAE in both seasons (0.837–0.918)
but the worst ILP input. MAE-optimal forecasts do not minimize the structure
of rank-error correlations that matter for ILP selection. Players at the top
of the predicted ranking tend to have correlated errors — the optimizer
selects them together, and they together underperform.

**Semi-variance gives a genuine risk-return trade-off.** In 2024–25,
semivar λ=0.5 reduces CV from 0.300 to 0.260 (−13%) at a cost of only −34
total points (−1.9%). Mean-variance penalizes star performers and is not
recommended.

**Lagrangian relaxation is viable.** 97.7–98.1% of full ILP performance with
~50 subgradient iterations. Integrality gap < 5% in all tested gameweeks.

## Duality Analysis

Shadow prices from the LP relaxation (averaged across 33 GWs):

| Constraint | Mean shadow price | Interpretation |
|------------|:-----------------:|----------------|
| Budget | ~0.6–0.8 pts/£1m | Marginal value of one additional £1m of budget |
| GK quota | ~0 | GK constraint rarely binds (single GK per format) |
| DEF/MID quota | ~0 | Range constraints: both min and max slack in most GWs |
| FWD quota (max=3) | Occasionally positive | Top FWDs are frequently budget-constrained |
| Team cap (top-6) | Occasionally positive | Arsenal, City, Chelsea caps bind in strong GWs |

The budget constraint has the largest and most consistent shadow price,
confirming that capital allocation is the primary driver of squad quality.

## Integrality Gap

The integrality gap $(z^*_{LP} - z^*_{ILP}) / z^*_{LP}$ is below 5% in all
evaluated gameweeks. This tight gap validates:
1. The LP relaxation bound is informative
2. Lagrangian relaxation (which relaxes to the LP bound) loses little
3. The feasible integer solution is close to the fractional optimum
