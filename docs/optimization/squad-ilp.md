# Squad Selection ILP

## Two-Level Formulation

The optimizer solves a single joint ILP for squad selection (Level 1) and lineup
selection (Level 2). Binary variables $s_i \in \{0,1\}$ indicate squad membership;
$x_i \in \{0,1\}$ indicate starting lineup membership.

**Objective (risk-averse):**

$$\max_{s,x} \sum_{i} (\hat{\mu}_i - \lambda \cdot \rho_i)\, x_i$$

where $\lambda \geq 0$ is the risk-aversion parameter and $\rho_i$ is the risk
measure (see [Risk Penalties](#risk-penalties)). Setting $\lambda = 0$ gives the
risk-neutral objective.

**Level 1 — Squad constraints:**

| Constraint | Expression |
|------------|-----------|
| Squad size | $\sum_i s_i = 15$ |
| Position quotas | $s_{GK}=2$, $s_{DEF}=5$, $s_{MID}=5$, $s_{FWD}=3$ |
| Budget | $\sum_i c_i s_i \leq 100$ (applied to the **full 15-player squad**) |
| Team diversity | $\sum_{i \in \text{team}_t} s_i \leq 3 \;\forall t$ |

**Level 2 — Lineup constraints:**

| Constraint | Expression |
|------------|-----------|
| Lineup size | $\sum_i x_i = 11$ |
| Subset of squad | $x_i \leq s_i \;\forall i$ |
| GK | $\sum_{i:p(i)=GK} x_i = 1$ |
| DEF | $3 \leq \sum_{i:p(i)=DEF} x_i \leq 5$ |
| MID | $2 \leq \sum_{i:p(i)=MID} x_i \leq 5$ |
| FWD | $1 \leq \sum_{i:p(i)=FWD} x_i \leq 3$ |

!!! warning "Budget applies to the full squad"
    The £100m budget constraint covers all 15 players, not just the 11
    starters. Formulations that apply the budget only to the lineup are
    incorrect and produce infeasible squads.

## Risk Penalties

Both penalties keep the objective **linear** because $x_i$ is binary and the
risk coefficient $\rho_i$ is computed by the inference pipeline before solving.

### Mean-Variance Penalty

$$\rho_i^{\text{MV}} = \sqrt{\hat{\sigma}^2_i}$$

Penalizes all variance symmetrically. Tends to penalize star performers with
high upside, so should be preferred only when downside risk estimates are
unavailable.

### Semi-Variance Penalty

$$\rho_i^{\text{SV}} = \hat{\sigma}^-_i = \sqrt{\mathbb{E}[\max(0, \hat{\mu}_i - P_i)^2]}$$

Under a Gaussian predictive distribution, $\rho_i^{\text{SV}} = \hat{\sigma}_i / \sqrt{2}$.
Penalizes only downside deviations, leaving upside variance intact. This is the
preferred risk measure and is returned directly by `enriched_predict`.

**Backtest result:** Semivar $\lambda=0.5$ in 2024–25 reduces CV from 0.300 to
0.260 (−13%) at a cost of only −1.9% total points (1791 → 1757).

### Downside Risk from TFT

When TFT quantile outputs are available, the spread
$\delta_i = q_{50,i} - q_{10,i}$ is used directly as $\rho_i$. This is a
distribution-free, data-driven risk measure.

## Oracle Score

The oracle provides the performance ceiling for any forecasting system:

$$\text{Oracle}(t) = \max_{s,x} \sum_i P^{\text{actual}}_i(t)\, x_i \quad \text{s.t.\ all constraints}$$

The same `TwoLevelILPOptimizer` is called with actual realized points as
inputs. This is **not** the sum of the top-11 scorers — budget, position
quotas, and team caps are all enforced.

Oracle averages 138.2 pts/GW (2023–24) and 132.4 pts/GW (2024–25).
The best strategy (ILP + Enriched) achieves ~42% of oracle.

## Double Gameweek Handling in the ILP

`aggregate_dgw_timeseries` (called automatically in the data layer) ensures
inference always outputs **per-fixture** expected points. Before the optimizer
call, `scale_predictions_for_dgw` applies the multiplier:

| Quantity | Scaling | Rationale |
|----------|---------|-----------|
| $\hat{\mu}_i$ | $\times n_i$ | EP adds linearly under match independence |
| $\hat{\sigma}^2_i$ | $\times n_i$ | Variance adds under independence |
| $\hat{\sigma}^-_i$ | $\times \sqrt{n_i}$ | Semi-deviation scales as $\sqrt{n}$ |
| BGW player ($n_i=0$) | $\hat{\mu}_i = 0$ | No fixture; naturally excluded by ILP |

DGW players ($n_i=2$) receive doubled expected points as their objective
coefficient and become the most attractive picks — which is correct.

See [Double Gameweeks](../inference/double-gameweek.md) for the full design.

## Lagrangian Relaxation

The budget constraint is dualized with multiplier $\mu \geq 0$:

$$L(\mu) = \max_{s,x \in \{0,1\}} \sum_i (\hat{\mu}_i - \lambda\hat{\rho}_i)\, x_i + \mu\left(B - \sum_i c_i s_i\right)$$

The dual $D(\mu) = \min_{\mu \geq 0} L(\mu)$ is minimized via subgradient ascent.
At each iteration, the unconstrained inner problem decomposes into independent
per-player problems solved greedily. Subgradient:
$g_k = B - \sum_i c_i s^*_i(\mu_k)$.

**Backtest result:** Lagrangian relaxation recovers 97.7% (2024–25) and 98.1%
(2023–24) of full ILP performance, converging in ~50 iterations.

## Valid Formations

The lineup constraints admit 8 valid formations:
3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-2-3, 5-3-2, 5-4-1.
Formation is determined automatically by the optimizer — no manual selection needed.

## Usage

```python
from fplx.selection.optimizer import TwoLevelILPOptimizer

optimizer = TwoLevelILPOptimizer(budget=100.0, risk_aversion=0.5)
squad = optimizer.optimize(
    players=players,
    expected_points=ep,
    expected_variance=var,        # for mean-variance penalty
    downside_risk=downside_risks, # overrides variance if provided
)
# squad.squad_players  -> 15 Player objects
# squad.lineup.players -> 11 starters
# squad.lineup.captain -> highest-EP player
```
