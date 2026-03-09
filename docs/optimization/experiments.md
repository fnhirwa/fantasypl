# Numerical Experiments

## Planned Comparisons

Using forecast data from Part I, the squad-selection model will be solved two ways:

### (a) Branch-and-Bound (Exact)

Standard MILP solver (CBC via PuLP) using branch-and-bound. Should find the provably optimal integer solution for 11 players from ~600 candidates.

- **Solver:** PuLP with `PULP_CBC_CMD`
- **Expected:** Finds global optimum; runtime < 1 second for single-period
- **Metric:** Optimal objective value $z^*_{ILP}$, solve time

### (b) First-Order Method (Approximate)

Relax integrality and apply a Lagrangian dual approach:

1. Relax the budget constraint into the objective with multiplier $\lambda$
2. Solve the inner problem (becomes a per-position sorting problem)
3. Update $\lambda$ via subgradient ascent
4. Round the relaxed solution to obtain a feasible integer solution

- **Expected:** Near-optimal solution faster per iteration, but may require tuning step size
- **Metric:** Objective value vs. iterations, distance from $z^*_{ILP}$

## Evaluation Metrics

| Metric | Definition | Purpose |
|--------|-----------|---------|
| Objective value | $\sum E[P_i] \cdot x_i^*$ | Solution quality |
| Solve time | Wall-clock seconds | Computational cost |
| Integrality gap | $(z^*_{LP} - z^*_{ILP}) / z^*_{LP}$ | LP relaxation tightness |
| Optimality gap | $(z^*_{ILP} - z_{\text{method}}) / z^*_{ILP}$ | Suboptimality of heuristic |
| Constraint analysis | Which constraints are binding | Resource scarcity insight |

## Backtesting Protocol

!!! note "Planned"
    Replay 38 gameweeks of a historical season:

    1. At each gameweek, run Part I inference to produce $E[P_i]$, $\text{Var}[P_i]$
    2. Solve the ILP (both risk-neutral and risk-averse variants)
    3. Record actual points scored by the selected squad
    4. Compare against:
        - Oracle (best possible squad with hindsight)
        - Baseline (rolling average → greedy selection)
        - Risk-neutral ILP vs. risk-averse ILP

    **Key question:** Does propagating uncertainty from Part I into Part II (via $\lambda > 0$) reduce the optimality gap, especially in volatile weeks?

## Expected Outcomes

Based on the proposal hypothesis:

- Branch-and-bound should solve all single-period instances optimally in under 1 second
- The first-order approach should converge to within 1-2% of optimal in ~50 iterations
- Shadow prices should reveal that budget is the tightest constraint (highest $\lambda_B$)
- Risk-averse selection ($\lambda > 0$) should produce more consistent weekly scores, especially around injury-heavy gameweeks
