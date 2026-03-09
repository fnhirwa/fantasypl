# Squad Selection ILP

## Formulation

The squad selection problem is a binary integer linear program.

**Decision variables:** $x_i \in \{0, 1\}$ — whether player $i$ is in the squad.

**Objective (risk-neutral):**

$$\max \sum_{i=1}^{n} E[P_i] \cdot x_i$$

**Objective (risk-averse, mean-variance):**

$$\max \sum_{i=1}^{n} E[P_i] \cdot x_i \;-\; \lambda \sum_{i=1}^{n} \sqrt{\text{Var}[P_i]} \cdot x_i$$

where $\lambda \geq 0$ controls risk aversion. Since $x_i$ is binary, $\sqrt{\text{Var}[P_i]} \cdot x_i$ is **linear** in $x_i$ — the problem remains a standard ILP.

## Constraints

**Squad size:**

$$\sum_{i=1}^{n} x_i = 11$$

**Budget:**

$$\sum_{i=1}^{n} \text{cost}_i \cdot x_i \leq B \quad (B = 100.0)$$

**Formation** (position quotas):

| Position | Min | Max |
|----------|:---:|:---:|
| GK | 1 | 1 |
| DEF | 3 | 5 |
| MID | 2 | 5 |
| FWD | 1 | 3 |

$$\text{min}_p \leq \sum_{i \in \text{pos}_p} x_i \leq \text{max}_p \quad \forall p \in \{\text{GK}, \text{DEF}, \text{MID}, \text{FWD}\}$$

**Team diversity** (max 3 from any real-world team):

$$\sum_{i \in \text{team}_t} x_i \leq 3 \quad \forall t$$

## Valid Formations

The constraints admit 8 valid formations: 3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-2-3, 5-3-2, 5-4-1.

## Implementation

### ILP Solver (PuLP)

[`ILPOptimizer`][fplx.selection.optimizer.ILPOptimizer] uses PuLP with the CBC backend for branch-and-bound:

```python
from fplx.selection.optimizer import ILPOptimizer

optimizer = ILPOptimizer(budget=100.0)
squad = optimizer.optimize(players, expected_points, formation=None)
```

When `formation=None`, the optimizer finds the optimal formation automatically.

### Greedy Heuristic

[`GreedyOptimizer`][fplx.selection.optimizer.GreedyOptimizer] sorts players by value (expected points / price) per position. Fast but not guaranteed optimal:

```python
from fplx.selection.optimizer import GreedyOptimizer

optimizer = GreedyOptimizer(budget=100.0)
squad = optimizer.optimize(players, expected_points, formation="4-3-3")
```

### Mean-Variance Extension

!!! note "Planned"
    The mean-variance objective replaces the ILP coefficient for each player:

    $$\text{score}_i = E[P_i] - \lambda \cdot \sqrt{\text{Var}[P_i]}$$

    This requires `expected_variance` from `FPLModel.fit()` (available when `model_type="inference"`).
    No solver changes needed — just modified objective coefficients.

## Multi-Period Extension

The proposal describes a multi-period formulation with transfer variables:

$$\max \sum_{t=1}^{T} \sum_{i=1}^{n} E[P_{i,t}] \cdot x_{i,t}$$

with transfer limits or penalties between consecutive gameweeks. This extends the single-period ILP with coupling constraints across time periods.
