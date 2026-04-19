# Duality & Shadow Prices

## LP Relaxation

Relaxing the integrality constraint ($x_i \in \{0,1\} \to x_i \in [0,1]$) yields a linear program whose dual provides economic interpretations of the constraints.

## Dual Problem

The Lagrangian of the LP relaxation introduces dual variables for each constraint:

| Constraint | Dual Variable | Interpretation |
|------------|:-------------:|----------------|
| Budget: $\sum \text{cost}_i \cdot x_i \leq B$ | $\lambda_B$ | Marginal value of one additional unit of budget |
| Position min: $\sum_{i \in p} x_i \geq \text{min}_p$ | $\mu_p^-$ | Cost of the minimum-position requirement |
| Position max: $\sum_{i \in p} x_i \leq \text{max}_p$ | $\mu_p^+$ | Value of an additional slot in position $p$ |
| Team cap: $\sum_{i \in t} x_i \leq 3$ | $\nu_t$ | Cost of the team diversity constraint for team $t$ |

## KKT Conditions

At the LP optimum, the KKT conditions give:

$$E[P_i] - \lambda_B \cdot \text{cost}_i + \mu_{p(i)}^{-} - \mu_{p(i)}^{+} - \nu_{t(i)} \leq 0 \quad \forall i$$

with equality when $x_i > 0$ (complementary slackness).

**Interpretation:** A player is selected ($x_i = 1$) only when their expected points minus the cost of their budget usage and constraint contributions is non-negative.

## Shadow Price Analysis

!!! note "Planned Experiment"
    Using the forecast data from Inference, we will:

    1. Solve the LP relaxation and extract dual variables
    2. Report $\lambda_B$ — the marginal value of budget
    3. Identify which position and team constraints are binding
    4. Compare shadow prices across gameweeks with different injury profiles

    **Expected insight:** $\lambda_B$ should be higher in gameweeks where top players are available (budget is scarce relative to talent). Team constraints for top-6 clubs should frequently be binding.

## Integrality Gap

The LP relaxation gives an upper bound on the ILP objective. The **integrality gap** measures how much is lost by requiring integer solutions:

$$\text{gap} = \frac{z_{LP}^* - z_{ILP}^*}{z_{LP}^*}$$

For FPL-sized instances (11 players from ~600 candidates), we expect a small gap because the problem has relatively few binding constraints compared to the number of variables.
