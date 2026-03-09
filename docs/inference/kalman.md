# Kalman Filter

The Kalman Filter tracks a **continuous** latent variable representing the player's true point potential. It handles **gradual form drift** — a player slowly improving or declining across gameweeks.

## State-Space Model

$$x_{t+1} = x_t + w_t, \quad w_t \sim \mathcal{N}(0, Q_t)$$

$$y_t = x_t + v_t, \quad v_t \sim \mathcal{N}(0, R_t)$$

- $x_t$ — true (unobserved) point potential at gameweek $t$
- $y_t$ — observed weekly points (noisy measurement)
- $Q_t$ — process noise (how fast true form can drift)
- $R_t$ — observation noise (how noisy weekly points are)

The random-walk model ($x_{t+1} = x_t + w_t$) means form drifts gradually. $Q$ controls drift speed: large $Q$ allows rapid changes, small $Q$ assumes stable form.

## Predict-Update Cycle

At each timestep:

**Predict:**

$$\hat{x}_{t|t-1} = \hat{x}_{t-1|t-1}$$

$$P_{t|t-1} = P_{t-1|t-1} + Q_t$$

**Update:**

$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R_t} \quad \text{(Kalman gain)}$$

$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t \cdot (y_t - \hat{x}_{t|t-1})$$

$$P_{t|t} = (1 - K_t) \cdot P_{t|t-1}$$

!!! info "Kalman Gain Interpretation"
    When $Q_t$ is large → $P_{t|t-1}$ is large → $K_t$ is large → the filter trusts the **observation** more.
    When $R_t$ is large → $K_t$ is small → the filter trusts the **prior** more.
    This is exactly how news and fixtures should affect inference.

## Adaptive Noise (Signal Injection)

Both $Q_t$ and $R_t$ can be overridden per-timestep:

| Signal | Parameter | Multiplier | Effect |
|--------|-----------|:----------:|--------|
| Injury news | $Q_t$ | 5.0 | Form may have jumped; widen state uncertainty |
| Doubtful news | $Q_t$ | 2.0 | Moderate form uncertainty |
| Easy fixture (difficulty 1) | $R_t$ | 0.8 | Points more predictable, trust observation |
| Hard fixture (difficulty 5) | $R_t$ | 1.5 | Points less predictable, trust prior |

```python
kf = KalmanFilter(Q=1.0, R=4.0, x0=4.0, P0=2.0)

# Injury at gameweek 20: process noise inflated
kf.inject_process_shock(timestep=20, multiplier=5.0)

# Hard fixture at gameweek 21: observation noise inflated
kf.inject_observation_noise(timestep=21, factor=1.5)

x_est, P_est = kf.filter(observations)
```

## RTS Smoother

The Rauch-Tung-Striebel smoother runs a backward pass after the forward Kalman filter, producing refined estimates that use **all** data (past and future):

$$L_t = \frac{P_{t|t}}{P_{t+1|t}}$$

$$\hat{x}_{t|T} = \hat{x}_{t|t} + L_t \cdot (\hat{x}_{t+1|T} - \hat{x}_{t+1|t})$$

Available via [`KalmanFilter.smooth()`][fplx.inference.kalman.KalmanFilter.smooth].

## Default Parameters

| Parameter | Default | Meaning |
|-----------|:-------:|---------|
| $Q$ | 1.0 | Process noise — moderate form drift |
| $R$ | 4.0 | Observation noise — weekly points are noisy |
| $x_0$ | 4.0 | Initial estimate — league-average points |
| $P_0$ | 2.0 | Initial uncertainty |

!!! tip "Tuning"
    $Q$ should be **higher** for young/volatile players and **lower** for established veterans.
    $R$ can be estimated from the empirical variance of weekly points across the league.

## API

::: fplx.inference.kalman.KalmanFilter
    options:
      show_source: false
      members:
        - filter
        - predict_next
        - smooth
        - inject_process_shock
        - inject_observation_noise
