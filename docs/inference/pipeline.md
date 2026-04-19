# Inference Pipeline

The core contribution of FPLX. Each player is modeled independently through a
multi-component system that tracks discrete form states (MV-HMM), continuous
point potential (Kalman Filter), and structural statistical patterns (Enriched
predictor). Their outputs are blended via calibrated inverse-variance weighting.

## Overview

```mermaid
graph TD
    RAW[Raw per-fixture data] --> AGG[aggregate_dgw_timeseries\none row / GW, points_norm]
    AGG --> E[Enriched Predictor]
    AGG --> HMM[MV-HMM]
    AGG --> KF[Kalman Filter]
    N[NewsSignal] -->|Transition perturbation| HMM
    N -->|Process noise shock| KF
    HMM -->|E[Y], Var[Y]| BL[Calibrated Blend]
    E -->|mu_e, var_e, DR| BL
    KF -->|x̂, P| BL
    BL -->|per-fixture E[P], Var[P], DR| SC[scale_predictions_for_dgw\n× n_fixtures]
    SC -->|full-GW E[P], Var[P]| OPT[Two-Level ILP]
```

## Components

### Enriched Predictor

The single strongest individual model. Constructs a per-player feature vector
including: EWMA of xG, xA, BPS, clean sheets, cards, own goals, penalties;
fixture difficulty (home/away factor, opponent strength); and rolling statistics
at windows [3, 5, 10]. A Ridge regression maps these features to expected points.

The enriched predictor also computes a **semi-variance** (downside risk below
E[P]) used directly by the risk-averse ILP objective.

Key principle: **availability prediction is separated from form prediction.**
The enriched predictor is fitted only on played gameweeks (minutes > 0), then
availability (from recent minutes history) is applied as a multiplicative factor.
This prevents the "Injured" state from absorbing variance that belongs to the
form distribution.

### Multivariate HMM

Each player is represented by a discrete hidden state
$S_t \in \{	ext{Injured}, 	ext{Slump}, 	ext{Average}, 	ext{Good}, 	ext{Star}\}$.

The MV-HMM uses position-specific feature vectors (not raw points scalars):

| Feature | GK | DEF | MID | FWD |
|---------|:--:|:---:|:---:|:---:|
| xPts (composite) | ✓ | ✓ | ✓ | ✓ |
| Minutes fraction | ✓ | ✓ | ✓ | ✓ |

Emission distributions are diagonal Gaussians per state. Parameters are learned
via Baum-Welch EM with MAP regularization toward position-level priors
(weight `prior_weight=0.85`), preventing overfitting on short histories.

!!! warning "Scalar HMM / KF alone are worse than rolling average"
    In both backtested seasons, the scalar HMM and KF individually *increase*
    MSE above the rolling average baseline. Probabilistic machinery only delivers
    gains when paired with structural observations (xG, xA, BPS, fixture context).
    The MV-HMM with position features is the correct architecture.

### Kalman Filter

Tracks a continuous latent variable $x_t$ via a random-walk model:

$$x_{t+1} = x_t + w_t, \quad w_t \sim \mathcal{N}(0, Q_t)$$
$$y_t = x_t + v_t, \quad v_t \sim \mathcal{N}(0, R_t)$$

The filter yields minimum-MSE estimate $\hat{x}_t$ and posterior variance $P_t$.

### News Signal Injection

News is injected **inside** the inference process, not as a post-hoc multiplier.
When news indicates an injury:

| Category | HMM boost | KF $Q_t$ multiplier |
|----------|-----------|:-------------------:|
| Unavailable | Injured ×10 | 5.0× |
| Doubtful | Injured ×3, Slump ×2 | 2.0× |
| Rotation | Slump ×2 | 1.5× |
| Positive | Good ×2 | 1.0× |

**Anticipatory advantage:** when news arrives before the gameweek (e.g.
"doubtful for Saturday"), the pipeline adjusts predictions before observing 0
points. Observation-only methods must wait one gameweek.

### Calibrated Blend

The Enriched predictor and MV-HMM are blended per player:

$$\hat\mu = lpha \hat\mu_	ext{enr} + (1-lpha) \hat\mu_	ext{mv}$$
$$\hat\sigma^2 = lpha^2 \hat\sigma^2_	ext{enr} + (1-lpha)^2 \hat\sigma^2_	ext{mv}$$

$lpha$ is calibrated per player via a rolling one-step MSE grid search over
the player's historical data. Default $lpha = 0.8$ (enriched-dominant) when
fewer than 8 gameweeks of history are available.

The KF output is fused with the blend via inverse-variance weighting (the
guarantee that $\hat\sigma^2_	ext{fused} \leq \min(\hat\sigma^2)$ holds).

## Ablation Results

| Model | MSE 2023–24 | MSE 2024–25 |
|-------|:-----------:|:-----------:|
| Rolling average (baseline) | 4.269 | 4.727 |
| EWMA | 4.172 | 4.599 |
| HMM scalar | 4.574 | 5.045 |
| KF scalar | 4.291 | 4.715 |
| Fused scalar | 4.295 | 4.726 |
| MV-HMM | 4.509 | 4.875 |
| Enriched | 3.643 | 4.197 |
| **Enriched + MV-HMM blend** | **3.656** | **4.145** |
| TFT q50 | 4.197 | 4.573 |

95% calibration coverage: 91.3% (2023–24) and 90.1% (2024–25).

## Double Gameweeks

DGW handling is fully integrated into the data layer.
`aggregate_dgw_timeseries` is called automatically inside
`VaastavLoader.build_player_objects` (historical) and `build_timeseries`
(live API), so the inference pipeline always receives one row per GW with
`points_norm` (per-fixture normalised) as the training target.
`scale_predictions_for_dgw` is the only DGW-aware step after inference.

See [Double Gameweeks](double-gameweek.md) for full details.
