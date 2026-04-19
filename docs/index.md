# FPLX

**Stochastic Inference & Constrained Optimization for Fantasy Premier League**

FPLX treats FPL squad selection as a dynamic decision problem under uncertainty.
It models each player's form as a latent variable, fuses noisy evidence via
probabilistic filters, and feeds the resulting distributions into a constrained
optimizer. Validated on two full historical seasons (2023–24 and 2024–25,
GW6–38, ≈24 000 player-gameweek observations each).

| | Component | Focus |
|---|-----------|-------|
| :material-brain: | [**Inference**](inference/index.md) | MV-HMM + Enriched predictor, inverse-variance blend, news signal injection |
| :material-chart-box: | [**Optimization**](optimization/index.md) | Two-level ILP, mean-variance and semi-variance risk penalties, Lagrangian relaxation |

---

## System Overview

``` mermaid
graph LR
    A[FPL API] --> B[Data Loading]
    B --> C[News / Fixtures]
    B --> D[Player History]
    B --> E[DGW Detection]
    C --> F[News Injection]
    D --> G[Enriched Predictor]
    D --> H[MV-HMM]
    D --> I[Kalman Filter]
    F -->|Transition perturb.| H
    F -->|Q_t inflate| I
    H --> J[Calibrated Blend]
    G --> J
    I --> J
    E -->|n_fixtures scale| J
    J -->|E[P], Var[P], DR| K[Two-Level ILP]
    K --> L[15-player Squad + 11-player Lineup]
```

## Pipeline: Perceive → Infer → Act

| Layer | Components | Role |
|-------|------------|------|
| **Perceive** | `NewsSignal`, `FixtureSignal`, `NewsCollector`, `double_gameweek` | Parse news text, fixture difficulty, detect DGW/BGW |
| **Infer** | `MultivariateHMM`, `KalmanFilter`, `enriched_predict`, `fuse_estimates` | Track latent form states, continuous point potential, calibrated blend |
| **Act** | `TwoLevelILPOptimizer`, `LagrangianOptimizer` | Select optimal 15+11 squad under budget, formation, and team constraints |

---

## Key Results (Backtest GW6–38)

| Season | Best MSE | vs. baseline | Best strategy | % of oracle |
|--------|----------|:------------:|---------------|:-----------:|
| 2023–24 | 3.656 (Blend) | −14.4% | ILP + Enriched: **1949 pts** | 42.7% |
| 2024–25 | 4.145 (Blend) | −12.3% | ILP + Enriched: **1791 pts** | 41.0% |

Oracle averages 138.2 / 132.4 pts/GW. Lagrangian relaxation recovers 97.7–98.1% of full ILP.

---

## Quick Links

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [Double Gameweeks](inference/double-gameweek.md)
- [Examples](examples.md)
