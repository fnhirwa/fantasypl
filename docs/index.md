# FPLX

**Stochastic Inference & Constrained Optimization for Fantasy Premier League**

FPLX treats FPL squad selection as a dynamic decision problem under uncertainty. It models each player's form as a latent variable, fuses noisy evidence via probabilistic filters, and feeds the resulting distributions into a constrained optimizer.

This project spans two components:

| | Component | Focus |
|---|-----------|-------|
| :material-brain: | [**Inference**](inference/index.md) | HMM + Kalman Filter for probabilistic point forecasting with news signal injection |
| :material-chart-box: | [**Optimization**](optimization/index.md) | ILP formulation, duality analysis, and robust squad selection under uncertainty |

---

## System Overview

``` mermaid
graph LR
    A[FPL API] --> B[Data Loading]
    B --> C[News Collection]
    B --> D[Player History]
    C --> E[Signal Processing]
    D --> F[HMM + Kalman Filter]
    E -->|Transition perturbation| F
    E -->|Noise modulation| F
    F --> G[Fusion]
    G -->|"E[P], Var[P]"| H[ILP / Greedy Optimizer]
    H --> I[Optimal Squad]
```

## Pipeline: Perceive → Infer → Act

| Layer | Components | Role |
|-------|------------|------|
| **Perceive** | `NewsSignal`, `FixtureSignal`, `NewsCollector` | Parse news text, fixture difficulty, collect per-gameweek snapshots |
| **Infer** | `HMMInference`, `KalmanFilter`, `fuse_estimates` | Track latent form states, continuous point potential, fuse with uncertainty |
| **Act** | `GreedyOptimizer`, `ILPOptimizer` | Select optimal squad under budget, formation, and team constraints |

---

## Quick Links

- :material-rocket-launch: [**Installation**](getting-started/installation.md) — Set up FPLX in minutes
- :material-code-braces: [**Quick Start**](getting-started/quickstart.md) — Run your first squad selection
- :material-api: [**API Reference**](api/index.md) — Auto-generated from docstrings
- :material-test-tube: [**Examples**](examples.md) — Runnable scripts against the live FPL API
