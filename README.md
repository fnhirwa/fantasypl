# FPLX — Stochastic Inference & Constrained Optimization for Fantasy Premier League

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

FPLX treats FPL squad selection as a **dynamic decision problem under uncertainty**. Instead of producing static point projections, it models each player's underlying form as a latent variable, fuses noisy evidence (match stats, injury news, fixture difficulty) via probabilistic filters, and feeds the resulting distributions into a constrained optimizer.

The system has three layers: **Perceive** (news signals, fixture data) → **Infer** (HMM + Kalman Filter → fused predictions with uncertainty) → **Act** (ILP/Greedy squad selection).

---

## Architecture

```
FPL API (bootstrap-static, element-summary, fixtures)
│
├─ fplx/data/           Data loading & news collection
│   ├─ loaders.py           FPL API client, CSV loader, player enrichment
│   ├─ news_collector.py    Per-gameweek news snapshots from API
│   └─ schemas.py           Pydantic validation schemas
│
├─ fplx/signals/        External signal processing
│   ├─ news.py              NLP: parse injury/availability from news text
│   ├─ fixtures.py          Fixture difficulty & congestion scoring
│   └─ stats.py             Statistical performance aggregation
│
├─ fplx/inference/      Probabilistic inference pipeline (core contribution)
│   ├─ hmm.py               Hidden Markov Model (5-state form tracking)
│   ├─ kalman.py            Kalman Filter (continuous point potential)
│   ├─ fusion.py            Inverse-variance weighting of HMM + KF
│   └─ pipeline.py          Per-player orchestrator with signal injection
│
├─ fplx/timeseries/     Feature engineering
│   ├─ transforms.py        Rolling, lag, EWMA, trend, consistency features
│   └─ features.py          Feature engineering pipeline (40+ features)
│
├─ fplx/models/         Prediction models (baselines & ML)
│   ├─ baseline.py          Rolling mean, EWMA, form-based heuristics
│   ├─ regression.py        Ridge, XGBoost, LightGBM with rolling CV
│   ├─ ensemble.py          Weighted & adaptive ensemble
│   └─ rolling_cv.py        Time-aware cross-validation splits
│
├─ fplx/selection/      Squad optimization
│   ├─ constraints.py       Formation, budget, team diversity constraints
│   └─ optimizer.py         Greedy & ILP (PuLP) optimizers
│
├─ fplx/core/           Domain objects
│   ├─ player.py            Player dataclass
│   ├─ squad.py             Squad dataclass with validation
│   └─ matchweek.py         Gameweek context
│
├─ fplx/utils/          Configuration & validation
│   ├─ config.py            Nested config with dot-notation access
│   └─ validation.py        Data quality checks & imputation
│
└─ fplx/api/
    └─ interface.py         FPLModel: high-level orchestrator
```

---

## Inference Pipeline

The core contribution. Each player is modeled independently:

```
                         NewsSignal ──────┐
                                          ▼
Points history ──► HMM (discrete) ──► Transition perturbation
     │               │                    │
     │               ▼                    │
     │          P(S_t | data)             │
     │          {Injured, Slump,          │
     │           Average, Good, Star}     │
     │               │                    │
     ├──► Kalman Filter (continuous) ◄────┘
     │          x̂_t, P_t              Process noise shock
     │               │
     │          FixtureSignal ──► Observation noise modulation
     │               │
     ▼               ▼
  Fusion (inverse-variance weighting)
     │
     ▼
  E[P], Var[P]  ──►  ILP Optimizer
```

**HMM** tracks 5 discrete form states: `{Injured, Slump, Average, Good, Star}`. Each state has a Gaussian emission model defining expected points. The Forward-Backward algorithm computes smoothed posteriors; Viterbi decoding finds the most likely state sequence. Baum-Welch (EM) learns transition and emission parameters from data.

**Kalman Filter** tracks a continuous latent variable representing the player's true point potential. A random-walk state model captures gradual form drift. The filter produces optimal minimum-MSE estimates with uncertainty bounds at each gameweek.

**News injection** is where signals enter the inference. When `NewsSignal` classifies news as "ruled out," the HMM transition matrix for that timestep is perturbed (10x boost toward Injured state), and the Kalman Filter's process noise is inflated (true form may have jumped). This is fundamentally different from the post-hoc scalar multiplier used in static pipelines.

**Fixture injection** modulates the Kalman Filter's observation noise. Harder opponents produce noisier point observations (R multiplied by 1.5 for difficulty-5 fixtures, 0.8 for difficulty-1).

**Fusion** combines HMM and Kalman outputs via inverse-variance weighting. The fused variance is always lower than either component alone. The output is a `(mean, variance)` pair per player that feeds into the optimizer.

---

## Quick Start

### Installation

```bash
git clone https://github.com/fnhirwa/fantasypl.git
cd fantasypl
pip install -e "."

# With ML models
pip install -e ".[ml]"

# With ILP optimizer
pip install -e ".[optimization]"

# Everything
pip install -e ".[all]"
```

### Inference Pipeline (recommended)

```python
from fplx import FPLModel

model = FPLModel(
    budget=100,
    horizon=1,
    formation="auto",
    config={"model_type": "inference"}
)

model.load_data(source="api")   # fetches FPL API + collects news
model.fit()                     # runs HMM + KF + fusion per player

squad = model.select_best_11()
print(squad.summary())

# Uncertainty is available for downstream use
for pid, ep in sorted(model.expected_points.items(), key=lambda x: -x[1])[:10]:
    var = model.expected_variance[pid]
    print(f"  Player {pid}: E[P]={ep:.2f}, std={var**0.5:.2f}")
```

### Legacy Pipeline (baselines)

```python
from fplx import FPLModel

# Rolling average baseline
model = FPLModel(budget=100, config={"model_type": "baseline"})
model.load_data(source="api")
model.fit()
squad = model.select_best_11()

# XGBoost (requires fplx[ml])
model = FPLModel(budget=100, config={"model_type": "xgboost"})
```

### Direct Inference (without FPLModel)

```python
import numpy as np
from fplx.inference.pipeline import PlayerInferencePipeline
from fplx.signals.news import NewsSignal

# Player's gameweek-by-gameweek points
points = np.array([6, 8, 5, 7, 9, 3, 6, 8, 7, 5, 0, 0, 0, 1, 2])

pipeline = PlayerInferencePipeline()
pipeline.ingest_observations(points)

# Inject injury news at gameweek 11
news = NewsSignal().generate_signal("Ruled out for 3 weeks")
pipeline.inject_news(news, timestep=10)

result = pipeline.run()
ep_mean, ep_var = pipeline.predict_next()

print(f"Next GW: E[P]={ep_mean:.2f}, std={ep_var**0.5:.2f}")
print(f"Current state: {result.viterbi_path[-1]}")  # 0=Injured,...,4=Star
print(f"P(Injured): {result.smoothed_beliefs[-1, 0]:.3f}")
```

---

## Examples

Run from the repo root. Each example hits the live FPL API.

```bash
# 1. Inspect what news data the API provides (run first)
python tests/test_fpl_news_api.py

# 2. Test news → inference pipeline on flagged players
python tests/test_news_inference.py

# 3. Deep-dive visualization for a specific player
python tests/test_play_inf_viz.py
python tests/test_play_inf_viz.py --player-id 301

# 4. Batch inference with squad ranking and risk adjustment
python tests/test_batch_inference.py --top 50
python tests/test_batch_inference.py --position MID
```

---

## News Signal Integration

FPLX uses news data from the FPL API itself. Every player in the `bootstrap-static` response includes:

| Field | Example | Usage |
|-------|---------|-------|
| `news` | `"Knee injury - expected back 01 Feb"` | Parsed by `NewsSignal` |
| `status` | `"i"` (injured), `"d"` (doubtful), `"a"` (available) | Classified into perturbation category |
| `chance_of_playing_next_round` | `25` (percent) | Augments news text |

`NewsCollector` snapshots this data per gameweek (persisted as JSON in `~/.fplx/news/`). `NewsSnapshot.to_news_signal_input()` enriches the raw text with status codes and chance percentages, then routes through the existing `NewsSignal.generate_signal()` parser.

The parsed signal maps to inference perturbations:

| News Category | HMM Perturbation | KF Process Noise |
|---------------|-------------------|------------------|
| Unavailable (`"ruled out"`, status=`i`) | Injured state ×10 | Q × 5.0 |
| Doubtful (`"late fitness test"`, status=`d`) | Injured ×3, Slump ×2 | Q × 2.0 |
| Rotation (`"rotation risk"`, `"benched"`) | Slump ×2, Average ×1.5 | Q × 1.5 |
| Positive (`"back in training"`, status=`a`) | Good ×2, Star ×1.5 | Q × 1.0 |
| Neutral (no news) | No perturbation | No change |

No external scraping required.

---

## Configuration

All components are configurable via the `config` dict:

```python
config = {
    # Prediction mode: "inference", "baseline", "form_based",
    # "ridge", "xgboost", "lightgbm", "ensemble"
    "model_type": "inference",

    # Optimizer: "greedy" or "ilp" (requires pulp)
    "optimizer": "greedy",

    # Inference parameters (only used when model_type="inference")
    "inference": {
        "hmm_params": {
            # Override default transition matrix, emission params, initial dist
        },
        "kf_params": {
            "Q": 1.0,    # Process noise (form drift rate)
            "R": 4.0,    # Observation noise (weekly point variance)
            "x0": 4.0,   # Initial state estimate
            "P0": 2.0,   # Initial uncertainty
        },
    },

    # Feature engineering (used by legacy ML models)
    "feature_engineering": {
        "rolling_windows": [3, 5, 10],
        "lag_periods": [1, 2, 3],
        "ewma_alphas": [0.3, 0.5],
    },

    # Signal weights (used by StatsSignal)
    "signals": {
        "stats_weights": {
            "points_mean": 0.3,
            "xG_mean": 0.15,
            "xA_mean": 0.15,
            "minutes_consistency": 0.2,
            "form_trend": 0.2,
        },
    },
}
```

---

## Project Structure

```
fantasypl/
├── fplx/                    Python package
│   ├── api/                 High-level interface (FPLModel)
│   ├── core/                Domain objects (Player, Squad, Matchweek)
│   ├── data/                Data loading & news collection
│   ├── inference/           HMM, Kalman Filter, fusion, pipeline
│   ├── models/              Baseline, regression, ensemble models
│   ├── selection/           Squad optimization (Greedy, ILP)
│   ├── signals/             News, fixture, stats signal processing
│   ├── timeseries/          Feature engineering transforms
│   └── utils/               Config, validation, imputation
├── examples/                Runnable examples (hit live FPL API)
├── tests/                   Test suite
├── pyproject.toml           Build config, dependencies, tool settings
└── README.md
```

---

## Development

```bash
git clone https://github.com/fnhirwa/fantasypl.git
cd fantasypl
python -m venv env
source env/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format
ruff format fplx/
ruff check fplx/ --fix
```

---

## Roadmap

### Done
- [x] Data loading from FPL API
- [x] Time-series feature engineering (40+ features)
- [x] Baseline models (rolling mean, EWMA, form-based)
- [x] ML models (Ridge, XGBoost, LightGBM)
- [x] Squad optimization (Greedy & ILP)
- [x] HMM for discrete form state tracking
- [x] Kalman Filter for continuous point potential
- [x] HMM + KF fusion with inverse-variance weighting
- [x] News signal injection into inference (dynamic transition perturbation)
- [x] Fixture difficulty → KF observation noise modulation
- [x] Per-gameweek news collection & persistence
- [x] Baum-Welch (EM) for HMM parameter learning

### In Progress
- [ ] Wire fixture data into `_fit_inference()` pipeline
- [ ] Mean-variance objective in ILP (robust optimization)
- [ ] Backtest engine (replay season with weekly news injection)

### Planned
- [ ] Shared HMM parameters by position (pool data across players)
- [ ] Captain selection with uncertainty-aware logic
- [ ] Transfer optimization (multi-period planning)
- [ ] Calibration analysis (coverage of 95% credible intervals)
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Web dashboard

---

## Research Context

This project implements the pipeline titled:

> *FPLX: A Framework for Stochastic Inference and Constrained Optimization in High-Variance Sports Environments*

The key insight is treating FPL squad selection as a **Perceive → Infer → Act** loop where uncertainty propagates end-to-end from observation noise through to squad selection. This contrasts with standard approaches that decouple forecasting from optimization and discard uncertainty at the interface.

References: Matthews et al. (AAAI 2012), Tamimi & Tran (IJCSS 2025), Ramezani (arXiv 2025), Brill et al. (arXiv 2024).

---

## License

MIT — see [LICENSE](LICENSE).

## Author

**Felix Hirwa Nshuti** — [hirwanshutiflx@gmail.com](mailto:hirwanshutiflx@gmail.com) — [@fnhirwa](https://github.com/fnhirwa)
