# FPLX - Fantasy Premier League Analysis & Optimization

**Production-ready Python library for FPL time-series analysis, player scoring, and optimal squad selection.**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

FPLX is a comprehensive Python library that combines:

- **Time-series analysis** of player performance data
- **News & injury signal processing** for availability predictions
- **Machine learning models** for expected points forecasting
- **Optimization algorithms** for squad selection (ILP & Greedy)
- **Clean, extensible API** for integration and customization

Think of it as `scikit-learn + pandas + FPL domain logic`.

---

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install fplx

# With ML support
pip install fplx[ml]

# With optimization support
pip install fplx[optimization]

# Everything
pip install fplx[all]
```

### Usage

```python
from fplx import FPLModel

# Initialize model
model = FPLModel(
    budget=100,
    horizon=3,
    formation="auto"
)

# Load data from FPL API
model.load_data(source='api')

# Fit model and generate predictions
model.fit()

# Select optimal 11-player squad
squad = model.select_best_11()

# View squad summary
print(squad.summary())
```

---

## 🏗️ Architecture

```
fplx/
├── data/           # Data loading & schemas
├── timeseries/     # Feature engineering & transforms
├── signals/        # Stats, news, fixture signals
├── models/         # ML models (baseline, regression, ensemble)
├── selection/      # Squad optimization algorithms
├── api/            # High-level user interface
└── utils/          # Configuration & validation
```

---

## 📊 Features

### 1. Data Loading

```python
from fplx.data.loaders import FPLDataLoader

loader = FPLDataLoader()
players = loader.load_players()  # From FPL API
history = loader.load_player_history(player_id=123)
```

### 2. Time-Series Feature Engineering

```python
from fplx.timeseries import FeatureEngineer

engineer = FeatureEngineer()
enriched_data = engineer.fit_transform(player_timeseries)

# Generates: rolling means, lags, EWMA, trends, consistency metrics
```

### 3. Signal Generation

```python
from fplx.signals import StatsSignal, NewsSignal, FixtureSignal

# Statistical signals
stats_signal = StatsSignal()
score = stats_signal.compute_signal(player_data)

# News signals
news_signal = NewsSignal()
signal = news_signal.generate_signal("Out for 2-3 weeks")
# Returns: {'availability': 0.0, 'minutes_risk': 0.0, 'adjustment_factor': 0.0}

# Fixture signals
fixture_signal = FixtureSignal()
advantage = fixture_signal.compute_fixture_advantage(
    team="Arsenal",
    upcoming_opponents=["Brighton", "Everton"],
    is_home=[True, False]
)
```

### 4. Prediction Models

```python
from fplx.models import BaselineModel, RegressionModel, EnsembleModel

# Baseline (fast)
baseline = BaselineModel(method='rolling_mean', window=5)
predictions = baseline.batch_predict(players_data)

# ML Regression
ml_model = RegressionModel(model_type='xgboost')
predictions = ml_model.fit_predict(y, X)

# Ensemble
ensemble = EnsembleModel(
    models=[baseline, ml_model],
    weights=[0.4, 0.6]
)
```

### 5. Squad Optimization

```python
from fplx.selection import GreedyOptimizer, ILPOptimizer

# Greedy (fast)
optimizer = GreedyOptimizer(budget=100)
squad = optimizer.optimize(players, expected_points, formation="3-4-3")

# ILP (optimal)
optimizer = ILPOptimizer(budget=100)
squad = optimizer.optimize(players, expected_points)
```

---

## 🔧 Advanced Usage

### Custom Configuration

```python
config = {
    'model_type': 'xgboost',
    'optimizer': 'ilp',
    'feature_engineering': {
        'rolling_windows': [3, 5, 10],
        'lag_periods': [1, 2, 3],
    }
}

model = FPLModel(budget=100, config=config)
```

### Loading Custom Data

```python
# From CSV
model.load_data(source='csv', filepath='my_data.csv')

# News data
model.load_news('news.json')

# Stats data
model.load_stats('player_stats.csv')
```

### Model Selection

```python
# Switch to different models
model.set_model('xgboost', n_estimators=200, max_depth=5)
model.set_optimizer('ilp')
```

---

## 📈 Time-Series Approach

FPLX uses sophisticated time-series methods adapted from the [MLSP Final Project](https://github.com/jahyun03/11755-MLSP-Final-Project):

- **Rolling Cross-Validation**: Time-aware train/test splits
- **Feature Engineering**: 40+ engineered features per player
- **Ensemble Models**: Combine multiple predictors
- **Signal Integration**: Merge stats, news, and fixtures

### Expected Points Formula

```
AdjustedScore = BaselinePoints × NewsAvailability × (1 - MinutesRisk) × FixtureAdvantage
```

---

## 🧪 Examples

Check out the `/examples` directory:

- `basic_usage.py` - Simple squad selection
- `custom_signals.py` - Add custom signals
- `advanced_optimization.py` - ILP optimization with constraints
- `backtesting.py` - Evaluate historical performance

---

## 🛠️ Development

### Setup

```bash
git clone https://github.com/fnhirwa/fantasypl.git
cd fantasypl
python -m venv env
source env/bin/activate
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Format Code

```bash
black fplx/
mypy fplx/
```

---

## 📝 Roadmap

### Phase 1 - MVP ✅
- [x] Data loading from FPL API
- [x] Time-series feature engineering
- [x] Baseline & ML models
- [x] Squad optimization (Greedy & ILP)
- [x] Clean API interface

### Phase 2 - Intelligence 🚧
- [ ] Advanced news parsing with NLP
- [ ] Fixture difficulty modeling
- [ ] Captain selection logic
- [ ] Transfer optimization

### Phase 3 - Advanced 🔮
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Backtesting engine
- [ ] Web dashboard
- [ ] Real-time updates

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- FPL API for data
- Time-series methods from [11755-MLSP-Final-Project](https://github.com/jahyun03/11755-MLSP-Final-Project)
- Inspiration from the FPL community

---

## 📧 Contact

**Felix Hirwa Nshuti**  
Email: hirwanshutiflx@gmail.com  
GitHub: [@fnhirwa](https://github.com/fnhirwa)

---

**Built with ❤️ for the FPL community**
