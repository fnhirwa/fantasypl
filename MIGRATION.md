# fplx

`fplx` is a Python library for fantasy football analytics.

---

## Architecture
`fplx` has a modular architecture with 7 specialized modules.

---

## Installation

```bash
# Basic
pip install -e .

# With ML support
pip install -e ".[ml]"

# With all features
pip install -e ".[all]"
```

---

## Usage

### Basic API Access

```python
from fplx.data.loaders import FPLDataLoader

loader = FPLDataLoader()
data = loader.fetch_bootstrap_data()
players = loader.load_players()
```

---

### Complete Workflow

```python
from fplx import FPLModel

# Initialize
model = FPLModel(budget=100, horizon=1, formation="auto")

# Load & process
model.load_data(source='api')
model.fit()

# Get optimal squad
squad = model.select_best_11()
print(squad.summary())
```

---

## Features

### 1. Time-Series Analysis
```python
from fplx.timeseries import FeatureEngineer

engineer = FeatureEngineer()
enriched = engineer.fit_transform(player_data)
```

### 2. Signal Processing
```python
from fplx.signals import NewsSignal

news = NewsSignal()
signal = news.generate_signal("Doubtful for next match")
# {'availability': 0.5, 'minutes_risk': 0.3, ...}
```

### 3. ML Models
```python
from fplx.models import RegressionModel, EnsembleModel

model = RegressionModel(model_type='xgboost')
ensemble = EnsembleModel([model1, model2], weights=[0.6, 0.4])
```

### 4. Optimization
```python
from fplx.selection import ILPOptimizer

optimizer = ILPOptimizer(budget=100)
squad = optimizer.optimize(players, expected_points)
```

---

## Module Mapping

| Old Module | New Module | Purpose |
|-----------|------------|---------|
| `fantasypl.parse_data` | `fplx.data.loaders` | Data loading |
| `fantasypl.api` | `fplx.api.interface` | High-level API |
| N/A | `fplx.timeseries` | Feature engineering |
| N/A | `fplx.signals` | Signal generation |
| N/A | `fplx.models` | ML models |
| N/A | `fplx.selection` | Optimization |
| N/A | `fplx.utils` | Utilities |

---

## Backward Compatibility

The old `fantasypl` module is still available for basic API access:

```python
# Still works (legacy)
from fantasypl.parse_data import FPL_GENERIC_INFO
```

But we recommend migrating to the new `fplx` interface for:
- Better performance
- More features
- Active development
- Better documentation

---

## Examples

See the `/examples` directory:
- `basic_usage.py` - Simple migration example
- `advanced_optimization.py` - Using new features
- `ml_models.py` - ML-based predictions

---

## Questions?

Open an issue on GitHub or contact hirwanshutiflx@gmail.com
