# FPLX - Project Structure

Complete reference for the FPLX library architecture.

---

## Directory Structure

```
fantasypl/
├── fplx/                          # New FPLX library (v0.2.0)
│   ├── __init__.py
│   ├── data/                      # Data loading & schemas
│   │   ├── __init__.py
│   │   ├── loaders.py            # FPL API & CSV loaders
│   │   └── schemas.py            # Player, Squad, Matchweek schemas
│   ├── timeseries/               # Time-series processing
│   │   ├── __init__.py
│   │   ├── transforms.py         # Rolling, lag, EWMA features
│   │   └── features.py           # Feature engineering pipeline
│   ├── signals/                  # Signal generation
│   │   ├── __init__.py
│   │   ├── stats.py             # Statistical signals
│   │   ├── news.py              # News/injury parsing
│   │   └── fixtures.py          # Fixture difficulty
│   ├── models/                   # Prediction models
│   │   ├── __init__.py
│   │   ├── baseline.py          # Heuristic models
│   │   ├── regression.py        # ML regressors
│   │   ├── ensemble.py          # Ensemble models
│   │   └── rolling_cv.py        # Time-series CV
│   ├── selection/                # Squad optimization
│   │   ├── __init__.py
│   │   ├── constraints.py       # Formation & budget rules
│   │   └── optimizer.py         # Greedy & ILP optimizers
│   ├── api/                      # Public API
│   │   ├── __init__.py
│   │   └── interface.py         # FPLModel class
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── config.py            # Configuration
│       └── validation.py        # Data validation
│
├── fantasypl/                     # Legacy library (v0.1.0)
│   ├── __init__.py
│   ├── api.py                    # (empty)
│   ├── parse_data.py            # Original FPL API wrapper
│   ├── search.py
│   └── utils.py
│
├── examples/                      # Usage examples
│   ├── basic_usage.py
│   ├── advanced_optimization.py
│   └── ml_models.py
│
├── tests/                         # Test suite (TODO)
│   └── test_*.py
│
├── pyproject.toml                 # Modern packaging config
├── setup.py                       # Legacy setup (deprecated)
├── requirements.txt               # Dependencies
├── README.md                      # Main documentation
├── MIGRATION.md                   # Migration guide
└── LICENSE
```

---

## Module Responsibilities

### `fplx.data`
- Load data from FPL API
- Load from CSV files
- Define data schemas (Player, Squad, Matchweek)
- Cache management

**Key Classes**: `FPLDataLoader`, `Player`, `Squad`, `Matchweek`

---

### `fplx.timeseries`
- Rolling window features
- Lag features
- EWMA (exponential moving average)
- Trend detection
- Consistency metrics

**Key Classes**: `FeatureEngineer`

**Key Functions**: `add_rolling_features`, `add_lag_features`, `add_ewma_features`

---

### `fplx.signals`
- Statistical performance signals
- News/injury text parsing
- Fixture difficulty calculation
- Signal aggregation

**Key Classes**: `StatsSignal`, `NewsSignal`, `NewsParser`, `FixtureSignal`

---

### `fplx.models`
- Baseline heuristic models
- ML regression models (Ridge, XGBoost, LightGBM)
- Ensemble models
- Rolling cross-validation

**Key Classes**: `BaselineModel`, `FormBasedModel`, `RegressionModel`, `EnsembleModel`, `RollingCV`

---

### `fplx.selection`
- Formation constraints
- Budget constraints
- Team diversity rules
- Greedy optimization
- ILP (Integer Linear Programming) optimization

**Key Classes**: `GreedyOptimizer`, `ILPOptimizer`, `FormationConstraints`, `BudgetConstraint`

---

### `fplx.api`
- High-level interface
- Workflow orchestration
- Configuration management
- Model/optimizer switching

**Key Classes**: `FPLModel`

---

### `fplx.utils`
- Configuration management
- Data validation
- Quality checks
- Missing data imputation

**Key Classes**: `Config`

**Key Functions**: `validate_data`, `check_data_quality`, `impute_missing`

---

## Key Design Patterns

### 1. Modular Architecture
Each module has a single, clear responsibility

### 2. Class-Based APIs
Consistent interfaces across modules

### 3. Configuration-Driven
Flexible configuration system

### 4. Type Hints
All functions and methods use type hints

### 5. Logging
Comprehensive logging for debugging

### 6. Extensibility
Easy to add custom signals, models, optimizers

---

## Data Flow

```
FPL API
   ↓
FPLDataLoader → Player objects
   ↓
FeatureEngineer → Enriched timeseries
   ↓
Signals (Stats, News, Fixtures) → Adjustment factors
   ↓
Models (Baseline, ML, Ensemble) → Expected points
   ↓
Optimizer (Greedy, ILP) → Optimal squad
   ↓
Squad object → Summary & visualization
```

---

## Configuration Structure

```python
{
    'model_type': 'baseline',
    'optimizer': 'greedy',
    'budget': 100.0,
    'horizon': 1,
    'formation': 'auto',
    'feature_engineering': {
        'rolling_windows': [3, 5, 10],
        'lag_periods': [1, 2, 3],
        'ewma_alphas': [0.3, 0.5],
    },
    'signals': {
        'stats_weights': {
            'points_mean': 0.3,
            'xG_mean': 0.15,
            'xA_mean': 0.15,
            'minutes_consistency': 0.2,
            'form_trend': 0.2,
        },
    },
}
```

---

## Testing Strategy

### Unit Tests
- Individual module functions
- Data loading & parsing
- Feature engineering
- Signal generation

### Integration Tests
- Full pipeline tests
- API endpoint tests
- Optimization validation

### Validation Tests
- Constraint satisfaction
- Budget compliance
- Formation rules

---

## Performance Considerations

### Caching
- Bootstrap data cached locally
- Player history cached per session

### Lazy Loading
- Enrich player history only when needed
- Optional ML models

### Optimization
- Greedy: O(n log n) - fast
- ILP: NP-hard - optimal but slower

---

## Extension Points

### Custom Signals
Implement signal interface:
```python
class CustomSignal:
    def compute_signal(self, player_data):
        # Your logic
        return score
```

### Custom Models
Inherit from base:
```python
class CustomModel(BaselineModel):
    def predict(self, player_data):
        # Your logic
        return prediction
```

### Custom Optimizers
Inherit from base:
```python
class CustomOptimizer(SquadOptimizer):
    def optimize(self, players, expected_points, formation):
        # Your logic
        return squad
```

---

## Dependencies

### Core
- numpy (arrays & numerical operations)
- pandas (data manipulation)
- requests (API calls)

### Optional - ML
- scikit-learn (preprocessing, metrics)
- xgboost (gradient boosting)
- lightgbm (light gradient boosting)

### Optional - Optimization
- pulp (linear programming)

### Optional - Deep Learning
- pytorch (neural networks)
- tensorflow (alternative)

---

## Version History

### v0.2.0 (Current)
- Complete restructure with modular architecture
- Time-series feature engineering
- ML models integration
- ILP optimization
- Clean API interface

For detailed API documentation, see individual module docstrings.
