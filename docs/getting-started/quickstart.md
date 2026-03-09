# Quick Start

## Inference Pipeline (Recommended)

The inference pipeline runs an HMM + Kalman Filter per player, injects news signals, and produces predictions with uncertainty.

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
```

After fitting, both expected points and variance are available:

```python
for pid, ep in sorted(model.expected_points.items(), key=lambda x: -x[1])[:10]:
    var = model.expected_variance[pid]
    print(f"Player {pid}: E[P]={ep:.2f}, std={var**0.5:.2f}")
```

## Direct Inference (Without FPLModel)

For more control, use the inference pipeline directly:

```python
import numpy as np
from fplx.inference.pipeline import PlayerInferencePipeline
from fplx.signals.news import NewsSignal

points = np.array([6, 8, 5, 7, 9, 3, 6, 8, 7, 5, 0, 0, 0, 1, 2])

pipeline = PlayerInferencePipeline()
pipeline.ingest_observations(points)

# Inject injury news at gameweek 11
news = NewsSignal().generate_signal("Ruled out for 3 weeks")
pipeline.inject_news(news, timestep=10)

result = pipeline.run()
ep_mean, ep_var = pipeline.predict_next()
print(f"Next GW: E[P]={ep_mean:.2f}, std={ep_var**0.5:.2f}")
```

## Legacy Baselines

```python
from fplx import FPLModel

# Rolling average
model = FPLModel(budget=100, config={"model_type": "baseline"})
model.load_data(source="api")
model.fit()
squad = model.select_best_11()
```

!!! note
    The legacy pipeline uses `FeatureEngineer` + `BaselineModel` and applies news as a post-hoc scalar multiplier. The inference pipeline integrates news directly into the probabilistic model.
