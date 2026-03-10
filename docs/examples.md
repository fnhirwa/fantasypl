# Examples

All examples hit the live FPL API. Run from the repo root.

## Inspect News Data

Fetches `bootstrap-static` and prints every player with active news, grouped by status. Saves raw data for offline use.

```bash
python tests/test_fpl_news_api.py
```

**What it shows:** raw news text, status codes, chance percentages — the exact signal your inference pipeline works with.

## Test News → Inference

The core validation. Picks flagged players + clean controls, runs inference with and without news injection, prints a comparison table.

```bash
python tests/test_news_inference.py
```

**What to check:** injured players should have negative E[P] shift, doubtful players slightly negative, controls zero.

## Visualize Player Inference

Deep dive on one player. Generates a 4-panel diagnostic plot (fused estimates, Viterbi path, smoothed posteriors, uncertainty).

```bash
python tests/test_play_inf_viz.py
python tests/test_play_inf_viz.py--player-id 301
```

## Batch Inference

Full squad ranking with news injection and risk-adjusted scoring.

```bash
python tests/test_batch_inference.py--top 50
python tests/test_batch_inference.py --position MID
```

## Toy Inference

Synthetic data with known ground truth to validate the HMM + Kalman + Fusion pipeline. No API needed.

```bash
python examples/toy_inference.py
```
