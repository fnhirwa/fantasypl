# Hidden Markov Model

The HMM tracks discrete form states for each player. It handles **abrupt regime shifts** — an injury doesn't gradually reduce points; it causes a sudden jump to a different operating regime.

## State Space

| Index | State | Emission $\mu$ | Emission $\sigma$ | Interpretation |
|-------|-------|:---------:|:---------:|----------------|
| 0 | Injured | 0.5 | 0.5 | Out or minimal cameo |
| 1 | Slump | 2.0 | 1.0 | Playing but underperforming |
| 2 | Average | 4.0 | 1.5 | Typical output |
| 3 | Good | 6.0 | 1.5 | Above-average returns |
| 4 | Star | 8.5 | 2.0 | Exceptional gameweek |

These are defaults. [`HMMInference.fit()`][fplx.inference.hmm.HMMInference.fit] re-estimates them via Baum-Welch.

## Transition Matrix

States are "sticky" (high self-transition) with gradual drift between adjacent states:

```
         Inj   Slump  Avg    Good   Star
Injured  [0.60, 0.25, 0.10, 0.05, 0.00]
Slump    [0.05, 0.50, 0.35, 0.08, 0.02]
Average  [0.02, 0.10, 0.55, 0.25, 0.08]
Good     [0.02, 0.05, 0.15, 0.55, 0.23]
Star     [0.01, 0.02, 0.07, 0.30, 0.60]
```

!!! note "Design Decision"
    Injured is structurally different from the other states — it represents availability, not form level. We keep it in the state space (rather than modeling it separately) because the transition dynamics between injury and form states are informative: an injured player typically recovers through Slump → Average, not directly to Star.

## Algorithms

| Algorithm | Computes | Complexity | Use |
|-----------|----------|:----------:|-----|
| Forward | $P(S_t \mid y_{1:t})$ | $O(TN^2)$ | Online filtering |
| Forward-Backward | $P(S_t \mid y_{1:T})$ | $O(TN^2)$ | Smoothed posteriors for fusion |
| Viterbi | Most likely state sequence | $O(TN^2)$ | Diagnostics, visualization |
| Baum-Welch | Learns $A$, $\mu$, $\sigma$ | $O(I \cdot TN^2)$ | Parameter training from data |
| `predict_next` | $E[Y_{T+1}]$, $\text{Var}[Y_{T+1}]$ | $O(N^2)$ | Forecast for optimizer |

## News Perturbation

When news is injected at timestep $t$, the transition matrix **for that timestep only** is modified:

$$A_t[i, j] \leftarrow A[i, j] \times \big(1 + c \cdot (b_j - 1)\big)$$

where $b_j$ is the boost factor for target state $j$ and $c \in [0, 1]$ is the news confidence. Each row is then renormalized to sum to 1.

This means even from the Star state, an "unavailable" signal makes transitioning to Injured 10× more likely — but the observation evidence still has a say. The perturbation is **not** a hard override.

```python
hmm.inject_news_perturbation(
    timestep=20,
    state_boost={0: 10.0, 1: 2.0},  # boost Injured×10, Slump×2
    confidence=0.9
)
```

## Baum-Welch Parameter Learning

The E-step uses Forward-Backward (already implemented). The M-step re-estimates:

- **Initial distribution** $\pi$ from $\gamma_0$
- **Transition matrix** $A[i,j]$ from expected transition counts $\xi_{t}[i,j]$
- **Emission parameters** $(\mu_s, \sigma_s)$ from $\gamma$-weighted observations

!!! warning "Data Requirements"
    Baum-Welch needs sufficient data per player. Players with <10 gameweeks may not have enough. A future improvement is learning shared parameters across all players in the same position, then fine-tuning per player.

## API

::: fplx.inference.hmm.HMMInference
    options:
      show_source: false
      members:
        - forward
        - forward_backward
        - viterbi
        - predict_next
        - inject_news_perturbation
        - fit
