# Probabilistic Inference

!!! abstract
    This component implements the inference pipeline.
    It demonstrates **Perceiving** (NLP signal processing), **Belief updating** (HMM/Kalman filtering),
    and feeds into the **Acting** stage (Optimization).

## Motivation

Fantasy managers face a high-variance environment: noisy match-day data and sudden regime shifts (injuries, transfers) make performance unpredictable. Standard FPL projections treat each week's data independently and ignore latent factors. FPLX models each player's underlying form as a **hidden state** that evolves over time, and uses probabilistic filters to separate true signal from noise.

## Approach

Each player is modeled through a dual-filter system:

``` mermaid
graph TD
    O["Observed Points<br/>(noisy)"] --> HMM["HMM<br/>Discrete form states"]
    O --> KF["Kalman Filter<br/>Continuous potential"]
    N["News Signal<br/>(injury, suspension)"] -->|Transition perturbation| HMM
    N -->|Process noise shock| KF
    F["Fixture Signal<br/>(difficulty)"] -->|Observation noise| KF
    HMM -->|"P(State), E[Y], Var[Y]"| FU[Inverse-Variance Fusion]
    KF -->|"x̂, P"| FU
    FU -->|"E[P], Var[P]"| OUT["Per-player forecast<br/>with uncertainty"]
```

**HMM** captures abrupt regime shifts (injury → slump → recovery) via discrete hidden states.

**Kalman Filter** captures gradual trends in scoring output via continuous latent tracking.

**News injection** is the key differentiator from static pipelines: signals enter *inside* the inference (perturbing transition probabilities and noise parameters), not as a post-hoc multiplier.

**Fusion** combines both outputs via inverse-variance weighting, producing forecasts that are always more certain than either component alone.

## Sections

- [**Inference Pipeline**](pipeline.md) — Full pipeline overview, fusion math, and usage
- [**News Signals**](news-signals.md) — How FPL API news feeds into inference, perturbation mapping
- [**HMM Details**](hmm.md) — State space, transition dynamics, Baum-Welch learning
- [**Kalman Filter**](kalman.md) — State-space model, adaptive noise, RTS smoother
