# Constrained Optimization

!!! abstract
    This component implements the squad-selection optimization.
    It takes the inference outputs from Inference (predicted points with uncertainty) and solves a constrained
    integer program to select the optimal 11-player squad.

## Problem Statement

Given $n$ players with predicted expected points $E[P_i]$ and prediction variance $\text{Var}[P_i]$ from the inference pipeline, select an 11-player squad that maximizes total expected points subject to FPL constraints.

This is a **multi-dimensional knapsack problem** — NP-hard in general, but tractable for FPL-sized instances via ILP solvers.

## Connection to Inference

The inference pipeline produces two quantities per player:

``` mermaid
graph LR
    INF["Inference"] -->|"E[P_i]"| OBJ["Objective coefficients"]
    INF -->|"Var[P_i]"| ROB["Robust penalty term"]
    OBJ --> ILP["ILP Solver"]
    ROB --> ILP
    ILP --> SQUAD["Optimal Squad"]
```

- **Point forecast** $E[P_i]$ → objective coefficient in the ILP
- **Uncertainty** $\text{Var}[P_i]$ → risk penalty in the robust variant

Without Inference, the optimizer receives flat point estimates with no uncertainty information. With Inference, the optimizer can hedge against unreliable forecasts.

## Sections

- [**Squad Selection ILP**](squad-ilp.md) — Problem formulation, constraints, implementation
- [**Duality & Shadow Prices**](duality.md) — LP relaxation, KKT conditions, marginal value analysis
- [**Numerical Experiments**](experiments.md) — Branch-and-bound vs. first-order methods, convergence comparison
