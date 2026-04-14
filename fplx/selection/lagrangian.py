"""Lagrangian dual decomposition for FPL squad selection.

Relaxes the budget constraint into the objective and solves via
subgradient ascent. The inner problem decomposes into per-position
sorting problems, each solvable in O(n log n).

This provides:
  - A dual upper bound on the ILP optimum
  - A near-optimal primal solution via rounding
  - Convergence diagnostics for the 18-660 report
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fplx.core.player import Player
from fplx.core.squad import FullSquad, Squad
from fplx.selection.constraints import (
    FormationConstraints,
    SquadQuotas,
)

logger = logging.getLogger(__name__)


@dataclass
class LagrangianResult:
    """Convergence diagnostics for the Lagrangian solver."""

    full_squad: Optional[FullSquad] = None
    primal_objective: float = 0.0
    dual_bound: float = 0.0
    duality_gap: float = 0.0
    n_iterations: int = 0
    converged: bool = False
    solve_time: float = 0.0
    # Per-iteration tracking for convergence plots
    dual_history: list[float] = field(default_factory=list)
    primal_history: list[float] = field(default_factory=list)
    lambda_history: list[float] = field(default_factory=list)
    budget_slack_history: list[float] = field(default_factory=list)


class LagrangianOptimizer:
    """
    Lagrangian relaxation for the FPL squad selection ILP.

    Relaxes the budget constraint into the objective:

        L(lambda) = max_{x in X} sum_i (mu_i - lambda * c_i) * x_i + lambda * B

    where X encodes squad size, position quotas, and team caps.
    The inner maximization decomposes: for each position, select
    the top-k players by modified score (mu_i - lambda * c_i).

    The dual problem min_{lambda >= 0} L(lambda) is solved via
    subgradient ascent.

    Parameters
    ----------
    budget : float
        Total budget (default 100.0).
    max_from_team : int
        Maximum players from same club.
    max_iter : int
        Maximum subgradient iterations.
    tol : float
        Convergence tolerance on duality gap.
    risk_aversion : float
        Mean-variance penalty (same as ILP).
    """

    def __init__(
        self,
        budget: float = 100.0,
        max_from_team: int = 3,
        max_iter: int = 200,
        tol: float = 0.01,
        risk_aversion: float = 0.0,
    ):
        self.budget = budget
        self.max_from_team = max_from_team
        self.max_iter = max_iter
        self.tol = tol
        self.risk_aversion = risk_aversion

    def _compute_modified_scores(
        self,
        players: list[Player],
        expected_points: dict[int, float],
        expected_variance: Optional[dict[int, float]],
        lam: float,
    ) -> dict[int, float]:
        """Compute mu_i - lambda * c_i - risk_penalty for each player."""
        scores = {}
        for p in players:
            ep = expected_points.get(p.id, 0.0)
            penalty = 0.0
            if self.risk_aversion > 0 and expected_variance:
                var_i = expected_variance.get(p.id, 0.0)
                penalty = self.risk_aversion * np.sqrt(max(var_i, 0.0))
            scores[p.id] = ep - penalty - lam * p.price
        return scores

    def _solve_inner(
        self,
        players: list[Player],
        scores: dict[int, float],
    ) -> tuple[list[Player], list[Player]]:
        """
        Solve the inner problem: select 15-player squad + 11-player lineup
        subject to position quotas and team caps (but NOT budget).

        Greedy per-position selection by modified score.
        Returns (squad_15, lineup_11).
        """
        # Group by position
        by_pos: dict[str, list[Player]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for p in players:
            by_pos[p.position].append(p)

        # Sort each position by modified score (descending)
        for pos in by_pos:
            by_pos[pos].sort(key=lambda p: scores[p.id], reverse=True)

        # Greedy squad selection with team cap enforcement
        squad_quotas = SquadQuotas.QUOTAS  # {GK:2, DEF:5, MID:5, FWD:3}
        squad: list[Player] = []
        team_counts: dict[str, int] = {}

        for pos in ["GK", "DEF", "MID", "FWD"]:
            count = 0
            for p in by_pos[pos]:
                if count >= squad_quotas[pos]:
                    break
                if team_counts.get(p.team, 0) >= self.max_from_team:
                    continue
                squad.append(p)
                team_counts[p.team] = team_counts.get(p.team, 0) + 1
                count += 1

        if len(squad) < 15:
            logger.warning("Inner problem: only %d squad players.", len(squad))
            return squad, squad[:11]

        # Select best 11 from squad
        lineup = self._best_lineup(squad, scores)
        return squad, lineup

    def _best_lineup(
        self,
        squad: list[Player],
        scores: dict[int, float],
    ) -> list[Player]:
        """Select best 11 from 15-player squad across all valid formations."""
        by_pos: dict[str, list[Player]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for p in squad:
            by_pos[p.position].append(p)
        for pos in by_pos:
            by_pos[pos].sort(key=lambda p: scores[p.id], reverse=True)

        best_lineup = None
        best_score = -np.inf

        for form_str in FormationConstraints.get_valid_formations():
            d, m, f = map(int, form_str.split("-"))
            targets = {"GK": 1, "DEF": d, "MID": m, "FWD": f}
            trial = []
            feasible = True
            for pos in ["GK", "DEF", "MID", "FWD"]:
                if len(by_pos[pos]) < targets[pos]:
                    feasible = False
                    break
                trial.extend(by_pos[pos][: targets[pos]])
            if feasible and len(trial) == 11:
                total_score = sum(scores[p.id] for p in trial)
                if total_score > best_score:
                    best_score = total_score
                    best_lineup = trial

        return best_lineup if best_lineup else squad[:11]

    def solve(
        self,
        players: list[Player],
        expected_points: dict[int, float],
        expected_variance: Optional[dict[int, float]] = None,
        best_known_primal: Optional[float] = None,
    ) -> LagrangianResult:
        """
        Solve via Lagrangian relaxation with subgradient ascent.

        Parameters
        ----------
        players : list[Player]
        expected_points : dict[int, float]
        expected_variance : dict[int, float], optional
        best_known_primal : float, optional
            Best known primal objective (e.g., from ILP).
            Used for better step size computation.

        Returns
        -------
        LagrangianResult
        """
        start_time = time.perf_counter()

        # Initialize lambda
        lam = 0.5  # initial budget multiplier
        best_dual = np.inf
        best_primal = -np.inf
        best_squad = None
        best_lineup = None

        # Step size parameters (Polyak-style)
        theta = 2.0
        theta_decay = 0.95
        no_improve_count = 0

        result = LagrangianResult()

        for k in range(self.max_iter):
            # Compute modified scores
            scores = self._compute_modified_scores(players, expected_points, expected_variance, lam)

            # Solve inner problem
            squad, lineup = self._solve_inner(players, scores)

            # Dual objective: L(lambda) = sum scores*x + lambda*B
            inner_value = sum(scores[p.id] for p in lineup)
            dual_obj = inner_value + lam * self.budget

            # Primal objective (original, without lambda penalty)
            primal_obj = sum(expected_points.get(p.id, 0.0) for p in lineup)
            if self.risk_aversion > 0 and expected_variance:
                for p in lineup:
                    primal_obj -= self.risk_aversion * np.sqrt(max(expected_variance.get(p.id, 0.0), 0.0))

            # Budget slack (subgradient)
            squad_cost = sum(p.price for p in squad)
            budget_slack = squad_cost - self.budget  # positive = over budget

            # Track best
            if dual_obj < best_dual:
                best_dual = dual_obj
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Only count as feasible primal if budget satisfied
            if squad_cost <= self.budget + 0.01 and primal_obj > best_primal:
                best_primal = primal_obj
                best_squad = squad
                best_lineup = lineup

            # Record history
            result.dual_history.append(float(dual_obj))
            result.primal_history.append(float(primal_obj))
            result.lambda_history.append(float(lam))
            result.budget_slack_history.append(float(budget_slack))

            # Convergence check
            gap = (best_dual - best_primal) / max(abs(best_dual), 1e-6)
            if gap < self.tol and best_primal > -np.inf:
                result.converged = True
                break

            # Step size (Polyak with target)
            target = best_known_primal if best_known_primal else best_primal
            step = 0.0 if abs(budget_slack) < 1e-08 else theta * (dual_obj - target) / budget_slack**2

            # Update lambda
            lam = max(0.0, lam + step * budget_slack)

            # Decay step size if no improvement
            if no_improve_count >= 5:
                theta *= theta_decay
                no_improve_count = 0

        elapsed = time.perf_counter() - start_time

        # Build FullSquad from best feasible solution
        if best_squad and best_lineup and len(best_squad) == 15 and len(best_lineup) == 11:
            pos_counts = {"DEF": 0, "MID": 0, "FWD": 0}
            for p in best_lineup:
                if p.position in pos_counts:
                    pos_counts[p.position] += 1
            formation = f"{pos_counts['DEF']}-{pos_counts['MID']}-{pos_counts['FWD']}"

            ep_lineup = sum(expected_points.get(p.id, 0.0) for p in best_lineup)
            captain = max(best_lineup, key=lambda p: expected_points.get(p.id, 0.0))

            lineup_obj = Squad(
                players=best_lineup,
                formation=formation,
                total_cost=sum(p.price for p in best_lineup),
                expected_points=ep_lineup,
                captain=captain,
            )
            result.full_squad = FullSquad(squad_players=best_squad, lineup=lineup_obj)

        result.primal_objective = best_primal
        result.dual_bound = best_dual
        result.duality_gap = (best_dual - best_primal) / max(abs(best_dual), 1e-6)
        result.n_iterations = k + 1
        result.solve_time = elapsed

        logger.info(
            "Lagrangian: %d iters, primal=%.1f, dual=%.1f, gap=%.2f%%, time=%.3fs",
            result.n_iterations,
            best_primal,
            best_dual,
            result.duality_gap * 100,
            elapsed,
        )

        return result


__all__ = ["LagrangianOptimizer", "LagrangianResult"]
