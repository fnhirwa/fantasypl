"""Squad optimization: two-level ILP, mean-variance, LP relaxation."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fplx.core.player import Player
from fplx.core.squad import FullSquad, Squad
from fplx.selection.base import BaseOptimizer
from fplx.selection.constraints import (
    FormationConstraints,
    SquadQuotas,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization outputs including duality analysis."""

    full_squad: FullSquad
    objective_value: float = 0.0
    solve_time: float = 0.0
    # LP relaxation analysis
    lp_objective: Optional[float] = None
    integrality_gap: Optional[float] = None
    shadow_prices: dict = field(default_factory=dict)
    binding_constraints: list = field(default_factory=list)


class TwoLevelILPOptimizer(BaseOptimizer):
    """
    Two-level ILP: select 15-player squad then 11-player lineup jointly.

    Supports risk-neutral and risk-averse (mean-variance) objectives.
    Also exposes LP relaxation for shadow price extraction.

    Parameters
    ----------
    budget : float
        Maximum total squad budget (applied to 15 players).
    max_from_team : int
        Maximum players from same club.
    risk_aversion : float
        Lambda for mean-variance penalty. 0 = risk-neutral.
    """

    def __init__(
        self,
        budget: float = 100.0,
        max_from_team: int = 3,
        risk_aversion: float = 0.0,
    ):
        self.budget = budget
        self.max_from_team = max_from_team
        self.risk_aversion = risk_aversion

        try:
            import pulp

            self.pulp = pulp
        except ImportError:
            raise ImportError("pulp required for ILP optimization. Install with: pip install pulp")

    def solve(self, players, **kwargs):
        """Solve the optimization problem."""
        return self.optimize(players, **kwargs)

    def _build_problem(
        self,
        players: list[Player],
        expected_points: dict[int, float],
        expected_variance: Optional[dict[int, float]] = None,
        relax: bool = False,
    ):
        """
        Build the two-level optimization problem.

        Parameters
        ----------
        players : list[Player]
            Available player pool.
        expected_points : dict[int, float]
            E[P_i] per player.
        expected_variance : dict[int, float], optional
            Var[P_i] per player. Required if risk_aversion > 0.
        relax : bool
            If True, use continuous variables [0,1] (LP relaxation).

        Returns
        -------
        prob : pulp.LpProblem
        s_vars : dict  (squad variables)
        x_vars : dict  (lineup variables)
        """
        pulp = self.pulp
        prob = pulp.LpProblem("FPL_TwoLevel", pulp.LpMaximize)

        cat = "Continuous" if relax else "Binary"

        # Decision variables
        s_vars = {p.id: pulp.LpVariable(f"s_{p.id}", 0, 1, cat=cat) for p in players}
        x_vars = {p.id: pulp.LpVariable(f"x_{p.id}", 0, 1, cat=cat) for p in players}

        # Objective: maximize expected points of starting 11
        # Risk-averse: subtract lambda * sqrt(Var) for each starter
        obj_coeffs = {}
        for p in players:
            ep = expected_points.get(p.id, 0.0)
            penalty = 0.0
            if self.risk_aversion > 0 and expected_variance:
                var_i = expected_variance.get(p.id, 0.0)
                penalty = self.risk_aversion * np.sqrt(max(var_i, 0.0))
            obj_coeffs[p.id] = ep - penalty

        prob += pulp.lpSum([obj_coeffs[p.id] * x_vars[p.id] for p in players])

        # --- Level 1: 15-player squad constraints ---
        # Squad size = 15
        prob += (
            pulp.lpSum([s_vars[p.id] for p in players]) == 15,
            "squad_size",
        )
        # Squad position quotas (exactly 2 GK, 5 DEF, 5 MID, 3 FWD)
        for pos, quota in SquadQuotas.QUOTAS.items():
            pos_players = [p for p in players if p.position == pos]
            prob += (
                pulp.lpSum([s_vars[p.id] for p in pos_players]) == quota,
                f"squad_quota_{pos}",
            )
        # Budget (applied to squad, not lineup)
        prob += (
            pulp.lpSum([p.price * s_vars[p.id] for p in players]) <= self.budget,
            "budget",
        )
        # Team diversity (applied to squad)
        teams = {p.team for p in players}
        for team in teams:
            team_players = [p for p in players if p.team == team]
            prob += (
                pulp.lpSum([s_vars[p.id] for p in team_players]) <= self.max_from_team,
                f"team_cap_{team}",
            )

        # --- Level 2: 11-player lineup constraints ---
        # Lineup size = 11
        prob += (
            pulp.lpSum([x_vars[p.id] for p in players]) == 11,
            "lineup_size",
        )
        # Coupling: can only start if in squad
        for p in players:
            prob += (x_vars[p.id] <= s_vars[p.id], f"coupling_{p.id}")
        # Formation constraints on lineup
        for pos, (lo, hi) in FormationConstraints.POSITION_LIMITS.items():
            pos_players = [p for p in players if p.position == pos]
            prob += (
                pulp.lpSum([x_vars[p.id] for p in pos_players]) >= lo,
                f"lineup_min_{pos}",
            )
            prob += (
                pulp.lpSum([x_vars[p.id] for p in pos_players]) <= hi,
                f"lineup_max_{pos}",
            )

        return prob, s_vars, x_vars

    def optimize(
        self,
        players: list[Player],
        expected_points: dict[int, float],
        expected_variance: Optional[dict[int, float]] = None,
        formation: Optional[str] = None,
    ) -> FullSquad:
        """
        Solve the two-level ILP.

        Parameters
        ----------
        players : list[Player]
            Available player pool.
        expected_points : dict[int, float]
            E[P_i] per player.
        expected_variance : dict[int, float], optional
            Var[P_i] per player.
        formation : Optional[str]
            Not used (formation is optimized automatically).

        Returns
        -------
        FullSquad
        """
        import time

        start = time.perf_counter()
        prob, s_vars, x_vars = self._build_problem(players, expected_points, expected_variance, relax=False)
        prob.solve(self.pulp.PULP_CBC_CMD(msg=0))
        elapsed = time.perf_counter() - start

        if prob.status != 1:
            logger.error("ILP solver did not find optimal solution (status=%d).", prob.status)

        # Extract solution
        squad_players = [p for p in players if s_vars[p.id].varValue and s_vars[p.id].varValue > 0.5]
        lineup_players = [p for p in players if x_vars[p.id].varValue and x_vars[p.id].varValue > 0.5]

        # Determine formation
        pos_counts = {"DEF": 0, "MID": 0, "FWD": 0}
        for p in lineup_players:
            if p.position in pos_counts:
                pos_counts[p.position] += 1
        formation_str = f"{pos_counts['DEF']}-{pos_counts['MID']}-{pos_counts['FWD']}"

        # Captain = highest expected points
        for p in lineup_players:
            p.expected_points = expected_points.get(p.id, 0.0)
        captain = (
            max(lineup_players, key=lambda p: expected_points.get(p.id, 0.0)) if lineup_players else None
        )

        total_ep = sum(expected_points.get(p.id, 0.0) for p in lineup_players)
        lineup_cost = sum(p.price for p in lineup_players)

        lineup = Squad(
            players=lineup_players,
            formation=formation_str,
            total_cost=lineup_cost,
            expected_points=total_ep,
            captain=captain,
        )
        full_squad = FullSquad(squad_players=squad_players, lineup=lineup)

        logger.info("ILP solved in %.3fs. Formation: %s. EP: %.2f", elapsed, formation_str, total_ep)
        return full_squad

    def solve_lp_relaxation(
        self,
        players: list[Player],
        expected_points: dict[int, float],
        expected_variance: Optional[dict[int, float]] = None,
    ) -> OptimizationResult:
        """
        Solve the LP relaxation and extract shadow prices.

        Returns
        -------
        OptimizationResult
            Contains LP objective, shadow prices, binding constraints.
        """
        import time

        start = time.perf_counter()
        prob, s_vars, x_vars = self._build_problem(players, expected_points, expected_variance, relax=True)
        prob.solve(self.pulp.PULP_CBC_CMD(msg=0))
        elapsed = time.perf_counter() - start

        lp_obj = self.pulp.value(prob.objective)

        # Extract shadow prices from constraints
        shadow_prices = {}
        binding = []
        for name, constraint in prob.constraints.items():
            slack = constraint.slack
            # PuLP: pi attribute gives the dual value for LP
            dual = constraint.pi if constraint.pi is not None else 0.0
            shadow_prices[name] = {
                "dual_value": dual,
                "slack": slack,
                "binding": abs(slack) < 1e-6,
            }
            if abs(slack) < 1e-6:
                binding.append(name)

        # Also solve ILP to compute integrality gap
        full_squad = self.optimize(players, expected_points, expected_variance)
        ilp_obj = full_squad.lineup.expected_points
        gap = (lp_obj - ilp_obj) / lp_obj if lp_obj > 0 else 0.0

        return OptimizationResult(
            full_squad=full_squad,
            objective_value=ilp_obj,
            solve_time=elapsed,
            lp_objective=lp_obj,
            integrality_gap=gap,
            shadow_prices=shadow_prices,
            binding_constraints=binding,
        )


class GreedyOptimizer(BaseOptimizer):
    """
    Greedy baseline: select best-value players per position.

    Fast heuristic for comparison. Selects 15-player squad, then
    picks best 11 as lineup.
    """

    def __init__(self, budget: float = 100.0, max_from_team: int = 3):
        self.budget = budget
        self.max_from_team = max_from_team

    def solve(self, players, **kwargs):
        return self.optimize(players, **kwargs)

    def optimize(
        self,
        players: list[Player],
        expected_points: dict[int, float],
        expected_variance: Optional[dict[int, float]] = None,
        formation: Optional[str] = None,
    ) -> FullSquad:
        """Greedy squad + lineup selection."""
        # Compute value = EP / price for each player
        for p in players:
            ep = expected_points.get(p.id, 0.0)
            p.expected_points = ep
            p._value = ep / max(p.price, 0.1)

        # Sort by value within each position
        by_pos: dict[str, list[Player]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for p in players:
            by_pos[p.position].append(p)
        for pos in by_pos:
            by_pos[pos].sort(key=lambda p: p._value, reverse=True)

        # Greedily fill squad (15 players)
        squad_quotas = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
        selected_squad: list[Player] = []
        team_counts: dict[str, int] = {}
        remaining = self.budget

        for pos in ["GK", "DEF", "MID", "FWD"]:
            count = 0
            for p in by_pos[pos]:
                if count >= squad_quotas[pos]:
                    break
                if team_counts.get(p.team, 0) >= self.max_from_team:
                    continue
                if p.price > remaining:
                    continue
                selected_squad.append(p)
                team_counts[p.team] = team_counts.get(p.team, 0) + 1
                remaining -= p.price
                count += 1

        if len(selected_squad) != 15:
            logger.warning("Greedy only picked %d squad players.", len(selected_squad))
            # Pad if needed (shouldn't happen with 600+ players)
            return self._fallback(selected_squad, expected_points)

        # Select best 11 from the 15
        lineup = self._select_lineup(selected_squad, expected_points, formation)
        return FullSquad(squad_players=selected_squad, lineup=lineup)

    def _select_lineup(
        self,
        squad: list[Player],
        expected_points: dict[int, float],
        formation: Optional[str] = None,
    ) -> Squad:
        """Pick best 11 from 15-player squad."""
        if formation and formation != "auto":
            def_t, mid_t, fwd_t = map(int, formation.split("-"))
        else:
            def_t, mid_t, fwd_t = 4, 4, 2  # default

        targets = {"GK": 1, "DEF": def_t, "MID": mid_t, "FWD": fwd_t}

        # Sort squad by EP within each position
        by_pos: dict[str, list[Player]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for p in squad:
            by_pos[p.position].append(p)
        for pos in by_pos:
            by_pos[pos].sort(key=lambda p: expected_points.get(p.id, 0.0), reverse=True)

        lineup: list[Player] = []
        for pos in ["GK", "DEF", "MID", "FWD"]:
            lineup.extend(by_pos[pos][: targets[pos]])

        if len(lineup) != 11:
            # Try auto-formation: pick best 11 across valid formations
            best_lineup = None
            best_ep = -1
            for form_str in FormationConstraints.get_valid_formations():
                d, m, f = map(int, form_str.split("-"))
                t = {"GK": 1, "DEF": d, "MID": m, "FWD": f}
                trial = []
                for pos in ["GK", "DEF", "MID", "FWD"]:
                    trial.extend(by_pos[pos][: t[pos]])
                if len(trial) == 11:
                    ep = sum(expected_points.get(p.id, 0.0) for p in trial)
                    if ep > best_ep:
                        best_ep = ep
                        best_lineup = trial
                        def_t, mid_t, fwd_t = d, m, f
            if best_lineup:
                lineup = best_lineup

        captain = max(lineup, key=lambda p: expected_points.get(p.id, 0.0)) if lineup else None
        total_ep = sum(expected_points.get(p.id, 0.0) for p in lineup)
        total_cost = sum(p.price for p in lineup)

        return Squad(
            players=lineup,
            formation=f"{def_t}-{mid_t}-{fwd_t}",
            total_cost=total_cost,
            expected_points=total_ep,
            captain=captain,
        )

    def _fallback(self, partial_squad, expected_points):
        """Emergency fallback if squad is incomplete."""
        raise RuntimeError(f"Could not form a 15-player squad. Only found {len(partial_squad)} players.")


# Backward compatibility alias
ILPOptimizer = TwoLevelILPOptimizer
SquadOptimizer = TwoLevelILPOptimizer
