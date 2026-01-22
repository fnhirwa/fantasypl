"""Squad optimization algorithms."""

import logging
from typing import Optional

from fplx.core.player import Player
from fplx.core.squad import Squad
from fplx.selection.base import BaseOptimizer
from fplx.selection.constraints import (
    BudgetConstraint,
    FormationConstraints,
    TeamDiversityConstraint,
)

logger = logging.getLogger(__name__)


class SquadOptimizer(BaseOptimizer):
    """
    Base class for squad optimization.

    Parameters
    ----------
    budget : float
        Maximum budget
    max_from_team : int
        Maximum players from same team
    """

    def __init__(self, budget: float = 100.0, max_from_team: int = 3):
        self.budget_constraint = BudgetConstraint(budget)
        self.formation_constraint = FormationConstraints()
        self.diversity_constraint = TeamDiversityConstraint(max_from_team)

    def solve(self, players, **kwargs):
        """Solve the optimization problem."""
        return self.optimize(players, **kwargs)

    def optimize(
        self,
        players: list[Player],
        expected_points: dict[int, float],
        formation: Optional[str] = None,
    ) -> Squad:
        """
        Optimize squad selection.

        Parameters
        ----------
        players : list[Player]
            Available players
        expected_points : dict[int, float]
            Expected points for each player (keyed by player.id)
        formation : Optional[str]
            Desired formation (e.g., "3-4-3"), None for auto

        Returns
        -------
        Squad
            Optimal squad
        """
        raise NotImplementedError("Subclasses must implement optimize()")

    def validate_squad(self, players: list[Player]) -> bool:
        """
        Validate if squad meets all constraints.

        Parameters
        ----------
        players : list[Player]
            Squad to validate

        Returns
        -------
        bool
            True if valid
        """
        return (
            self.formation_constraint.validate(players)
            and self.budget_constraint.validate(players)
            and self.diversity_constraint.validate(players)
        )


class GreedyOptimizer(SquadOptimizer):
    """
    Greedy optimizer: select best value players per position.

    Fast but not guaranteed optimal.
    """

    def optimize(
        self,
        players: list[Player],
        expected_points: dict[int, float],
        formation: Optional[str] = None,
    ) -> Squad:
        """
        Greedy optimization algorithm.

        Parameters
        ----------
        players : list[Player]
            Available players
        expected_points : dict[int, float]
            Expected points for each player
        formation : Optional[str]
            Desired formation

        Returns
        -------
        Squad
            Selected squad
        """
        # Attach expected points to players
        for p in players:
            p.expected_points = expected_points.get(p.id, 0.0)
            p.value = p.expected_points / max(p.price, 0.1)  # Points per £

        # Parse formation
        if formation and formation != "auto":
            def_target, mid_target, fwd_target = map(int, formation.split("-"))
        else:
            # Default balanced formation
            def_target, mid_target, fwd_target = 4, 3, 3

        targets = {
            "GK": 1,
            "DEF": def_target,
            "MID": mid_target,
            "FWD": fwd_target,
        }

        # Sort players by value within each position
        players_by_pos = {
            "GK": [],
            "DEF": [],
            "MID": [],
            "FWD": [],
        }

        for p in players:
            players_by_pos[p.position].append(p)

        for pos in players_by_pos:
            players_by_pos[pos].sort(key=lambda x: x.value, reverse=True)

        # Greedy selection
        selected = []
        remaining_budget = self.budget_constraint.max_budget

        # Select best players per position up to the target
        for pos in ["GK", "DEF", "MID", "FWD"]:
            target = targets[pos]
            candidates = players_by_pos[pos]
            count = 0
            for candidate in candidates:
                if count >= target:
                    break

                # Tentatively add to check constraints
                temp_squad = selected + [candidate]
                if self.diversity_constraint.validate(
                    temp_squad
                ) and self.budget_constraint.validate(temp_squad):
                    selected.append(candidate)
                    remaining_budget -= candidate.price
                    count += 1

        # If squad is not full, fill with cheapest available players
        if len(selected) < 11:
            logger.warning(
                f"Greedy selection only picked {len(selected)} players. Filling with cheapest options."
            )

            # Get all players not yet selected
            all_player_ids = {p.id for p in players}
            selected_ids = {p.id for p in selected}
            remaining_players = [p for p in players if p.id not in selected_ids]
            remaining_players.sort(key=lambda p: p.price)  # Sort by cheapest

            pos_counts = {
                pos: len([p for p in selected if p.position == pos])
                for pos in self.formation_constraint.POSITION_LIMITS
            }
            min_pos_counts = {
                pos: limits[0]
                for pos, limits in self.formation_constraint.POSITION_LIMITS.items()
            }

            # First, try to meet minimums
            for player in remaining_players:
                if len(selected) >= 11:
                    break
                pos = player.position
                if pos_counts[pos] < min_pos_counts[pos]:
                    temp_squad = selected + [player]
                    if self.diversity_constraint.validate(
                        temp_squad
                    ) and self.budget_constraint.validate(temp_squad):
                        selected.append(player)
                        pos_counts[pos] += 1

            # Then, fill remaining spots up to max
            for player in remaining_players:
                if len(selected) >= 11:
                    break

                # Re-check if player was already added in the first pass
                if player.id in {p.id for p in selected}:
                    continue

                pos = player.position
                if pos_counts[pos] < self.formation_constraint.MAX_PLAYERS[pos]:
                    temp_squad = selected + [player]
                    if self.diversity_constraint.validate(
                        temp_squad
                    ) and self.budget_constraint.validate(temp_squad):
                        selected.append(player)
                        pos_counts[pos] += 1
                if pos_counts[pos] < min_pos_counts[pos]:
                    temp_squad = selected + [player]
                    if self.diversity_constraint.validate(
                        temp_squad
                    ) and self.budget_constraint.validate(temp_squad):
                        selected.append(player)
                        pos_counts[pos] += 1

            # Then, fill remaining spots up to max
            for player in remaining_players:
                if len(selected) >= 11:
                    break

                # Re-check if player was already added in the first pass
                if player.id in {p.id for p in selected}:
                    continue

                pos = player.position
                if pos_counts[pos] < self.formation_constraint.MAX_PLAYERS[pos]:
                    temp_squad = selected + [player]
                    if self.diversity_constraint.validate(
                        temp_squad
                    ) and self.budget_constraint.validate(temp_squad):
                        selected.append(player)
                        pos_counts[pos] += 1

        # Validate and create squad
        if len(selected) != 11:
            logger.error(
                f"Optimizer could not form a valid 11-player squad. Only found {len(selected)}."
            )
            # Fallback to returning what we have, even if invalid

        total_expected = sum(p.expected_points for p in selected)
        total_cost = sum(p.price for p in selected)

        # Determine captain (highest expected points)
        captain = max(selected, key=lambda p: p.expected_points) if selected else None

        squad = Squad(
            players=selected,
            formation=f"{def_target}-{mid_target}-{fwd_target}",
            total_cost=total_cost,
            expected_points=total_expected,
            captain=captain,
        )

        return squad


class ILPOptimizer(SquadOptimizer):
    """
    Integer Linear Programming optimizer for optimal solution.

    Requires pulp library.
    """

    def __init__(self, budget: float = 100.0, max_from_team: int = 3):
        super().__init__(budget, max_from_team)
        try:
            import pulp

            self.pulp = pulp
        except ImportError:
            raise ImportError(
                "pulp required for ILP optimization. Install with: pip install pulp"
            )

    def optimize(
        self,
        players: list[Player],
        expected_points: dict[int, float],
        formation: Optional[str] = None,
    ) -> Squad:
        """
        ILP optimization for provably optimal squad.

        Parameters
        ----------
        players : list[Player]
            Available players
        expected_points : dict[int, float]
            Expected points for each player
        formation : Optional[str]
            Desired formation (if None, will optimize formation too)

        Returns
        -------
        Squad
            Optimal squad
        """
        prob = self.pulp.LpProblem("FPL_Squad_Selection", self.pulp.LpMaximize)

        # Decision variables
        player_vars = {
            p.id: self.pulp.LpVariable(f"player_{p.id}", cat="Binary") for p in players
        }

        # Objective: maximize expected points
        prob += self.pulp.lpSum([
            expected_points.get(p.id, 0) * player_vars[p.id] for p in players
        ])

        # Constraint: exactly 11 players
        prob += self.pulp.lpSum([player_vars[p.id] for p in players]) == 11

        # Constraint: budget
        prob += (
            self.pulp.lpSum([p.price * player_vars[p.id] for p in players])
            <= self.budget_constraint.max_budget
        )

        # Constraint: formation
        for pos, (min_count, max_count) in FormationConstraints.POSITION_LIMITS.items():
            pos_players = [p for p in players if p.position == pos]
            prob += (
                self.pulp.lpSum([player_vars[p.id] for p in pos_players]) >= min_count
            )
            prob += (
                self.pulp.lpSum([player_vars[p.id] for p in pos_players]) <= max_count
            )

        # Constraint: team diversity
        teams = set(p.team for p in players)
        for team in teams:
            team_players = [p for p in players if p.team == team]
            prob += (
                self.pulp.lpSum([player_vars[p.id] for p in team_players])
                <= self.diversity_constraint.max_from_team
            )

        # Solve
        prob.solve(self.pulp.PULP_CBC_CMD(msg=0))

        # Extract solution
        selected = [p for p in players if player_vars[p.id].varValue == 1]

        # Determine formation
        pos_counts = {"DEF": 0, "MID": 0, "FWD": 0}
        for p in selected:
            if p.position in pos_counts:
                pos_counts[p.position] += 1

        formation_str = f"{pos_counts['DEF']}-{pos_counts['MID']}-{pos_counts['FWD']}"

        # Calculate metrics
        total_expected = sum(expected_points.get(p.id, 0) for p in selected)
        total_cost = sum(p.price for p in selected)

        # Captain
        for p in selected:
            p.expected_points = expected_points.get(p.id, 0.0)
        captain = max(selected, key=lambda p: p.expected_points) if selected else None

        squad = Squad(
            players=selected,
            formation=formation_str,
            total_cost=total_cost,
            expected_points=total_expected,
            captain=captain,
        )

        return squad
