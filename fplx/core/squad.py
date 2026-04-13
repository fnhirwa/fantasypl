"""Squad and FullSquad domain objects."""

from dataclasses import dataclass, field
from typing import Optional

from fplx.core.player import Player


@dataclass
class Squad:
    """
    Represents an 11-player starting lineup.

    Attributes
    ----------
    players : list[Player]
        Selected starters (exactly 11).
    formation : str
        Formation string (e.g., "3-4-3").
    total_cost : float
        Total cost of the starting 11.
    expected_points : float
        Expected total points for the starting 11.
    captain : Optional[Player]
        Captain selection (earns double points).
    """

    players: list[Player]
    formation: str
    total_cost: float
    expected_points: float
    captain: Optional[Player] = None

    def __post_init__(self):
        if len(self.players) != 11:
            raise ValueError(f"Lineup must contain exactly 11 players, got {len(self.players)}.")

    def summary(self) -> str:
        """Returns a formatted string summary of the lineup."""
        pos_order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
        lines = [
            f"Formation: {self.formation}",
            f"Total Cost: £{self.total_cost:.1f}m",
            f"Expected Points: {self.expected_points:.2f}",
            f"Captain: {self.captain.name if self.captain else 'None'}",
            "",
            "--- Starting XI ---",
        ]
        for p in sorted(self.players, key=lambda x: pos_order.get(x.position, 9)):
            lines.append(f"  {p.name} ({p.position}, {p.team}, £{p.price}m)")
        return "\n".join(lines)


@dataclass
class FullSquad:
    """
    Represents a 15-player FPL squad with a selected 11-player lineup.

    The two-level FPL structure:
      Level 1: 15-player squad (2 GK, 5 DEF, 5 MID, 3 FWD) under budget.
      Level 2: 11-player starting lineup chosen from the squad each gameweek.

    Attributes
    ----------
    squad_players : list[Player]
        All 15 squad members.
    lineup : Squad
        The 11-player starting lineup (subset of squad_players).
    bench : list[Player]
        The 4 bench players.
    squad_cost : float
        Total cost of all 15 players.
    expected_points : float
        Expected points for the starting 11.
    """

    squad_players: list[Player]
    lineup: Squad
    bench: list[Player] = field(default_factory=list)
    squad_cost: float = 0.0
    expected_points: float = 0.0

    def __post_init__(self):
        if len(self.squad_players) != 15:
            raise ValueError(f"Squad must contain exactly 15 players, got {len(self.squad_players)}.")
        if not self.bench:
            lineup_ids = {p.id for p in self.lineup.players}
            self.bench = [p for p in self.squad_players if p.id not in lineup_ids]
        if not self.squad_cost:
            self.squad_cost = sum(p.price for p in self.squad_players)
        if not self.expected_points:
            self.expected_points = self.lineup.expected_points

    def summary(self) -> str:
        """Returns a formatted string summary of the full squad."""
        lines = [
            f"Squad Cost: £{self.squad_cost:.1f}m / £100.0m",
            f"Remaining Budget: £{100.0 - self.squad_cost:.1f}m",
            "",
            self.lineup.summary(),
            "",
            "--- Bench ---",
        ]
        for p in self.bench:
            lines.append(f"  {p.name} ({p.position}, {p.team}, £{p.price}m)")
        return "\n".join(lines)
