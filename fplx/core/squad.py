"""Squad domain object."""
from dataclasses import dataclass
from typing import Optional
from fplx.core.player import Player


@dataclass
class Squad:
    """
    Represents an optimal 11-player fantasy squad.
    
    Attributes
    ----------
    players : list[Player]
        Selected players (must be exactly 11)
    formation : str
        Formation string (e.g., "3-4-3")
    total_cost : float
        Total squad cost
    expected_points : float
        Expected total points
    captain : Optional[Player]
        Captain selection
    """
    players: list[Player]
    formation: str
    total_cost: float
    expected_points: float
    captain: Optional[Player] = None
    
    def __post_init__(self):
        if len(self.players) != 11:
            raise ValueError("Squad must contain exactly 11 players.")
            
    def summary(self) -> str:
        """Returns a formatted string summary of the squad."""
        summary_str = (
            f"Formation: {self.formation}\n"
            f"Total Cost: £{self.total_cost:.1f}m\n"
            f"Expected Points: {self.expected_points:.2f}\n"
            f"Captain: {self.captain.name if self.captain else 'None'}\n\n"
            "--- Squad ---\n"
        )
        for p in sorted(self.players, key=lambda x: x.position):
            summary_str += f"- {p.name} ({p.position}, {p.team}, £{p.price}m)\n"
        return summary_str
