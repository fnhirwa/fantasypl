"""Constraints for squad selection."""

from fplx.core.player import Player


class SquadQuotas:
    """
    Position quotas for the 15-player FPL squad.

    Rules:
    - 2 GK, 5 DEF, 5 MID, 3 FWD (exactly).
    - Total = 15 players.
    """

    QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    TOTAL = 15

    @classmethod
    def validate(cls, players: list[Player]) -> bool:
        if len(players) != cls.TOTAL:
            return False
        counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for p in players:
            counts[p.position] += 1
        return all(counts[pos] == quota for pos, quota in cls.QUOTAS.items())


class FormationConstraints:
    """
    Formation constraints for FPL squad.

    Rules:
    - Exactly 11 players
    - 1 GK
    - 3-5 DEF
    - 2-5 MID
    - 1-3 FWD
    """

    POSITION_LIMITS = {
        "GK": (1, 1),
        "DEF": (3, 5),
        "MID": (2, 5),
        "FWD": (1, 3),
    }
    TOTAL_PLAYERS = 11

    @classmethod
    def validate(cls, players: list[Player]) -> bool:
        """
        Check if squad satisfies formation constraints.

        Parameters
        ----------
        players : list[Player]
            List of players in squad

        Returns
        -------
        bool
            True if valid formation
        """
        if len(players) != cls.TOTAL_PLAYERS:
            return False
        counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for p in players:
            counts[p.position] += 1
        return all(lo <= counts[pos] <= hi for pos, (lo, hi) in cls.POSITION_LIMITS.items())

    @classmethod
    def get_valid_formations(cls) -> list[str]:
        """
        Get list of valid formation strings.

        Returns
        -------
        List[str]
            Valid formations (e.g., "3-4-3", "4-3-3")
        """
        formations = []
        for d in range(3, 6):
            for m in range(2, 6):
                for f in range(1, 4):
                    if d + m + f == 10:
                        formations.append(f"{d}-{m}-{f}")
        return formations


class BudgetConstraint:
    """Budget constraint for FPL squad (applied to 15-player squad)."""

    def __init__(self, max_budget: float = 100.0):
        self.max_budget = max_budget

    def validate(self, players: list[Player]) -> bool:
        return sum(p.price for p in players) <= self.max_budget

    def get_remaining_budget(self, players: list[Player]) -> float:
        return self.max_budget - sum(p.price for p in players)


class TeamDiversityConstraint:
    """Max players from same real-world team (default 3)."""

    def __init__(self, max_from_team: int = 3):
        self.max_from_team = max_from_team

    def validate(self, players: list[Player]) -> bool:
        counts: dict[str, int] = {}
        for p in players:
            counts[p.team] = counts.get(p.team, 0) + 1
        return all(c <= self.max_from_team for c in counts.values())
