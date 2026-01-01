"""Constraints for squad selection."""

from fplx.core import Player


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
        'GK': (1, 1),
        'DEF': (3, 5),
        'MID': (2, 5),
        'FWD': (1, 3),
    }
    
    TOTAL_PLAYERS = 11
    MAX_PLAYERS = {
        'GK': 1,
        'DEF': 5,
        'MID': 5,
        'FWD': 3,
    }
    
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
        
        # Count positions
        position_counts = {'GK': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        for p in players:
            position_counts[p.position] += 1
        
        # Check constraints
        for pos, (min_count, max_count) in cls.POSITION_LIMITS.items():
            count = position_counts[pos]
            if not (min_count <= count <= max_count):
                return False
        
        return True
    
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
        
        for def_count in range(3, 6):
            for mid_count in range(2, 6):
                for fwd_count in range(1, 4):
                    if def_count + mid_count + fwd_count == 10:
                        formations.append(f"{def_count}-{mid_count}-{fwd_count}")
        
        return formations


class BudgetConstraint:
    """
    Budget constraint for FPL squad.
    
    Parameters
    ----------
    max_budget : float
        Maximum total budget (default 100.0)
    """
    
    def __init__(self, max_budget: float = 100.0):
        self.max_budget = max_budget
    
    def validate(self, players: list[Player]) -> bool:
        """
        Check if squad is within budget.
        
        Parameters
        ----------
        players : list[Player]
            List of players in squad
            
        Returns
        -------
        bool
            True if within budget
        """
        total_cost = sum(p.price for p in players)
        return total_cost <= self.max_budget
    
    def get_total_cost(self, players: list[Player]) -> float:
        """
        Calculate total squad cost.
        
        Parameters
        ----------
        players : list[Player]
            List of players in squad
            
        Returns
        -------
        float
            Total cost
        """
        return sum(p.price for p in players)
    
    def get_remaining_budget(self, players: list[Player]) -> float:
        """
        Calculate remaining budget.
        
        Parameters
        ----------
        players : list[Player]
            List of players in squad
            
        Returns
        -------
        float
            Remaining budget
        """
        return self.max_budget - self.get_total_cost(players)


class TeamDiversityConstraint:
    """
    Constraint on maximum players from same team.
    
    Parameters
    ----------
    max_from_team : int
        Maximum players allowed from same team (default 3)
    """
    
    def __init__(self, max_from_team: int = 3):
        self.max_from_team = max_from_team
    
    def validate(self, players: list[Player]) -> bool:
        """
        Check if squad satisfies team diversity constraint.
        
        Parameters
        ----------
        players : list[Player]
            List of players in squad
            
        Returns
        -------
        bool
            True if constraint satisfied
        """
        team_counts = {}
        for p in players:
            team_counts[p.team] = team_counts.get(p.team, 0) + 1
        
        return all(count <= self.max_from_team for count in team_counts.values())
