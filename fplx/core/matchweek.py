"""Matchweek domain object."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Matchweek:
    """
    Represents a matchweek with global context.

    Attributes
    ----------
    gameweek : int
        Gameweek number
    date : datetime
        Date of the gameweek
    fixtures : list[dict]
        List of fixtures
    team_difficulty : dict[str, float]
        Team-level difficulty ratings
    """

    gameweek: int
    date: datetime
    fixtures: list[dict]
    team_difficulty: dict[str, float]
