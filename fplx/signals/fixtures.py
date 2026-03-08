"""Fixture difficulty signals."""

import logging
from typing import Optional

import pandas as pd

from fplx.signals.base import BaseSignal

logger = logging.getLogger(__name__)


class FixtureSignal(BaseSignal):
    """Generate signals based on fixture difficulty and schedule."""

    def __init__(self, difficulty_ratings: Optional[dict[str, int]] = None):
        """
        Initialize with team difficulty ratings.

        Parameters
        ----------
        difficulty_ratings : Optional[dict[str, int]]
            Team strength ratings (1-5, higher = harder opponent)
        """
        self.difficulty_ratings = difficulty_ratings or {}

    def generate_signal(self, data):
        """Generate fixture-based signal."""
        # This is a placeholder. The actual implementation would take
        # fixture data and compute a signal.
        return self.compute_fixture_advantage(
            data["team"], data["upcoming_opponents"], data["is_home"]
        )

    def set_difficulty_ratings(self, ratings: dict[str, int]):
        """
        Set or update difficulty ratings.

        Parameters
        ----------
        ratings : Dict[str, int]
            Team strength ratings
        """
        self.difficulty_ratings = ratings

    def compute_fixture_difficulty(
        self, team: str, upcoming_opponents: list[str], is_home: list[bool]
    ) -> float:
        """
        Compute fixture difficulty score for upcoming games.

        Parameters
        ----------
        team : str
            Player's team
        upcoming_opponents : list[str]
            List of upcoming opponent teams
        is_home : list[bool]
            Whether each fixture is home

        Returns
        -------
        float
            Difficulty score (lower = easier fixtures)
        """
        if not upcoming_opponents:
            return 3.0  # Neutral

        difficulties = []
        for opponent, home in zip(upcoming_opponents, is_home):
            # Get opponent difficulty
            diff = self.difficulty_ratings.get(opponent, 3)

            # Adjust for home advantage
            if home:
                diff = max(1, diff - 0.5)
            else:
                diff = min(5, diff + 0.5)

            difficulties.append(diff)

        # Average difficulty
        avg_difficulty = sum(difficulties) / len(difficulties)
        return avg_difficulty

    def compute_fixture_advantage(
        self, team: str, upcoming_opponents: list[str], is_home: list[bool]
    ) -> float:
        """
        Compute fixture advantage (inverse of difficulty).

        Higher score = easier fixtures = better for player.

        Parameters
        ----------
        team : str
            Player's team
        upcoming_opponents : list[str]
            List of upcoming opponent teams
        is_home : list[bool]
            Whether each fixture is home

        Returns
        -------
        float
            Advantage score (0-1, higher = better fixtures)
        """
        difficulty = self.compute_fixture_difficulty(team, upcoming_opponents, is_home)

        # Convert to advantage (invert and normalize)
        # difficulty: 1 (easiest) to 5 (hardest)
        # advantage: 1 (best) to 0 (worst)
        advantage = (6 - difficulty) / 5
        return max(0, min(1, advantage))

    def compute_fixture_congestion(
        self, fixtures: pd.DataFrame, team: str, days_window: int = 14
    ) -> float:
        """
        Compute fixture congestion (number of games in short period).

        Parameters
        ----------
        fixtures : pd.DataFrame
            Fixtures dataframe
        team : str
            Team name
        days_window : int
            Days to look ahead

        Returns
        -------
        float
            Congestion score (0-1, higher = more congested)
        """
        # Filter fixtures for the team
        team_fixtures = fixtures[
            (fixtures["team_h"] == team) | (fixtures["team_a"] == team)
        ]

        if team_fixtures.empty:
            return 0.0

        # Count fixtures in window
        num_fixtures = len(team_fixtures)

        # Normalize: 1 game/week = 0, 3+ games/week = 1
        games_per_week = num_fixtures / (days_window / 7)
        congestion = min(1.0, (games_per_week - 1) / 2)

        return max(0, congestion)

    def batch_compute_advantages(
        self, players_teams: dict[str, str], fixtures_data: dict[str, tuple]
    ) -> dict[str, float]:
        """
        Compute fixture advantages for multiple players.

        Parameters
        ----------
        players_teams : dict[str, str]
            Mapping of player ID to team
        fixtures_data : dict[str, tuple]
            Mapping of team to (opponents, is_home) tuples

        Returns
        -------
        dict[str, float]
            Dictionary of player fixture advantage scores
        """
        advantages = {}

        for player_id, team in players_teams.items():
            if team in fixtures_data:
                opponents, is_home = fixtures_data[team]
                advantage = self.compute_fixture_advantage(team, opponents, is_home)
                advantages[player_id] = advantage
            else:
                advantages[player_id] = 0.5  # Neutral

        return advantages
