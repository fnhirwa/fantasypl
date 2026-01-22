"""Player domain object."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Player:
    """
    Represents a Fantasy Premier League player.

    Attributes
    ----------
    id : int
        Unique player identifier
    name : str
        Player full name
    team : str
        Current team
    position : str
        Position (GK, DEF, MID, FWD)
    price : float
        Current price in FPL
    timeseries : pd.DataFrame
        Historical stats (points, xG, minutes, etc.)
    news : Optional[dict]
        Latest news/injury information
    """

    id: int
    name: str
    team: str
    position: str
    price: float
    timeseries: pd.DataFrame
    news: Optional[dict] = None

    @property
    def last_5_points(self) -> float:
        """Average points over last 5 gameweeks."""
        if len(self.timeseries) >= 5:
            return self.timeseries["points"].tail(5).mean()
        return self.timeseries["points"].mean()

    @property
    def availability(self) -> float:
        """Availability score (0-1) based on news."""
        if self.news is None:
            return 1.0
        return self.news.get("availability", 1.0)
