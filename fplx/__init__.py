"""
FPLX - Fantasy Premier League Time-Series Analysis & Squad Optimization

A production-ready Python library for:
- FPL player time-series data analysis
- News & injury signal integration
- Expected performance scoring
- Optimal 15-player squad and 11-player lineup selection
"""

__version__ = "0.3.0"

from fplx.api.interface import FPLModel
from fplx.core.matchweek import Matchweek
from fplx.core.player import Player
from fplx.core.squad import FullSquad, Squad
from fplx.data.loaders import FPLDataLoader

__all__ = [
    "FPLModel",
    "FPLDataLoader",
    "Player",
    "Matchweek",
    "Squad",
    "FullSquad",
]
