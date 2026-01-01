"""Data loaders for FPL data sources."""

import pandas as pd
import requests
from pathlib import Path
from typing import Optional
import logging

from fplx.core import Player

logger = logging.getLogger(__name__)


class FPLDataLoader:
    """
    Load and manage FPL data from various sources (API, CSV, cache).
    
    Parameters
    ----------
    cache_dir : Optional[Path]
        Directory to cache downloaded data
    """
    
    # FPL API Endpoints
    BASE_URL = "https://fantasy.premierleague.com/api"
    BOOTSTRAP_URL = f"{BASE_URL}/bootstrap-static/"
    FIXTURES_URL = f"{BASE_URL}/fixtures/"
    PLAYER_DETAIL_URL = f"{BASE_URL}/element-summary/{{player_id}}/"
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".fplx" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._bootstrap_data = None
        
    def fetch_bootstrap_data(self, force_refresh: bool = False) -> dict:
        """
        Fetch main FPL data (players, teams, gameweeks).
        
        Parameters
        ----------
        force_refresh : bool
            Force refresh even if cached
            
        Returns
        -------
        Dict
            Bootstrap data containing players, teams, events
        """
        cache_file = self.cache_dir / "bootstrap.json"
        
        if not force_refresh and cache_file.exists():
            import json
            with open(cache_file, 'r') as f:
                logger.info("Loading bootstrap data from cache")
                return json.load(f)
        
        logger.info("Fetching bootstrap data from FPL API")
        response = requests.get(self.BOOTSTRAP_URL)
        response.raise_for_status()
        
        data = response.json()
        
        # Cache the data
        import json
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        self._bootstrap_data = data
        return data
    
    def load_players(self, force_refresh: bool = False) -> list[Player]:
        """
        Load all players with basic info.
        
        Parameters
        ----------
        force_refresh : bool
            Force refresh from API
            
        Returns
        -------
        list[Player]
            List of Player objects
        """
        data = self.fetch_bootstrap_data(force_refresh)
        
        # Build team mapping
        teams = {t['id']: t['name'] for t in data['teams']}
        positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        players = []
        for element in data['elements']:
            # Create minimal timeseries (can be enriched later)
            ts_data = {
                'gameweek': [0],
                'points': [element.get('total_points', 0)],
                'minutes': [element.get('minutes', 0)],
                'form': [float(element.get('form', 0))],
            }
            
            player = Player(
                id=element['id'],
                name=element['web_name'],
                team=teams[element['team']],
                position=positions[element['element_type']],
                price=element['now_cost'] / 10.0,  # Convert to £m
                timeseries=pd.DataFrame(ts_data),
                news={'text': element.get('news', ''), 'availability': 1.0 if element.get('chance_of_playing_next_round') is None else element.get('chance_of_playing_next_round') / 100.0}
            )
            players.append(player)
        
        logger.info(f"Loaded {len(players)} players")
        return players
    
    def load_player_history(self, player_id: int) -> pd.DataFrame:
        """
        Load detailed historical data for a specific player.
        
        Parameters
        ----------
        player_id : int
            Player ID
            
        Returns
        -------
        pd.DataFrame
            Historical gameweek stats
        """
        url = self.PLAYER_DETAIL_URL.format(player_id=player_id)
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        history = pd.DataFrame(data['history'])
        
        # Rename columns for consistency
        if not history.empty:
            history = history.rename(columns={
                'round': 'gameweek',
                'total_points': 'points',
                'minutes': 'minutes',
                'goals_scored': 'goals',
                'assists': 'assists',
                'expected_goals': 'xG',
                'expected_assists': 'xA',
            })
        
        return history
    
    def load_fixtures(self) -> pd.DataFrame:
        """
        Load all fixtures.
        
        Returns
        -------
        pd.DataFrame
            Fixtures data
        """
        response = requests.get(self.FIXTURES_URL)
        response.raise_for_status()
        
        fixtures = pd.DataFrame(response.json())
        return fixtures
    
    def load_from_csv(self, filepath: Path) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Parameters
        ----------
        filepath : Path
            Path to CSV file
            
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        return df
    
    def enrich_player_history(self, players: list[Player]) -> list[Player]:
        """
        Enrich players with full historical data.
        
        Parameters
        ----------
        players : list[Player]
            List of players to enrich
            
        Returns
        -------
        list[Player]
            Players with enriched timeseries
        """
        enriched = []
        for player in players:
            try:
                history = self.load_player_history(player.id)
                if not history.empty:
                    player.timeseries = history
                enriched.append(player)
            except Exception as e:
                logger.warning(f"Could not load history for {player.name}: {e}")
                enriched.append(player)
        
        return enriched
