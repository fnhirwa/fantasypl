"""News and injury signal processing."""

import logging
import re

from fplx.signals.base import BaseSignal

logger = logging.getLogger(__name__)


class NewsParser:
    """
    Parse and interpret FPL news text into structured signals.
    """

    # Keyword patterns for availability
    UNAVAILABLE_PATTERNS = [
        r"out for \d+\s*(weeks?|months?)",
        r"ruled out",
        r"sidelined",
        r"long[- ]term injury",
        r"season[- ]ending",
        r"suspended",
    ]

    DOUBTFUL_PATTERNS = [
        r"doubtful",
        r"unlikely",
        r"50%",
        r"touch and go",
        r"late fitness test",
    ]

    ROTATION_PATTERNS = [
        r"rotation risk",
        r"benched",
        r"not starting",
        r"limited minutes",
    ]

    POSITIVE_PATTERNS = [
        r"back in training",
        r"available",
        r"fit",
        r"recovered",
        r"expected to start",
    ]

    def parse_availability(self, news_text: str) -> float:
        """
        Parse availability from news text.

        Parameters
        ----------
        news_text : str
            News text

        Returns
        -------
        float
            Availability score (0-1)
        """
        if not news_text or news_text.strip() == "":
            return 1.0

        text_lower = news_text.lower()

        # Check unavailable patterns
        for pattern in self.UNAVAILABLE_PATTERNS:
            if re.search(pattern, text_lower):
                return 0.0

        # Check doubtful patterns
        for pattern in self.DOUBTFUL_PATTERNS:
            if re.search(pattern, text_lower):
                return 0.5

        # Check positive patterns
        for pattern in self.POSITIVE_PATTERNS:
            if re.search(pattern, text_lower):
                return 0.9

        # Default: assume available if no negative signals
        return 1.0

    def parse_minutes_risk(self, news_text: str) -> float:
        """
        Parse minutes risk from news text.

        Parameters
        ----------
        news_text : str
            News text

        Returns
        -------
        float
            Minutes risk score (0-1, higher = more risk)
        """
        if not news_text or news_text.strip() == "":
            return 0.0

        text_lower = news_text.lower()

        # Check rotation patterns
        for pattern in self.ROTATION_PATTERNS:
            if re.search(pattern, text_lower):
                return 0.7

        # Check if doubtful (moderate risk)
        for pattern in self.DOUBTFUL_PATTERNS:
            if re.search(pattern, text_lower):
                return 0.3

        return 0.0

    def parse_confidence(self, news_text: str) -> float:
        """
        Estimate confidence in the parsed signal.

        Parameters
        ----------
        news_text : str
            News text

        Returns
        -------
        float
            Confidence score (0-1)
        """
        if not news_text or news_text.strip() == "":
            return 1.0  # High confidence when no news

        # Confidence based on clarity of news
        text_lower = news_text.lower()

        # High confidence patterns
        if any(
            re.search(p, text_lower) for p in ["ruled out", "confirmed", "definitely"]
        ):
            return 0.9

        # Medium confidence patterns
        if any(re.search(p, text_lower) for p in ["likely", "expected", "should"]):
            return 0.7

        # Low confidence patterns
        if any(re.search(p, text_lower) for p in ["maybe", "possible", "unclear"]):
            return 0.4

        return 0.6  # Default medium confidence


class NewsSignal(BaseSignal):
    """
    Generate structured news signals for players.
    """

    def __init__(self):
        self.parser = NewsParser()

    def generate_signal(self, news_text: str) -> dict[str, float]:
        """
        Generate signal from news text.

        Parameters
        ----------
        news_text : str
            News text

        Returns
        -------
        dict[str, float]
            Dictionary with availability, minutes_risk, confidence
        """
        availability = self.parser.parse_availability(news_text)
        minutes_risk = self.parser.parse_minutes_risk(news_text)
        confidence = self.parser.parse_confidence(news_text)

        return {
            "availability": availability,
            "minutes_risk": minutes_risk,
            "confidence": confidence,
            "adjustment_factor": availability * (1 - minutes_risk),
        }

    def batch_generate(self, news_dict: dict[str, str]) -> dict[str, dict[str, float]]:
        """
        Generate signals for multiple players.

        Parameters
        ----------
        news_dict : dict[str, str]
            Dictionary mapping player ID to news text

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary of player signals
        """
        signals = {}
        for player_id, news_text in news_dict.items():
            signals[player_id] = self.generate_signal(news_text)

        return signals
