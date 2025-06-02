
# --- strategies/DrawProbabilityPhysics.py ---
import numpy as np
from smarkets_Sports_Quant_Trading.backtester.strategies.strategy_interface.strategy_interface import BaseStrategy

import pandas as pd
import numpy as np
from smarkets_Sports_Quant_Trading.backtester.strategies.strategy_interface.strategy_interface import StrategyInterface

class DrawProbabilityPhysics(StrategyInterface):
    """Betting strategy estimating draw probability based on team rating differences."""
    
    def __init__(self, prob_threshold: float = 0.3, min_edge: float = 0.05, bankroll: float = 1000.0):
        """Initialize the strategy with probability threshold and betting parameters.
        
        Args:
            prob_threshold (float): Minimum draw probability to place a bet (default: 0.3).
            min_edge (float): Minimum edge to place a bet (default: 0.05).
            bankroll (float): Total betting capital (default: 1000.0).
        """
        super().__init__()
        self.prob_threshold = prob_threshold
        self.min_edge = min_edge
        self.bankroll = bankroll
    
    def calculate_implied_probability(self, odds: float) -> float:
        """Calculate implied probability from decimal odds.
        
        Args:
            odds (float): Decimal odds (e.g., 2.0 for even money).
            
        Returns:
            float: Implied probability (e.g., 0.5 for odds of 2.0).
        """
        return 1 / odds if odds > 1 else 0.0
    
    def estimate_draw_probability(self, home_rating: float, away_rating: float) -> float:
        """Estimate draw probability based on rating difference.
        
        Args:
            home_rating (float): Home team rating (e.g., Elo rating).
            away_rating (float): Away team rating (e.g., Elo rating).
            
        Returns:
            float: Estimated draw probability (0 to 1).
        """
        delta_rating = abs(home_rating - away_rating)
        # Physics-inspired exponential decay, normalized to approximate realistic draw probability
        raw_prob = np.exp(-delta_rating / 400)
        # Scale to typical draw probability range (e.g., 0.2 to 0.35 in soccer)
        scaled_prob = 0.35 * raw_prob / (1 + raw_prob)
        return min(max(scaled_prob, 0.0), 1.0)
    
    def calculate_stake(self, prob_draw: float, odds: float) -> float:
        """Calculate stake using a simplified Kelly Criterion.
        
        Args:
            prob_draw (float): Estimated probability of a draw.
            odds (float): Decimal odds for a draw.
            
        Returns:
            float: Recommended stake, capped at 10% of bankroll.
        """
        implied_prob = self.calculate_implied_probability(odds)
        edge = prob_draw - implied_prob
        fraction = edge / (odds - 1) if odds > 1 and edge > 0 else 0
        stake = self.bankroll * fraction
        return max(0, min(stake, self.bankroll * 0.1))  # Cap at 10% of bankroll
    
    def generate_signal(self, row: pd.Series) -> dict:
        """Generate a betting signal based on draw probability from rating differences.
        
        Args:
            row (pd.Series): Input row with 'home_rating', 'away_rating', 'draw_odds' columns.
            
        Returns:
            dict: Signal dictionary with 'team', 'side', 'odds', 'stake'.
        """
        signal = {'team': None, 'side': None, 'odds': 0.0, 'stake': 0.0}
        
        # Extract data, provide defaults for missing/invalid values
        home_rating = row.get('home_rating', 1500.0)
        away_rating = row.get('away_rating', 1500.0)
        draw_odds = row.get('draw_odds', 1.0)
        
        # Validate inputs
        if not all(isinstance(x, (int, float)) for x in [home_rating, away_rating, draw_odds]) or draw_odds <= 1.0:
            return signal
        
        # Estimate draw probability
        draw_prob = self.estimate_draw_probability(home_rating, away_rating)
        
        # Generate signal if probability exceeds threshold
        if draw_prob > self.prob_threshold:
            implied_prob = self.calculate_implied_probability(draw_odds)
            edge = draw_prob - implied_prob
            if edge > self.min_edge:
                stake = self.calculate_stake(draw_prob, draw_odds)
                if stake > 0:
                    signal = {
                        'team': 'draw',
                        'side': 'draw',
                        'odds': draw_odds,
                        'stake': round(stake, 2)
                    }
        
        return signal