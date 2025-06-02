# --- strategies/ExpectedGoalsEdge.py ---

import pandas as pd
import numpy as np
from smarkets_Sports_Quant_Trading.backtester.strategies.strategy_interface.strategy_interface import StrategyInterface

class ExpectedGoalsEdge(StrategyInterface):
    """Betting strategy based on expected goals (xG) differential for home team advantage."""
    
    def __init__(self, xg_threshold: float = 0.5, min_edge: float = 0.05, bankroll: float = 1000.0):
        """Initialize the strategy with xG threshold and betting parameters.
        
        Args:
            xg_threshold (float): Minimum xG differential to place a bet (default: 0.5).
            min_edge (float): Minimum edge to place a bet (default: 0.05).
            bankroll (float): Total betting capital (default: 1000.0).
        """
        super().__init__()
        self.xg_threshold = xg_threshold
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
    
    def estimate_win_probability(self, xg_home: float, xg_away: float) -> float:
        """Estimate home win probability based on xG differential.
        
        Args:
            xg_home (float): Expected goals for home team.
            xg_away (float): Expected goals for away team.
            
        Returns:
            float: Estimated home win probability (0 to 1).
        """
        if xg_home <= 0 or xg_away <= 0:
            return 0.0
        # Simplified model: probability based on xG differential
        diff = xg_home - xg_away
        # Logistic function to map xG differential to probability
        prob = 1 / (1 + np.exp(-diff))
        return min(max(prob, 0.0), 1.0)
    
    def calculate_stake(self, prob_win: float, odds: float) -> float:
        """Calculate stake using a simplified Kelly Criterion.
        
        Args:
            prob_win (float): Estimated probability of winning.
            odds (float): Decimal odds.
            
        Returns:
            float: Recommended stake, capped at 10% of bankroll.
        """
        implied_prob = self.calculate_implied_probability(odds)
        edge = prob_win - implied_prob
        fraction = edge / (odds - 1) if odds > 1 and edge > 0 else 0
        stake = self.bankroll * fraction
        return max(0, min(stake, self.bankroll * 0.1))  # Cap at 10% of bankroll
    
    def generate_signal(self, row: pd.Series) -> dict:
        """Generate a betting signal based on expected goals differential.
        
        Args:
            row (pd.Series): Input row with 'home_xg', 'away_xg', 'home_odds' columns.
            
        Returns:
            dict: Signal dictionary with 'team', 'side', 'odds', 'stake'.
        """
        signal = {'team': None, 'side': None, 'odds': 0.0, 'stake': 0.0}
        
        # Extract data, provide defaults for missing/invalid values
        xg_home = row.get('home_xg', 1.2)
        xg_away = row.get('away_xg', 0.9)
        home_odds = row.get('home_odds', 1.0)
        
        # Validate inputs
        if not all(isinstance(x, (int, float)) for x in [xg_home, xg_away, home_odds]) or home_odds <= 1.0:
            return signal
        
        # Check xG differential
        xg_diff = xg_home - xg_away
        if xg_diff > self.xg_threshold:
            # Estimate win probability
            prob_win = self.estimate_win_probability(xg_home, xg_away)
            implied_prob = self.calculate_implied_probability(home_odds)
            edge = prob_win - implied_prob
            
            # Generate signal if edge exceeds minimum threshold
            if edge > self.min_edge:
                stake = self.calculate_stake(prob_win, home_odds)
                if stake > 0:
                    signal = {
                        'team': 'home',
                        'side': 'home',
                        'odds': home_odds,
                        'stake': round(stake, 2)
                    }
        
        return signal