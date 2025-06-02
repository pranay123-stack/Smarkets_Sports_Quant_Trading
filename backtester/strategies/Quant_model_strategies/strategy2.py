# --- strategies/BayesianValueBet.py ---
import pandas as pd
import numpy as np
from smarkets_Sports_Quant_Trading.backtester.strategies.strategy_interface.strategy_interface import StrategyInterface

class BayesianValueBet(StrategyInterface):
    """Bayesian-inspired betting strategy comparing prior win probability to implied odds."""
    
    def __init__(self, prior_win_prob: float, min_edge: float = 0.05, bankroll: float = 1000.0):
        """Initialize the strategy with prior win probability and betting parameters.
        
        Args:
            prior_win_prob (float): Prior probability of home team winning (0 to 1).
            min_edge (float): Minimum edge to place a bet (default: 0.05).
            bankroll (float): Total betting capital (default: 1000.0).
        """
        super().__init__()
        self.prior = prior_win_prob
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
    
    def calculate_stake(self, edge: float, odds: float) -> float:
        """Calculate stake using a simplified Kelly Criterion.
        
        Args:
            edge (float): Betting edge (prior probability - implied probability).
            odds (float): Decimal odds.
            
        Returns:
            float: Recommended stake, capped at 10% of bankroll.
        """
        fraction = edge / (odds - 1) if odds > 1 and edge > 0 else 0
        stake = self.bankroll * fraction
        return max(0, min(stake, self.bankroll * 0.1))  # Cap at 10% of bankroll
    
    def generate_signal(self, row: pd.Series) -> dict:
        """Generate a betting signal for a single row based on Bayesian value betting.
        
        Args:
            row (pd.Series): Input row with at least 'home_odds' column for home team odds.
            
        Returns:
            dict: Signal dictionary with 'team', 'side', 'odds', 'stake'.
        """
        signal = {'team': None, 'side': None, 'odds': 0.0, 'stake': 0.0}
        
        # Extract home odds, default to 1.0 if missing or invalid
        home_odds = row.get('home_odds', 1.0)
        if not isinstance(home_odds, (int, float)) or home_odds <= 1.0:
            return signal
        
        # Calculate edge
        implied_prob = self.calculate_implied_probability(home_odds)
        edge = self.prior - implied_prob
        
        # Generate signal if edge exceeds minimum threshold
        if edge > self.min_edge:
            stake = self.calculate_stake(edge, home_odds)
            if stake > 0:
                signal = {
                    'team': 'home',
                    'side': 'home',
                    'odds': home_odds,
                    'stake': round(stake, 2)
                }
        
        return signal