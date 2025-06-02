# --- strategies/UnderdogBiasTheory.py ---
import numpy as np
from smarkets_Sports_Quant_Trading.backtester.strategies.strategy_interface.strategy_interface import BaseStrategy
import pandas as pd
import numpy as np
from smarkets_Sports_Quant_Trading.backtester.strategies.strategy_interface.strategy_interface import StrategyInterface

class UnderdogBiasTheory(StrategyInterface):
    """Betting strategy exploiting perceived market bias towards underdog teams with high odds."""
    
    def __init__(self, odds_threshold: float = 4.0, min_edge: float = 0.05, bankroll: float = 1000.0, bias_adjustment: float = 0.1):
        """Initialize the strategy with underdog odds threshold and betting parameters.
        
        Args:
            odds_threshold (float): Minimum odds to consider a team an underdog (default: 4.0).
            min_edge (float): Minimum edge to place a bet (default: 0.05).
            bankroll (float): Total betting capital (default: 1000.0).
            bias_adjustment (float): Adjustment to implied probability to account for market bias (default: 0.1).
        """
        super().__init__()
        self.odds_threshold = odds_threshold
        self.min_edge = min_edge
        self.bankroll = bankroll
        self.bias_adjustment = bias_adjustment
    
    def calculate_implied_probability(self, odds: float) -> float:
        """Calculate implied probability from decimal odds.
        
        Args:
            odds (float): Decimal odds (e.g., 2.0 for even money).
            
        Returns:
            float: Implied probability (e.g., 0.5 for odds of 2.0).
        """
        return 1 / odds if odds > 1 else 0.0
    
    def estimate_win_probability(self, implied_prob: float) -> float:
        """Estimate true win probability by adjusting implied probability for market bias.
        
        Args:
            implied_prob (float): Implied probability from odds.
            
        Returns:
            float: Adjusted probability (0 to 1).
        """
        # Increase perceived probability to account for underdog bias
        adjusted_prob = implied_prob + self.bias_adjustment
        return min(max(adjusted_prob, 0.0), 1.0)
    
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
        """Generate a betting signal for underdog teams based on high odds.
        
        Args:
            row (pd.Series): Input row with 'home_odds', 'away_odds' columns.
            
        Returns:
            dict: Signal dictionary with 'team', 'side', 'odds', 'stake'.
        """
        signal = {'team': None, 'side': None, 'odds': 0.0, 'stake': 0.0}
        
        # Extract data, provide defaults for missing/invalid values
        home_odds = row.get('home_odds', 1.0)
        away_odds = row.get('away_odds', 1.0)
        
        # Validate inputs
        if not all(isinstance(x, (int, float)) for x in [home_odds, away_odds]):
            return signal
        
        # Check for underdog opportunities
        outcomes = []
        if home_odds > self.odds_threshold and home_odds > 1.0:
            outcomes.append(('home', home_odds))
        if away_odds > self.odds_threshold and away_odds > 1.0:
            outcomes.append(('away', away_odds))
        
        # Select the best underdog opportunity
        for team, odds in outcomes:
            implied_prob = self.calculate_implied_probability(odds)
            prob_win = self.estimate_win_probability(implied_prob)
            edge = prob_win - implied_prob
            if edge > self.min_edge:
                stake = self.calculate_stake(prob_win, odds)
                if stake > 0:
                    signal = {
                        'team': team,
                        'side': team,
                        'odds': odds,
                        'stake': round(stake, 2)
                    }
                    break  # Take the first valid underdog bet
        
        return signal