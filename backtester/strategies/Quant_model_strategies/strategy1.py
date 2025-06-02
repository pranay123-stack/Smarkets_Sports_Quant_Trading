import pandas as pd
from smarkets_Sports_Quant_Trading.backtester.strategies.strategy_interface.strategy_interface import StrategyInterface

class KellyBettingStrategy(StrategyInterface):
    """Quantitative betting strategy using Kelly Criterion and statistical probabilities."""
    
    def __init__(self, bankroll: float = 1000.0, min_edge: float = 0.05):
        """Initialize the strategy with bankroll and minimum edge.
        
        Args:
            bankroll (float): Total betting capital (default: 1000.0).
            min_edge (float): Minimum expected edge to place a bet (default: 0.05).
        """
        super().__init__()
        self.bankroll = bankroll
        self.min_edge = min_edge
    
    def calculate_implied_probability(self, odds: float) -> float:
        """Calculate implied probability from decimal odds.
        
        Args:
            odds (float): Decimal odds (e.g., 2.0 for even money).
            
        Returns:
            float: Implied probability (e.g., 0.5 for odds of 2.0).
        """
        return 1 / odds if odds > 1 else 0.0
    
    def calculate_kelly_stake(self, prob_win: float, odds: float) -> float:
        """Calculate stake using Kelly Criterion.
        
        Args:
            prob_win (float): Estimated probability of winning.
            odds (float): Decimal odds.
            
        Returns:
            float: Recommended stake, capped at 10% of bankroll.
        """
        edge = prob_win * (odds - 1) - (1 - prob_win)
        fraction = edge / (odds - 1) if odds > 1 and edge > 0 else 0
        stake = self.bankroll * fraction
        return max(0, min(stake, self.bankroll * 0.1))  # Cap at 10% of bankroll
    
    def generate_signal(self, row: pd.Series) -> dict:
        """Generate a betting signal for a single row using statistical probabilities.
        
        Args:
            row (pd.Series): Input row with columns:
                            - 'home_odds', 'away_odds', 'draw_odds': float
                            - 'home_win_rate', 'away_win_rate': float (historical win rates)
                            - 'draw_rate': float (historical draw rate)
            
        Returns:
            dict: Signal dictionary with 'team', 'side', 'odds', 'stake'.
        """
        signal = {'team': None, 'side': None, 'odds': 0.0, 'stake': 0.0}
        
        # Extract data from row
        home_odds = row.get('home_odds', 1.0)
        away_odds = row.get('away_odds', 1.0)
        draw_odds = row.get('draw_odds', 1.0)
        home_prob = row.get('home_win_rate', 0.0)
        away_prob = row.get('away_win_rate', 0.0)
        draw_prob = row.get('draw_rate', 0.0)
        
        # Normalize probabilities to sum to 1
        total_prob = home_prob + away_prob + draw_prob
        if total_prob > 0:
            home_prob /= total_prob
            away_prob /= total_prob
            draw_prob /= total_prob
        
        # Calculate expected value for each outcome
        outcomes = [
            ('home', home_prob, home_odds),
            ('away', away_prob, away_odds),
            ('draw', draw_prob, draw_odds)
        ]
        
        best_edge = -float('inf')
        best_outcome = None
        
        for team, prob, odds in outcomes:
            if prob <= 0 or odds <= 1:
                continue
            implied_prob = self.calculate_implied_probability(odds)
            edge = prob - implied_prob
            if edge > self.min_edge and edge > best_edge:
                best_edge = edge
                best_outcome = (team, odds, prob)
        
        if best_outcome:
            team, odds, prob = best_outcome
            stake = self.calculate_kelly_stake(prob, odds)
            if stake > 0:
                signal = {
                    'team': team,
                    'side': team,
                    'odds': odds,
                    'stake': round(stake, 2)
                }
        
        return signal