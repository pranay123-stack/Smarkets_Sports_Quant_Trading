import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from timeseries_strategy_interface import TimeSeriesStrategyInterface

class ARIMAXGBet(TimeSeriesStrategyInterface):
    """Time series betting strategy using ARIMA to forecast xG differential for value bets."""
    
    def __init__(self, bankroll: float = 1000.0, min_edge: float = 0.05, sequence_length: int = 10, interaction_scale: float = 400.0):
        """Initialize the ARIMA strategy with betting and time series parameters.
        
        Args:
            bankroll (float): Total betting capital (default: 1000.0).
            min_edge (float): Minimum edge to place a bet (default: 0.05).
            sequence_length (int): Number of recent matches for time series (default: 10).
            interaction_scale (float): Scale for physics-inspired draw probability (default: 400.0).
        """
        super().__init__()
        self.bankroll = bankroll
        self.min_edge = min_edge
        self.sequence_length = sequence_length
        self.interaction_scale = interaction_scale
        self.model = None
        self.history = []  # Store xG_diff and draw_prob time series
    
    def _calculate_draw_prob(self, home_rating: float, away_rating: float) -> float:
        """Calculate physics-inspired draw probability (from ParticleInteractionBet).
        
        Args:
            home_rating (float): Home team rating.
            away_rating (float): Away team rating.
            
        Returns:
            float: Estimated draw probability.
        """
        delta_rating = abs(home_rating - away_rating)
        interaction_strength = np.exp(-(delta_rating ** 2) / (2 * self.interaction_scale ** 2))
        return 0.35 * interaction_strength
    
    def build_model(self, data: pd.DataFrame) -> None:
        """Build the ARIMA model using historical xG differential and draw probability.
        
        Args:
            data (pd.DataFrame): Historical data with 'home_xg', 'away_xg', 'home_rating', 'away_rating'.
        """
        xG_diff = data['home_xg'] - data['away_xg']
        draw_probs = data.apply(lambda row: self._calculate_draw_prob(row['home_rating'], row['away_rating']), axis=1)
        self.history = list(zip(xG_diff, draw_probs))
        if len(self.history) >= self.sequence_length:
            xG_series = np.array([h[0] for h in self.history[-self.sequence_length:]])
            exog = np.array([h[1] for h in self.history[-self.sequence_length:]])
            try:
                self.model = ARIMA(xG_series, exog=exog, order=(1, 0, 1)).fit()
            except:
                self.model = None
    
    def forecast(self, data: pd.DataFrame) -> np.ndarray:
        """Forecast the next xG differential using the ARIMA model.
        
        Args:
            data (pd.DataFrame): Recent data with 'home_xg', 'away_xg', 'home_rating', 'away_rating'.
            
        Returns:
            np.ndarray: Forecasted xG differential (single value).
        """
        if self.model is None or len(data) < 1:
            return np.array([0.0])
        xG_diff = data['home_xg'] - data['away_xg']
        draw_probs = data.apply(lambda row: self._calculate_draw_prob(row['home_rating'], row['away_rating']), axis=1)
        self.history.extend(list(zip(xG_diff, draw_probs)))
        recent_history = self.history[-self.sequence_length:]
        xG_series = np.array([h[0] for h in recent_history])
        exog = np.array([h[1] for h in recent_history])
        try:
            self.model = ARIMA(xG_series, exog=exog, order=(1, 0, 1)).fit()
            forecast = self.model.forecast(steps=1, exog=np.array([[draw_probs.iloc[-1]]]))
            return np.array([forecast[0]])
        except:
            return np.array([0.0])
    
    def update_model(self, new_data: pd.Series) -> None:
        """Update the ARIMA model with a new match data point.
        
        Args:
            new_data (pd.Series): New data with 'home_xg', 'away_xg', 'home_rating', 'away_rating'.
        """
        xG_diff = new_data.get('home_xg', 1.2) - new_data.get('away_xg', 0.9)
        draw_prob = self._calculate_draw_prob(new_data.get('home_rating', 1500.0), new_data.get('away_rating', 1500.0))
        self.history.append((xG_diff, draw_prob))
        if len(self.history) >= self.sequence_length:
            xG_series = np.array([h[0] for h in self.history[-self.sequence_length:]])
            exog = np.array([h[1] for h in self.history[-self.sequence_length:]])
            try:
                self.model = ARIMA(xG_series, exog=exog, order=(1, 0, 1)).fit()
            except:
                self.model = None
    
    def calculate_stake(self, prob: float, odds: float) -> float:
        """Calculate stake using a simplified Kelly Criterion.
        
        Args:
            prob (float): Predicted probability of the outcome.
            odds (float): Decimal odds.
            
        Returns:
            float: Recommended stake, capped at 10% of bankroll.
        """
        implied_prob = 1 / odds if odds > 1 else 0.0
        edge = prob - implied_prob
        fraction = edge / (odds - 1) if odds > 1 and edge > 0 else 0
        stake = self.bankroll * fraction
        return max(0, min(stake, self.bankroll * 0.1))
    
    def generate_signal(self, row: pd.Series) -> dict:
        """Generate a betting signal based on forecasted xG differential.
        
        Args:
            row (pd.Series): Input row with 'home_xg', 'away_xg', 'home_rating', 'away_rating', 'home_odds', 'away_odds', 'draw_odds'.
            
        Returns:
            dict: Signal dictionary with 'team', 'side', 'odds', 'stake'.
        """
        signal = {'team': None, 'side': None, 'odds': 0.0, 'stake': 0.0}
        
        # Validate inputs
        home_odds = row.get('home_odds', 1.0)
        away_odds = row.get('away_odds', 1.0)
        draw_odds = row.get('draw_odds', 1.0)
        if not all(isinstance(x, (int, float)) for x in [home_odds, away_odds, draw_odds]) or max(home_odds, away_odds, draw_odds) <= 1.0:
            return signal
        
        # Forecast xG differential
        xG_diff_forecast = self.forecast(pd.DataFrame([row]))[0]
        
        # Convert xG differential to win probability (logistic model, inspired by ExpectedGoalsEdge)
        prob_home = 1 / (1 + np.exp(-xG_diff_forecast))
        prob_away = 1 / (1 + np.exp(xG_diff_forecast))
        prob_draw = self._calculate_draw_prob(row.get('home_rating', 1500.0), row.get('away_rating', 1500.0))
        # Normalize probabilities
        total_prob = prob_home + prob_away + prob_draw
        if total_prob > 0:
            prob_home /= total_prob
            prob_away /= total_prob
            prob_draw /= total_prob
        
        # Select best value bet
        outcomes = [('home', home_odds, prob_home), ('away', away_odds, prob_away), ('draw', draw_odds, prob_draw)]
        for team, odds, prob in outcomes:
            edge = prob - (1 / odds if odds > 1 else 0.0)
            if edge > self.min_edge:
                stake = self.calculate_stake(prob, odds)
                if stake > 0:
                    signal = {
                        'team': team,
                        'side': team,
                        'odds': odds,
                        'stake': round(stake, 2)
                    }
                    break
        
        return signal