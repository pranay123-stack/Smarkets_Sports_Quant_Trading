import pandas as np
import pandas as pd
import tensorflow as tf
from dl_strategy_interface import DLStrategyInterface

class LSTMOutcomePredictor(DLStrategyInterface):
    """DL betting strategy using LSTM to predict match outcomes from sequential team performance."""
    
    def __init__(self, bankroll: float = 1000.0, min_edge: float = 0.05, sequence_length: int = 5):
        """Initialize the LSTM strategy with betting parameters.
        
        Args:
            bankroll (float): Total betting capital (default: 1000.0).
            min_edge (float): Minimum edge to place a bet (default: 0.05).
            sequence_length (int): Number of recent matches to consider (default: 5).
        """
        super().__init__()
        self.bankroll = bankroll
        self.min_edge = min_edge
        self.sequence_length = sequence_length
        self.model = self._build_model()
        self.label_map = {'home': 0, 'away': 1, 'draw': 2}
    
    def _build_model(self) -> tf.keras.Model:
        """Build an LSTM model for outcome prediction."""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, input_shape=(self.sequence_length, 3)),  # 3 features per match: xG_diff, rating_diff, odds_diff
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 outcomes
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model
    
    def _prepare_sequence(self, row: pd.Series) -> np.ndarray:
        """Prepare a sequence of recent match features for a single row.
        
        Args:
            row (pd.Series): Input row with 'recent_matches' (list of dicts with 'xG_diff', 'rating_diff', 'odds_diff').
            
        Returns:
            np.ndarray: Sequence shape (1, sequence_length, 3).
        """
        recent_matches = row.get('recent_matches', [])
        sequence = np.zeros((self.sequence_length, 3))
        for i, match in enumerate(recent_matches[-self.sequence_length:]):
            sequence[i] = [
                match.get('xG_diff', 0.0),
                match.get('rating_diff', 0.0),
                match.get('odds_diff', 0.0)
            ]
        return sequence.reshape(1, self.sequence_length, 3)
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the LSTM model on sequence data.
        
        Args:
            X (np.ndarray): Feature sequences (shape: samples, sequence_length, features).
            y (np.ndarray): Target labels (0: home, 1: away, 2: draw).
        """
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    
    def fine_tune(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fine-tune the LSTM model with new data.
        
        Args:
            X (np.ndarray): New feature sequence (shape: 1, sequence_length, features).
            y (np.ndarray): New target label (shape: 1).
        """
        self.model.fit(X, y, epochs=1, verbose=0)
    
    def calculate_stake(self, prob: float, odds: float) -> float:
        """Calculate stake using Kelly Criterion.
        
        Args:
            prob (float): Predicted probability.
            odds (float): Decimal odds.
            
        Returns:
            float: Recommended stake.
        """
        implied_prob = 1 / odds if odds > 1 else 0.0
        edge = prob - implied_prob
        fraction = edge / (odds - 1) if odds > 1 and edge > 0 else 0
        return max(0, min(self.bankroll * fraction, self.bankroll * 0.1))
    
    def generate_signal(self, row: pd.Series) -> dict:
        """Generate a betting signal using LSTM predictions.
        
        Args:
            row (pd.Series): Input row with 'recent_matches', 'home_odds', 'away_odds', 'draw_odds'.
            
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
        
        # Prepare sequence
        sequence = self._prepare_sequence(row)
        
        # Predict probabilities
        probs = self.model.predict(sequence, verbose=0)[0]
        outcomes = [('home', home_odds, probs[0]), ('away', away_odds, probs[1]), ('draw', draw_odds, probs[2])]
        
        # Select best value bet
        for team, odds, prob in outcomes:
            if prob > 0.3:  # Threshold for confidence
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