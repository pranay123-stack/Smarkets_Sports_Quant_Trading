import pandas as pd
import numpy as np
import tensorflow as tf
from nn_strategy_interface import NNStrategyInterface

class FeedForwardValueBet(NNStrategyInterface):
    """Neural network betting strategy using a feedforward NN to predict value bets."""
    
    def __init__(self, bankroll: float = 1000.0, min_edge: float = 0.05, interaction_scale: float = 400.0):
        """Initialize the feedforward NN strategy with betting parameters.
        
        Args:
            bankroll (float): Total betting capital (default: 1000.0).
            min_edge (float): Minimum edge to place a bet (default: 0.05).
            interaction_scale (float): Scale for physics-inspired draw probability (default: 400.0).
        """
        super().__init__()
        self.bankroll = bankroll
        self.min_edge = min_edge
        self.interaction_scale = interaction_scale
        self.model = None
        self.label_map = {'home': 0, 'away': 1, 'draw': 2}
        self.build_model()
    
    def build_model(self) -> None:
        """Build a feedforward neural network for outcome probability prediction."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(5,)),  # 5 features: rating_diff, xG_diff, home_odds, away_odds, draw_prob
            tf.keras.layers.Dropout(0.2),  # Regularization
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 outcomes: home, away, draw
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')
    
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
    
    def _prepare_features(self, row: pd.Series) -> np.ndarray:
        """Prepare feature vector for a single row.
        
        Args:
            row (pd.Series): Input row with 'home_rating', 'away_rating', 'home_xg', 'away_xg', 'home_odds', 'away_odds'.
            
        Returns:
            np.ndarray: Feature vector [rating_diff, xG_diff, home_odds, away_odds, draw_prob].
        """
        home_rating = row.get('home_rating', 1500.0)
        away_rating = row.get('away_rating', 1500.0)
        rating_diff = home_rating - away_rating
        xG_diff = row.get('home_xg', 1.2) - row.get('away_xg', 0.9)
        home_odds = row.get('home_odds', 1.0)
        away_odds = row.get('away_odds', 1.0)
        draw_prob = self._calculate_draw_prob(home_rating, away_rating)
        return np.array([[rating_diff, xG_diff, home_odds, away_odds, draw_prob]])
    
    def train_model(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, epochs: int = 10) -> None:
        """Train the feedforward NN model on input features and labels.
        
        Args:
            X (np.ndarray): Feature matrix (shape: samples, 5).
            y (np.ndarray): Target labels (0: home, 1: away, 2: draw).
            batch_size (int): Number of samples per batch.
            epochs (int): Number of training epochs.
        """
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)
    
    def update_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fine-tune the NN model with new data.
        
        Args:
            X (np.ndarray): New feature matrix (shape: 1, 5).
            y (np.ndarray): New target label (shape: 1).
        """
        self.model.fit(X, y, epochs=1, verbose=0)
    
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
        """Generate a betting signal using feedforward NN predictions.
        
        Args:
            row (pd.Series): Input row with 'home_rating', 'away_rating', 'home_xg', 'away_xg', 'home_odds', 'away_odds', 'draw_odds'.
            
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
        
        # Prepare features
        features = self._prepare_features(row)
        
        # Predict probabilities
        probs = self.model.predict(features, verbose=0)[0]
        outcomes = [('home', home_odds, probs[0]), ('away', away_odds, probs[1]), ('draw', draw_odds, probs[2])]
        
        # Select best value bet
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