import pandas as pd
import numpy as np
import tensorflow as tf
from drl_strategy_interface import DRLStrategyInterface

class PPOLiveDrawBet(DRLStrategyInterface):
    """DRL betting strategy using PPO for in-play draw betting."""
    
    def __init__(self, learning_rate: float = 0.001, bankroll: float = 1000.0):
        """Initialize the PPO strategy with DRL parameters.
        
        Args:
            learning_rate (float): Learning rate for neural network (default: 0.001).
            bankroll (float): Total betting capital (default: 1000.0).
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.bankroll = bankroll
        self.actions = ['draw', 'none']
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.memory = []  # Store (state, action, reward, prob) transitions
        self.batch_size = 32
    
    def _build_actor(self) -> tf.keras.Model:
        """Build actor network for PPO policy."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),  # 6 features: score_diff, time, xG_diff, draw_odds, bankroll, draw_prob
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(self.actions), activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def _build_critic(self) -> tf.keras.Model:
        """Build critic network for PPO value estimation."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def _get_state(self, row: pd.Series) -> np.ndarray:
        """Convert row to state vector for PPO, including physics-inspired draw probability.
        
        Args:
            row (pd.Series): Input row with live data.
            
        Returns:
            np.ndarray: State vector [score_diff, time_remaining, xG_diff, draw_odds, bankroll, draw_prob].
        """
        score_diff = row.get('score_diff', 0.0)
        time_remaining = row.get('time_remaining', 90.0) / 90.0
        xg_diff = row.get('home_xg', 1.2) - row.get('away_xg', 0.9)
        draw_odds = row.get('draw_odds', 1.0)
        bankroll = self.bankroll / 1000.0
        # Physics-inspired draw probability (from ParticleInteractionBet)
        home_rating = row.get('home_rating', 1500.0)
        away_rating = row.get('away_rating', 1500.0)
        delta_rating = abs(home_rating - away_rating)
        draw_prob = 0.35 * np.exp(-(delta_rating ** 2) / (2 * 400 ** 2))
        return np.array([score_diff, time_remaining, xg_diff, draw_odds, bankroll, draw_prob])
    
    def initialize_model(self, data: pd.DataFrame) -> None:
        """Initialize the PPO model using historical in-play data.
        
        Args:
            data (pd.DataFrame): Historical data with live match features and 'result'.
        """
        for _, row in data.iterrows():
            state = self._get_state(row)
            action = np.random.choice(self.actions)
            reward = 0.0
            if action == 'draw':
                odds = row.get('draw_odds', 1.0)
                reward = (odds - 1) * 100 if row.get('result') == 'draw' else -100
            probs = self.actor.predict(state.reshape(1, -1), verbose=0)[0]
            self.memory.append((state, action, reward, probs[self.actions.index(action)]))
        self._train_batch()
    
    def _train_batch(self) -> None:
        """Train PPO model on a batch of experiences (simplified)."""
        if len(self.memory) < self.batch_size:
            return
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = np.array([self.memory[i][0] for i in batch])
        actions = [self.actions.index(self.memory[i][1]) for i in batch]
        rewards = np.array([self.memory[i][2] for i in batch])
        
        # Simplified PPO update: train critic
        values = self.critic.predict(states, verbose=0)
        advantages = rewards - values.flatten()
        self.critic.fit(states, rewards, epochs=1, verbose=0)
        
        # Train actor (simplified)
        action_probs = self.actor.predict(states, verbose=0)
        target_probs = action_probs.copy()
        for i, action in enumerate(actions):
            target_probs[i][action] += 0.1 * advantages[i]
        self.actor.fit(states, target_probs, epochs=1, verbose=0)
    
    def update_model(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray) -> None:
        """Update PPO model with new experience.
        
        Args:
            state (np.ndarray): Current state vector.
            action (str): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state vector.
        """
        probs = self.actor.predict(state.reshape(1, -1), verbose=0)[0]
        self.memory.append((state, action, reward, probs[self.actions.index(action)]))
        self._train_batch()
    
    def generate_signal(self, row: pd.Series) -> dict:
        """Generate a betting signal using PPO policy.
        
        Args:
            row (pd.Series): Input row with live match data.
            
        Returns:
            dict: Signal dictionary with 'team', 'side', 'odds', 'stake'.
        """
        signal = {'team': None, 'side': None, 'odds': 0.0, 'stake': 0.0}
        
        draw_odds = row.get('draw_odds', 1.0)
        if not isinstance(draw_odds, (int, float)) or draw_odds <= 1.0:
            return signal
        
        state = self._get_state(row)
        probs = self.actor.predict(state.reshape(1, -1), verbose=0)[0]
        action = self.actions[np.argmax(probs)]
        
        if action == 'draw':
            stake = min(100.0, self.bankroll * 0.1)
            signal = {
                'team': 'draw',
                'side': 'draw',
                'odds': draw_odds,
                'stake': round(stake, 2)
            }
        
        return signal