import pandas as pd
import numpy as np
from rl_strategy_interface import RLStrategyInterface

class SARSAUnderdogBet(RLStrategyInterface):
    """RL betting strategy using SARSA to bet on underdog teams with high odds."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, exploration_rate: float = 0.1, bankroll: float = 1000.0):
        """Initialize the SARSA strategy with RL parameters.
        
        Args:
            learning_rate (float): Learning rate for Q-table updates (default: 0.1).
            discount_factor (float): Discount factor for future rewards (default: 0.9).
            exploration_rate (float): Probability of random action (default: 0.1).
            bankroll (float): Total betting capital (default: 1000.0).
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.bankroll = bankroll
        self.actions = ['home', 'away', 'none']
        self.q_table = {}
        self.state_bins = {
            'rating_diff': np.linspace(-1000, 1000, 11),
            'odds': np.linspace(2.0, 10.0, 11)  # Focus on high odds for underdogs
        }
        self.last_state_action = None
    
    def _get_state(self, row: pd.Series) -> tuple:
        """Convert row to discrete state (rating difference, odds).
        
        Args:
            row (pd.Series): Input row with 'home_rating', 'away_rating', 'home_odds', 'away_odds'.
            
        Returns:
            tuple: Discretized state (rating_diff_bin, odds_bin).
        """
        home_rating = row.get('home_rating', 1500.0)
        away_rating = row.get('away_rating', 1500.0)
        home_odds = row.get('home_odds', 1.0)
        away_odds = row.get('away_odds', 1.0)
        rating_diff = home_rating - away_rating
        odds = home_odds if home_odds > away_odds else away_odds
        rating_diff_bin = np.digitize(rating_diff, self.state_bins['rating_diff'], right=True)
        odds_bin = np.digitize(odds, self.state_bins['odds'], right=True)
        return (rating_diff_bin, odds_bin)
    
    def _get_action_values(self, state: tuple) -> dict:
        """Get Q-values for all actions in the given state.
        
        Args:
            state (tuple): Discretized state.
            
        Returns:
            dict: Q-values for each action, initialized to 0 if state is new.
        """
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.actions}
        return self.q_table[state]
    
    def initialize_policy(self, data: pd.DataFrame) -> None:
        """Initialize the SARSA policy using historical data.
        
        Args:
            data (pd.DataFrame): Historical data with 'home_rating', 'away_rating', 'home_odds', 'away_odds', 'result'.
        """
        for _, row in data.iterrows():
            state = self._get_state(row)
            action = np.random.choice(self.actions)
            reward = 0.0
            if action != 'none':
                odds = row.get(f'{action}_odds', 1.0)
                reward = (odds - 1) * 100 if row.get('result') == action else -100
            next_state = state  # Simplified: static state for pre-match
            self.update_policy(state, action, reward, next_state)
    
    def update_policy(self, state: tuple, action: str, reward: float, next_state: tuple) -> None:
        """Update SARSA policy with state-action-reward transition.
        
        Args:
            state (tuple): Current state.
            action (str): Action taken.
            reward (float): Reward received.
            next_state (tuple): Next state.
        """
        next_action = max(self._get_action_values(next_state), key=self._get_action_values(next_state).get)
        current_q = self._get_action_values(state)[action]
        next_q = self._get_action_values(next_state)[next_action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        self.q_table[state][action] = new_q
    
    def generate_signal(self, row: pd.Series) -> dict:
        """Generate a betting signal using SARSA policy.
        
        Args:
            row (pd.Series): Input row with 'home_rating', 'away_rating', 'home_odds', 'away_odds'.
            
        Returns:
            dict: Signal dictionary with 'team', 'side', 'odds', 'stake'.
        """
        signal = {'team': None, 'side': None, 'odds': 0.0, 'stake': 0.0}
        
        home_odds = row.get('home_odds', 1.0)
        away_odds = row.get('away_odds', 1.0)
        if not all(isinstance(x, (int, float)) for x in [home_odds, away_odds]) or max(home_odds, away_odds) <= 1.0:
            return signal
        
        state = self._get_state(row)
        if np.random.random() < self.exploration_rate:
            action = np.random.choice(self.actions)
        else:
            action = max(self._get_action_values(state), key=self._get_action_values(state).get)
        
        if action != 'none' and ((action == 'home' and home_odds > 4.0) or (action == 'away' and away_odds > 4.0)):
            odds = row.get(f'{action}_odds', 1.0)
            stake = min(100.0, self.bankroll * 0.1)  # Simplified stake
            signal = {
                'team': action,
                'side': action,
                'odds': odds,
                'stake': round(stake, 2)
            }
        
        self.last_state_action = (state, action)
        return signal