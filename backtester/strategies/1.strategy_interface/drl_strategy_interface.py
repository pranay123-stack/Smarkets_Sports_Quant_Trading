from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class DRLStrategyInterface(ABC):
    """Abstract base class for DRL betting strategies."""
    
    def __init__(self):
        """Initialize the strategy with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def initialize_model(self, data: pd.DataFrame) -> None:
        """Initialize the DRL model (neural network) using historical data.
        
        Args:
            data (pd.DataFrame): Historical data (e.g., match stats, odds, outcomes).
        """
        pass
    
    @abstractmethod
    def update_model(self, state: np.ndarray, action: str, reward: float, next_state: np.ndarray) -> None:
        """Update the DRL model with a new experience.
        
        Args:
            state (np.ndarray): Current state vector (e.g., match features).
            action (str): Action taken (e.g., 'draw', 'none').
            reward (float): Reward from the action (e.g., profit/loss).
            next_state (np.ndarray): Next state vector.
        """
        pass
    
    @abstractmethod
    def generate_signal(self, row: pd.Series) -> dict:
        """Generate a betting signal for a single row.
        
        Args:
            row (pd.Series): Input row with match data (e.g., odds, team stats).
            
        Returns:
            dict: Signal dictionary with keys:
                  - 'team': 'home', 'away', or 'draw'
                  - 'side': same as team
                  - 'odds': float, betting odds
                  - 'stake': float, recommended stake
        """
        pass