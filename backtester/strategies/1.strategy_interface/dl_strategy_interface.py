from abc import StrategyInterface, abstractmethod
import pandas as pd
import numpy as np

class DLStrategyInterface(StrategyInterface):
    """Abstract base class for DL betting strategies."""
    
    def __init__(self):
        """Initialize the strategy with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the DL model using input features and labels.
        
        Args:
            X (np.ndarray): Feature matrix (e.g., match stats, odds).
            y (np.ndarray): Target labels (e.g., match outcomes).
        """
        pass
    
    @abstractmethod
    def fine_tune(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fine-tune the DL model with new data.
        
        Args:
            X (np.ndarray): New feature matrix.
            y (np.ndarray): New target labels.
        """
        pass
    
    @abstractmethod
    def generate_signal(self, row: pd.Series) -> dict:
        """
        Generate a betting signal for a single row.
        
        Args:
            row (pd.Series): Input row with match data (e.g., odds, team stats).
        
        Returns:
            dict: Dictionary containing the signal with the following keys:
                - 'team': 'home', 'away', or 'draw'
                - 'side': same as team
                - 'odds': float, betting odds
                - 'stake': float, recommended stake
        """
        pass