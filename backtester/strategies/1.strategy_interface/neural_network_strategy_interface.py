from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class NNStrategyInterface(ABC):
    """Abstract base class for neural network-based betting strategies."""
    
    def __init__(self):
        """Initialize the strategy with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def build_model(self) -> None:
        """Build or initialize the neural network model architecture."""
        pass
    
    @abstractmethod
    def train_model(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, epochs: int = 10) -> None:
        """Train the neural network model on input features and labels.
        
        Args:
            X (np.ndarray): Feature matrix (e.g., match stats, odds).
            y (np.ndarray): Target labels (e.g., match outcomes or probabilities).
            batch_size (int): Number of samples per batch (default: 32).
            epochs (int): Number of training epochs (default: 10).
        """
        pass
    
    @abstractmethod
    def update_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the neural network model with new data (e.g., incremental training).
        
        Args:
            X (np.ndarray): New feature matrix (single or small batch).
            y (np.ndarray): New target labels.
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