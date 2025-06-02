from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class TimeSeriesStrategyInterface(ABC):
    """Abstract base class for time series-based betting strategies."""
    
    def __init__(self):
        """Initialize the strategy with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def build_model(self, data: pd.DataFrame) -> None:
        """Build or initialize the time series model using historical data.
        
        Args:
            data (pd.DataFrame): Historical data with temporal features (e.g., match stats, odds).
        """
        pass
    
    @abstractmethod
    def forecast(self, data: pd.DataFrame) -> np.ndarray:
        """Generate forecasts for the next time step(s) using the time series model.
        
        Args:
            data (pd.DataFrame): Recent temporal data to base the forecast on.
            
        Returns:
            np.ndarray: Forecasted values (e.g., predicted xG, probabilities).
        """
        pass
    
    @abstractmethod
    def update_model(self, new_data: pd.Series) -> None:
        """Update the time series model with new data.
        
        Args:
            new_data (pd.Series): New data point (e.g., latest match stats).
        """
        pass
    
    @abstractmethod
    def generate_signal(self, row: pd.Series) -> dict:
        """Generate a betting signal for a single row.
        
        Args:
            row (pd.Series): Input row with match data (e.g., odds, team stats, recent performance).
            
        Returns:
            dict: Signal dictionary with keys:
                  - 'team': 'home', 'away', or 'draw'
                  - 'side': same as team
                  - 'odds': float, betting odds
                  - 'stake': float, recommended stake
        """
        pass