from abc import ABC, abstractmethod
import pandas as pd

class StrategyInterface(ABC):
    """Abstract base class defining the interface for betting strategies."""
    
    def __init__(self):
        """Initialize the strategy with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def generate_signal(self, row: pd.Series) -> dict:
        """Generate a betting signal for a single row.
        
        Args:
            row (pd.Series): Input row containing match data (e.g., odds, team stats).
            
        Returns:
            dict: Signal dictionary with keys:
                  - 'team': 'home', 'away', or 'draw'
                  - 'side': same as team
                  - 'odds': float, betting odds
                  - 'stake': float, recommended stake
        """
        pass