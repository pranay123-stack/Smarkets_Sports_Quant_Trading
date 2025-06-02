from abc import ABC, abstractmethod
from typing import Dict, Any

class PortfolioManagerInterface(ABC):
    """Abstract base class for portfolio managers handling betting strategies."""
    
    def __init__(self):
        """Initialize the portfolio manager with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def add_strategy(self, strategy_name: str) -> None:
        """Add a new strategy to the portfolio.
        
        Args:
            strategy_name (str): Unique name of the strategy.
        """
        pass
    
    @abstractmethod
    def update_strategy(self, strategy_name: str, pnl: float) -> None:
        """Update the strategy's portfolio with profit/loss and related metrics.
        
        Args:
            strategy_name (str): Name of the strategy.
            pnl (float): Profit or loss from the strategy's bet.
        """
        pass
    
    @abstractmethod
    def get_strategy_metrics(self, strategy_name: str) -> Dict[str, Any]:
        """Retrieve performance metrics for a specific strategy.
        
        Args:
            strategy_name (str): Name of the strategy.
            
        Returns:
            Dict[str, Any]: Metrics (e.g., bankroll, ROI, drawdown).
        """
        pass
    
    @abstractmethod
    def get_overall_metrics(self) -> Dict[str, Any]:
        """Retrieve overall portfolio performance metrics.
        
        Returns:
            Dict[str, Any]: Metrics (e.g., bankroll, ROI, drawdown).
        """
        pass