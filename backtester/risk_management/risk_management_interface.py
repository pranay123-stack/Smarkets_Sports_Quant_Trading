from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class RiskManagerInterface(ABC):
    """Abstract base class for risk managers in sports betting portfolios."""
    
    def __init__(self):
        """Initialize the risk manager with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def configure_strategy(self, strategy: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Configure risk parameters for a strategy.
        
        Args:
            strategy (str): Unique name of the strategy.
            config (Optional[Dict[str, Any]]): Custom risk configuration (e.g., max drawdown).
        """
        pass
    
    @abstractmethod
    def update_bankroll(self, strategy: str, bankroll: float) -> None:
        """Update the strategy's bankroll and related risk metrics (e.g., drawdown).
        
        Args:
            strategy (str): Name of the strategy.
            bankroll (float): Current bankroll for the strategy.
        """
        pass
    
    @abstractmethod
    def update_loss_streak(self, strategy: str, won_last_bet: bool) -> None:
        """Update the strategy's loss streak based on the last bet's outcome.
        
        Args:
            strategy (str): Name of the strategy.
            won_last_bet (bool): True if the last bet was won, False otherwise.
        """
        pass
    
    @abstractmethod
    def should_stop_trading(self, strategy: str) -> bool:
        """Determine if the strategy should stop trading due to risk thresholds.
        
        Args:
            strategy (str): Name of the strategy.
            
        Returns:
            bool: True if trading should stop, False otherwise.
        """
        pass
    
    @abstractmethod
    def calculate_dynamic_stake(self, strategy: str) -> float:
        """Calculate the dynamic stake for a strategy based on its risk profile.
        
        Args:
            strategy (str): Name of the strategy.
            
        Returns:
            float: Recommended stake amount.
        """
        pass
    
    @abstractmethod
    def get_status(self, strategy: str) -> Dict[str, Any]:
        """Retrieve the risk status for a specific strategy.
        
        Args:
            strategy (str): Name of the strategy.
            
        Returns:
            Dict[str, Any]: Risk metrics (e.g., loss streak, drawdown, bankroll).
        """
        pass
    
    @abstractmethod
    def all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve risk statuses for all strategies.
        
        Returns:
            Dict[str, Dict[str, Any]]: Risk metrics for each strategy.
        """
        pass