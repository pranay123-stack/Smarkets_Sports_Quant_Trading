from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable

class ExecutionManagerInterface(ABC):
    """Abstract base class for execution managers in sports betting systems."""
    
    def __init__(self):
        """Initialize the execution manager with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def configure(self, slippage_fn: Optional[Callable[[float], float]] = None, 
                  rejection_rate: Optional[float] = None, 
                  min_stake: Optional[float] = None, 
                  max_stake: Optional[float] = None) -> None:
        """Configure execution parameters.
        
        Args:
            slippage_fn (Optional[Callable[[float], float]]): Function to apply slippage to odds.
            rejection_rate (Optional[float]): Probability of bet rejection (0 to 1).
            min_stake (Optional[float]): Minimum stake amount.
            max_stake (Optional[float]): Maximum stake amount.
        """
        pass
    
    @abstractmethod
    def execute_bet(self, strategy: str, match_id: str, side: str, requested_odds: float, stake: float) -> Dict[str, Any]:
        """Execute a bet with slippage and rejection checks.
        
        Args:
            strategy (str): Name of the strategy.
            match_id (str): Unique identifier for the match.
            side (str): Betting side (e.g., 'home', 'away', 'draw').
            requested_odds (float): Requested odds.
            stake (float): Stake amount.
            
        Returns:
            Dict[str, Any]: Execution result (e.g., {'status': 'executed', 'executed_odds': float, 'stake': float}).
        """
        pass
    
    @abstractmethod
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Retrieve the log of executed bets.
        
        Returns:
            List[Dict[str, Any]]: List of executed bets.
        """
        pass
    
    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        """Generate a summary of execution performance.
        
        Returns:
            Dict[str, Any]: Summary metrics (e.g., total executed, volume).
        """
        pass