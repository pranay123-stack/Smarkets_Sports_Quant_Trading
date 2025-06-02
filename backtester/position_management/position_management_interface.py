from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class PositionManagerInterface(ABC):
    """Abstract base class for position managers in sports betting systems."""
    
    def __init__(self):
        """Initialize the position manager with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def open_position(self, strategy: str, match_id: str, side: str, stake: float, odds: float) -> None:
        """Open a new betting position for a strategy.
        
        Args:
            strategy (str): Name of the strategy.
            match_id (str): Unique identifier for the match.
            side (str): Betting side (e.g., 'home', 'away', 'draw').
            stake (float): Stake amount.
            odds (float): Betting odds.
        """
        pass
    
    @abstractmethod
    def close_position(self, strategy: str, match_id: str, result_side: str) -> float:
        """Close an open position and calculate P&L.
        
        Args:
            strategy (str): Name of the strategy.
            match_id (str): Unique identifier for the match.
            result_side (str): Outcome of the match (e.g., 'home', 'away', 'draw').
            
        Returns:
            float: Profit or loss from the position.
        """
        pass
    
    @abstractmethod
    def evaluate_strategy_risk(self, strategy: str) -> tuple[float, float]:
        """Evaluate the risk exposure and average odds for a strategy's open positions.
        
        Args:
            strategy (str): Name of the strategy.
            
        Returns:
            tuple[float, float]: Total exposure (stake) and average odds.
        """
        pass
    
    @abstractmethod
    def force_close_all(self, strategy: str) -> int:
        """Force-close all open positions for a strategy.
        
        Args:
            strategy (str): Name of the strategy.
            
        Returns:
            int: 1 if positions were closed, 0 if none existed.
        """
        pass
    
    @abstractmethod
    def get_positions(self, strategy: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve positions for a strategy or all strategies.
        
        Args:
            strategy (Optional[str]): Name of the strategy (None for all).
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Positions by strategy.
        """
        pass