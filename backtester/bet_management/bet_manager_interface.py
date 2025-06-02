from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
import pandas as pd

class BetManagerInterface(ABC):
    """Abstract base class for bet managers in sports betting systems."""
    
    def __init__(self):
        """Initialize the bet manager with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def log_bet(self, strategy: str, date: str, team: str, side: str, odds: float, stake: float, 
                pnl: float, bankroll: float, drawdown: float, loss_streak: int) -> None:
        """Log a new bet for a strategy.
        
        Args:
            strategy (str): Name of the strategy.
            date (str): Date of the bet.
            team (str): Team bet on (e.g., 'home', 'away', 'draw').
            side (str): Betting side (same as team).
            odds (float): Betting odds.
            stake (float): Stake amount.
            pnl (float): Profit or loss from the bet.
            bankroll (float): Current bankroll.
            drawdown (float): Current drawdown.
            loss_streak (int): Current loss streak.
        """
        pass
    
    @abstractmethod
    def remove_bets_by_condition(self, strategy: str, condition_fn: Callable[[Dict[str, Any]], bool]) -> int:
        """Remove bets for a strategy based on a condition.
        
        Args:
            strategy (str): Name of the strategy.
            condition_fn (Callable[[Dict[str, Any]], bool]): Function returning True for bets to remove.
            
        Returns:
            int: Number of bets removed.
        """
        pass
    
    @abstractmethod
    def adjust_bets_by_risk(self, strategy: str, max_stake: Optional[float] = None, 
                           portfolio_risk_level: Optional[float] = None) -> None:
        """Adjust bet stakes based on risk parameters.
        
        Args:
            strategy (str): Name of the strategy.
            max_stake (Optional[float]): Maximum stake per bet.
            portfolio_risk_level (Optional[float]): Risk level to scale stakes (0 to 1).
        """
        pass
    
    @abstractmethod
    def export(self, strategy: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Export bets for a strategy or all strategies.
        
        Args:
            strategy (Optional[str]): Name of the strategy (None for all).
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Bets by strategy.
        """
        pass
    
    @abstractmethod
    def summary(self, strategy: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Generate a summary of betting performance.
        
        Args:
            strategy (Optional[str]): Name of the strategy (None for all).
            
        Returns:
            Dict[str, Dict[str, Any]]: Summary metrics by strategy.
        """
        pass