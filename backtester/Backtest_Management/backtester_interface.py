from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class BacktesterInterface(ABC):
    """Abstract base class for backtesters in sports betting systems."""
    
    def __init__(self):
        """Initialize the backtester with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def configure(self, strategy: 'StrategyInterface', config: Dict[str, Any]) -> None:
        """Configure the backtester with a strategy and settings.
        
        Args:
            strategy (StrategyInterface): Betting strategy to backtest.
            config (Dict[str, Any]): Configuration (e.g., data path, risk settings).
        """
        pass
    
    @abstractmethod
    def run_backtest(self) -> Optional[Dict[str, Any]]:
        """Run the backtest and return results.
        
        Returns:
            Optional[Dict[str, Any]]: Backtest summary metrics or None if failed.
        """
        pass
    
    @abstractmethod
    def get_report(self) -> pd.DataFrame:
        """Retrieve the detailed backtest report.
        
        Returns:
            pd.DataFrame: Report with bet details (e.g., date, stake, P&L).
        """
        pass