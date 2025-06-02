from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, Optional
from smarkets_Sports_Quant_Trading.backtester.Backtest_Management.backtester import Backtester

class BacktestRunnerInterface(ABC):
    """Abstract base class for backtest runners in sports betting systems."""
    
    def __init__(self):
        """Initialize the backtest runner with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def run_backtests(self, config_path: str, backtester_class: Type[Backtester], 
                      strategies: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Run backtests for one or more strategies.
        
        Args:
            config_path (str): Path to the configuration file.
            backtester_class (Type[Backtester]): Backtester class to use.
            strategies (Optional[List[Dict[str, Any]]]): List of strategy configurations.
            
        Returns:
            List[Dict[str, Any]]: List of backtest results.
        """
        pass