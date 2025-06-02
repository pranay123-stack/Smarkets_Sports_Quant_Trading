from abc import ABC, abstractmethod
from typing import Dict, Type
from smarkets_Sports_Quant_Trading.backtester.strategies.strategy_interface.strategy_interface import StrategyInterface
from smarkets_Sports_Quant_Trading.backtester.Backtest_Management.backtester import Backtester

class BacktestAppInterface(ABC):
    """Abstract base class for backtest application UIs."""
    
    def __init__(self):
        """Initialize the backtest app with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def run(self, strategy_registry: Dict[str, Type[StrategyInterface]], 
            backtester_types: Dict[str, Type[Backtester]]) -> None:
        """Run the backtest application.
        
        Args:
            strategy_registry (Dict[str, Type[StrategyInterface]]): Mapping of strategy names to classes.
            backtester_types (Dict[str, Type[Backtester]]): Mapping of backtest modes to backtester classes.
        """
        pass