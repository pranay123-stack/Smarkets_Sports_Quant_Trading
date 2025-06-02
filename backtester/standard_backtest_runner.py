import json
import pandas as pd
import os
from typing import List, Dict, Any, Type, Optional
from backtest_runner_interface import BacktestRunnerInterface
from smarkets_Sports_Quant_Trading.backtester.Backtest_Management.backtester import Backtester
from smarkets_Sports_Quant_Trading.backtester.strategies.strategy_interface.strategy_interface import StrategyInterface
from smarkets_Sports_Quant_Trading.backtester.strategies.arbitrage_strategy import ArbitrageStrategy
from smarkets_Sports_Quant_Trading.backtester.strategies.arima_xg_bet import ARIMAXGBet

class StandardBacktestRunner(BacktestRunnerInterface):
    """Standard backtest runner for executing backtests."""
    
    def __init__(self):
        """Initialize the backtest runner."""
        super().__init__()
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_backtests(self, config_path: str, backtester_class: Type[Backtester], 
                      strategies: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Run backtests for one or more strategies."""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Failed to load config: {e}")
            return []
        
        if strategies is None:
            strategies = config.get('strategies', [config.get('strategy')])
        
        results = []
        
        for strategy_config in strategies:
            try:
                strategy_name = strategy_config.get('name')
                strategy_map = {
                    'ArbitrageStrategy': ArbitrageStrategy,
                    'ARIMAXGBet': ARIMAXGBet,
                    # Add other strategies
                }
                
                strategy_class = strategy_map.get(strategy_name)
                if not strategy_class:
                    print(f"❌ Unknown strategy: {strategy_name}")
                    continue
                
                strategy = strategy_class(strategy_config)
                backtest_config = config.copy()
                backtest_config['strategy'] = strategy_config
                
                backtester = backtester_class(strategy=strategy, config=backtest_config)
                result = backtester.run_backtest()
                
                if result:
                    summary_df = pd.DataFrame([result])
                    summary_path = os.path.join(self.results_dir, f"{strategy.name}_summary.csv")
                    summary_df.to_csv(summary_path, index=False)
                    
                    report_df = backtester.get_report()
                    report_path = os.path.join(self.results_dir, f"{strategy.name}_report.csv")
                    report_df.to_csv(report_path, index=False)
                    
                    results.append({
                        'strategy': strategy.name,
                        'result': result,
                        'summary_path': summary_path,
                        'report_path': report_path
                    })
                    print(f"✅ Backtest completed for {strategy.name}. Results saved to {summary_path}")
                
            except Exception as e:
                print(f"❌ Backtest failed for {strategy_config.get('name', 'unknown')}: {e}")
        
        return results