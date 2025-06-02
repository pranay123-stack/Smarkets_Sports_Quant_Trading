import pandas as pd
from datetime import datetime
import os
import traceback
import random
import streamlit as st
from smarkets_Sports_Quant_Trading.backtester.Logger.logger import Logger
from smarkets_Sports_Quant_Trading.backtester.risk_management.risk_management import RiskManager
from smarkets_Sports_Quant_Trading.backtester.bet_management.bet_management import BetManager
from smarkets_Sports_Quant_Trading.backtester.Execution_Management.execution_management import ExecutionManager
from smarkets_Sports_Quant_Trading.backtester.position_management.position_management import PositionManager
from smarkets_Sports_Quant_Trading.backtester.portfolio_management.portfolio_management import PortfolioManager
from smarkets_Sports_Quant_Trading.backtester.Data_loader.live_data_loader import LiveDataLoader
from smarkets_Sports_Quant_Trading.backtester.Data_Handler.data_handler import DataHandler
from smarkets_Sports_Quant_Trading.backtester.strategies.strategy_interface.strategy_interface import StrategyInterface
from smarkets_Sports_Quant_Trading.backtester.strategy_metrics.metrics import Metrics
from typing import Dict, Any, Optional
import numpy as np

class PaperBacktester(BacktesterInterface):
    """Backtester for paper trading with live API data."""
    
    def __init__(self):
        """Initialize the backtester."""
        super().__init__()
        self.strategy = None
        self.config = None
        self.logger = None
        self.bet_manager = BetManager()
        self.execution_manager = ExecutionManager()
        self.position_manager = PositionManager()
        self.risk_manager = None
        self.portfolio_manager = PortfolioManager()
        self.loader = None
        self.handler = DataHandler()
        self.metrics = Metrics()
        self.strategy_report_path = None
        self.live_bet_container = None  # Streamlit container for live bets
    
    def configure(self, strategy: StrategyInterface, config: Dict[str, Any]) -> None:
        """Configure the backtester."""
        self.strategy = strategy
        self.config = config
        self.logger = Logger(log_dir="Logs")
        self.risk_manager = RiskManager(config.get('risk', {}))
        self.loader = LiveDataLoader(api_url=config.get('api_url', 'https://api.sportsdata.io'), 
                                     api_key=config.get('api_key'))
        self.strategy_report_path = os.path.join("strategy_report", f"{self.strategy.name}_paper_report.csv")
        os.makedirs("strategy_report", exist_ok=True)
        # Initialize Streamlit container for live bets
        self.live_bet_container = st.empty() if 'st' in globals() else None
    
    def run_backtest(self) -> Optional[Dict[str, Any]]:
        """Run the paper backtest with live API data."""
        try:
            # Load live data (simulated real-time stream)
            df = self.loader.load_data('live_matches', parse_dates=['Date'])
            df = self.handler.clean_data(df)
            df = self.handler.enforce_consistency(df)
            df = self.handler.fill_missing_values(df, method="ffill")
            df = self.handler.normalize_features(df, self.config.get('normalize_columns', []))
            
            self.logger.info("Live data loaded and preprocessed")
            
            report_logs = []
            live_bets = []  # Track live bets for display
            
            # Process live data (simulated single pass for paper trading)
            for index, row in df.iterrows():
                try:
                    date = row.get('Date', datetime.now())
                    self.logger.new_log_cycle(index, date)
                    
                    signal = self.strategy.generate_signal(row)
                    signal = self.adjust_signal_by_volatility(signal, row)  # Quant model
                    
                    if not signal or self.risk_manager.should_stop_trading(self.strategy.name):
                        continue
                    
                    odds = signal.get('odds')
                    stake = self.risk_manager.calculate_bet_size(self.strategy.name, odds)
                    
                    if stake == 0:
                        continue
                    
                    execution_result = self.execution_manager.execute_bet(
                        strategy=self.strategy.name,
                        match_id=row.get('MatchID', f"live_{index}"),
                        side=signal.get('side'),
                        requested_odds=odds,
                        stake=stake
                    )
                    
                    if execution_result['status'] != 'executed':
                        continue
                    
                    # Simulate outcome for paper trading
                    result = row.get('FTR', 'H' if random.random() < 0.5 else 'A')  # Simulated
                    pnl = stake * (execution_result['executed_odds'] - 1) if result == signal['team'] else -stake
                    
                    # Log bet
                    self.bet_manager.log_bet(
                        strategy=self.strategy.name,
                        date=date,
                        team=signal.get('team'),
                        side=signal.get('side'),
                        odds=execution_result['executed_odds'],
                        stake=stake,
                        pnl=pnl,
                        bankroll=self.risk_manager.get_bankroll(self.strategy.name),
                        drawdown=self.risk_manager.get_drawdown(self.strategy.name),
                        loss_streak=self.risk_manager.get_loss_streak(self.strategy.name)
                    )
                    
                    # Update position and portfolio
                    self.position_manager.update_position(
                        strategy=self.strategy.name,
                        match_id=row.get('MatchID', f"live_{index}"),
                        pnl=pnl
                    )
                    self.risk_manager.update(self.strategy.name, pnl)
                    self.portfolio_manager.update(self.strategy.name, pnl)
                    
                    # Log and display live bet
                    bet_details = {
                        "Date": date,
                        "MatchID": row.get('MatchID', f"live_{index}"),
                        "Team": signal.get('team'),
                        "Side": signal.get('side'),
                        "Odds": execution_result['executed_odds'],
                        "Stake": stake,
                        "PnL": pnl,
                        "Bankroll": self.risk_manager.get_bankroll(self.strategy.name)
                    }
                    live_bets.append(bet_details)
                    self.logger.log_bet_details(self.strategy.name, signal, pnl, stake)
                    
                    # Display live bet in Streamlit (if available)
                    if self.live_bet_container:
                        with self.live_bet_container:
                            st.write(f"Live Bet: {bet_details['Team']} ({bet_details['Side']}) at {bet_details['Odds']:.2f} odds, Stake: ₹{bet_details['Stake']:.2f}")
                    
                    report_logs.append(bet_details)
                
                except Exception as e:
                    self.logger.error(f"Error in match index {index}: {e}\n{traceback.format_exc()}")
            
            # Final reporting
            results = self.bet_manager.summary(self.strategy.name)
            report_df = pd.DataFrame(report_logs)
            report_df.to_csv(self.strategy_report_path, index=False)
            
            self.logger.info(f"Paper Backtest Summary: {results}")
            
            # Ensure metrics match historical backtester format
            if results:
                results['pnl_curve'] = [log['PnL'] for log in report_logs]
                results.setdefault('roi', (results.get('total_profit', 0) / initial_bankroll) * 100 if initial_bankroll > 0 else 0)
                results.setdefault('bets_placed', len(report_logs))
            
            return results
        
        except Exception as e:
            self.logger.error(f"Paper backtest failed: {e}\n{traceback.format_exc()}")
            return None
    
    def get_report(self) -> pd.DataFrame:
        """Retrieve the paper backtest report."""
        if os.path.exists(self.strategy_report_path):
            return pd.read_csv(self.strategy_report_path)
        return pd.DataFrame()
    
    def adjust_signal_by_volatility(self, signal: Optional[Dict[str, Any]], row: pd.Series) -> Optional[Dict[str, Any]]:
        """Adjust signal based on odds volatility (non-interface method).
        
        Args:
            signal (Optional[Dict[str, Any]]): Strategy signal.
            row (pd.Series): Match data.
            
        Returns:
            Optional[Dict[str, Any]]: Adjusted signal.
        """
        if not signal:
            return None
        
        odds_history = row.get('odds_history', [signal.get('odds', 1.0)])
        volatility = np.std(odds_history) if len(odds_history) > 1 else 0.0
        
        if volatility > 0.5:
            signal = signal.copy()
            signal['stake'] = signal.get('stake', 0) * 0.5
        
        return signal