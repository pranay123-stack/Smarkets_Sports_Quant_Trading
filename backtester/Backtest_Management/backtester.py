import pandas as pd
from datetime import datetime
import os
import traceback
from smarkets_Sports_Quant_Trading.backtester.Logger.logger import Logger
from smarkets_Sports_Quant_Trading.backtester.risk_management.risk_management import RiskManager
from smarkets_Sports_Quant_Trading.backtester.bet_management.bet_management import BetManager
from smarkets_Sports_Quant_Trading.backtester.Execution_Management.execution_management import ExecutionManager
from smarkets_Sports_Quant_Trading.backtester.position_management.position_management import PositionManager
from smarkets_Sports_Quant_Trading.backtester.portfolio_management.portfolio_management import PortfolioManager
from smarkets_Sports_Quant_Trading.backtester.Data_loader.data_loader import DataLoader
from smarkets_Sports_Quant_Trading.backtester.Data_Handler.data_handler import DataHandler
from smarkets_Sports_Quant_Trading.backtester.strategies.strategy_interface.strategy_interface import StrategyInterface
from stastical_tool import StatisticalTools
from  probability_tools  import  ProbabilitySportQuantModels
from Quant_Model import QuantSportsModels
from ml_models import MLModels
from smarkets_Sports_Quant_Trading.backtester.strategy_metrics.metrics import Metrics

class Backtester:
    def __init__(self, strategy: StrategyInterface, config: dict):
        self.strategy = strategy
        self.config = config
        self.logger = Logger(log_dir="Logs")
        self.bet_manager = BetManager()
        self.execution_manager = ExecutionManager()
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(config['risk'])
        self.portfolio_manager = PortfolioManager()
        self.loader = DataLoader()
        self.handler = DataHandler()
        self.stats = StatisticalTools()
        self.prob = ProbabilitySportQuantModels()
        self.quant = QuantSportsModels()
        self.ml = MLModels()
        self.metrics = Metrics()

        self.strategy_report_path = os.path.join("strategy_report", f"{self.strategy.name}_report.csv")
        os.makedirs("strategy_report", exist_ok=True)

    def run_backtest(self):
        try:
            # Load and preprocess data
            df = self.loader.load(self.config['data_path'])
            df = self.handler.clean_data(df)
            df = self.handler.enforce_consistency(df)
            df = self.handler.fill_missing_values(df)
            df = self.handler.normalize_features(df, self.config['normalize_columns'])

            self.logger.info("Data loaded and preprocessed")

            # Initialize report tracking
            report_logs = []

            # Iterate through each match
            for index, row in df.iterrows():
                try:
                    date = row.get('Date', datetime.now())
                    self.logger.new_log_cycle(index, date)

                    signal = self.strategy.generate_signal(row)

                    if not signal or self.risk_manager.should_stop_trading(self.strategy.name):
                        continue

                    odds = signal.get('odds')
                    stake = self.risk_manager.calculate_bet_size(self.strategy.name, odds)

                    if stake == 0:
                        continue

                    pnl = self.execution_manager.execute(signal, stake)
                    self.bet_manager.log_bet(
                        strategy=self.strategy.name,
                        date=date,
                        team=signal.get('team'),
                        side=signal.get('side'),
                        odds=odds,
                        stake=stake,
                        pnl=pnl,
                        bankroll=self.risk_manager.get_bankroll(self.strategy.name),
                        drawdown=self.risk_manager.get_drawdown(self.strategy.name),
                        loss_streak=self.risk_manager.get_loss_streak(self.strategy.name)
                    )

                    self.position_manager.update_position(
                        strategy=self.strategy.name,
                        match_id=row.get('MatchID', index),
                        pnl=pnl
                    )

                    self.risk_manager.update(self.strategy.name, pnl)
                    self.portfolio_manager.update(self.strategy.name, pnl)

                    self.logger.log_bet_details(self.strategy.name, signal, pnl, stake)

                    # Append to report
                    report_logs.append({
                        "Date": date,
                        "MatchID": row.get('MatchID', index),
                        "Team": signal.get('team'),
                        "Side": signal.get('side'),
                        "Odds": odds,
                        "Stake": stake,
                        "PnL": pnl,
                        "Bankroll": self.risk_manager.get_bankroll(self.strategy.name)
                    })

                except Exception as e:
                    self.logger.error(f"Error in match index {index}: {e}\n{traceback.format_exc()}")

            # Final reporting
            results = self.bet_manager.summary(self.strategy.name)
            report_df = pd.DataFrame(report_logs)
            report_df.to_csv(self.strategy_report_path, index=False)

            self.logger.info(f"Backtest Summary for {self.strategy.name}: {results}")
            self.logger.info(f"Report saved at: {self.strategy_report_path}")

            return results

        except Exception as e:
            self.logger.error(f"Backtest failed: {e}\n{traceback.format_exc()}")
            return None
