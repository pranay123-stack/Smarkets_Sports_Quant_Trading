import streamlit as st
import pandas as pd
import json
import os
from typing import Dict, Type
from backtest_app_interface import BacktestAppInterface
from smarkets_Sports_Quant_Trading.backtester.Backtest_Management.backtester import Backtester
from smarkets_Sports_Quant_Trading.backtester.Backtest_Management.standard_backtester import StandardBacktester
from smarkets_Sports_Quant_Trading.backtester.Backtest_Management.paper_backtester import PaperBacktester
from smarkets_Sports_Quant_Trading.backtester.strategies.strategy_interface.strategy_interface import StrategyInterface
from smarkets_Sports_Quant_Trading.backtester.strategies.poisson_goal_model import PoissonGoalModel
from smarkets_Sports_Quant_Trading.backtester.strategies.bayesian_value_bet import BayesianValueBet
from smarkets_Sports_Quant_Trading.backtester.strategies.expected_goals_edge import ExpectedGoalsEdge
from smarkets_Sports_Quant_Trading.backtester.strategies.draw_probability_physics import DrawProbabilityPhysics
from smarkets_Sports_Quant_Trading.backtester.strategies.underdog_bias_theory import UnderdogBiasTheory
from smarkets_Sports_Quant_Trading.backtester.strategies.ml_model_strategy import MLModelStrategy

class StandardBacktestApp(BacktestAppInterface):
    """Standard Streamlit app for running backtests."""
    
    def __init__(self):
        """Initialize the backtest app."""
        super().__init__()
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        st.set_page_config(page_title="⚽ Elite Sports Quant Backtester", layout="wide")
    
    def load_config(self, config_path: str = "config/backtest_config.json") -> Dict[str, Any]:
        """Load the configuration file."""
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st.warning("Default configuration used.")
            return {
                "normalize_columns": ["B365H", "B365D", "B365A"],
                "risk": {
                    "initial_bankroll": 10000,
                    "max_drawdown_pct": 0.25,
                    "kelly_fraction": 0.5,
                    "max_loss_streak": 4,
                    "dynamic_bet_size": True
                },
                "odds_fields": ["B365H", "B365D", "B365A"]
            }
    
    def run(self, strategy_registry: Dict[str, Type[StrategyInterface]], 
            backtester_types: Dict[str, Type[Backtester]]) -> None:
        """Run the Streamlit app."""
        st.title("📊 Elite Sports Quant Backtester")
        
        mode = st.selectbox("Backtest Mode", options=list(backtester_types.keys()), index=0)
        backtester_class = backtester_types[mode]
        
        data_source = None
        if mode == "Historical":
            uploaded_file = st.file_uploader("Upload match odds CSV", type=["csv"])
            if uploaded_file:
                data_source = uploaded_file
                df = pd.read_csv(uploaded_file)
        else:
            api_url = st.text_input("API URL", value="https://api.sportsdata.io")
            api_key = st.text_input("API Key", type="password")
            if api_url and api_key:
                config = self.load_config()
                config['api_url'] = api_url
                config['api_key'] = api_key
                data_source = "live_matches"
        
        selected_strategies = st.multiselect(
            "Select strategies to run",
            options=list(strategy_registry.keys()),
            default=["PoissonGoalModel"]
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            min_edge = st.slider("Minimum Edge %", 0.0, 20.0, 5.0) / 100
        with col2:
            stake = st.number_input("Stake per Bet", value=100)
        with col3:
            initial_bankroll = st.number_input("Starting Bankroll", value=10000)
        
        if st.button("🚀 Run Backtest") and data_source and selected_strategies:
            config = self.load_config()
            config['risk']['initial_bankroll'] = initial_bankroll
            results = []
            combined_pnl = []
            
            for strategy_name in selected_strategies:
                st.subheader(f"📈 Running: {strategy_name}")
                
                strategy_config = config.copy()
                strategy_config['data_path'] = data_source
                strategy_config['strategy'] = {
                    "name": strategy_name,
                    "min_edge": min_edge,
                    "odds_fields": config.get('odds_fields', ["B365H", "B365D", "B365A"])
                }
                
                try:
                    strategy_class = strategy_registry[strategy_name]
                    strategy_instance = strategy_class(strategy_config["strategy"])
                    backtester = backtester_class(strategy=strategy_instance, config=strategy_config)
                    result = backtester.run_backtest()
                    
                    if result:
                        results.append(result)
                        combined_pnl.append(result.get("pnl_curve", []))
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("ROI", f"{result.get('roi', 0):.2f}%")
                        with col2:
                            st.metric("Profit", f"₹{result.get('total_profit', 0):.2f}")
                        with col3:
                            st.metric("Bets Placed", result.get('bets_placed', 0))
                        
                        if "pnl_curve" in result:
                            st.line_chart(pd.Series(result["pnl_curve"], name=strategy_name))
                        
                        summary_df = pd.DataFrame([result])
                        summary_path = os.path.join(self.results_dir, f"{strategy_name}_summary.csv")
                        summary_df.to_csv(summary_path, index=False)
                        st.success(f"Results saved to {summary_path}")
                    
                except Exception as e:
                    st.error(f"Backtest failed for {strategy_name}: {e}")
            
            if len(combined_pnl) > 1 and all(combined_pnl):
                st.subheader("📉 Portfolio Performance")
                df_combined = pd.DataFrame(combined_pnl).sum(axis=0)
                st.line_chart(df_combined)
                final_bankroll = df_combined.iloc[-1] + initial_bankroll if len(df_combined) > 0 else initial_bankroll
                total_profit = df_combined.iloc[-1] if len(df_combined) > 0 else 0
                roi = (total_profit / (len(df_combined) * stake)) * 100 if len(df_combined) > 0 else 0
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Portfolio Final Bankroll", f"₹{final_bankroll:.2f}")
                with col2:
                    st.metric("Portfolio ROI", f"{roi:.2f}%")

# Run the app
if __name__ == "__main__":
    STRATEGIES = {
        "PoissonGoalModel": PoissonGoalModel,
        "BayesianValueBet": BayesianValueBet,
        "ExpectedGoalsEdge": ExpectedGoalsEdge,
        "DrawProbabilityPhysics": DrawProbabilityPhysics,
        "UnderdogBiasTheory": UnderdogBiasTheory,
        "MLModelStrategy": MLModelStrategy
    }
    BACKTESTER_TYPES = {
        "Historical": StandardBacktester,
        "Paper": PaperBacktester
    }
    app = StandardBacktestApp()
    app.run(strategy_registry=STRATEGIES, backtester_types=BACKTESTER_TYPES)