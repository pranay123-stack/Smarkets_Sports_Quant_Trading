import numpy as np
import pandas as pd

class Metrics:
    @staticmethod
    def compute_basic_metrics(bets_df):
        total_bets = len(bets_df)
        wins = bets_df[bets_df['P&L'] > 0].shape[0]
        losses = bets_df[bets_df['P&L'] < 0].shape[0]
        total_profit = bets_df['P&L'].sum()
        avg_pnl = bets_df['P&L'].mean()
        win_rate = wins / total_bets * 100 if total_bets > 0 else 0
        roi = (total_profit / bets_df['Stake'].sum()) * 100 if bets_df['Stake'].sum() > 0 else 0

        return {
            'Total Bets': total_bets,
            'Wins': wins,
            'Losses': losses,
            'Total Profit': total_profit,
            'Average P&L': avg_pnl,
            'Win Rate (%)': win_rate,
            'ROI (%)': roi
        }

    @staticmethod
    def compute_risk_metrics(bets_df):
        equity_curve = bets_df['Bankroll']
        peak = equity_curve.cummax()
        drawdown = (peak - equity_curve)
        max_drawdown = drawdown.max()
        avg_drawdown = drawdown.mean()

        pnl = bets_df['P&L']
        std_dev = np.std(pnl)
        sharpe = pnl.mean() / std_dev if std_dev != 0 else 0

        return {
            'Max Drawdown': max_drawdown,
            'Average Drawdown': avg_drawdown,
            'Sharpe Ratio': sharpe
        }

    @staticmethod
    def compute_strategy_metrics(strategy_bets):
        metrics = {}
        for strat, bets in strategy_bets.items():
            df = pd.DataFrame(bets)
            basic = Metrics.compute_basic_metrics(df)
            risk = Metrics.compute_risk_metrics(df)
            metrics[strat] = {**basic, **risk}
        return metrics

    @staticmethod
    def compute_profit_factor(bets_df):
        gains = bets_df[bets_df['P&L'] > 0]['P&L'].sum()
        losses = abs(bets_df[bets_df['P&L'] < 0]['P&L'].sum())
        return gains / losses if losses > 0 else np.inf
