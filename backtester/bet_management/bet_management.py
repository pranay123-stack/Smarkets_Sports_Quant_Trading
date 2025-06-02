class BetManager:
    def __init__(self):
        self.strategy_bets = {}

    def log_bet(self, strategy, date, team, side, odds, stake, pnl, bankroll, drawdown, loss_streak):
        if strategy not in self.strategy_bets:
            self.strategy_bets[strategy] = []
        self.strategy_bets[strategy].append({
            'Date': date,
            'Team': team,
            'Side': side,
            'Odds': odds,
            'Stake': stake,
            'P&L': pnl,
            'Bankroll': bankroll,
            'Drawdown': drawdown,
            'Loss Streak': loss_streak
        })

    def remove_bets_by_condition(self, strategy, condition_fn):
        if strategy in self.strategy_bets:
            original_count = len(self.strategy_bets[strategy])
            self.strategy_bets[strategy] = [bet for bet in self.strategy_bets[strategy] if not condition_fn(bet)]
            removed_count = original_count - len(self.strategy_bets[strategy])
            return removed_count
        return 0

    def adjust_bets_by_risk(self, strategy, max_stake=None, portfolio_risk_level=None):
        if strategy not in self.strategy_bets:
            return

        for bet in self.strategy_bets[strategy]:
            if max_stake is not None and bet['Stake'] > max_stake:
                bet['Stake'] = max_stake
            if portfolio_risk_level is not None:
                bet['Stake'] *= (1 - portfolio_risk_level)

    def export(self, strategy=None):
        if strategy:
            return self.strategy_bets.get(strategy, [])
        return self.strategy_bets

    def summary(self, strategy=None):
        import pandas as pd
        from collections import defaultdict

        result = defaultdict(dict)

        strategies = [strategy] if strategy else self.strategy_bets.keys()
        for strat in strategies:
            data = self.strategy_bets[strat]
            if not data:
                continue

            df = pd.DataFrame(data)
            wins = df[df['P&L'] > 0].shape[0]
            losses = df[df['P&L'] < 0].shape[0]
            total_bets = len(df)
            total_profit = df['P&L'].sum()
            avg_pnl = df['P&L'].mean()
            win_rate = wins / total_bets * 100 if total_bets > 0 else 0
            max_drawdown = df['Drawdown'].max() if 'Drawdown' in df else 0

            result[strat] = {
                'Total Bets': total_bets,
                'Wins': wins,
                'Losses': losses,
                'Total Profit': total_profit,
                'Average P&L': avg_pnl,
                'Win Rate (%)': win_rate,
                'Max Drawdown': max_drawdown
            }

        return dict(result)




