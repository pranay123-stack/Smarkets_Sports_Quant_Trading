# --- portfolio_management.py ---
class PortfolioManager:
    def __init__(self, initial_bankroll):
        self.initial_bankroll = initial_bankroll
        self.portfolios = {}
        self.overall_metrics = {
            'bankroll': initial_bankroll,
            'total_profit': 0,
            'total_loss': 0,
            'max_drawdown': 0,
            'trade_history': []
        }

    def add_strategy(self, strategy_name):
        self.portfolios[strategy_name] = {
            'bankroll': self.initial_bankroll,
            'peak': self.initial_bankroll,
            'total_profit': 0,
            'total_loss': 0,
            'max_drawdown': 0,
            'loss_streak': 0,
            'win_streak': 0,
            'trade_history': [],
            'equity_curve': [],
            'active': True
        }

    def update_strategy(self, strategy_name, pnl):
        p = self.portfolios[strategy_name]
        if not p['active']:
            return

        p['bankroll'] += pnl
        p['total_profit'] += pnl if pnl > 0 else 0
        p['total_loss'] += pnl if pnl < 0 else 0
        p['peak'] = max(p['peak'], p['bankroll'])

        drawdown = p['peak'] - p['bankroll']
        p['max_drawdown'] = max(p['max_drawdown'], drawdown)

        if pnl < 0:
            p['loss_streak'] += 1
            p['win_streak'] = 0
        else:
            p['win_streak'] += 1
            p['loss_streak'] = 0

        p['equity_curve'].append(p['bankroll'])
        p['trade_history'].append(pnl)

        self._update_overall(pnl)

        # Strategy deactivation logic (e.g., 5 loss streak or >30% drawdown)
        if p['loss_streak'] >= 5 or (drawdown / p['peak']) > 0.3:
            p['active'] = False

    def _update_overall(self, pnl):
        self.overall_metrics['bankroll'] += pnl
        self.overall_metrics['total_profit'] += pnl if pnl > 0 else 0
        self.overall_metrics['total_loss'] += pnl if pnl < 0 else 0
        peak = max(self.initial_bankroll, self.overall_metrics['bankroll'])
        drawdown = peak - self.overall_metrics['bankroll']
        self.overall_metrics['max_drawdown'] = max(self.overall_metrics['max_drawdown'], drawdown)
        self.overall_metrics['trade_history'].append(pnl)

    def get_strategy_metrics(self, strategy_name):
        p = self.portfolios[strategy_name]
        roi = (p['bankroll'] - self.initial_bankroll) / self.initial_bankroll * 100
        win_rate = sum(1 for t in p['trade_history'] if t > 0) / len(p['trade_history']) * 100 if p['trade_history'] else 0
        avg_gain = sum(t for t in p['trade_history'] if t > 0) / max(1, sum(1 for t in p['trade_history'] if t > 0))
        avg_loss = sum(t for t in p['trade_history'] if t < 0) / max(1, sum(1 for t in p['trade_history'] if t < 0))
        profit_factor = abs(avg_gain / avg_loss) if avg_loss != 0 else float('inf')

        return {
            'Final Bankroll': p['bankroll'],
            'ROI (%)': roi,
            'Total Profit': p['total_profit'],
            'Total Loss': p['total_loss'],
            'Max Drawdown': p['max_drawdown'],
            'Win Streak': p['win_streak'],
            'Loss Streak': p['loss_streak'],
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor,
            'Active': p['active']
        }

    def get_overall_metrics(self):
        roi = (self.overall_metrics['bankroll'] - self.initial_bankroll) / self.initial_bankroll * 100
        trades = self.overall_metrics['trade_history']
        win_rate = sum(1 for t in trades if t > 0) / len(trades) * 100 if trades else 0
        avg_gain = sum(t for t in trades if t > 0) / max(1, sum(1 for t in trades if t > 0))
        avg_loss = sum(t for t in trades if t < 0) / max(1, sum(1 for t in trades if t < 0))
        profit_factor = abs(avg_gain / avg_loss) if avg_loss != 0 else float('inf')

        return {
            'Final Bankroll': self.overall_metrics['bankroll'],
            'ROI (%)': roi,
            'Total Profit': self.overall_metrics['total_profit'],
            'Total Loss': self.overall_metrics['total_loss'],
            'Max Drawdown': self.overall_metrics['max_drawdown'],
            'Win Rate (%)': win_rate,
            'Profit Factor': profit_factor
        }
