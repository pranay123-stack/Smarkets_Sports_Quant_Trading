class RiskManager:
    def __init__(self):
        self.strategy_risk = {}
        self.default_config = {
            'max_loss_streak': 5,
            'max_drawdown_pct': 0.2,
            'max_exposure': 0.15,  # % of bankroll
            'dynamic_stake_pct': 0.02,  # 2% of bankroll
            'min_bankroll': 500
        }

    def configure_strategy(self, strategy, config=None):
        self.strategy_risk[strategy] = {
            'loss_streak': 0,
            'peak_bankroll': 0,
            'current_drawdown': 0,
            'bankroll': 0,
            'config': config if config else self.default_config.copy()
        }

    def update_bankroll(self, strategy, bankroll):
        if strategy not in self.strategy_risk:
            self.configure_strategy(strategy)

        risk = self.strategy_risk[strategy]
        if bankroll > risk['peak_bankroll']:
            risk['peak_bankroll'] = bankroll
        drawdown = risk['peak_bankroll'] - bankroll
        risk['current_drawdown'] = drawdown / risk['peak_bankroll'] if risk['peak_bankroll'] > 0 else 0
        risk['bankroll'] = bankroll

    def update_loss_streak(self, strategy, won_last_bet):
        if strategy not in self.strategy_risk:
            self.configure_strategy(strategy)
        if won_last_bet:
            self.strategy_risk[strategy]['loss_streak'] = 0
        else:
            self.strategy_risk[strategy]['loss_streak'] += 1

    def should_stop_trading(self, strategy):
        risk = self.strategy_risk.get(strategy, {})
        cfg = risk.get('config', self.default_config)
        return (
            risk.get('loss_streak', 0) >= cfg['max_loss_streak'] or
            risk.get('current_drawdown', 0) >= cfg['max_drawdown_pct'] or
            risk.get('bankroll', 0) <= cfg['min_bankroll']
        )

    def calculate_dynamic_stake(self, strategy):
        risk = self.strategy_risk.get(strategy, {})
        bankroll = risk.get('bankroll', 0)
        stake = bankroll * risk.get('config', self.default_config)['dynamic_stake_pct']
        return max(1, round(stake, 2))

    def get_status(self, strategy):
        risk = self.strategy_risk.get(strategy, {})
        return {
            'Loss Streak': risk.get('loss_streak', 0),
            'Drawdown %': round(risk.get('current_drawdown', 0) * 100, 2),
            'Bankroll': risk.get('bankroll', 0),
            'Should Stop': self.should_stop_trading(strategy)
        }

    def all_statuses(self):
        return {s: self.get_status(s) for s in self.strategy_risk}
