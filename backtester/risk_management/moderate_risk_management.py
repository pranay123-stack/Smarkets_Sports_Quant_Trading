from risk_manager_interface import RiskManagerInterface
from typing import Dict, Any, Optional

class ModerateRiskManager(RiskManagerInterface):
    """Moderate risk manager for sports betting strategies."""
    
    def __init__(self):
        """Initialize the moderate risk manager with default configuration."""
        super().__init__()
        self.strategy_risk = {}
        self.default_config = {
            'max_loss_streak': 7,  # More lenient than conservative
            'max_drawdown_pct': 0.3,  # 30% drawdown
            'max_exposure': 0.2,  # 20% of bankroll
            'dynamic_stake_pct': 0.05,  # 5% of bankroll
            'min_bankroll': 300  # Lower threshold
        }
    
    def configure_strategy(self, strategy: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Configure risk parameters for a strategy."""
        self.strategy_risk[strategy] = {
            'loss_streak': 0,
            'peak_bankroll': 0.0,
            'current_drawdown': 0.0,
            'bankroll': 0.0,
            'config': config if config else self.default_config.copy()
        }
    
    def update_bankroll(self, strategy: str, bankroll: float) -> None:
        """Update the strategy's bankroll and drawdown."""
        if strategy not in self.strategy_risk:
            self.configure_strategy(strategy)
        
        risk = self.strategy_risk[strategy]
        risk['bankroll'] = bankroll
        risk['peak_bankroll'] = max(risk['peak_bankroll'], bankroll)
        drawdown = risk['peak_bankroll'] - bankroll
        risk['current_drawdown'] = drawdown / risk['peak_bankroll'] if risk['peak_bankroll'] > 0 else 0
    
    def update_loss_streak(self, strategy: str, won_last_bet: bool) -> None:
        """Update the strategy's loss streak."""
        if strategy not in self.strategy_risk:
            self.configure_strategy(strategy)
        
        risk = self.strategy_risk[strategy]
        risk['loss_streak'] = 0 if won_last_bet else risk['loss_streak'] + 1
    
    def should_stop_trading(self, strategy: str) -> bool:
        """Check if the strategy should stop trading."""
        risk = self.strategy_risk.get(strategy, {})
        cfg = risk.get('config', self.default_config)
        return (
            risk.get('loss_streak', 0) >= cfg['max_loss_streak'] or
            risk.get('current_drawdown', 0) >= cfg['max_drawdown_pct'] or
            risk.get('bankroll', 0) <= cfg['min_bankroll']
        )
    
    def calculate_dynamic_stake(self, strategy: str) -> float:
        """Calculate the dynamic stake for a strategy."""
        risk = self.strategy_risk.get(strategy, {})
        bankroll = risk.get('bankroll', 0.0)
        stake = bankroll * risk.get('config', self.default_config)['dynamic_stake_pct']
        return max(1.0, round(stake, 2))
    
    def get_status(self, strategy: str) -> Dict[str, Any]:
        """Retrieve the risk status for a strategy."""
        risk = self.strategy_risk.get(strategy, {})
        return {
            'Loss Streak': risk.get('loss_streak', 0),
            'Drawdown %': round(risk.get('current_drawdown', 0) * 100, 2),
            'Bankroll': risk.get('bankroll', 0.0),
            'Should Stop': self.should_stop_trading(strategy)
        }
    
    def all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve risk statuses for all strategies."""
        return {strategy: self.get_status(strategy) for strategy in self.strategy_risk}