from portfolio_manager_interface import PortfolioManagerInterface
from typing import Dict, Any

class AggressivePortfolioManager(PortfolioManagerInterface):
    """Portfolio manager with aggressive stake allocation and lenient deactivation."""
    
    def __init__(self, initial_bankroll: float = 1000.0, max_stake_fraction: float = 0.2):
        """Initialize the aggressive portfolio manager.
        
        Args:
            initial_bankroll (float): Starting capital (default: 1000.0).
            max_stake_fraction (float): Maximum bankroll fraction per bet (default: 0.2).
        """
        super().__init__()
        self.initial_bankroll = initial_bankroll
        self.max_stake_fraction = max_stake_fraction
        self.portfolios = {}
        self.overall_metrics = {
            'bankroll': initial_bankroll,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'trade_history': []
        }
    
    def add_strategy(self, strategy_name: str) -> None:
        """Add a new strategy with aggressive initial metrics."""
        self.portfolios[strategy_name] = {
            'bankroll': self.initial_bankroll,
            'peak': self.initial_bankroll,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'loss_streak': 0,
            'win_streak': 0,
            'trade_history': [],
            'equity_curve': [],
            'active': True
        }
    
    def update_strategy(self, strategy_name: str, pnl: float) -> None:
        """Update the strategy with aggressive risk tolerance."""
        if strategy_name not in self.portfolios:
            return
        
        p = self.portfolios[strategy_name]
        if not p['active']:
            return
        
        # Update strategy metrics
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
        
        # Lenient deactivation: 7 loss streak or 50% drawdown
        if p['loss_streak'] >= 7 or (drawdown / p['peak'] > 0.5 if p['peak'] > 0 else False):
            p['active'] = False
        
        self._update_overall(pnl)
    
    def _update_overall(self, pnl: float) -> None:
        """Update overall portfolio metrics."""
        self.overall_metrics['bankroll'] += pnl
        self.overall_metrics['total_profit'] += pnl if pnl > 0 else 0
        self.overall_metrics['total_loss'] += pnl if pnl < 0 else 0
        peak = max(self.initial_bankroll, self.overall_metrics['bankroll'])
        drawdown = peak - self.overall_metrics['bankroll']
        self.overall_metrics['max_drawdown'] = max(self.overall_metrics['max_drawdown'], drawdown)
        self.overall_metrics['trade_history'].append(pnl)
    
    def get_strategy_metrics(self, strategy_name: str) -> Dict[str, Any]:
        """Retrieve aggressive strategy metrics."""
        if strategy_name not in self.portfolios:
            return {}
        
        p = self.portfolios[strategy_name]
        roi = (p['bankroll'] - self.initial_bankroll) / self.initial_bankroll * 100 if self.initial_bankroll > 0 else 0
        trades = p['trade_history']
        win_rate = sum(1 for t in trades if t > 0) / len(trades) * 100 if trades else 0
        avg_gain = sum(t for t in trades if t > 0) / max(1, sum(1 for t in trades if t > 0)) if any(t > 0 for t in trades) else 0
        avg_loss = sum(t for t in trades if t < 0) / max(1, sum(1 for t in trades if t < 0)) if any(t < 0 for t in trades) else 0
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
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """Retrieve overall aggressive portfolio metrics."""
        roi = (self.overall_metrics['bankroll'] - self.initial_bankroll) / self.initial_bankroll * 100 if self.initial_bankroll > 0 else 0
        trades = self.overall_metrics['trade_history']
        win_rate = sum(1 for t in trades if t > 0) / len(trades) * 100 if trades else 0
        avg_gain = sum(t for t in trades if t > 0) / max(1, sum(1 for t in trades if t > 0)) if any(t > 0 for t in trades) else 0
        avg_loss = sum(t for t in trades if t < 0) / max(1, sum(1 for t in trades if t < 0)) if any(t < 0 for t in trades) else 0
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