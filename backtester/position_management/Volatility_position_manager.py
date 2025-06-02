from position_manager_interface import PositionManagerInterface
from typing import Dict, List, Any, Optional
import numpy as np

class VolatilityPositionManager(PositionManagerInterface):
    """Position manager with volatility-based stake adjustment for sports betting."""
    
    def __init__(self):
        """Initialize the position manager."""
        super().__init__()
        self.positions = {}
    
    def open_position(self, strategy: str, match_id: str, side: str, stake: float, odds: float) -> None:
        """Open a new betting position with volatility adjustment."""
        if strategy not in self.positions:
            self.positions[strategy] = []
        
        # Adjust stake based on odds volatility
        adjusted_stake = self.adjust_for_volatility(row=None, stake=stake, odds=odds)
        
        if adjusted_stake > 0:
            self.positions[strategy].append({
                'Match ID': match_id,
                'Side': side,
                'Stake': adjusted_stake,
                'Odds': odds,
                'Status': 'open'
            })
    
    def close_position(self, strategy: str, match_id: str, result_side: str) -> float:
        """Close an open position and calculate P&L."""
        if strategy not in self.positions:
            return 0.0
        
        for pos in self.positions[strategy]:
            if pos['Match ID'] == match_id and pos['Status'] == 'open':
                pos['Status'] = 'closed'
                pos['Result'] = result_side
                pos['P&L'] = pos['Stake'] * (pos['Odds'] - 1) if result_side == pos['Side'] else -pos['Stake']
                return pos['P&L']
        return 0.0
    
    def evaluate_strategy_risk(self, strategy: str) -> tuple[float, float]:
        """Evaluate risk exposure and average odds."""
        if strategy not in self.positions:
            return 0.0, 0.0
        
        open_positions = [p for p in self.positions[strategy] if p['Status'] == 'open']
        total_exposure = sum(p['Stake'] for p in open_positions)
        avg_odds = sum(p['Odds'] for p in open_positions) / len(open_positions) if open_positions else 0.0
        return total_exposure, avg_odds
    
    def force_close_all(self, strategy: str) -> int:
        """Force-close all open positions."""
        if strategy not in self.positions:
            return 0
        
        for pos in self.positions[strategy]:
            if pos['Status'] == 'open':
                pos['Status'] = 'forced_close'
                pos['P&L'] = -pos['Stake']
        return 1 if any(p['Status'] == 'forced_close' for p in self.positions[strategy]) else 0
    
    def get_positions(self, strategy: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve positions for a strategy or all strategies."""
        if strategy:
            return {strategy: self.positions.get(strategy, [])}
        return self.positions
    
    def adjust_for_volatility(self, row: Optional[pd.Series], stake: float, odds: float) -> float:
        """Adjust stake based on odds volatility (non-interface method).
        
        Args:
            row (Optional[pd.Series]): Match data (e.g., odds history).
            stake (float): Proposed stake.
            odds (float): Current odds.
            
        Returns:
            float: Adjusted stake.
        """
        # Simplified volatility: assume odds variance from historical data
        odds_history = row.get('odds_history', [odds]) if row else [odds]
        volatility = np.std(odds_history) if len(odds_history) > 1 else 0.0
        
        # Reduce stake if volatility is high
        if volatility > 0.5:  # Arbitrary threshold
            return stake * 0.5  # Halve stake for high volatility
        return stake