from position_manager_interface import PositionManagerInterface
from typing import Dict, List, Any, Optional
import numpy as np

class AdvancedPositionManager(PositionManagerInterface):
    """Advanced position manager with position limits and correlation tracking."""
    
    def __init__(self, max_positions: int = 10, max_exposure: float = 0.2):
        """Initialize the advanced position manager.
        
        Args:
            max_positions (int): Max open positions per strategy (default: 10).
            max_exposure (float): Max exposure as fraction of bankroll (default: 0.2).
        """
        super().__init__()
        self.positions = {}
        self.max_positions = max_positions
        self.max_exposure = max_exposure
    
    def open_position(self, strategy: str, match_id: str, side: str, stake: float, odds: float) -> None:
        """Open a new betting position with limits."""
        if strategy not in self.positions:
            self.positions[strategy] = []
        
        open_positions = [p for p in self.positions[strategy] if p['Status'] == 'open']
        if len(open_positions) >= self.max_positions:
            return  # Ignore if max positions reached
        
        bankroll = open_positions[0].get('bankroll', 1000.0) if open_positions else 1000.0
        total_exposure = sum(p['Stake'] for p in open_positions) + stake
        if total_exposure / bankroll > self.max_exposure:
            stake = max(0, bankroll * self.max_exposure - sum(p['Stake'] for p in open_positions))
        
        if stake > 0:
            self.positions[strategy].append({
                'Match ID': match_id,
                'Side': side,
                'Stake': stake,
                'Odds': odds,
                'Status': 'open',
                'Bankroll': bankroll
            })
    
    def close_position(self, strategy: str, match_id: str, result_side: str) -> float:
        """Close a position with P&L and correlation adjustment."""
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
        """Evaluate risk with correlation tracking."""
        if strategy not in self.positions:
            return 0.0, 0.0
        
        open_positions = [p for p in self.positions[strategy] if p['Status'] == 'open']
        total_exposure = sum(p['Stake'] for p in open_positions)
        avg_odds = sum(p['Odds'] for p in open_positions) / len(open_positions) if open_positions else 0.0
        
        # Simplified correlation: count same-side positions
        sides = [p['Side'] for p in open_positions]
        correlation_risk = sum(sides.count(s) - 1 for s in set(sides)) / len(open_positions) if open_positions else 0.0
        total_exposure *= (1 + correlation_risk)  # Adjust for correlation
        
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
        """Retrieve positions."""
        if strategy:
            return {strategy: self.positions.get(strategy, [])}
        return self.positions