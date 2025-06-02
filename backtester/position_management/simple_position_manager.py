from position_manager_interface import PositionManagerInterface
from typing import Dict, List, Any, Optional

class SimplePositionManager(PositionManagerInterface):
    """Simple position manager with basic tracking for sports betting."""
    
    def __init__(self):
        """Initialize the position manager."""
        super().__init__()
        self.positions = {}
    
    def open_position(self, strategy: str, match_id: str, side: str, stake: float, odds: float) -> None:
        """Open a new betting position with minimal data."""
        if strategy not in self.positions:
            self.positions[strategy] = []
        
        self.positions[strategy].append({
            'Match ID': match_id,
            'Stake': stake,
            'Status': 'open'
        })
    
    def close_position(self, strategy: str, match_id: str, result_side: str) -> float:
        """Close a position with fixed P&L (win: stake, loss: -stake)."""
        if strategy not in self.positions:
            return 0.0
        
        for pos in self.positions[strategy]:
            if pos['Match ID'] == match_id and pos['Status'] == 'open':
                pos['Status'] = 'closed'
                pos['P&L'] = pos['Stake'] if result_side == 'home' else -pos['Stake']  # Simplified
                return pos['P&L']
        return 0.0
    
    def evaluate_strategy_risk(self, strategy: str) -> tuple[float, float]:
        """Evaluate risk with total stake only."""
        if strategy not in self.positions:
            return 0.0, 0.0
        
        open_positions = [p for p in self.positions[strategy] if p['Status'] == 'open']
        total_exposure = sum(p['Stake'] for p in open_positions)
        return total_exposure, 0.0  # No odds tracking
    
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