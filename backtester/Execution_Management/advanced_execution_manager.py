import random
from execution_manager_interface import ExecutionManagerInterface
from typing import Dict, List, Any, Optional, Callable
import numpy as np

class AdvancedExecutionManager(ExecutionManagerInterface):
    """Advanced execution manager with dynamic slippage and quant model integration."""
    
    def __init__(self):
        """Initialize the execution manager."""
        super().__init__()
        self.executed_bets = []
        self.slippage_model = lambda odds: odds - random.uniform(0.01, 0.1) * odds / 2  # Dynamic slippage
        self.rejection_rate = 0.02
        self.min_stake = 10.0
        self.max_stake = 1000.0
    
    def configure(self, slippage_fn: Optional[Callable[[float], float]] = None, 
                  rejection_rate: Optional[float] = None, 
                  min_stake: Optional[float] = None, 
                  max_stake: Optional[float] = None) -> None:
        """Configure execution parameters."""
        if slippage_fn is not None:
            self.slippage_model = slippage_fn
        if rejection_rate is not None:
            self.rejection_rate = rejection_rate
        if min_stake is not None:
            self.min_stake = min_stake
        if max_stake is not None:
            self.max_stake = max_stake
    
    def execute_bet(self, strategy: str, match_id: str, side: str, requested_odds: float, stake: float) -> Dict[str, Any]:
        """Execute a bet with dynamic slippage and rejection."""
        # Dynamic rejection based on stake size
        adjusted_rejection_rate = self.rejection_rate * (1 + stake / self.max_stake)
        if random.random() < adjusted_rejection_rate:
            return {'status': 'rejected', 'reason': 'Bookmaker rejection'}
        
        if stake < self.min_stake:
            return {'status': 'rejected', 'reason': 'Stake below minimum'}
        if stake > self.max_stake:
            return {'status': 'rejected', 'reason': 'Stake above maximum'}
        
        # Apply dynamic slippage
        final_odds = max(1.01, self.slippage_model(requested_odds))
        
        self.executed_bets.append({
            'Strategy': strategy,
            'Match ID': match_id,
            'Side': side,
            'Requested Odds': requested_odds,
            'Executed Odds': final_odds,
            'Stake': stake,
            'Timestamp': pd.Timestamp.now()  # Non-interface addition
        })
        
        return {
            'status': 'executed',
            'executed_odds': final_odds,
            'stake': stake
        }
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Retrieve the log of executed bets."""
        return self.executed_bets
    
    def summary(self) -> Dict[str, Any]:
        """Generate an advanced summary with quant metrics."""
        if not self.executed_bets:
            return {'Total Executed': 0, 'Total Volume': 0.0, 'Avg Slippage': 0.0}
        
        total_volume = sum(b['Stake'] for b in self.executed_bets)
        slippage = [b['Requested Odds'] - b['Executed Odds'] for b in self.executed_bets]
        avg_slippage = np.mean(slippage) if slippage else 0.0
        
        return {
            'Total Executed': len(self.executed_bets),
            'Total Volume': total_volume,
            'Avg Slippage': avg_slippage
        }
    
    def calculate_dynamic_slippage(self, odds: float, stake: float) -> float:
        """Calculate slippage based on stake size (non-interface method).
        
        Args:
            odds (float): Requested odds.
            stake (float): Stake amount.
            
        Returns:
            float: Adjusted odds.
        """
        # Increase slippage for larger stakes
        slippage_factor = random.uniform(0.01, 0.1) * (1 + stake / self.max_stake)
        return odds - slippage_factor