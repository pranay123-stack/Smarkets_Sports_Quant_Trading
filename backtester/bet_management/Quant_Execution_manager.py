import random
import pandas as pd
import numpy as np
from execution_manager_interface import ExecutionManagerInterface
from typing import Dict, List, Any, Optional, Callable

class QuantExecutionManager(ExecutionManagerInterface):
    """Execution manager with TWAP-based bet execution for sports betting."""
    
    def __init__(self):
        """Initialize the execution manager."""
        super().__init__()
        self.executed_bets = []
        self.slippage_model = lambda odds: odds - random.uniform(0.01, 0.05)
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
        """Execute a bet, optionally using TWAP for large stakes."""
        if stake > self.max_stake / 2:  # Use TWAP for large bets
            return self.execute_twap(strategy, match_id, side, requested_odds, stake)
        
        if random.random() < self.rejection_rate:
            return {'status': 'rejected', 'reason': 'Bookmaker rejection'}
        
        if stake < self.min_stake:
            return {'status': 'rejected', 'reason': 'Stake below minimum'}
        if stake > self.max_stake:
            return {'status': 'rejected', 'reason': 'Stake above maximum'}
        
        final_odds = max(1.01, self.slippage_model(requested_odds))
        
        self.executed_bets.append({
            'Strategy': strategy,
            'Match ID': match_id,
            'Side': side,
            'Requested Odds': requested_odds,
            'Executed Odds': final_odds,
            'Stake': stake,
            'Timestamp': pd.Timestamp.now()
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
        """Generate a summary with TWAP metrics."""
        if not self.executed_bets:
            return {'Total Executed': 0, 'Total Volume': 0.0, 'TWAP Usage': 0}
        
        total_volume = sum(b['Stake'] for b in self.executed_bets)
        twap_bets = sum(1 for b in self.executed_bets if 'TWAP' in b.get('Execution Type', ''))
        
        return {
            'Total Executed': len(self.executed_bets),
            'Total Volume': total_volume,
            'TWAP Usage': twap_bets
        }
    
    def execute_twap(self, strategy: str, match_id: str, side: str, requested_odds: float, stake: float) -> Dict[str, Any]:
        """Execute a bet using TWAP (non-interface method).
        
        Args:
            strategy (str): Name of the strategy.
            match_id (str): Unique identifier for the match.
            side (str): Betting side.
            requested_odds (float): Requested odds.
            stake (float): Total stake amount.
            
        Returns:
            Dict[str, Any]: Execution result.
        """
        # Split stake into 5 smaller bets over time (simulated)
        num_splits = 5
        split_stake = stake / num_splits
        
        if split_stake < self.min_stake:
            return {'status': 'rejected', 'reason': 'Split stake below minimum'}
        
        executed_odds = []
        total_stake = 0.0
        
        for i in range(num_splits):
            if random.random() < self.rejection_rate:
                continue  # Simulate partial rejection
            
            # Simulate odds fluctuation over time
            time_adjustment = random.uniform(-0.1, 0.1) * requested_odds
            current_odds = max(1.01, requested_odds + time_adjustment)
            final_odds = max(1.01, self.slippage_model(current_odds))
            
            self.executed_bets.append({
                'Strategy': strategy,
                'Match ID': f"{match_id}_twap_{i}",
                'Side': side,
                'Requested Odds': current_odds,
                'Executed Odds': final_odds,
                'Stake': split_stake,
                'Execution Type': 'TWAP',
                'Timestamp': pd.Timestamp.now()
            })
            
            executed_odds.append(final_odds)
            total_stake += split_stake
        
        if not executed_odds:
            return {'status': 'rejected', 'reason': 'All TWAP splits rejected'}
        
        avg_odds = np.mean(executed_odds)
        return {
            'status': 'executed',
            'executed_odds': avg_odds,
            'stake': total_stake
        }