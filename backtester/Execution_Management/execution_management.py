import random

class ExecutionManager:
    def __init__(self):
        self.executed_bets = []
        self.slippage_model = lambda odds: odds - random.uniform(0.01, 0.05)  # Default slippage
        self.rejection_rate = 0.02  # 2% chance of bet rejection
        self.min_stake = 10
        self.max_stake = 1000

    def configure(self, slippage_fn=None, rejection_rate=None, min_stake=None, max_stake=None):
        if slippage_fn:
            self.slippage_model = slippage_fn
        if rejection_rate is not None:
            self.rejection_rate = rejection_rate
        if min_stake:
            self.min_stake = min_stake
        if max_stake:
            self.max_stake = max_stake

    def execute_bet(self, strategy, match_id, side, requested_odds, stake):
        # Reject based on rejection probability
        if random.random() < self.rejection_rate:
            return {'status': 'rejected', 'reason': 'Bookmaker rejection'}

        # Check stake bounds
        if stake < self.min_stake:
            return {'status': 'rejected', 'reason': 'Stake below minimum'}
        if stake > self.max_stake:
            return {'status': 'rejected', 'reason': 'Stake above maximum'}

        # Apply slippage
        final_odds = max(1.01, self.slippage_model(requested_odds))

        # Assume market fills order
        self.executed_bets.append({
            'Strategy': strategy,
            'Match ID': match_id,
            'Side': side,
            'Requested Odds': requested_odds,
            'Executed Odds': final_odds,
            'Stake': stake
        })

        return {
            'status': 'executed',
            'executed_odds': final_odds,
            'stake': stake
        }

    def get_execution_log(self):
        return self.executed_bets

    def summary(self):
        return {
            'Total Executed': len(self.executed_bets),
            'Total Volume': sum([b['Stake'] for b in self.executed_bets])
        }
