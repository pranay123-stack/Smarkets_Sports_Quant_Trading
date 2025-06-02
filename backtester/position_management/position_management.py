class PositionManager:
    def __init__(self):
        self.positions = {}

    def open_position(self, strategy, match_id, side, stake, odds):
        if strategy not in self.positions:
            self.positions[strategy] = []

        self.positions[strategy].append({
            'Match ID': match_id,
            'Side': side,
            'Stake': stake,
            'Odds': odds,
            'Status': 'open'
        })

    def close_position(self, strategy, match_id, result_side):
        if strategy not in self.positions:
            return 0

        for pos in self.positions[strategy]:
            if pos['Match ID'] == match_id and pos['Status'] == 'open':
                pos['Status'] = 'closed'
                pos['Result'] = result_side
                if result_side == pos['Side']:
                    pos['P&L'] = pos['Stake'] * (pos['Odds'] - 1)
                else:
                    pos['P&L'] = -pos['Stake']
                return pos['P&L']
        return 0

    def evaluate_strategy_risk(self, strategy):
        if strategy not in self.positions:
            return 0, 0

        open_positions = [p for p in self.positions[strategy] if p['Status'] == 'open']
        total_exposure = sum([p['Stake'] for p in open_positions])
        avg_odds = sum([p['Odds'] for p in open_positions]) / len(open_positions) if open_positions else 0

        return total_exposure, avg_odds

    def force_close_all(self, strategy):
        if strategy not in self.positions:
            return 0
        for p in self.positions[strategy]:
            if p['Status'] == 'open':
                p['Status'] = 'forced_close'
                p['P&L'] = -p['Stake']
        return 1

    def get_positions(self, strategy=None):
        if strategy:
            return self.positions.get(strategy, [])
        return self.positions
