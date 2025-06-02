import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
file_path = "Smarkets_Sports_Quant_Trading\Arbitrage_detector_strategy\Data\E0.csv"
raw_df = pd.read_csv(file_path)

# Keep necessary columns
df = raw_df[['FTR', 'B365H', 'B365D', 'B365A']].dropna()
df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

# Calculate implied probabilities
for side in ['H', 'D', 'A']:
    df[f'prob_{side}'] = 1 / df[f'B365{side}']
df['total_prob'] = df[['prob_H', 'prob_D', 'prob_A']].sum(axis=1)
for side in ['H', 'D', 'A']:
    df[f'norm_prob_{side}'] = df[f'prob_{side}'] / df['total_prob']

# Feature engineering
X = df[['norm_prob_H', 'norm_prob_D', 'norm_prob_A']]
y = df['FTR']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Arbitrage detection logic
threshold = 0.05  # Minimum edge to consider a value bet
stake = 100
bankroll = 10000

log = []
probs = model.predict_proba(X_test)

for i in range(len(X_test)):
    row = X_test.iloc[i]
    odds_row = raw_df.loc[X_test.index[i], ['B365H', 'B365D', 'B365A']]
    true_result = y_test.iloc[i]

    implied = [1 / odds_row[f'B365H'], 1 / odds_row[f'B365D'], 1 / odds_row[f'B365A']]
    implied = [x / sum(implied) for x in implied]
    predicted = probs[i]

    edges = [predicted[j] - implied[j] for j in range(3)]
    max_edge = max(edges)
    pick = edges.index(max_edge)

    if max_edge > threshold:
        odds = odds_row[pick]
        win = pick == true_result
        profit = (odds - 1) * stake if win else -stake
        bankroll += profit

        log.append({
            'Match': i + 1,
            'Model Pick': ['Home', 'Draw', 'Away'][pick],
            'Actual Result': ['Home', 'Draw', 'Away'][true_result],
            'Edge': round(max_edge, 3),
            'Odds': round(odds, 2),
            'Profit': round(profit, 2),
            'New Bankroll': round(bankroll, 2)
        })

# Create log DataFrame and display
log_df = pd.DataFrame(log)
print("\nArbitrage Bets Executed:")
print(log_df)
roi = ((bankroll - 10000) / (stake * len(log_df))) * 100 if len(log_df) else 0
print(f"\n✅ Total Profit: ₹{round(bankroll - 10000, 2)} | ROI: {roi:.2f}% | Bets Placed: {len(log_df)}")

# Save for Streamlit reuse
log_df.to_csv("Smarkets_Sports_Quant_Trading\Arbitrage_detector_strategy/arbitrage_bet_log.csv", index=False)
