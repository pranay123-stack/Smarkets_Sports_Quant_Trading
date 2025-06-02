import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# === Load and preprocess data ===
file_path = "Smarkets_Sports_Quant_Trading\Football_outcome_predictor_bets_strategy\Data/E0.csv"  # Use relative path
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Ensure 'E0.csv' is in the 'Data' folder.")
    exit(1)

# Keep necessary columns
df = df[['FTR', 'B365H', 'B365D', 'B365A']].dropna()

# Map match result to numerical values
df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

# Compute implied probabilities
df['odds_home_prob'] = 1 / df['B365H']
df['odds_draw_prob'] = 1 / df['B365D']
df['odds_away_prob'] = 1 / df['B365A']

# Normalize probabilities (to remove overround)
df['norm'] = df['odds_home_prob'] + df['odds_draw_prob'] + df['odds_away_prob']
df['book_home_prob'] = df['odds_home_prob'] / df['norm']
df['book_draw_prob'] = df['odds_draw_prob'] / df['norm']
df['book_away_prob'] = df['odds_away_prob'] / df['norm']

# Define features and target
features = ['book_home_prob', 'book_draw_prob', 'book_away_prob']
X = df[features]
y = df['FTR']
odds_df = df[['B365H', 'B365D', 'B365A']]  # For odds reference later

# === Train-test split ===
X_train, X_test, y_train, y_test, odds_train, odds_test = train_test_split(
    X, y, odds_df, test_size=0.2, random_state=42
)

# === Load pre-trained model ===
try:
    with open("Smarkets_Sports_Quant_Trading\Football_outcome_predictor_bets_strategy\Trained_ML_Models/model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Pre-trained model loaded from models/model.pkl")
except FileNotFoundError:
    print("Error: Pre-trained model (models/model.pkl) not found. Please ensure the model file exists.")
    exit(1)

# === Predict probabilities ===
probs = model.predict_proba(X_test)

# === Simulate betting strategy ===
stake = 100
bankroll = 10000
min_edge = 0.05  # minimum model edge vs bookmaker implied prob

total_profit = 0
bet_log = []
labels = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}

for i in range(len(X_test)):
    row = X_test.iloc[i]
    odds = odds_test.iloc[i]
    pred_prob = probs[i]
    actual = y_test.iloc[i]

    # Implied bookmaker probabilities
    implied_probs = [
        1 / odds['B365H'],
        1 / odds['B365D'],
        1 / odds['B365A']
    ]
    norm = sum(implied_probs)
    implied_probs = [x / norm for x in implied_probs]

    # Calculate edges
    best_edge = -1
    selected_outcome = None
    for outcome in range(3):
        edge = pred_prob[outcome] - implied_probs[outcome]
        if edge > min_edge and edge > best_edge:
            best_edge = edge
            selected_outcome = outcome

    if selected_outcome is not None:
        price = odds.iloc[selected_outcome]
        if selected_outcome == actual:
            profit = (price - 1) * stake
        else:
            profit = -stake
        total_profit += profit
        bankroll += profit

        bet_log.append({
            'Match': i + 1,
            'Model Pick': labels[selected_outcome],
            'Actual Result': labels[actual],
            'Edge': round(best_edge, 3),
            'Odds': round(price, 2),
            'Profit': round(profit, 2),
            'Bankroll': round(bankroll, 2)
        })

# === Convert to DataFrame and save ===
bet_df = pd.DataFrame(bet_log)

# Print summary
print("\n=== Betting Simulation Log ===\n")
print(bet_df)

roi = (total_profit / (len(bet_df) * stake)) * 100 if len(bet_df) > 0 else 0
print(f"\n✅ Total Profit: ₹{round(total_profit, 2)}")
print(f"💼 Final Bankroll: ₹{round(bankroll, 2)}")
print(f"📊 ROI: {roi:.2f}%")
print(f"🎯 Bets Placed: {len(bet_df)}")

# Save results with relative path
output_path = "Smarkets_Sports_Quant_Trading\Football_outcome_predictor_bets_strategy/betting_simulation_log.csv"
bet_df.to_csv(output_path, index=False)
print(f"\n📁 Log saved to: {output_path}")