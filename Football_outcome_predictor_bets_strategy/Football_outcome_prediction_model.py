import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load data
file_path = "Smarkets_Sports_Quant_Trading\Football_outcome_predictor_bets_strategy\Data\E0.csv"
df = pd.read_csv(file_path)

# Keep only required columns
df = df[['FTR', 'B365H', 'B365D', 'B365A']].dropna()

# Map result to numeric
df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

# Convert odds to implied probabilities
df['odds_home_prob'] = 1 / df['B365H']
df['odds_draw_prob'] = 1 / df['B365D']
df['odds_away_prob'] = 1 / df['B365A']

# Normalize to remove overround
df['norm'] = df['odds_home_prob'] + df['odds_draw_prob'] + df['odds_away_prob']
df['book_home_prob'] = df['odds_home_prob'] / df['norm']
df['book_draw_prob'] = df['odds_draw_prob'] / df['norm']
df['book_away_prob'] = df['odds_away_prob'] / df['norm']

# Define features and target
features = ['book_home_prob', 'book_draw_prob', 'book_away_prob']
X = df[features]
y = df['FTR']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)

# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Simulate live predictions on test set
labels = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
print("\n=== Simulated Live Predictions ===\n")

correct = 0
for i in range(len(X_test)):
    input_row = X_test.iloc[[i]]
    actual = y_test.iloc[i]
    predicted = model.predict(input_row)[0]
    prob = model.predict_proba(input_row)[0]

    print(f"Match {i+1}")
    print("→ Actual:    ", labels[actual])
    print("→ Predicted: ", labels[predicted])
    print(f"→ Probabilities - Home: {prob[0]:.2f}, Draw: {prob[1]:.2f}, Away: {prob[2]:.2f}")
    print("---")

    if predicted == actual:
        correct += 1

# Final accuracy
accuracy = correct / len(X_test)
print(f"\n✅ Simulated Prediction Accuracy: {accuracy*100:.2f}%")
