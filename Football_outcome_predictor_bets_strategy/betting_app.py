import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="⚽ Betting Edge Predictor", layout="wide")
st.title("⚽ Football Betting Strategy Simulator")

st.markdown("""
Use Football (Soccer) data from Football-Data.co.uk
Upload a CSV containing pre-match bookmaker odds (`B365H`, `B365D`, `B365A`).  
This app will:
- Predict outcomes using a trained ML model
- Detect profitable edges
- Simulate bets and bankroll over time
- Plot ROI vs Edge Threshold and Bankroll Curve
""")

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload Match Odds CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if not all(col in df.columns for col in ['B365H', 'B365D', 'B365A']):
        st.error("CSV must contain columns: B365H, B365D, B365A")
        st.stop()

    # Compute implied & normalized probs
    df['odds_home_prob'] = 1 / df['B365H']
    df['odds_draw_prob'] = 1 / df['B365D']
    df['odds_away_prob'] = 1 / df['B365A']
    norm = df[['odds_home_prob', 'odds_draw_prob', 'odds_away_prob']].sum(axis=1)
    df['book_home_prob'] = df['odds_home_prob'] / norm
    df['book_draw_prob'] = df['odds_draw_prob'] / norm
    df['book_away_prob'] = df['odds_away_prob'] / norm

    # --- Load and train model from base data ---
    base_df = pd.read_csv("Smarkets_Sports_Quant_Trading\Football_outcome_predictor_bets_strategy\Data\E0.csv").dropna(subset=['B365H', 'B365D', 'B365A'])
    base_df['FTR'] = base_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    base_df['odds_home_prob'] = 1 / base_df['B365H']
    base_df['odds_draw_prob'] = 1 / base_df['B365D']
    base_df['odds_away_prob'] = 1 / base_df['B365A']
    base_df['norm'] = base_df[['odds_home_prob', 'odds_draw_prob', 'odds_away_prob']].sum(axis=1)
    base_df['book_home_prob'] = base_df['odds_home_prob'] / base_df['norm']
    base_df['book_draw_prob'] = base_df['odds_draw_prob'] / base_df['norm']
    base_df['book_away_prob'] = base_df['odds_away_prob'] / base_df['norm']

    features = ['book_home_prob', 'book_draw_prob', 'book_away_prob']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(base_df[features], base_df['FTR'])

    # Predictions
    probs = model.predict_proba(df[features])
    preds = model.predict(df[features])
    labels = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}

    # --- User Config ---
    min_edge = st.slider("Minimum Edge % to Place Bet", 0.0, 20.0, 5.0) / 100
    stake = st.number_input("Stake per Bet", value=100)
    initial_bankroll = st.number_input("Starting Bankroll", value=10000)

    # --- Simulate Bets ---
    bankroll = initial_bankroll
    total_profit = 0
    bet_log = []

    for i in range(len(df)):
        implied_probs = [1 / df.loc[i, 'B365H'], 1 / df.loc[i, 'B365D'], 1 / df.loc[i, 'B365A']]
        implied_probs = [x / sum(implied_probs) for x in implied_probs]
        model_probs = probs[i]
        edge = [model_probs[j] - implied_probs[j] for j in range(3)]
        best_edge = max(edge)
        selected = edge.index(best_edge)
        odds = df.iloc[i, [df.columns.get_loc('B365H') + selected]].values[0]

        if best_edge > min_edge:
            profit = (odds - 1) * stake if preds[i] == selected else -stake
            bankroll += profit
            total_profit += profit
            bet_log.append({
                'Match': i + 1,
                'Model Pick': labels[selected],
                'Edge': round(best_edge, 3),
                'Odds': round(odds, 2),
                'Profit': round(profit, 2),
                'Bankroll': round(bankroll, 2)
            })

    bet_df = pd.DataFrame(bet_log)

    # --- Results ---
    st.subheader("📊 Betting Simulation Results")
    if not bet_df.empty:
        st.dataframe(bet_df)
        roi = (total_profit / (len(bet_df) * stake)) * 100 if len(bet_df) > 0 else 0
        st.success(f"✅ Total Profit: ₹{round(total_profit, 2)} | Final Bankroll: ₹{round(bankroll, 2)} | ROI: {roi:.2f}%")

        # ROI vs Edge Threshold
        st.subheader("📈 ROI vs Edge Threshold")
        edge_range = np.linspace(0.01, 0.2, 20)
        roi_list = []
        for threshold in edge_range:
            bkr = initial_bankroll
            profit = 0
            for i in range(len(df)):
                implied = [1 / df.loc[i, 'B365H'], 1 / df.loc[i, 'B365D'], 1 / df.loc[i, 'B365A']]
                implied = [x / sum(implied) for x in implied]
                mp = probs[i]
                e = [mp[j] - implied[j] for j in range(3)]
                max_e = max(e)
                sel = e.index(max_e)
                odd = df.iloc[i, [df.columns.get_loc('B365H') + sel]].values[0]
                if max_e > threshold:
                    p = (odd - 1) * stake if preds[i] == sel else -stake
                    profit += p
                    bkr += p
            r = (profit / (stake * len(df))) * 100 if len(df) > 0 else 0
            roi_list.append(r)

        fig1, ax1 = plt.subplots()
        ax1.plot(edge_range * 100, roi_list, marker='o')
        ax1.set_xlabel("Edge Threshold (%)")
        ax1.set_ylabel("ROI (%)")
        ax1.set_title("ROI vs Edge Threshold")
        st.pyplot(fig1)

        # Bankroll Over Time
        st.subheader("📉 Bankroll Over Time")
        fig2, ax2 = plt.subplots()
        ax2.plot(bet_df['Match'], bet_df['Bankroll'], color='green')
        ax2.set_xlabel("Match")
        ax2.set_ylabel("Bankroll")
        ax2.set_title("Bankroll Over Time")
        st.pyplot(fig2)
    else:
        st.warning("No bets placed. Try lowering your edge threshold.")
