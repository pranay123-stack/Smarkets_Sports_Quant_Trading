import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="⚽ Betting Edge Predictor", layout="wide")
st.title("⚽ Football Betting Strategy Simulator")

st.markdown("""
Use Football (Soccer) data from [Football-Data.co.uk](https://www.football-data.co.uk/).
This app uses a pre-trained RandomForest model to:
- **Predict outcomes** and simulate bets on uploaded odds data (`B365H`, `B365D`, `B365A`).
- **Simulate a betting strategy** on historical data with actual results (`FTR`, `B365H`, `B365D`, `B365A`).
""")

# --- Load pre-trained model ---
try:
    with open("Smarkets_Sports_Quant_Trading\Football_outcome_predictor_bets_strategy\Trained_ML_Models/model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Pre-trained model (models/model.pkl) not found. Please ensure the model file is in the repository.")
    st.stop()

# --- Prediction-Based Betting Section ---
st.subheader("🔍 Predict Outcomes and Simulate Bets")
st.markdown("Upload a CSV with columns `B365H`, `B365D`, `B365A` to predict match outcomes and simulate bets based on edge detection.")

uploaded_file_pred = st.file_uploader("Upload Match Odds CSV for Prediction", type=["csv"], key="pred_uploader")

if uploaded_file_pred:
    df = pd.read_csv(uploaded_file_pred)

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

    # Predictions
    features = ['book_home_prob', 'book_draw_prob', 'book_away_prob']
    probs = model.predict_proba(df[features])
    preds = model.predict(df[features])
    labels = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}

    # User Config
    min_edge = st.slider("Minimum Edge % to Place Bet", 0.0, 20.0, 5.0, key="pred_edge") / 100
    stake = st.number_input("Stake per Bet", value=100, key="pred_stake")
    initial_bankroll = st.number_input("Starting Bankroll", value=10000, key="pred_bankroll")

    # Simulate Bets
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
            # No actual outcome (FTR) available, assume prediction for simulation
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

    # Results
    st.subheader("📊 Prediction-Based Betting Results")
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

# --- Strategy Simulation Section ---
st.subheader("🎯 Simulate Betting Strategy on Historical Data")
st.markdown("Upload a CSV with columns `FTR`, `B365H`, `B365D`, `B365A` to simulate a betting strategy using actual match outcomes.")

uploaded_file_strategy = st.file_uploader("Upload Historical Data CSV for Strategy", type=["csv"], key="strategy_uploader")

if uploaded_file_strategy:
    df_strategy = pd.read_csv(uploaded_file_strategy)

    if not all(col in df_strategy.columns for col in ['FTR', 'B365H', 'B365D', 'B365A']):
        st.error("CSV must contain columns: FTR, B365H, B365D, B365A")
        st.stop()

    # Preprocess data
    df_strategy['FTR'] = df_strategy['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    df_strategy['odds_home_prob'] = 1 / df_strategy['B365H']
    df_strategy['odds_draw_prob'] = 1 / df_strategy['B365D']
    df_strategy['odds_away_prob'] = 1 / df_strategy['B365A']
    norm = df_strategy[['odds_home_prob', 'odds_draw_prob', 'odds_away_prob']].sum(axis=1)
    df_strategy['book_home_prob'] = df_strategy['odds_home_prob'] / norm
    df_strategy['book_draw_prob'] = df_strategy['odds_draw_prob'] / norm
    df_strategy['book_away_prob'] = df_strategy['odds_away_prob'] / norm

    # Strategy Parameters
    min_edge_strategy = st.slider("Minimum Edge % to Place Bet", 0.0, 20.0, 5.0, key="strategy_edge") / 100
    stake_strategy = st.number_input("Stake per Bet", value=100, key="strategy_stake")
    initial_bankroll_strategy = st.number_input("Starting Bankroll", value=10000, key="strategy_bankroll")

    # Simulate Strategy
    bankroll = initial_bankroll_strategy
    total_profit = 0
    bet_log = []
    labels = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}

    probs = model.predict_proba(df_strategy[features])
    X_strategy = df_strategy[features]
    odds_strategy = df_strategy[['B365H', 'B365D', 'B365A']]

    for i in range(len(df_strategy)):
        row = X_strategy.iloc[i]
        odds = odds_strategy.iloc[i]
        pred_prob = probs[i]
        actual = df_strategy['FTR'].iloc[i]

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
            if edge > min_edge_strategy and edge > best_edge:
                best_edge = edge
                selected_outcome = outcome

        if selected_outcome is not None:
            price = odds.iloc[selected_outcome]
            profit = (price - 1) * stake_strategy if selected_outcome == actual else -stake_strategy
            bankroll += profit
            total_profit += profit
            bet_log.append({
                'Match': i + 1,
                'Model Pick': labels[selected_outcome],
                'Actual Result': labels[actual],
                'Edge': round(best_edge, 3),
                'Odds': round(price, 2),
                'Profit': round(profit, 2),
                'Bankroll': round(bankroll, 2)
            })

    bet_df_strategy = pd.DataFrame(bet_log)

    # Strategy Results
    st.subheader("📊 Strategy Simulation Results")
    if not bet_df_strategy.empty:
        st.dataframe(bet_df_strategy)
        roi = (total_profit / (len(bet_df_strategy) * stake_strategy)) * 100 if len(bet_df_strategy) > 0 else 0
        st.success(f"✅ Total Profit: ₹{round(total_profit, 2)} | Final Bankroll: ₹{round(bankroll, 2)} | ROI: {roi:.2f}%")

        # ROI vs Edge Threshold
        st.subheader("📈 ROI vs Edge Threshold")
        edge_range = np.linspace(0.01, 0.2, 20)
        roi_list = []
        for threshold in edge_range:
            bkr = initial_bankroll_strategy
            profit = 0
            for i in range(len(df_strategy)):
                implied = [1 / df_strategy.loc[i, 'B365H'], 1 / df_strategy.loc[i, 'B365D'], 1 / df_strategy.loc[i, 'B365A']]
                implied = [x / sum(implied) for x in implied]
                mp = probs[i]
                e = [mp[j] - implied[j] for j in range(3)]
                max_e = max(e)
                sel = np.argmax(e)
                odd = df_strategy.iloc[i, [df_strategy.columns.get_loc('B365H') + sel]].values[0]
                actual = df_strategy['FTR'].iloc[i]
                if max_e > threshold:
                    p = (odd - 1) * stake_strategy if actual == sel else -stake_strategy
                    profit += p
                    bkr += p
            r = (profit / (stake_strategy * len(df_strategy))) * 100 if len(df_strategy) > 0 else 0
            roi_list.append(r)

        fig3, ax3 = plt.subplots()
        ax3.plot(edge_range * 100, roi_list, marker='o')
        ax3.set_xlabel("Edge Threshold (%)")
        ax3.set_ylabel("ROI (%)")
        ax3.set_title("ROI vs Edge Threshold (Strategy)")
        st.pyplot(fig3)

        # Bankroll Over Time
        st.subheader("📉 Bankroll Over Time")
        fig4, ax4 = plt.subplots()
        ax4.plot(bet_df_strategy['Match'], bet_df_strategy['Bankroll'], color='green')
        ax4.set_xlabel("Match")
        ax4.set_ylabel("Bankroll")
        ax4.set_title("Bankroll Over Time (Strategy)")
        st.pyplot(fig4)
    else:
        st.warning("No bets placed. Try lowering your edge threshold.")