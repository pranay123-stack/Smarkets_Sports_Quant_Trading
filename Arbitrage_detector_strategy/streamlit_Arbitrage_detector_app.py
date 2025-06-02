import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="⚽ Arbitrage Detector", layout="wide")
st.title("⚽ Market Inefficiency Arbitrage Detector")

st.markdown("""
Use Football (Soccer) data from Football-Data.co.uk
Upload a CSV file (e.g. `E0.csv`) with bookmaker odds: `B365H`, `B365D`, `B365A`.  
This app will train a predictive model, compare model vs market probabilities, detect value bets,
and simulate bankroll growth over time.
""")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    if not all(col in raw_df.columns for col in ['FTR', 'B365H', 'B365D', 'B365A']):
        st.error("CSV must contain columns: FTR, B365H, B365D, B365A")
        st.stop()

    df = raw_df[['FTR', 'B365H', 'B365D', 'B365A']].dropna()
    df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

    # Calculate normalized implied probabilities
    for side in ['H', 'D', 'A']:
        df[f'prob_{side}'] = 1 / df[f'B365{side}']
    df['total_prob'] = df[['prob_H', 'prob_D', 'prob_A']].sum(axis=1)
    for side in ['H', 'D', 'A']:
        df[f'norm_prob_{side}'] = df[f'prob_{side}'] / df['total_prob']

    # Feature preparation
    features = ['norm_prob_H', 'norm_prob_D', 'norm_prob_A']
    X = df[features]
    y = df['FTR']

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    probs = model.predict_proba(X)
    preds = model.predict(X)

    # User input config
    threshold = st.slider("Edge Threshold (%)", 0.0, 20.0, 5.0) / 100
    stake = st.number_input("Stake per Bet", value=100)
    initial_bankroll = st.number_input("Initial Bankroll", value=10000)

    # Simulate arbitrage betting
    bankroll = initial_bankroll
    log = []
    for i in range(len(X)):
        implied = [1 / df.iloc[i]['B365H'], 1 / df.iloc[i]['B365D'], 1 / df.iloc[i]['B365A']]
        implied = [x / sum(implied) for x in implied]
        model_prob = probs[i]
        edges = [model_prob[j] - implied[j] for j in range(3)]
        max_edge = max(edges)
        pick = edges.index(max_edge)
        odds = df.iloc[i][['B365H', 'B365D', 'B365A']].values[pick]
        actual = y.iloc[i]

        if max_edge > threshold:
            win = pick == actual
            profit = (odds - 1) * stake if win else -stake
            bankroll += profit
            log.append({
                'Match': i + 1,
                'Model Pick': ['Home', 'Draw', 'Away'][pick],
                'Actual Result': ['Home', 'Draw', 'Away'][actual],
                'Edge': round(max_edge, 3),
                'Odds': round(odds, 2),
                'Profit': round(profit, 2),
                'Bankroll': round(bankroll, 2)
            })

    log_df = pd.DataFrame(log)

    # Results
    st.subheader("📊 Arbitrage Simulation Results")
    if not log_df.empty:
        st.dataframe(log_df)
        roi = ((bankroll - initial_bankroll) / (stake * len(log_df))) * 100
        st.success(f"✅ Profit: ₹{round(bankroll - initial_bankroll, 2)} | ROI: {roi:.2f}% | Bets Placed: {len(log_df)}")

        # ROI vs Edge Threshold (simulate)
        st.subheader("📈 ROI vs Edge Threshold")
        thresholds = np.linspace(0.01, 0.2, 20)
        roi_curve = []
        for th in thresholds:
            bkr, profit = initial_bankroll, 0
            for i in range(len(X)):
                implied = [1 / df.iloc[i]['B365H'], 1 / df.iloc[i]['B365D'], 1 / df.iloc[i]['B365A']]
                implied = [x / sum(implied) for x in implied]
                model_prob = probs[i]
                edges = [model_prob[j] - implied[j] for j in range(3)]
                e = max(edges)
                sel = edges.index(e)
                odd = df.iloc[i][['B365H', 'B365D', 'B365A']].values[sel]
                if e > th:
                    p = (odd - 1) * stake if preds[i] == sel else -stake
                    bkr += p
                    profit += p
            roi_th = (profit / (stake * len(X))) * 100
            roi_curve.append(roi_th)

        fig, ax = plt.subplots()
        ax.plot(thresholds * 100, roi_curve, marker='o')
        ax.set_xlabel("Edge Threshold (%)")
        ax.set_ylabel("ROI (%)")
        ax.set_title("ROI vs Edge Threshold")
        st.pyplot(fig)

        # Bankroll curve
        st.subheader("📉 Bankroll Over Time")
        fig2, ax2 = plt.subplots()
        ax2.plot(log_df['Match'], log_df['Bankroll'], color='green')
        ax2.set_xlabel("Match")
        ax2.set_ylabel("Bankroll")
        ax2.set_title("Bankroll Curve")
        st.pyplot(fig2)
    else:
        st.warning("No bets met the edge threshold. Try lowering the threshold.")
