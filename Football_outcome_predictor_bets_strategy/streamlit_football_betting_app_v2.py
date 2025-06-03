import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="⚽ Betting Edge Predictor", layout="wide")
st.title("⚽ Football Betting Strategy Simulator")

st.markdown("""
Use Football (Soccer) data from [Football-Data.co.uk](https://www.football-data.co.uk/).
This app uses a pre-trained RandomForest model to:
- **Predict outcomes** and simulate bets on uploaded odds data (`B365H`, `B365D`, `B365A`).
- **Simulate a betting strategy** on historical data with actual results (`FTR`, `B365H`, `B365D`, `B365A`).
""")

# Dynamically determine the repository root and model path
def get_model_path():
    # Get the directory of the current script (Football_outcome_predictor_bets_strategy)
    app_dir = os.path.dirname(os.path.abspath(__file__))
    # Move up one level to Smarkets_Sports_Quant_Trading (repo root)
    repo_root = os.path.dirname(app_dir)
    # Construct path to model.pkl relative to repo root
    model_path = os.path.join(repo_root, "Football_outcome_predictor_bets_strategy", "Trained_ML_Models", "model.pkl")
    
    # Fallback: Try case-insensitive or common variations
    fallback_path = os.path.join(app_dir, "Trained_ML_Models", "model.pkl")
    lowercase_path = os.path.join(repo_root, "football_outcome_predictor_bets_strategy", "trained_ml_models", "model.pkl")
    
    # Return the first valid path
    for path in [model_path, fallback_path, lowercase_path]:
        if os.path.exists(path):
            return path
    return model_path  # Return primary path for error message

# Get model path
model_path = get_model_path()

# Debug: Display the resolved model path
st.write(f"Attempting to load model from: {os.path.abspath(model_path)}")

# Load pre-trained model
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    st.success(f"✅ Pre-trained model loaded from {model_path}")
except FileNotFoundError:
    st.error(f"❌ Pre-trained model not found at {model_path}. Please ensure 'model.pkl' exists in Smarkets_Sports_Quant_Trading/Football_outcome_predictor_bets_strategy/Trained_ML_Models/ within the repository.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# File uploader for CSV
st.subheader("📂 Upload Data")
data_file = st.file_uploader("Upload CSV file (e.g., E3.csv)", type=["csv"])

# Check if file is uploaded
if data_file is not None:
    # Load CSV data
    try:
        df = pd.read_csv(data_file)
        st.success("✅ CSV file uploaded successfully!")
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {e}")
        st.stop()

    # Validate required columns
    required_columns = ['FTR', 'B365H', 'B365D', 'B365A']
    if not all(col in df.columns for col in required_columns):
        st.error(f"❌ CSV must contain columns: {', '.join(required_columns)}")
        st.stop()

    # Preprocess data
    df = df[required_columns].dropna()
    df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

    # Compute implied probabilities
    df['odds_home_prob'] = 1 / df['B365H']
    df['odds_draw_prob'] = 1 / df['B365D']
    df['odds_away_prob'] = 1 / df['B365A']

    # Normalize probabilities
    df['norm'] = df['odds_home_prob'] + df['odds_draw_prob'] + df['odds_away_prob']
    df['book_home_prob'] = df['odds_home_prob'] / df['norm']
    df['book_draw_prob'] = df['odds_draw_prob'] / df['norm']
    df['book_away_prob'] = df['odds_away_prob'] / df['norm']

    # Define features, target, and odds
    features = ['book_home_prob', 'book_draw_prob', 'book_away_prob']
    X = df[features]
    y = df['FTR']
    odds_df = df[['B365H', 'B365D', 'B365A']]

    # Predict probabilities
    try:
        probs = model.predict_proba(X)
    except Exception as e:
        st.error(f"❌ Error predicting with model: {e}")
        st.stop()

    # Betting strategy parameters
    st.subheader("🎯 Betting Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        stake = st.slider("💰 Stake per Bet (₹)", 50, 500, 100)
    with col2:
        bankroll = st.slider("🏦 Initial Bankroll (₹)", 5000, 50000, 10000)
    with col3:
        min_edge = st.slider("📈 Minimum Edge", 0.01, 0.2, 0.05, step=0.01)

    # Simulate betting strategy
    total_profit = 0
    bet_log = []
    labels = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}

    for i in range(len(X)):
        row = X.iloc[i]
        odds = odds_df.iloc[i]
        pred_prob = probs[i]
        actual = y.iloc[i]

        # Implied bookmaker probabilities
        implied_probs = [1 / odds['B365H'], 1 / odds['B365D'], 1 / odds['B365A']]
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
            profit = (price - 1) * stake if selected_outcome == actual else -stake
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

    # Display results
    bet_df = pd.DataFrame(bet_log)
    st.subheader("📊 Betting Simulation Log")
    st.dataframe(bet_df, use_container_width=True)

    roi = (total_profit / (len(bet_df) * stake)) * 100 if len(bet_df) > 0 else 0
    st.markdown(f"""
    ### 📈 Summary
    - **Total Profit**: ₹{round(total_profit, 2)}  
    - **Final Bankroll**: ₹{round(bankroll, 2)}  
    - **ROI**: {roi:.2f}%  
    - **Bets Placed**: {len(bet_df)}
    """)

    # Download button for results
    output = BytesIO()
    bet_df.to_csv(output, index=False)
    output.seek(0)
    st.download_button(
        label="📥 Download Betting Log as CSV",
        data=output,
        file_name="betting_simulation_log.csv",
        mime="text/csv"
    )
else:
    st.info("📤 Please upload a CSV file to start the simulation.")