import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self):
        pass

    def clean_data(self, df):
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Drop rows with clearly invalid entries
        df = df.dropna(subset=['FTR'])

        # Remove unrealistic odds
        for col in ['B365H', 'B365D', 'B365A']:
            if col in df.columns:
                df = df[(df[col] > 1.01) & (df[col] < 1000)]

        # Standardize column names (remove whitespace)
        df.columns = df.columns.str.strip()

        return df.reset_index(drop=True)

    def normalize_features(self, df, feature_cols):
        df = df.copy()
        for col in feature_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std != 0:
                    df[col] = (df[col] - mean) / std
                else:
                    df[col] = 0
        return df

    def fill_missing_values(self, df, method="zero"):
        if method == "zero":
            return df.fillna(0)
        elif method == "mean":
            return df.fillna(df.mean())
        elif method == "ffill":
            return df.fillna(method="ffill")
        elif method == "bfill":
            return df.fillna(method="bfill")
        else:
            raise ValueError("Unsupported fill method")

    def enforce_consistency(self, df):
        # Ensure all required columns exist
        required_cols = ['FTHG', 'FTAG', 'FTR']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Make sure outcome is valid
        df = df[df['FTR'].isin(['H', 'D', 'A'])]

        return df

    def detect_anomalies(self, df, z_thresh=3):
        anomalies = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_zscore = (df[col] - df[col].mean()) / df[col].std()
            anomalies[col] = df[np.abs(col_zscore) > z_thresh].shape[0]
        return anomalies
