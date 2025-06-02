import pandas as pd
import numpy as np
from data_handler_interface import DataHandlerInterface
from typing import Dict, List, Any

class AdvancedDataHandler(DataHandlerInterface):
    """Advanced data handler with quant model preprocessing for sports betting."""
    
    def __init__(self):
        """Initialize the data handler."""
        super().__init__()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the input data with advanced checks."""
        df = df.drop_duplicates()
        df = df.dropna(subset=['FTR'])
        
        for col in ['B365H', 'B365D', 'B365A']:
            if col in df.columns:
                df = df[(df[col] > 1.01) & (df[col] < 1000)]
        
        # Remove outliers using IQR
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        
        df.columns = df.columns.str.strip()
        return df.reset_index(drop=True)
    
    def normalize_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Normalize features with robust scaling."""
        df = df.copy()
        for col in feature_cols:
            if col in df.columns:
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                if iqr != 0:
                    df[col] = (df[col] - median) / iqr
                else:
                    df[col] = 0
        return df
    
    def fill_missing_values(self, df: pd.DataFrame, method: str = "zero") -> pd.DataFrame:
        """Fill missing values with interpolation for time-series data."""
        if method == "interpolate":
            return df.interpolate(method="linear", limit_direction="both")
        elif method == "zero":
            return df.fillna(0)
        elif method == "mean":
            return df.fillna(df.mean())
        elif method == "ffill":
            return df.fillna(method="ffill")
        elif method == "bfill":
            return df.fillna(method="bfill")
        else:
            raise ValueError("Unsupported fill method")
    
    def enforce_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data consistency with advanced checks."""
        required_cols = ['FTHG', 'FTAG', 'FTR']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df = df[df['FTR'].isin(['H', 'D', 'A'])]
        
        # Ensure non-negative goals
        df = df[(df['FTHG'] >= 0) & (df['FTAG'] >= 0)]
        return df
    
    def detect_anomalies(self, df: pd.DataFrame, z_thresh: float = 3) -> Dict[str, Any]:
        """Detect anomalies with quant model (Isolation Forest)."""
        from sklearn.ensemble import IsolationForest
        
        anomalies = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            return anomalies
        
        X = df[numeric_cols].fillna(0)
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        preds = iso_forest.fit_predict(X)
        
        for col in numeric_cols:
            anomalies[col] = np.sum(preds == -1)  # Count outliers
        return anomalies
    
    def calculate_momentum_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Calculate momentum features (non-interface method).
        
        Args:
            df (pd.DataFrame): Input data.
            window (int): Rolling window size (default: 5).
            
        Returns:
            pd.DataFrame: Data with momentum features.
        """
        df = df.copy()
        if 'goal_diff' in df.columns:
            df['momentum_goal_diff'] = df['goal_diff'].rolling(window=window).mean()
        return df