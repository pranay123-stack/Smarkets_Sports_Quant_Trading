import pandas as pd
import numpy as np
from data_handler_interface import DataHandlerInterface
from typing import Dict, List, Any

class StandardDataHandler(DataHandlerInterface):
    """Standard data handler for sports betting match data."""
    
    def __init__(self):
        """Initialize the data handler."""
        super().__init__()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the input data."""
        df = df.drop_duplicates()
        df = df.dropna(subset=['FTR'])
        
        for col in ['B365H', 'B365D', 'B365A']:
            if col in df.columns:
                df = df[(df[col] > 1.01) & (df[col] < 1000)]
        
        df.columns = df.columns.str.strip()
        return df.reset_index(drop=True)
    
    def normalize_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Normalize specified features."""
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
    
    def fill_missing_values(self, df: pd.DataFrame, method: str = "zero") -> pd.DataFrame:
        """Fill missing values."""
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
    
    def enforce_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data consistency."""
        required_cols = ['FTHG', 'FTAG', 'FTR']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df = df[df['FTR'].isin(['H', 'D', 'A'])]
        return df
    
    def detect_anomalies(self, df: pd.DataFrame, z_thresh: float = 3) -> Dict[str, Any]:
        """Detect anomalies using z-scores."""
        anomalies = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_zscore = (df[col] - df[col].mean()) / df[col].std()
            anomalies[col] = df[np.abs(col_zscore) > z_thresh].shape[0]
        return anomalies