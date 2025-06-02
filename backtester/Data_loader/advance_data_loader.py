import pandas as pd
import os
import numpy as np
from data_loader_interface import DataLoaderInterface
from typing import List, Tuple, Optional
import requests  # For API simulation

class AdvancedDataLoader(DataLoaderInterface):
    """Advanced data loader with real-time data and quant model preprocessing."""
    
    def __init__(self, data_dir: str = "data", api_url: Optional[str] = None):
        """Initialize the data loader.
        
        Args:
            data_dir (str): Directory containing data files.
            api_url (Optional[str]): URL for real-time data API.
        """
        super().__init__()
        self.data_dir = data_dir
        self.api_url = api_url
        self.loaded_files = {}
    
    def load_data(self, source: str, parse_dates: Optional[List[str]] = None, 
                  index_col: Optional[str] = None) -> pd.DataFrame:
        """Load data from CSV or API."""
        if source.startswith('http') or self.api_url:
            # Simulate API fetch
            try:
                url = source if source.startswith('http') else f"{self.api_url}/{source}"
                response = requests.get(url)
                response.raise_for_status()
                df = pd.DataFrame(response.json())  # Simulated
            except Exception as e:
                raise ConnectionError(f"Failed to fetch data from API: {e}")
        else:
            filepath = os.path.join(self.data_dir, source)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"CSV file not found: {filepath}")
            df = pd.read_csv(filepath, parse_dates=parse_dates, index_col=index_col)
        
        self.loaded_files[source] = df
        return df
    
    def list_data_sources(self) -> List[str]:
        """List available data sources (CSV or API endpoints)."""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        api_endpoints = ['live_odds', 'match_stats'] if self.api_url else []  # Simulated
        return csv_files + api_endpoints
    
    def preprocess_match_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess match data with quant model features."""
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Match data missing required columns: {required_cols}")
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['goal_diff'] = df['FTHG'] - df['FTAG']
        df['shot_diff'] = df.get('HS', 0) - df.get('AS', 0)
        df['shot_target_diff'] = df.get('HST', 0) - df.get('AST', 0)
        
        # Quant model: Exponential moving average of goal difference
        df['ema_goal_diff'] = self.calculate_ema(df['goal_diff'], span=5)
        
        # Normalize odds
        for b in ['B365H', 'B365D', 'B365A']:
            if b in df.columns:
                df[f'odds_{b}'] = 1 / df[b]
        if all(f'odds_{b}' in df.columns for b in ['B365H', 'B365D', 'B365A']):
            total_prob = df[[f'odds_B365H', f'odds_B365D', f'odds_B365A']].sum(axis=1)
            df['book_home_prob'] = df['odds_B365H'] / total_prob
            df['book_draw_prob'] = df['odds_B365D'] / total_prob
            df['book_away_prob'] = df['odds_B365A'] / total_prob
        
        return df.dropna()
    
    def get_features_and_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and labels with quant model features."""
        features = [
            'goal_diff', 'shot_diff', 'shot_target_diff', 'ema_goal_diff',
            'book_home_prob', 'book_draw_prob', 'book_away_prob'
        ]
        X = df[features].fillna(0)
        y = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        return X, y
    
    def calculate_ema(self, series: pd.Series, span: int = 5) -> pd.Series:
        """Calculate exponential moving average (non-interface method).
        
        Args:
            series (pd.Series): Input series (e.g., goal_diff).
            span (int): EMA span (default: 5).
            
        Returns:
            pd.Series: EMA values.
        """
        return series.ewm(span=span, adjust=False).mean()