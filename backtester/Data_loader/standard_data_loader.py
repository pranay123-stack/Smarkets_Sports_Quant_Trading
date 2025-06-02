import pandas as pd
import os
from data_loader_interface import DataLoaderInterface
from typing import List, Tuple, Optional

class StandardDataLoader(DataLoaderInterface):
    """Standard data loader for sports betting match data."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data loader.
        
        Args:
            data_dir (str): Directory containing data files (default: 'data').
        """
        super().__init__()
        self.data_dir = data_dir
        self.loaded_files = {}
    
    def load_data(self, source: str, parse_dates: Optional[List[str]] = None, 
                  index_col: Optional[str] = None) -> pd.DataFrame:
        """Load data from a CSV file."""
        filepath = os.path.join(self.data_dir, source)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=parse_dates, index_col=index_col)
        self.loaded_files[source] = df
        return df
    
    def list_data_sources(self) -> List[str]:
        """List available CSV files."""
        return [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
    
    def preprocess_match_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess match data."""
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Match data missing required columns: {required_cols}")
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['goal_diff'] = df['FTHG'] - df['FTAG']
        df['shot_diff'] = df.get('HS', 0) - df.get('AS', 0)
        df['shot_target_diff'] = df.get('HST', 0) - df.get('AST', 0)
        df['corner_diff'] = df.get('HC', 0) - df.get('AC', 0)
        df['foul_diff'] = df.get('HF', 0) - df.get('AF', 0)
        df['yellow_diff'] = df.get('HY', 0) - df.get('AY', 0)
        df['red_diff'] = df.get('HR', 0) - df.get('AR', 0)
        
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
        """Extract features and labels."""
        features = [
            'goal_diff', 'shot_diff', 'shot_target_diff', 'corner_diff',
            'foul_diff', 'yellow_diff', 'red_diff',
            'book_home_prob', 'book_draw_prob', 'book_away_prob'
        ]
        X = df[features].fillna(0)
        y = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        return X, y