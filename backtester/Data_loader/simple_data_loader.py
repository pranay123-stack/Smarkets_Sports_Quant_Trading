import pandas as pd
import os
from data_loader_interface import DataLoaderInterface
from typing import List, Tuple, Optional

class SimpleDataLoader(DataLoaderInterface):
    """Simple data loader with basic preprocessing for sports betting."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data loader."""
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
        """Perform minimal preprocessing."""
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Match data missing required columns: {required_cols}")
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        return df.dropna()
    
    def get_features_and_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract basic features and labels."""
        features = ['B365H', 'B365D', 'B365A'] if all(col in df.columns for col in ['B365H', 'B365D', 'B365A']) else []
        X = df[features].fillna(0) if features else pd.DataFrame(index=df.index)
        y = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        return X, y