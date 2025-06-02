from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd

class DataHandlerInterface(ABC):
    """Abstract base class for data handlers in sports betting systems."""
    
    def __init__(self):
        """Initialize the data handler with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the input data (e.g., remove duplicates, invalid entries).
        
        Args:
            df (pd.DataFrame): Input data.
            
        Returns:
            pd.DataFrame: Cleaned data.
        """
        pass
    
    @abstractmethod
    def normalize_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Normalize specified features (e.g., z-score normalization).
        
        Args:
            df (pd.DataFrame): Input data.
            feature_cols (List[str]): Columns to normalize.
            
        Returns:
            pd.DataFrame: Data with normalized features.
        """
        pass
    
    @abstractmethod
    def fill_missing_values(self, df: pd.DataFrame, method: str = "zero") -> pd.DataFrame:
        """Fill missing values using the specified method.
        
        Args:
            df (pd.DataFrame): Input data.
            method (str): Fill method (e.g., 'zero', 'mean', 'ffill', 'bfill').
            
        Returns:
            pd.DataFrame: Data with filled values.
        """
        pass
    
    @abstractmethod
    def enforce_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure data consistency (e.g., required columns, valid outcomes).
        
        Args:
            df (pd.DataFrame): Input data.
            
        Returns:
            pd.DataFrame: Consistent data.
        """
        pass
    
    @abstractmethod
    def detect_anomalies(self, df: pd.DataFrame, z_thresh: float = 3) -> Dict[str, Any]:
        """Detect anomalies in numeric columns using z-scores.
        
        Args:
            df (pd.DataFrame): Input data.
            z_thresh (float): Z-score threshold for anomalies (default: 3).
            
        Returns:
            Dict[str, Any]: Anomalies per column (e.g., count of outliers).
        """
        pass