from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import pandas as pd

class DataLoaderInterface(ABC):
    """Abstract base class for data loaders in sports betting systems."""
    
    def __init__(self):
        """Initialize the data loader with its class name."""
        self.name = self.__class__.__name__
    
    @abstractmethod
    def load_data(self, source: str, parse_dates: Optional[List[str]] = None, 
                  index_col: Optional[str] = None) -> pd.DataFrame:
        """Load data from a source (e.g., CSV file).
        
        Args:
            source (str): Data source (e.g., filename, URL).
            parse_dates (Optional[List[str]]): Columns to parse as dates.
            index_col (Optional[str]): Column to set as index.
            
        Returns:
            pd.DataFrame: Loaded data.
        """
        pass
    
    @abstractmethod
    def list_data_sources(self) -> List[str]:
        """List available data sources.
        
        Returns:
            List[str]: List of data source identifiers (e.g., filenames).
        """
        pass
    
    @abstractmethod
    def preprocess_match_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess match data (e.g., calculate features, normalize odds).
        
        Args:
            df (pd.DataFrame): Raw match data.
            
        Returns:
            pd.DataFrame: Preprocessed data.
        """
        pass
    
    @abstractmethod
    def get_features_and_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and labels from preprocessed data.
        
        Args:
            df (pd.DataFrame): Preprocessed match data.
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) and labels (y).
        """
        pass