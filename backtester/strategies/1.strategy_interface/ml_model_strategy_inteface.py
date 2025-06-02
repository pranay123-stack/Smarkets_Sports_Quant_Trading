from abc import ABC, abstractmethod
import pandas as pd

class MLModelStrategyInterface(ABC):
    """Abstract base class defining the interface for ML model strategies."""
    
    @abstractmethod
    def __init__(self, df: pd.DataFrame, label_col: str):
        """Initialize the strategy with a DataFrame and target label column.
        
        Args:
            df (pd.DataFrame): Input data for training and prediction.
            label_col (str): Name of the target label column.
        """
        pass
    
    @abstractmethod
    def train_models(self, features: list):
        """Train the models using the specified features.
        
        Args:
            features (list): List of feature column names to use for training.
        """
        pass
    
    @abstractmethod
    def predict(self, row: pd.Series, features: list, model_name: str, **kwargs) -> tuple:
        """Make a prediction for a single row using the specified model.
        
        Args:
            row (pd.Series): Input row for prediction.
            features (list): List of feature column names.
            model_name (str): Name of the model to use for prediction.
            **kwargs: Additional arguments (e.g., odds for betting strategies).
            
        Returns:
            tuple: Prediction result, typically (prediction, confidence/probability, stake).
        """
        pass