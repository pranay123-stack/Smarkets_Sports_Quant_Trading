from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from smarkets_Sports_Quant_Trading.backtester.strategies.strategy_interface.ml_model_strategy_inteface import MLModelStrategyInterface

class MLStrategy1(MLModelStrategyInterface):
    """ML betting strategy implementing the interface for sports outcome prediction."""
    
    def __init__(self, df: pd.DataFrame, label_col: str = 'is_home_win'):
        """Initialize the strategy with a DataFrame and target label column.
        
        Args:
            df (pd.DataFrame): Input data for training and prediction.
            label_col (str): Name of the target label column (e.g., 'is_home_win').
        """
        self.df = df.copy()
        self.label_col = label_col
        self.model_dict = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
            'LightGBM': LGBMClassifier(random_state=42)
        }
        self.trained_models = {}
    
    def train_models(self, features: list):
        """Train the models using the specified features.
        
        Args:
            features (list): List of feature column names to use for training.
        """
        X = self.df[features]
        y = self.df[self.label_col]
        for name, model in self.model_dict.items():
            model.fit(X, y)
            self.trained_models[name] = model
    
    def predict(self, row: pd.Series, features: list, model_name: str = 'RandomForest', **kwargs) -> tuple:
        """Make a betting prediction for a single row using the specified model.
        
        Args:
            row (pd.Series): Input row for prediction.
            features (list): List of feature column names.
            model_name (str): Name of the model to use for prediction.
            **kwargs: Additional arguments, e.g., odds (default=2.0), min_prob (default=0.55).
            
        Returns:
            tuple: (bet, odds, stake) where bet is 'home', 'away', or None.
        """
        odds = kwargs.get('odds', 2.0)
        min_prob = kwargs.get('min_prob', 0.55)
        if model_name not in self.trained_models:
            return (None, 0, 0)
        X_row = row[features].values.reshape(1, -1)
        prob = self.trained_models[model_name].predict_proba(X_row)[0]
        # Assuming binary classification: prob[1] is probability of home win, prob[0] is away win
        if prob[1] > min_prob:
            return ('home', odds, 100)
        elif prob[0] > min_prob:
            return ('away', odds, 100)
        return (None, 0, 0)