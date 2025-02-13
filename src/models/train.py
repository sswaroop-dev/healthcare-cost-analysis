"""
Model training utilities for healthcare cost prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib

class ModelTrainer:
    """
    Class for training different regression models for healthcare cost prediction.
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of dataset to include in the test split
            random_state (int): Random state for reproducibility
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self._split_data()
        
    def _split_data(self):
        """Split data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
    def setup_random_forest(self, params: Optional[Dict[str, Any]] = None):
        """
        Setup Random Forest model with parameters.
        
        Args:
            params (Dict[str, Any], optional): Model parameters
        """
        default_params = {
            'n_estimators': 100,
            'max_features': int(np.sqrt(self.X.shape[1])),
            'random_state': self.random_state
        }
        if params:
            default_params.update(params)
        
        self.models['random_forest'] = RandomForestRegressor(**default_params)
        
    def setup_gradient_boosting(self, params: Optional[Dict[str, Any]] = None):
        """
        Setup Gradient Boosting model with parameters.
        
        Args:
            params (Dict[str, Any], optional): Model parameters
        """
        default_params = {
            'n_estimators': 500,
            'learning_rate': 0.01,
            'random_state': self.random_state
        }
        if params:
            default_params.update(params)
        
        self.models['gradient_boosting'] = GradientBoostingRegressor(**default_params)
        
    def setup_all_models(self, rf_params: Optional[Dict[str, Any]] = None, 
                        gb_params: Optional[Dict[str, Any]] = None):
        """
        Setup all models with given parameters.
        
        Args:
            rf_params (Dict[str, Any], optional): Random Forest parameters
            gb_params (Dict[str, Any], optional): Gradient Boosting parameters
        """
        self.setup_random_forest(rf_params)
        self.setup_gradient_boosting(gb_params)
        
    def train_model(self, model_name: str):
        """
        Train a specific model.
        
        Args:
            model_name (str): Name of the model to train
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Please set up the model first.")
            
        print(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(self.X_train, self.y_train)
        self.trained_models[model_name] = model
        
    def train_all_models(self):
        """Train all setup models."""
        for model_name in self.models:
            self.train_model(model_name)
            
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path where to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found or not trained yet.")
            
        joblib.dump(self.trained_models[model_name], filepath)
        
    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        """
        Get feature importance for a trained model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found or not trained yet.")
            
        model = self.trained_models[model_name]
        importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)

def main():
    """
    Main function to demonstrate model training pipeline.
    """
    try:
        # Load engineered features
        X = pd.read_csv('../../data/processed/engineered_features.csv')
        y = pd.read_csv('../../data/processed/processed_healthcare_costs.csv')['cost']
        
        # Initialize trainer
        trainer = ModelTrainer(X, y)
        
        # Setup and train models
        print("Setting up models...")
        trainer.setup_all_models()
        
        print("Training models...")
        trainer.train_all_models()
        
        # Save models
        print("Saving trained models...")
        trainer.save_model('random_forest', '../../models/random_forest.joblib')
        trainer.save_model('gradient_boosting', '../../models/gradient_boosting.joblib')
        
        # Print feature importance
        for model_name in ['random_forest', 'gradient_boosting']:
            print(f"\nFeature Importance for {model_name}:")
            print(trainer.get_feature_importance(model_name))
            
        print("\nModel training completed successfully!")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")

if __name__ == "__main__":
    main()
