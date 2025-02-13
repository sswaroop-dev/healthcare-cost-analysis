"""
Model evaluation utilities for healthcare cost prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap

class ModelEvaluator:
    """
    Class for evaluating regression models for healthcare cost prediction.
    """
    def __init__(self, trained_models: Dict, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series):
        """
        Initialize model evaluator.
        
        Args:
            trained_models (Dict): Dictionary of trained models
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Testing features
            y_train (pd.Series): Training target
            y_test (pd.Series): Testing target
        """
        self.trained_models = trained_models
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.predictions = {}
        self.metrics = {}
        
    def make_predictions(self):
        """Make predictions for all models on test data."""
        for name, model in self.trained_models.items():
            self.predictions[name] = model.predict(self.X_test)
            
    def calculate_metrics(self) -> Dict:
        """
        Calculate evaluation metrics for all models.
        
        Returns:
            Dict: Dictionary containing evaluation metrics for each model
        """
        for name, y_pred in self.predictions.items():
            self.metrics[name] = {
                'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
                'MAE': mean_absolute_error(self.y_test, y_pred),
                'R2': r2_score(self.y_test, y_pred),
                'MPE': np.mean((self.y_test - y_pred) / self.y_test) * 100,
                'MAPE': np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
            }
        return self.metrics
    
    def get_shap_values(self, model_name: str, max_display: int = 10):
        """
        Calculate SHAP values for feature importance interpretation.
        
        Args:
            model_name (str): Name of the model to analyze
            max_display (int): Maximum number of features to display
            
        Returns:
            tuple: SHAP explainer and values
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found.")
            
        model = self.trained_models[model_name]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_test)
        
        return explainer, shap_values
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get comparison of all models' performance metrics.
        
        Returns:
            pd.DataFrame: Comparison of model metrics
        """
        if not self.metrics:
            self.calculate_metrics()
            
        comparison = pd.DataFrame(self.metrics).T
        return comparison
    
    def get_prediction_errors(self, model_name: str) -> pd.DataFrame:
        """
        Get prediction errors analysis for a specific model.
        
        Args:
            model_name (str): Name of the model to analyze
            
        Returns:
            pd.DataFrame: DataFrame with actual, predicted values and errors
        """
        if model_name not in self.predictions:
            raise ValueError(f"No predictions found for model {model_name}")
            
        errors_df = pd.DataFrame({
            'Actual': self.y_test,
            'Predicted': self.predictions[model_name],
            'Error': self.y_test - self.predictions[model_name],
            'Percentage_Error': ((self.y_test - self.predictions[model_name]) / self.y_test) * 100
        })
        return errors_df
    
    def evaluate_all_models(self) -> Dict:
        """
        Run complete evaluation pipeline for all models.
        
        Returns:
            Dict: Dictionary containing all evaluation results
        """
        self.make_predictions()
        metrics = self.calculate_metrics()
        
        evaluation_results = {
            'metrics': metrics,
            'comparison': self.get_model_comparison(),
            'errors': {name: self.get_prediction_errors(name) 
                      for name in self.trained_models.keys()}
        }
        
        return evaluation_results

def main():
    """
    Main function to demonstrate model evaluation pipeline.
    """
    try:
        # Load data and models
        from train import ModelTrainer
        X = pd.read_csv('../../data/processed/engineered_features.csv')
        y = pd.read_csv('../../data/processed/processed_healthcare_costs.csv')['cost']
        
        # Train models
        trainer = ModelTrainer(X, y)
        trainer.setup_all_models()
        trainer.train_all_models()
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            trainer.trained_models,
            trainer.X_train,
            trainer.X_test,
            trainer.y_train,
            trainer.y_test
        )
        
        # Run evaluation
        print("Evaluating models...")
        results = evaluator.evaluate_all_models()
        
        # Print results
        print("\nModel Comparison:")
        print(results['comparison'])
        
        # Save evaluation results
        results['comparison'].to_csv('../../reports/model_comparison.csv')
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")

if __name__ == "__main__":
    main()
