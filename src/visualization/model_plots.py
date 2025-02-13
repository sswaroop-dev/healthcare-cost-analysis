"""
Model visualization utilities for healthcare cost prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, Optional
import os

class ModelVisualizer:
    """
    Class for creating model-related visualizations.
    """
    def __init__(self, models: Dict, X_test: pd.DataFrame, y_test: pd.Series,
                 save_path: str = '../../reports/figures/'):
        """
        Initialize model visualizer.
        
        Args:
            models (Dict): Dictionary of trained models
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            save_path (str): Path to save generated plots
        """
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.save_path = save_path
        self.create_save_dir()
        
        # Set default style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def create_save_dir(self):
        """Create directory for saving plots if it doesn't exist."""
        os.makedirs(self.save_path, exist_ok=True)
        
    def plot_feature_importance(self, model_name: str, save: bool = True) -> None:
        """
        Plot feature importance for a specific model.
        
        Args:
            model_name (str): Name of the model
            save (bool): Whether to save the plot
        """
        model = self.models[model_name]
        importance = pd.DataFrame({
            'feature': self.X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['importance'])
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        
        if save:
            plt.savefig(os.path.join(self.save_path, f'feature_importance_{model_name}.png'))
            plt.close()
        else:
            plt.show()
            
    def plot_prediction_scatter(self, model_name: str, save: bool = True) -> None:
        """
        Plot predicted vs actual values.
        
        Args:
            model_name (str): Name of the model
            save (bool): Whether to save the plot
        """
        model = self.models[model_name]
        predictions = model.predict(self.X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, predictions, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2)
        plt.title(f'Predicted vs Actual Values - {model_name}')
        plt.xlabel('Actual Cost')
        plt.ylabel('Predicted Cost')
        
        if save:
            plt.savefig(os.path.join(self.save_path, f'prediction_scatter_{model_name}.png'))
            plt.close()
        else:
            plt.show()
            
    def plot_residuals(self, model_name: str, save: bool = True) -> None:
        """
        Plot residuals analysis.
        
        Args:
            model_name (str): Name of the model
            save (bool): Whether to save the plot
        """
        model = self.models[model_name]
        predictions = model.predict(self.X_test)
        residuals = self.y_test - predictions
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residuals vs Predicted
        ax1.scatter(predictions, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.set_xlabel('Predicted Cost')
        ax1.set_ylabel('Residuals')
        
        # Residuals distribution
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_title('Residuals Distribution')
        ax2.set_xlabel('Residuals')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_path, f'residuals_{model_name}.png'))
            plt.close()
        else:
            plt.show()
            
    def plot_shap_summary(self, model_name: str, save: bool = True) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            model_name (str): Name of the model
            save (bool): Whether to save the plot
        """
        model = self.models[model_name]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_test)
        
        plt.figure()
        shap.summary_plot(shap_values, self.X_test, show=False)
        plt.title(f'SHAP Summary - {model_name}')
        
        if save:
            plt.savefig(os.path.join(self.save_path, f'shap_summary_{model_name}.png'))
            plt.close()
        else:
            plt.show()
            
    def create_model_dashboard(self, model_name: str) -> None:
        """
        Create comprehensive model visualization dashboard.
        
        Args:
            model_name (str): Name of the model
        """
        self.plot_feature_importance(model_name)
        self.plot_prediction_scatter(model_name)
        self.plot_residuals(model_name)
        self.plot_shap_summary(model_name)

def main():
    """
    Main function to demonstrate model visualization pipeline.
    """
    try:
        # Load data and models
        from ..models.train import ModelTrainer
        
        # Load data
        X = pd.read_csv('../../data/processed/engineered_features.csv')
        y = pd.read_csv('../../data/processed/processed_healthcare_costs.csv')['cost']
        
        # Train models
        trainer = ModelTrainer(X, y)
        trainer.setup_all_models()
        trainer.train_all_models()
        
        # Initialize visualizer
        visualizer = ModelVisualizer(
            trainer.trained_models,
            trainer.X_test,
            trainer.y_test
        )
        
        # Create visualizations for each model
        print("Creating model visualizations...")
        for model_name in trainer.trained_models:
            print(f"Processing {model_name}...")
            visualizer.create_model_dashboard(model_name)
            
        print("Model visualizations completed successfully!")
        print(f"Plots saved in:
