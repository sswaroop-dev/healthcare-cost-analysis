"""
Report visualization utilities for healthcare cost analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ReportVisualizer:
    """
    Class for creating report visualizations summarizing analysis results.
    """
    def __init__(self, save_path: str = '../../reports/figures/'):
        """
        Initialize report visualizer.
        
        Args:
            save_path (str): Path to save generated plots
        """
        self.save_path = save_path
        self.create_save_dir()
        
        # Set default styling
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def create_save_dir(self):
        """Create directory for saving plots if it doesn't exist."""
        os.makedirs(self.save_path, exist_ok=True)

    def plot_model_comparison(self, metrics_dict: Dict[str, Dict], save: bool = True) -> None:
        """
        Plot comparison of different models' performance metrics.
        
        Args:
            metrics_dict (Dict[str, Dict]): Dictionary containing model metrics
            save (bool): Whether to save the plot
        """
        metrics_df = pd.DataFrame(metrics_dict).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # RMSE
        sns.barplot(x=metrics_df.index, y='RMSE', data=metrics_df, ax=axes[0,0])
        axes[0,0].set_title('Root Mean Square Error')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # R2
        sns.barplot(x=metrics_df.index, y='R2', data=metrics_df, ax=axes[0,1])
        axes[0,1].set_title('R-squared Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # MAE
        sns.barplot(x=metrics_df.index, y='MAE', data=metrics_df, ax=axes[1,0])
        axes[1,0].set_title('Mean Absolute Error')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # MAPE
        sns.barplot(x=metrics_df.index, y='MAPE', data=metrics_df, ax=axes[1,1])
        axes[1,1].set_title('Mean Absolute Percentage Error')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_path, 'model_comparison.png'))
            plt.close()
        else:
            plt.show()

    def plot_feature_importance_comparison(self, importance_dict: Dict[str, pd.DataFrame], 
                                         top_n: int = 10, save: bool = True) -> None:
        """
        Plot comparison of feature importance across different models.
        
        Args:
            importance_dict (Dict[str, pd.DataFrame]): Dictionary containing feature importance for each model
            top_n (int): Number of top features to show
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        for model_name, importance_df in importance_dict.items():
            top_features = importance_df.nlargest(top_n, 'importance')
            plt.barh(y=top_features['feature'] + f' ({model_name})', 
                    width=top_features['importance'], 
                    label=model_name, alpha=0.7)
        
        plt.title(f'Top {top_n} Feature Importance Comparison Across Models')
        plt.xlabel('Importance Score')
        plt.legend()
        
        if save:
            plt.savefig(os.path.join(self.save_path, 'feature_importance_comparison.png'))
            plt.close()
        else:
            plt.show()

    def plot_error_analysis(self, error_dict: Dict[str, pd.DataFrame], save: bool = True) -> None:
        """
        Plot error analysis for different models.
        
        Args:
            error_dict (Dict[str, pd.DataFrame]): Dictionary containing prediction errors for each model
            save (bool): Whether to save the plot
        """
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=['Error Distribution', 'Error vs Predicted',
                                         'Percentage Error Distribution', 'Error by Cost Range'])
        
        for model_name, error_df in error_dict.items():
            # Error Distribution
            fig.add_trace(
                go.Histogram(x=error_df['Error'], name=f'{model_name} - Error',
                           opacity=0.7),
                row=1, col=1
            )
            
            # Error vs Predicted
            fig.add_trace(
                go.Scatter(x=error_df['Predicted'], y=error_df['Error'],
                          mode='markers', name=f'{model_name} - Error vs Pred',
                          opacity=0.7),
                row=1, col=2
            )
            
            # Percentage Error Distribution
            fig.add_trace(
                go.Histogram(x=error_df['Percentage_Error'],
                           name=f'{model_name} - % Error',
                           opacity=0.7),
                row=2, col=1
            )
            
            # Error by Cost Range
            error_df['cost_range'] = pd.qcut(error_df['Actual'], q=5)
            mean_errors = error_df.groupby('cost_range')['Error'].mean()
            
            fig.add_trace(
                go.Bar(x=[str(x) for x in mean_errors.index],
                      y=mean_errors.values,
                      name=f'{model_name} - Error by Range',
                      opacity=0.7),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Comprehensive Error Analysis")
        
        if save:
            fig.write_html(os.path.join(self.save_path, 'error_analysis.html'))
        else:
            fig.show()

    def create_summary_report(self, 
                            metrics_dict: Dict[str, Dict],
                            importance_dict: Dict[str, pd.DataFrame],
                            error_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Create comprehensive summary report with all visualizations.
        
        Args:
            metrics_dict (Dict[str, Dict]): Model performance metrics
            importance_dict (Dict[str, pd.DataFrame]): Feature importance data
            error_dict (Dict[str, pd.DataFrame]): Prediction error data
        """
        self.plot_model_comparison(metrics_dict)
        self.plot_feature_importance_comparison(importance_dict)
        self.plot_error_analysis(error_dict)
        
        # Create summary table
        summary_df = pd.DataFrame(metrics_dict).T
        summary_df.to_csv(os.path.join(self.save_path, 'model_summary.csv'))

def main():
    """
    Main function to demonstrate report visualization pipeline.
    """
    try:
        # Example usage with dummy data
        metrics_dict = {
            'RandomForest': {'RMSE': 3487.17, 'R2': 0.573, 'MAE': 1946.08, 'MAPE': 106.91},
            'GradientBoosting': {'RMSE': 3359.59, 'R2': 0.573, 'MAE': 1901.23, 'MAPE': 119.61}
        }
        
        importance_dict = {
            'RandomForest': pd.DataFrame({
                'feature': ['age', 'bmi', 'smoker'],
                'importance': [0.3, 0.2, 0.5]
            }),
            'GradientBoosting': pd.DataFrame({
                'feature': ['age', 'bmi', 'smoker'],
                'importance': [0.25, 0.15, 0.6]
            })
        }
        
        error_dict = {
            'RandomForest': pd.DataFrame({
                'Actual': np.random.normal(5000, 1000, 100),
                'Predicted': np.random.normal(5000, 1000, 100)
            })
        }
        error_dict['RandomForest']['Error'] = error_dict['RandomForest']['Actual'] - error_dict['RandomForest']['Predicted']
        error_dict['RandomForest']['Percentage_Error'] = (error_dict['RandomForest']['Error'] / error_dict['RandomForest']['Actual']) * 100
        
        # Initialize visualizer
        visualizer = ReportVisualizer()
        
        # Create reports
        print("Creating report visualizations...")
        visualizer.create_summary_report(metrics_dict, importance_dict, error_dict)
        
        print("Report visualizations completed successfully!")
        print(f"Plots saved in: {visualizer.save_path}")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")

if __name__ == "__main__":
    main()
