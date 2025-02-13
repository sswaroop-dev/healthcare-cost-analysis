"""
Exploratory Data Analysis visualization utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from typing import Optional, List
import os

class EDAVisualizer:
    """
    Class for creating EDA visualizations for healthcare cost analysis.
    """
    def __init__(self, data: pd.DataFrame, save_path: str = '../../reports/figures/'):
        """
        Initialize EDA visualizer.
        
        Args:
            data (pd.DataFrame): Dataset for visualization
            save_path (str): Path to save generated plots
        """
        self.data = data
        self.save_path = save_path
        self.create_save_dir()
        
        # Set style for consistent visualization
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def create_save_dir(self):
        """Create directory for saving plots if it doesn't exist."""
        os.makedirs(self.save_path, exist_ok=True)
        
    def plot_cost_distribution(self, save: bool = True) -> None:
        """Plot distribution of healthcare costs."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='cost', bins=50, kde=True)
        plt.title('Distribution of Healthcare Costs')
        plt.xlabel('Cost (USD)')
        plt.ylabel('Frequency')
        
        if save:
            plt.savefig(os.path.join(self.save_path, 'cost_distribution.png'))
            plt.close()
        else:
            plt.show()
            
    def plot_correlation_matrix(self, save: bool = True) -> None:
        """Plot correlation matrix for numerical variables."""
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.data[numerical_cols].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numerical Variables')
        
        if save:
            plt.savefig(os.path.join(self.save_path, 'correlation_matrix.png'))
            plt.close()
        else:
            plt.show()
            
    def plot_age_cost_relationship(self, save: bool = True) -> None:
        """Plot relationship between age and cost with smoking status."""
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=self.data, x='age', y='cost', hue='smoker', alpha=0.6)
        plt.title('Age vs. Cost by Smoking Status')
        plt.xlabel('Age')
        plt.ylabel('Cost (USD)')
        
        if save:
            plt.savefig(os.path.join(self.save_path, 'age_cost_relationship.png'))
            plt.close()
        else:
            plt.show()
            
    def plot_categorical_analysis(self, save: bool = True) -> None:
        """Plot cost distributions for categorical variables."""
        categorical_vars = ['smoker', 'exercise', 'yearly_physical', 'location_type']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cost Distribution by Categorical Variables')
        
        for i, var in enumerate(categorical_vars):
            if var in self.data.columns:
                ax = axes[i//2, i%2]
                sns.boxplot(data=self.data, x=var, y='cost', ax=ax)
                ax.set_title(f'Cost by {var}')
                ax.tick_params(rotation=45)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_path, 'categorical_analysis.png'))
            plt.close()
        else:
            plt.show()
            
    def plot_bmi_analysis(self, save: bool = True) -> None:
        """Plot BMI-related visualizations."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # BMI distribution
        sns.histplot(data=self.data, x='bmi', bins=30, kde=True, ax=ax1)
        ax1.set_title('BMI Distribution')
        
        # BMI vs Cost
        sns.scatterplot(data=self.data, x='bmi', y='cost', hue='smoker', ax=ax2)
        ax2.set_title('BMI vs Cost by Smoking Status')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.save_path, 'bmi_analysis.png'))
            plt.close()
        else:
            plt.show()
            
    def plot_geographic_distribution(self, save: bool = True) -> None:
        """Plot geographic distribution of costs."""
        avg_cost_by_location = self.data.groupby('location')['cost'].mean().reset_index()
        
        fig = px.choropleth(avg_cost_by_location,
                          locations='location',
                          locationmode='USA-states',
                          color='cost',
                          scope='usa',
                          color_continuous_scale='Viridis',
                          title='Average Healthcare Costs by State')
        
        if save:
            fig.write_html(os.path.join(self.save_path, 'geographic_distribution.html'))
        else:
            fig.show()
            
    def create_eda_dashboard(self) -> None:
        """Create comprehensive EDA dashboard."""
        self.plot_cost_distribution()
        self.plot_correlation_matrix()
        self.plot_age_cost_relationship()
        self.plot_categorical_analysis()
        self.plot_bmi_analysis()
        self.plot_geographic_distribution()
