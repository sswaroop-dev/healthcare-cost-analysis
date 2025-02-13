"""
Exploratory Data Analysis visualization utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from typing import Optional, Tuple, List
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
        
        # Set default style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def create_save_dir(self):
        """Create directory for saving plots if it doesn't exist."""
        os.makedirs(self.save_path, exist_ok=True)
        
    def plot_cost_distribution(self, save: bool = True) -> None:
        """
        Plot distribution of healthcare costs.
        
        Args:
            save (bool): Whether to save the plot
        """
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
            
    def plot_cost_by_category(self, category: str, save: bool = True) -> None:
        """
        Plot cost distribution by categorical variable.
        
        Args:
            category (str): Categorical variable to analyze
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.data, x=category, y='cost')
        plt.title(f'Healthcare Costs by {category}')
        plt.xticks(rotation=45)
        plt.xlabel(category)
        plt.ylabel('Cost (USD)')
        
        if save:
            plt.savefig(os.path.join(self.save_path, f'cost_by_{category}.png'))
            plt.close()
        else:
            plt.show()
            
    def plot_correlation_matrix(self, save: bool = True) -> None:
        """
        Plot correlation matrix for numerical variables.
        
        Args:
            save (bool): Whether to save the plot
        """
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.data[numerical_cols].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        
        if save:
            plt.savefig(os.path.join(self.save_path, 'correlation_matrix.png'))
            plt.close()
        else:
            plt.show()
            
    def plot_age_cost_relationship(self, save: bool = True) -> None:
        """
        Plot relationship between age and cost with smoking status.
        
        Args:
            save (bool): Whether to save the plot
        """
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
            
    def plot_bmi_cost_relationship(self, save: bool = True) -> None:
        """
        Plot relationship between BMI and cost.
        
        Args:
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x='bmi', y='cost', alpha=0.5)
        plt.title('BMI vs. Cost')
        plt.xlabel('BMI')
        plt.ylabel('Cost (USD)')
        
        if save:
            plt.savefig(os.path.join(self.save_path, 'bmi_cost_relationship.png'))
            plt.close()
        else:
            plt.show()
            
    def plot_geographic_distribution(self, save: bool = True) -> None:
        """
        Plot average costs by location.
        
        Args:
            save (bool): Whether to save the plot
        """
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
        """Create comprehensive EDA dashboard with all plots."""
        self.plot_cost_distribution()
        self.plot_correlation_matrix()
        self.plot_age_cost_relationship()
        self.plot_bmi_cost_relationship()
        self.plot_geographic_distribution()
        
        categorical_vars = ['smoker', 'exercise', 'married', 'education_level']
        for var in categorical_vars:
            if var in self.data.columns:
                self.plot_cost_by_category(var)

def main():
    """
    Main function to demonstrate EDA visualization pipeline.
    """
    try:
        # Load preprocessed data
        data = pd.read_csv('../../data/processed/processed_healthcare_costs.csv')
        
        # Initialize visualizer
        visualizer = EDAVisualizer(data)
        
        # Create all plots
        print("Creating EDA visualizations...")
        visualizer.create_eda_dashboard()
        
        print("EDA visualizations completed successfully!")
        print(f"Plots saved in: {visualizer.save_path}")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")

if __name__ == "__main__":
    main()
