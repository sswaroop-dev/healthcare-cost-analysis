"""
Feature engineering and selection utilities for healthcare cost analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

class VIFSelector:
    """
    Class for handling multicollinearity using Variance Inflation Factor (VIF).
    """
    def __init__(self, threshold: float = 5.0):
        """
        Initialize VIF selector.
        
        Args:
            threshold (float): VIF threshold for feature elimination
        """
        self.threshold = threshold
        self.vif_scores = None
        self.selected_features = None
        
    def calculate_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VIF for all features.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            pd.DataFrame: DataFrame with VIF scores
        """
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(X.shape[1])]
        self.vif_scores = vif_data.sort_values('VIF', ascending=False)
        return self.vif_scores
    
    def select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features based on VIF threshold.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            pd.DataFrame: DataFrame with selected features
        """
        features = X.columns.tolist()
        while True:
            vif_data = self.calculate_vif(X[features])
            if vif_data['VIF'].max() <= self.threshold:
                break
            features.remove(vif_data.iloc[0]['Feature'])
        
        self.selected_features = features
        return X[features]

class FeatureEngineer:
    """
    Class for feature engineering and selection.
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """
        Initialize feature engineer.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """
        self.X = X
        self.y = y
        self.scaler = StandardScaler()
        
    def create_interaction_terms(self) -> pd.DataFrame:
        """
        Create interaction terms for relevant features.
        
        Returns:
            pd.DataFrame: DataFrame with added interaction terms
        """
        # Add age and smoking interaction
        if 'age' in self.X.columns and 'smoker' in self.X.columns:
            self.X['age_smoker'] = self.X['age'] * self.X['smoker']
            
        # Add bmi and smoking interaction
        if 'bmi' in self.X.columns and 'smoker' in self.X.columns:
            self.X['bmi_smoker'] = self.X['bmi'] * self.X['smoker']
            
        # Add exercise and age interaction
        if 'exercise' in self.X.columns and 'age' in self.X.columns:
            self.X['exercise_age'] = self.X['exercise'] * self.X['age']
            
        return self.X
    
    def scale_features(self) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        numerical_cols = self.X.select_dtypes(include=['float64', 'int64']).columns
        self.X[numerical_cols] = self.scaler.fit_transform(self.X[numerical_cols])
        return self.X
    
    def select_top_features(self, k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features using f_regression.
        
        Args:
            k (int): Number of features to select
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Selected features DataFrame and feature names
        """
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(self.X, self.y)
        selected_features = self.X.columns[selector.get_support()].tolist()
        
        return pd.DataFrame(X_selected, columns=selected_features), selected_features
    
    def process_features(self, create_interactions: bool = True, 
                        scale: bool = True, 
                        vif_threshold: Optional[float] = None,
                        n_select: Optional[int] = None) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline.
        
        Args:
            create_interactions (bool): Whether to create interaction terms
            scale (bool): Whether to scale features
            vif_threshold (float, optional): VIF threshold for feature selection
            n_select (int, optional): Number of top features to select
            
        Returns:
            pd.DataFrame: Processed features
        """
        if create_interactions:
            self.create_interaction_terms()
            
        if scale:
            self.scale_features()
            
        if vif_threshold is not None:
            vif_selector = VIFSelector(threshold=vif_threshold)
            self.X = vif_selector.select_features(self.X)
            
        if n_select is not None:
            self.X, selected_features = self.select_top_features(k=n_select)
            
        return self.X

def main():
    """
    Main function to demonstrate feature engineering pipeline.
    """
    try:
        # Load preprocessed data
        data = pd.read_csv('../../data/processed/processed_healthcare_costs.csv')
        X = data.drop('cost', axis=1)
        y = data['cost']
        
        # Initialize feature engineer
        engineer = FeatureEngineer(X, y)
        
        # Run feature engineering pipeline
        print("Starting feature engineering...")
        X_processed = engineer.process_features(
            create_interactions=True,
            scale=True,
            vif_threshold=5.0,
            n_select=10
        )
        
        # Save engineered features
        X_processed.to_csv('../../data/processed/engineered_features.csv', index=False)
        print("Feature engineering completed successfully!")
        
        # Print feature information
        print("\nFeature Engineering Results:")
        print(f"Original number of features: {X.shape[1]}")
        print(f"Final number of features: {X_processed.shape[1]}")
        print("\nSelected features:")
        print(X_processed.columns.tolist())
        
    except Exception as e:
        print(f"Error during feature engineering: {str(e)}")

if __name__ == "__main__":
    main()
