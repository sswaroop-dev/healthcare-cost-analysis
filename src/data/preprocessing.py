"""
Data preprocessing utilities for healthcare cost analysis.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

class DataPreprocessor:
    """
    A class to handle all data preprocessing tasks for healthcare cost analysis.
    """
    def __init__(self, file_path: str):
        """
        Initialize the preprocessor with the data file path.
        
        Args:
            file_path (str): Path to the healthcare cost dataset
        """
        self.file_path = file_path
        self.data = None
        self.numeric_columns = ['age', 'bmi', 'children', 'hypertension', 'cost']
        self.categorical_columns = ['smoker', 'location', 'location_type', 
                                  'education_level', 'yearly_physical', 
                                  'exercise', 'married', 'gender']
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from the specified file path.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            self.data = pd.read_csv(self.file_path)
            # Drop the 'X' column as it's just an identifier
            if 'X' in self.data.columns:
                self.data = self.data.drop('X', axis=1)
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the data file at {self.file_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def check_missing_values(self) -> pd.Series:
        """
        Check for missing values in the dataset.
        
        Returns:
            pd.Series: Count of missing values per column
        """
        return self.data.isnull().sum()
    
    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Returns:
            pd.DataFrame: Dataset with handled missing values
        """
        # Convert specific columns to numeric
        self.data['bmi'] = pd.to_numeric(self.data['bmi'], errors='coerce')
        self.data['hypertension'] = pd.to_numeric(self.data['hypertension'], errors='coerce')
        
        # Handle missing values for numeric columns
        for col in self.numeric_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        # Handle missing values for categorical columns
        for col in self.categorical_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        
        return self.data
    
    def encode_categorical_variables(self) -> pd.DataFrame:
        """
        Encode categorical variables in the dataset.
        
        Returns:
            pd.DataFrame: Dataset with encoded categorical variables
        """
        encoding_map = {
            "smoker": {"yes": 1, "no": 0},
            "location_type": {"Urban": 1, "Country": 0},
            "yearly_physical": {"Yes": 1, "No": 0},
            "exercise": {"Active": 1, "Not-Active": 0},
            "married": {"Married": 1, "Not_Married": 0},
            "gender": {"male": 1, "female": 0}
        }
        
        # Apply encoding for specified categorical variables
        for column, mapping in encoding_map.items():
            if column in self.data.columns:
                self.data[column] = self.data[column].map(mapping)
        
        # One-hot encode location and education_level
        if 'location' in self.data.columns:
            location_dummies = pd.get_dummies(self.data['location'], prefix='location')
            self.data = pd.concat([self.data, location_dummies], axis=1)
            self.data.drop('location', axis=1, inplace=True)
            
        if 'education_level' in self.data.columns:
            education_dummies = pd.get_dummies(self.data['education_level'], prefix='education')
            self.data = pd.concat([self.data, education_dummies], axis=1)
            self.data.drop('education_level', axis=1, inplace=True)
        
        return self.data
    
    def validate_data(self) -> bool:
        """
        Validate the processed data for any anomalies.
        
        Returns:
            bool: True if validation passes, raises exception otherwise
        """
        # Check for any remaining missing values
        if self.data.isnull().any().any():
            raise ValueError("Data still contains missing values after preprocessing")
            
        # Check for invalid values in numeric columns
        for col in self.numeric_columns:
            if col in self.data.columns:
                if (self.data[col] < 0).any():
                    raise ValueError(f"Negative values found in {col}")
        
        return True
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline.
        
        Returns:
            pd.DataFrame: Fully preprocessed dataset
        """
        self.load_data()
        self.handle_missing_values()
        self.encode_categorical_variables()
        self.validate_data()
        return self.data
    
    def get_feature_target_split(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split the dataset into features and target variable.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features (X) and target variable (y)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call preprocess_data() first.")
            
        X = self.data.drop('cost', axis=1)
        y = self.data['cost']
        
        return X, y

def main():
    """
    Main function to demonstrate the preprocessing pipeline.
    """
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor('../../data/healthcare_costs.csv')
        
        # Run preprocessing pipeline
        print("Starting data preprocessing...")
        processed_data = preprocessor.preprocess_data()
        
        # Get feature and target split
        X, y = preprocessor.get_feature_target_split()
        
        # Save processed data
        processed_data.to_csv('../../data/processed/processed_healthcare_costs.csv', index=False)
        print("Preprocessing completed successfully!")
        
        # Print basic statistics
        print("\nDataset Statistics:")
        print(f"Number of samples: {len(processed_data)}")
        print(f"Number of features: {len(X.columns)}")
        print("\nFeature names:")
        print(X.columns.tolist())
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")

if __name__ == "__main__":
    main()
