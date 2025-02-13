"""
Helper functions for healthcare cost analysis project.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime

def load_config(config_path: str = '../../config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        return {}

def setup_logging(log_path: str = '../../logs') -> None:
    """
    Setup logging configuration.
    
    Args:
        log_path (str): Path to store log files
    """
    create_directory(log_path)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_path, f'healthcare_analysis_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_directory(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path (str): Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)

def format_currency(amount: float) -> str:
    """
    Format amount as currency string.
    
    Args:
        amount (float): Amount to format
        
    Returns:
        str: Formatted currency string
    """
    return f"${amount:,.2f}"

def validate_numeric_column(data: pd.DataFrame, column: str) -> bool:
    """
    Validate if a column contains valid numeric data.
    
    Args:
        data (pd.DataFrame): DataFrame to check
        column (str): Column name to validate
        
    Returns:
        bool: True if column is valid numeric
    """
    try:
        pd.to_numeric(data[column])
        return True
    except (ValueError, TypeError):
        return False

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value (float): Original value
        new_value (float): New value
        
    Returns:
        float: Percentage change
    """
    try:
        return ((new_value - old_value) / abs(old_value)) * 100
    except ZeroDivisionError:
        return np.inf if new_value > 0 else -np.inf

def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    Validate if date range is valid.
    
    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        bool: True if date range is valid
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        return start <= end
    except ValueError:
        return False

def get_file_paths(directory: str, extension: str = '.csv') -> list:
    """
    Get all files with specific extension in directory.
    
    Args:
        directory (str): Directory to search
        extension (str): File extension to filter
        
    Returns:
        list: List of file paths
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) 
            if f.endswith(extension)]

def save_dataframe(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """
    Save DataFrame with proper directory creation.
    
    Args:
        df (pd.DataFrame): DataFrame to save
        filepath (str): Path to save file
        index (bool): Whether to save index
    """
    directory = os.path.dirname(filepath)
    create_directory(directory)
    df.to_csv(filepath, index=index)

def load_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame with error handling.
    
    Args:
        filepath (str): Path to load file from
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"Empty file: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading file {filepath}: {str(e)}")
        raise

class Timer:
    """Simple timer class for performance monitoring."""
    
    def __init__(self, name: str = ''):
        """Initialize timer with optional name."""
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        return self
        
    def __exit__(self, *args):
        """End timing and print duration."""
        duration = datetime.now() - self.start_time
        logging.info(f"{self.name} took {duration}")

def main():
    """
    Main function to demonstrate utility functions.
    """
    try:
        # Setup logging
        setup_logging()
        logging.info("Testing utility functions...")
        
        # Test directory creation
        test_dir = '../../test'
        create_directory(test_dir)
        logging.info(f"Created directory: {test_dir}")
        
        # Test currency formatting
        amount = 1234.5678
        formatted = format_currency(amount)
        logging.info(f"Formatted currency: {formatted}")
        
        # Test timer
        with Timer("Test operation"):
            # Simulate some work
            pd.DataFrame(np.random.randn(1000, 1000))
            
        logging.info("Utility functions test completed successfully!")
        
    except Exception as e:
        logging.error(f"Error testing utilities: {str(e)}")

if __name__ == "__main__":
    main()
