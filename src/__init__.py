"""
Healthcare Cost Analysis Package

This package provides tools for analyzing healthcare costs using machine learning.
"""

from . import data
from . import features
from . import models
from . import visualization
from . import utils

# Package metadata
__version__ = '1.0.0'
__author__ = 'sswaroop-dev'
__description__ = 'Healthcare cost analysis and prediction tools'

# Import key classes for easier access
from .data.preprocessing import DataPreprocessor
from .features.feature_engineering import FeatureEngineer, VIFSelector
from .models.train import ModelTrainer
from .models.evaluate import ModelEvaluator
from .visualization.eda_plots import EDAVisualizer
from .visualization.model_plots import ModelVisualizer
from .visualization.report_plots import ReportVisualizer

# Define public interface
__all__ = [
    # Main modules
    'data',
    'features',
    'models',
    'visualization',
    'utils',
    
    # Key classes
    'DataPreprocessor',
    'FeatureEngineer',
    'VIFSelector',
    'ModelTrainer',
    'ModelEvaluator',
    'EDAVisualizer',
    'ModelVisualizer',
    'ReportVisualizer',
    
    # Metadata
    '__version__',
    '__author__',
    '__author_email__',
    '__description__'
]

# Set default logging configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

def get_version():
    """Return the package version."""
    return __version__

def setup():
    """
    Initialize the package with default configuration.
    
    This function sets up:
    - Logging configuration
    - Default directories
    - Basic configuration
    """
    from .utils.helpers import setup_logging, create_directory
    
    # Setup logging
    setup_logging()
    
    # Create necessary directories
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'reports/figures',
        'logs'
    ]
    
    for directory in directories:
        create_directory(f'../../{directory}')
        
    logging.info(f"Healthcare Cost Analysis Package v{__version__} initialized")
    
def run_pipeline(data_path: str):
    """
    Run the complete analysis pipeline.
    
    Args:
        data_path (str): Path to the raw data file
    """
    try:
        logging.info("Starting analysis pipeline...")
        
        # Data preprocessing
        preprocessor = DataPreprocessor(data_path)
        processed_data = preprocessor.preprocess_data()
        X, y = preprocessor.get_feature_target_split()
        
        # Feature engineering
        engineer = FeatureEngineer(X, y)
        features = engineer.process_features()
        
        # Model training
        trainer = ModelTrainer(features, y)
        trainer.setup_all_models()
        trainer.train_all_models()
        
        # Model evaluation
        evaluator = ModelEvaluator(
            trainer.trained_models,
            trainer.X_test,
            trainer.y_test
        )
        results = evaluator.evaluate_all_models()
        
        # Visualizations
        eda_viz = EDAVisualizer(processed_data)
        eda_viz.create_eda_dashboard()
        
        model_viz = ModelVisualizer(
            trainer.trained_models,
            trainer.X_test,
            trainer.y_test
        )
        
        for model_name in trainer.trained_models:
            model_viz.create_model_dashboard(model_name)
            
        logging.info("Analysis pipeline completed successfully!")
        return results
        
    except Exception as e:
        logging.error(f"Error in analysis pipeline: {str(e)}")
        raise
