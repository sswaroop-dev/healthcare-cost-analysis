# Source Code Documentation

## Directory Structure

```
src/
├── __init__.py                 # Package initialization and pipeline setup
├── data/                       # Data handling modules
│   ├── __init__.py
│   └── preprocessing.py        # Data preprocessing utilities
│
├── features/                   # Feature engineering modules
│   ├── __init__.py
│   └── feature_engineering.py  # Feature creation and selection
│
├── models/                     # Model training and evaluation
│   ├── __init__.py
│   ├── train.py               # Model training utilities
│   └── evaluate.py            # Model evaluation utilities
│
├── visualization/             # Visualization modules
│   ├── __init__.py
│   ├── eda_plots.py          # Exploratory data analysis plots
│   └── model_plots.py        # Model visualization utilities
│
└── utils/                    # Utility functions
    ├── __init__.py
    └── helpers.py           # Common helper functions
```

## Module Descriptions

### 1. Data Module (`data/`)
- **Purpose**: Handle all data preprocessing tasks
- **Key Components**:
  - `DataPreprocessor`: Class for data cleaning and preparation
  - Handles missing values
  - Performs categorical encoding
  - Manages data type conversions

### 2. Features Module (`features/`)
- **Purpose**: Feature engineering and selection
- **Key Components**:
  - `FeatureEngineer`: Creates and transforms features
  - `VIFSelector`: Handles multicollinearity using VIF
  - Implements feature selection methods
  - Creates interaction terms

### 3. Models Module (`models/`)
- **Purpose**: Model training and evaluation
- **Key Components**:
  - `ModelTrainer`: Trains various regression models
    - Random Forest
    - Gradient Boosting
  - `ModelEvaluator`: Evaluates model performance
    - Calculates metrics
    - Performs model comparison
    - Generates SHAP analysis

### 4. Visualization Module (`visualization/`)
- **Purpose**: Create data and model visualizations
- **Key Components**:
  - `EDAVisualizer`: Exploratory data analysis plots
  - `ModelVisualizer`: Model-specific visualizations
  - Supports various plot types and interactive visualizations

### 5. Utils Module (`utils/`)
- **Purpose**: Common utility functions
- **Key Components**:
  - File operations
  - Logging setup
  - Directory management
  - Data validation
  - Performance monitoring

## Usage Examples

### Basic Usage
```python
from src.data import DataPreprocessor
from src.features import FeatureEngineer
from src.models import ModelTrainer, ModelEvaluator

# Initialize preprocessor
preprocessor = DataPreprocessor('data/raw/healthcare_costs.csv')
processed_data = preprocessor.preprocess_data()

# Feature engineering
engineer = FeatureEngineer(processed_data)
features = engineer.process_features()

# Train models
trainer = ModelTrainer(features)
trainer.train_all_models()
```

### Running Complete Pipeline
```python
from src import run_pipeline

# Run entire analysis pipeline
results = run_pipeline('data/raw/healthcare_costs.csv')
```

## Key Features

1. **Modular Design**
   - Each component is independent and reusable
   - Clear separation of concerns
   - Easy to extend and modify

2. **Comprehensive Documentation**
   - Detailed docstrings
   - Type hints
   - Usage examples

3. **Error Handling**
   - Robust error checking
   - Informative error messages
   - Logging support

4. **Performance Optimization**
   - Efficient data processing
   - Memory management
   - Progress tracking

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- shap
- plotly
