# Healthcare Cost Analysis and Prediction 🏥💰

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📊 Project Overview

A comprehensive analysis and prediction system for healthcare costs using machine learning. This project analyzes various factors affecting healthcare expenses and provides predictive models to estimate future costs.

### 🎯 Key Objectives

- Analyze patterns in healthcare costs
- Identify key cost drivers
- Develop accurate prediction models
- Provide actionable insights for cost optimization

## 🗂️ Project Structure

```
healthcare-cost-analysis/
├── data/                      # Data directory
│   ├── README_data.md        # Data documentation
│   └── healthcare_costs.csv  # Dataset
│
├── src/                      # Source code
│   ├── data/                # Data processing
│   ├── features/            # Feature engineering
│   ├── models/             # ML models
│   ├── utils/              # Utilities
│   └── visualization/      # Visualization tools
│
├── LICENSE                  # License file
└── README.md               # Project documentation
```

## 🔍 Features

### Data Analysis
- Comprehensive data preprocessing
- Missing value handling
- Categorical variable encoding
- Feature correlation analysis

### Machine Learning Models
- Random Forest Regression
- Gradient Boosting
- Model performance comparison
- Feature importance analysis

### Visualizations
- Interactive dashboards
- Cost distribution analysis
- Geographic cost variations
- Model performance metrics

## 🛠️ Technologies Used

- **Python** 3.8+
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Model Interpretation**: SHAP

## 📈 Key Findings

1. **Cost Drivers**
   - Smoking status is the strongest predictor
   - Age has significant impact
   - Geographic location influences costs

2. **Model Performance**
   - Gradient Boosting shows highest accuracy
   - Feature importance reveals key factors
   - Model validates on unseen data

## 🚀 Getting Started

### Prerequisites
```bash
python -m pip install --upgrade pip
```

### Installation
```bash
# Clone the repository
git clone https://github.com/[username]/healthcare-cost-analysis.git

# Navigate to project directory
cd healthcare-cost-analysis

# Install dependencies
pip install -r requirements.txt
```

### Usage
```python
from src import run_pipeline

# Run the complete analysis
results = run_pipeline('data/healthcare_costs.csv')
```

## 📝 Analysis Steps

1. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Feature scaling

2. **Exploratory Analysis**
   - Distribution analysis
   - Correlation studies
   - Geographic patterns

3. **Model Development**
   - Feature engineering
   - Model training
   - Performance evaluation

4. **Results Interpretation**
   - SHAP analysis
   - Feature importance
   - Error analysis

## 💡 Applications

- Healthcare cost prediction
- Risk factor identification
- Resource allocation optimization
- Policy decision support

## 📈 Results

The analysis revealed that:
- Smoking increases healthcare costs by approximately 12,000 units
- Age and BMI show strong correlation with costs
- Geographic location impacts healthcare expenses
- Model achieves 57% accuracy in cost prediction

## 🤝 Contributing

We welcome contributions! Please read our CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
