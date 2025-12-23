# Predicting Loan Payback - Kaggle Competition

A machine learning project for predicting whether a loan will be paid back or not (binary classification).

## Competition Details
- **Task:** Binary Classification
- **Metric:** ROC-AUC
- **Platform:** Kaggle Playground Series

## Project Structure

```
├── notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb    # Feature engineering pipeline
│   ├── 02_Model_Recommendations.md     # Model recommendations and insights
│   └── 03_LGBM_Model.ipynb             # LightGBM model training
├── data/
│   ├── train.csv                       # Training data
│   ├── test.csv                        # Test data
│   └── processed/                      # Feature-engineered datasets
├── models/
│   └── lgbm_loan_model.txt             # Trained LightGBM model
├── results/
│   └── lgbm_fe_submission_v2.csv       # Submission file
├── app.py                              # Streamlit app (to be implemented)
└── README.md
```

## Getting Started

### 1. Feature Engineering
Run `notebooks/02_Feature_Engineering.ipynb` to create feature-engineered datasets:
- Creates `data/processed/train_fe.csv`
- Creates `data/processed/test_fe.csv`

### 2. Model Training
Run `notebooks/03_LGBM_Model.ipynb` to:
- Train LightGBM model with 5-fold cross-validation
- Save the trained model to `models/lgbm_loan_model.txt`
- Generate submission file

### 3. Key Features

**Original Features:**
- Numerical: `annual_income`, `debt_to_income_ratio`, `credit_score`, `loan_amount`, `interest_rate`
- Categorical: `gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, `grade_subgrade`

**Engineered Features:**
- `grade`: Extracted from `grade_subgrade` (A-F)
- `subgrade_num`: Extracted subgrade number (1-5)
- `grade_score`: Numeric score (A=6, B=5, C=4, D=3, E=2, F=1)
- `has_stable_income`: Binary indicator for stable employment
- `is_unemployed`: Binary indicator for unemployment
- `risk_score`: Combined risk metric

## Model Performance

**Cross-Validation Results:**
- Mean AUC: ~0.9215
- Std AUC: ~0.0006

## Model Usage

Load the trained model for predictions:

```python
import lightgbm as lgb
import pandas as pd

# Load model
model = lgb.Booster(model_file='models/lgbm_loan_model.txt')

# Prepare features (same as training)
# ... apply feature engineering ...

# Predict
probability = model.predict(X)[0]  # Returns probability of loan being paid back
```

## Requirements

```
pandas
numpy
lightgbm
scikit-learn
matplotlib
seaborn
jupyter
```

## License

MIT
