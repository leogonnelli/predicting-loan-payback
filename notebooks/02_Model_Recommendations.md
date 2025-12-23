# Model Recommendations - First Strong Model

## Key Insights from EDA

### Dataset Characteristics:
- **Imbalanced**: 79.88% paid back vs 20.12% not paid back
- **Metric**: ROC-AUC (good for imbalanced data)
- **Size**: ~594k training samples, ~255k test samples
- **Features**: Mix of numerical (6) and categorical (6) features

### Strong Predictive Features:
1. **`grade_subgrade`** - ⭐⭐⭐⭐⭐ (CRITICAL)
   - Range: 60-95% payback rate
   - Clear hierarchy: A (95%) > B (93%) > C (85%) > D (72%) > E (65%) > F (63%)
   - **Recommendation**: Extract grade (A-F) and subgrade number (1-5) as separate features

2. **`employment_status`** - ⭐⭐⭐⭐⭐ (CRITICAL)
   - Retired: 99.72% payback
   - Employed/Self-employed: ~89% payback
   - Unemployed: 7.76% payback ⚠️
   - **Recommendation**: Create binary feature "has_stable_income" (Employed/Self-employed/Retired)

3. **`debt_to_income_ratio`** - ⭐⭐⭐⭐
   - Correlation: -0.34 (strongest numerical feature)
   - Clear separation between classes

4. **`credit_score`** - ⭐⭐⭐⭐
   - Correlation: 0.23
   - Higher score = higher payback probability

5. **`interest_rate`** - ⭐⭐⭐
   - Correlation: -0.13
   - Lower rate = higher payback probability

### Moderate Features:
- `loan_purpose`: 77-82% range
- `education_level`: 78-83% range

### Weak Features:
- `annual_income`: Very weak correlation (0.01)
- `loan_amount`: Negligible correlation (-0.00)
- `gender`, `marital_status`: Likely minimal impact

---

## Recommended First Model: **LightGBM**

### Why LightGBM?
1. ✅ **Excellent for tabular data** (standard in Kaggle competitions)
2. ✅ **Handles imbalanced data well** (scale_pos_weight parameter)
3. ✅ **Fast training** on large datasets (600k+ samples)
4. ✅ **Native categorical feature support** (can pass categories directly)
5. ✅ **Good defaults** - works well out of the box
6. ✅ **Feature importance** - easy to interpret

### Alternative: XGBoost or CatBoost
- **XGBoost**: Similar performance, slightly more tuning required
- **CatBoost**: Excellent with categorical features, good default handling

---

## Feature Engineering Strategy

### 1. Extract from `grade_subgrade`:
```python
# Extract grade letter (A-F) - very predictive
train_df['grade'] = train_df['grade_subgrade'].str[0]

# Extract subgrade number (1-5)
train_df['subgrade_num'] = train_df['grade_subgrade'].str[1:].astype(int)

# Optional: Create grade score (A=6, B=5, ..., F=1)
grade_map = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1}
train_df['grade_score'] = train_df['grade'].map(grade_map)
```

### 2. Employment Status Engineering:
```python
# Binary stable income indicator
train_df['has_stable_income'] = train_df['employment_status'].isin(
    ['Employed', 'Self-employed', 'Retired']
).astype(int)

# High-risk employment
train_df['is_unemployed'] = (train_df['employment_status'] == 'Unemployed').astype(int)
```

### 3. Derived Numerical Features:
```python
# Loan-to-income ratio (already identified in EDA)
train_df['loan_to_income_ratio'] = train_df['loan_amount'] / train_df['annual_income']

# Monthly payment estimate
train_df['estimated_monthly_payment'] = train_df['loan_amount'] * (train_df['interest_rate'] / 100 / 12)

# Payment burden (annual payment / annual income)
train_df['payment_burden'] = (train_df['estimated_monthly_payment'] * 12) / train_df['annual_income']

# Combined risk score (lower is better)
train_df['risk_score'] = train_df['debt_to_income_ratio'] + (train_df['interest_rate'] / 100)
```

### 4. Categorical Encoding:
- **LightGBM**: Can handle categoricals natively (recommended)
- **Alternative**: Target encoding for high-cardinality features

---

## Model Configuration

### LightGBM Parameters (Starting Point):
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
    # Handle imbalanced data
    'scale_pos_weight': 0.25  # negative_samples / positive_samples ≈ 119k/475k
}
```

### Training Strategy:
```python
# Stratified K-Fold (5 folds recommended)
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Early stopping with validation set
# Use ~20% for validation
```

---

## Expected Performance

### Baseline Expectations:
- **First model (no tuning)**: ROC-AUC ~0.85-0.90
- **With feature engineering**: ROC-AUC ~0.90-0.93
- **After hyperparameter tuning**: ROC-AUC ~0.93-0.95
- **Ensemble**: ROC-AUC ~0.95+

### Feature Importance (Expected Order):
1. `grade_subgrade` / `grade` / `grade_score`
2. `employment_status` / `has_stable_income`
3. `debt_to_income_ratio`
4. `credit_score`
5. `interest_rate`
6. Derived features (loan_to_income_ratio, payment_burden)

---

## Next Steps After First Model

1. **Analyze feature importance** - verify expectations
2. **Error analysis** - where does model fail?
3. **Hyperparameter tuning** - Optuna/Hyperopt
4. **Try other models** - XGBoost, CatBoost, Neural Networks
5. **Ensemble** - blend multiple models
6. **Advanced feature engineering** - interactions, polynomial features

---

## Quick Start Code Structure

```python
# 1. Feature Engineering
# 2. Prepare features (separate numerical/categorical)
# 3. Set up cross-validation
# 4. Train LightGBM with early stopping
# 5. Predict on test set
# 6. Submit predictions
```

