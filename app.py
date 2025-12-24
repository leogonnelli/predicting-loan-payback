import streamlit as st
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Loan Payback Prediction",
    page_icon="💰",
    layout="wide"
)

# Title
st.title("🏦 Loan Payback Probability Predictor")
st.markdown("Enter the client's information below to predict the probability of loan repayment.")

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained Lasso Logistic Regression model"""
    model_path = 'models/lasso_logistic_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the model is trained.")
        return None
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

# Feature engineering function (same as in notebooks)
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to match training pipeline"""
    out = df.copy()
    
    # Extract grade and subgrade
    out['grade'] = out['grade_subgrade'].str[0]
    out['subgrade_num'] = out['grade_subgrade'].str[1:].astype(int)
    grade_map = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1}
    out['grade_score'] = out['grade'].map(grade_map)
    
    # Employment indicators
    out['has_stable_income'] = out['employment_status'].isin(['Employed', 'Self-employed', 'Retired']).astype(int)
    out['is_unemployed'] = (out['employment_status'] == 'Unemployed').astype(int)
    
    # Risk score
    out['risk_score'] = out['debt_to_income_ratio'] + (out['interest_rate'] / 100)
    
    return out

def prepare_prediction_input(user_inputs, model_data):
    """Prepare user inputs for prediction with feature engineering and preprocessing"""
    df = pd.DataFrame([user_inputs])
    df = add_features(df)
    
    # Get feature columns from model_data (same order as training)
    categorical_features = model_data['categorical_features']
    numerical_features = model_data['numerical_features']
    feature_cols = categorical_features + numerical_features
    
    X = df[feature_cols].copy()
    
    # Apply preprocessing pipeline (OneHotEncoder + StandardScaler)
    preprocessor = model_data['preprocessor']
    X_processed = preprocessor.transform(X)
    
    return X_processed

# Main form
with st.form("loan_prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Financial Information")
        
        annual_income = st.number_input(
            "Annual Income ($)",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            help="Client's annual income in dollars"
        )
        
        debt_to_income_ratio = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01,
            format="%.2f",
            help="Ratio of monthly debt payments to monthly income (0-1)"
        )
        
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=700,
            step=10,
            help="Client's credit score (typically 300-850)"
        )
        
        loan_amount = st.number_input(
            "Loan Amount ($)",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
            help="Requested loan amount in dollars"
        )
        
        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=12.5,
            step=0.1,
            format="%.2f",
            help="Annual interest rate percentage"
        )
        
        grade_subgrade = st.selectbox(
            "Loan Grade & Subgrade",
            options=['A1', 'A2', 'A3', 'A4', 'A5',
                    'B1', 'B2', 'B3', 'B4', 'B5',
                    'C1', 'C2', 'C3', 'C4', 'C5',
                    'D1', 'D2', 'D3', 'D4', 'D5',
                    'E1', 'E2', 'E3', 'E4', 'E5',
                    'F1', 'F2', 'F3', 'F4', 'F5'],
            index=12,  # Default to C3
            help="Loan risk grade assigned by the lender"
        )
    
    with col2:
        st.subheader("👤 Personal Information")
        
        gender = st.selectbox(
            "Gender",
            options=['Female', 'Male', 'Other']
        )
        
        marital_status = st.selectbox(
            "Marital Status",
            options=['Single', 'Married', 'Divorced', 'Widowed']
        )
        
        education_level = st.selectbox(
            "Education Level",
            options=['High School', "Bachelor's", "Master's", 'PhD', 'Other']
        )
        
        employment_status = st.selectbox(
            "Employment Status",
            options=['Employed', 'Self-employed', 'Retired', 'Student', 'Unemployed'],
            help="Current employment situation"
        )
        
        loan_purpose = st.selectbox(
            "Loan Purpose",
            options=['Debt consolidation', 'Home', 'Car', 'Business', 
                    'Education', 'Medical', 'Vacation', 'Other']
        )
    
    # Submit button
    submitted = st.form_submit_button("🔮 Predict Loan Payback Probability", use_container_width=True)

# Process prediction when form is submitted
if submitted:
    # Collect all inputs
    user_inputs = {
        'annual_income': annual_income,
        'debt_to_income_ratio': debt_to_income_ratio,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'grade_subgrade': grade_subgrade,
        'gender': gender,
        'marital_status': marital_status,
        'education_level': education_level,
        'employment_status': employment_status,
        'loan_purpose': loan_purpose
    }
    
    # Load model
    model_data = load_model()
    
    if model_data is not None:
        try:
            # Prepare features with preprocessing
            X_processed = prepare_prediction_input(user_inputs, model_data)
            
            # Get model from model_data
            model = model_data['model']
            
            # Make prediction (predict_proba returns [prob_class_0, prob_class_1])
            probability = model.predict_proba(X_processed)[0, 1]
            probability_percent = probability * 100
            
            # Display results
            st.markdown("---")
            st.subheader("📈 Prediction Results")
            
            # Create columns for result display
            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
            
            with result_col2:
                # Color-coded result box
                if probability_percent >= 70:
                    color = "🟢"
                    risk_level = "LOW RISK"
                    recommendation = "✅ **Recommendation: APPROVE**"
                elif probability_percent >= 50:
                    color = "🟡"
                    risk_level = "MODERATE RISK"
                    recommendation = "⚠️ **Recommendation: REVIEW CAREFULLY**"
                else:
                    color = "🔴"
                    risk_level = "HIGH RISK"
                    recommendation = "❌ **Recommendation: REJECT or REQUIRE ADDITIONAL CONDITIONS**"
                
                # Main probability display
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; border-radius: 10px; 
                            background-color: {'#e8f5e9' if probability_percent >= 70 else '#fff3e0' if probability_percent >= 50 else '#ffebee'}; 
                            border: 2px solid {'#4caf50' if probability_percent >= 70 else '#ff9800' if probability_percent >= 50 else '#f44336'};'>
                    <h2 style='margin: 0; color: {'#2e7d32' if probability_percent >= 70 else '#e65100' if probability_percent >= 50 else '#c62828'};'>
                        {color} {risk_level}
                    </h2>
                    <h1 style='margin: 10px 0; font-size: 3em; color: {'#1b5e20' if probability_percent >= 70 else '#bf360c' if probability_percent >= 50 else '#b71c1c'};'>
                        {probability_percent:.2f}%
                    </h1>
                    <p style='font-size: 1.1em; margin: 0;'>
                        Probability of Loan Repayment
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("")
                st.markdown(f"### {recommendation}")
                
                # Additional details
                with st.expander("📋 View Detailed Information"):
                    st.markdown(f"""
                    **Financial Summary:**
                    - Annual Income: ${annual_income:,.2f}
                    - Loan Amount: ${loan_amount:,.2f}
                    - Interest Rate: {interest_rate:.2f}%
                    - Debt-to-Income Ratio: {debt_to_income_ratio:.2%}
                    - Credit Score: {credit_score}
                    
                    **Personal Information:**
                    - Gender: {gender}
                    - Marital Status: {marital_status}
                    - Education: {education_level}
                    - Employment: {employment_status}
                    - Loan Purpose: {loan_purpose}
                    - Loan Grade: {grade_subgrade}
                    
                    **Prediction Confidence:**
                    - Repayment Probability: {probability_percent:.4f}%
                    - Default Risk: {100 - probability_percent:.2f}%
                    """)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application uses a trained **Lasso Logistic Regression** model to predict 
    the probability that a client will pay back their loan.
    
    **Model Characteristics:**
    - **Type**: Lasso (L1-regularized) Logistic Regression
    - **CV AUC**: ~0.85-0.90 (varies based on regularization)
    - **C Parameter**: Optimized via cross-validation
    
    **Advantages:**
    - ✅ **Interpretable**: Clear coefficient relationships
    - ✅ **Intuitive**: Monotonic relationships (e.g., higher interest → lower probability)
    - ✅ **Fast**: Quick predictions for real-time use
    
    ### How to Use:
    1. Fill in all client information fields
    2. Click "Predict Loan Payback Probability"
    3. Review the risk assessment and recommendation
    
    ### Risk Categories:
    - 🟢 **Low Risk** (≥70%): High probability of repayment
    - 🟡 **Moderate Risk** (50-70%): Requires careful review
    - 🔴 **High Risk** (<50%): Low probability of repayment
    """)
    
    st.markdown("---")
    st.markdown("**Note**: This is a predictive tool. Final loan decisions should consider additional factors and regulatory requirements.")


