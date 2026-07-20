"""
Insurance Fraud Detection System
Developed by: Mehfooz Mehmood Khan
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ========================================================
# Configuration & Helper Functions
# ========================================================

st.set_page_config(page_title="Insurance Fraud Detection", layout="centered")

@st.cache_resource
def load_models():
    """Loads the pre-trained machine learning model and encoders."""
    try:
        model = joblib.load('decision_tree_model.pkl')
        encoders = joblib.load('label_encoders (2).pkl')
        return model, encoders
    except FileNotFoundError as e:
        st.error(f"Missing required file: {e.filename}. Please ensure all .pkl files are in the directory.")
        st.stop()

@st.cache_data
def load_data():
    """Loads the sample dataset for random claim generation."""
    try:
        df = pd.read_csv('insurance_claims (1).csv')
        return df
    except FileNotFoundError:
        st.error("Missing dataset: 'insurance_claims (1).csv'. Please ensure it is in the directory.")
        st.stop()

# Define feature sets
CATEGORICAL_FEATURES = [
    'policy_state', 'policy_csl', 'insured_sex', 'insured_education_level',
    'insured_occupation', 'insured_hobbies', 'insured_relationship',
    'incident_type', 'collision_type', 'incident_severity',
    'authorities_contacted', 'incident_state', 'incident_city',
    'property_damage', 'police_report_available', 'auto_make', 'auto_model'
]

NUMERICAL_FEATURES = [
    'months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium',
    'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
    'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
    'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim',
    'auto_year', 'days_to_incident'
]

ALL_FEATURES = (
    ['months_as_customer', 'age', 'policy_state', 'policy_csl', 'policy_deductable', 
     'policy_annual_premium', 'umbrella_limit', 'insured_sex', 'insured_education_level', 
     'insured_occupation', 'insured_hobbies', 'insured_relationship', 'capital-gains', 
     'capital-loss', 'incident_type', 'collision_type', 'incident_severity', 
     'authorities_contacted', 'incident_state', 'incident_city', 'incident_hour_of_the_day', 
     'number_of_vehicles_involved', 'property_damage', 'bodily_injuries', 'witnesses', 
     'police_report_available', 'total_claim_amount', 'injury_claim', 'property_claim', 
     'vehicle_claim', 'auto_make', 'auto_model', 'auto_year', 'days_to_incident']
)

def main():
    # Load required assets
    model, label_encoders = load_models()
    df = load_data()

    # ========================================================
    # SECTION 1: Header
    # ========================================================
    st.title("🛡 Insurance Fraud Detection System")
    st.subheader("Machine Learning Based Insurance Claim Fraud Prediction")
    st.markdown("---")

    # ========================================================
    # SECTION 2: Project Overview
    # ========================================================
    st.write("### Project Overview")
    st.write(
        "Insurance fraud costs the industry billions annually. This application uses a Machine Learning "
        "model trained on historical data to automatically flag potentially fraudulent claims, "
        "helping to reduce financial losses and streamline the investigation process."
    )
    st.markdown("---")

    # ========================================================
    # SECTION 3: Dataset Summary
    # ========================================================
    st.write("### Dataset Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", "1000")
        st.metric("Target Variable", "Fraud Reported")
    with col2:
        st.metric("Total Features", "34")
        st.metric("Selected Model", "Decision Tree")
        
    st.markdown("---")

    # ========================================================
    # SECTION 4: Model Comparison Dashboard
    # ========================================================
    st.write("### Model Comparison Dashboard")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
        accuracies = [73, 80, 79]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax.bar(models, accuracies, color=colors)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        
        # Display percentage values above each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval}%", ha='center', va='bottom', fontweight='bold')
            
        st.pyplot(fig)
        
    with col2:
        st.info(
            "🏆 **Best Model**\n\n"
            "**Decision Tree**\n\n"
            "**Accuracy :** 80%\n\n"
            "**Precision :** 0.57\n\n"
            "**Recall :** 0.71\n\n"
            "**F1 Score :** 0.64"
        )

    st.markdown("---")

    # ========================================================
    # SECTION 5: Load Sample & Prediction Form
    # ========================================================
    st.write("### Claim Details Input")
    
    # Initialize session state for claim data
    if 'current_claim' not in st.session_state:
        st.session_state.current_claim = df.sample(1).iloc[0].to_dict()

    if st.button("📄 Load Random Claim"):
        st.session_state.current_claim = df.sample(1).iloc[0].to_dict()

    st.write("Review or modify the features below before running the prediction.")
    
    user_input = {}
    col1, col2 = st.columns(2)
    
    # Dynamically distribute feature inputs across two columns
    for i, feature in enumerate(ALL_FEATURES):
        target_col = col1 if i % 2 == 0 else col2
        current_val = st.session_state.current_claim.get(feature, None)
        
        with target_col:
            if feature in CATEGORICAL_FEATURES:
                options = list(df[feature].dropna().unique())
                if current_val not in options and pd.notna(current_val):
                    options.append(current_val)
                    
                index = options.index(current_val) if pd.notna(current_val) else 0
                user_input[feature] = st.selectbox(feature.replace('_', ' ').title(), options=options, index=index)
            else:
                default_val = float(current_val) if pd.notna(current_val) else 0.0
                user_input[feature] = st.number_input(feature.replace('_', ' ').title(), value=default_val)

    st.markdown("---")

    # ========================================================
    # SECTION 6 & 7: Prediction Results & Explanation
    # ========================================================
    if st.button("🚀 Predict Fraud", type="primary"):
        st.write("## Prediction Results")
        
        input_df = pd.DataFrame([user_input])
        
        try:
            # Safely encode categorical features
            for col in CATEGORICAL_FEATURES:
                if hasattr(label_encoders, 'get') and col in label_encoders:
                    le = label_encoders[col]
                    val = input_df[col].iloc[0]
                    
                    if val in le.classes_:
                        input_df[col] = int(le.transform([val])[0])
                    else:
                        input_df[col] = 0
                else:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
                
            # Force all numeric types before passing to the model
            input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            
            # Ensure correct column order for model
            ordered_df = input_df[ALL_FEATURES] 
            
            # Generate final prediction
            prediction = model.predict(ordered_df)
            

            # Display Result
            if prediction[0] == 1:
                st.error("## ❌ FRAUD")
            else:
                st.success("## ✅ NON-FRAUD")
                
            # Explanation
            st.write("#### Explanation")
            
            if prediction[0] == 1:
                st.warning("This insurance claim contains characteristics similar to previously identified fraudulent claims. Further manual investigation is recommended.")
            else:
                st.info("This insurance claim appears consistent with genuine claims based on the trained machine learning model.")
            
            # ========================================================
            # SECTION 8: Prediction Summary
            # ========================================================
            # Prediction Summary
            st.subheader("Prediction Summary")
            
            if prediction[0] == 1:
                st.warning(
                    """
                    The model predicts that this insurance claim is **likely to be Fraud**
                    based on the provided claim information.
            
                    This prediction is intended to assist decision-making and should be
                    verified through further investigation before taking any action.
                    """
                )
            else:
                st.info(
                    """
                    The model predicts that this insurance claim is **likely to be Non-Fraud**
                    based on the provided claim information.
            
                    This prediction is intended to assist decision-making and should not be
                    considered a final verdict without expert review.
                    """
                )

             st.markdown("---")

    # ========================================================
    # SECTION 9: About Section
    # ========================================================
    st.write("### About")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model Information**")
        st.write("- Model: Decision Tree Classifier")
        st.write("- Accuracy: 80%")
    with col2:
        st.write("**Libraries Used**")
        st.write("- Python\n- Pandas\n- NumPy\n- Scikit-learn\n- Streamlit")

    

if __name__ == "__main__":
    main()
