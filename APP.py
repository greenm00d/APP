import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('XGBoost.pkl')

# Define feature options
embryos = {    
    0: 'no (1)',    
    1: 'yes (2)'
}

# Define feature options
biochemical = {    
    0: 'no (1)',    
    1: 'yes (2)'
}

# Define feature names
feature_names = [    
    "age", "infertility_time", "menarche_age", "AMH", "gn_dose", "gn_days", "oocyte", "embryos", "biochemical"
]

# Streamlit user interface
st.title("Heart Disease Predictor")

# age: numerical input
age = st.number_input("Age:", min_value=1, max_value=120, value=50)

# age: numerical input
infertility_time = st.number_input("infertility_time:", min_value=1, max_value=120, value=50)

# age: numerical input
menarche_age = st.number_input("menarche_age:", min_value=1, max_value=120, value=50)

# age: numerical input
AMH = st.number_input("AMH:", min_value=1, max_value=120, value=50)

# age: numerical input
gn_dose = st.number_input("gn_dose:", min_value=1, max_value=120, value=50)

# age: numerical input
gn_days = st.number_input("gn_days:", min_value=1, max_value=120, value=50)

# age: numerical input
oocyte = st.number_input("oocyte:", min_value=1, max_value=120, value=50)

# cp: categorical selection
embryos = st.selectbox("embryos:", options=list(embryos.keys()), format_func=lambda x: embryos[x])

# cp: categorical selection
biochemical = st.selectbox("biochemical:", options=list(biochemical.keys()), format_func=lambda x: biochemical[x])

# Process inputs and make predictions
feature_values = [age, infertility_time, menarche_age, AMH, gn_dose, gn_days, oocyte, embryos, biochemical]
features = np.array([feature_values])

if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(features)[0]    
    predicted_proba = model.predict_proba(features)[0]
    
    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}")    
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    
    # Generate advice based on prediction results    
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1:        
        advice = (            
            f"According to our model, you have a high risk of heart disease. "            
            f"The model predicts that your probability of having heart disease is {probability:.1f}%. "            
            "While this is just an estimate, it suggests that you may be at significant risk. "            
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "            
            "to ensure you receive an accurate diagnosis and necessary treatment."        
        )
    else:        
        advice = (            
            f"According to our model, you have a low risk of heart disease. "            
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "            
            "However, maintaining a healthy lifestyle is still very important. "            
            "I recommend regular check-ups to monitor your heart health, "            
            "and to seek medical advice promptly if you experience any symptoms."        
        )
    st.write(advice)
    
    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    
    st.image("shap_force_plot.png")
