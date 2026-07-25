import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# App Title and Description
st.title("🏦 Loan Eligibility Predictor")
st.write("Determine loan approval status using Machine Learning.")
st.divider()

# Load Model & Encoders (Safely using Try/Except block)
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("random_model.pkl")
        gender_encoder = joblib.load("gender_encoder.pkl")
        married_encoder = joblib.load("married_encoder.pkl")
        education_encoder = joblib.load("education_encoder.pkl")
        self_emp_encoder = joblib.load("self_employed_encoder.pkl")
        property_encoder = joblib.load("property_area_encoder.pkl")
        return model, gender_encoder, married_encoder, education_encoder, self_emp_encoder, property_encoder
    except FileNotFoundError:
        return None, None, None, None, None, None

model, gender_le, married_le, edu_le, self_emp_le, prop_le = load_assets()

# Fallback options if encoder files are missing during initial testing
gender_options = gender_le.classes_ if gender_le else ["Male", "Female"]
married_options = married_le.classes_ if married_le else ["No", "Yes"]
education_options = edu_le.classes_ if edu_le else ["Graduate", "Not Graduate"]
self_emp_options = self_emp_le.classes_ if self_emp_le else ["No", "Yes"]
property_options = prop_le.classes_ if prop_le else ["Urban", "Semiurban", "Rural"]

# --- USER INPUT FORM ---
st.subheader("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", gender_options)
    married = st.selectbox("Marital Status", married_options)
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education Level", education_options)

with col2:
    self_employed = st.selectbox("Self Employed?", self_emp_options)
    applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0, step=500)
    property_area = st.selectbox("Property Area", property_options)

st.divider()

st.subheader("Loan Details")
col3, col4 = st.columns(2)

with col3:
    loan_amount = st.number_input("Loan Amount (in Thousands)", min_value=1.0, value=150.0, step=5.0)
    loan_term = st.selectbox("Loan Term (Months)", [360, 180, 240, 120, 84, 60], index=0)

with col4:
    credit_history = st.selectbox("Credit History Meets Guidelines?", ["Yes (1.0)", "No (0.0)"])
    credit_history_val = 1.0 if credit_history.startswith("Yes") else 0.0

st.divider()

# --- PREDICTION LOGIC ---
if st.button("Predict Loan Status", type="primary", use_container_width=True):
    
    # Encode categorical inputs
    encoded_gender = gender_le.transform([gender])[0] if gender_le else (1 if gender == "Male" else 0)
    encoded_married = married_le.transform([married])[0] if married_le else (1 if married == "Yes" else 0)
    encoded_education = edu_le.transform([education])[0] if edu_le else (0 if education == "Graduate" else 1)
    encoded_self_emp = self_emp_le.transform([self_employed])[0] if self_emp_le else (1 if self_employed == "Yes" else 0)
    encoded_property = prop_le.transform([property_area])[0] if prop_le else 0
    
    # Handle dependents encoding ('3+' becomes 3)
    dep_val = 3 if dependents == "3+" else int(dependents)

    # Format dataframe to match the exact order expected by the model
    input_data = pd.DataFrame([{
        "Gender": encoded_gender,
        "Married": encoded_married,
        "Dependents": dep_val,
        "Education": encoded_education,
        "Self_Employed": encoded_self_emp,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history_val,
        "Property_Area": encoded_property
    }])

    # Make Prediction
    if model:
        prediction = model.predict(input_data)[0]
    else:
        # Fallback dummy logic if .pkl file isn't found
        st.warning("⚠️ Pre-trained model (`random_model.pkl`) not found. Showing simulated result.")
        prediction = 1 if (credit_history_val == 1.0 and applicant_income > 2500) else 0

    # Display Results
    if prediction == 1 or prediction == 'Y':
        st.success("🎉 **Congratulations! The loan request is likely to be APPROVED.**")
    else:
        st.error("❌ **Sorry, the loan request is likely to be REJECTED.**")