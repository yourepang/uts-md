import streamlit as st
import pandas as pd
import pickle

# Load model dan preprocessing tools
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('ord_encoder.pkl', 'rb') as f:
    ord_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Loan Approval Prediction")

# Input form
with st.form(key='loan_form'):
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_gender = st.selectbox("Gender", options=["Male", "Female"])
    person_education = st.selectbox("Education", options=["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    person_income = st.number_input("Income", min_value=0, value=50000)
    person_emp_exp = st.number_input("Employment Experience (in years)", min_value=0, value=5)
    person_home_ownership = st.selectbox("Home Ownership", options=["OWN", "MORTGAGE", "RENT"])
    loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
    loan_intent = st.selectbox("Loan Intent", options=["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, value=5.5)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, value=0.3)
    cb_person_cred_hist_length = st.number_input("Credit History Length (in years)", min_value=0, value=10)
    credit_score = st.number_input("Credit Score", min_value=0, value=700)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", options=["Yes", "No"])

    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Preprocessing
    person_gender = 'male' if person_gender == "Male" else 'female'
    previous_loan_defaults_on_file = 1 if previous_loan_defaults_on_file == "Yes" else 0

    data = pd.DataFrame({
        'person_age': [person_age],
        'person_gender': [person_gender],
        'person_education': [person_education],
        'person_income': [person_income],
        'person_emp_exp': [person_emp_exp],
        'person_home_ownership': [person_home_ownership],
        'loan_amnt': [loan_amnt],
        'loan_intent': [loan_intent],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'credit_score': [credit_score],
        'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
    })

    # Binary encode
    data['person_gender'] = data['person_gender'].replace({'male': 1, 'female': 0})

    # One-hot encode
    encoded = encoder.transform(data[['loan_intent', 'person_home_ownership']])
    df_encoded = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(), index=data.index)
    data = pd.concat([data, df_encoded], axis=1)

    # Drop original categorical columns
    data.drop(columns=['loan_intent', 'person_home_ownership'], inplace=True)

    # Ordinal encode education
    data[['person_education']] = ord_encoder.transform(data[['person_education']])

    # Scaling
    cols_to_scale = [
        'person_age',
        'person_income',
        'person_emp_exp',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'credit_score'
    ]
    data[cols_to_scale] = scaler.transform(data[cols_to_scale])

    # Prediction
    prediction = model.predict(data)[0]

    result = "Approved ✅" if prediction == 1 else "Not Approved ❌"
    st.subheader("Loan Prediction Result:")
    st.success(result)
