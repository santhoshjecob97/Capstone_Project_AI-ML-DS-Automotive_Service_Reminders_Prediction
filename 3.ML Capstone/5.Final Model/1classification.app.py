import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load your trained pipeline (make sure this file is in your repo/app working directory)
@st.cache_resource
def load_model():
    return joblib.load('Class_service_reminder_model1.pkl')

model = load_model()

st.title("AI-Driven Customer Service Reminder System")
st.markdown(
    "Personalised, AI-powered vehicle service prediction and message generator for dealerships."
)

st.header("Enter Customer & Vehicle Details")

# Input features: You can add more based on your model
age_of_vehicle = st.number_input("Age of Vehicle (years)", min_value=0, max_value=30, value=5)
odometer_reading = st.number_input("Odometer Reading (km)", min_value=0, max_value=500000, value=25000)
last_service_kms = st.number_input("Km at Last Service", min_value=0, max_value=500000, value=20000)
avg_kms_per_month = st.number_input("Average Km per Month", min_value=0, max_value=5000, value=1000)
last_service_cost = st.number_input("Last Service Cost (INR)", min_value=0, max_value=100000, value=4000)
feedback_score = st.slider("Feedback Score", min_value=1, max_value=5, value=4)
days_since_last_service = st.number_input("Days Since Last Service", min_value=0, max_value=365, value=90)
next_service_due_days = st.number_input("Days Until Next Service Due", min_value=-365, max_value=365, value=60)

customer_type = st.selectbox("Customer Type", ['Retail', 'Fleet', 'Corporate'])
AMC_status = st.selectbox("AMC Status", ['Active', 'Not Subscribed', 'Expired'])
warranty_status = st.selectbox("Warranty Status", ['Active', 'Expired'])
insurance_status = st.selectbox("Insurance Status", ['Active', 'Expired'])
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Electric'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
customer_feedback = st.text_area("Customer Feedback (text)", value="Service was smooth and efficient.")

# The order/layout of columns in the input must match your model's expected order
feature_dict = {
    'age_of_vehicle': age_of_vehicle,
    'odometer_reading': odometer_reading,
    'last_service_kms': last_service_kms,
    'avg_kms_per_month': avg_kms_per_month,
    'last_service_cost': last_service_cost,
    'feedback_score': feedback_score,
    'days_since_last_service': days_since_last_service,
    'next_service_due_days': next_service_due_days,
    'customer_type': customer_type,
    'AMC_status': AMC_status,
    'warranty_status': warranty_status,
    'insurance_status': insurance_status,
    'fuel_type': fuel_type,
    'transmission': transmission,
    'customer_feedback': customer_feedback
}

if st.button("Predict & Generate Reminder"):
    input_df = pd.DataFrame([feature_dict])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    if hasattr(model, 'predict_proba'):
        probability = np.max(model.predict_proba(input_df))
        st.write(f"**Confidence:** {probability:.2f}")

    # Example of simple post-processing (your real pipeline may have more)
    urgency_map = {1: "Service Needed Soon (Urgent)", 0: "No Immediate Service Needed"}
    st.success(f"Prediction: {urgency_map[prediction]}")

    # Dynamic message generation based on simple logic—replace with your own generator
    if prediction == 1:
        if feedback_score <= 2:
            msg = (
                f"Dear {customer_type} Customer,\n"
                "We noticed your previous feedback and want to ensure a better experience this time.\n"
                "Your vehicle is due for service soon. Book now to stay safe on the road.\n"
                "Special offer: As an apology, enjoy a 15% discount on your next service."
            )
            channels = "Phone, WhatsApp, Email"
        else:
            msg = (
                f"Dear {customer_type} Customer,\n"
                "Your vehicle is due for service soon. Please schedule your appointment."
            )
            channels = "WhatsApp, Email, SMS"
    else:
        msg = "No immediate service is due! Thank you for being a valued customer."
        channels = "Email"

    st.markdown("#### Reminder Message")
    st.code(msg)
    st.info(f"**Recommended Communication Channels:** {channels}")

st.caption("Powered by Machine Learning (RandomForest, GradientBoosting, LightGBM) and NLP (TF-IDF)")

# -----------------
# To deploy:
# - Place this script and your saved `service_reminder_model.pkl` in the same directory/repo.
# - Run locally with: `streamlit run app.py`
# - Deploy to Streamlit Cloud by pushing these files to GitHub and linking the repo in Streamlit Cloud dashboard.
