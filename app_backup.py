
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Enter customer details to predict the probability of churn."
)

st.divider()

# -----------------------------
# Customer Information
# -----------------------------

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:

    tenure = st.number_input(
        "Tenure Months",
        min_value=0,
        value=12
    )

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

with col2:

    phone = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No internet service", "No", "Yes"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No internet service", "No", "Yes"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No internet service", "No", "Yes"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No internet service", "No", "Yes"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No internet service", "No", "Yes"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No internet service", "No", "Yes"]
    )

st.divider()

# -----------------------------
# Account Information
# -----------------------------

st.subheader("Account Information")

col3, col4 = st.columns(2)

with col3:

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col4:

    payment = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Churn"):

    # Derived features

    avg_monthly = (
        total / tenure
        if tenure > 0
        else monthly
    )

    total_services = 0

    services = [
        phone,
        multiple_lines,
        internet,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies
    ]

    for service in services:
        if service not in [
            "No",
            "No internet service",
            "No phone service"
        ]:
            total_services += 1

    # Tenure group

    if tenure <= 12:
        tenure_group = "New"

    elif tenure <= 24:
        tenure_group = "Medium"

    elif tenure <= 48:
        tenure_group = "Loyal"

    else:
        tenure_group = "Very Loyal"

    # -----------------------------
    # Create 35 features
    # -----------------------------

    customer = pd.DataFrame(
        np.zeros(
            (1, len(features))
        ),
        columns=features
    )

    # Numeric features

    customer["Tenure Months"] = tenure
    customer["Monthly Charges"] = monthly
    customer["Total Charges"] = total
    customer["AvgMonthlyCharge"] = avg_monthly
    customer["TotalServices"] = total_services

    # Customer information

    if gender == "Male":
        customer["Gender_Male"] = 1

    if senior == "Yes":
        customer["Senior Citizen_Yes"] = 1

    if partner == "Yes":
        customer["Partner_Yes"] = 1

    if dependents == "Yes":
        customer["Dependents_Yes"] = 1

    if phone == "Yes":
        customer["Phone Service_Yes"] = 1

    # Multiple Lines

    if multiple_lines == "No phone service":
        customer["Multiple Lines_No phone service"] = 1

    elif multiple_lines == "Yes":
        customer["Multiple Lines_Yes"] = 1

    # Internet

    if internet == "Fiber optic":
        customer["Internet Service_Fiber optic"] = 1

    elif internet == "No":
        customer["Internet Service_No"] = 1

    # Other services

    service_data = {
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies
    }

    for name, value in service_data.items():

        if value == "No internet service":

            customer[
                f"{name}_No internet service"
            ] = 1

        elif value == "Yes":

            customer[
                f"{name}_Yes"
            ] = 1

    # Contract

    if contract == "One year":
        customer["Contract_One year"] = 1

    elif contract == "Two year":
        customer["Contract_Two year"] = 1

    # Paperless Billing

    if paperless == "Yes":
        customer["Paperless Billing_Yes"] = 1

    # Payment Method

    payment_columns = {
        "Credit card (automatic)":
            "Payment Method_Credit card (automatic)",

        "Electronic check":
            "Payment Method_Electronic check",

        "Mailed check":
            "Payment Method_Mailed check"
    }

    if payment in payment_columns:

        customer[
            payment_columns[payment]
        ] = 1

    # Tenure Group

    if tenure_group == "Medium":

        customer[
            "TenureGroup_Medium"
        ] = 1

    elif tenure_group == "Loyal":

        customer[
            "TenureGroup_Loyal"
        ] = 1

    elif tenure_group == "Very Loyal":

        customer[
            "TenureGroup_Very Loyal"
        ] = 1

    # Exact feature order

    customer = customer[features]

    # -----------------------------
    # Scale + Predict
    # -----------------------------

    customer_scaled = scaler.transform(customer)

    probability = model.predict_proba(
        customer_scaled
    )[0][1]

    # -----------------------------
    # Display Result
    # -----------------------------

    st.divider()

    st.subheader("🎯 Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if probability >= 0.5:

        st.error(
            "🔴 Customer Will Churn"
        )

    else:

        st.success(
            "🟢 Customer Will Not Churn"
        )
