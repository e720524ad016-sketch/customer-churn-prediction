import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #0f172a;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main Header */

.main-header {
    background: linear-gradient(135deg, #1e293b, #172554);
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 25px;
    border: 1px solid #334155;
}

.main-header h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 5px;
}

.main-header p {
    color: #cbd5e1;
    font-size: 17px;
    margin-bottom: 0;
}

/* Status */

.status {
    background: #052e16;
    color: #86efac;
    padding: 8px 16px;
    border-radius: 20px;
    display: inline-block;
    font-size: 14px;
    margin-top: 15px;
}

/* Section headings */

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: white;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Input boxes */

.stNumberInput, .stSelectbox {
    color: white;
}

/* Button */

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 700;
    border: none;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #6d28d9);
}

/* Result Card */

.result-card {
    background: #1e293b;
    border: 1px solid #475569;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin-top: 25px;
}

.result-title {
    color: #cbd5e1;
    font-size: 18px;
}

.result-number {
    color: white;
    font-size: 50px;
    font-weight: 800;
    margin: 10px 0;
}

.risk-high {
    background: #450a0a;
    color: #fca5a5;
    padding: 12px 25px;
    border-radius: 25px;
    display: inline-block;
    font-weight: 700;
}

.risk-low {
    background: #052e16;
    color: #86efac;
    padding: 12px 25px;
    border-radius: 25px;
    display: inline-block;
    font-weight: 700;
}

/* Info cards */

.info-card {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
    margin-top: 15px;
}

.info-card h4 {
    color: white;
}

.info-card p {
    color: #cbd5e1;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 40px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-header">

<h1>📊 Customer Churn Prediction</h1>

<p>
AI-powered customer retention risk analysis using Machine Learning
</p>

<div class="status">
● Machine Learning Model Active
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# CUSTOMER PROFILE
# =========================================================

st.markdown(
    '<div class="section-title">👤 Customer Profile</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    tenure = st.number_input(
        "Tenure Months",
        min_value=0,
        value=12
    )

with col2:
    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

with col3:
    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

with col4:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )


col1, col2, col3, col4 = st.columns(4)

with col1:
    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

with col2:
    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

with col3:
    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

with col4:
    phone = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


# =========================================================
# SERVICES
# =========================================================

st.markdown(
    '<div class="section-title">🌐 Services</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

with col2:
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col3:
    online_security = st.selectbox(
        "Online Security",
        ["No internet service", "No", "Yes"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    online_backup = st.selectbox(
        "Online Backup",
        ["No internet service", "No", "Yes"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["No internet service", "No", "Yes"]
    )

with col3:
    tech_support = st.selectbox(
        "Tech Support",
        ["No internet service", "No", "Yes"]
    )


col1, col2 = st.columns(2)

with col1:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No internet service", "No", "Yes"]
    )

with col2:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No internet service", "No", "Yes"]
    )


# =========================================================
# ACCOUNT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">💳 Account Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col3:
    payment = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict = st.button("🔮  PREDICT CUSTOMER CHURN")


# =========================================================
# PREDICTION LOGIC
# =========================================================

if predict:

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


    # =====================================================
    # CREATE FEATURES
    # =====================================================

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

        customer[
            "Multiple Lines_No phone service"
        ] = 1

    elif multiple_lines == "Yes":

        customer[
            "Multiple Lines_Yes"
        ] = 1


    # Internet

    if internet == "Fiber optic":

        customer[
            "Internet Service_Fiber optic"
        ] = 1

    elif internet == "No":

        customer[
            "Internet Service_No"
        ] = 1


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

        customer[
            "Contract_One year"
        ] = 1

    elif contract == "Two year":

        customer[
            "Contract_Two year"
        ] = 1


    # Paperless Billing

    if paperless == "Yes":

        customer[
            "Paperless Billing_Yes"
        ] = 1


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


    # =====================================================
    # SCALE + PREDICT
    # =====================================================

    customer_scaled = scaler.transform(customer)

    probability = model.predict_proba(
        customer_scaled
    )[0][1]
    # ==============================
# MODEL INTERPRETABILITY
# ==============================

if hasattr(model, "coef_"):

    coefficients = model.coef_[0]

    contributions = customer_scaled[0] * coefficients

    explanation = pd.DataFrame({
        "Feature": features,
        "Contribution": contributions
    })

    explanation["Importance"] = explanation["Contribution"].abs()

    top_features = (
        explanation
        .sort_values("Importance", ascending=False)
        .head(10)
        .sort_values("Contribution")
    )

    st.markdown("### 🔍 Why this prediction?")

    st.caption(
        "Positive values increase churn risk, while negative values reduce churn risk."
    )

    st.bar_chart(
        top_features.set_index("Feature")["Contribution"],
        horizontal=True
    )

    # Risk increasing factors
    positive_factors = (
        explanation[explanation["Contribution"] > 0]
        .sort_values("Contribution", ascending=False)
        .head(5)
    )

    # Risk reducing factors
    negative_factors = (
        explanation[explanation["Contribution"] < 0]
        .sort_values("Contribution")
        .head(5)
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="info-card">
            <h4>🔴 Factors Increasing Churn Risk</h4>
            """,
            unsafe_allow_html=True
        )

        if len(positive_factors) > 0:

            for _, row in positive_factors.iterrows():

                st.write(
                    f"• **{row['Feature']}** "
                    f"(+{row['Contribution']:.3f})"
                )

        else:
            st.write("No strong positive risk factors.")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:

        st.markdown(
            """
            <div class="info-card">
            <h4>🟢 Factors Reducing Churn Risk</h4>
            """,
            unsafe_allow_html=True
        )

        if len(negative_factors) > 0:

            for _, row in negative_factors.iterrows():

                st.write(
                    f"• **{row['Feature']}** "
                    f"({row['Contribution']:.3f})"
                )

        else:
            st.write("No strong protective factors.")

        st.markdown("</div>", unsafe_allow_html=True)


    # =====================================================
    # RESULT
    # =====================================================

    percentage = probability * 100

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if probability >= 0.5:

        st.markdown(f"""
        <div class="result-card">

        <div class="result-title">
        Churn Probability
        </div>

        <div class="result-number">
        {percentage:.2f}%
        </div>

        <div class="risk-high">
        🔴 HIGH RISK — Customer Will Churn
        </div>

        <br><br>

        <p style="color:#cbd5e1;">
        This customer has a high probability of leaving the service.
        Consider proactive retention strategies.
        </p>

        </div>
        """, unsafe_allow_html=True)


    else:

        st.markdown(f"""
        <div class="result-card">

        <div class="result-title">
        Churn Probability
        </div>

        <div class="result-number">
        {percentage:.2f}%
        </div>

        <div class="risk-low">
        🟢 LOW RISK — Customer Will Not Churn
        </div>

        <br><br>

        <p style="color:#cbd5e1;">
        This customer currently has a lower probability of leaving the service.
        Continue maintaining good customer engagement.
        </p>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

Customer Churn Prediction • Machine Learning Application

</div>
""", unsafe_allow_html=True)
