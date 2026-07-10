import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background:
        linear-gradient(rgba(6, 11, 31, 0.88), rgba(6, 11, 31, 0.92)),
        url("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero-section {
    padding: 55px 45px;
    border-radius: 30px;
    background: linear-gradient(135deg, rgba(59,130,246,0.35), rgba(139,92,246,0.30));
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 20px 60px rgba(0,0,0,0.45);
    text-align: center;
    margin-bottom: 35px;
}

.hero-title {
    font-size: 58px;
    font-weight: 900;
    background: linear-gradient(90deg, #ffffff, #93c5fd, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 21px;
    color: #e5e7eb;
    margin-top: 15px;
}

.hero-tags {
    margin-top: 25px;
}

.hero-tag {
    display: inline-block;
    padding: 9px 18px;
    margin: 6px;
    border-radius: 999px;
    background: rgba(255,255,255,0.13);
    color: #ffffff;
    font-weight: 600;
    border: 1px solid rgba(255,255,255,0.20);
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #111827);
}

h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #f9fafb;
}

.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #f9fafb !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #ffffff;
}

.stButton > button {
    border-radius: 18px;
    height: 58px;
    font-size: 19px;
    font-weight: 800;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #6d28d9);
    color: white;
}

.info-card {
    padding: 25px;
    border-radius: 24px;
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid rgba(255,255,255,0.16);
    box-shadow: 0 10px 35px rgba(0,0,0,0.30);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
    <div class="hero-title">Customer Churn Intelligence</div>
    <div class="hero-subtitle">
        Predict customer churn risk using customer profile, service usage, contract type, and billing behavior.
    </div>
    <div class="hero-tags">
        <span class="hero-tag">Machine Learning</span>
        <span class="hero-tag">Customer Retention</span>
        <span class="hero-tag">Business Analytics</span>
        <span class="hero-tag">Smart Prediction</span>
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([2, 1])

with left:
    st.subheader("👤 Customer Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])

    with c2:
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure in Months", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with c3:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    st.subheader("🛡️ Service Details")

    s1, s2, s3 = st.columns(3)

    with s1:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with s2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with s3:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.subheader("💳 Billing Details")

    b1, b2, b3 = st.columns(3)

    with b1:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    with b2:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with b3:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

    total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

with right:
    st.markdown("""
    <div class="info-card">
        <h3>📌 Customer Insights</h3>
        <p>
        This dashboard evaluates customer behavior and highlights possible churn risk factors.
        </p>
        <br>
        <h4>🎯 Key Risk Factors</h4>
        <p>• Month-to-month contract</p>
        <p>• Short customer tenure</p>
        <p>• Electronic check payment</p>
        <p>• High monthly charges</p>
        <p>• No tech support</p>
        <p>• No online security</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

predict = st.button("🔍 Predict Customer Churn", use_container_width=True)

if predict:
    risk_score = 0
    reasons = []

    if contract == "Month-to-month":
        risk_score += 3
        reasons.append("Customer uses month-to-month contract")

    if internet_service == "Fiber optic":
        risk_score += 2
        reasons.append("Customer uses fiber optic internet")

    if payment_method == "Electronic check":
        risk_score += 2
        reasons.append("Customer pays using electronic check")

    if tenure < 12:
        risk_score += 2
        reasons.append("Customer tenure is less than 12 months")
    elif tenure < 24:
        risk_score += 1
        reasons.append("Customer tenure is below 24 months")

    if monthly_charges > 80:
        risk_score += 2
        reasons.append("Monthly charges are high")
    elif monthly_charges > 60:
        risk_score += 1
        reasons.append("Monthly charges are moderately high")

    if online_security == "No":
        risk_score += 1
        reasons.append("Customer has no online security")

    if tech_support == "No":
        risk_score += 1
        reasons.append("Customer has no tech support")

    if paperless_billing == "Yes":
        risk_score += 1
        reasons.append("Customer uses paperless billing")

    st.subheader("📢 Prediction Result")

    r1, r2, r3 = st.columns(3)

    if risk_score >= 7:
        prediction = "Likely to Churn"
        risk_level = "High Risk"
        with r1:
            st.error("⚠️ Likely to Churn")
        with r2:
            st.metric("Risk Level", "High")
        with r3:
            st.metric("Risk Score", risk_score)

    elif risk_score >= 4:
        prediction = "Medium Churn Risk"
        risk_level = "Medium Risk"
        with r1:
            st.warning("⚠️ Medium Churn Risk")
        with r2:
            st.metric("Risk Level", "Medium")
        with r3:
            st.metric("Risk Score", risk_score)

    else:
        prediction = "Likely to Stay"
        risk_level = "Low Risk"
        with r1:
            st.success("✅ Likely to Stay")
        with r2:
            st.metric("Risk Level", "Low")
        with r3:
            st.metric("Risk Score", risk_score)

    risk_pct = round(min(risk_score / 12, 1.0) * 100, 1)
    st.subheader("📊 Risk Score Indicator")
    st.write(f"**Risk confidence:** {risk_pct}%")

    st.subheader("🔎 Main Reasons Behind Prediction")
    if reasons:
        for reason in reasons:
            st.write("•", reason)
    else:
        st.write("No major churn risk factors detected.")

    customer_data = pd.DataFrame({
        "Gender": [gender],
        "Senior Citizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "Tenure": [tenure],
        "Phone Service": [phone_service],
        "Multiple Lines": [multiple_lines],
        "Internet Service": [internet_service],
        "Online Security": [online_security],
        "Online Backup": [online_backup],
        "Device Protection": [device_protection],
        "Tech Support": [tech_support],
        "Streaming TV": [streaming_tv],
        "Streaming Movies": [streaming_movies],
        "Contract": [contract],
        "Paperless Billing": [paperless_billing],
        "Payment Method": [payment_method],
        "Monthly Charges": [monthly_charges],
        "Total Charges": [total_charges],
        "Prediction": [prediction],
        "Risk Level": [risk_level],
        "Risk Score": [risk_score]
    })

    st.subheader("📋 Customer Prediction Summary")
    st.dataframe(customer_data, use_container_width=True)

    csv = customer_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction Report",
        data=csv,
        file_name="customer_churn_prediction_report.csv",
        mime="text/csv",
        use_container_width=True
    )