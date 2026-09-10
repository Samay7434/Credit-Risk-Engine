import sys
from pathlib import Path

# ----------------------------------------------------
# VERY IMPORTANT
# ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------
# Normal imports
# ----------------------------------------------------

import os
import subprocess

import pandas as pd
import plotly.express as px
import streamlit as st
import shap
import matplotlib.pyplot as plt

from src.models.predictor import CreditRiskPredictor
from src.utils.risk_score import calculate_risk_score
from src.utils.decision_engine import loan_decision
from src.explainability.shap_explainer import CreditRiskExplainer
from src.utils.report_generator import generate_report

from src.config import (
    INPUT_DATA_PATH,
    OUTPUT_DATA_PATH,
)

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Engine",
    page_icon="🏦",
    layout="wide",
)

st.markdown("""
<style>

.main {
    background-color:#f5f7fa;
}

h1{
    color:#0f172a;
}

div[data-testid="metric-container"]{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,.08);
}

.stButton>button{
    width:100%;
    border-radius:10px;
    height:45px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
# 🏦 Enterprise Credit Risk Underwriting Platform

### AI Powered Loan Approval System

---
""")

st.markdown(
    """
This system provides:

- Credit Risk Prediction
- Batch Loan Underwriting
- Portfolio Analytics
- Model Training
"""
)

# ---------------------------------------------------
# Load Predictor
# ---------------------------------------------------

predictor = CreditRiskPredictor()

explainer = CreditRiskExplainer()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/bank-building.png",
    width=90,
)

st.sidebar.title("Credit Risk Engine")

st.sidebar.markdown("---")

menu = st.sidebar.radio(

    "Navigation",

    [
        "🏦 Underwriting",
        "📂 Batch Processing",
        "📊 Portfolio Dashboard",
        "⚙ Model Training",
    ]

)

# ===================================================
# SINGLE PREDICTION
# ===================================================

if menu == "🏦 Underwriting":

    st.header("Single Loan Prediction")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("Age", 18, 100, 35)

        income = st.number_input(
            "Income",
            10000,
            500000,
            65000,
        )

        loan_amount = st.number_input(
            "Loan Amount",
            1000,
            100000,
            18000,
        )

        credit_score = st.number_input(
            "Credit Score",
            300,
            850,
            720,
        )

        months_employed = st.number_input(
            "Months Employed",
            0,
            500,
            60,
        )

        num_credit_lines = st.number_input(
            "Credit Lines",
            1,
            20,
            4,
        )

    with col2:

        interest_rate = st.number_input(
            "Interest Rate",
            1.0,
            40.0,
            9.5,
        )

        loan_term = st.selectbox(
            "Loan Term",
            [12, 24, 36, 48, 60]
        )

        dti = st.slider(
            "DTI Ratio",
            0.0,
            1.0,
            0.32,
        )

        education = st.selectbox(

            "Education",

            [

                "High School",

                "Bachelor's",

                "Master's",

                "PhD",

            ]

        )

        employment = st.selectbox(

            "Employment",

            [

                "Full-time",

                "Part-time",

                "Self-employed",

                "Unemployed",

            ]

        )

        marital = st.selectbox(

            "Marital Status",

            [

                "Single",

                "Married",

                "Divorced",

            ]

        )

        mortgage = st.selectbox(

            "Mortgage",

            [

                "Yes",

                "No",

            ]

        )

        dependents = st.selectbox(

            "Dependents",

            [

                "Yes",

                "No",

            ]

        )

        loan_purpose = st.selectbox(

            "Loan Purpose",

            [

                "Home",

                "Auto",

                "Business",

                "Education",

                "Other",

            ]

        )

        cosigner = st.selectbox(

            "Co-Signer",

            [

                "Yes",

                "No",

            ]

        )

    if st.button("Predict Credit Risk"):

        customer = {

            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines,
            "InterestRate": interest_rate,
            "LoanTerm": loan_term,
            "DTIRatio": dti,
            "Education": education,
            "EmploymentType": employment,
            "MaritalStatus": marital,
            "HasMortgage": mortgage,
            "HasDependents": dependents,
            "LoanPurpose": loan_purpose,
            "HasCoSigner": cosigner,

        }

        prediction, probability = predictor.predict(customer)

        risk_score = calculate_risk_score(probability)

        decision = loan_decision(risk_score)

        report = generate_report(
            customer,
            probability,
            risk_score,
            decision,
        )

        encoded_df, shap_values = explainer.explain(customer)

        st.success("Prediction Complete")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Default Probability",
            f"{probability:.2%}"
        )

        c2.metric(
            "Risk Score",
            f"{risk_score}/1000"
        )

        if decision == "Approved":

            c3.success("✅ APPROVED")

        elif decision == "Manual Review":

            c3.warning("🟡 MANUAL REVIEW")

        else:

            c3.error("🔴 REJECTED")

        # =====================================================
        # SHAP EXPLANATION STARTS HERE
        # =====================================================

        st.divider()

        st.header("🏦 Enterprise Underwriting Report")

        st.markdown("### Application Summary")

        left, right = st.columns(2)

        with left:

            st.write(
                f"**Application ID:** {report['ApplicationID']}"
            )

            st.write(
                f"**Risk Category:** {report['RiskCategory']}"
            )

        with right:

            st.write(
                f"**Probability of Default:** {report['Probability']}%"
            )

            st.write(
                f"**Decision:** {report['Decision']}"
            )

        st.divider()

        st.subheader("Primary Approval Factors")

        for item in report["ApprovalFactors"]:

            st.success(item)

        st.subheader("Primary Risk Factors")

        if report["RiskFactors"]:

            for item in report["RiskFactors"]:

                st.warning(item)

        else:

            st.success("No major risk factors detected.")

        st.subheader("Policy Validation")

        for item in report["PolicyChecks"]:

            st.info(item)

        st.divider()

        st.subheader("Model Information")

        st.write("Algorithm : LightGBM")

        st.write("Hyperparameter Optimization : Optuna")

        st.write("Experiment Tracking : MLflow")

        with st.expander("Technical Explanation (SHAP)"):

            fig = plt.figure(figsize=(10, 4))

            shap.plots.waterfall(

                shap.Explanation(

                    values=shap_values[0],

                    base_values=explainer.explainer.expected_value,

                    data=encoded_df.iloc[0],

                    feature_names=encoded_df.columns,

                ),

                show=False,

            )

            st.pyplot(fig)


# ===================================================
# BATCH UNDERWRITING
# ===================================================

elif menu == "📂 Batch Processing":

    st.header("Batch Loan Underwriting")

    uploaded_file = st.file_uploader(

        "Upload Loan Applications CSV",

        type=["csv"]

    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.dataframe(df)

        df.to_csv(
            INPUT_DATA_PATH,
            index=False,
        )

        if st.button("Run Batch Underwriting"):

            subprocess.run(

                [

                    "python",

                    "-m",

                    "src.models.batch_predict",

                ]

            )

            st.success("Batch Underwriting Completed!")

            report = pd.read_csv(OUTPUT_DATA_PATH)

            st.dataframe(report)

            with open(
                OUTPUT_DATA_PATH,
                "rb",
            ) as file:

                st.download_button(

                    "Download Underwriting Report",

                    file,

                    file_name="underwriting_report.csv",

                    mime="text/csv",

                )

# ===================================================
# PORTFOLIO ANALYTICS
# ===================================================

elif menu == "📊 Portfolio Dashboard":

    st.header("🏦 Loan Portfolio Analytics")

    if os.path.exists(OUTPUT_DATA_PATH):

        df = pd.read_csv(OUTPUT_DATA_PATH)

        total = len(df)

        approved = (
            df["LoanDecision"] == "Approved"
        ).sum()

        review = (
            df["LoanDecision"] == "Manual Review"
        ).sum()

        rejected = (
            df["LoanDecision"] == "Rejected"
        ).sum()

        avg_score = round(
            df["RiskScore"].mean(),
            2,
        )

        avg_default = round(
            df["DefaultProbability"].mean() * 100,
            2,
        )

        approval_rate = round(
        approved / total * 100,
        2,
        )

        rejection_rate = round(
        rejected / total * 100,
        2,
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Applications",
            total,
        )

        c2.metric(
            "Approval Rate",
            f"{approval_rate}%"
        )

        c3.metric(
            "Rejection Rate",
            f"{rejection_rate}%"
        )

        c4.metric(
            "Manual Review",
            review,
        )

        c5.metric(
            "Average Risk Score",
            avg_score,
        )

        st.divider()

        left, right = st.columns(2)

        with left:

            fig = px.pie(

                df,

                names="LoanDecision",

                title="Loan Decision Distribution",

            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with right:

            fig2 = px.histogram(

                df,

                x="RiskScore",

                nbins=20,

                title="Risk Score Distribution",

            )

            st.plotly_chart(
                fig2,
                use_container_width=True,
            )

            st.divider()

            st.subheader("Risk Category Distribution")

            risk_counts = (
                df["RiskCategory"]
                .value_counts()
                .reset_index()
            )

            risk_counts.columns = [
                "RiskCategory",
                "Count",
            ]

            fig3 = px.bar(

                risk_counts,

                x="RiskCategory",

                y="Count",

                title="Loan Risk Categories",

            )

            st.plotly_chart(
                fig3,
                width="stretch",
            )

        st.divider()

        st.subheader("Average Default Probability")

        st.progress(avg_default / 100)

        st.write(
            f"{avg_default}%"
        )

        st.divider()

        st.subheader("Top 10 Highest Risk Applications")

        high_risk = (

            df.sort_values(

                "RiskScore",

                ascending=False,

            )

            .head(10)

        )

        st.dataframe(

            high_risk,

            width="stretch",

            hide_index=True,

        )

        st.divider()

        st.subheader("Latest Underwriting Report")

        st.dataframe(

            df,

            use_container_width=True,

            hide_index=True,

        )

        csv = df.to_csv(index=False)

        st.download_button(

            "Download Report",

            csv,

            "underwriting_report.csv",

            "text/csv",

        )

    else:

        st.warning(
            "Run Batch Underwriting first."
        )

# ===================================================
# TRAIN MODEL
# ===================================================

elif menu == "⚙ Model Training":

    st.header("Model Training")

    st.write(
        "Train the baseline LightGBM model."
    )

    if st.button("Train Model"):

        with st.spinner("Training..."):

            subprocess.run(

                [

                    "python",

                    "-m",

                    "src.models.train",

                ]

            )

        st.success("Training Completed Successfully!")

st.markdown("---")

st.caption(
    "Enterprise Credit Risk Engine • LightGBM • Optuna • MLflow • Streamlit"
)