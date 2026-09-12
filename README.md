# 🏦 Credit Risk Engine

### Enterprise AI-Powered Credit Risk Underwriting Platform

An AI-powered credit underwriting system designed to assess loan default risk and support automated lending decisions using machine learning.

The system uses **LightGBM** for credit risk prediction, **Optuna** for automated hyperparameter optimization, **MLflow** for experiment tracking, and **SHAP** for model explainability.

---

## 🚀 Live Application

🌐 **Live Demo:**  
https://credit-risk-engine-ai.streamlit.app/

💻 **GitHub Repository:**  
https://github.com/Samay7434/Credit-Risk-Engine

---

## 📌 Project Overview

Credit Risk Engine is an end-to-end machine learning application that evaluates loan applications and estimates the probability that a borrower may default.

The platform transforms borrower information into:

- Default Probability
- Risk Score
- Risk Category
- Loan Decision
- Explainable Risk Factors

The application provides both **single-loan underwriting** and **batch loan processing**, along with portfolio-level analytics and model training capabilities.

---

## 🎯 Objectives

The primary objectives of the Credit Risk Engine are:

- Automate credit risk assessment
- Estimate borrower default probability
- Support data-driven loan approval decisions
- Process multiple loan applications simultaneously
- Provide interpretable model predictions
- Track machine learning experiments
- Provide portfolio-level risk analytics

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Loan Application   │
                    │      Data Input      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Data Preprocessing │
                    │   & Feature Handling  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      LightGBM        │
                    │   Credit Risk Model  │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
             ┌──────────────┐      ┌──────────────┐
             │ Default      │      │    SHAP      │
             │ Probability  │      │ Explainability│
             └──────┬───────┘      └──────────────┘
                    │
                    ▼
             ┌──────────────┐
             │  Risk Score  │
             │   0 - 1000  │
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────┐
             │ Risk Category│
             └──────┬───────┘
                    │
                    ▼
             ┌──────────────────┐
             │ Loan Decision    │
             │                  │
             │ Approved         │
             │ Manual Review    │
             │ Rejected         │
             └──────────────────┘
