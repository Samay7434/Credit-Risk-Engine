"""Streamlit interface for the Credit Risk Engine."""

from pathlib import Path
import json
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split

from src.config import (
    MODEL_PATH, ENCODER_PATH, RAW_DATA_PATH, RANDOM_STATE, TEST_SIZE,
    CANDIDATE_METRICS_PATH,
)
from src.models.batch_predict import score_applications
from src.models.predictor import CreditRiskPredictor
from src.utils.metrics import evaluate_model
from src.utils.risk_category import risk_category
from src.utils.risk_score import calculate_risk_score
from src.utils.decision_engine import loan_decision


st.set_page_config(page_title="Credit Risk Engine", page_icon="CR", layout="wide")
st.markdown("""
<style>
  :root { --navy:#10243a; --blue:#2864dc; --ink:#172b40; --muted:#667789; }
  [data-testid="stAppViewContainer"] { background:#f5f7fa; color:var(--ink); }
  [data-testid="stSidebar"] { background:var(--navy); }
  [data-testid="stSidebar"] * { color:#edf3fb; }
  [data-testid="stMetric"] { background:#fff; border:1px solid #e3e9f0; padding:16px 18px; border-radius:10px; }
  h1,h2,h3 { color:var(--ink); letter-spacing:-.02em; }
  div.stButton > button, div.stDownloadButton > button { border-radius:7px; font-weight:600; }
  @media (max-width: 700px) { [data-testid="stMetric"] { padding:12px; } }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model artifacts…")
def get_predictor(model_mtime: float, encoder_mtime: float):
    return CreditRiskPredictor()


@st.cache_data(show_spinner="Evaluating saved model on the holdout data…")
def evaluate_saved_model(model_mtime: float, encoder_mtime: float, data_mtime: float):
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {RAW_DATA_PATH}")
    predictor = get_predictor(model_mtime, encoder_mtime)
    data = pd.read_csv(RAW_DATA_PATH).drop(columns=["LoanID"], errors="ignore")
    if "Default" not in data:
        raise ValueError("The evaluation dataset must contain the Default target.")
    X, y = data.drop(columns=["Default"]), data["Default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    for column, encoder in predictor.encoders.items():
        if column in X_test:
            X_test[column] = encoder.transform(X_test[column].astype(str))
    X_test = X_test[predictor.feature_names]
    return evaluate_model(predictor.model, X_test, y_test)


def app_status():
    missing = [str(path.name) for path in (MODEL_PATH, ENCODER_PATH) if not path.exists()]
    return "Model ready" if not missing else "Model artifacts missing: " + ", ".join(missing)


def model_timestamps():
    return (MODEL_PATH.stat().st_mtime, ENCODER_PATH.stat().st_mtime)


st.sidebar.markdown("## Credit Risk Engine")
st.sidebar.caption("Credit Risk & Portfolio Analytics")
st.sidebar.divider()
page = st.sidebar.radio("Workspace", ["Executive dashboard", "Single assessment", "Batch analysis", "Portfolio analytics", "Model performance", "Explainability", "Model training", "About"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(app_status())
st.sidebar.caption("Decision support only · Review outputs before use")

st.title(page)
st.caption("Credit risk analysis workspace")

if page == "Executive dashboard":
    st.markdown("Review the saved model, assess an application, or analyze a batch. Portfolio metrics appear after you score a batch in this session.")
    batch = st.session_state.get("batch_result")
    if batch is None:
        st.info("No applications have been processed in this session. Start with a single assessment or upload a CSV in Batch analysis.")
    else:
        risk = batch["RiskCategory"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0).rename_axis("Risk category").reset_index(name="Applications")
        cols = st.columns(3)
        cols[0].metric("Applications processed", f"{len(batch):,}", help="Applications scored in the current session.")
        cols[1].metric("High risk", f"{(batch.RiskCategory == 'High').sum():,}", help="Model probability mapped to the existing risk score thresholds.")
        cols[2].metric("Average predicted default probability", f"{batch.DefaultProbability.mean():.1%}", help="Mean model probability across this session's scored applications.")
        fig = px.bar(risk, x="Risk category", y="Applications", color="Risk category", title="Predicted risk category distribution", color_discrete_map={"Low":"#3c8c72", "Medium":"#d59b32", "High":"#bd5261"})
        st.plotly_chart(fig, width="stretch")
    st.subheader("Model status")
    st.write(app_status())
    st.caption("Observed portfolio data is not available unless supplied and scored. Predictions shown here describe model outputs, not observed defaults.")

elif page == "Single assessment":
    try:
        predictor = get_predictor(*model_timestamps())
        fields = predictor.feature_names
        categories = {name: list(encoder.classes_) for name, encoder in predictor.encoders.items()}
        numeric_specs = {
            "Age": (18, 69, 43, 1), "Income": (15000, 149999, 82499, 1000),
            "LoanAmount": (5000, 249999, 127579, 1000), "CreditScore": (300, 849, 574, 1),
            "MonthsEmployed": (0, 119, 60, 1), "NumCreditLines": (1, 4, 2, 1),
            "InterestRate": (2.0, 25.0, 13.5, .1), "LoanTerm": (12, 60, 36, 12),
            "DTIRatio": (.1, .9, .5, .01),
        }
        customer = {}
        with st.form("assessment"):
            st.caption("Enter values within the ranges represented by the training dataset.")
            numeric_fields = [name for name in fields if name in numeric_specs]
            categorical_fields = [name for name in fields if name in categories]
            left, right = st.columns(2)
            for i, name in enumerate(numeric_fields):
                lo, hi, default, step = numeric_specs[name]
                with (left if i % 2 == 0 else right):
                    customer[name] = st.number_input(name, min_value=lo, max_value=hi, value=default, step=step, help=f"Training data range: {lo} to {hi}.")
            for i, name in enumerate(categorical_fields):
                with (left if i % 2 == 0 else right):
                    customer[name] = st.selectbox(name, categories[name])
            submitted = st.form_submit_button("Assess application", type="primary")
        if submitted:
            prediction, probability = predictor.predict(customer)
            score = calculate_risk_score(probability)
            category = risk_category(score)
            decision = loan_decision(score)
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted default probability", f"{probability:.1%}")
            c2.metric("Risk score", f"{score} / 1000")
            c3.metric("Risk category", category)
            st.subheader("Decision support")
            st.write(f"**Rule based outcome:** {decision}")
            st.caption("This outcome follows the repository's current score thresholds. It is a policy heuristic, separate from the model prediction, and is not a lending decision.")
            st.write(f"Model class prediction: {'Default' if prediction == 1 else 'No default'}")
    except Exception as exc:
        st.error(f"Assessment unavailable: {exc}")

elif page == "Batch analysis":
    uploaded = st.file_uploader("Upload application data (CSV)", type=["csv"], help="Use the model input columns. An optional LoanID column is retained in the result.")
    if uploaded:
        try:
            source = pd.read_csv(uploaded)
            st.caption(f"{len(source):,} rows · {len(source.columns)} columns")
            st.dataframe(source.head(8), width="stretch", hide_index=True)
            if st.button("Score applications", type="primary"):
                with st.spinner("Scoring applications…"):
                    result = score_applications(source)
                    st.session_state["batch_result"] = result
                st.success(f"Scored {len(result):,} applications.")
                st.dataframe(result, width="stretch", hide_index=True)
                st.download_button("Download scored CSV", result.to_csv(index=False), "credit_risk_results.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process CSV: {exc}")

elif page == "Portfolio analytics":
    batch = st.session_state.get("batch_result")
    if batch is None:
        st.info("No scored applications in this session yet. Run Batch analysis to see portfolio summaries.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Applications", f"{len(batch):,}")
        c2.metric("High risk", f"{(batch.RiskCategory == 'High').sum():,}")
        c3.metric("Average predicted default probability", f"{batch.DefaultProbability.mean():.1%}")
        c4.metric("Average risk score", f"{batch.RiskScore.mean():.0f} / 1000")
        counts = batch.RiskCategory.value_counts().reindex(["Low", "Medium", "High"], fill_value=0).rename_axis("Risk category").reset_index(name="Applications")
        st.plotly_chart(px.bar(counts, x="Risk category", y="Applications", title="Predicted risk category distribution"), width="stretch")
        if "LoanAmount" in batch:
            st.plotly_chart(px.box(batch, x="RiskCategory", y="LoanAmount", category_orders={"RiskCategory":["Low","Medium","High"]}, title="Loan amount by predicted risk category", labels={"LoanAmount":"Loan amount"}), width="stretch")
        st.caption("All figures summarize model predictions for this session's uploaded batch; they are not observed default rates.")

elif page == "Model performance":
    st.write("Evaluation of the saved model artifact on the reproducible stratified holdout split from the configured raw dataset.")
    try:
        model_time, encoder_time = model_timestamps()
        data_time = RAW_DATA_PATH.stat().st_mtime if RAW_DATA_PATH.exists() else 0.0
        metrics = evaluate_saved_model(model_time, encoder_time, data_time)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        c2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        c3.metric("Precision", f"{metrics['precision']:.1%}")
        c4.metric("Recall", f"{metrics['recall']:.1%}")
        c1, c2, c3 = st.columns(3)
        c1.metric("F1 score", f"{metrics['f1_score']:.3f}")
        c2.metric("Threshold", f"{metrics['threshold']:.2f}")
        c3.metric("Holdout observations", f"{sum(sum(row) for row in metrics['confusion_matrix']):,}")
        cm = metrics["confusion_matrix"]
        st.plotly_chart(px.imshow(cm, text_auto=True, x=["Predicted no default", "Predicted default"], y=["Actual no default", "Actual default"], labels={"x":"Prediction", "y":"Observed target", "color":"Applications"}, title="Confusion matrix at threshold 0.50", color_continuous_scale="Blues"), width="stretch")
        st.caption("Target: Default (1 = default). Holdout split: 20%, stratified, random_state=42. Metrics describe the saved artifact on this dataset and split; they are not a claim of current live-service performance.")
    except Exception as exc:
        st.error(f"Evaluation unavailable: {exc}")

elif page == "Explainability":
    try:
        model = get_predictor(*model_timestamps()).model
        if not hasattr(model, "feature_importances_"):
            st.info("This saved model does not provide feature importance values.")
        else:
            importance = pd.DataFrame({"Feature": get_predictor(*model_timestamps()).feature_names, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
            st.plotly_chart(px.bar(importance, x="Importance", y="Feature", orientation="h", title="LightGBM split feature importance", labels={"Importance":"Split importance", "Feature":"Input feature"}), width="stretch")
            st.caption("Split importance counts how often a feature is used in model splits. It does not show causal impact, direction, or an individual applicant's reason for a prediction.")
    except Exception as exc:
        st.error(f"Explainability unavailable: {exc}")

elif page == "Model training":
    st.write("Train a baseline LightGBM candidate on the configured raw dataset and evaluate it on the stratified holdout.")
    st.info("Training saves a matched model and encoder pair under `models/candidates/`. It does not replace the artifacts used by the app.")
    if not RAW_DATA_PATH.exists():
        st.error(f"Training data is unavailable: {RAW_DATA_PATH}")
    elif st.button("Train candidate model", type="primary"):
        with st.spinner("Training candidate model. This may take several minutes…"):
            result = subprocess.run(
                [sys.executable, "-m", "src.models.train"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
        if result.returncode == 0:
            st.success("Candidate training completed. The active model was not changed.")
            if result.stdout:
                st.code(result.stdout[-8000:])
        else:
            st.error("Candidate training failed.")
            st.code((result.stderr or result.stdout)[-8000:])
    if CANDIDATE_METRICS_PATH.exists():
        try:
            candidate_metrics = json.loads(CANDIDATE_METRICS_PATH.read_text(encoding="utf-8"))
            st.subheader("Latest candidate evaluation")
            st.json(candidate_metrics)
            st.caption("Candidate metrics are separate from the active model results shown on Model performance.")
        except (OSError, json.JSONDecodeError) as exc:
            st.warning(f"Candidate metrics could not be read: {exc}")

elif page == "About":
    st.subheader("Credit Risk & Portfolio Analytics")
    st.write("A Streamlit application for estimating default probabilities with a saved LightGBM classifier and summarizing scored applications.")
    st.markdown("**Technology:** Python · Streamlit · LightGBM · scikit-learn · Plotly · SHAP dependency available in the environment")
    st.markdown("**Target:** `Default` in the included loan dataset; class 1 represents default.")
    st.warning("Model outputs are estimates for analysis and require human review. The score thresholds are illustrative code rules, not validated credit policy. Do not use this tool as the sole basis for lending or eligibility decisions.")
    st.caption("Model version and training timestamp are not currently recorded in the repository's artifacts.")
