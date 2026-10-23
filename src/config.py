from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_PATH = DATA_DIR / "raw" / "Loan_default.csv"

PROCESSED_DATA_PATH = (
    DATA_DIR / "processed" / "processed_loan_data.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "lightgbm_credit_risk.pkl"

ENCODER_PATH = MODEL_DIR / "label_encoders.pkl"

CANDIDATE_DIR = MODEL_DIR / "candidates"
CANDIDATE_MODEL_PATH = CANDIDATE_DIR / "lightgbm_credit_risk_candidate.pkl"
CANDIDATE_ENCODER_PATH = CANDIDATE_DIR / "label_encoders_candidate.pkl"
CANDIDATE_METRICS_PATH = CANDIDATE_DIR / "evaluation_metrics.json"

MLRUNS_PATH = PROJECT_ROOT / "mlruns"

INPUT_DATA_PATH = DATA_DIR / "input" / "loan_applications.csv"

OUTPUT_DATA_PATH = DATA_DIR / "output" / "underwriting_report.csv"

RANDOM_STATE = 42

TEST_SIZE = 0.20
