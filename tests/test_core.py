import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.models.predictor import CreditRiskPredictor
from src.models.batch_predict import score_applications
from src.utils.metrics import evaluate_model
from src.utils.risk_category import risk_category
from src.utils.risk_score import calculate_risk_score
from src.utils.decision_engine import loan_decision


class FakeModel:
    feature_name_ = ["Age", "Education"]
    classes_ = np.array([0, 1])

    def predict(self, frame):
        return np.array([1])

    def predict_proba(self, frame):
        return np.array([[.2, .8]])


class CoreTests(unittest.TestCase):
    def predictor(self):
        enc = LabelEncoder().fit(["Bachelor", "PhD"])
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path, encoder_path = Path(temp_dir) / "model", Path(temp_dir) / "encoder"
            model_path.touch()
            encoder_path.touch()
            with patch("src.models.predictor._load_artifacts", return_value=(FakeModel(), {"Education": enc})):
                return CreditRiskPredictor(model_path, encoder_path)

    def test_prediction_schema_and_probability(self):
        prediction, probability = self.predictor().predict({"Age": 40, "Education": "PhD"})
        self.assertEqual(prediction, 1)
        self.assertEqual(probability, .8)

    def test_missing_extra_and_unknown_values_rejected(self):
        predictor = self.predictor()
        for customer in ({"Age": 40}, {"Age": 40, "Education": "PhD", "extra": 1}, {"Age": 40, "Education": "Unknown"}):
            with self.assertRaises(ValueError):
                predictor.predict(customer)

    def test_batch_schema_and_output_columns(self):
        class BatchPredictor:
            feature_names = ["Age", "Education"]
            def predict(self, row):
                return 1, .42
        scored = score_applications(pd.DataFrame([{"Age": 45, "Education": "PhD"}]), BatchPredictor())
        self.assertEqual(scored.loc[0, "DefaultProbability"], .42)
        self.assertEqual(scored.loc[0, "RiskCategory"], "Medium")
        with self.assertRaises(ValueError):
            score_applications(pd.DataFrame([{"Age": 45}]), BatchPredictor())

    def test_batch_allows_observed_target_without_using_it_as_feature(self):
        class BatchPredictor:
            feature_names = ["Age", "Education"]
            seen = None
            def predict(self, row):
                self.seen = row
                return 0, .12
        predictor = BatchPredictor()
        scored = score_applications(
            pd.DataFrame([{"Age": 45, "Education": "PhD", "Default": 1}]), predictor
        )
        self.assertEqual(predictor.seen, {"Age": 45, "Education": "PhD"})
        self.assertEqual(scored.loc[0, "Default"], 1)
        self.assertEqual(scored.loc[0, "DefaultProbability"], .12)
        with self.assertRaisesRegex(ValueError, "only 0 or 1"):
            score_applications(
                pd.DataFrame([{"Age": 45, "Education": "PhD", "Default": 2}]), predictor
            )

    def test_risk_boundaries_and_score_range_mapping(self):
        self.assertEqual(calculate_risk_score(0), 0)
        self.assertEqual(calculate_risk_score(1), 1000)
        with self.assertRaises(ValueError):
            calculate_risk_score(1.01)
        self.assertEqual(risk_category(299), "Low")
        self.assertEqual(risk_category(300), "Medium")
        self.assertEqual(risk_category(700), "High")
        self.assertEqual(loan_decision(300), "Manual Review")
        self.assertEqual(loan_decision(600), "Rejected")

    def test_metrics_respect_threshold(self):
        class MetricModel:
            def predict_proba(self, _):
                return np.array([[.9, .1], [.4, .6], [.2, .8]])
        metrics = evaluate_model(MetricModel(), None, np.array([0, 1, 1]), threshold=.7)
        self.assertEqual(metrics["threshold"], .7)
        self.assertEqual(metrics["confusion_matrix"].tolist(), [[1, 0], [1, 1]])
        with self.assertRaises(ValueError):
            evaluate_model(MetricModel(), None, np.array([0, 1, 1]), threshold=1.1)


if __name__ == "__main__":
    unittest.main()
