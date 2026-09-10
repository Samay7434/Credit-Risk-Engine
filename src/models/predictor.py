import joblib
import pandas as pd

from src.config import (
    MODEL_PATH,
    ENCODER_PATH,
)


class CreditRiskPredictor:

    def __init__(self):

        self.model = joblib.load(MODEL_PATH)

        self.encoders = joblib.load(
            ENCODER_PATH
        )

    def predict(self, customer):

        df = pd.DataFrame([customer])

        for column, encoder in self.encoders.items():

            df[column] = encoder.transform(
                df[column]
            )

        prediction = self.model.predict(df)[0]

        probability = self.model.predict_proba(df)[0][1]

        return prediction, probability