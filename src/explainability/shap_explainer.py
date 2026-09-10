import joblib
import pandas as pd
import shap

from src.config import (
    MODEL_PATH,
    ENCODER_PATH,
)


class CreditRiskExplainer:

    def __init__(self):

        self.model = joblib.load(MODEL_PATH)

        self.encoders = joblib.load(
            ENCODER_PATH
        )

        self.explainer = shap.TreeExplainer(
            self.model
        )

    def explain(self, customer):

        df = pd.DataFrame([customer])

        for column, encoder in self.encoders.items():

            df[column] = encoder.transform(
                df[column]
            )

        shap_values = self.explainer.shap_values(df)

        print("\n" + "=" * 70)
        print("SHAP DEBUG")
        print("=" * 70)
        print("TYPE:", type(shap_values))

        if hasattr(shap_values, "shape"):
            print("SHAPE:", shap_values.shape)

        try:
            print("LEN:", len(shap_values))
        except Exception as e:
            print("NO LEN:", e)

        print("VALUE:")
        print(shap_values)
        print("=" * 70 + "\n")

        return df, shap_values
    
        print("EXPECTED VALUE:")
        print(self.explainer.expected_value)