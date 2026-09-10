import pandas as pd

from src.data.loader import load_data


def analyze_target():

    df = load_data()

    print("=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)

    print(df["Default"].value_counts())

    print("\nPercentage")

    print(
        df["Default"].value_counts(normalize=True) * 100
    )


if __name__ == "__main__":
    analyze_target()