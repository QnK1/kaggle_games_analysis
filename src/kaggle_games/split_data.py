import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_reviews = df["positive"] + df["negative"]
    positive_ratio = np.where(
        total_reviews > 0, df["positive"] / total_reviews, 0
    )

    df["is_hit"] = ((df["positive"] >= 1000) & (positive_ratio >= 0.80)).astype(
        int
    )

    # Prevent target leakage
    df.drop(columns=["positive", "negative", "estimated_owners"], inplace=True)

    X = df.drop(columns=["is_hit"])
    y = df["is_hit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
