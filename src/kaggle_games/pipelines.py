import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)


class SteamFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies all EDA cleaning and feature
    engineering steps to incoming raw data.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X_out = X.copy()

        mnar_cols = [
            "metacritic_url",
            "reviews",
            "website",
            "support_url",
            "support_email",
            "short_description",
            "detailed_description",
            "about_the_game",
        ]
        for col in mnar_cols:
            if col in X_out.columns:
                X_out[f"has_{col}"] = X_out[col].notna().astype(int)

        if "release_date" in X_out.columns:
            dates = pd.to_datetime(X_out["release_date"], errors="coerce")
            X_out["release_year"] = dates.dt.year.fillna(0).astype(int)
            X_out["release_month"] = dates.dt.month.fillna(0).astype(int)

        if "supported_languages" in X_out.columns:
            X_out["language_count"] = (
                X_out["supported_languages"].str.len().fillna(0).astype(int)
            )
        if "tags" in X_out.columns:
            X_out["tag_count"] = X_out["tags"].str.len().fillna(0).astype(int)

        if "categories" in X_out.columns:
            X_out["is_multiplayer"] = (
                X_out["categories"]
                .astype(str)
                .str.contains("Multiplayer", case=False, na=False)
                .astype(int)
            )
        if "genres" in X_out.columns:
            X_out["is_free_to_play"] = (
                X_out["genres"]
                .astype(str)
                .str.contains("Free to Play", case=False, na=False)
                .astype(int)
            )

        for col in ["mac", "linux", "windows"]:
            if col in X_out.columns:
                X_out[col] = X_out[col].astype(float)

        cols_to_drop = [
            "name",
            "developers",
            "publishers",
            "screenshots",
            "movies",
            "header_image",
            "notes",
            "release_date",
            "supported_languages",
            "full_audio_languages",
            "packages",
            "categories",
            "genres",
            "tags",
            "metacritic_url",
            "reviews",
            "website",
            "support_url",
            "support_email",
            "short_description",
            "detailed_description",
            "about_the_game",
            "score_rank",
            "windows",
            "has_score_rank",
            "estimated_owners",
            "recommendations",
            "user_score",
            "positive",
            "negative",
        ]

        existing_drops = [c for c in cols_to_drop if c in X_out.columns]
        X_out = X_out.drop(columns=existing_drops)

        return X_out


def get_lr_pipeline() -> Pipeline:
    skewed_features = [
        "price",
        "peak_ccu",
        "achievements",
        "average_playtime_forever",
        "average_playtime_2weeks",
        "median_playtime_forever",
        "median_playtime_2weeks",
        "dlc_count",
    ]

    standard_features = [
        "metacritic_score",
        "discount",
        "language_count",
        "tag_count",
        "release_year",
        "required_age",
    ]

    categorical_features = [
        "release_month",
        "mac",
        "linux",
        "is_multiplayer",
        "is_free_to_play",
        "has_metacritic_url",
        "has_reviews",
        "has_website",
        "has_support_url",
        "has_support_email",
        "has_short_description",
        "has_detailed_description",
        "has_about_the_game",
    ]

    log_transformer = FunctionTransformer(
        np.log1p, validate=False, feature_names_out="one-to-one"
    )
    skewed_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", log_transformer),
            ("scaler", StandardScaler()),
        ]
    )

    standard_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("skewed", skewed_transformer, skewed_features),
            ("standard", standard_transformer, standard_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("engineer", SteamFeatureEngineer()),
            ("preprocessor", preprocessor),
        ]
    )


def get_rf_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("engineer", SteamFeatureEngineer()),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
