"""Deterministic classifier, performance metrics and basic feature monitoring."""

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CATEGORIES, FEATURES


def features(frame):
    return frame.loc[:, list(FEATURES)]


def labels(frame):
    return frame["y"].eq("yes").astype("int64")


def train_model(train):
    numeric = [column for column in FEATURES if column not in CATEGORIES]
    preprocess = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORIES)),
        (StandardScaler(), numeric),
    )
    estimator = make_pipeline(preprocess, LogisticRegression(C=0.2, max_iter=500))
    return estimator.fit(features(train), labels(train))


def evaluate(model, frame):
    expected = labels(frame)
    scores = model.predict_proba(features(frame))[:, 1]
    return {
        "roc_auc": float(roc_auc_score(expected, scores)),
        "average_precision": float(average_precision_score(expected, scores)),
        "positive_rate": float(expected.mean()),
    }


def _distribution(values, cuts):
    if cuts is not None:
        values = pd.Series(np.searchsorted(cuts, values.to_numpy(), side="right"))
    return {str(key): float(value) for key, value in values.value_counts(normalize=True).items()}


def reference_profile(train):
    profile = {}
    for column in FEATURES:
        # Fixed reference deciles become the bins for later monitoring windows.
        cuts = (
            None if column in CATEGORIES
            else np.unique(train[column].quantile(np.arange(1, 10) / 10)).tolist()
        )
        profile[column] = {"cuts": cuts, "proportions": _distribution(train[column], cuts)}
    return profile


def compare_distributions(profile, evaluation, threshold):
    drift = {}
    for column, reference in profile.items():
        expected = reference["proportions"]
        observed = _distribution(evaluation[column], reference["cuts"])
        distance = sum(abs(expected.get(key, 0) - observed.get(key, 0))
                       for key in expected.keys() | observed.keys()) / 2
        drift[column] = {"total_variation": distance, "flagged": distance >= threshold}
    return drift
