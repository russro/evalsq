"""Simple classification models for SPY direction prediction."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import FEATURES


def make_models() -> dict:
    """Fresh, unfitted model zoo. Single source of truth for all heuristics."""
    return {
        "LogReg": LogisticRegression(max_iter=500),
        "RF": RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=42, n_jobs=-1),
    }


def train_test_split_by_year(df: pd.DataFrame, cutoff: int = 2018):
    return df[df.index.year < cutoff], df[df.index.year >= cutoff]


def fit_models(train: pd.DataFrame) -> tuple[StandardScaler, dict]:
    """Return (scaler, {name: fitted model})."""
    scaler = StandardScaler().fit(train[FEATURES])
    X, y = scaler.transform(train[FEATURES]), train["target"]
    return scaler, {name: m.fit(X, y) for name, m in make_models().items()}


def predict(model, scaler: StandardScaler, df: pd.DataFrame):
    return model.predict(scaler.transform(df[FEATURES]))


def predict_proba(model, scaler: StandardScaler, df: pd.DataFrame):
    """P(next day up)."""
    return model.predict_proba(scaler.transform(df[FEATURES]))[:, 1]


def accuracy(model, scaler: StandardScaler, df: pd.DataFrame) -> float:
    return float((predict(model, scaler, df) == df["target"]).mean())
