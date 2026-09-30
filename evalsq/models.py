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


# ── Deployment zoo (monthly walk-forward in deploy.py) ───────────────────────

MOM = ["ret1", "ret5", "ret20", "ma10", "ma50"]
VOL = ["ret1", "ret5", "vol20"]
CAL = ["dow", "month"]


def make_zoo() -> dict:
    """Fresh deployment zoo: {name: (feature list, unfitted pipeline)}.

    Different model families on different feature sets, so no single model should win every month.
    """
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline

    def pipe(model):
        return make_pipeline(StandardScaler(), model)

    return {
        "LogReg": (FEATURES, pipe(LogisticRegression(max_iter=500))),
        "LogReg-mom": (MOM, pipe(LogisticRegression(max_iter=500))),
        "RF": (FEATURES, pipe(RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=0))),
        "ExtraTrees-vol": (VOL + CAL, pipe(ExtraTreesClassifier(n_estimators=100, min_samples_leaf=20, random_state=0))),
        "GBM": (FEATURES + CAL, pipe(HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.05, random_state=0))),
        "GBM-mom": (MOM, pipe(HistGradientBoostingClassifier(max_iter=100, max_depth=2, learning_rate=0.05, random_state=0))),
        "kNN": (FEATURES, pipe(KNeighborsClassifier(n_neighbors=100))),
        "MLP": (FEATURES, pipe(MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-2, max_iter=300, early_stopping=True, random_state=0))),
    }


def make_grid(n: int = 25, seed: int = 0) -> dict:
    """n random GBM configs (depth, learning rate, iterations, leaf size, feature set). Backup winner's-curse demo."""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline

    rng = np.random.default_rng(seed)
    feats = [FEATURES, MOM, VOL + CAL, FEATURES + CAL]
    grid = {}
    for i in range(n):
        model = HistGradientBoostingClassifier(
            max_depth=int(rng.integers(2, 6)),
            learning_rate=float(10 ** rng.uniform(-2.3, -0.7)),
            max_iter=int(rng.integers(20, 150)),
            min_samples_leaf=int(rng.integers(10, 200)),
            random_state=i,
        )
        grid[f"gbm{i:03d}"] = (feats[int(rng.integers(len(feats)))], make_pipeline(StandardScaler(), model))
    return grid
