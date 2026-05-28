"""Classical ML baselines: Logistic Regression, SVM (RBF), Random Forest."""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def make_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=1000,
            random_state=config.RANDOM_SEED,
            C=1.0,
        )),
    ])


def make_svm() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(
            kernel="rbf",
            probability=True,
            random_state=config.RANDOM_SEED,
            C=1.0,
            gamma="scale",
        )),
    ])


def make_rf() -> Pipeline:
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )),
    ])


CLASSICAL_MODELS = {
    "LogisticRegression": make_lr,
    "SVM":               make_svm,
    "RandomForest":      make_rf,
}
