"""The nine classifiers benchmarked, with their inner-loop search grids.

Every model is wrapped in a Pipeline with StandardScaler so that scaling
statistics are fitted inside the training partition only, never on the
validation or test partition.
"""
from __future__ import annotations

from sklearn.ensemble import (GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_ORDER = [
    "LogisticRegression", "SVM", "KNN", "DecisionTree", "RandomForest",
    "GradientBoosting", "XGBoost", "LightGBM", "CatBoost",
]


def build_models(seed: int = 0) -> dict[str, tuple[Pipeline, dict]]:
    """Return {name: (pipeline, param_grid)} for all nine classifiers."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    def pipe(est):
        return Pipeline([("scaler", StandardScaler()), ("clf", est)])

    models: dict[str, tuple[Pipeline, dict]] = {
        "LogisticRegression": (
            pipe(LogisticRegression(max_iter=5000, random_state=seed)),
            {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "SVM": (
            pipe(SVC(kernel="rbf", probability=True, random_state=seed)),
            {"clf__C": [0.1, 1.0, 10.0],
             "clf__gamma": ["scale", 0.01]},
        ),
        "KNN": (
            pipe(KNeighborsClassifier()),
            {"clf__n_neighbors": [3, 5, 7, 9],
             "clf__weights": ["uniform", "distance"]},
        ),
        "DecisionTree": (
            pipe(DecisionTreeClassifier(random_state=seed)),
            {"clf__max_depth": [3, 5, None],
             "clf__min_samples_leaf": [1, 3]},
        ),
        "RandomForest": (
            pipe(RandomForestClassifier(n_estimators=300, n_jobs=1,
                                        random_state=seed)),
            {"clf__max_depth": [None, 5],
             "clf__min_samples_leaf": [1, 3]},
        ),
        "GradientBoosting": (
            pipe(GradientBoostingClassifier(random_state=seed)),
            {"clf__n_estimators": [100, 200],
             "clf__learning_rate": [0.05, 0.1],
             "clf__max_depth": [2, 3]},
        ),
        "XGBoost": (
            pipe(XGBClassifier(n_estimators=200, subsample=0.8,
                               colsample_bytree=0.8, eval_metric="logloss",
                               n_jobs=1, random_state=seed,
                               tree_method="hist")),
            {"clf__learning_rate": [0.05, 0.1],
             "clf__max_depth": [2, 3]},
        ),
        "LightGBM": (
            pipe(LGBMClassifier(n_estimators=200, min_child_samples=5,
                                n_jobs=1, random_state=seed, verbose=-1)),
            {"clf__learning_rate": [0.05, 0.1],
             "clf__num_leaves": [7, 15]},
        ),
        "CatBoost": (
            pipe(CatBoostClassifier(iterations=200, verbose=0,
                                    allow_writing_files=False,
                                    thread_count=1, random_seed=seed)),
            {"clf__depth": [2, 4],
             "clf__learning_rate": [0.05, 0.1]},
        ),
    }
    return models
