"""
models.py - Model definitions, pipelines, and training under one identical protocol.
Five model families:
1. Logistic Regression (Baseline)
2. K-Nearest Neighbours (KNN, scaled)
3. Decision Tree (depth tuned via CV)
4. Random Forest
5. HistGradientBoosting (Gradient Boosting)
"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from src.features import build_preprocessor

def get_model_pipelines(dt_max_depth: int = 10, random_state: int = 42):
    """
    Construct the 5 model pipelines using a shared preprocessing ColumnTransformer.
    Protocol:
    - Identical feature pipeline fitted only on training folds.
    - Scaled numeric inputs for distance/gradient based estimators.
    - Fixed random_state across stochastic algorithms.
    """
    preprocessor = build_preprocessor()

    pipelines = {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state, solver="lbfgs"))
        ]),
        "K-Nearest Neighbours": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", KNeighborsClassifier(n_neighbors=5, weights="uniform", n_jobs=-1))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", DecisionTreeClassifier(max_depth=dt_max_depth, min_samples_leaf=10, random_state=random_state))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=random_state))
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=12, random_state=random_state))
        ])
    }
    return pipelines

def tune_decision_tree_depth(X_train, y_train, depths=[3, 5, 8, 10, 12, 15, 20], cv_folds=5, random_state=42):
    """
    Tune Decision Tree depth using 5-fold cross-validation on the training set only.
    """
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DecisionTreeClassifier(min_samples_leaf=10, random_state=random_state))
    ])
    param_grid = {"classifier__max_depth": depths}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_params_["classifier__max_depth"], grid.best_score_, grid.cv_results_

def cross_validate_models(pipelines, X_train, y_train, cv_folds=5, scoring="roc_auc", random_state=42):
    """
    Cross-validate all candidate models on the training set.
    The held-out test set is NEVER touched during cross-validation.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_results = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_results[name] = {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "fold_scores": scores.tolist()
        }
    return cv_results
