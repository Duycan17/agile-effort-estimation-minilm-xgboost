"""Fold-local base learners and low-flexibility meta-learner."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from storypoint_oof.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


class TrackedRidge(BaseEstimator, RegressorMixin):
    """Ridge regression with an auditable L-BFGS optimization trajectory."""

    def __init__(self, alpha: float = 1.0, max_iter: int = 500, tol: float = 1e-10):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x: Any, y: Any) -> TrackedRidge:
        features = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if features.ndim != 2 or target.ndim != 1:
            raise ValueError("TrackedRidge expects a 2D X and a 1D y")
        if len(features) != len(target):
            raise ValueError("X and y have inconsistent row counts")
        n_rows, n_features = features.shape

        def components(parameters: np.ndarray) -> tuple[float, float, float]:
            intercept = parameters[0]
            coefficients = parameters[1:]
            residual = features @ coefficients + intercept - target
            mse = float(np.mean(np.square(residual)))
            penalty = float(self.alpha * np.dot(coefficients, coefficients) / n_rows)
            return mse, penalty, mse + penalty

        def objective_and_gradient(
            parameters: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            intercept = parameters[0]
            coefficients = parameters[1:]
            residual = features @ coefficients + intercept - target
            objective = float(
                np.dot(residual, residual)
                + self.alpha * np.dot(coefficients, coefficients)
            )
            gradient = np.empty_like(parameters)
            gradient[0] = 2.0 * residual.sum()
            gradient[1:] = 2.0 * (features.T @ residual + self.alpha * coefficients)
            return objective, gradient

        initial = np.zeros(n_features + 1, dtype=np.float64)
        initial[0] = float(target.mean())
        history: list[dict[str, float | int]] = []

        def record(parameters: np.ndarray) -> None:
            mse, penalty, objective = components(parameters)
            history.append(
                {
                    "iteration": len(history),
                    "mse_loss": mse,
                    "ridge_penalty": penalty,
                    "objective_per_sample": objective,
                }
            )

        record(initial)
        result = minimize(
            objective_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            callback=record,
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )
        if not np.allclose(result.x, initial) and (
            len(history) == 1
            or not np.allclose(
                history[-1]["objective_per_sample"], components(result.x)[2]
            )
        ):
            record(result.x)
        self.intercept_ = float(result.x[0])
        self.coef_ = np.asarray(result.x[1:], dtype=np.float64)
        self.n_features_in_ = n_features
        self.loss_history_ = pd.DataFrame(history)
        self.optimization_success_ = bool(result.success)
        self.optimization_message_ = str(result.message)
        self.n_iter_ = int(result.nit)
        return self

    def predict(self, x: Any) -> np.ndarray:
        features = np.asarray(x, dtype=np.float64)
        return features @ self.coef_ + self.intercept_


def make_text_model(alpha: float) -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("regressor", TrackedRidge(alpha=alpha))]
    )


def make_tabular_model(seed: int, n_estimators: int) -> Pipeline:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    preprocessing = ColumnTransformer(
        [
            ("categorical", categorical, list(CATEGORICAL_COLUMNS)),
            ("numeric", numeric, list(NUMERIC_COLUMNS)),
        ],
        remainder="drop",
    )
    regressor = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        eval_metric=["mae", "rmse"],
    )
    return Pipeline([("preprocessing", preprocessing), ("regressor", regressor)])


def fit_tabular_model_with_history(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: Any,
    x_validation: pd.DataFrame | None = None,
    y_validation: Any | None = None,
) -> tuple[Pipeline, dict[str, dict[str, list[float]]]]:
    """Fit a tabular pipeline and retain XGBoost's per-round evaluation history.

    Validation data are transformed by the preprocessor fitted on the training
    fold. No early stopping is used, so recording the metrics does not alter the
    declared number of boosting rounds or the experimental protocol.
    """
    preprocessing = model.named_steps["preprocessing"]
    regressor = model.named_steps["regressor"]
    transformed_train = preprocessing.fit_transform(x_train, y_train)
    eval_set = [(transformed_train, y_train)]
    if (x_validation is None) != (y_validation is None):
        raise ValueError("x_validation and y_validation must be provided together")
    if x_validation is not None:
        transformed_validation = preprocessing.transform(x_validation)
        eval_set.append((transformed_validation, y_validation))
    regressor.fit(transformed_train, y_train, eval_set=eval_set, verbose=False)
    return model, regressor.evals_result()


def make_meta_model(alpha: float) -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("regressor", TrackedRidge(alpha=alpha))]
    )
