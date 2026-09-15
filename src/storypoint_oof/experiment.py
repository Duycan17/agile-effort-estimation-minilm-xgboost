"""Manual OOF stacking and complete reproducibility artifact bundle."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from storypoint_oof.data import StoryPointDataset
from storypoint_oof.evaluation import (
    evaluate_predictions,
    save_figures,
    save_reference_comparison_figure,
)
from storypoint_oof.models import (
    fit_tabular_model_with_history,
    make_meta_model,
    make_tabular_model,
    make_text_model,
)
from storypoint_oof.splitting import SplitPlan, make_split_plan

TABLE3_TAWOS_REFERENCE = {
    "Gradient Boosting": (0.4369, 0.3553, 0.3360, 0.5797, 0.2482),
    "XGBoost": (0.4230, 0.3356, 0.3136, 0.5600, 0.2983),
    "LightGBM": (0.4177, 0.3332, 0.3031, 0.5506, 0.3218),
    "CatBoost": (0.4254, 0.3384, 0.3158, 0.5620, 0.2934),
    "Histogram Gradient Boosting": (0.4253, 0.3378, 0.3184, 0.5642, 0.2877),
}


@dataclass(frozen=True)
class ExperimentConfig:
    split_strategy: str = "group"
    test_size: float = 0.2
    folds: int = 5
    seed: int = 42
    text_ridge_alpha: float = 10.0
    meta_ridge_alpha: float = 1.0
    xgb_estimators: int = 300
    bootstrap_repetitions: int = 2000
    target_transform: str = "raw"


@dataclass
class ExperimentResult:
    plan: SplitPlan
    metrics: pd.DataFrame
    comparisons: pd.DataFrame
    raw_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    training_history: pd.DataFrame
    fold_metrics: pd.DataFrame
    ridge_training_history: pd.DataFrame
    ridge_fit_summary: pd.DataFrame
    models: dict[str, Any]


def _history_frame(
    history: dict[str, dict[str, list[float]]], fold: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dataset_names = {"validation_0": "train", "validation_1": "validation"}
    for dataset_name, metrics in history.items():
        for metric, values in metrics.items():
            rows.extend(
                {
                    "fit": fold,
                    "dataset": dataset_names.get(dataset_name, dataset_name),
                    "metric": metric,
                    "boosting_round": boosting_round,
                    "value": float(value),
                }
                for boosting_round, value in enumerate(values, start=1)
            )
    return pd.DataFrame(rows)


def summarize_xgboost_history(history: pd.DataFrame) -> pd.DataFrame:
    """Summarize the best and final monitored round for each XGBoost fit."""
    rows: list[dict[str, Any]] = []
    for fit_name in sorted(history["fit"].unique()):
        for metric in sorted(history["metric"].unique()):
            selected = history.loc[
                (history["fit"] == fit_name) & (history["metric"] == metric)
            ]
            train = selected.loc[selected["dataset"] == "train"].set_index(
                "boosting_round"
            )["value"]
            validation = selected.loc[selected["dataset"] == "validation"].set_index(
                "boosting_round"
            )["value"]
            if validation.empty:
                best_round = int(train.index[-1])
                best_validation = np.nan
                train_at_best = float(train.iloc[-1])
            else:
                best_round = int(validation.idxmin())
                best_validation = float(validation.loc[best_round])
                train_at_best = float(train.loc[best_round])
            rows.append(
                {
                    "fit": fit_name,
                    "metric": metric,
                    "best_monitored_round": best_round,
                    "train_at_best_round": train_at_best,
                    "validation_at_best_round": best_validation,
                    "generalization_gap_at_best": best_validation - train_at_best,
                    "final_round": int(train.index[-1]),
                    "final_train_value": float(train.iloc[-1]),
                    "final_validation_value": (
                        float(validation.iloc[-1]) if not validation.empty else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _ridge_history_frame(model: Any, component: str, fit_name: str) -> pd.DataFrame:
    history = model.named_steps["regressor"].loss_history_.copy()
    history.insert(0, "fit", fit_name)
    history.insert(0, "component", component)
    return history


def _ridge_fit_row(model: Any, component: str, fit_name: str) -> dict[str, Any]:
    regressor = model.named_steps["regressor"]
    history = regressor.loss_history_
    return {
        "component": component,
        "fit": fit_name,
        "optimization_success": regressor.optimization_success_,
        "optimizer_iterations": regressor.n_iter_,
        "recorded_states": len(history),
        "initial_objective_per_sample": float(history["objective_per_sample"].iloc[0]),
        "final_objective_per_sample": float(history["objective_per_sample"].iloc[-1]),
        "objective_reduction_percent": float(
            100
            * (
                1
                - history["objective_per_sample"].iloc[-1]
                / history["objective_per_sample"].iloc[0]
            )
        ),
        "optimizer_message": regressor.optimization_message_,
    }


def run_experiment(
    dataset: StoryPointDataset,
    embeddings: np.ndarray,
    config: ExperimentConfig,
) -> ExperimentResult:
    if embeddings.shape[0] != len(dataset.target):
        raise ValueError("Embedding rows do not align with dataset rows")
    if config.target_transform == "raw":
        model_target = dataset.target.copy()
    elif config.target_transform == "log1p":
        model_target = np.log1p(dataset.target)
    else:
        raise ValueError("target_transform must be 'raw' or 'log1p'")
    plan = make_split_plan(
        dataset.groups,
        strategy=config.split_strategy,
        test_size=config.test_size,
        folds=config.folds,
        seed=config.seed,
    )
    oof_text = np.full(len(dataset.target), np.nan)
    oof_tabular = np.full(len(dataset.target), np.nan)
    history_frames: list[pd.DataFrame] = []
    ridge_history_frames: list[pd.DataFrame] = []
    ridge_fit_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    for fold_number, fit, validation in plan.folds():
        text_model = make_text_model(config.text_ridge_alpha)
        tabular_model = make_tabular_model(config.seed, config.xgb_estimators)
        text_model.fit(embeddings[fit], model_target[fit])
        fold_label = f"fold_{fold_number}"
        ridge_history_frames.append(
            _ridge_history_frame(text_model, "text_ridge", fold_label)
        )
        ridge_fit_rows.append(_ridge_fit_row(text_model, "text_ridge", fold_label))
        tabular_model, fold_history = fit_tabular_model_with_history(
            tabular_model,
            dataset.tabular.iloc[fit],
            model_target[fit],
            dataset.tabular.iloc[validation],
            model_target[validation],
        )
        history_frames.append(_history_frame(fold_history, fold_label))
        text_validation = text_model.predict(embeddings[validation])
        tabular_validation = tabular_model.predict(dataset.tabular.iloc[validation])
        oof_text[validation] = text_validation
        oof_tabular[validation] = tabular_validation
        for model_name, prediction in (
            ("Text Ridge", text_validation),
            ("Tabular XGBoost", tabular_validation),
        ):
            errors = np.abs(model_target[validation] - prediction)
            fold_metric_rows.append(
                {
                    "fold": int(fold_number),
                    "model": model_name,
                    "n_validation": int(len(validation)),
                    "MAE": float(np.mean(errors)),
                    "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
                }
            )
    if (
        np.isnan(oof_text[plan.train_indices]).any()
        or np.isnan(oof_tabular[plan.train_indices]).any()
    ):
        raise RuntimeError("OOF predictions are incomplete")

    meta_features = np.column_stack([oof_text, oof_tabular])
    meta_model = make_meta_model(config.meta_ridge_alpha)
    meta_model.fit(meta_features[plan.train_indices], model_target[plan.train_indices])
    ridge_history_frames.append(
        _ridge_history_frame(meta_model, "stacking_ridge", "oof_meta_fit")
    )
    ridge_fit_rows.append(_ridge_fit_row(meta_model, "stacking_ridge", "oof_meta_fit"))

    final_text = make_text_model(config.text_ridge_alpha)
    final_tabular = make_tabular_model(config.seed, config.xgb_estimators)
    final_text.fit(embeddings[plan.train_indices], model_target[plan.train_indices])
    ridge_history_frames.append(
        _ridge_history_frame(final_text, "text_ridge", "final_full_train")
    )
    ridge_fit_rows.append(_ridge_fit_row(final_text, "text_ridge", "final_full_train"))
    final_tabular, final_history = fit_tabular_model_with_history(
        final_tabular,
        dataset.tabular.iloc[plan.train_indices],
        model_target[plan.train_indices],
    )
    history_frames.append(_history_frame(final_history, "final_full_train"))
    text_test = final_text.predict(embeddings[plan.test_indices])
    tabular_test = final_tabular.predict(dataset.tabular.iloc[plan.test_indices])
    test_meta_features = np.column_stack([text_test, tabular_test])
    stacking_test = meta_model.predict(test_meta_features)
    train_target = model_target[plan.train_indices]
    predictions = {
        "Training Mean": np.full(len(plan.test_indices), np.mean(train_target)),
        "Training Median": np.full(len(plan.test_indices), np.median(train_target)),
        "Text Ridge": np.maximum(text_test, 0),
        "Tabular XGBoost": np.maximum(tabular_test, 0),
        "Simple Average": np.maximum((text_test + tabular_test) / 2, 0),
        "OOF Stacking Ridge": np.maximum(stacking_test, 0),
    }
    y_test = model_target[plan.test_indices]
    metrics, comparisons = evaluate_predictions(
        y_test,
        predictions,
        dataset.groups[plan.test_indices],
        repetitions=config.bootstrap_repetitions,
        seed=config.seed,
    )
    if config.target_transform == "log1p":
        raw_predictions = {
            name: np.maximum(np.expm1(value), 0) for name, value in predictions.items()
        }
    else:
        raw_predictions = predictions
    raw_metrics, _ = evaluate_predictions(
        dataset.target[plan.test_indices],
        raw_predictions,
        dataset.groups[plan.test_indices],
        repetitions=config.bootstrap_repetitions,
        seed=config.seed,
    )
    oof_frame = pd.DataFrame(
        {
            "row_id": dataset.row_ids[plan.train_indices],
            "project_key": dataset.project_keys[plan.train_indices],
            "oof_fold": plan.fold_id[plan.train_indices],
            "observed_story_point": dataset.target[plan.train_indices],
            "observed_model_target": model_target[plan.train_indices],
            "text_oof_prediction": oof_text[plan.train_indices],
            "tabular_oof_prediction": oof_tabular[plan.train_indices],
        }
    )
    test_frame = pd.DataFrame(
        {
            "row_id": dataset.row_ids[plan.test_indices],
            "project_key": dataset.project_keys[plan.test_indices],
            "observed_model_target": y_test,
            "observed_story_point": dataset.target[plan.test_indices],
            **{name: value for name, value in predictions.items()},
            **{
                f"{name} (raw story points)": value
                for name, value in raw_predictions.items()
            },
        }
    )
    return ExperimentResult(
        plan=plan,
        metrics=metrics,
        comparisons=comparisons,
        raw_metrics=raw_metrics,
        oof_predictions=oof_frame,
        test_predictions=test_frame,
        training_history=pd.concat(history_frames, ignore_index=True),
        fold_metrics=pd.DataFrame(fold_metric_rows),
        ridge_training_history=pd.concat(ridge_history_frames, ignore_index=True),
        ridge_fit_summary=pd.DataFrame(ridge_fit_rows),
        models={
            "text_ridge": final_text,
            "tabular_xgboost": final_tabular,
            "stacking_ridge": meta_model,
        },
    )


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _table3_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ("MAE", "MdAE", "MSE", "RMSE", "R2")
    rows = [
        {
            "source": "paper Table 3",
            "model": model,
            **dict(zip(columns, values, strict=True)),
        }
        for model, values in TABLE3_TAWOS_REFERENCE.items()
    ]
    for model in ("Text Ridge", "Tabular XGBoost", "OOF Stacking Ridge"):
        observed = metrics.loc[metrics["model"] == model].iloc[0]
        rows.append(
            {
                "source": "this compatible run",
                "model": model,
                **{column: observed[column] for column in columns},
            }
        )
    return pd.DataFrame(rows)


def write_artifacts(
    output_dir: Path,
    dataset: StoryPointDataset,
    embedding_metadata: dict[str, Any],
    config: ExperimentConfig,
    result: ExperimentResult,
    *,
    smoke: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "models").mkdir()
    (output_dir / "figures").mkdir()
    (output_dir / "training_history").mkdir()
    config_payload = asdict(config)
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    split_frame = pd.DataFrame(
        {
            "row_id": dataset.row_ids,
            "project_key": dataset.project_keys,
            "outer_split": np.where(
                np.isin(np.arange(len(dataset.target)), result.plan.test_indices),
                "test",
                "train",
            ),
            "oof_fold": result.plan.fold_id,
        }
    )
    split_frame.to_csv(output_dir / "splits.csv", index=False)
    result.oof_predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    result.test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    result.metrics.to_csv(output_dir / "metrics.csv", index=False)
    result.raw_metrics.to_csv(output_dir / "metrics_raw_story_points.csv", index=False)
    result.comparisons.to_csv(output_dir / "comparisons.csv", index=False)
    result.training_history.to_csv(
        output_dir / "training_history" / "xgboost_history_long.csv", index=False
    )
    summarize_xgboost_history(result.training_history).to_csv(
        output_dir / "training_history" / "xgboost_history_summary.csv", index=False
    )
    result.fold_metrics.to_csv(
        output_dir / "training_history" / "oof_fold_metrics.csv", index=False
    )
    result.ridge_training_history.to_csv(
        output_dir / "training_history" / "ridge_history_long.csv", index=False
    )
    result.ridge_fit_summary.to_csv(
        output_dir / "training_history" / "ridge_fit_summary.csv", index=False
    )
    if config.target_transform == "log1p" and config.split_strategy == "random":
        table3_comparison = _table3_comparison(result.metrics)
        table3_comparison.to_csv(output_dir / "table3_comparison.csv", index=False)
        save_reference_comparison_figure(table3_comparison, output_dir / "figures")
    for name, model in result.models.items():
        joblib.dump(model, output_dir / "models" / f"{name}.joblib")
    save_figures(
        result.metrics,
        result.test_predictions["observed_model_target"].to_numpy(),
        {
            name: result.test_predictions[name].to_numpy()
            for name in (
                "Text Ridge",
                "Tabular XGBoost",
                "Simple Average",
                "OOF Stacking Ridge",
            )
        },
        output_dir / "figures",
        scale_label=(
            "log1p(story point)"
            if config.target_transform == "log1p"
            else "story points"
        ),
        oof_predictions=result.oof_predictions,
        fold_metrics=result.fold_metrics,
        training_history=result.training_history,
        ridge_training_history=result.ridge_training_history,
        models=result.models,
    )

    figure_captions = {
        "absolute_error_ecdf": (
            "Empirical cumulative distributions of absolute outer-test errors; "
            "curves nearer the upper-left indicate more accurate predictions."
        ),
        "mae_comparison": (
            "Outer-test MAE for all evaluated approaches with 95% clustered "
            "bootstrap confidence intervals."
        ),
        "oof_error_complementarity": (
            "Relationship between the text and tabular learners' absolute OOF "
            "errors, illustrating their complementary error patterns."
        ),
        "oof_fold_stability": (
            "Validation MAE of both base learners across the inner OOF folds."
        ),
        "pipeline_architecture": (
            "Leakage-safe OOF stacking workflow used in the proposed approach."
        ),
        "predicted_vs_observed": (
            "Observed versus predicted effort on the untouched outer test set."
        ),
        "prediction_distribution": (
            "Distributions of observed targets and model predictions on the "
            "outer test set."
        ),
        "residual_distribution": (
            "Outer-test residual distributions for the base learners and stacker."
        ),
        "residual_vs_fitted": (
            "Residual-versus-fitted diagnostics for heteroscedasticity and bias."
        ),
        "stacker_coefficients": (
            "Coefficients learned by Ridge from standardized OOF base predictions."
        ),
        "xgboost_feature_importance": (
            "Top tabular predictors according to the final XGBoost model's "
            "normalized feature importance."
        ),
        "xgboost_final_training_history": (
            "Training MAE and RMSE over boosting rounds for the final XGBoost fit."
        ),
        "xgboost_oof_learning_curves": (
            "Mean training and validation learning curves across OOF folds; shaded "
            "bands denote one standard deviation."
        ),
        "ridge_optimization_history": (
            "Optimization histories for the text Ridge base learner and Ridge "
            "stacking meta-learner."
        ),
    }
    inventory_rows = []
    for pdf_path in sorted((output_dir / "figures").glob("*.pdf")):
        inventory_rows.append(
            {
                "figure": pdf_path.stem,
                "pdf": pdf_path.name,
                "png": f"{pdf_path.stem}.png",
                "suggested_caption": figure_captions.get(pdf_path.stem, ""),
            }
        )
    pd.DataFrame(inventory_rows).to_csv(
        output_dir / "figures" / "figure_inventory.csv", index=False
    )
    figure_readme_lines = [
        "# Publication figure inventory",
        "",
        "Use PDF files for LaTeX/Overleaf and PNG files for quick preview.",
        "",
    ]
    for row in inventory_rows:
        figure_readme_lines.extend(
            [f"## `{row['figure']}`", "", str(row["suggested_caption"]), ""]
        )
    (output_dir / "figures" / "README.md").write_text(
        "\n".join(figure_readme_lines), encoding="utf-8"
    )

    history_readme = (
        "# Training-history artifacts\n\n"
        "`xgboost_history_long.csv` contains the actual per-boosting-round MAE "
        "and RMSE reported by XGBoost for every OOF training/validation fold and "
        "the final full-training fit. The final fit has no validation curve "
        "because the untouched outer test set is never used for monitoring.\n\n"
        "`xgboost_history_summary.csv` reports the best monitored round, final "
        "round, and train-validation gap for each fold and metric.\n\n"
        "`ridge_history_long.csv` contains the actual L-BFGS optimization states "
        "for every Text Ridge fold/final fit and the Stacking Ridge meta-fit. "
        "`ridge_fit_summary.csv` records convergence status and objective "
        "reduction. MiniLM remains frozen and has no training loss.\n"
    )
    (output_dir / "training_history" / "README.md").write_text(
        history_readme, encoding="utf-8"
    )

    packages = {}
    tracked_packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "torch",
        "transformers",
    )
    for package in tracked_packages:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke_test": smoke,
        "publication_evidence": not smoke and config.split_strategy == "group",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git": _git_state(),
        "data_audit": dataset.audit,
        "evaluation_target": {
            "transform": config.target_transform,
            "metrics_csv_scale": (
                "log1p(story_point)"
                if config.target_transform == "log1p"
                else "raw_story_points"
            ),
            "raw_metrics_file": "metrics_raw_story_points.csv",
        },
        "split_audit": result.plan.audit,
        "embedding": embedding_metadata,
        "leakage_controls": {
            "outer_test_untouched_until_final_evaluation": True,
            "meta_learner_trained_only_on_oof_predictions": True,
            "tabular_preprocessing_refit_inside_each_fold": True,
            "transformer_frozen_and_target_free": True,
            "post_estimation_and_identifier_predictors_excluded": True,
            "outer_test_excluded_from_xgboost_monitoring": True,
        },
        "training_history": {
            "xgboost": "training_history/xgboost_history_long.csv",
            "xgboost_summary": "training_history/xgboost_history_summary.csv",
            "oof_fold_metrics": "training_history/oof_fold_metrics.csv",
            "ridge": "training_history/ridge_history_long.csv",
            "ridge_summary": "training_history/ridge_fit_summary.csv",
            "encoder_epoch_history": None,
            "encoder_epoch_history_reason": "MiniLM encoder is frozen",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    best = result.metrics.sort_values("MAE").iloc[0]
    caveat = (
        "This is a smoke test on one small project and is not publication evidence."
        if smoke
        else (
            "This run reproduces the Table 3 comparison protocol; its random "
            "issue split does not establish unseen-project generalization."
            if config.split_strategy == "random"
            else "This run follows the declared cross-project protocol."
        )
    )
    metric_scale = (
        "log1p(story_point)"
        if config.target_transform == "log1p"
        else "raw story points"
    )
    summary = (
        "# Run summary\n\n"
        f"{caveat}\n\n"
        f"Best observed outer-test MAE: **{best['MAE']:.4f}** "
        f"({best['model']}) on "
        f"**{metric_scale}**. "
        "Model selection must not be based on this single "
        "test set.\n\n"
        "See `manifest.json` for provenance and leakage controls, `metrics.csv` for "
        "uncertainty intervals, and `comparisons.csv` for paired effects.\n"
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
