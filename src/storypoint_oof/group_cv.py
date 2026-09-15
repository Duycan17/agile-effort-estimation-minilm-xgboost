"""Five-fold cross-project evaluation of OOF stacking versus XGBoost."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold

from storypoint_oof.cli import PROJECT_ROOT
from storypoint_oof.data import StoryPointDataset, load_tawos
from storypoint_oof.embeddings import EmbeddingConfig, load_or_create_embeddings
from storypoint_oof.evaluation import point_metrics
from storypoint_oof.models import make_meta_model, make_tabular_model, make_text_model

MODEL_NAMES = (
    "Text Ridge",
    "Tabular XGBoost",
    "Simple Average",
    "OOF Stacking Ridge",
)


@dataclass(frozen=True)
class GroupCVConfig:
    outer_folds: int = 5
    inner_folds: int = 5
    seed: int = 42
    target_transform: str = "log1p"
    text_ridge_alpha: float = 10.0
    meta_ridge_alpha: float = 1.0
    xgb_estimators: int = 300
    inference_repetitions: int = 100_000


def _target(dataset: StoryPointDataset, transform: str) -> np.ndarray:
    if transform == "log1p":
        return np.log1p(dataset.target)
    if transform == "raw":
        return dataset.target.copy()
    raise ValueError("target_transform must be 'raw' or 'log1p'")


def _fit_outer_fold(
    dataset: StoryPointDataset,
    embeddings: np.ndarray,
    target: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    config: GroupCVConfig,
    outer_fold: int,
) -> dict[str, np.ndarray]:
    inner = GroupKFold(
        n_splits=config.inner_folds,
        shuffle=True,
        random_state=config.seed + outer_fold,
    )
    oof_text = np.full(len(outer_train), np.nan)
    oof_tabular = np.full(len(outer_train), np.nan)
    train_groups = dataset.groups[outer_train]
    for fit_relative, validation_relative in inner.split(
        outer_train, groups=train_groups
    ):
        fit = outer_train[fit_relative]
        validation = outer_train[validation_relative]
        if set(dataset.groups[fit]) & set(dataset.groups[validation]):
            raise RuntimeError("Inner project leakage detected")
        text_model = make_text_model(config.text_ridge_alpha)
        tabular_model = make_tabular_model(config.seed, config.xgb_estimators)
        text_model.fit(embeddings[fit], target[fit])
        tabular_model.fit(dataset.tabular.iloc[fit], target[fit])
        oof_text[validation_relative] = text_model.predict(embeddings[validation])
        oof_tabular[validation_relative] = tabular_model.predict(
            dataset.tabular.iloc[validation]
        )
    if np.isnan(oof_text).any() or np.isnan(oof_tabular).any():
        raise RuntimeError("Inner OOF predictions are incomplete")

    meta = make_meta_model(config.meta_ridge_alpha)
    meta.fit(np.column_stack([oof_text, oof_tabular]), target[outer_train])
    final_text = make_text_model(config.text_ridge_alpha)
    final_tabular = make_tabular_model(config.seed, config.xgb_estimators)
    final_text.fit(embeddings[outer_train], target[outer_train])
    final_tabular.fit(dataset.tabular.iloc[outer_train], target[outer_train])
    text_prediction = final_text.predict(embeddings[outer_test])
    tabular_prediction = final_tabular.predict(dataset.tabular.iloc[outer_test])
    stacking_prediction = meta.predict(
        np.column_stack([text_prediction, tabular_prediction])
    )
    return {
        "Text Ridge": np.maximum(text_prediction, 0),
        "Tabular XGBoost": np.maximum(tabular_prediction, 0),
        "Simple Average": np.maximum((text_prediction + tabular_prediction) / 2, 0),
        "OOF Stacking Ridge": np.maximum(stacking_prediction, 0),
    }


def cross_project_predictions(
    dataset: StoryPointDataset,
    embeddings: np.ndarray,
    config: GroupCVConfig,
) -> pd.DataFrame:
    if len(embeddings) != len(dataset.target):
        raise ValueError("Embedding rows do not align with dataset rows")
    target = _target(dataset, config.target_transform)
    predictions = {name: np.full(len(target), np.nan) for name in MODEL_NAMES}
    outer_fold_ids = np.full(len(target), -1, dtype=int)
    outer = GroupKFold(
        n_splits=config.outer_folds,
        shuffle=True,
        random_state=config.seed,
    )
    for outer_fold, (train, test) in enumerate(
        outer.split(np.arange(len(target)), groups=dataset.groups)
    ):
        train = np.asarray(train, dtype=int)
        test = np.asarray(test, dtype=int)
        if set(dataset.groups[train]) & set(dataset.groups[test]):
            raise RuntimeError("Outer project leakage detected")
        fold_predictions = _fit_outer_fold(
            dataset, embeddings, target, train, test, config, outer_fold
        )
        for name, values in fold_predictions.items():
            predictions[name][test] = values
        outer_fold_ids[test] = outer_fold
        print(
            f"Completed outer fold {outer_fold + 1}/{config.outer_folds}: "
            f"train={len(train)}, test={len(test)}, "
            f"test_projects={len(np.unique(dataset.groups[test]))}"
        )
    if outer_fold_ids.min() < 0 or any(
        np.isnan(values).any() for values in predictions.values()
    ):
        raise RuntimeError("Outer predictions are incomplete")
    frame = pd.DataFrame(
        {
            "row_id": dataset.row_ids,
            "project_key": dataset.project_keys,
            "outer_fold": outer_fold_ids,
            "observed_model_target": target,
            "observed_story_point": dataset.target,
            **predictions,
        }
    )
    if config.target_transform == "log1p":
        for name in MODEL_NAMES:
            frame[f"{name} (raw story points)"] = np.maximum(np.expm1(frame[name]), 0)
    return frame


def _equal_project_inference(
    projects: pd.DataFrame, repetitions: int, seed: int
) -> tuple[dict[str, Any], np.ndarray]:
    difference = projects["MAE_difference_stacking_minus_xgboost"].to_numpy()
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(difference), size=(repetitions, len(difference)), endpoint=False
    )
    bootstrap = difference[sampled].mean(axis=1)
    signs = rng.choice((-1.0, 1.0), size=(repetitions, len(difference)))
    randomized = (signs * difference).mean(axis=1)
    observed = float(difference.mean())
    p_two_sided = (np.count_nonzero(np.abs(randomized) >= abs(observed)) + 1) / (
        repetitions + 1
    )
    p_one_sided = (np.count_nonzero(randomized <= observed) + 1) / (repetitions + 1)
    signed_rank = wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
    return (
        {
            "primary_estimand": (
                "equal-project mean of MAE(OOF Stacking Ridge) - MAE(Tabular XGBoost)"
            ),
            "negative_favors_stacking": True,
            "projects": len(difference),
            "mean_difference": observed,
            "median_difference": float(np.median(difference)),
            "ci_95_low": float(np.percentile(bootstrap, 2.5)),
            "ci_95_high": float(np.percentile(bootstrap, 97.5)),
            "bootstrap_repetitions": repetitions,
            "sign_flip_two_sided_p_value": float(p_two_sided),
            "sign_flip_one_sided_p_value_stacking_better": float(p_one_sided),
            "wilcoxon_two_sided_p_value": float(signed_rank.pvalue),
            "projects_stacking_better": int(np.count_nonzero(difference < 0)),
            "projects_xgboost_better": int(np.count_nonzero(difference > 0)),
            "seed": seed,
            "confirmatory_success": bool(
                observed < 0
                and np.percentile(bootstrap, 97.5) < 0
                and p_two_sided < 0.05
            ),
        },
        bootstrap,
    )


def evaluate_cross_project_predictions(
    predictions: pd.DataFrame, config: GroupCVConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray]:
    observed = predictions["observed_model_target"].to_numpy()
    aggregate_rows = [
        {"model": name, **point_metrics(observed, predictions[name].to_numpy())}
        for name in MODEL_NAMES
    ]
    project_rows = []
    for project, frame in predictions.groupby("project_key", sort=True):
        row: dict[str, float | int | str] = {
            "project_key": str(project),
            "issues": len(frame),
            "outer_fold": int(frame["outer_fold"].iloc[0]),
        }
        y_project = frame["observed_model_target"].to_numpy()
        for name in MODEL_NAMES:
            row[f"MAE_{name}"] = point_metrics(y_project, frame[name].to_numpy())["MAE"]
        row["MAE_difference_stacking_minus_xgboost"] = (
            row["MAE_OOF Stacking Ridge"] - row["MAE_Tabular XGBoost"]
        )
        project_rows.append(row)
    projects = pd.DataFrame(project_rows)
    inference, bootstrap = _equal_project_inference(
        projects, config.inference_repetitions, config.seed
    )
    return pd.DataFrame(aggregate_rows), projects, inference, bootstrap


def _save_figures(
    projects: pd.DataFrame,
    inference: dict[str, Any],
    bootstrap: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = projects.sort_values("MAE_difference_stacking_minus_xgboost")
    colors = np.where(
        ordered["MAE_difference_stacking_minus_xgboost"] < 0,
        "#228833",
        "#EE6677",
    )
    fig, ax = plt.subplots(figsize=(7, 7.5), constrained_layout=True)
    ax.barh(
        ordered["project_key"],
        ordered["MAE_difference_stacking_minus_xgboost"],
        color=colors,
    )
    ax.axvline(0, color="#333333", linestyle="--")
    ax.set_xlabel("Project MAE difference: stacking − XGBoost")
    ax.set_title("Cross-project paired effects (negative favors stacking)")
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"project_effects.{extension}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.7), constrained_layout=True)
    ax.hist(bootstrap, bins=60, color="#4477AA", alpha=0.85)
    ax.axvline(0, color="#333333", linestyle="--", label="No difference")
    ax.axvline(
        inference["mean_difference"],
        color="#AA3377",
        label=f"Observed: {inference['mean_difference']:.4f}",
    )
    ax.set_xlabel("Equal-project mean MAE difference")
    ax.set_ylabel("Bootstrap samples")
    ax.set_title(
        f"95% CI [{inference['ci_95_low']:.4f}, {inference['ci_95_high']:.4f}]"
    )
    ax.legend(frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"project_bootstrap.{extension}", dpi=300)
    plt.close(fig)


def write_group_cv_artifacts(
    output_dir: Path,
    dataset: StoryPointDataset,
    embedding_metadata: dict[str, Any],
    config: GroupCVConfig,
    predictions: pd.DataFrame,
    aggregate: pd.DataFrame,
    projects: pd.DataFrame,
    inference: dict[str, Any],
    bootstrap: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    figures = output_dir / "figures"
    predictions.to_csv(output_dir / "cross_project_predictions.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    projects.to_csv(output_dir / "per_project_metrics.csv", index=False)
    (output_dir / "inference.json").write_text(
        json.dumps(inference, indent=2), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "data_audit": dataset.audit,
        "embedding": embedding_metadata,
        "protocol": {
            "outer_split": "5-fold shuffled GroupKFold by project_key",
            "inner_split": "5-fold shuffled GroupKFold on outer training only",
            "target": config.target_transform + "(story_point)",
            "primary_statistical_unit": "project",
            "hyperparameters_tuned_on_outer_predictions": False,
            "every_project_outer_test_once": True,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _save_figures(projects, inference, bootstrap, figures)
    conclusion = (
        "The prespecified superiority criterion was met."
        if inference["confirmatory_success"]
        else "The prespecified superiority criterion was not met."
    )
    summary = (
        "# Cross-project evaluation summary\n\n"
        f"{conclusion}\n\n"
        f"Equal-project mean paired MAE difference: "
        f"**{inference['mean_difference']:.4f}**; 95% CI "
        f"[{inference['ci_95_low']:.4f}, {inference['ci_95_high']:.4f}]; "
        f"two-sided sign-flip p = {inference['sign_flip_two_sided_p_value']:.4f}.\n"
    )
    (output_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to a local pre-estimation TAWOS CSV; the dataset is not distributed.",
    )
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-repetitions", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output path already exists: {args.output_dir}")
    dataset = load_tawos(args.data, project="all")
    embedding_config = EmbeddingConfig(
        model_name=args.model_name,
        revision=args.model_revision,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    embeddings, embedding_metadata = load_or_create_embeddings(
        dataset.row_ids, dataset.texts, embedding_config, args.cache_dir
    )
    config = GroupCVConfig(
        seed=args.seed, inference_repetitions=args.inference_repetitions
    )
    predictions = cross_project_predictions(dataset, embeddings, config)
    aggregate, projects, inference, bootstrap = evaluate_cross_project_predictions(
        predictions, config
    )
    write_group_cv_artifacts(
        args.output_dir,
        dataset,
        embedding_metadata,
        config,
        predictions,
        aggregate,
        projects,
        inference,
        bootstrap,
    )
    print(
        aggregate[["model", "MAE", "MdAE", "MSE", "RMSE", "R2"]].to_string(index=False)
    )
    print(json.dumps(inference, indent=2))
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
