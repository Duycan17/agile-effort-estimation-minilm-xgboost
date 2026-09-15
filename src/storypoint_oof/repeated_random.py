"""Repeated random 80/20 evaluation with leakage-safe inner OOF stacking."""

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
from scipy.stats import t

from storypoint_oof.cli import PROJECT_ROOT
from storypoint_oof.data import StoryPointDataset, load_tawos
from storypoint_oof.embeddings import EmbeddingConfig, load_or_create_embeddings
from storypoint_oof.experiment import ExperimentConfig, run_experiment


@dataclass(frozen=True)
class RepeatedRandomConfig:
    repetitions: int = 30
    first_seed: int = 42
    test_size: float = 0.2
    inner_folds: int = 5
    target_transform: str = "log1p"
    text_ridge_alpha: float = 10.0
    meta_ridge_alpha: float = 1.0
    xgb_estimators: int = 300


def corrected_resampled_t_test(
    differences: np.ndarray, test_size: float, alpha: float = 0.05
) -> dict[str, Any]:
    """Nadeau-Bengio corrected test for repeated random train/test splits."""
    differences = np.asarray(differences, dtype=float)
    if len(differences) < 2:
        raise ValueError("At least two repeated splits are required")
    mean = float(differences.mean())
    variance = float(differences.var(ddof=1))
    train_size = 1 - test_size
    correction = 1 / len(differences) + test_size / train_size
    standard_error = float(np.sqrt(correction * variance))
    if standard_error == 0:
        statistic = float("-inf") if mean < 0 else float("inf")
    else:
        statistic = mean / standard_error
    degrees_of_freedom = len(differences) - 1
    critical = float(t.ppf(1 - alpha / 2, degrees_of_freedom))
    return {
        "primary_estimand": (
            "mean repeated-split MAE(OOF Stacking Ridge) - MAE(Tabular XGBoost)"
        ),
        "negative_favors_stacking": True,
        "mean_difference": mean,
        "median_difference": float(np.median(differences)),
        "standard_deviation_across_splits": float(np.sqrt(variance)),
        "corrected_standard_error": standard_error,
        "variance_correction": correction,
        "t_statistic": float(statistic),
        "degrees_of_freedom": degrees_of_freedom,
        "ci_95_low": mean - critical * standard_error,
        "ci_95_high": mean + critical * standard_error,
        "two_sided_p_value": float(2 * t.sf(abs(statistic), degrees_of_freedom)),
        "one_sided_p_value_stacking_better": float(
            t.cdf(statistic, degrees_of_freedom)
        ),
        "splits_stacking_better": int(np.count_nonzero(differences < 0)),
        "splits_xgboost_better": int(np.count_nonzero(differences > 0)),
    }


def run_repeated_random(
    dataset: StoryPointDataset,
    embeddings: np.ndarray,
    config: RepeatedRandomConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metric_frames = []
    prediction_frames = []
    for repetition in range(config.repetitions):
        seed = config.first_seed + repetition
        experiment_config = ExperimentConfig(
            split_strategy="random",
            test_size=config.test_size,
            folds=config.inner_folds,
            seed=seed,
            text_ridge_alpha=config.text_ridge_alpha,
            meta_ridge_alpha=config.meta_ridge_alpha,
            xgb_estimators=config.xgb_estimators,
            bootstrap_repetitions=10,
            target_transform=config.target_transform,
        )
        result = run_experiment(dataset, embeddings, experiment_config)
        metrics = result.metrics.copy()
        metrics.insert(0, "seed", seed)
        metric_frames.append(metrics)
        predictions = result.test_predictions.copy()
        predictions.insert(0, "seed", seed)
        prediction_frames.append(predictions)
        selected = metrics.set_index("model")
        difference = (
            selected.loc["OOF Stacking Ridge", "MAE"]
            - selected.loc["Tabular XGBoost", "MAE"]
        )
        print(
            f"Completed split {repetition + 1}/{config.repetitions}, "
            f"seed={seed}, paired MAE difference={difference:+.6f}"
        )
    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pivot = metrics.pivot(index="seed", columns="model", values="MAE")
    differences = (pivot["OOF Stacking Ridge"] - pivot["Tabular XGBoost"]).to_numpy()
    inference = corrected_resampled_t_test(differences, config.test_size)
    mean_xgboost_mae = float(pivot["Tabular XGBoost"].mean())
    mean_stacking_mae = float(pivot["OOF Stacking Ridge"].mean())
    inference.update(
        {
            "mean_xgboost_mae": mean_xgboost_mae,
            "mean_stacking_mae": mean_stacking_mae,
            "relative_mae_reduction_percent": (
                100 * (mean_xgboost_mae - mean_stacking_mae) / mean_xgboost_mae
            ),
            "repetitions": config.repetitions,
            "first_seed": config.first_seed,
            "last_seed": config.first_seed + config.repetitions - 1,
            "target_scale": f"{config.target_transform}(story_point)",
            "outer_protocol": "repeated random issue-level 80/20 holdout",
            "inner_protocol": "five shuffled OOF folds on training issues only",
            "confirmatory_success": bool(
                inference["mean_difference"] < 0
                and inference["ci_95_high"] < 0
                and inference["two_sided_p_value"] < 0.05
            ),
            "claim_boundary": (
                "Generalization to new issues from the sampled project distribution; "
                "not generalization to unseen projects."
            ),
        }
    )
    return metrics, predictions, inference


def _save_figure(metrics: pd.DataFrame, inference: dict[str, Any], path: Path) -> None:
    pivot = metrics.pivot(index="seed", columns="model", values="MAE")
    differences = pivot["OOF Stacking Ridge"] - pivot["Tabular XGBoost"]
    colors = np.where(differences < 0, "#228833", "#EE6677")
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.bar(differences.index.astype(str), differences, color=colors)
    ax.axhline(0, color="#333333", linestyle="--")
    ax.axhline(
        inference["mean_difference"],
        color="#AA3377",
        label=f"Mean: {inference['mean_difference']:.4f}",
    )
    ax.set_xlabel("Random split seed")
    ax.set_ylabel("MAE difference: stacking − XGBoost")
    ax.set_title("Repeated random 80/20 paired effects")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.legend(frameon=False, loc="upper right")
    for extension in ("png", "pdf"):
        fig.savefig(path / f"paired_effects_by_seed.{extension}", dpi=300)
    plt.close(fig)


def write_artifacts(
    output_dir: Path,
    dataset: StoryPointDataset,
    embedding_metadata: dict[str, Any],
    config: RepeatedRandomConfig,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    inference: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    figures = output_dir / "figures"
    figures.mkdir()
    metrics.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions_by_seed.csv", index=False)
    summary = (
        metrics.groupby("model")[["MAE", "MdAE", "MSE", "RMSE", "R2"]]
        .agg(["mean", "std"])
        .sort_values(("MAE", "mean"))
    )
    summary.to_csv(output_dir / "metrics_summary.csv")
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
            "random_issue_split": True,
            "outer_test_untouched_within_each_repetition": True,
            "meta_learner_uses_inner_oof_predictions_only": True,
            "hyperparameters_fixed_before_repeated_evaluation": True,
            "corrected_resampled_t_test": True,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _save_figure(metrics, inference, figures)
    conclusion = (
        "The fixed superiority criterion was met."
        if inference["confirmatory_success"]
        else "The fixed superiority criterion was not met."
    )
    (output_dir / "SUMMARY.md").write_text(
        "# Repeated random-split evaluation\n\n"
        f"{conclusion}\n\n"
        f"Mean paired MAE difference: **{inference['mean_difference']:.4f}**; "
        f"corrected 95% CI [{inference['ci_95_low']:.4f}, "
        f"{inference['ci_95_high']:.4f}]; corrected two-sided p = "
        f"{inference['two_sided_p_value']:.4g}.\n",
        encoding="utf-8",
    )


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
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output path already exists: {args.output_dir}")
    dataset = load_tawos(args.data, project="all")
    embedding_config = EmbeddingConfig(
        batch_size=args.batch_size, device=args.device, seed=args.first_seed
    )
    embeddings, embedding_metadata = load_or_create_embeddings(
        dataset.row_ids, dataset.texts, embedding_config, args.cache_dir
    )
    config = RepeatedRandomConfig(
        repetitions=args.repetitions, first_seed=args.first_seed
    )
    metrics, predictions, inference = run_repeated_random(dataset, embeddings, config)
    write_artifacts(
        args.output_dir,
        dataset,
        embedding_metadata,
        config,
        metrics,
        predictions,
        inference,
    )
    print(json.dumps(inference, indent=2))
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
