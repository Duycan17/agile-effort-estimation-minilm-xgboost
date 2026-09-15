"""Paired, project-cluster-aware comparison of two regressors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _cluster_bootstrap(
    differences: np.ndarray,
    groups: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    unique = np.unique(groups)
    rows = {group: np.flatnonzero(groups == group) for group in unique}
    estimates = np.empty(repetitions)
    for repetition in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([rows[group] for group in sampled])
        estimates[repetition] = differences[index].mean()
    return estimates


def _cluster_sign_flip(
    differences: np.ndarray,
    groups: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Monte Carlo paired randomization test, flipping whole projects."""
    unique = np.unique(groups)
    cluster_sums = np.array([differences[groups == group].sum() for group in unique])
    observed = differences.mean()
    simulated = np.empty(repetitions)
    chunk_size = 10_000
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        signs = rng.choice((-1.0, 1.0), size=(stop - start, len(unique)))
        simulated[start:stop] = signs @ cluster_sums / len(differences)
    two_sided = (np.count_nonzero(np.abs(simulated) >= abs(observed)) + 1) / (
        repetitions + 1
    )
    one_sided = (np.count_nonzero(simulated <= observed) + 1) / (repetitions + 1)
    return float(two_sided), float(one_sided)


def compare_predictions(
    frame: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    repetitions: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray]:
    required = {"project_key", "observed_model_target", baseline, candidate}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    observed = frame["observed_model_target"].to_numpy(dtype=float)
    baseline_error = np.abs(observed - frame[baseline].to_numpy(dtype=float))
    candidate_error = np.abs(observed - frame[candidate].to_numpy(dtype=float))
    differences = candidate_error - baseline_error
    groups = frame["project_key"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    bootstrap = _cluster_bootstrap(differences, groups, repetitions, rng)
    permutation_two_sided, permutation_one_sided = _cluster_sign_flip(
        differences, groups, repetitions, rng
    )

    project_rows = []
    for project in sorted(np.unique(groups)):
        values = differences[groups == project]
        project_rows.append(
            {
                "project_key": project,
                "issues": len(values),
                "mean_paired_absolute_error_difference": values.mean(),
                "median_paired_absolute_error_difference": np.median(values),
                "candidate_wins_fraction": np.mean(values < 0),
            }
        )
    projects = pd.DataFrame(project_rows)
    project_wilcoxon = wilcoxon(
        projects["mean_paired_absolute_error_difference"],
        alternative="two-sided",
        zero_method="wilcox",
    )
    baseline_mae = float(baseline_error.mean())
    candidate_mae = float(candidate_error.mean())
    result: dict[str, object] = {
        "estimand": f"MAE({candidate}) - MAE({baseline})",
        "negative_favors_candidate": True,
        "target_scale": "log1p(story_point)",
        "test_issues": len(frame),
        "test_projects": int(len(np.unique(groups))),
        "baseline_mae": baseline_mae,
        "candidate_mae": candidate_mae,
        "observed_difference": float(differences.mean()),
        "relative_mae_reduction_percent": float(
            100 * (baseline_mae - candidate_mae) / baseline_mae
        ),
        "cluster_bootstrap": {
            "repetitions": repetitions,
            "seed": seed,
            "ci_method": "project-cluster percentile bootstrap",
            "ci_95_low": float(np.percentile(bootstrap, 2.5)),
            "ci_95_high": float(np.percentile(bootstrap, 97.5)),
            "bootstrap_fraction_below_zero": float(np.mean(bootstrap < 0)),
        },
        "primary_cluster_sign_flip_test": {
            "repetitions": repetitions,
            "seed": seed,
            "two_sided_p_value": permutation_two_sided,
            "one_sided_p_value_candidate_better": permutation_one_sided,
            "unit_of_sign_flip": "project",
        },
        "project_level_sensitivity": {
            "estimand": "median of project-level mean error differences",
            "projects_candidate_better": int(
                np.count_nonzero(projects["mean_paired_absolute_error_difference"] < 0)
            ),
            "projects_baseline_better": int(
                np.count_nonzero(projects["mean_paired_absolute_error_difference"] > 0)
            ),
            "median_project_difference": float(
                projects["mean_paired_absolute_error_difference"].median()
            ),
            "wilcoxon_two_sided_p_value": float(project_wilcoxon.pvalue),
        },
        "interpretation_guardrail": (
            "This is a post-hoc analysis of one random split. Statistical evidence "
            "does not establish stability across seeds or unseen projects."
        ),
    }
    return result, projects, bootstrap


def write_results(
    result: dict[str, object],
    projects: pd.DataFrame,
    bootstrap: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stacking_vs_xgboost_statistics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    projects.to_csv(output_dir / "stacking_vs_xgboost_by_project.csv", index=False)

    observed = float(result["observed_difference"])
    bootstrap_result = result["cluster_bootstrap"]
    assert isinstance(bootstrap_result, dict)
    fig, ax = plt.subplots(figsize=(6.5, 3.7), constrained_layout=True)
    ax.hist(bootstrap, bins=60, color="#4477AA", alpha=0.85)
    ax.axvline(0, color="#333333", linestyle="--", label="No difference")
    ax.axvline(observed, color="#AA3377", label=f"Observed: {observed:.4f}")
    ax.set_xlabel("MAE difference: OOF stacking − XGBoost")
    ax.set_ylabel("Project-cluster bootstrap samples")
    ax.set_title(
        "Uncertainty of paired MAE difference\n"
        f"95% CI [{bootstrap_result['ci_95_low']:.4f}, "
        f"{bootstrap_result['ci_95_high']:.4f}]"
    )
    ax.legend(frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"stacking_vs_xgboost_bootstrap.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", default="Tabular XGBoost")
    parser.add_argument("--candidate", default="OOF Stacking Ridge")
    parser.add_argument("--repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.predictions)
    result, projects, bootstrap = compare_predictions(
        frame,
        baseline=args.baseline,
        candidate=args.candidate,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    write_results(result, projects, bootstrap, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
