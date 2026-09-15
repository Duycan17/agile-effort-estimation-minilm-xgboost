"""Point estimates, paired uncertainty, and publication figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

MODEL_COLORS = {
    "Training Mean": "#999999",
    "Training Median": "#666666",
    "Text Ridge": "#4477AA",
    "Tabular XGBoost": "#EE6677",
    "Simple Average": "#CCBB44",
    "OOF Stacking Ridge": "#228833",
}


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MdAE": float(median_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_percent": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def _resample_indices(
    groups: np.ndarray, repetitions: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    samples: list[np.ndarray] = []
    if len(unique_groups) < 2:
        for _ in range(repetitions):
            samples.append(rng.integers(0, len(groups), size=len(groups)))
        return samples
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for _ in range(repetitions):
        selected = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        samples.append(np.concatenate([group_rows[group] for group in selected]))
    return samples


def evaluate_predictions(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    groups: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = _resample_indices(groups, repetitions, seed)
    rows: list[dict[str, float | str]] = []
    for model, prediction in predictions.items():
        metrics = point_metrics(y_true, prediction)
        row: dict[str, float | str] = {"model": model, **metrics}
        for metric in ("MAE", "MdAE", "RMSE", "MAPE_percent"):
            values = [
                point_metrics(y_true[index], prediction[index])[metric]
                for index in samples
            ]
            row[f"{metric}_ci_low"] = float(np.percentile(values, 2.5))
            row[f"{metric}_ci_high"] = float(np.percentile(values, 97.5))
        rows.append(row)

    baseline_error = np.abs(y_true - predictions["Tabular XGBoost"])
    comparison_rows: list[dict[str, float | str]] = []
    for model in ("Text Ridge", "Simple Average", "OOF Stacking Ridge"):
        candidate_error = np.abs(y_true - predictions[model])
        observed = float(np.mean(candidate_error - baseline_error))
        bootstrap = [
            float(np.mean(candidate_error[index] - baseline_error[index]))
            for index in samples
        ]
        comparison_rows.append(
            {
                "candidate": model,
                "baseline": "Tabular XGBoost",
                "metric": "paired_MAE_difference",
                "difference": observed,
                "ci_low": float(np.percentile(bootstrap, 2.5)),
                "ci_high": float(np.percentile(bootstrap, 97.5)),
                "negative_favors_candidate": True,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(comparison_rows)


def save_figures(
    metrics: pd.DataFrame,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
    *,
    scale_label: str = "story points",
    oof_predictions: pd.DataFrame | None = None,
    fold_metrics: pd.DataFrame | None = None,
    training_history: pd.DataFrame | None = None,
    ridge_training_history: pd.DataFrame | None = None,
    models: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )

    ordered = metrics.sort_values("MAE", ascending=True)
    lower = ordered["MAE"] - ordered["MAE_ci_low"]
    upper = ordered["MAE_ci_high"] - ordered["MAE"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)
    ax.barh(
        ordered["model"],
        ordered["MAE"],
        xerr=np.vstack([lower, upper]),
        color=[MODEL_COLORS[name] for name in ordered["model"]],
        capsize=3,
    )
    ax.set_xlabel(f"Mean absolute error ({scale_label})")
    ax.set_title("Outer-test performance with 95% bootstrap intervals")
    ax.grid(axis="x", alpha=0.25)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"mae_comparison.{extension}", bbox_inches="tight")
    plt.close(fig)

    selected = ["Text Ridge", "Tabular XGBoost", "OOF Stacking Ridge"]
    minimum = min(
        float(y_true.min()), *(float(predictions[name].min()) for name in selected)
    )
    maximum = max(
        float(y_true.max()), *(float(predictions[name].max()) for name in selected)
    )
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    for ax, name in zip(axes, selected, strict=True):
        ax.scatter(
            y_true,
            predictions[name],
            s=20,
            alpha=0.75,
            color=MODEL_COLORS[name],
            edgecolor="white",
            linewidth=0.3,
        )
        ax.plot([minimum, maximum], [minimum, maximum], "--", color="#333333")
        ax.set_title(name)
        ax.set_xlabel(f"Observed ({scale_label})")
        ax.set_ylabel(f"Predicted ({scale_label})")
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"predicted_vs_observed.{extension}", bbox_inches="tight"
        )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)
    for name in selected:
        residual = y_true - predictions[name]
        ax.hist(
            residual,
            bins="auto",
            histtype="step",
            linewidth=1.6,
            label=name,
            color=MODEL_COLORS[name],
        )
    ax.axvline(0, linestyle="--", color="#333333", linewidth=1)
    ax.set_xlabel(f"Residual: observed - predicted ({scale_label})")
    ax.set_ylabel("Issues")
    ax.set_title("Outer-test residual distributions")
    ax.legend(frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"residual_distribution.{extension}", bbox_inches="tight"
        )
    plt.close(fig)

    _save_residual_vs_fitted(y_true, predictions, output_dir, scale_label)
    _save_absolute_error_ecdf(y_true, predictions, output_dir, scale_label)
    _save_prediction_distributions(y_true, predictions, output_dir, scale_label)
    if oof_predictions is not None:
        _save_oof_error_complementarity(oof_predictions, output_dir, scale_label)
    if fold_metrics is not None:
        _save_fold_stability(fold_metrics, output_dir, scale_label)
    if training_history is not None:
        _save_xgboost_learning_curves(training_history, output_dir, scale_label)
    if ridge_training_history is not None:
        _save_ridge_optimization_history(ridge_training_history, output_dir)
    if models is not None:
        _save_stacker_coefficients(models, output_dir)
        _save_xgboost_feature_importance(models, output_dir)

    _save_architecture_figure(output_dir)


def _save_both(fig: Any, output_dir: Path, stem: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight")
    plt.close(fig)


def _selected_predictions() -> tuple[str, ...]:
    return ("Text Ridge", "Tabular XGBoost", "OOF Stacking Ridge")


def _save_residual_vs_fitted(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
    scale_label: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1), constrained_layout=True)
    for ax, name in zip(axes, _selected_predictions(), strict=True):
        fitted = predictions[name]
        residual = y_true - fitted
        ax.hexbin(fitted, residual, gridsize=35, mincnt=1, cmap="Blues")
        ax.axhline(0, linestyle="--", color="#333333", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel(f"Fitted ({scale_label})")
        ax.set_ylabel(f"Residual ({scale_label})")
    _save_both(fig, output_dir, "residual_vs_fitted")


def _save_absolute_error_ecdf(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
    scale_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    for name in _selected_predictions():
        error = np.sort(np.abs(y_true - predictions[name]))
        cumulative = np.arange(1, len(error) + 1) / len(error)
        ax.plot(error, cumulative, label=name, color=MODEL_COLORS[name], linewidth=1.7)
    ax.set_xlabel(f"Absolute error ({scale_label})")
    ax.set_ylabel("Cumulative proportion of issues")
    ax.set_title("Outer-test absolute-error distribution")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_both(fig, output_dir, "absolute_error_ecdf")


def _save_prediction_distributions(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: Path,
    scale_label: str,
) -> None:
    values = np.concatenate(
        [y_true, *(predictions[name] for name in _selected_predictions())]
    )
    bins = np.histogram_bin_edges(values, bins="fd")
    if len(bins) > 81:
        bins = np.linspace(float(values.min()), float(values.max()), 81)
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    ax.hist(
        y_true,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2.1,
        color="#222222",
        label="Observed",
    )
    for name in _selected_predictions():
        ax.hist(
            predictions[name],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=MODEL_COLORS[name],
            label=name,
        )
    ax.set_xlabel(scale_label)
    ax.set_ylabel("Density")
    ax.set_title("Observed and predicted target distributions")
    ax.legend(frameon=False)
    _save_both(fig, output_dir, "prediction_distribution")


def _save_oof_error_complementarity(
    oof: pd.DataFrame, output_dir: Path, scale_label: str
) -> None:
    observed = oof["observed_model_target"].to_numpy()
    text_error = np.abs(observed - oof["text_oof_prediction"].to_numpy())
    tabular_error = np.abs(observed - oof["tabular_oof_prediction"].to_numpy())
    maximum = float(np.quantile(np.concatenate([text_error, tabular_error]), 0.99))
    fig, ax = plt.subplots(figsize=(5.2, 4.5), constrained_layout=True)
    density = ax.hexbin(
        tabular_error,
        text_error,
        gridsize=45,
        mincnt=1,
        cmap="viridis",
        extent=(0, maximum, 0, maximum),
    )
    ax.plot([0, maximum], [0, maximum], "--", color="#333333", linewidth=1)
    ax.set_xlim(0, maximum)
    ax.set_ylim(0, maximum)
    ax.set_xlabel(f"Tabular XGBoost OOF absolute error ({scale_label})")
    ax.set_ylabel(f"Text Ridge OOF absolute error ({scale_label})")
    ax.set_title("OOF error complementarity (99% display range)")
    fig.colorbar(density, ax=ax, label="Issues per hexagon")
    _save_both(fig, output_dir, "oof_error_complementarity")


def _save_fold_stability(
    fold_metrics: pd.DataFrame, output_dir: Path, scale_label: str
) -> None:
    models = ("Text Ridge", "Tabular XGBoost")
    folds = sorted(fold_metrics["fold"].unique())
    positions = np.arange(len(folds))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)
    for offset, name in zip((-width / 2, width / 2), models, strict=True):
        selected = fold_metrics.loc[fold_metrics["model"] == name].set_index("fold")
        ax.bar(
            positions + offset,
            selected.loc[folds, "MAE"],
            width,
            label=name,
            color=MODEL_COLORS[name],
        )
    ax.set_xticks(positions, [str(fold + 1) for fold in folds])
    ax.set_xlabel("OOF fold")
    ax.set_ylabel(f"Validation MAE ({scale_label})")
    ax.set_title("Base-learner stability across OOF folds")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save_both(fig, output_dir, "oof_fold_stability")


def _save_xgboost_learning_curves(
    history: pd.DataFrame, output_dir: Path, scale_label: str
) -> None:
    oof = history.loc[history["fit"] != "final_full_train"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for ax, metric in zip(axes, ("mae", "rmse"), strict=True):
        for dataset_name, color in (("train", "#4477AA"), ("validation", "#EE6677")):
            selected = oof.loc[
                (oof["metric"] == metric) & (oof["dataset"] == dataset_name)
            ]
            aggregate = selected.groupby("boosting_round")["value"].agg(["mean", "std"])
            rounds = aggregate.index.to_numpy()
            mean = aggregate["mean"].to_numpy()
            spread = aggregate["std"].fillna(0).to_numpy()
            ax.plot(rounds, mean, color=color, label=dataset_name.capitalize())
            ax.fill_between(
                rounds, mean - spread, mean + spread, color=color, alpha=0.16
            )
        ax.set_xlabel("Boosting round")
        ax.set_ylabel(f"{metric.upper()} ({scale_label})")
        ax.set_title(f"OOF XGBoost {metric.upper()} (mean $\\pm$ SD)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    _save_both(fig, output_dir, "xgboost_oof_learning_curves")

    final = history.loc[
        (history["fit"] == "final_full_train") & (history["dataset"] == "train")
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)
    for metric, color in (("mae", "#4477AA"), ("rmse", "#EE6677")):
        selected = final.loc[final["metric"] == metric]
        ax.plot(
            selected["boosting_round"],
            selected["value"],
            label=metric.upper(),
            color=color,
            linewidth=1.6,
        )
    ax.set_xlabel("Boosting round")
    ax.set_ylabel(f"Training metric ({scale_label})")
    ax.set_title("Final XGBoost fit on the complete outer-training set")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_both(fig, output_dir, "xgboost_final_training_history")


def _save_stacker_coefficients(models: dict[str, Any], output_dir: Path) -> None:
    stacker = models["stacking_ridge"]
    coefficients = stacker.named_steps["regressor"].coef_
    labels = ("Text Ridge prediction", "Tabular XGBoost prediction")
    fig, ax = plt.subplots(figsize=(5.7, 3.2), constrained_layout=True)
    bars = ax.barh(
        labels,
        coefficients,
        color=(MODEL_COLORS["Text Ridge"], MODEL_COLORS["Tabular XGBoost"]),
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Ridge coefficient on standardized OOF prediction")
    ax.set_title("Learned stacking meta-model weights")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    _save_both(fig, output_dir, "stacker_coefficients")


def _save_ridge_optimization_history(history: pd.DataFrame, output_dir: Path) -> None:
    # Stack the panels vertically so labels remain legible when the figure is
    # scaled to the single-column text width of an A4 proceedings paper.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.4), constrained_layout=True)
    panels = (
        ("text_ridge", "Text Ridge fits", MODEL_COLORS["Text Ridge"]),
        (
            "stacking_ridge",
            "Stacking Ridge meta-fit",
            MODEL_COLORS["OOF Stacking Ridge"],
        ),
    )
    for ax, (component, title, color) in zip(axes, panels, strict=True):
        selected = history.loc[history["component"] == component]
        for fit_name, trajectory in selected.groupby("fit", sort=True):
            normalized = (
                trajectory["objective_per_sample"]
                / trajectory["objective_per_sample"].iloc[0]
            )
            is_final = fit_name == "final_full_train"
            is_meta = component == "stacking_ridge"
            ax.plot(
                trajectory["iteration"],
                normalized,
                color=color,
                alpha=1.0 if is_final or is_meta else 0.32,
                linewidth=2.8 if is_final or is_meta else 1.6,
                marker="o" if is_meta else None,
                markersize=6 if is_meta else None,
                label=(
                    "Final full-training fit"
                    if is_final
                    else "OOF fold fits"
                    if fit_name == "fold_0"
                    else "Ridge meta-fit"
                    if is_meta
                    else "_nolegend_"
                ),
            )
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Normalized Ridge objective", fontsize=12)
        ax.set_title(title, fontsize=13, pad=7)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.legend(frameon=False, fontsize=10.5, loc="best")
        ax.margins(x=0.03)
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _save_both(fig, output_dir, "ridge_optimization_history")


def _save_xgboost_feature_importance(
    models: dict[str, Any], output_dir: Path, top_n: int = 20
) -> None:
    pipeline = models["tabular_xgboost"]
    preprocessing = pipeline.named_steps["preprocessing"]
    regressor = pipeline.named_steps["regressor"]
    names = np.asarray(preprocessing.get_feature_names_out(), dtype=str)
    names = np.asarray(
        [name.replace("categorical__", "").replace("numeric__", "") for name in names]
    )
    importance = np.asarray(regressor.feature_importances_)
    count = min(top_n, len(importance))
    selected = np.argsort(importance)[-count:]
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.barh(
        names[selected], importance[selected], color=MODEL_COLORS["Tabular XGBoost"]
    )
    ax.set_xlabel("XGBoost feature importance (gain-normalized)")
    ax.set_title(f"Top {count} tabular features in the final model")
    ax.grid(axis="x", alpha=0.25)
    _save_both(fig, output_dir, "xgboost_feature_importance")


def save_reference_comparison_figure(
    comparison: pd.DataFrame, output_dir: Path
) -> None:
    """Plot same-scale paper and reproduction metrics without implying pairing."""
    selected = comparison.loc[
        comparison["model"].isin(
            ["XGBoost", "LightGBM", "Tabular XGBoost", "OOF Stacking Ridge"]
        )
    ].copy()
    short_names = {
        "OOF Stacking Ridge": "OOF Stacking",
        "Tabular XGBoost": "XGBoost",
    }
    selected["label"] = [
        f"{'Paper' if source == 'paper Table 3' else 'Ours'}: "
        f"{short_names.get(model, model)}"
        for source, model in zip(selected["source"], selected["model"], strict=True)
    ]
    colors = [
        "#8C8C8C" if source == "paper Table 3" else "#228833"
        for source in selected["source"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), constrained_layout=True)
    for ax, metric, direction in zip(
        axes,
        ("MAE", "RMSE", "R2"),
        ("lower is better", "lower is better", "higher is better"),
        strict=True,
    ):
        ax.barh(selected["label"], selected[metric], color=colors)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel(f"{metric} ({direction})")
        ax.grid(axis="x", alpha=0.25)
        for index, value in enumerate(selected[metric]):
            ax.text(value, index, f" {value:.4f}", va="center", fontsize=8)
    axes[0].set_title("Same random 80/20 split family")
    axes[1].set_title("Target: log1p(story point)")
    axes[2].set_title("TAWOS: 43,732 issues")
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"table3_direct_comparison.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)


def _save_architecture_figure(output_dir: Path) -> None:
    """Save a compact, manuscript-ready diagram of the leakage-safe workflow."""
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        label: str,
        color: str,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center")

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.2},
        )

    box(0.2, 2.0, 1.25, 1.0, "Outer-training\nissues", "#DDDDDD")
    box(2.0, 3.3, 2.0, 1.0, "Frozen transformer\n+ fold-local Ridge", "#CCDDF0")
    box(2.0, 0.7, 2.0, 1.0, "Fold-local preprocessing\n+ XGBoost", "#F5CCD1")
    box(4.8, 2.0, 1.6, 1.0, "OOF base\npredictions", "#EFE5A8")
    box(7.0, 2.0, 1.25, 1.0, "Ridge\nmeta-learner", "#C9E8D0")
    box(8.8, 2.0, 1.0, 1.0, "Final\nprediction", "#DDDDDD")
    arrow((1.45, 2.65), (2.0, 3.65))
    arrow((1.45, 2.35), (2.0, 1.35))
    arrow((4.0, 3.8), (4.8, 2.75))
    arrow((4.0, 1.2), (4.8, 2.25))
    arrow((6.4, 2.5), (7.0, 2.5))
    arrow((8.25, 2.5), (8.8, 2.5))
    ax.text(
        5.0,
        4.75,
        "Each OOF prediction is produced without fitting on its own row",
        ha="center",
        va="center",
        weight="bold",
    )
    ax.text(
        5.0,
        0.15,
        "The untouched outer test enters only after both base learners are refitted",
        ha="center",
        va="center",
        style="italic",
    )
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"pipeline_architecture.{extension}", bbox_inches="tight"
        )
    plt.close(fig)
