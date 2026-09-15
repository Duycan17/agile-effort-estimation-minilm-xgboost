"""Command-line entrypoint for reproducible story-point experiments."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from storypoint_oof.data import load_tawos
from storypoint_oof.embeddings import EmbeddingConfig, load_or_create_embeddings
from storypoint_oof.experiment import ExperimentConfig, run_experiment, write_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen text embeddings + tabular XGBoost + leakage-safe OOF stacking"
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to a local pre-estimation TAWOS CSV; the dataset is not distributed.",
    )
    parser.add_argument(
        "--protocol",
        choices=("strict", "table3-compatible"),
        default="strict",
        help=(
            "strict evaluates raw story points with the requested split; "
            "table3-compatible fixes all-data random 80/20 seed 42 and evaluates "
            "log1p(story_point)"
        ),
    )
    parser.add_argument(
        "--project", default="all", help="all, smallest, or project key"
    )
    parser.add_argument("--split", choices=("auto", "group", "random"), default="auto")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-name", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--text-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--meta-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Select the smallest project and run a lightweight non-evidential check",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")

    if args.protocol == "table3-compatible" and args.smoke:
        raise ValueError("Table 3 compatibility requires the complete TAWOS dataset")
    if args.protocol == "table3-compatible":
        project = "all"
        split = "random"
        test_size = 0.2
        folds = 5
        seed = 42
        target_transform = "log1p"
        print(
            "Using Table-3-compatible protocol: all rows, random 80/20, "
            "5 OOF folds, seed 42, log1p(story_point)."
        )
    else:
        project = "smallest" if args.smoke else args.project
        split = args.split
        test_size = args.test_size
        folds = args.folds
        seed = args.seed
        target_transform = "raw"
    if split == "auto":
        split = "random" if project != "all" else "group"
    if project != "all" and split == "group":
        raise ValueError("A single-project run cannot use project-group splitting")
    bootstrap_repetitions = 200 if args.smoke else args.bootstrap_repetitions
    xgb_estimators = 100 if args.smoke else args.xgb_estimators
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / f"run_{stamp}"
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")

    print(f"Loading data: {args.data}")
    dataset = load_tawos(args.data, project=project)
    print(
        f"Rows={len(dataset.target)}, projects={len(set(dataset.groups))}, "
        f"selected={dataset.audit['selected_project']}"
    )
    embedding_config = EmbeddingConfig(
        model_name=args.model_name,
        revision=args.model_revision,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        seed=seed,
    )
    embeddings, embedding_metadata = load_or_create_embeddings(
        dataset.row_ids, dataset.texts, embedding_config, args.cache_dir
    )
    config = ExperimentConfig(
        split_strategy=split,
        test_size=test_size,
        folds=folds,
        seed=seed,
        text_ridge_alpha=args.text_ridge_alpha,
        meta_ridge_alpha=args.meta_ridge_alpha,
        xgb_estimators=xgb_estimators,
        bootstrap_repetitions=bootstrap_repetitions,
        target_transform=target_transform,
    )
    result = run_experiment(dataset, embeddings, config)
    write_artifacts(
        output_dir,
        dataset,
        embedding_metadata,
        config,
        result,
        smoke=args.smoke,
    )
    print(
        result.metrics[["model", "MAE", "MdAE", "MSE", "RMSE", "R2"]]
        .sort_values("MAE")
        .to_string(index=False)
    )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
