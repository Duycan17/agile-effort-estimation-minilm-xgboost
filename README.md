# StoryPoint OOF

Publication-oriented reference implementation of a multimodal story-point
regressor:

1. a frozen pretrained transformer converts `title + body` into text embeddings;
2. Ridge regression maps embeddings to story points;
3. a leakage-safe `ColumnTransformer` + XGBoost pipeline learns from tabular
   pre-estimation variables;
4. a Ridge meta-learner combines **out-of-fold (OOF)** base predictions;
5. an untouched outer test set is evaluated once.

The project is intentionally separate from `ages_repro`. It reads the existing
TAWOS CSV but has its own environment, lockfile, source package, tests, protocol,
and run artifacts. The repository does not distribute TAWOS data, trained models,
embedding caches, or experimental outputs.

## Quick start

```bash
uv sync
uv run pytest
uv run storypoint-oof --data /path/to/tawos_pre_estimation.csv --smoke
```

`--smoke` automatically selects the project with the fewest eligible rows. It
uses a deterministic issue-level split and is only a runtime/data-contract
check. It must not be reported as evidence that the model generalizes to unseen
projects.

For the predeclared cross-project experiment:

```bash
uv run storypoint-oof \
  --data /path/to/tawos_pre_estimation.csv \
  --project all \
  --split group \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --model-revision main \
  --bootstrap-repetitions 2000 \
  --output-dir outputs/tawos_group_seed42
```

For a direct, apples-to-apples comparison with TAWOS Table 3:

```bash
uv run storypoint-oof \
  --data /path/to/tawos_pre_estimation.csv \
  --protocol table3-compatible \
  --output-dir outputs/tawos_table3_compatible_seed42
```

This profile fixes the scope inferred from `ages_repro`: all 43,732 eligible
TAWOS issues, a seeded random 80/20 issue split, and metrics on
`log1p(story_point)`. It writes `table3_comparison.csv` and also reports
back-transformed performance in `metrics_raw_story_points.csv`. Random splitting
supports reproduction of Table 3, but not claims about unseen projects.

Run the paired, project-cluster-aware statistical comparison after an experiment:

```bash
uv run storypoint-stats \
  outputs/tawos_table3_compatible_seed42/test_predictions.csv \
  --output-dir outputs/tawos_table3_compatible_seed42/statistics \
  --repetitions 100000 \
  --seed 42
```

For the fixed, simplest cross-project superiority test:

```bash
uv run storypoint-groupcv \
  --data /path/to/tawos_pre_estimation.csv \
  --batch-size 64 \
  --inference-repetitions 100000 \
  --seed 42 \
  --output-dir outputs/tawos_groupcv5_seed42
```

This performs five outer project-group folds and five inner project-group OOF
folds. Each of the 39 projects is outer test exactly once. The primary contrast
is the equal-project mean MAE difference between OOF stacking and XGBoost, with
a project bootstrap interval and paired project sign-flip test.

For the standard random issue-level pipeline matching Table 3:

```bash
uv run storypoint-repeated-random \
  --data /path/to/tawos_pre_estimation.csv \
  --repetitions 30 \
  --first-seed 42 \
  --batch-size 64 \
  --output-dir outputs/tawos_repeated_random30_seed42
```

This repeats random 80/20 evaluation while retaining inner OOF stacking and
uses a corrected resampled t-test for the paired model comparison.

After the first download, replace `main` with the resolved model commit recorded
in `manifest.json` for archival reproduction.

## Leakage controls

- The outer test rows are never used to fit a model, imputer, encoder, scaler,
  meta-learner, or hyperparameter.
- Each OOF prediction comes from base models fitted without that row.
- Tabular imputation and one-hot encoding live inside the scikit-learn pipeline,
  so they are refitted inside every fold.
- Frozen embeddings are computed row-by-row without targets or corpus fitting;
  caching all rows is therefore not target leakage.
- Publication mode uses project groups in both the outer split and inner folds.
- IDs, `estimation_date`, and `days_to_estimation` are excluded from predictors.

See [PROTOCOL.md](PROTOCOL.md) for the estimand, primary endpoint, claim boundary,
and artifact contract.

## Run artifacts

Every completed run writes:

- `config.json`, `manifest.json`, and `summary.md`;
- `splits.csv`, `oof_predictions.csv`, and `test_predictions.csv`;
- `metrics.csv` with bootstrap confidence intervals;
- `metrics_raw_story_points.csv` for performance on the original target scale;
- `table3_comparison.csv` for Table-3-compatible runs;
- `comparisons.csv` for paired MAE differences against tabular XGBoost;
- fitted scikit-learn/XGBoost models under `models/`;
- actual per-round XGBoost train/validation MAE and RMSE, OOF fold metrics, and
  compact summaries under `training_history/`;
- publication-ready PNG and vector PDF figures under `figures/`, plus a figure
  inventory with suggested manuscript captions.

The frozen MiniLM encoder has no training loss because it is not fine-tuned.
Both Ridge components use a tracked L-BFGS solver for the standard Ridge
objective, so their optimization states and convergence summaries are archived
alongside the XGBoost learning curves.

The smoke run is deliberately small. Do not choose models or rewrite hypotheses
from its test result.
