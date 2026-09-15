# TAWOS Table 3 reproduction audit

## What was established from `ages_repro`

- The historical TAWOS export contains 43,732 issues from 39 projects and 12
  repositories.
- The current pre-estimation snapshot contains the same 43,732 positive
  story-point observations, while excluding post-outcome fields from the model
  allow-list.
- The AgES experiment evaluates a `log1p`-transformed effort target with a
  seeded random 80/20 issue split.
- Applying that split family to all TAWOS rows and evaluating
  `log1p(story_point)` reproduces the numerical scale of Table 3. A diagnostic
  XGBoost run produced MAE 0.4182, RMSE 0.5527, and R2 0.3085, close to the
  reported 0.4230, 0.5600, and 0.2983. This convergence is evidence for the
  inferred protocol, although the paper's exact feature and implementation
  details are not present in the repository history.

## Compatible protocol used here

- Scope: all 43,732 eligible TAWOS issues.
- Outer split: shuffled random 80/20, seed 42 (34,985 train; 8,747 test).
- Inner stacking split: five shuffled folds, seed 42, outer-training rows only.
- Model target and headline metrics: `log1p(story_point)`.
- Original-scale metrics: predictions are back-transformed with `expm1` and
  reported separately.
- Predictors: pre-estimation allow-list only. IDs, estimation timing, and
  post-outcome variables remain excluded.

This profile is intended for a numerical Table 3 comparison. The strict group
protocol remains necessary for claims about generalization to unseen projects.

## Completed compatible run

| Model | MAE | MdAE | MSE | RMSE | R2 |
|---|---:|---:|---:|---:|---:|
| Paper Table 3 LightGBM | 0.4177 | 0.3332 | 0.3031 | 0.5506 | 0.3218 |
| Paper Table 3 XGBoost | 0.4230 | 0.3356 | 0.3136 | 0.5600 | 0.2983 |
| This run Tabular XGBoost | 0.4262 | 0.3366 | 0.3192 | 0.5650 | 0.2774 |
| This run OOF Stacking Ridge | **0.4071** | **0.3246** | **0.2841** | **0.5330** | **0.3568** |

OOF stacking has the best observed point estimates. Its paired MAE difference
against this run's tabular XGBoost is -0.0190, with a project-cluster bootstrap
95% interval of [-0.0352, 0.0038]. Because the interval includes zero, the
improvement should not yet be described as statistically conclusive.
