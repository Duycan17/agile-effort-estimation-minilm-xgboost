# Predeclared experimental protocol

## Two explicitly separated estimands

The primary strict protocol predicts raw story points and holds out complete
projects. The `table3-compatible` reproduction profile instead uses all eligible
TAWOS rows, a random 80/20 issue split with seed 42, five shuffled OOF folds,
and fits/evaluates on `log1p(story_point)`. Results from these profiles must not
be presented together without explicit target-scale and split labels.

The compatibility profile retains the pre-estimation feature allow-list and
leakage-safe OOF construction. It changes only the dataset scope, split, and
target scale needed for numerical comparison; it does not reintroduce historical
post-outcome variables.

## Research question

Does creation/pre-estimation issue text provide predictive information beyond a
tabular XGBoost model when estimating story points for previously unseen Agile
projects?

## Model roles

- Primary baseline: tabular XGBoost.
- Text baseline: frozen transformer embeddings + Ridge.
- Primary model: OOF Stacking Ridge.
- Sanity baseline: unweighted average of the two base predictions.
- Primary endpoint: test MAE in raw story points.
- Secondary endpoints: MdAE, RMSE, R2, and MAPE.

## Primary protocol

- Dataset: all eligible rows in the pre-estimation TAWOS snapshot.
- Outer split: `GroupShuffleSplit`, grouped by `project_key`, 20% test groups,
  seed 42.
- Inner OOF: five-fold `GroupKFold` on outer-training projects.
- Text: `title + body`, max 256 tokens, attention-mask mean pooling, L2
  normalization, frozen encoder.
- Tabular processing: fold-local median/mode imputation and one-hot encoding.
- Meta-learner: standardized Ridge regression fitted only on OOF predictions.
- Uncertainty: 2,000 project-cluster bootstrap repetitions on the outer test.

## Smoke protocol

The smallest eligible project is selected deterministically. Because only one
project remains, a seeded random outer split and shuffled K-fold OOF are used.
This verifies execution and leakage invariants but has no cross-project external
validity and is not publication evidence.

## Claim boundary

This is predictive observational evidence. It can show incremental predictive
value under the declared split; it cannot show that textual or tabular features
cause effort. Stacking is called better than XGBoost only when the paired
bootstrap interval for `MAE_stacking - MAE_xgboost` is below zero.
