# Repeated random issue-level protocol

## Question

Does frozen-text plus tabular OOF stacking reduce MAE relative to tabular
XGBoost for new issues drawn from the same sampled project distribution?

## Fixed design

- Data: all 43,732 eligible TAWOS issues.
- Target and metrics: `log1p(story_point)`, matching the Table 3 scale.
- Outer evaluation: 30 random 80/20 issue-level holdouts, seeds 42 through 71.
- Inner stacking: five shuffled OOF folds using each outer-training set only.
- Models and hyperparameters: fixed before repeated evaluation.
- Primary effect: mean across splits of
  `MAE(OOF stacking) - MAE(tabular XGBoost)`; negative favors stacking.
- Inference: Nadeau-Bengio corrected resampled t-test, accounting for overlap
  among repeated train/test samples.
- Superiority criterion: effect below zero, corrected upper 95% CI below zero,
  and corrected two-sided p-value below 0.05.

The estimand concerns new issues under random issue-level sampling. It does not
support claims about generalization to entirely unseen projects.
