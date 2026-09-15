# Fixed cross-project superiority protocol

## Question

Does frozen-text plus tabular OOF stacking reduce cross-project MAE relative to
tabular XGBoost?

## Fixed design

- Data: all 43,732 eligible TAWOS issues from 39 projects.
- Target: `log1p(story_point)`.
- Outer evaluation: five shuffled `GroupKFold` folds by `project_key`, seed 42.
- Inner stacking: five shuffled `GroupKFold` OOF folds on outer-training projects.
- Models and hyperparameters: frozen at their Table-3-compatible values; no
  selection based on outer predictions.
- Primary unit: project, with every project weighted equally.
- Primary effect: project mean of
  `MAE(OOF stacking) - MAE(tabular XGBoost)`; negative favors stacking.
- Inference: 100,000 project bootstrap repetitions for the 95% percentile CI
  and 100,000 paired project sign flips for the two-sided p-value.
- Superiority criterion: observed effect below zero, upper 95% CI below zero,
  and two-sided p-value below 0.05.

Aggregate issue-weighted metrics, win counts, the one-sided p-value, and the
Wilcoxon signed-rank result are secondary descriptions.
