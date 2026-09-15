import numpy as np
import pandas as pd

from storypoint_oof.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    StoryPointDataset,
)
from storypoint_oof.group_cv import (
    GroupCVConfig,
    _equal_project_inference,
    cross_project_predictions,
)


def test_group_cv_predicts_every_row_once_without_project_overlap() -> None:
    groups = np.repeat([f"P{index}" for index in range(6)], 4)
    rows = len(groups)
    tabular = pd.DataFrame(
        {
            CATEGORICAL_COLUMNS[0]: ["repo"] * rows,
            CATEGORICAL_COLUMNS[1]: groups,
            **{name: np.arange(rows, dtype=float) for name in NUMERIC_COLUMNS},
        }
    )
    dataset = StoryPointDataset(
        row_ids=np.array([f"row-{index}" for index in range(rows)]),
        texts=["text"] * rows,
        tabular=tabular,
        target=np.tile([1.0, 2.0, 3.0, 5.0], 6),
        groups=groups,
        project_keys=groups,
        audit={},
    )
    embeddings = np.random.default_rng(42).normal(size=(rows, 8))
    config = GroupCVConfig(
        outer_folds=3,
        inner_folds=2,
        xgb_estimators=2,
        inference_repetitions=100,
    )

    predictions = cross_project_predictions(dataset, embeddings, config)

    assert len(predictions) == rows
    assert predictions[list(("Text Ridge", "Tabular XGBoost"))].notna().all().all()
    assert predictions.groupby("project_key")["outer_fold"].nunique().eq(1).all()


def test_equal_project_inference_does_not_weight_large_projects() -> None:
    projects = pd.DataFrame(
        {"MAE_difference_stacking_minus_xgboost": [-0.3, -0.2, 0.1]}
    )

    result, bootstrap = _equal_project_inference(projects, 100, 42)

    assert np.isclose(result["mean_difference"], -0.4 / 3)
    assert bootstrap.shape == (100,)
