import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from storypoint_oof.experiment import summarize_xgboost_history
from storypoint_oof.models import TrackedRidge


def test_history_summary_uses_lowest_validation_round() -> None:
    history = pd.DataFrame(
        {
            "fit": ["fold_0"] * 6,
            "dataset": ["train"] * 3 + ["validation"] * 3,
            "metric": ["mae"] * 6,
            "boosting_round": [1, 2, 3, 1, 2, 3],
            "value": [0.8, 0.6, 0.5, 0.9, 0.7, 0.75],
        }
    )

    summary = summarize_xgboost_history(history).iloc[0]

    assert summary["best_monitored_round"] == 2
    assert summary["validation_at_best_round"] == 0.7
    assert np.isclose(summary["generalization_gap_at_best"], 0.1)
    assert summary["final_validation_value"] == 0.75


def test_tracked_ridge_matches_standard_ridge_and_records_loss() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(120, 5))
    target = features @ np.array([1.0, -0.4, 0.2, 0.0, 0.7]) + 2.5

    tracked = TrackedRidge(alpha=1.0).fit(features, target)
    reference = Ridge(alpha=1.0).fit(features, target)

    assert tracked.optimization_success_
    assert np.allclose(
        tracked.predict(features), reference.predict(features), atol=1e-5
    )
    objective = tracked.loss_history_["objective_per_sample"].to_numpy()
    assert len(objective) >= 2
    assert np.all(np.diff(objective) <= 1e-10)
    assert objective[-1] < objective[0]
