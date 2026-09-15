import numpy as np
import pandas as pd

from storypoint_oof.statistical_analysis import compare_predictions


def test_clustered_comparison_reports_negative_candidate_effect() -> None:
    frame = pd.DataFrame(
        {
            "project_key": np.repeat(["A", "B", "C"], 4),
            "observed_model_target": np.arange(12, dtype=float),
            "baseline": np.arange(12, dtype=float) + 1.0,
            "candidate": np.arange(12, dtype=float) + 0.5,
        }
    )

    result, projects, bootstrap = compare_predictions(
        frame,
        baseline="baseline",
        candidate="candidate",
        repetitions=100,
        seed=42,
    )

    assert result["observed_difference"] == -0.5
    assert result["relative_mae_reduction_percent"] == 50.0
    assert len(projects) == 3
    assert bootstrap.shape == (100,)
