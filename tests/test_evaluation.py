import numpy as np

from storypoint_oof.evaluation import point_metrics


def test_point_metrics_include_table3_columns() -> None:
    metrics = point_metrics(np.array([1.0, 2.0]), np.array([1.0, 4.0]))

    assert metrics["MAE"] == 1.0
    assert metrics["MdAE"] == 1.0
    assert metrics["MSE"] == 2.0
    assert np.isclose(metrics["RMSE"], np.sqrt(2.0))
    assert metrics["R2"] == -7.0
