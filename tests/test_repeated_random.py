import numpy as np

from storypoint_oof.repeated_random import corrected_resampled_t_test


def test_corrected_resampled_test_preserves_effect_direction() -> None:
    differences = np.array([-0.03, -0.02, -0.01, -0.04, -0.02])

    result = corrected_resampled_t_test(differences, test_size=0.2)

    assert result["mean_difference"] < 0
    assert result["t_statistic"] < 0
    assert result["one_sided_p_value_stacking_better"] < 0.5
    assert result["variance_correction"] == 0.45
