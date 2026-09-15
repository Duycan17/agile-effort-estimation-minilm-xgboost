import numpy as np

from storypoint_oof.splitting import make_split_plan


def test_group_plan_has_no_row_or_project_overlap() -> None:
    groups = np.repeat(["A", "B", "C", "D", "E", "F", "G"], 5)
    plan = make_split_plan(groups, strategy="group", test_size=0.2, folds=5, seed=42)
    assert not set(plan.train_indices) & set(plan.test_indices)
    assert not set(groups[plan.train_indices]) & set(groups[plan.test_indices])
    assert plan.audit["outer_project_overlap"] == 0
    assert all(value == 0 for value in plan.audit["inner_project_overlaps"])
    assert np.all(plan.fold_id[plan.train_indices] >= 0)
    assert np.all(plan.fold_id[plan.test_indices] == -1)


def test_each_training_row_is_validation_once() -> None:
    groups = np.array(["ONE"] * 30)
    plan = make_split_plan(groups, strategy="random", test_size=0.2, folds=5, seed=7)
    seen = []
    for _, fit, validation in plan.folds():
        assert not set(fit) & set(validation)
        assert not set(validation) & set(plan.test_indices)
        seen.extend(validation.tolist())
    assert sorted(seen) == sorted(plan.train_indices.tolist())
