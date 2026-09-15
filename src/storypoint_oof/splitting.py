"""Outer holdout and inner OOF split plans with explicit audits."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    train_test_split,
)


@dataclass(frozen=True)
class SplitPlan:
    train_indices: np.ndarray
    test_indices: np.ndarray
    fold_id: np.ndarray
    audit: dict[str, Any]

    def folds(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        for fold in sorted(np.unique(self.fold_id[self.train_indices]).tolist()):
            validation = self.train_indices[self.fold_id[self.train_indices] == fold]
            fit = self.train_indices[self.fold_id[self.train_indices] != fold]
            yield int(fold), fit, validation


def make_split_plan(
    groups: np.ndarray,
    *,
    strategy: str,
    test_size: float,
    folds: int,
    seed: int,
) -> SplitPlan:
    if strategy not in {"group", "random"}:
        raise ValueError("strategy must be 'group' or 'random'")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    if folds < 2:
        raise ValueError("folds must be at least two")

    groups = np.asarray(groups)
    indices = np.arange(len(groups), dtype=int)
    if strategy == "group":
        if len(np.unique(groups)) < folds + 1:
            raise ValueError("Group mode requires at least folds + 1 projects")
        outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_rel, test_rel = next(outer.split(indices, groups=groups))
        train_indices = indices[train_rel]
        test_indices = indices[test_rel]
        training_group_count = len(np.unique(groups[train_indices]))
        if training_group_count < folds:
            raise ValueError(
                "Outer split leaves too few training projects for GroupKFold: "
                f"{training_group_count} projects for {folds} folds"
            )
        inner = GroupKFold(n_splits=folds).split(
            train_indices, groups=groups[train_indices]
        )
    else:
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, shuffle=True, random_state=seed
        )
        inner = KFold(n_splits=folds, shuffle=True, random_state=seed).split(
            train_indices
        )

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    fold_id = np.full(len(indices), -1, dtype=int)
    fold_overlaps: list[int] = []
    for fold, (fit_rel, validation_rel) in enumerate(inner):
        fit = train_indices[np.asarray(fit_rel, dtype=int)]
        validation = train_indices[np.asarray(validation_rel, dtype=int)]
        fold_id[validation] = fold
        fold_overlaps.append(
            int(len(set(groups[fit].tolist()) & set(groups[validation].tolist())))
        )

    train_groups = set(groups[train_indices].tolist())
    test_groups = set(groups[test_indices].tolist())
    outer_overlap = len(train_groups & test_groups)
    if set(train_indices) & set(test_indices):
        raise RuntimeError("Outer train/test row overlap detected")
    if np.any(fold_id[train_indices] < 0):
        raise RuntimeError("Every outer-training row must receive one OOF fold")
    if strategy == "group" and (outer_overlap or any(fold_overlaps)):
        raise RuntimeError("Project leakage detected in group split")

    return SplitPlan(
        train_indices=train_indices,
        test_indices=test_indices,
        fold_id=fold_id,
        audit={
            "strategy": strategy,
            "seed": seed,
            "requested_test_fraction": test_size,
            "realized_test_fraction": len(test_indices) / len(indices),
            "total_rows": len(indices),
            "outer_train_rows": len(train_indices),
            "outer_test_rows": len(test_indices),
            "folds": folds,
            "unique_projects": len(np.unique(groups)),
            "outer_train_projects": len(train_groups),
            "outer_test_projects": len(test_groups),
            "outer_project_overlap": outer_overlap,
            "inner_project_overlaps": fold_overlaps,
            "target_used_to_construct_splits": False,
        },
    )
