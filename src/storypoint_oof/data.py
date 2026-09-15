"""Strict loading of pre-estimation TAWOS variables."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ("repository", "project_key")
NUMERIC_COLUMNS = (
    "title_length_chars",
    "title_word_count",
    "body_length_chars",
    "body_word_count",
    "has_description_code",
    "description_code_length_chars",
    "comment_count_before_estimation",
    "unique_commenters_before_estimation",
    "comment_text_length_before_estimation",
    "comment_code_length_before_estimation",
    "changelog_count_before_estimation",
    "changed_fields_before_estimation",
    "project_age_days_at_creation",
    "project_prior_issues",
    "creator_prior_created_issues_global",
    "creator_prior_created_issues_project",
    "reporter_prior_reported_issues_global",
    "reporter_prior_reported_issues_project",
    "contributors_expertise_global",
    "contributors_expertise_project",
    "created_year",
    "created_month",
    "created_weekday",
    "created_hour",
)

EXCLUDED_LEAKAGE_OR_ID_COLUMNS = (
    "story_point",
    "estimation_date",
    "days_to_estimation",
    "jira_id",
    "creator_id",
    "reporter_id",
    "number",
    "project_name",
)


@dataclass(frozen=True)
class StoryPointDataset:
    row_ids: np.ndarray
    texts: list[str]
    tabular: pd.DataFrame
    target: np.ndarray
    groups: np.ndarray
    project_keys: np.ndarray
    audit: dict[str, Any]

    def __post_init__(self) -> None:
        sizes = {
            len(self.row_ids),
            len(self.texts),
            len(self.tabular),
            len(self.target),
            len(self.groups),
            len(self.project_keys),
        }
        if len(sizes) != 1:
            raise ValueError(f"Dataset views are misaligned: {sorted(sizes)}")
        if len(np.unique(self.row_ids)) != len(self.row_ids):
            raise ValueError("Stable row IDs must be unique")
        if not np.isfinite(self.target).all() or np.any(self.target <= 0):
            raise ValueError("Story points must be finite and strictly positive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonblank(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def load_tawos(path: Path, project: str = "all") -> StoryPointDataset:
    """Load TAWOS through an explicit predictor allow-list.

    `project="smallest"` selects the least frequent eligible project after all
    global cleaning and deduplication decisions have been applied.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    required = {
        "repository",
        "project_key",
        "number",
        "created_at",
        "title",
        "body",
        "story_point",
        *NUMERIC_COLUMNS[:-4],
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required TAWOS columns: {missing}")

    frame = raw.copy()
    initial_rows = len(frame)
    frame["story_point"] = pd.to_numeric(frame["story_point"], errors="coerce")
    valid = (
        frame["story_point"].notna()
        & (frame["story_point"] > 0)
        & _nonblank(frame["repository"])
        & _nonblank(frame["project_key"])
        & (_nonblank(frame["title"]) | _nonblank(frame["body"]))
    )
    frame = frame.loc[valid].copy()
    frame["_row_id"] = (
        frame["repository"].astype(str).str.strip()
        + "::"
        + frame["project_key"].astype(str).str.strip()
        + "::"
        + frame["number"].astype(str).str.strip()
    )
    before_deduplication = len(frame)
    frame = frame.drop_duplicates("_row_id", keep="first").reset_index(drop=True)
    after_deduplication = len(frame)

    eligible_project_counts = frame.groupby("project_key").size().sort_values()
    selected_project: str | None = None
    if project == "smallest":
        minimum = int(eligible_project_counts.iloc[0])
        selected_project = sorted(
            eligible_project_counts[eligible_project_counts == minimum].index.astype(
                str
            )
        )[0]
    elif project != "all":
        selected_project = project
    if selected_project is not None:
        frame = frame.loc[
            frame["project_key"].astype(str) == selected_project
        ].reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"Project not found after eligibility filtering: {project}"
            )

    created = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    tabular = frame.loc[:, list(CATEGORICAL_COLUMNS)].copy()
    for column in NUMERIC_COLUMNS[:-4]:
        tabular[column] = pd.to_numeric(frame[column], errors="coerce")
    tabular["created_year"] = created.dt.year.astype(float)
    tabular["created_month"] = created.dt.month.astype(float)
    tabular["created_weekday"] = created.dt.weekday.astype(float)
    tabular["created_hour"] = created.dt.hour.astype(float)

    titles = frame["title"].fillna("").astype(str).str.strip()
    bodies = frame["body"].fillna("").astype(str).str.strip()
    texts = [
        f"Title: {title}\nDescription: {body}"
        for title, body in zip(titles, bodies, strict=True)
    ]
    audit = {
        "source_path": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "initial_rows": int(initial_rows),
        "eligible_rows_before_deduplication": int(before_deduplication),
        "eligible_rows_after_deduplication": int(after_deduplication),
        "duplicates_removed": int(before_deduplication - after_deduplication),
        "analysis_rows": int(len(frame)),
        "eligible_projects_before_selection": int(len(eligible_project_counts)),
        "project_selection": project,
        "selected_project": selected_project,
        "selected_project_rows": int(len(frame)) if selected_project else None,
        "smallest_eligible_project": str(eligible_project_counts.index[0]),
        "smallest_eligible_project_rows": int(eligible_project_counts.iloc[0]),
        "missing_body_rows": int(frame["body"].isna().sum()),
        "target": "story_point",
        "target_scale": "raw_story_points",
        "prediction_time": "story-point estimation",
        "categorical_predictors": list(CATEGORICAL_COLUMNS),
        "numeric_predictors": list(NUMERIC_COLUMNS),
        "excluded_columns": list(EXCLUDED_LEAKAGE_OR_ID_COLUMNS),
    }
    return StoryPointDataset(
        row_ids=frame["_row_id"].to_numpy(dtype=str),
        texts=texts,
        tabular=tabular,
        target=frame["story_point"].to_numpy(dtype=float),
        groups=frame["project_key"].astype(str).to_numpy(),
        project_keys=frame["project_key"].astype(str).to_numpy(),
        audit=audit,
    )
