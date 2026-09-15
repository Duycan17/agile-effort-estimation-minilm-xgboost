from pathlib import Path

import pandas as pd

from storypoint_oof.data import EXCLUDED_LEAKAGE_OR_ID_COLUMNS, load_tawos


def test_loader_uses_allow_list_and_selects_smallest_project(tmp_path: Path) -> None:
    rows = []
    for project, count in (("BIG", 8), ("SMALL", 6)):
        for number in range(count):
            rows.append(
                {
                    "repository": "repo",
                    "project_key": project,
                    "number": number,
                    "created_at": "2024-01-01T00:00:00Z",
                    "title": f"Issue {number}",
                    "body": "Description",
                    "story_point": 1 + number % 3,
                    "estimation_date": "2024-01-02T00:00:00Z",
                    "days_to_estimation": 1,
                    "jira_id": number,
                    "creator_id": 1,
                    "reporter_id": 1,
                    "project_name": project,
                    **{
                        column: float(number)
                        for column in (
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
                        )
                    },
                }
            )
    path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    dataset = load_tawos(path, project="smallest")

    assert dataset.audit["selected_project"] == "SMALL"
    assert len(dataset.target) == 6
    assert not set(EXCLUDED_LEAKAGE_OR_ID_COLUMNS) & set(dataset.tabular.columns)
