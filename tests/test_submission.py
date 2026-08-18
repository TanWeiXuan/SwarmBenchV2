from pathlib import Path

import pytest

from swarmbench.competition.submission import (
    MAX_SUBMISSION_BYTES,
    ChangedFile,
    SubmissionValidationError,
    aggregate_calibration,
    calibration_opponents,
    smoke_test,
    validate_calibration_artifact,
    validate_pr_files,
    validate_source,
)
from swarmbench.competition.glicko2 import GlickoRating, update_rating
from swarmbench.competition.publisher import accept_calibration
from swarmbench.competition.ratings import RatingRecord
from swarmbench.version import CALIBRATION_VERSION, ENGINE_VERSION


def test_correct_single_file_pr_shape() -> None:
    assert validate_pr_files([ChangedFile("submissions/alice/controller.py", 123)], "Alice") == "submissions/alice/controller.py"


@pytest.mark.parametrize(
    ("files", "author"),
    [
        ([ChangedFile("wrong/alice/controller.py", 1)], "alice"),
        ([ChangedFile("submissions/bob/controller.py", 1)], "alice"),
        ([ChangedFile("submissions/alice/controller.txt", 1)], "alice"),
        ([ChangedFile("submissions/alice/controller.py", 1), ChangedFile("README.md", 1)], "alice"),
        ([ChangedFile("submissions/alice/controller.py", MAX_SUBMISSION_BYTES + 1)], "alice"),
        ([ChangedFile("submissions/alice/controller.py", 1, True)], "alice"),
    ],
)
def test_invalid_pr_shapes_are_rejected(files, author) -> None:
    if files[0].path.startswith("wrong/"):
        assert validate_pr_files(files, author) is None
    else:
        with pytest.raises(SubmissionValidationError):
            validate_pr_files(files, author)


def test_non_submission_bot_pr_is_ignored() -> None:
    assert validate_pr_files([ChangedFile("leaderboard/ratings.json", 20)], "swarmbench-bot") is None


def write_simple_controller(path: Path) -> Path:
    path.write_text(
        """from swarmbench import BaseSwarmController
class SwarmController(BaseSwarmController):
    def initialize(self, game_info): self.goal = game_info.target_goal
    def step(self, state):
        return {drone.id: (4.0 if self.goal.center[0] > drone.position[0] else -4.0, 0.0) for drone in state.own_drones}
""",
        encoding="utf-8",
    )
    return path


def test_source_and_empty_arena_smoke(tmp_path: Path) -> None:
    controller = write_simple_controller(tmp_path / "controller.py")
    validate_source(controller)
    result = smoke_test(controller)
    assert result["score"] > 0
    assert result["timing"]["hard_timeouts"] == 0


def test_source_rejects_unsupported_import(tmp_path: Path) -> None:
    controller = write_simple_controller(tmp_path / "controller.py")
    controller.write_text("import pandas\n" + controller.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SubmissionValidationError, match="unsupported import"):
        validate_source(controller)


def valid_artifact():
    return {
        "schema": "swarmbench-calibration-v2",
        "engine_version": ENGINE_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "submission_id": "alice/controller",
        "submission_path": "submissions/alice/controller.py",
        "head_sha": "a" * 40,
        "opponents": ["rush"],
        "match_count": 2,
        "wins": 1,
        "draws": 0,
        "losses": 1,
        "score_mean": 30.0,
        "timing": {"mean": 0.01, "p95": 0.02, "max": 0.03, "missed_updates": 0},
        "provisional_rating": 1500.0,
        "deviation": 250.0,
        "volatility": 0.06,
    }


def test_calibration_schema_accepts_valid_primitive_data() -> None:
    assert validate_calibration_artifact(valid_artifact())["match_count"] == 2


def test_calibration_schema_rejects_bad_counts_and_sha() -> None:
    artifact = valid_artifact()
    artifact["head_sha"] = "$(malicious)"
    with pytest.raises(SubmissionValidationError):
        validate_calibration_artifact(artifact)


def test_trusted_acceptance_checks_path_and_sha() -> None:
    artifact = valid_artifact()
    accepted = accept_calibration(
        artifact,
        {
            "rush": RatingRecord("rush", "Rush", "SwarmBench", rating=1400, built_in=True),
            "bob/existing": RatingRecord("bob/existing", "Existing", "bob", rating=1600),
        },
        expected_path="submissions/alice/controller.py",
        expected_sha="a" * 40,
    )
    assert accepted["alice/controller"].built_in is False
    assert accepted["rush"].rating == 1400
    assert accepted["bob/existing"].rating == 1600
    with pytest.raises(SubmissionValidationError):
        accept_calibration(artifact, accepted, expected_path="submissions/alice/other.py", expected_sha="a" * 40)


def test_calibration_includes_baselines_and_nearby_community_without_subject() -> None:
    ratings = {
        name: RatingRecord(name, name, "SwarmBench", built_in=True)
        for name in ("rush", "defend", "greedy_value", "assignment", "potential_field", "marksman", "convoy")
    }
    ratings.update(
        {
            "alice/controller": RatingRecord("alice/controller", "Controller", "alice", rating=1700),
            "bob/near": RatingRecord("bob/near", "Near", "bob", rating=1690),
            "carol/far": RatingRecord("carol/far", "Far", "carol", rating=1100),
        }
    )

    opponents = calibration_opponents(ratings, "submissions/alice/controller.py")

    assert [record.controller_id for record in opponents] == [
        "rush",
        "defend",
        "greedy_value",
        "assignment",
        "potential_field",
        "marksman",
        "convoy",
        "bob/near",
        "carol/far",
    ]


def test_calibration_uses_fixed_opponent_rating() -> None:
    game = {
        "opponent": "bob/strong",
        "opponent_rating": 2000.0,
        "opponent_deviation": 80.0,
        "opponent_volatility": 0.05,
        "side": "A",
        "result": 1.0,
        "score_for": 1,
        "score_against": 0,
        "timing": {"mean": 0.01, "p95": 0.02, "max": 0.03, "missed_updates": 0, "hard_timeouts": 0, "invalid_actions": 0},
    }
    part = {
        "schema": "swarmbench-calibration-seed-v2",
        "engine_version": ENGINE_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "submission_id": "submissions/alice/controller.py",
        "head_sha": "a" * 40,
        "seed_index": 0,
        "opponents": ["bob/strong"],
        "games": [game],
    }

    artifact = aggregate_calibration(
        [part],
        submission_id="submissions/alice/controller.py",
        submission_path="submissions/alice/controller.py",
        head_sha="a" * 40,
    )
    expected = update_rating(GlickoRating(), [(GlickoRating(2000.0, 80.0, 0.05), 1.0)])

    assert artifact["provisional_rating"] == pytest.approx(expected.rating)


def test_large_calibration_pool_spans_community_ratings() -> None:
    ratings = {
        name: RatingRecord(name, name, "SwarmBench", built_in=True)
        for name in ("rush", "defend", "greedy_value", "assignment", "potential_field", "marksman", "convoy")
    }
    ratings.update(
        {
            f"author{index}/controller": RatingRecord(
                f"author{index}/controller", "Controller", f"author{index}", rating=1000 + index * 100
            )
            for index in range(12)
        }
    )

    community = calibration_opponents(ratings, "submissions/alice/new.py")[7:]

    assert len(community) == 8
    assert {record.rating for record in community} >= {1000, 1100, 2000, 2100}
