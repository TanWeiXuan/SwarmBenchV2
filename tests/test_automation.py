import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from swarmbench.competition import automation
from swarmbench.competition.automation import (
    live_report,
    prepare_plan,
    progress_summary,
    reconcile_tournament_ratings,
    resolve_seed,
    validate_plan,
)
from swarmbench.competition.publisher import leaderboard_markdown, update_readme_leaderboard
from swarmbench.competition.ratings import RatingRecord
from swarmbench.competition.tournament import MAX_TOURNAMENT_BATCHES
from swarmbench.version import ENGINE_VERSION, TOURNAMENT_FORMAT_VERSION


def records() -> dict[str, RatingRecord]:
    return {
        f"c{index}": RatingRecord(f"c{index}", f"C{index}", "alice", rating=1400 + index * 50)
        for index in range(5)
    }


def test_seed_resolution_handles_integer_string_and_empty() -> None:
    assert resolve_seed("42", "10") == 42
    assert resolve_seed("named-seed", "10") == resolve_seed("named-seed", "99")
    assert resolve_seed(None, "10") == resolve_seed(None, "10")


def test_graphql_failure_includes_cli_error(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = subprocess.CompletedProcess([], 1, stdout="", stderr="gh: permission denied")
    monkeypatch.setattr(automation.subprocess, "run", lambda *args, **kwargs: completed)
    with pytest.raises(RuntimeError, match="permission denied"):
        automation._gh_graphql("query { viewer { login } }")


def test_create_discussion_reuses_the_latest_matching_tournament(monkeypatch: pytest.MonkeyPatch) -> None:
    data = prepare_plan(records(), seed=42, mode="official", size="small", run_id="123", repository="owner/repo")
    monkeypatch.setattr(automation, "_discussion_category", lambda *_args: ("R1", "C1"))

    def graphql(query: str, **_fields: str):
        assert "createDiscussion" not in query
        return {
            "data": {
                "repository": {
                    "discussions": {
                        "nodes": [
                            {
                                "id": "D2",
                                "url": "https://example.test/discussions/2",
                                "title": "Ranking Tournament #123 — 2026-08-19",
                            },
                            {
                                "id": "D1",
                                "url": "https://example.test/discussions/1",
                                "title": "Ranking Tournament #123 — 2026-08-18",
                            },
                        ]
                    }
                }
            }
        }

    monkeypatch.setattr(automation, "_gh_graphql", graphql)

    discussion = automation.create_discussion(data, "owner/repo")

    assert discussion == {
        "id": "D2",
        "url": "https://example.test/discussions/2",
        "title": "Ranking Tournament #123 — 2026-08-19",
    }


def test_failed_stage_only_reports_tournament_work() -> None:
    jobs = [
        {"name": "Maintain live tournament Discussion", "conclusion": "failure"},
        {"name": "Compute batch 3 (untrusted controllers)", "conclusion": "timed_out"},
    ]
    assert automation._failed_stage(jobs) == "Compute batch 3 (untrusted controllers)"


def test_automation_plan_round_trip_and_batch_coverage() -> None:
    data = prepare_plan(records(), seed=42, mode="official", size="small", run_id="123", repository="owner/repo")
    plan, restored = validate_plan(json.loads(json.dumps(data)))
    assert restored == records()
    assert len(plan.batches) == min(MAX_TOURNAMENT_BATCHES, max(1, len(plan.games)))
    assert set(game.game_id for game in plan.games) == set().union(*map(set, plan.batches))


def test_load_batches_orders_double_digit_indexes_numerically(tmp_path: Path) -> None:
    for index in (10, 2, 0):
        (tmp_path / f"batch-{index}.json").write_text(json.dumps({"batch_index": index}), encoding="utf-8")

    assert [batch["batch_index"] for batch in automation._load_batches(tmp_path)] == [0, 2, 10]


def test_plan_tampering_is_rejected() -> None:
    data = prepare_plan(records(), seed=42, mode="official", size="small", run_id="123", repository="owner/repo")
    data["engine_version"] = "tampered"
    with pytest.raises(ValueError):
        validate_plan(data)


def test_plan_rejects_duplicate_batch_game_assignment() -> None:
    data = prepare_plan(records(), seed=42, mode="official", size="small", run_id="123", repository="owner/repo")
    data["batches"][1].append(data["batches"][0][0])

    with pytest.raises(ValueError, match="batch coverage mismatch"):
        validate_plan(data)


def test_tournament_rating_reconciliation_preserves_later_controllers() -> None:
    planned = records()
    tournament_after = {
        controller_id: replace(record, rating=record.rating + 25, games=8, wins=5, draws=1, losses=2)
        for controller_id, record in planned.items()
    }
    later = RatingRecord("bob/new", "New", "bob", rating=1777, version_sha="a" * 40)
    current = {**planned, later.controller_id: later}

    merged = reconcile_tournament_ratings(planned, tournament_after, current)

    assert merged == {**tournament_after, later.controller_id: later}
    assert current == {**planned, later.controller_id: later}


def test_tournament_rating_reconciliation_rejects_changed_participant() -> None:
    planned = records()
    tournament_after = dict(planned)
    current = dict(planned)
    current["c0"] = replace(current["c0"], rating=current["c0"].rating + 1)

    with pytest.raises(ValueError, match="rating state changed during tournament: c0"):
        reconcile_tournament_ratings(planned, tournament_after, current)


def test_tournament_rating_reconciliation_rejects_wrong_result_set() -> None:
    planned = records()
    tournament_after = dict(planned)
    tournament_after.pop("c0")

    with pytest.raises(ValueError, match="does not match the planned controllers"):
        reconcile_tournament_ratings(planned, tournament_after, planned)


def test_progress_summary_validates_each_completed_batch() -> None:
    data = prepare_plan(records(), seed=42, mode="official", size="small", run_id="123", repository="owner/repo")
    plan, _ = validate_plan(data)
    games = {game.game_id: game for game in plan.games}
    batches = []
    for batch_index in range(2):
        batch = {
            "format_version": TOURNAMENT_FORMAT_VERSION,
            "engine_version": ENGINE_VERSION,
            "tournament_seed": plan.seed,
            "batch_index": batch_index,
            "expected_game_ids": sorted(plan.batches[batch_index]),
            "games": [],
        }
        for game_id in plan.batches[batch_index]:
            game = games[game_id]
            batch["games"].append(
                {
                    "game_id": game.game_id,
                    "pairing_id": game.pairing_id,
                    "controller_a": game.controller_a,
                    "controller_b": game.controller_b,
                    "scenario_seed": game.scenario_seed,
                    "score_a": 1 if batch_index == 0 else 0,
                    "score_b": 0,
                    "result_a": 1.0 if batch_index == 0 else 0.5,
                    "stats_a": {},
                    "stats_b": {},
                }
            )
        batches.append(batch)
    summary = progress_summary(data, list(reversed(batches)), 1)
    assert "Completed:" in summary and "provisional" in summary
    current_count = len(batches[1]["games"])
    assert (
        f"Current batch: {current_count} games — 0 side-A wins, {current_count} draws, "
        "0 side-B wins; aggregate score 0–0"
    ) in summary
    assert "#### Current batch matchups" in summary
    assert "| Matchup | Games | Left W-D-L | Score (left–right) |" in summary


def test_matchup_progress_table_is_side_neutral_and_compact() -> None:
    games = [
        {"controller_a": "z", "controller_b": "a", "result_a": 1.0, "score_a": 3, "score_b": 1},
        {"controller_a": "a", "controller_b": "z", "result_a": 0.5, "score_a": 2, "score_b": 2},
    ]
    games.extend(
        {"controller_a": f"left-{index}", "controller_b": f"right-{index}", "result_a": 1.0, "score_a": 1, "score_b": 0}
        for index in range(10)
    )

    lines = automation._matchup_progress_lines(games)

    assert "| `a` vs `z` | 2 | 0-1-1 | 3–5 |" in lines
    assert sum(line.startswith("| `") for line in lines) == 10
    assert lines[-1] == "_Showing 10 of 11 matchups in this batch; complete pairing details follow in the final report._"


def test_live_report_owns_discussion_until_final_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = prepare_plan(records(), seed=42, mode="exhibition", size="small", run_id="123", repository="owner/repo")
    plan, _ = validate_plan(data)
    games = {game.game_id: game for game in plan.games}
    batches = {}
    for index, game_ids in enumerate(plan.batches):
        batches[f"tournament-batch-{index}"] = {
            "format_version": TOURNAMENT_FORMAT_VERSION,
            "engine_version": ENGINE_VERSION,
            "tournament_seed": plan.seed,
            "batch_index": index,
            "expected_game_ids": sorted(game_ids),
            "games": [
                {
                    "game_id": games[game_id].game_id,
                    "pairing_id": games[game_id].pairing_id,
                    "controller_a": games[game_id].controller_a,
                    "controller_b": games[game_id].controller_b,
                    "scenario_seed": games[game_id].scenario_seed,
                    "score_a": 0,
                    "score_b": 0,
                    "result_a": 0.5,
                    "stats_a": {},
                    "stats_b": {},
                }
                for game_id in game_ids
            ],
        }
    artifacts = set(batches) | {"tournament-results"}
    jobs = [
        {"name": f"Compute batch {index + 1} (untrusted controllers)", "conclusion": "success"}
        for index in range(len(plan.batches))
    ] + [{"name": "Validate all batches and publish atomically", "conclusion": "success"}]
    comments = []
    downloads = []
    artifact_checks = 0

    def available_artifacts(*_args) -> set[str]:
        nonlocal artifact_checks
        artifact_checks += 1
        if artifact_checks == 1:
            return {f"tournament-batch-{len(plan.batches) - 1}"}
        return artifacts

    def download(_repository: str, _run_id: str, name: str, destination: Path) -> None:
        downloads.append(name)
        destination.mkdir(parents=True)
        if name in batches:
            index = int(name.rsplit("-", 1)[1])
            (destination / f"batch-{index}.json").write_text(json.dumps(batches[name]), encoding="utf-8")
        else:
            (destination / "tournament-result.json").write_text(
                json.dumps(
                    {
                        "format_version": TOURNAMENT_FORMAT_VERSION,
                        "mode": "exhibition",
                        "game_count": len(plan.games),
                        "discussion_body": "## Status: COMPLETE\n\nDone.",
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(automation, "create_discussion", lambda *_args: {"id": "D1", "url": "https://example.test/1"})
    monkeypatch.setattr(automation, "_artifact_names", available_artifacts)
    monkeypatch.setattr(automation, "_run_jobs", lambda *_args: jobs)
    monkeypatch.setattr(automation, "_download_artifact", download)
    monkeypatch.setattr(automation, "_add_comment", lambda _id, body: comments.append(body))

    discussion = live_report(data, "owner/repo", "123", tmp_path, poll_seconds=0)

    assert discussion["id"] == "D1"
    assert downloads[0] == f"tournament-batch-{len(plan.batches) - 1}"
    assert len(comments) == 6 and comments[-1].startswith("### Final result")
    assert "## Status: COMPLETE" in comments[-1]


def test_readme_leaderboard_contains_community_only(tmp_path: Path) -> None:
    state = {
        "rush": RatingRecord("rush", "Rush", "SwarmBench", 1800, built_in=True),
        "alice/a": RatingRecord("alice/a", "A", "alice", 1700, wins=2, games=2),
    }
    block = leaderboard_markdown(state)
    assert "alice" in block and "Rush" not in block
    readme = tmp_path / "README.md"
    readme.write_text("before\n<!-- LEADERBOARD_START -->\nold\n<!-- LEADERBOARD_END -->\nafter\n", encoding="utf-8")
    update_readme_leaderboard(readme, state)
    assert "| 1 | A | alice |" in readme.read_text(encoding="utf-8")
