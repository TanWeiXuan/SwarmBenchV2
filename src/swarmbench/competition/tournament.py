"""Sharded tournament planning, compute, validation, and atomic publication."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from swarmbench.controllers.baselines import BASELINE_NAMES, baseline_path
from swarmbench.match import run_match
from swarmbench.replay import save_replay
from swarmbench.version import ENGINE_VERSION, TOURNAMENT_FORMAT_VERSION

from .matchmaking import MatchmakingEntry, ScheduledGame, schedule_games, select_pairings
from .ratings import RatingRecord, apply_rating_period, load_ratings, ratings_to_dict


MAX_TOURNAMENT_BATCHES = 19


@dataclass(frozen=True, slots=True)
class TournamentPlan:
    seed: int
    mode: str
    pairings: tuple[tuple[str, str], ...]
    games: tuple[ScheduledGame, ...]
    batches: tuple[tuple[str, ...], ...]


SIZE_PRESETS = {
    "small": (2, 1),
    "default": (8, 4),
    "large": (12, 8),
}


def create_plan(records: dict[str, RatingRecord], seed: int, *, mode: str, size: str) -> TournamentPlan:
    if mode not in {"official", "exhibition"} or size not in SIZE_PRESETS:
        raise ValueError("invalid tournament mode or size")
    target_opponents, scenario_count = SIZE_PRESETS[size]
    entries = [MatchmakingEntry(key, record.rating) for key, record in sorted(records.items())]
    pairings = select_pairings(entries, seed, target_opponents)
    games = schedule_games(pairings, seed, scenario_count)
    batch_count = min(MAX_TOURNAMENT_BATCHES, max(1, len(games)))
    batches = tuple(tuple(game.game_id for game in games[index::batch_count]) for index in range(batch_count))
    return TournamentPlan(seed, mode, pairings, games, batches)


def execute_batch(
    plan: TournamentPlan,
    batch_index: int,
    controller_paths: dict[str, Path],
    *,
    duration: float = 90.0,
    backend: str = "local",
    replay_dir: Path | None = None,
) -> dict[str, Any]:
    expected = set(plan.batches[batch_index])
    results = []
    representative = None
    closest = None
    for game in plan.games:
        if game.game_id not in expected:
            continue
        match = run_match(
            controller_paths[game.controller_a],
            controller_paths[game.controller_b],
            seed=game.scenario_seed,
            duration=duration,
            backend=backend,
        )
        score = 0.5 if match.winner is None else (1.0 if match.winner.value == "A" else 0.0)
        results.append(
            {
                "game_id": game.game_id,
                "pairing_id": game.pairing_id,
                "controller_a": game.controller_a,
                "controller_b": game.controller_b,
                "scenario_seed": game.scenario_seed,
                "score_a": match.score_a,
                "score_b": match.score_b,
                "result_a": score,
                "reason": match.reason,
                "stats_a": match.stats_a,
                "stats_b": match.stats_b,
            }
        )
        if replay_dir is not None:
            representative = representative or match.replay
            difference = abs(match.score_a - match.score_b)
            if closest is None or difference < closest[0]:
                closest = (difference, match.replay)
    replay_artifacts = []
    if replay_dir is not None:
        replay_dir.mkdir(parents=True, exist_ok=True)
        if representative is not None:
            destination = save_replay(representative, replay_dir / f"representative-b{batch_index}.json.gz")
            replay_artifacts.append(destination.name)
        if closest is not None:
            destination = save_replay(closest[1], replay_dir / f"closest-b{batch_index}.json.gz")
            replay_artifacts.append(destination.name)
    return {
        "format_version": TOURNAMENT_FORMAT_VERSION,
        "engine_version": ENGINE_VERSION,
        "tournament_seed": plan.seed,
        "batch_index": batch_index,
        "expected_game_ids": sorted(expected),
        "games": results,
        "replay_artifacts": replay_artifacts,
    }


def validate_batch(plan: TournamentPlan, batch: Any, batch_index: int) -> list[dict[str, Any]]:
    if not isinstance(batch, dict):
        raise ValueError("batch must be an object")
    if (
        batch.get("format_version") != TOURNAMENT_FORMAT_VERSION
        or batch.get("engine_version") != ENGINE_VERSION
        or batch.get("tournament_seed") != plan.seed
        or batch.get("batch_index") != batch_index
    ):
        raise ValueError("batch identity mismatch")
    expected = set(plan.batches[batch_index])
    if set(batch.get("expected_game_ids", [])) != expected or not isinstance(batch.get("games"), list):
        raise ValueError("batch schedule mismatch")
    seen = set()
    games_by_id = {game.game_id: game for game in plan.games}
    for result in batch["games"]:
        game_id = result.get("game_id")
        if game_id in seen or game_id not in expected:
            raise ValueError("duplicate or unexpected result")
        scheduled = games_by_id[game_id]
        if (
            result.get("controller_a") != scheduled.controller_a
            or result.get("controller_b") != scheduled.controller_b
            or result.get("scenario_seed") != scheduled.scenario_seed
            or result.get("result_a") not in {0.0, 0.5, 1.0}
        ):
            raise ValueError("result does not match schedule")
        for score_key in ("score_a", "score_b"):
            if not isinstance(result.get(score_key), int) or not 0 <= result[score_key] <= 60:
                raise ValueError("invalid score")
        seen.add(game_id)
    if seen != expected:
        raise ValueError("missing batch results")
    return batch["games"]


@dataclass(frozen=True, slots=True)
class TournamentOutcome:
    games: tuple[dict[str, Any], ...]
    ratings_before: dict[str, RatingRecord]
    ratings_after: dict[str, RatingRecord]


def aggregate_batches(
    plan: TournamentPlan,
    batches: list[dict[str, Any]],
    ratings: dict[str, RatingRecord],
) -> TournamentOutcome:
    if len(batches) != len(plan.batches):
        raise ValueError("all planned batches are required")
    all_games = tuple(result for index, batch in enumerate(batches) for result in validate_batch(plan, batch, index))
    if len({result["game_id"] for result in all_games}) != len(plan.games):
        raise ValueError("tournament result set is incomplete")
    observations = [(result["controller_a"], result["controller_b"], float(result["result_a"])) for result in all_games]
    after = apply_rating_period(ratings, observations) if plan.mode == "official" else dict(ratings)
    return TournamentOutcome(all_games, dict(ratings), after)


def tournament_cli(*, seed: int, size: str, mode: str) -> int:
    root = Path.cwd()
    ratings_path = root / "leaderboard" / "ratings.json"
    ratings = load_ratings(ratings_path)
    plan = create_plan(ratings, seed, mode=mode, size=size)
    paths = {name: baseline_path(name) for name in BASELINE_NAMES}
    batches = [execute_batch(plan, index, paths) for index in range(len(plan.batches))]
    outcome = aggregate_batches(plan, batches, ratings)
    if mode == "official":
        ratings_path.write_text(json.dumps(ratings_to_dict(outcome.ratings_after), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{mode} tournament: {len(plan.pairings)} pairings, {len(outcome.games)} games")
    return 0
