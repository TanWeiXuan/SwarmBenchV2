from copy import deepcopy

import pytest

from swarmbench.competition.matchmaking import MatchmakingEntry, schedule_games, select_pairings
from swarmbench.competition.ratings import RatingRecord, load_ratings, save_ratings
from swarmbench.competition.tournament import MAX_TOURNAMENT_BATCHES, aggregate_batches, create_plan
from swarmbench.version import ENGINE_VERSION, TOURNAMENT_FORMAT_VERSION


def records(count: int = 7) -> dict[str, RatingRecord]:
    return {
        f"c{index}": RatingRecord(f"c{index}", f"Controller {index}", f"author{index}", rating=1200 + index * 100)
        for index in range(count)
    }


def test_matchmaking_is_deterministic_unique_and_contains_exploration() -> None:
    entries = [MatchmakingEntry(key, value.rating) for key, value in records(12).items()]
    first = select_pairings(entries, 99, target_opponents=4)
    assert first == select_pairings(entries, 99, target_opponents=4)
    assert len(first) == len(set(first))
    assert all(left < right for left, right in first)
    gaps = [abs(int(left[1:]) - int(right[1:])) for left, right in first]
    assert any(gap <= 2 for gap in gaps)
    assert any(gap >= 4 for gap in gaps)


def test_schedule_swaps_sides_for_every_scenario() -> None:
    games = schedule_games((("alpha", "beta"),), 42, scenario_count=4)
    assert len(games) == 8
    for index in range(0, len(games), 2):
        first, second = games[index : index + 2]
        assert first.scenario_seed == second.scenario_seed
        assert (first.controller_a, first.controller_b) == (second.controller_b, second.controller_a)


def test_plan_uses_balanced_nonempty_parallel_batches() -> None:
    plan = create_plan(records(25), 42, mode="official", size="default")
    sizes = [len(batch) for batch in plan.batches]
    assert len(sizes) == MAX_TOURNAMENT_BATCHES
    assert min(sizes) > 0
    assert max(sizes) - min(sizes) <= 1

    small = create_plan(records(2), 42, mode="exhibition", size="small")
    assert len(small.batches) == len(small.games)
    assert all(len(batch) == 1 for batch in small.batches)


def complete_batches(plan):
    games = {game.game_id: game for game in plan.games}
    batches = []
    for index, ids in enumerate(plan.batches):
        results = []
        for game_id in ids:
            game = games[game_id]
            results.append(
                {
                    "game_id": game.game_id,
                    "pairing_id": game.pairing_id,
                    "controller_a": game.controller_a,
                    "controller_b": game.controller_b,
                    "scenario_seed": game.scenario_seed,
                    "score_a": 1,
                    "score_b": 0,
                    "result_a": 1.0,
                    "reason": "TIME_LIMIT",
                    "stats_a": {},
                    "stats_b": {},
                }
            )
        batches.append(
            {
                "format_version": TOURNAMENT_FORMAT_VERSION,
                "engine_version": ENGINE_VERSION,
                "tournament_seed": plan.seed,
                "batch_index": index,
                "expected_game_ids": sorted(ids),
                "games": results,
            }
        )
    return batches


def test_official_updates_only_after_every_valid_batch() -> None:
    state = records()
    plan = create_plan(state, 5, mode="official", size="small")
    batches = complete_batches(plan)
    before = deepcopy(state)
    with pytest.raises(ValueError):
        aggregate_batches(plan, batches[:-1], state)
    assert state == before
    outcome = aggregate_batches(plan, batches, state)
    assert outcome.ratings_after != state
    assert all(record.games > 0 for record in outcome.ratings_after.values())


def test_exhibition_does_not_modify_ratings() -> None:
    state = records()
    plan = create_plan(state, 5, mode="exhibition", size="small")
    outcome = aggregate_batches(plan, complete_batches(plan), state)
    assert outcome.ratings_after == state


def test_official_games_update_baseline_and_community_ratings() -> None:
    state = {
        "rush": RatingRecord("rush", "Rush", "SwarmBench", built_in=True),
        "defend": RatingRecord("defend", "Defend", "SwarmBench", built_in=True),
        "alice/controller": RatingRecord("alice/controller", "Controller", "alice"),
    }
    plan = create_plan(state, 5, mode="official", size="small")
    assert {frozenset((game.controller_a, game.controller_b)) for game in plan.games} == {
        frozenset(("rush", "defend")),
        frozenset(("rush", "alice/controller")),
        frozenset(("defend", "alice/controller")),
    }
    batches = complete_batches(plan)
    for batch in batches:
        for game in batch["games"]:
            winner = "rush" if "rush" in {game["controller_a"], game["controller_b"]} else "defend"
            winner_is_a = game["controller_a"] == winner
            game["score_a"], game["score_b"] = ((1, 0) if winner_is_a else (0, 1))
            game["result_a"] = 1.0 if winner_is_a else 0.0
    outcome = aggregate_batches(plan, batches, state)

    assert all(outcome.ratings_after[key].games > 0 for key in state)
    assert all(outcome.ratings_after[key] != state[key] for key in state)
    assert outcome.ratings_after["rush"].rating > state["rush"].rating
    assert outcome.ratings_after["alice/controller"].rating < state["alice/controller"].rating


def test_missing_duplicate_or_wrong_identity_is_rejected() -> None:
    state = records()
    plan = create_plan(state, 5, mode="official", size="small")
    batches = complete_batches(plan)
    batches[0]["games"][0]["controller_a"] = "tampered"
    with pytest.raises(ValueError):
        aggregate_batches(plan, batches, state)


def test_rating_state_round_trip(tmp_path) -> None:
    path = tmp_path / "ratings.json"
    save_ratings(records(), path)
    assert load_ratings(path) == records()
