"""Deterministic rating-aware pair and side-swap schedule generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from random import Random


@dataclass(frozen=True, slots=True)
class MatchmakingEntry:
    controller_id: str
    rating: float


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    game_id: str
    pairing_id: str
    controller_a: str
    controller_b: str
    scenario_seed: int


def select_pairings(entries: list[MatchmakingEntry], seed: int, target_opponents: int = 8) -> tuple[tuple[str, str], ...]:
    if target_opponents < 1:
        return ()
    ordered = sorted(entries, key=lambda entry: entry.controller_id)
    target = min(target_opponents, max(0, len(ordered) - 1))
    rng = Random(seed)
    pairs: set[tuple[str, str]] = set()
    degree = {entry.controller_id: 0 for entry in ordered}
    ratings = {entry.controller_id: entry.rating for entry in ordered}

    candidates = []
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            candidates.append((abs(left.rating - right.rating), rng.random(), left.controller_id, right.controller_id))
    nearby = sorted(candidates)
    broad = sorted(candidates, key=lambda item: item[1])

    for controller in ordered:
        controller_id = controller.controller_id
        slot = 0
        while degree[controller_id] < target:
            pool = broad if slot % 4 == 3 else nearby
            options = [
                item
                for item in pool
                if controller_id in item[2:]
                and tuple(sorted(item[2:])) not in pairs
                and degree[item[2] if item[3] == controller_id else item[3]] < target
            ]
            if not options:
                options = [item for item in pool if controller_id in item[2:] and tuple(sorted(item[2:])) not in pairs]
            if not options:
                break
            _, _, left, right = options[0]
            pair = tuple(sorted((left, right)))
            pairs.add(pair)
            degree[left] += 1
            degree[right] += 1
            slot += 1

    return tuple(sorted(pairs, key=lambda pair: (ratings[pair[0]] + ratings[pair[1]], pair)))


def _scenario_seed(tournament_seed: int, pairing: tuple[str, str], index: int) -> int:
    payload = f"{tournament_seed}:{pairing[0]}:{pairing[1]}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def schedule_games(
    pairings: tuple[tuple[str, str], ...],
    tournament_seed: int,
    scenario_count: int = 4,
) -> tuple[ScheduledGame, ...]:
    games = []
    for pairing_index, pair in enumerate(pairings):
        pairing_id = f"p{pairing_index:04d}"
        for scenario_index in range(scenario_count):
            seed = _scenario_seed(tournament_seed, pair, scenario_index)
            games.append(ScheduledGame(f"{pairing_id}-s{scenario_index:02d}-ab", pairing_id, pair[0], pair[1], seed))
            games.append(ScheduledGame(f"{pairing_id}-s{scenario_index:02d}-ba", pairing_id, pair[1], pair[0], seed))
    return tuple(games)

