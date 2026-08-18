"""Current-only authoritative rating state."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .glicko2 import GlickoRating, simultaneous_update

RATINGS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RatingRecord:
    controller_id: str
    display_name: str
    author: str
    rating: float = 1500.0
    deviation: float = 350.0
    volatility: float = 0.06
    wins: int = 0
    draws: int = 0
    losses: int = 0
    games: int = 0
    version_sha: str = ""
    built_in: bool = False

    @property
    def glicko(self) -> GlickoRating:
        return GlickoRating(self.rating, self.deviation, self.volatility)


def ratings_to_dict(records: dict[str, RatingRecord]) -> dict[str, Any]:
    return {
        "schema_version": RATINGS_SCHEMA_VERSION,
        "controllers": [asdict(records[key]) for key in sorted(records)],
    }


def save_ratings(records: dict[str, RatingRecord], path: str | Path) -> None:
    Path(path).write_text(json.dumps(ratings_to_dict(records), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_ratings(path: str | Path) -> dict[str, RatingRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != RATINGS_SCHEMA_VERSION or not isinstance(data.get("controllers"), list):
        raise ValueError("invalid ratings schema")
    records = {}
    for item in data["controllers"]:
        record = RatingRecord(**item)
        if record.controller_id in records or record.games != record.wins + record.draws + record.losses:
            raise ValueError("invalid rating record")
        records[record.controller_id] = record
    return records


def apply_rating_period(records: dict[str, RatingRecord], games: list[tuple[str, str, float]]) -> dict[str, RatingRecord]:
    updated = simultaneous_update({key: value.glicko for key, value in records.items()}, games)
    counters = {key: [0, 0, 0] for key in records}
    for left, right, score in games:
        if score == 1.0:
            counters[left][0] += 1
            counters[right][2] += 1
        elif score == 0.0:
            counters[right][0] += 1
            counters[left][2] += 1
        else:
            counters[left][1] += 1
            counters[right][1] += 1
    return {
        key: replace(
            record,
            rating=updated[key].rating,
            deviation=updated[key].deviation,
            volatility=updated[key].volatility,
            wins=record.wins + counters[key][0],
            draws=record.draws + counters[key][1],
            losses=record.losses + counters[key][2],
            games=record.games + sum(counters[key]),
        )
        for key, record in records.items()
    }

