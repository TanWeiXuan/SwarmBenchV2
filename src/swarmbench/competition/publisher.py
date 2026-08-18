"""Trusted publication helpers that consume validated primitive artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

from .ratings import RatingRecord, load_ratings, save_ratings
from .submission import SubmissionValidationError, validate_calibration_artifact

LEADERBOARD_START = "<!-- LEADERBOARD_START -->"
LEADERBOARD_END = "<!-- LEADERBOARD_END -->"


def accept_calibration(
    artifact: dict,
    ratings: dict[str, RatingRecord],
    *,
    expected_path: str,
    expected_sha: str,
) -> dict[str, RatingRecord]:
    validate_calibration_artifact(artifact)
    normalized = expected_path.replace("\\", "/")
    if artifact["submission_path"] != normalized or artifact["head_sha"] != expected_sha:
        raise SubmissionValidationError("artifact does not match merged submission")
    parts = PurePosixPath(normalized).parts
    if len(parts) != 3 or parts[0] != "submissions":
        raise SubmissionValidationError("invalid merged submission path")
    controller_id = f"{parts[1]}/{PurePosixPath(parts[2]).stem}"
    updated = dict(ratings)
    previous = updated.get(controller_id)
    updated[controller_id] = RatingRecord(
        controller_id,
        PurePosixPath(parts[2]).stem.replace("_", " ").title(),
        parts[1],
        float(artifact["provisional_rating"]),
        float(artifact["deviation"]),
        float(artifact["volatility"]),
        version_sha=expected_sha,
        built_in=False,
        wins=previous.wins if previous else 0,
        draws=previous.draws if previous else 0,
        losses=previous.losses if previous else 0,
        games=previous.games if previous else 0,
    )
    return updated


def leaderboard_markdown(ratings: dict[str, RatingRecord]) -> str:
    community = sorted((record for record in ratings.values() if not record.built_in), key=lambda item: (-item.rating, item.controller_id))[:10]
    lines = [
        LEADERBOARD_START,
        "| Rank | Controller | Author | Rating | RD | W | D | L | Games |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if community:
        for rank, record in enumerate(community, 1):
            lines.append(
                f"| {rank} | {record.display_name} | {record.author} | {record.rating:.0f} | {record.deviation:.0f} | {record.wins} | {record.draws} | {record.losses} | {record.games} |"
            )
    else:
        lines.append("| — | No community controllers yet | — | — | — | — | — | — | — |")
    lines.append(LEADERBOARD_END)
    return "\n".join(lines)


def update_readme_leaderboard(readme: Path, ratings: dict[str, RatingRecord]) -> None:
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(LEADERBOARD_START), text.find(LEADERBOARD_END)
    if start < 0 or end < start:
        raise ValueError("README leaderboard markers are missing")
    end += len(LEADERBOARD_END)
    readme.write_text(text[:start] + leaderboard_markdown(ratings) + text[end:], encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--readme", type=Path)
    parser.add_argument("--expected-path")
    parser.add_argument("--expected-path-file", type=Path)
    parser.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    if bool(args.expected_path) == bool(args.expected_path_file):
        parser.error("provide exactly one of --expected-path or --expected-path-file")
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    updated = accept_calibration(
        artifact,
        load_ratings(args.ratings),
        expected_path=args.expected_path_file.read_text(encoding="utf-8").strip() if args.expected_path_file else args.expected_path,
        expected_sha=args.expected_sha,
    )
    save_ratings(updated, args.ratings)
    if args.readme:
        update_readme_leaderboard(args.readme, updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
