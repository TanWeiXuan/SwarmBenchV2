"""Trusted tournament workflow preparation, reporting, and publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swarmbench.controllers.baselines import BASELINE_NAMES, baseline_path
from swarmbench.version import CONTROLLER_API_VERSION, ENGINE_VERSION, SCENARIO_GENERATOR_VERSION, TOURNAMENT_FORMAT_VERSION

from .matchmaking import ScheduledGame
from .publisher import update_readme_leaderboard
from .ratings import RatingRecord, load_ratings, ratings_to_dict, save_ratings
from .tournament import TournamentPlan, aggregate_batches, create_plan, execute_batch, validate_batch


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def resolve_seed(value: str | None, run_id: str) -> int:
    if value:
        try:
            return int(value) % (2**63)
        except ValueError:
            payload = value.encode()
    else:
        payload = f"github-run:{run_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _record_dict(record: RatingRecord) -> dict[str, Any]:
    return asdict(record)


def prepare_plan(
    ratings: dict[str, RatingRecord],
    *,
    seed: int,
    mode: str,
    size: str,
    run_id: str,
    repository: str,
) -> dict[str, Any]:
    plan = create_plan(ratings, seed, mode=mode, size=size)
    return {
        "format_version": TOURNAMENT_FORMAT_VERSION,
        "engine_version": ENGINE_VERSION,
        "controller_api_version": CONTROLLER_API_VERSION,
        "generator_version": SCENARIO_GENERATOR_VERSION,
        "tournament_id": run_id,
        "run_url": f"https://github.com/{repository}/actions/runs/{run_id}",
        "started_at": _now(),
        "seed": plan.seed,
        "mode": plan.mode,
        "size": size,
        "ratings": [_record_dict(ratings[key]) for key in sorted(ratings)],
        "pairings": [list(pair) for pair in plan.pairings],
        "games": [asdict(game) for game in plan.games],
        "batches": [list(batch) for batch in plan.batches],
    }


def validate_plan(data: Any) -> tuple[TournamentPlan, dict[str, RatingRecord]]:
    if not isinstance(data, dict) or data.get("format_version") != TOURNAMENT_FORMAT_VERSION or data.get("engine_version") != ENGINE_VERSION:
        raise ValueError("invalid tournament plan identity")
    try:
        records = {}
        for item in data["ratings"]:
            record = RatingRecord(**item)
            if record.controller_id in records:
                raise ValueError("duplicate controller")
            records[record.controller_id] = record
        pairings = tuple(tuple(pair) for pair in data["pairings"])
        games = tuple(ScheduledGame(**game) for game in data["games"])
        batches = tuple(tuple(batch) for batch in data["batches"])
        plan = TournamentPlan(int(data["seed"]), str(data["mode"]), pairings, games, batches)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid tournament plan: {error}") from error
    if len(batches) != 5 or set(game.game_id for game in games) != set().union(*map(set, batches)):
        raise ValueError("plan batch coverage mismatch")
    if any(game.controller_a not in records or game.controller_b not in records for game in games):
        raise ValueError("plan references unknown controller")
    return plan, records


def reconcile_tournament_ratings(
    planned: dict[str, RatingRecord],
    tournament_after: dict[str, RatingRecord],
    current: dict[str, RatingRecord],
) -> dict[str, RatingRecord]:
    """Merge a deterministic tournament result into the latest rating state."""

    if set(tournament_after) != set(planned):
        raise ValueError("tournament rating result does not match the planned controllers")
    merged = dict(current)
    for controller_id, planned_record in planned.items():
        if current.get(controller_id) != planned_record:
            raise ValueError(f"rating state changed during tournament: {controller_id}")
        merged[controller_id] = tournament_after[controller_id]
    return merged


def controller_paths(records: dict[str, RatingRecord], root: Path) -> dict[str, Path]:
    paths = {}
    for controller_id, record in records.items():
        if record.built_in:
            if controller_id not in BASELINE_NAMES:
                raise ValueError(f"unknown built-in controller: {controller_id}")
            paths[controller_id] = baseline_path(controller_id)
        else:
            parts = controller_id.split("/", 1)
            if len(parts) != 2:
                raise ValueError("invalid community controller ID")
            paths[controller_id] = root / "submissions" / parts[0] / f"{parts[1]}.py"
            if not paths[controller_id].is_file():
                raise ValueError(f"controller file is missing: {controller_id}")
    return paths


def _gh_graphql(query: str, **fields: str) -> dict[str, Any]:
    command = ["gh", "api", "graphql", "-f", f"query={query}"]
    for key, value in fields.items():
        command.extend(("-f", f"{key}={value}"))
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"GitHub GraphQL request failed: {detail}")
    return json.loads(completed.stdout)


def _gh_json(endpoint: str) -> dict[str, Any]:
    completed = subprocess.run(["gh", "api", endpoint], check=False, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"GitHub API request failed: {detail}")
    return json.loads(completed.stdout)


def _discussion_category(owner: str, name: str) -> tuple[str, str]:
    query = "query($owner:String!,$name:String!){repository(owner:$owner,name:$name){id discussionCategories(first:100){nodes{id name}}}}"
    data = _gh_graphql(query, owner=owner, name=name)["data"]["repository"]
    category = next((item for item in data["discussionCategories"]["nodes"] if item["name"] == "Tournament Results"), None)
    if category is None:
        raise RuntimeError("Tournament Results Discussion category is missing")
    return data["id"], category["id"]


def initial_discussion_body(data: dict[str, Any]) -> str:
    return "\n".join(
        (
            "## Initial status: RUNNING",
            "",
            f"- Tournament ID: `{data['tournament_id']}`",
            f"- Mode: **{data['mode']}**",
            f"- [GitHub Actions run]({data['run_url']})",
            f"- Engine/API/generator: `{data['engine_version']}` / `{data['controller_api_version']}` / `{data['generator_version']}`",
            f"- Tournament seed: `{data['seed']}`",
            f"- Controllers: {len(data['ratings'])}",
            f"- Planned pairings: {len(data['pairings'])}",
            f"- Planned games: {len(data['games'])}",
            f"- Started: {data['started_at']}",
            "",
            "The latest bot reply is authoritative for current status. Provisional progress will be posted at approximately 20% increments. Ratings change only after every batch validates.",
        )
    )


def create_discussion(data: dict[str, Any], repository: str) -> dict[str, str]:
    owner, name = repository.split("/", 1)
    repository_id, category_id = _discussion_category(owner, name)
    kind = "Ranking" if data["mode"] == "official" else "Exhibition"
    title_prefix = f"{kind} Tournament #{data['tournament_id']} —"
    title = f"{title_prefix} {datetime.now(UTC).date().isoformat()}"
    query = "query($owner:String!,$name:String!,$categoryId:ID!){repository(owner:$owner,name:$name){discussions(first:100,categoryId:$categoryId,orderBy:{field:CREATED_AT,direction:DESC}){nodes{id url title}}}}"
    existing = _gh_graphql(query, owner=owner, name=name, categoryId=category_id)["data"]["repository"]["discussions"]["nodes"]
    match = next((item for item in existing if str(item["title"]).startswith(title_prefix)), None)
    if match is not None:
        return {"id": match["id"], "url": match["url"], "title": match["title"]}
    mutation = "mutation($repositoryId:ID!,$categoryId:ID!,$title:String!,$body:String!){createDiscussion(input:{repositoryId:$repositoryId,categoryId:$categoryId,title:$title,body:$body}){discussion{id url}}}"
    result = _gh_graphql(
        mutation,
        repositoryId=repository_id,
        categoryId=category_id,
        title=title,
        body=initial_discussion_body(data),
    )
    discussion = result["data"]["createDiscussion"]["discussion"]
    return {"id": discussion["id"], "url": discussion["url"], "title": title}


def _add_comment(discussion_id: str, body: str) -> None:
    mutation = "mutation($discussionId:ID!,$body:String!){addDiscussionComment(input:{discussionId:$discussionId,body:$body}){comment{id}}}"
    _gh_graphql(mutation, discussionId=discussion_id, body=body)


def _artifact_names(repository: str, run_id: str) -> set[str]:
    data = _gh_json(f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100")
    return {item["name"] for item in data["artifacts"] if not item.get("expired", False)}


def _run_jobs(repository: str, run_id: str) -> list[dict[str, Any]]:
    data = _gh_json(f"repos/{repository}/actions/runs/{run_id}/jobs?filter=latest&per_page=100")
    return data["jobs"]


def _failed_stage(jobs: list[dict[str, Any]]) -> str | None:
    bad = {"action_required", "cancelled", "failure", "stale", "startup_failure", "timed_out"}
    for job in jobs:
        name = str(job.get("name", ""))
        if (name.startswith("Compute batch") or name == "Validate all batches and publish atomically") and job.get("conclusion") in bad:
            return name
    return None


def _job_succeeded(jobs: list[dict[str, Any]], name: str) -> bool:
    return any(job.get("name") == name and job.get("conclusion") == "success" for job in jobs)


def _download_artifact(repository: str, run_id: str, name: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["gh", "run", "download", run_id, "--repo", repository, "--name", name, "--dir", str(destination)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"failed to download {name}: {detail}")


def _mark_failed(data: dict[str, Any], discussion_id: str, stage: str) -> None:
    comment = f"## Status: FAILED\n\nThe workflow stopped during `{stage}`. Official ratings were not partially updated."
    _add_comment(discussion_id, comment)


def live_report(
    data: dict[str, Any],
    repository: str,
    run_id: str,
    work: Path,
    *,
    poll_seconds: float = 10.0,
    timeout_seconds: float = 9_900.0,
) -> dict[str, str]:
    """Own one Discussion while isolated jobs publish validated JSON artifacts."""

    validate_plan(data)
    discussion = create_discussion(data, repository)
    print(f"Tournament Discussion: {discussion['url']}", flush=True)
    deadline = time.monotonic() + timeout_seconds

    def wait_for(name: str, job_name: str) -> None:
        while True:
            jobs = _run_jobs(repository, run_id)
            failed = _failed_stage(jobs)
            if failed:
                raise RuntimeError(f"{failed} failed")
            if name in _artifact_names(repository, run_id) and _job_succeeded(jobs, job_name):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {name}")
            time.sleep(poll_seconds)

    try:
        for index in range(5):
            artifact = f"tournament-batch-{index}"
            wait_for(artifact, f"Compute batch {index + 1} (untrusted controllers)")
            _download_artifact(repository, run_id, artifact, work / artifact)
            summary = progress_summary(data, _load_batches(work), index)
            _add_comment(discussion["id"], summary)

        wait_for("tournament-results", "Validate all batches and publish atomically")
        result_dir = work / "tournament-results"
        _download_artifact(repository, run_id, "tournament-results", result_dir)
        result = json.loads((result_dir / "tournament-result.json").read_text(encoding="utf-8"))
        if (
            result.get("format_version") != TOURNAMENT_FORMAT_VERSION
            or result.get("mode") != data["mode"]
            or result.get("game_count") != len(data["games"])
            or not str(result.get("discussion_body", "")).startswith("## Status: COMPLETE")
        ):
            raise ValueError("invalid tournament result artifact")
        body = result["discussion_body"]
        _add_comment(discussion["id"], "### Final result\n\n" + body)
    except Exception as error:
        try:
            _mark_failed(data, discussion["id"], str(error))
        except Exception:
            pass
        raise
    return discussion


def _load_batches(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.rglob("batch-*.json"))]


def _matchup_progress_lines(games: list[dict[str, Any]]) -> list[str]:
    matchups: dict[tuple[str, str], list[int]] = {}
    for game in games:
        left, right = sorted((game["controller_a"], game["controller_b"]))
        totals = matchups.setdefault((left, right), [0, 0, 0, 0, 0, 0])
        left_is_a = game["controller_a"] == left
        result = game["result_a"] if left_is_a else 1.0 - game["result_a"]
        totals[0] += 1
        if result == 1.0:
            totals[1] += 1
        elif result == 0.5:
            totals[2] += 1
        else:
            totals[3] += 1
        totals[4] += game["score_a"] if left_is_a else game["score_b"]
        totals[5] += game["score_b"] if left_is_a else game["score_a"]
    if not matchups:
        return []
    limit = 10
    lines = [
        "#### Current batch matchups",
        "",
        "| Matchup | Games | Left W-D-L | Score (left–right) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for (left, right), (count, wins, draws, losses, score_left, score_right) in list(matchups.items())[:limit]:
        lines.append(f"| `{left}` vs `{right}` | {count} | {wins}-{draws}-{losses} | {score_left}–{score_right} |")
    if len(matchups) > limit:
        lines.extend(("", f"_Showing {limit} of {len(matchups)} matchups in this batch; complete pairing details follow in the final report._"))
    return lines


def progress_summary(data: dict[str, Any], batches: list[dict[str, Any]], completed_index: int) -> str:
    plan, _ = validate_plan(data)
    games = []
    current_games = []
    for index, batch in enumerate(batches):
        validated = validate_batch(plan, batch, index)
        games.extend(validated)
        if index == completed_index:
            current_games = validated
    expected_completed = sum(len(plan.batches[index]) for index in range(completed_index + 1))
    if len(games) != expected_completed:
        raise ValueError("progress result count mismatch")
    side_a_wins = sum(game["result_a"] == 1.0 for game in current_games)
    draws = sum(game["result_a"] == 0.5 for game in current_games)
    side_b_wins = len(current_games) - side_a_wins - draws
    score_a = sum(game["score_a"] for game in current_games)
    score_b = sum(game["score_b"] for game in current_games)
    hard = sum(int(game[side].get("hard_timeouts", 0)) for game in games for side in ("stats_a", "stats_b"))
    soft = sum(int(game[side].get("missed_updates", 0)) for game in games for side in ("stats_a", "stats_b"))
    exceptions = sum(int(game[side].get("exceptions", 0)) for game in games for side in ("stats_a", "stats_b"))
    percent = round(100 * len(games) / len(plan.games)) if plan.games else 100
    pairings = len({game["pairing_id"] for game in games})
    lines = [
        f"### Progress — {percent}%",
        "",
        f"Completed: {len(games)} / {len(plan.games)} games",
        f"Current batch: {len(current_games)} games — {side_a_wins} side-A wins, {draws} draws, {side_b_wins} side-B wins; aggregate score {score_a}–{score_b}",
        f"Pairings touched: {pairings} / {len(plan.pairings)}",
        f"Controller exceptions: {exceptions}",
        f"Hard timeouts: {hard}",
        f"Soft-deadline misses: {soft}",
    ]
    matchup_lines = _matchup_progress_lines(current_games)
    if matchup_lines:
        lines.extend(("", *matchup_lines))
    lines.extend(("", "Current results are provisional; Glicko-2 updates are applied only when the full rating period completes."))
    return "\n".join(lines)


def _game_totals(games: tuple[dict[str, Any], ...]) -> tuple[int, int, int]:
    wins = sum(game["result_a"] == 1.0 for game in games)
    draws = sum(game["result_a"] == 0.5 for game in games)
    losses = len(games) - wins - draws
    return wins, draws, losses


def final_report(data: dict[str, Any], outcome) -> str:
    plan, _ = validate_plan(data)
    wins, draws, losses = _game_totals(outcome.games)
    hard = sum(int(game[side].get("hard_timeouts", 0)) for game in outcome.games for side in ("stats_a", "stats_b"))
    soft = sum(int(game[side].get("missed_updates", 0)) for game in outcome.games for side in ("stats_a", "stats_b"))
    exceptions = sum(int(game[side].get("exceptions", 0)) for game in outcome.games for side in ("stats_a", "stats_b"))
    aggregate = {controller_id: [0, 0, 0] for controller_id in outcome.ratings_before}
    for game in outcome.games:
        left, right, score = game["controller_a"], game["controller_b"], game["result_a"]
        if score == 1.0:
            aggregate[left][0] += 1
            aggregate[right][2] += 1
        elif score == 0.0:
            aggregate[right][0] += 1
            aggregate[left][2] += 1
        else:
            aggregate[left][1] += 1
            aggregate[right][1] += 1
    deltas = sorted(
        ((key, outcome.ratings_after[key].rating - record.rating) for key, record in outcome.ratings_before.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    closest = min(outcome.games, key=lambda game: (abs(game["score_a"] - game["score_b"]), game["game_id"])) if outcome.games else None
    upsets = []
    timing_means = []
    timing_p95 = []
    timing_maximums = []
    for game in outcome.games:
        for side in ("stats_a", "stats_b"):
            timing_means.append(float(game[side].get("mean", 0.0)))
            timing_p95.append(float(game[side].get("p95", 0.0)))
            timing_maximums.append(float(game[side].get("max", 0.0)))
        if game["result_a"] in {0.0, 1.0}:
            winner = game["controller_a"] if game["result_a"] == 1.0 else game["controller_b"]
            loser = game["controller_b"] if game["result_a"] == 1.0 else game["controller_a"]
            gap = outcome.ratings_before[loser].rating - outcome.ratings_before[winner].rating
            if gap > 0:
                upsets.append((gap, game["game_id"], winner, loser))
    lines = [
        "## Status: COMPLETE",
        "",
        f"- Tournament ID: `{data['tournament_id']}`",
        f"- Mode: **{plan.mode}**",
        f"- Started / ended: {data['started_at']} / {_now()}",
        f"- [GitHub Actions run]({data['run_url']})",
        f"- Engine/API/generator: `{data['engine_version']}` / `{data['controller_api_version']}` / `{data['generator_version']}`",
        f"- Tournament seed: `{plan.seed}`",
        f"- Controllers / pairings / games: {len(outcome.ratings_before)} / {len(plan.pairings)} / {len(outcome.games)}",
        f"- Side-A W/D/L: {wins} / {draws} / {losses}",
        f"- Exceptions / hard timeouts / soft misses: {exceptions} / {hard} / {soft}",
        f"- Timing mean / worst p95 / maximum: {(sum(timing_means) / len(timing_means) if timing_means else 0.0) * 1000:.2f} / {max(timing_p95, default=0.0) * 1000:.2f} / {max(timing_maximums, default=0.0) * 1000:.2f} ms",
        "",
        "### Rating period",
        "",
        "| Controller | Before | After | Delta | W-D-L |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for controller_id in sorted(outcome.ratings_before):
        before, after = outcome.ratings_before[controller_id], outcome.ratings_after[controller_id]
        w, d, loss = aggregate[controller_id]
        lines.append(f"| {controller_id} | {before.rating:.1f} | {after.rating:.1f} | {after.rating - before.rating:+.1f} | {w}-{d}-{loss} |")
    if deltas:
        lines.extend(("", f"Biggest gain: `{deltas[0][0]}` ({deltas[0][1]:+.1f}); biggest loss: `{deltas[-1][0]}` ({deltas[-1][1]:+.1f})."))
    if closest:
        lines.append(f"Closest game: `{closest['game_id']}` — {closest['score_a']}-{closest['score_b']}.")
    if upsets:
        gap, game_id, winner, loser = max(upsets)
        lines.append(f"Notable upset: `{winner}` defeated `{loser}` in `{game_id}` despite a {gap:.1f}-point pre-period rating gap.")
    if plan.mode == "official":
        top = sorted((record for record in outcome.ratings_after.values() if not record.built_in), key=lambda record: (-record.rating, record.controller_id))[:10]
        lines.extend(("", "### Community top 10", ""))
        if top:
            lines.extend(f"{index}. `{record.controller_id}` — {record.rating:.1f} ± {record.deviation:.1f}" for index, record in enumerate(top, 1))
        else:
            lines.append("No community controllers have been accepted yet.")
    lines.extend(("", "### Pairing details", ""))
    for pairing in plan.pairings:
        selected = [game for game in outcome.games if {game["controller_a"], game["controller_b"]} == set(pairing)]
        first_wins = sum((game["result_a"] if game["controller_a"] == pairing[0] else 1 - game["result_a"]) == 1.0 for game in selected)
        pairing_draws = sum(game["result_a"] == 0.5 for game in selected)
        lines.append(f"- `{pairing[0]}` vs `{pairing[1]}`: {first_wins} wins / {pairing_draws} draws / {len(selected) - first_wins - pairing_draws} losses for `{pairing[0]}`")
    lines.extend(("", "Selected compact replay JSON files are attached to this workflow run's `tournament-replays` artifact."))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--ratings", type=Path, required=True)
    prepare.add_argument("--mode", required=True)
    prepare.add_argument("--size", required=True)
    prepare.add_argument("--seed")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--repository", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    compute = subparsers.add_parser("compute")
    compute.add_argument("--plan", type=Path, required=True)
    compute.add_argument("--batch", type=int, required=True)
    compute.add_argument("--output", type=Path, required=True)
    compute.add_argument("--replay-dir", type=Path, required=True)
    compute.add_argument("--duration", type=float, default=90.0)
    reporter = subparsers.add_parser("live-report")
    reporter.add_argument("--plan", type=Path, required=True)
    reporter.add_argument("--repository", required=True)
    reporter.add_argument("--run-id", required=True)
    reporter.add_argument("--work", type=Path, required=True)
    final = subparsers.add_parser("final")
    final.add_argument("--plan", type=Path, required=True)
    final.add_argument("--batches", type=Path, required=True)
    final.add_argument("--ratings", type=Path, required=True)
    final.add_argument("--readme", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "prepare":
        seed = resolve_seed(args.seed, args.run_id)
        data = prepare_plan(load_ratings(args.ratings), seed=seed, mode=args.mode, size=args.size, run_id=args.run_id, repository=args.repository)
        args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "compute":
        data = json.loads(args.plan.read_text(encoding="utf-8"))
        plan, records = validate_plan(data)
        result = execute_batch(
            plan,
            args.batch,
            controller_paths(records, Path.cwd()),
            duration=args.duration,
            backend=os.environ.get("SWARMBENCH_BACKEND", "local"),
            replay_dir=args.replay_dir,
        )
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "live-report":
        data = json.loads(args.plan.read_text(encoding="utf-8"))
        live_report(data, args.repository, args.run_id, args.work)
    elif args.command == "final":
        data = json.loads(args.plan.read_text(encoding="utf-8"))
        plan, records = validate_plan(data)
        outcome = aggregate_batches(plan, _load_batches(args.batches), records)
        report_body = final_report(data, outcome)
        if plan.mode == "official":
            published_ratings = reconcile_tournament_ratings(records, outcome.ratings_after, load_ratings(args.ratings))
            save_ratings(published_ratings, args.ratings)
            update_readme_leaderboard(args.readme, published_ratings)
        args.output.write_text(
            json.dumps(
                {
                    "format_version": TOURNAMENT_FORMAT_VERSION,
                    "mode": plan.mode,
                    "game_count": len(outcome.games),
                    "discussion_body": report_body,
                    "ratings_before": ratings_to_dict(outcome.ratings_before),
                    "ratings_after": ratings_to_dict(outcome.ratings_after),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
