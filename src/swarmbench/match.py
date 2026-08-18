"""Complete isolated match orchestration."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from swarmbench.api import CONTROLLER_PERIOD, DEFAULT_MATCH_DURATION, PHYSICS_DT, TANK_WEAPON_SPEC, DroneStatus, GameInfo, GameState, Team
from swarmbench.controller_runner import ControllerError, ControllerProcess, ControllerTimeout, StepResult
from swarmbench.engine import EventType, Scenario, Simulator, generate_scenario
from swarmbench.replay import Replay
from swarmbench.version import CONTROLLER_API_VERSION


def controller_seed(match_seed: int, team: Team) -> int:
    digest = hashlib.sha256(f"swarmbench:{match_seed}:{team.value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def controller_identity(path: Path) -> dict[str, str]:
    return {"id": path.stem, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def game_info(scenario: Scenario, team: Team) -> GameInfo:
    return GameInfo(
        team,
        scenario.width,
        scenario.height,
        scenario.target_goal(team),
        scenario.own_goal(team),
        scenario.obstacles,
        scenario.drone_specs,
        scenario.team_drones(team),
        scenario.team_drones(team.opponent),
        scenario.seed,
        scenario.generator_version,
        controller_seed(scenario.seed, team),
        CONTROLLER_API_VERSION,
        TANK_WEAPON_SPEC,
    )


def game_state(simulator: Simulator, team: Team) -> GameState:
    return GameState(
        simulator.time,
        simulator.snapshots(team),
        simulator.snapshots(team.opponent),
        simulator.scores[team],
        simulator.scores[team.opponent],
        simulator.projectile_snapshots(),
    )


@dataclass(frozen=True, slots=True)
class MatchResult:
    score_a: int
    score_b: int
    winner: Team | None
    reason: str
    replay: Replay
    stats_a: dict
    stats_b: dict


def _event_dict(event) -> dict:
    return {
        "time": event.time,
        "type": event.event_type.value,
        "drone_ids": list(event.drone_ids),
        "position": list(event.position),
        "team": event.team.value if event.team else None,
        "points": event.points,
        "projectile_id": event.projectile_id,
    }


def run_match(
    controller_a_path: str | Path,
    controller_b_path: str | Path,
    *,
    seed: int,
    duration: float = DEFAULT_MATCH_DURATION,
    scenario: Scenario | None = None,
    soft_deadline: float = 0.5,
    hard_timeout: float = 5.0,
    backend: str = "local",
) -> MatchResult:
    path_a, path_b = Path(controller_a_path).resolve(), Path(controller_b_path).resolve()
    scenario = scenario or generate_scenario(seed)
    simulator = Simulator(scenario)
    runner_a = ControllerProcess(path_a, soft_deadline=soft_deadline, hard_timeout=hard_timeout, backend=backend)
    runner_b = ControllerProcess(path_b, soft_deadline=soft_deadline, hard_timeout=hard_timeout, backend=backend)
    action_changes: list[dict] = []
    replay_events: list[dict] = []
    recorded_commands = {Team.A: {}, Team.B: {}}
    forfeiting_team: Team | None = None
    reason = "TIME_LIMIT"

    def record_controller_failure(team: Team, error: BaseException) -> None:
        nonlocal forfeiting_team, reason
        forfeiting_team = team
        event_type = EventType.CONTROLLER_TIMEOUT if isinstance(error, ControllerTimeout) else EventType.CONTROLLER_EXCEPTION
        reason = event_type.value
        replay_events.append(
            {"time": simulator.time, "type": event_type.value, "drone_ids": [], "position": [0.0, 0.0], "team": team.value, "points": 0}
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            init = {
                Team.A: executor.submit(runner_a.initialize, game_info(scenario, Team.A)),
                Team.B: executor.submit(runner_b.initialize, game_info(scenario, Team.B)),
            }
            for team in Team:
                try:
                    init[team].result()
                except ControllerError as error:
                    record_controller_failure(team, error)
        total_ticks = round(duration / PHYSICS_DT)
        for tick in range(total_ticks):
            if forfeiting_team is not None:
                break
            if tick % round(CONTROLLER_PERIOD / PHYSICS_DT) == 0:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {
                        Team.A: executor.submit(runner_a.step, game_state(simulator, Team.A)),
                        Team.B: executor.submit(runner_b.step, game_state(simulator, Team.B)),
                    }
                    results: dict[Team, StepResult] = {}
                    for team in Team:
                        try:
                            results[team] = futures[team].result()
                        except ControllerError as error:
                            record_controller_failure(team, error)
                if forfeiting_team is not None:
                    break
                for team, result in results.items():
                    if result.missed_soft_deadline:
                        replay_events.append(
                            {"time": simulator.time, "type": EventType.CONTROLLER_TIMEOUT.value, "deadline": "soft", "drone_ids": [], "position": [0.0, 0.0], "team": team.value, "points": 0}
                        )
                    changed = {
                        drone_id: command
                        for drone_id, command in result.accepted_changes.items()
                        if recorded_commands[team].get(drone_id) != command
                    }
                    if changed:
                        recorded_commands[team].update(changed)
                        simulator.set_commands(team, changed)
                    fired_events = simulator.fire(team, result.fire_requests)
                    replay_events.extend(_event_dict(event) for event in fired_events)
                    fired = {
                        event.drone_ids[0]: result.fire_requests[event.drone_ids[0]]
                        for event in fired_events
                    }
                    if changed or fired:
                        action_changes.append(
                            {
                                "time": simulator.time,
                                "team": team.value,
                                "actions": {str(key): list(value) for key, value in sorted(changed.items())},
                                "fires": {str(key): list(value) for key, value in sorted(fired.items())},
                            }
                        )
            replay_events.extend(_event_dict(event) for event in simulator.step())
            if not any(drone.status is DroneStatus.ACTIVE for drone in simulator.snapshots()):
                reason = "ALL_DRONES_RESOLVED"
                break
    finally:
        runner_a.close()
        runner_b.close()

    score_a, score_b = simulator.scores[Team.A], simulator.scores[Team.B]
    if forfeiting_team is not None:
        winner = forfeiting_team.opponent
    elif score_a > score_b:
        winner = Team.A
    elif score_b > score_a:
        winner = Team.B
    else:
        winner = None
    result_text = winner.value if winner else "DRAW"
    replay = Replay(
        scenario,
        controller_identity(path_a),
        controller_identity(path_b),
        action_changes,
        replay_events,
        simulator.time,
        {"A": score_a, "B": score_b},
        result_text,
    )
    return MatchResult(score_a, score_b, winner, reason, replay, runner_a.stats.summary(), runner_b.stats.summary())
