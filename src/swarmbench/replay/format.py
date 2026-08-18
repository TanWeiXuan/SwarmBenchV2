"""Compact, versioned JSON replay format and deterministic reconstruction."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from swarmbench.api import CircleObstacle, DroneSnapshot, DroneSpec, DroneStatus, DroneType, GoalZone, ProjectileSnapshot, RectangleObstacle, Team
from swarmbench.engine import Scenario, Simulator
from swarmbench.version import CONTROLLER_API_VERSION, ENGINE_VERSION, REPLAY_VERSION


class ReplayValidationError(ValueError):
    pass


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    obstacles = []
    for obstacle in scenario.obstacles:
        if isinstance(obstacle, CircleObstacle):
            obstacles.append({"kind": "circle", "center": list(obstacle.center), "radius": obstacle.radius})
        else:
            obstacles.append(
                {
                    "kind": "rectangle",
                    "x_min": obstacle.x_min,
                    "x_max": obstacle.x_max,
                    "y_min": obstacle.y_min,
                    "y_max": obstacle.y_max,
                }
            )
    return {
        "seed": scenario.seed,
        "generator_version": scenario.generator_version,
        "width": scenario.width,
        "height": scenario.height,
        "goal_for_a": vars_goal(scenario.goal_for_a),
        "goal_for_b": vars_goal(scenario.goal_for_b),
        "obstacles": obstacles,
        "drone_specs": [
            {
                "drone_type": drone_type.value,
                "max_speed": spec.max_speed,
                "max_acceleration": spec.max_acceleration,
                "max_jerk": spec.max_jerk,
                "point_value": spec.point_value,
            }
            for drone_type, spec in scenario.drone_specs
        ],
        "drones": [
            {
                "id": drone.id,
                "team": drone.team.value,
                "drone_type": drone.drone_type.value,
                "position": list(drone.position),
                "velocity": list(drone.velocity),
                "acceleration": list(drone.acceleration),
                "shots_remaining": drone.shots_remaining,
                "next_fire_time": drone.next_fire_time,
            }
            for drone in scenario.drones
        ],
    }


def vars_goal(goal: GoalZone) -> dict[str, float]:
    return {"x_min": goal.x_min, "x_max": goal.x_max, "y_min": goal.y_min, "y_max": goal.y_max}


def scenario_from_dict(data: dict[str, Any]) -> Scenario:
    try:
        obstacles = []
        for item in data["obstacles"]:
            if item["kind"] == "circle":
                obstacles.append(CircleObstacle((float(item["center"][0]), float(item["center"][1])), float(item["radius"])))
            elif item["kind"] == "rectangle":
                obstacles.append(
                    RectangleObstacle(float(item["x_min"]), float(item["x_max"]), float(item["y_min"]), float(item["y_max"]))
                )
            else:
                raise ReplayValidationError("unknown obstacle kind")
        drones = tuple(
            DroneSnapshot(
                int(item["id"]),
                Team(item["team"]),
                DroneType(item["drone_type"]),
                (float(item["position"][0]), float(item["position"][1])),
                (float(item["velocity"][0]), float(item["velocity"][1])),
                (float(item["acceleration"][0]), float(item["acceleration"][1])),
                shots_remaining=int(item["shots_remaining"]) if item.get("shots_remaining") is not None else None,
                next_fire_time=float(item["next_fire_time"]) if item.get("next_fire_time") is not None else None,
            )
            for item in data["drones"]
        )
        goal_a = data["goal_for_a"]
        goal_b = data["goal_for_b"]
        specs = tuple(
            (
                DroneType(item["drone_type"]),
                DroneSpec(
                    float(item["max_speed"]),
                    float(item["max_acceleration"]),
                    float(item["max_jerk"]),
                    int(item["point_value"]),
                ),
            )
            for item in data["drone_specs"]
        )
        return Scenario(
            int(data["seed"]),
            int(data["generator_version"]),
            float(data["width"]),
            float(data["height"]),
            GoalZone(*(float(goal_a[key]) for key in ("x_min", "x_max", "y_min", "y_max"))),
            GoalZone(*(float(goal_b[key]) for key in ("x_min", "x_max", "y_min", "y_max"))),
            tuple(obstacles),
            drones,
            specs,
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        if isinstance(error, ReplayValidationError):
            raise
        raise ReplayValidationError(f"invalid scenario: {error}") from error


@dataclass(slots=True)
class Replay:
    scenario: Scenario
    controller_a: dict[str, str]
    controller_b: dict[str, str]
    action_changes: list[dict[str, Any]]
    events: list[dict[str, Any]]
    final_time: float
    final_scores: dict[str, int]
    result: str
    engine_version: str = ENGINE_VERSION
    controller_api_version: int = CONTROLLER_API_VERSION
    replay_version: int = REPLAY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_version": self.replay_version,
            "engine_version": self.engine_version,
            "controller_api_version": self.controller_api_version,
            "generator_version": self.scenario.generator_version,
            "scenario_seed": self.scenario.seed,
            "controllers": {"A": self.controller_a, "B": self.controller_b},
            "scenario": scenario_to_dict(self.scenario),
            "action_changes": self.action_changes,
            "events": self.events,
            "final_time": self.final_time,
            "final_scores": self.final_scores,
            "result": self.result,
        }


def validate_replay(data: Any) -> Replay:
    if not isinstance(data, dict) or data.get("replay_version") != REPLAY_VERSION:
        raise ReplayValidationError("unsupported replay version")
    try:
        scenario = scenario_from_dict(data["scenario"])
        if int(data["scenario_seed"]) != scenario.seed or int(data["generator_version"]) != scenario.generator_version:
            raise ReplayValidationError("scenario identity mismatch")
        controllers = data["controllers"]
        action_changes = data["action_changes"]
        events = data["events"]
        scores = data["final_scores"]
        if not isinstance(action_changes, list) or not isinstance(events, list):
            raise ReplayValidationError("actions and events must be lists")
        previous_time = -1.0
        for change in action_changes:
            timestamp = float(change["time"])
            if (
                timestamp < previous_time
                or change["team"] not in {"A", "B"}
                or not isinstance(change["actions"], dict)
                or not isinstance(change.get("fires", {}), dict)
            ):
                raise ReplayValidationError("invalid action change")
            previous_time = timestamp
        final_time = float(data["final_time"])
        if final_time < 0 or int(scores["A"]) < 0 or int(scores["B"]) < 0:
            raise ReplayValidationError("invalid final values")
        return Replay(
            scenario,
            {str(key): str(value) for key, value in controllers["A"].items()},
            {str(key): str(value) for key, value in controllers["B"].items()},
            action_changes,
            events,
            final_time,
            {"A": int(scores["A"]), "B": int(scores["B"])},
            str(data["result"]),
            str(data["engine_version"]),
            int(data["controller_api_version"]),
            int(data["replay_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ReplayValidationError):
            raise
        raise ReplayValidationError(f"invalid replay: {error}") from error


def save_replay(replay: Replay, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(replay.to_dict(), sort_keys=True, separators=(",", ":"))
    if destination.suffix == ".gz":
        with gzip.open(destination, "wt", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        destination.write_text(payload + "\n", encoding="utf-8")
    return destination


def load_replay(path: str | Path) -> Replay:
    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
    else:
        data = json.loads(source.read_text(encoding="utf-8"))
    return validate_replay(data)


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    time: float
    drones: tuple[DroneSnapshot, ...]
    scores: tuple[int, int]
    projectiles: tuple[ProjectileSnapshot, ...] = ()


def reconstruct_frames(replay: Replay, every_ticks: int = 1) -> Iterator[ReplayFrame]:
    simulator = Simulator(replay.scenario)
    by_tick: dict[int, list[dict[str, Any]]] = {}
    for change in replay.action_changes:
        tick = round(float(change["time"]) / simulator.dt)
        by_tick.setdefault(tick, []).append(change)
    total_ticks = round(replay.final_time / simulator.dt)
    yield ReplayFrame(0.0, simulator.snapshots(), (0, 0), simulator.projectile_snapshots())
    for tick in range(total_ticks):
        for change in by_tick.get(tick, []):
            actions = {
                int(drone_id): (float(command[0]), float(command[1]))
                for drone_id, command in change["actions"].items()
            }
            simulator.set_commands(Team(change["team"]), actions)
            fires = {
                int(drone_id): (float(direction[0]), float(direction[1]))
                for drone_id, direction in change.get("fires", {}).items()
            }
            simulator.fire(Team(change["team"]), fires)
        simulator.step()
        if (tick + 1) % every_ticks == 0 or tick + 1 == total_ticks:
            yield ReplayFrame(
                simulator.time,
                simulator.snapshots(),
                (simulator.scores[Team.A], simulator.scores[Team.B]),
                simulator.projectile_snapshots(),
            )


def verify_reconstruction(replay: Replay) -> None:
    final = None
    for final in reconstruct_frames(replay):
        pass
    if final is None or final.scores != (replay.final_scores["A"], replay.final_scores["B"]):
        raise ReplayValidationError("reconstructed score does not match replay")

