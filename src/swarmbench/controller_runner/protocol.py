"""Versioned JSON-only controller IPC codecs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from swarmbench.api import (
    CircleObstacle,
    DestructionReason,
    DroneSnapshot,
    DroneSpec,
    DroneStatus,
    DroneType,
    GameInfo,
    GameState,
    GoalZone,
    RectangleObstacle,
    ProjectileSnapshot,
    Team,
    WeaponSpec,
)

PROTOCOL_VERSION = 2


class ProtocolError(RuntimeError):
    pass


def _goal_to_dict(goal: GoalZone) -> dict[str, float]:
    return asdict(goal)


def _goal_from_dict(data: dict[str, Any]) -> GoalZone:
    return GoalZone(*(float(data[key]) for key in ("x_min", "x_max", "y_min", "y_max")))


def _obstacle_to_dict(obstacle: CircleObstacle | RectangleObstacle) -> dict[str, Any]:
    if isinstance(obstacle, CircleObstacle):
        return {"kind": "circle", "center": list(obstacle.center), "radius": obstacle.radius}
    return {"kind": "rectangle", **asdict(obstacle)}


def _obstacle_from_dict(data: dict[str, Any]) -> CircleObstacle | RectangleObstacle:
    if data.get("kind") == "circle":
        center = data["center"]
        return CircleObstacle((float(center[0]), float(center[1])), float(data["radius"]))
    if data.get("kind") == "rectangle":
        return RectangleObstacle(*(float(data[key]) for key in ("x_min", "x_max", "y_min", "y_max")))
    raise ProtocolError("invalid obstacle kind")


def _drone_to_dict(drone: DroneSnapshot) -> dict[str, Any]:
    return {
        "id": drone.id,
        "team": drone.team.value,
        "drone_type": drone.drone_type.value,
        "position": list(drone.position),
        "velocity": list(drone.velocity),
        "acceleration": list(drone.acceleration),
        "status": drone.status.value,
        "destruction_reason": drone.destruction_reason.value if drone.destruction_reason else None,
        "shots_remaining": drone.shots_remaining,
        "next_fire_time": drone.next_fire_time,
    }


def _drone_from_dict(data: dict[str, Any]) -> DroneSnapshot:
    reason = data.get("destruction_reason")
    return DroneSnapshot(
        int(data["id"]),
        Team(data["team"]),
        DroneType(data["drone_type"]),
        (float(data["position"][0]), float(data["position"][1])),
        (float(data["velocity"][0]), float(data["velocity"][1])),
        (float(data["acceleration"][0]), float(data["acceleration"][1])),
        DroneStatus(data["status"]),
        DestructionReason(reason) if reason else None,
        int(data["shots_remaining"]) if data.get("shots_remaining") is not None else None,
        float(data["next_fire_time"]) if data.get("next_fire_time") is not None else None,
    )


def _projectile_to_dict(projectile: ProjectileSnapshot) -> dict[str, Any]:
    return {
        "id": projectile.id,
        "team": projectile.team.value,
        "source_drone_id": projectile.source_drone_id,
        "position": list(projectile.position),
        "velocity": list(projectile.velocity),
    }


def _projectile_from_dict(data: dict[str, Any]) -> ProjectileSnapshot:
    return ProjectileSnapshot(
        int(data["id"]),
        Team(data["team"]),
        int(data["source_drone_id"]),
        (float(data["position"][0]), float(data["position"][1])),
        (float(data["velocity"][0]), float(data["velocity"][1])),
    )


def game_info_to_dict(info: GameInfo) -> dict[str, Any]:
    return {
        "team": info.team.value,
        "arena_width": info.arena_width,
        "arena_height": info.arena_height,
        "target_goal": _goal_to_dict(info.target_goal),
        "own_goal": _goal_to_dict(info.own_goal),
        "obstacles": [_obstacle_to_dict(item) for item in info.obstacles],
        "drone_specs": [
            {"drone_type": kind.value, **asdict(spec)} for kind, spec in info.drone_specs
        ],
        "own_initial_drones": [_drone_to_dict(item) for item in info.own_initial_drones],
        "opponent_initial_drones": [_drone_to_dict(item) for item in info.opponent_initial_drones],
        "scenario_seed": info.scenario_seed,
        "generator_version": info.generator_version,
        "controller_seed": info.controller_seed,
        "controller_api_version": info.controller_api_version,
        "weapon_spec": asdict(info.weapon_spec),
    }


def game_info_from_dict(data: dict[str, Any]) -> GameInfo:
    specs = tuple(
        (
            DroneType(item["drone_type"]),
            DroneSpec(float(item["max_speed"]), float(item["max_acceleration"]), float(item["max_jerk"]), int(item["point_value"])),
        )
        for item in data["drone_specs"]
    )
    weapon = data["weapon_spec"]
    return GameInfo(
        Team(data["team"]),
        float(data["arena_width"]),
        float(data["arena_height"]),
        _goal_from_dict(data["target_goal"]),
        _goal_from_dict(data["own_goal"]),
        tuple(_obstacle_from_dict(item) for item in data["obstacles"]),
        specs,
        tuple(_drone_from_dict(item) for item in data["own_initial_drones"]),
        tuple(_drone_from_dict(item) for item in data["opponent_initial_drones"]),
        int(data["scenario_seed"]),
        int(data["generator_version"]),
        int(data["controller_seed"]),
        int(data["controller_api_version"]),
        WeaponSpec(
            float(weapon["projectile_speed"]),
            int(weapon["magazine_size"]),
            float(weapon["cooldown"]),
            float(weapon["initial_lockout"]),
        ),
    )


def game_state_to_dict(state: GameState) -> dict[str, Any]:
    return {
        "time": state.time,
        "own_drones": [_drone_to_dict(item) for item in state.own_drones],
        "opponent_drones": [_drone_to_dict(item) for item in state.opponent_drones],
        "own_score": state.own_score,
        "opponent_score": state.opponent_score,
        "projectiles": [_projectile_to_dict(item) for item in state.projectiles],
    }


def game_state_from_dict(data: dict[str, Any]) -> GameState:
    return GameState(
        float(data["time"]),
        tuple(_drone_from_dict(item) for item in data["own_drones"]),
        tuple(_drone_from_dict(item) for item in data["opponent_drones"]),
        int(data["own_score"]),
        int(data["opponent_score"]),
        tuple(_projectile_from_dict(item) for item in data["projectiles"]),
    )


def request(sequence: int, command: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"protocol_version": PROTOCOL_VERSION, "sequence": sequence, "command": command, "payload": payload}


def validate_request(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict) or message.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("invalid protocol version")
    if not isinstance(message.get("sequence"), int) or message["sequence"] < 0:
        raise ProtocolError("invalid sequence")
    if message.get("command") not in {"initialize", "step", "shutdown"} or not isinstance(message.get("payload"), dict):
        raise ProtocolError("invalid request")
    return message


def validate_response(message: Any, expected_sequence: int) -> dict[str, Any]:
    if not isinstance(message, dict) or message.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("invalid response protocol")
    if message.get("sequence") != expected_sequence:
        raise ProtocolError("stale or incorrect response sequence")
    if message.get("status") not in {"ok", "error"}:
        raise ProtocolError("invalid response status")
    return message

