"""Small public value types shared by controllers and the engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Team(str, Enum):
    A = "A"
    B = "B"

    @property
    def opponent(self) -> "Team":
        return Team.B if self is Team.A else Team.A


class DroneType(str, Enum):
    SCOUT = "SCOUT"
    TRANSPORT = "TRANSPORT"
    TANK = "TANK"


class DroneStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ELIMINATED = "ELIMINATED"
    SCORED = "SCORED"


class DestructionReason(str, Enum):
    VEHICLE_COLLISION = "VEHICLE_COLLISION"
    PROJECTILE_HIT = "PROJECTILE_HIT"
    OBSTACLE_CRASH = "OBSTACLE_CRASH"


@dataclass(frozen=True, slots=True)
class DroneSpec:
    max_speed: float
    max_acceleration: float
    max_jerk: float
    point_value: int


@dataclass(frozen=True, slots=True)
class WeaponSpec:
    projectile_speed: float
    magazine_size: int
    cooldown: float
    initial_lockout: float


DRONE_SPECS = {
    DroneType.SCOUT: DroneSpec(5.0, 4.0, 16.0, 1),
    DroneType.TRANSPORT: DroneSpec(2.5, 2.0, 8.0, 5),
    DroneType.TANK: DroneSpec(1.5, 1.2, 4.8, 1),
}
DYNAMICS_VARIATION = 0.20
DRONE_COUNT_RANGES = {
    DroneType.SCOUT: (10, 14),
    DroneType.TRANSPORT: (4, 8),
    DroneType.TANK: (2, 4),
}
TANK_WEAPON_SPEC = WeaponSpec(20.0, 5, 4.0, 5.0)

DRONE_RADIUS = 0.25
VEHICLE_COLLISION_RADIUS = 0.75
INTERCEPT_RADIUS = VEHICLE_COLLISION_RADIUS
PROJECTILE_CONTACT_RADIUS = DRONE_RADIUS
PHYSICS_DT = 0.05
CONTROLLER_PERIOD = 0.10
DEFAULT_MATCH_DURATION = 90.0
ARENA_WIDTH = 100.0
ARENA_HEIGHT = 60.0

Vec2 = tuple[float, float]
DroneCommand = Vec2 | dict[str, Vec2]

