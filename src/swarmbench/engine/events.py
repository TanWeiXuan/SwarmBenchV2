from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from swarmbench.api import Team, Vec2


class EventType(str, Enum):
    OBSTACLE_CRASH = "OBSTACLE_CRASH"
    PROJECTILE_BLOCKED = "PROJECTILE_BLOCKED"
    PROJECTILE_HIT = "PROJECTILE_HIT"
    VEHICLE_COLLISION = "VEHICLE_COLLISION"
    PROJECTILE_EXIT = "PROJECTILE_EXIT"
    PROJECTILE_FIRED = "PROJECTILE_FIRED"
    GOAL = "GOAL"
    CONTROLLER_TIMEOUT = "CONTROLLER_TIMEOUT"
    CONTROLLER_EXCEPTION = "CONTROLLER_EXCEPTION"


EVENT_PRIORITY = {
    EventType.PROJECTILE_FIRED: -1,
    EventType.PROJECTILE_BLOCKED: 0,
    EventType.OBSTACLE_CRASH: 1,
    EventType.PROJECTILE_HIT: 2,
    EventType.VEHICLE_COLLISION: 3,
    EventType.GOAL: 4,
    EventType.PROJECTILE_EXIT: 5,
    EventType.CONTROLLER_TIMEOUT: 6,
    EventType.CONTROLLER_EXCEPTION: 6,
}


@dataclass(frozen=True, slots=True)
class GameEvent:
    time: float
    event_type: EventType
    drone_ids: tuple[int, ...]
    position: Vec2
    team: Team | None = None
    points: int = 0
    projectile_id: int | None = None

    @property
    def sort_key(self) -> tuple[float, int, tuple[int, ...], int]:
        return (self.time, EVENT_PRIORITY[self.event_type], self.drone_ids, self.projectile_id or -1)

