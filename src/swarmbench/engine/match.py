"""Headless authoritative v2 game state and continuous event resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from math import hypot, isfinite

from swarmbench.api import (
    DRONE_RADIUS,
    PHYSICS_DT,
    PROJECTILE_CONTACT_RADIUS,
    TANK_WEAPON_SPEC,
    VEHICLE_COLLISION_RADIUS,
    DestructionReason,
    DroneSnapshot,
    DroneStatus,
    DroneType,
    ProjectileSnapshot,
    Team,
    Vec2,
)

from .arena import Scenario
from .collisions import (
    contact_position,
    swept_arena_exit,
    swept_goal_contact,
    swept_obstacle_contact,
    swept_points_contact,
)
from .dynamics import DynamicState, advance_dynamics
from .events import EventType, GameEvent


@dataclass(slots=True)
class _Drone:
    snapshot: DroneSnapshot
    desired_acceleration: Vec2 = (0.0, 0.0)


@dataclass(slots=True)
class _Projectile:
    snapshot: ProjectileSnapshot


class Simulator:
    """Deterministic physics and rules with no controller or rendering dependency."""

    def __init__(self, scenario: Scenario, dt: float = PHYSICS_DT) -> None:
        self.scenario = scenario
        self.dt = dt
        self.time = 0.0
        self.scores = {Team.A: 0, Team.B: 0}
        self.drones: dict[int, _Drone] = {}
        for drone in scenario.drones:
            if drone.drone_type is DroneType.TANK:
                drone = replace(
                    drone,
                    shots_remaining=TANK_WEAPON_SPEC.magazine_size,
                    next_fire_time=TANK_WEAPON_SPEC.initial_lockout,
                )
            self.drones[drone.id] = _Drone(drone)
        self.projectiles: dict[int, _Projectile] = {}
        self._next_projectile_id = 0
        self.events: list[GameEvent] = []

    def snapshots(self, team: Team | None = None) -> tuple[DroneSnapshot, ...]:
        return tuple(
            runtime.snapshot
            for drone_id, runtime in sorted(self.drones.items())
            if team is None or runtime.snapshot.team is team
        )

    def projectile_snapshots(self) -> tuple[ProjectileSnapshot, ...]:
        return tuple(runtime.snapshot for _, runtime in sorted(self.projectiles.items()))

    def set_commands(self, team: Team, commands: dict[int, Vec2 | dict[str, Vec2]]) -> None:
        fire_requests: dict[int, Vec2] = {}
        for drone_id, command in commands.items():
            runtime = self.drones.get(drone_id)
            if runtime is not None and runtime.snapshot.team is team and runtime.snapshot.status is DroneStatus.ACTIVE:
                if isinstance(command, dict):
                    acceleration = command.get("acceleration")
                    if acceleration is not None:
                        runtime.desired_acceleration = acceleration
                    fire_direction = command.get("fire_direction")
                    if fire_direction is not None:
                        fire_requests[drone_id] = fire_direction
                else:
                    runtime.desired_acceleration = command
        if fire_requests:
            self.fire(team, fire_requests)

    def fire(self, team: Team, requests: dict[int, Vec2]) -> tuple[GameEvent, ...]:
        fired: list[GameEvent] = []
        for drone_id, direction in sorted(requests.items()):
            runtime = self.drones.get(drone_id)
            if runtime is None:
                continue
            drone = runtime.snapshot
            if (
                drone.team is not team
                or drone.status is not DroneStatus.ACTIVE
                or drone.drone_type is not DroneType.TANK
                or not drone.shots_remaining
                or drone.next_fire_time is None
                or self.time + 1e-12 < drone.next_fire_time
            ):
                continue
            try:
                x, y = float(direction[0]), float(direction[1])
            except (TypeError, ValueError, IndexError):
                continue
            magnitude = hypot(x, y)
            if magnitude == 0.0 or not isfinite(magnitude):
                continue
            velocity = (x / magnitude * TANK_WEAPON_SPEC.projectile_speed, y / magnitude * TANK_WEAPON_SPEC.projectile_speed)
            projectile_id = self._next_projectile_id
            self._next_projectile_id += 1
            projectile = ProjectileSnapshot(projectile_id, team, drone_id, drone.position, velocity)
            self.projectiles[projectile_id] = _Projectile(projectile)
            runtime.snapshot = replace(
                drone,
                shots_remaining=drone.shots_remaining - 1,
                next_fire_time=self.time + TANK_WEAPON_SPEC.cooldown,
            )
            event = GameEvent(
                self.time,
                EventType.PROJECTILE_FIRED,
                (drone_id,),
                drone.position,
                team,
                projectile_id=projectile_id,
            )
            self.events.append(event)
            fired.append(event)
        return tuple(fired)

    def _candidate_events(
        self,
        before: dict[int, DroneSnapshot],
        after: dict[int, DroneSnapshot],
        projectile_before: dict[int, ProjectileSnapshot],
        projectile_after: dict[int, ProjectileSnapshot],
    ) -> list[GameEvent]:
        candidates: list[GameEvent] = []
        active_ids = sorted(before)
        for drone_id in active_ids:
            old, new = before[drone_id], after[drone_id]
            obstacle_times = [
                contact
                for obstacle in self.scenario.obstacles
                if (contact := swept_obstacle_contact(old.position, new.position, obstacle, DRONE_RADIUS)) is not None
            ]
            if obstacle_times:
                t = min(obstacle_times)
                candidates.append(GameEvent(t, EventType.OBSTACLE_CRASH, (drone_id,), contact_position(old.position, new.position, t)))
            goal_t = swept_goal_contact(old.position, new.position, self.scenario.target_goal(old.team))
            if goal_t is not None:
                candidates.append(
                    GameEvent(
                        goal_t,
                        EventType.GOAL,
                        (drone_id,),
                        contact_position(old.position, new.position, goal_t),
                        old.team,
                        self.scenario.spec_for(old.drone_type).point_value,
                    )
                )

        for left_id, right_id in combinations(active_ids, 2):
            contact = swept_points_contact(
                before[left_id].position,
                after[left_id].position,
                before[right_id].position,
                after[right_id].position,
                VEHICLE_COLLISION_RADIUS,
            )
            if contact is not None:
                left_position = contact_position(before[left_id].position, after[left_id].position, contact)
                right_position = contact_position(before[right_id].position, after[right_id].position, contact)
                position = ((left_position[0] + right_position[0]) / 2, (left_position[1] + right_position[1]) / 2)
                candidates.append(GameEvent(contact, EventType.VEHICLE_COLLISION, (left_id, right_id), position))

        for projectile_id, old_projectile in sorted(projectile_before.items()):
            new_projectile = projectile_after[projectile_id]
            obstacle_times = [
                contact
                for obstacle in self.scenario.obstacles
                if (contact := swept_obstacle_contact(old_projectile.position, new_projectile.position, obstacle, 0.0)) is not None
            ]
            if obstacle_times:
                t = min(obstacle_times)
                candidates.append(
                    GameEvent(
                        t,
                        EventType.PROJECTILE_BLOCKED,
                        (),
                        contact_position(old_projectile.position, new_projectile.position, t),
                        old_projectile.team,
                        projectile_id=projectile_id,
                    )
                )
            for drone_id in active_ids:
                if drone_id == old_projectile.source_drone_id:
                    continue
                contact = swept_points_contact(
                    old_projectile.position,
                    new_projectile.position,
                    before[drone_id].position,
                    after[drone_id].position,
                    PROJECTILE_CONTACT_RADIUS,
                )
                if contact is not None:
                    candidates.append(
                        GameEvent(
                            contact,
                            EventType.PROJECTILE_HIT,
                            (drone_id,),
                            contact_position(old_projectile.position, new_projectile.position, contact),
                            old_projectile.team,
                            projectile_id=projectile_id,
                        )
                    )
            exit_t = swept_arena_exit(
                old_projectile.position,
                new_projectile.position,
                self.scenario.width,
                self.scenario.height,
            )
            if exit_t is not None:
                candidates.append(
                    GameEvent(
                        exit_t,
                        EventType.PROJECTILE_EXIT,
                        (),
                        contact_position(old_projectile.position, new_projectile.position, exit_t),
                        old_projectile.team,
                        projectile_id=projectile_id,
                    )
                )
        return sorted(candidates, key=lambda event: event.sort_key)

    def step(self) -> tuple[GameEvent, ...]:
        before = {
            drone_id: runtime.snapshot
            for drone_id, runtime in self.drones.items()
            if runtime.snapshot.status is DroneStatus.ACTIVE
        }
        after: dict[int, DroneSnapshot] = {}
        for drone_id, old in sorted(before.items()):
            runtime = self.drones[drone_id]
            dynamic = advance_dynamics(
                DynamicState(old.position, old.velocity, old.acceleration),
                runtime.desired_acceleration,
                self.scenario.spec_for(old.drone_type),
                self.dt,
            )
            after[drone_id] = replace(
                old,
                position=dynamic.position,
                velocity=dynamic.velocity,
                acceleration=dynamic.acceleration,
            )

        projectile_before = {projectile_id: runtime.snapshot for projectile_id, runtime in self.projectiles.items()}
        projectile_after = {
            projectile_id: replace(
                projectile,
                position=(
                    projectile.position[0] + projectile.velocity[0] * self.dt,
                    projectile.position[1] + projectile.velocity[1] * self.dt,
                ),
            )
            for projectile_id, projectile in projectile_before.items()
        }

        active_drones = set(before)
        active_projectiles = set(projectile_before)
        resolved: list[GameEvent] = []
        for candidate in self._candidate_events(before, after, projectile_before, projectile_after):
            if candidate.event_type in {EventType.PROJECTILE_BLOCKED, EventType.PROJECTILE_EXIT}:
                if candidate.projectile_id not in active_projectiles:
                    continue
                active_projectiles.remove(candidate.projectile_id)
            elif candidate.event_type is EventType.PROJECTILE_HIT:
                drone_id = candidate.drone_ids[0]
                if candidate.projectile_id not in active_projectiles or drone_id not in active_drones:
                    continue
                active_projectiles.remove(candidate.projectile_id)
                active_drones.remove(drone_id)
                after[drone_id] = replace(
                    after[drone_id],
                    position=candidate.position,
                    status=DroneStatus.ELIMINATED,
                    destruction_reason=DestructionReason.PROJECTILE_HIT,
                )
            elif candidate.event_type is EventType.VEHICLE_COLLISION:
                if not all(drone_id in active_drones for drone_id in candidate.drone_ids):
                    continue
                for drone_id in candidate.drone_ids:
                    active_drones.remove(drone_id)
                    after[drone_id] = replace(
                        after[drone_id],
                        position=candidate.position,
                        status=DroneStatus.ELIMINATED,
                        destruction_reason=DestructionReason.VEHICLE_COLLISION,
                    )
            elif candidate.event_type is EventType.OBSTACLE_CRASH:
                drone_id = candidate.drone_ids[0]
                if drone_id not in active_drones:
                    continue
                active_drones.remove(drone_id)
                after[drone_id] = replace(
                    after[drone_id],
                    position=candidate.position,
                    status=DroneStatus.ELIMINATED,
                    destruction_reason=DestructionReason.OBSTACLE_CRASH,
                )
            elif candidate.event_type is EventType.GOAL:
                drone_id = candidate.drone_ids[0]
                if drone_id not in active_drones:
                    continue
                active_drones.remove(drone_id)
                after[drone_id] = replace(after[drone_id], position=candidate.position, status=DroneStatus.SCORED)
                self.scores[after[drone_id].team] += candidate.points
            else:
                continue
            event = replace(candidate, time=self.time + candidate.time * self.dt)
            resolved.append(event)

        for drone_id, snapshot in after.items():
            self.drones[drone_id].snapshot = snapshot
        self.projectiles = {
            projectile_id: _Projectile(projectile_after[projectile_id])
            for projectile_id in sorted(active_projectiles)
        }
        self.time += self.dt
        self.events.extend(resolved)
        return tuple(resolved)
