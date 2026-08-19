"""Prioritized SIPP traffic scheduling with conservative Tank fire control.

The controller deliberately keeps the authoritative engine outside the
submission.  It uses the engine's public snapshots and its deterministic
geometry/dynamics helpers for prediction, while treating SIPP as a timing
layer above a small jerk-aware tracker.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import acos, cos, hypot, isfinite, pi, sin, sqrt
from typing import Iterable, Sequence

from swarmbench import (
    BaseSwarmController,
    CircleObstacle,
    DroneSnapshot,
    DroneSpec,
    DroneStatus,
    DroneType,
    GameInfo,
    GameState,
    RectangleObstacle,
    Team,
)
from swarmbench.api import Vec2
from swarmbench.engine.collisions import (
    swept_arena_exit,
    swept_obstacle_contact,
    swept_points_contact,
)
from swarmbench.engine.dynamics import DynamicState, advance_dynamics, clip_vector


EPS = 1.0e-9
MERGE_EPS = 1.0e-7
VEHICLE_COLLISION_RADIUS = 0.75
RESERVATION_MARGIN = 0.15
RESERVATION_RADIUS = VEHICLE_COLLISION_RADIUS + RESERVATION_MARGIN
PATH_CLEARANCE = 0.35
ROADMAP_MARGIN = 0.18
PLANNING_HORIZON = 12.0
REPLAN_PERIOD = 0.7
EDGE_SPEED_FACTOR = 0.80
EDGE_TRACKING_SLACK = 0.35
DEPARTURE_GRANULARITY = 0.05


def _distance(left: Vec2, right: Vec2) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def _lerp(left: Vec2, right: Vec2, fraction: float) -> Vec2:
    return (
        left[0] + (right[0] - left[0]) * fraction,
        left[1] + (right[1] - left[1]) * fraction,
    )


def _segment_bounds(left: Vec2, right: Vec2, padding: float) -> tuple[float, float, float, float]:
    return (
        min(left[0], right[0]) - padding,
        max(left[0], right[0]) + padding,
        min(left[1], right[1]) - padding,
        max(left[1], right[1]) + padding,
    )


def _bounds_overlap(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1] and left[2] <= right[3] and right[2] <= left[3]


def _segment_clear(
    left: Vec2,
    right: Vec2,
    obstacles: tuple[CircleObstacle | RectangleObstacle, ...],
) -> bool:
    padding = 0.25 + PATH_CLEARANCE
    return all(swept_obstacle_contact(left, right, obstacle, padding) is None for obstacle in obstacles)


@dataclass(frozen=True, slots=True)
class ReservationSegment:
    start_time: float
    end_time: float
    start: Vec2
    end: Vec2

    @property
    def velocity(self) -> Vec2:
        duration = self.end_time - self.start_time
        if duration <= EPS:
            return (0.0, 0.0)
        return (
            (self.end[0] - self.start[0]) / duration,
            (self.end[1] - self.start[1]) / duration,
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return _segment_bounds(self.start, self.end, RESERVATION_RADIUS)


@dataclass(frozen=True, slots=True)
class TimedPoint:
    time: float
    position: Vec2


@dataclass(frozen=True, slots=True)
class SippPlan:
    points: tuple[TimedPoint, ...]
    arrival_time: float

    @property
    def destination(self) -> Vec2:
        return self.points[-1].position


def reservation_segments(trajectory: Sequence[tuple[float, float, float] | TimedPoint]) -> tuple[ReservationSegment, ...]:
    """Convert timestamped points into the linear segments used by SIPP."""
    points: list[TimedPoint] = []
    for item in trajectory:
        if isinstance(item, TimedPoint):
            point = item
        else:
            point = TimedPoint(float(item[0]), (float(item[1]), float(item[2])))
        if points and point.time < points[-1].time - MERGE_EPS:
            raise ValueError("trajectory times must be monotonic")
        points.append(point)
    segments: list[ReservationSegment] = []
    for left, right in zip(points, points[1:]):
        if right.time <= left.time + EPS:
            continue
        segments.append(ReservationSegment(left.time, right.time, left.position, right.position))
    return tuple(segments)


def forbidden_interval_at_point(
    point: Vec2,
    segment: ReservationSegment,
    radius: float = RESERVATION_RADIUS,
    lower: float | None = None,
    upper: float | None = None,
) -> tuple[float, float] | None:
    """Return the exact time interval where a moving reservation reaches ``point``."""
    start_time = segment.start_time if lower is None else max(segment.start_time, lower)
    end_time = segment.end_time if upper is None else min(segment.end_time, upper)
    if end_time < start_time - EPS:
        return None
    if not _bounds_overlap(_segment_bounds(segment.start, segment.end, radius), (point[0], point[0], point[1], point[1])):
        return None

    vx, vy = segment.velocity
    rx, ry = segment.start[0] - point[0], segment.start[1] - point[1]
    a = vx * vx + vy * vy
    b = 2.0 * (rx * vx + ry * vy)
    c = rx * rx + ry * ry - radius * radius
    if a <= EPS:
        return (start_time, end_time) if c <= EPS else None
    discriminant = b * b - 4.0 * a * c
    if discriminant < -EPS:
        return None
    root = sqrt(max(0.0, discriminant))
    first = (-b - root) / (2.0 * a) + segment.start_time
    last = (-b + root) / (2.0 * a) + segment.start_time
    first, last = max(first, start_time), min(last, end_time)
    return (first, last) if first <= last + EPS else None


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((left, right) for left, right in intervals if right >= left - EPS)
    merged: list[tuple[float, float]] = []
    for left, right in ordered:
        if not merged or left > merged[-1][1] + MERGE_EPS:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    return merged


def safe_intervals(
    point: Vec2,
    reservations: Sequence[ReservationSegment],
    now: float,
    horizon: float,
    radius: float = RESERVATION_RADIUS,
) -> tuple[tuple[float, float], ...]:
    """Return the complement of analytically computed reservation intervals."""
    if horizon < now - EPS:
        return ()
    forbidden = []
    point_bounds = (point[0], point[0], point[1], point[1])
    for segment in reservations:
        if not _bounds_overlap(segment.bounds, point_bounds):
            continue
        interval = forbidden_interval_at_point(point, segment, radius, now, horizon)
        if interval is not None:
            forbidden.append(interval)
    merged = _merge_intervals(forbidden)
    result: list[tuple[float, float]] = []
    cursor = now
    for left, right in merged:
        left, right = max(now, left), min(horizon, right)
        if left > cursor + MERGE_EPS:
            result.append((cursor, left))
        cursor = max(cursor, right)
        if cursor >= horizon - MERGE_EPS:
            break
    if cursor < horizon - MERGE_EPS or not result and not merged:
        result.append((cursor, horizon))
    return tuple(result)


def edge_conflicts_reservation(
    start: Vec2,
    end: Vec2,
    depart_time: float,
    arrive_time: float,
    reservation: ReservationSegment,
    radius: float = RESERVATION_RADIUS,
) -> bool:
    """Continuously test a candidate edge against one moving reservation."""
    overlap_start = max(depart_time, reservation.start_time)
    overlap_end = min(arrive_time, reservation.end_time)
    if overlap_end < overlap_start - EPS:
        return False
    candidate_duration = max(EPS, arrive_time - depart_time)
    candidate_velocity = (
        (end[0] - start[0]) / candidate_duration,
        (end[1] - start[1]) / candidate_duration,
    )
    reserved_velocity = reservation.velocity
    candidate_at_start = _lerp(start, end, (overlap_start - depart_time) / candidate_duration)
    reserved_at_start = _lerp(
        reservation.start,
        reservation.end,
        (overlap_start - reservation.start_time) / max(EPS, reservation.end_time - reservation.start_time),
    )
    relative = (candidate_at_start[0] - reserved_at_start[0], candidate_at_start[1] - reserved_at_start[1])
    relative_velocity = (
        candidate_velocity[0] - reserved_velocity[0],
        candidate_velocity[1] - reserved_velocity[1],
    )
    duration = max(0.0, overlap_end - overlap_start)
    speed_squared = relative_velocity[0] ** 2 + relative_velocity[1] ** 2
    candidates = [0.0, duration]
    if speed_squared > EPS:
        closest = -(
            relative[0] * relative_velocity[0] + relative[1] * relative_velocity[1]
        ) / speed_squared
        candidates.append(min(duration, max(0.0, closest)))
    return any(
        hypot(
            relative[0] + relative_velocity[0] * candidate,
            relative[1] + relative_velocity[1] * candidate,
        ) < radius - EPS
        for candidate in candidates
    )


def edge_conflicts(
    start: Vec2,
    end: Vec2,
    depart_time: float,
    arrive_time: float,
    reservations: Sequence[ReservationSegment],
) -> bool:
    edge_bounds = _segment_bounds(start, end, RESERVATION_RADIUS)
    return any(
        _bounds_overlap(edge_bounds, reservation.bounds)
        and edge_conflicts_reservation(start, end, depart_time, arrive_time, reservation)
        for reservation in reservations
    )


class SpatialGraph:
    """Small visibility roadmap shared by static routing and SIPP."""

    def __init__(self, points: Sequence[Vec2], adjacency: Sequence[Sequence[int]]) -> None:
        self.points = tuple(points)
        self.adjacency = tuple(tuple(sorted(neighbors)) for neighbors in adjacency)

    @classmethod
    def from_obstacles(
        cls,
        obstacles: tuple[CircleObstacle | RectangleObstacle, ...],
    ) -> "SpatialGraph":
        points: list[Vec2] = []
        clearance = 0.25 + PATH_CLEARANCE + ROADMAP_MARGIN
        for obstacle in obstacles:
            if isinstance(obstacle, CircleObstacle):
                radius = obstacle.radius + clearance
                for index in range(12):
                    angle = 2.0 * pi * index / 12.0
                    points.append((obstacle.center[0] + radius * cos(angle), obstacle.center[1] + radius * sin(angle)))
            else:
                points.extend(
                    (
                        (obstacle.x_min - clearance, obstacle.y_min - clearance),
                        (obstacle.x_min - clearance, obstacle.y_max + clearance),
                        (obstacle.x_max + clearance, obstacle.y_min - clearance),
                        (obstacle.x_max + clearance, obstacle.y_max + clearance),
                    )
                )
        deduplicated: list[Vec2] = []
        for point in points:
            if not any(_distance(point, existing) < 1.0e-6 for existing in deduplicated):
                deduplicated.append(point)
        adjacency = [[] for _ in deduplicated]
        for left_index, left in enumerate(deduplicated):
            for right_index in range(left_index):
                right = deduplicated[right_index]
                if _segment_clear(left, right, obstacles):
                    adjacency[left_index].append(right_index)
                    adjacency[right_index].append(left_index)
        return cls(deduplicated, adjacency)

    def with_endpoints(self, start: Vec2, goal: Vec2, obstacles: tuple[CircleObstacle | RectangleObstacle, ...]) -> "SpatialGraph":
        points = list(self.points) + [start, goal]
        adjacency = [list(neighbors) for neighbors in self.adjacency] + [[], []]
        start_index, goal_index = len(points) - 2, len(points) - 1
        for index, point in enumerate(points[:-2]):
            if _segment_clear(start, point, obstacles):
                adjacency[start_index].append(index)
                adjacency[index].append(start_index)
            if _segment_clear(goal, point, obstacles):
                adjacency[goal_index].append(index)
                adjacency[index].append(goal_index)
        if _segment_clear(start, goal, obstacles):
            adjacency[start_index].append(goal_index)
            adjacency[goal_index].append(start_index)
        return SpatialGraph(points, adjacency)

    def static_distances(self, goal_index: int) -> tuple[float, ...]:
        distances = [float("inf")] * len(self.points)
        distances[goal_index] = 0.0
        queue = [(0.0, goal_index)]
        while queue:
            distance, node = heappop(queue)
            if distance > distances[node] + EPS:
                continue
            for neighbor in self.adjacency[node]:
                candidate = distance + _distance(self.points[node], self.points[neighbor])
                if candidate < distances[neighbor] - EPS:
                    distances[neighbor] = candidate
                    heappush(queue, (candidate, neighbor))
        return tuple(distances)


class SippPlanner:
    """Prioritized Safe Interval Path Planner over a small spatial roadmap."""

    def __init__(
        self,
        graph: SpatialGraph,
        obstacles: tuple[CircleObstacle | RectangleObstacle, ...],
        reservations: Sequence[ReservationSegment],
        now: float,
        horizon: float = PLANNING_HORIZON,
        speed: float = 1.0,
    ) -> None:
        self.graph = graph
        self.obstacles = obstacles
        self.reservations = tuple(reservations)
        self.now = now
        self.horizon = horizon
        self.speed = max(0.1, speed)

    def _edge_duration(self, graph: SpatialGraph, left: int, right: int) -> float:
        distance = _distance(graph.points[left], graph.points[right])
        return distance / (EDGE_SPEED_FACTOR * self.speed) + EDGE_TRACKING_SLACK

    def _earliest_departure(
        self,
        graph: SpatialGraph,
        left: int,
        right: int,
        earliest: float,
        latest: float,
        duration: float,
    ) -> float | None:
        if latest < earliest - EPS:
            return None
        departure = earliest
        while departure <= latest + EPS:
            arrival = departure + duration
            if not edge_conflicts(
                graph.points[left],
                graph.points[right],
                departure,
                arrival,
                self.reservations,
            ):
                return departure
            departure += DEPARTURE_GRANULARITY
        return None

    def plan(self, start: Vec2, goal: Vec2) -> SippPlan | None:
        graph = self.graph.with_endpoints(start, goal, self.obstacles)
        start_index, goal_index = len(graph.points) - 2, len(graph.points) - 1
        intervals = [safe_intervals(point, self.reservations, self.now, self.now + self.horizon) for point in graph.points]
        start_interval = next(
            (index for index, interval in enumerate(intervals[start_index]) if interval[0] - EPS <= self.now <= interval[1] + EPS),
            None,
        )
        if start_interval is None:
            return None

        heuristic_distances = graph.static_distances(goal_index)
        root = (start_index, start_interval)
        best: dict[tuple[int, int], float] = {root: self.now}
        parent: dict[tuple[int, int], tuple[tuple[int, int], float] | None] = {root: None}
        queue: list[tuple[float, float, int, int]] = []
        heappush(queue, (self.now + heuristic_distances[start_index] / self.speed, self.now, start_index, start_interval))
        goal_state: tuple[int, int] | None = None

        while queue:
            _, arrival, node, interval_id = heappop(queue)
            state = (node, interval_id)
            if arrival > best.get(state, float("inf")) + EPS:
                continue
            if node == goal_index:
                goal_state = state
                break
            current_interval = intervals[node][interval_id]
            for neighbor in graph.adjacency[node]:
                duration = self._edge_duration(graph, node, neighbor)
                for neighbor_interval_id, neighbor_interval in enumerate(intervals[neighbor]):
                    earliest = max(arrival, neighbor_interval[0] - duration)
                    latest = min(current_interval[1], neighbor_interval[1] - duration)
                    departure = self._earliest_departure(graph, node, neighbor, earliest, latest, duration)
                    if departure is None:
                        continue
                    neighbor_arrival = departure + duration
                    neighbor_state = (neighbor, neighbor_interval_id)
                    if neighbor_arrival >= best.get(neighbor_state, float("inf")) - EPS:
                        continue
                    best[neighbor_state] = neighbor_arrival
                    parent[neighbor_state] = (state, departure)
                    heuristic = heuristic_distances[neighbor] / self.speed
                    heappush(queue, (neighbor_arrival + heuristic, neighbor_arrival, neighbor, neighbor_interval_id))
                    break

        if goal_state is None:
            return None
        states = []
        current = goal_state
        while current is not None:
            states.append(current)
            previous = parent[current]
            current = previous[0] if previous is not None else None
        states.reverse()
        timed: list[TimedPoint] = [TimedPoint(self.now, graph.points[start_index])]
        for child in states[1:]:
            previous = parent[child]
            if previous is None:
                continue
            parent_state, departure = previous
            parent_node = parent_state[0]
            child_node = child[0]
            parent_arrival = best[parent_state]
            if departure > parent_arrival + MERGE_EPS:
                timed.append(TimedPoint(departure, graph.points[parent_node]))
            timed.append(TimedPoint(best[child], graph.points[child_node]))
        return SippPlan(tuple(timed), best[goal_state])


def _static_route(graph: SpatialGraph, obstacles: tuple[CircleObstacle | RectangleObstacle, ...], start: Vec2, goal: Vec2) -> tuple[Vec2, ...]:
    """Fallback route used when reservations make SIPP temporarily infeasible."""
    augmented = graph.with_endpoints(start, goal, obstacles)
    start_index, goal_index = len(augmented.points) - 2, len(augmented.points) - 1
    distances = [float("inf")] * len(augmented.points)
    parent: list[int | None] = [None] * len(augmented.points)
    distances[start_index] = 0.0
    queue = [(0.0, start_index)]
    while queue:
        distance, node = heappop(queue)
        if distance > distances[node] + EPS:
            continue
        if node == goal_index:
            break
        for neighbor in augmented.adjacency[node]:
            candidate = distance + _distance(augmented.points[node], augmented.points[neighbor])
            if candidate < distances[neighbor] - EPS:
                distances[neighbor] = candidate
                parent[neighbor] = node
                heappush(queue, (candidate, neighbor))
    if not isfinite(distances[goal_index]):
        return (start, goal)
    route = []
    current: int | None = goal_index
    while current is not None:
        route.append(augmented.points[current])
        current = parent[current]
    route.reverse()
    return tuple(route)


def _goal_target(goal, drone: DroneSnapshot, lane: float) -> Vec2:
    y = min(goal.y_max - 0.7, max(goal.y_min + 0.7, lane))
    return (goal.center[0], y)


class SwarmController(BaseSwarmController):
    """Initial SIPP-integrated movement controller; fire control is added next."""

    def initialize(self, game_info: GameInfo) -> None:
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.direction = 1.0 if self.team is Team.A else -1.0
        self.graph = SpatialGraph.from_obstacles(self.obstacles)
        ordered = sorted(game_info.own_initial_drones, key=lambda drone: (drone.drone_type.value, drone.id))
        self.lanes = {
            drone.id: self.goal.y_min
            + 0.75
            + (self.goal.y_max - self.goal.y_min - 1.5) * (index + 0.5) / max(1, len(ordered))
            for index, drone in enumerate(ordered)
        }
        self.plans: dict[int, SippPlan] = {}
        self.predictions: dict[int, tuple[tuple[float, float, float], ...]] = {}
        self.last_plan_time = -float("inf")
        self.aging: dict[int, int] = {}

    def _destination(self, drone: DroneSnapshot, state: GameState) -> Vec2:
        return _goal_target(self.goal, drone, self.lanes.get(drone.id, self.goal.center[1]))

    def _priority(self, drone: DroneSnapshot, state: GameState) -> float:
        remaining = _distance(drone.position, self.goal.center)
        time_left = max(1.0, 90.0 - state.time)
        urgency = remaining / max(0.1, self.specs[drone.drone_type].max_speed) / time_left
        value = self.specs[drone.drone_type].point_value
        class_bonus = 20.0 if drone.drone_type is DroneType.TRANSPORT else 10.0 if drone.drone_type is DroneType.TANK else 0.0
        return 100.0 * urgency + class_bonus + min(10.0, 2.0 * self.aging.get(drone.id, 0)) - 0.01 * drone.id + value

    def _steer(self, drone: DroneSnapshot, target: Vec2, *, hold: bool = False) -> Vec2:
        spec = self.specs[drone.drone_type]
        dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
        distance = hypot(dx, dy)
        if hold or distance < 1.0e-6:
            desired_velocity = (0.0, 0.0)
            forward = (self.direction, 0.0)
        else:
            forward = (dx / distance, dy / distance)
            desired_speed = min(spec.max_speed, sqrt(max(0.0, 2.0 * spec.max_acceleration * distance)))
            desired_velocity = (forward[0] * desired_speed, forward[1] * desired_speed)
        ax = 2.25 * (desired_velocity[0] - drone.velocity[0])
        ay = 2.25 * (desired_velocity[1] - drone.velocity[1])
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                center, radius = obstacle.center, obstacle.radius
            else:
                center = ((obstacle.x_min + obstacle.x_max) * 0.5, (obstacle.y_min + obstacle.y_max) * 0.5)
                radius = hypot(obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min) * 0.5
            ox, oy = center[0] - drone.position[0], center[1] - drone.position[1]
            along = ox * forward[0] + oy * forward[1]
            lateral = ox * -forward[1] + oy * forward[0]
            safe = radius + 1.1
            if -0.5 < along < 8.5 and abs(lateral) < safe:
                side = -1.0 if lateral > 0.0 else 1.0
                if abs(lateral) < 0.05:
                    side = 1.0 if drone.id % 2 == 0 else -1.0
                strength = spec.max_acceleration * 1.6 * (1.0 - max(0.0, along) / 8.5)
                ax += -forward[1] * side * strength
                ay += forward[0] * side * strength
            center_distance = hypot(ox, oy)
            surface_distance = center_distance - radius
            if EPS < surface_distance < 2.6:
                strength = spec.max_acceleration * (2.6 - surface_distance) / 2.6
                ax -= ox / center_distance * strength
                ay -= oy / center_distance * strength
        return clip_vector((ax, ay), spec.max_acceleration)

    def _plan_routes(self, state: GameState, active: list[DroneSnapshot]) -> dict[int, SippPlan]:
        reservations: list[ReservationSegment] = []
        routes: dict[int, SippPlan] = {}
        for drone in sorted(active, key=lambda item: (-self._priority(item, state), item.id)):
            target = self._destination(drone, state)
            planner = SippPlanner(
                self.graph,
                self.obstacles,
                reservations,
                state.time,
                speed=self.specs[drone.drone_type].max_speed,
            )
            plan = planner.plan(drone.position, target)
            if plan is None:
                fallback = _static_route(self.graph, self.obstacles, drone.position, target)
                timed = []
                elapsed = state.time
                for left, right in zip(fallback, fallback[1:]):
                    timed.append(TimedPoint(elapsed, left))
                    elapsed += _distance(left, right) / max(0.1, self.specs[drone.drone_type].max_speed * EDGE_SPEED_FACTOR) + EDGE_TRACKING_SLACK
                timed.append(TimedPoint(elapsed, fallback[-1]))
                plan = SippPlan(tuple(timed), elapsed)
                self.aging[drone.id] = min(5, self.aging.get(drone.id, 0) + 1)
            else:
                self.aging[drone.id] = max(0, self.aging.get(drone.id, 0) - 1)
            routes[drone.id] = plan
            prediction = self._predict_route(drone, plan, state.time, 3.0)
            self.predictions[drone.id] = prediction
            reservations.extend(reservation_segments(prediction))
        return routes

    def _predict_route(
        self,
        drone: DroneSnapshot,
        plan: SippPlan,
        now: float,
        horizon: float,
    ) -> tuple[tuple[float, float, float], ...]:
        dynamic = DynamicState(drone.position, drone.velocity, drone.acceleration)
        samples = [(now, dynamic.position[0], dynamic.position[1])]
        for index in range(round(horizon / 0.05)):
            timestamp = now + index * 0.05
            target, hold = self._plan_target(plan, timestamp)
            command = self._steer(
                DroneSnapshot(
                    drone.id,
                    drone.team,
                    drone.drone_type,
                    dynamic.position,
                    dynamic.velocity,
                    dynamic.acceleration,
                    DroneStatus.ACTIVE,
                ),
                target,
                hold=hold,
            )
            dynamic = advance_dynamics(dynamic, command, self.specs[drone.drone_type], 0.05)
            samples.append((timestamp + 0.05, dynamic.position[0], dynamic.position[1]))
        return tuple(samples)

    @staticmethod
    def _plan_target(plan: SippPlan, timestamp: float) -> tuple[Vec2, bool]:
        points = plan.points
        for index in range(1, len(points)):
            if timestamp <= points[index].time + EPS:
                previous = points[index - 1]
                if points[index].time - previous.time > MERGE_EPS and _distance(previous.position, points[index].position) > EPS:
                    if timestamp < points[index].time - MERGE_EPS:
                        return points[index].position, False
                return previous.position, True
        return points[-1].position, True

    def _emergency_guard(self, drone: DroneSnapshot, command: Vec2, state: GameState, priority: float) -> Vec2:
        for friend in state.own_drones:
            if friend.id == drone.id or friend.status is not DroneStatus.ACTIVE:
                continue
            if self._priority(friend, state) <= priority:
                continue
            relative = (friend.position[0] - drone.position[0], friend.position[1] - drone.position[1])
            relative_velocity = (friend.velocity[0] - drone.velocity[0], friend.velocity[1] - drone.velocity[1])
            speed_squared = relative_velocity[0] ** 2 + relative_velocity[1] ** 2
            closest = 0.0
            if speed_squared > EPS:
                closest = max(0.0, min(1.0, -(
                    relative[0] * relative_velocity[0] + relative[1] * relative_velocity[1]
                ) / speed_squared))
            miss = hypot(relative[0] + relative_velocity[0] * closest, relative[1] + relative_velocity[1] * closest)
            if miss < 0.82:
                spec = self.specs[drone.drone_type]
                away = hypot(relative[0], relative[1])
                if away < EPS:
                    return clip_vector((-drone.velocity[0] * 3.0, -drone.velocity[1] * 3.0), spec.max_acceleration)
                yield_acceleration = (-relative[0] / away * spec.max_acceleration, -relative[1] / away * spec.max_acceleration)
                braking = (-drone.velocity[0] * 3.0, -drone.velocity[1] * 3.0)
                return clip_vector((yield_acceleration[0] + braking[0], yield_acceleration[1] + braking[1]), spec.max_acceleration)
        return command

    def step(self, state: GameState) -> dict[int, tuple[float, float] | dict[str, Vec2]]:
        active = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        if state.time - self.last_plan_time >= REPLAN_PERIOD - EPS or set(self.plans) != {drone.id for drone in active}:
            self.plans = self._plan_routes(state, active)
            self.last_plan_time = state.time
        actions = {}
        for drone in active:
            plan = self.plans.get(drone.id)
            if plan is None:
                command = self._steer(drone, self._destination(drone, state))
            else:
                target, hold = self._plan_target(plan, state.time)
                command = self._steer(drone, target, hold=hold)
            actions[drone.id] = self._emergency_guard(drone, command, state, self._priority(drone, state))
        return actions
