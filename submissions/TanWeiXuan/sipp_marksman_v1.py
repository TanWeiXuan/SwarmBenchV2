"""Prioritized SIPP traffic scheduling with conservative Tank fire control.

The controller deliberately keeps the authoritative engine outside the
submission.  It uses the engine's public snapshots and its deterministic
geometry/dynamics helpers for prediction, while treating SIPP as a timing
layer above a small jerk-aware tracker.
"""

from __future__ import annotations

from heapq import heappop, heappush
from math import atan2, cos, hypot, isfinite, pi, sin, sqrt
from typing import Iterable, NamedTuple, Sequence

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
from swarmbench.api import CONTROLLER_PERIOD, PHYSICS_DT, PROJECTILE_CONTACT_RADIUS, TANK_WEAPON_SPEC, WeaponSpec
from swarmbench.engine.collisions import (
    swept_arena_exit,
    swept_obstacle_contact,
    swept_points_contact,
)
from swarmbench.engine.dynamics import DynamicState, advance_dynamics, clip_vector


EPS = 1.0e-9
MERGE_EPS = 1.0e-7
VEHICLE_COLLISION_RADIUS = 0.75
# Keep the analytic reservation primitive close to the engine's center
# collision radius.  The controller asks SIPP for a wider tracking tube below,
# because the jerk-limited low-level tracker can lag a timed roadmap edge.
RESERVATION_MARGIN = 0.15
RESERVATION_RADIUS = VEHICLE_COLLISION_RADIUS + RESERVATION_MARGIN
TRACKING_RESERVATION_MARGIN = 0.60
TRACKING_RESERVATION_RADIUS = VEHICLE_COLLISION_RADIUS + TRACKING_RESERVATION_MARGIN
PATH_CLEARANCE = 0.35
ROADMAP_MARGIN = 0.18
PLANNING_HORIZON = 12.0
REPLAN_PERIOD = 0.7
EDGE_SPEED_FACTOR = 0.80
EDGE_TRACKING_SLACK = 0.35
DEPARTURE_GRANULARITY = 0.10
RESERVATION_SAMPLE_PERIOD = 0.30
RECEDING_GOAL_FRACTION = 0.55
PREDICTION_HORIZON = 6.0
EMERGENCY_GUARD_HORIZON = 4.5


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


class ReservationSegment(NamedTuple):
    start_time: float
    end_time: float
    start: Vec2
    end: Vec2
    velocity: Vec2
    bounds: tuple[float, float, float, float]


class TimedPoint(NamedTuple):
    time: float
    position: Vec2


class SippPlan(NamedTuple):
    points: tuple[TimedPoint, ...]
    arrival_time: float

    @property
    def destination(self) -> Vec2:
        return self.points[-1].position


def reservation_segments(
    trajectory: Sequence[tuple[float, float, float] | TimedPoint],
    radius: float = RESERVATION_RADIUS,
) -> tuple[ReservationSegment, ...]:
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
        duration = right.time - left.time
        velocity = (
            (right.position[0] - left.position[0]) / duration,
            (right.position[1] - left.position[1]) / duration,
        )
        segments.append(
            ReservationSegment(
                left.time,
                right.time,
                left.position,
                right.position,
                velocity,
                _segment_bounds(left.position, right.position, radius),
            )
        )
    return tuple(segments)


def _coarsen_trajectory(
    trajectory: Sequence[tuple[float, float, float]],
    period: float = RESERVATION_SAMPLE_PERIOD,
) -> tuple[tuple[float, float, float], ...]:
    """Keep the forward-simulated path while bounding SIPP reservation work."""
    if len(trajectory) <= 2:
        return tuple(trajectory)
    selected = [trajectory[0]]
    next_time = trajectory[0][0] + period
    for sample in trajectory[1:-1]:
        if sample[0] + EPS >= next_time:
            selected.append(sample)
            next_time = sample[0] + period
    selected.append(trajectory[-1])
    return tuple(selected)


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
        reservation_radius: float = RESERVATION_RADIUS,
    ) -> None:
        self.graph = graph
        self.obstacles = obstacles
        self.reservations = tuple(reservations)
        self.now = now
        self.horizon = horizon
        self.speed = max(0.1, speed)
        self.reservation_radius = max(0.0, reservation_radius)

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
        edge_bounds = _segment_bounds(graph.points[left], graph.points[right], self.reservation_radius)
        relevant = tuple(
            reservation for reservation in self.reservations if _bounds_overlap(edge_bounds, reservation.bounds)
        )
        if not relevant:
            return earliest
        departure = earliest
        while departure <= latest + EPS:
            arrival = departure + duration
            if not any(
                edge_conflicts_reservation(
                    graph.points[left],
                    graph.points[right],
                    departure,
                    arrival,
                    reservation,
                    self.reservation_radius,
                )
                for reservation in relevant
            ):
                return departure
            departure += DEPARTURE_GRANULARITY
        return None

    def plan(self, start: Vec2, goal: Vec2) -> SippPlan | None:
        graph = self.graph.with_endpoints(start, goal, self.obstacles)
        start_index, goal_index = len(graph.points) - 2, len(graph.points) - 1
        intervals = [
            safe_intervals(
                point,
                self.reservations,
                self.now,
                self.now + self.horizon,
                self.reservation_radius,
            )
            for point in graph.points
        ]
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


def _receding_goal(route: Sequence[Vec2], speed: float, horizon: float) -> Vec2:
    """Choose a reachable local route point for finite-horizon replanning."""
    budget = max(2.0, speed * horizon * RECEDING_GOAL_FRACTION)
    travelled = 0.0
    for left, right in zip(route, route[1:]):
        edge = _distance(left, right)
        if travelled + edge > budget:
            fraction = max(0.0, min(1.0, (budget - travelled) / max(EPS, edge)))
            return _lerp(left, right, fraction)
        travelled += edge
    return route[-1]


class ShotEvaluation(NamedTuple):
    target_id: int
    aim_direction: Vec2
    hit_fraction: float
    hard_evasion_hit_fraction: float
    mean_margin: float
    target_hit_time: float | None
    friendly_first_fraction: float
    terrain_first_fraction: float


class _ShotOutcome(NamedTuple):
    first_kind: str | None
    first_id: int | None
    hit_time: float | None
    margin: float


def _unit(vector: Vec2) -> Vec2:
    length = hypot(vector[0], vector[1])
    if length <= EPS:
        return (0.0, 0.0)
    return (vector[0] / length, vector[1] / length)


def _add(left: Vec2, right: Vec2) -> Vec2:
    return (left[0] + right[0], left[1] + right[1])


def _scale(vector: Vec2, factor: float) -> Vec2:
    return (vector[0] * factor, vector[1] * factor)


def _minimum_relative_distance(left_start: Vec2, left_end: Vec2, right_start: Vec2, right_end: Vec2) -> float:
    relative_start = (left_start[0] - right_start[0], left_start[1] - right_start[1])
    relative_velocity = (
        (left_end[0] - left_start[0]) - (right_end[0] - right_start[0]),
        (left_end[1] - left_start[1]) - (right_end[1] - right_start[1]),
    )
    speed_squared = relative_velocity[0] ** 2 + relative_velocity[1] ** 2
    if speed_squared <= EPS:
        return hypot(*relative_start)
    closest = max(
        0.0,
        min(
            1.0,
            -(
                relative_start[0] * relative_velocity[0]
                + relative_start[1] * relative_velocity[1]
            )
            / speed_squared,
        ),
    )
    return hypot(
        relative_start[0] + relative_velocity[0] * closest,
        relative_start[1] + relative_velocity[1] * closest,
    )


class ConservativeFireControl:
    """Deterministic limited-ammunition fire control with hard safety gates."""

    NORMAL_HIT_THRESHOLD = 0.90
    NORMAL_HARD_THRESHOLD = 0.75
    URGENT_HIT_THRESHOLD = 0.84
    URGENT_HARD_THRESHOLD = 0.60
    MAX_TARGETS_TO_EVALUATE = 2
    MAX_FLIGHT_TIME = 6.0
    AIM_OFFSETS_DEGREES = (0.0, 0.40, -0.40, 1.0, -1.0)
    HARD_MODES = frozenset({"brake", "left", "right", "forward_left", "forward_right", "brake_left", "brake_right"})

    def __init__(
        self,
        team: Team,
        width: float,
        height: float,
        obstacles: tuple[CircleObstacle | RectangleObstacle, ...],
        specs: dict[DroneType, DroneSpec],
        weapon: WeaponSpec,
        own_goal,
    ) -> None:
        self.team = team
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.specs = specs
        self.weapon = weapon
        self.own_goal = own_goal
        self.histories: dict[int, list[tuple[float, Vec2, Vec2, Vec2]]] = {}
        self.last_evaluations: tuple[ShotEvaluation, ...] = ()

    def observe(self, state: GameState) -> None:
        cutoff = state.time - 1.0
        active_ids = set()
        for enemy in state.opponent_drones:
            if enemy.status is not DroneStatus.ACTIVE:
                continue
            active_ids.add(enemy.id)
            history = self.histories.setdefault(enemy.id, [])
            history.append((state.time, enemy.position, enemy.velocity, enemy.acceleration))
            self.histories[enemy.id] = [sample for sample in history if sample[0] >= cutoff]
        for enemy_id in tuple(self.histories):
            if enemy_id not in active_ids and self.histories[enemy_id][-1][0] < cutoff:
                del self.histories[enemy_id]

    @staticmethod
    def _lead_point(shooter: DroneSnapshot, target: DroneSnapshot, projectile_speed: float) -> Vec2 | None:
        rx = target.position[0] - shooter.position[0]
        ry = target.position[1] - shooter.position[1]
        vx, vy = target.velocity
        a = vx * vx + vy * vy - projectile_speed * projectile_speed
        b = 2.0 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry
        roots: list[float] = []
        if abs(a) <= EPS:
            if abs(b) > EPS:
                roots.append(-c / b)
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant >= -EPS:
                root = sqrt(max(0.0, discriminant))
                roots.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        positive = [value for value in roots if value > EPS and isfinite(value)]
        if not positive:
            return None
        time = min(positive)
        point = (target.position[0] + target.velocity[0] * time, target.position[1] + target.velocity[1] * time)
        return point if all(isfinite(value) for value in point) else None

    def _maneuver_command(
        self,
        target: DroneSnapshot,
        dynamic: DynamicState,
        mode: str,
        elapsed: float,
        aim_direction: Vec2,
        shooter_position: Vec2,
        switch_time: float,
    ) -> Vec2:
        spec = self.specs[target.drone_type]
        if elapsed < CONTROLLER_PERIOD - EPS:
            return clip_vector(target.acceleration, spec.max_acceleration)
        if mode == "switch_left_right":
            mode = "left" if elapsed < switch_time else "right"
        elif mode == "switch_right_left":
            mode = "right" if elapsed < switch_time else "left"
        elif mode == "switch_brake_left":
            mode = "brake" if elapsed < switch_time else "left"
        elif mode == "switch_brake_right":
            mode = "brake" if elapsed < switch_time else "right"

        velocity_direction = _unit(dynamic.velocity)
        if velocity_direction == (0.0, 0.0):
            velocity_direction = aim_direction
        forward = velocity_direction
        brake = (-forward[0], -forward[1])
        left = (-aim_direction[1], aim_direction[0])
        cross = aim_direction[0] * (dynamic.position[1] - shooter_position[1]) - aim_direction[1] * (dynamic.position[0] - shooter_position[0])
        away = (-left[0], -left[1]) if cross > 0.0 else left
        vectors = {
            "continue": dynamic.acceleration,
            "zero": (0.0, 0.0),
            "brake": _scale(brake, spec.max_acceleration),
            "forward": _scale(forward, spec.max_acceleration),
            "left": _scale(left, spec.max_acceleration),
            "right": _scale((-left[0], -left[1]), spec.max_acceleration),
            "half_left": _scale(left, spec.max_acceleration * 0.5),
            "half_right": _scale((-left[0], -left[1]), spec.max_acceleration * 0.5),
            "forward_left": _scale(_unit(_add(forward, left)), spec.max_acceleration),
            "forward_right": _scale(_unit(_add(forward, (-left[0], -left[1]))), spec.max_acceleration),
            "brake_left": _scale(_unit(_add(brake, left)), spec.max_acceleration),
            "brake_right": _scale(_unit(_add(brake, (-left[0], -left[1]))), spec.max_acceleration),
            "dodge_left": _scale(away, spec.max_acceleration),
            "dodge_right": _scale((-away[0], -away[1]), spec.max_acceleration),
        }
        return clip_vector(vectors.get(mode, (0.0, 0.0)), spec.max_acceleration)

    def _target_trajectory(
        self,
        shooter: DroneSnapshot,
        target: DroneSnapshot,
        aim_direction: Vec2,
        mode: str,
        flight_time: float,
    ) -> tuple[Vec2, ...]:
        dynamic = DynamicState(target.position, target.velocity, target.acceleration)
        samples = [dynamic.position]
        switch_time = max(0.5, min(1.5, flight_time * 0.45))
        for index in range(round(flight_time / PHYSICS_DT)):
            elapsed = index * PHYSICS_DT
            command = self._maneuver_command(
                target,
                dynamic,
                mode,
                elapsed,
                aim_direction,
                shooter.position,
                switch_time,
            )
            dynamic = advance_dynamics(dynamic, command, self.specs[target.drone_type], PHYSICS_DT)
            samples.append(dynamic.position)
        return tuple(samples)

    @staticmethod
    def _prediction_position(prediction: Sequence[tuple[float, float, float]], timestamp: float) -> Vec2:
        if not prediction:
            return (0.0, 0.0)
        if timestamp <= prediction[0][0]:
            return (prediction[0][1], prediction[0][2])
        for left, right in zip(prediction, prediction[1:]):
            if timestamp <= right[0] + EPS:
                fraction = (timestamp - left[0]) / max(EPS, right[0] - left[0])
                return _lerp((left[1], left[2]), (right[1], right[2]), fraction)
        return (prediction[-1][1], prediction[-1][2])

    def _other_vehicle_position(
        self,
        vehicle: DroneSnapshot,
        elapsed: float,
        state: GameState,
        predictions: dict[int, tuple[tuple[float, float, float], ...]],
    ) -> Vec2:
        if vehicle.team is self.team and vehicle.id in predictions:
            return self._prediction_position(predictions[vehicle.id], state.time + elapsed)
        return (
            vehicle.position[0] + vehicle.velocity[0] * elapsed,
            vehicle.position[1] + vehicle.velocity[1] * elapsed,
        )

    def _build_vehicle_tracks(
        self,
        state: GameState,
        predictions: dict[int, tuple[tuple[float, float, float], ...]],
        steps: int,
    ) -> dict[int, tuple[Vec2, ...]]:
        tracks: dict[int, tuple[Vec2, ...]] = {}
        for vehicle in tuple(state.own_drones) + tuple(state.opponent_drones):
            prediction = predictions.get(vehicle.id)
            if prediction:
                positions = []
                for index in range(steps + 1):
                    timestamp = state.time + index * PHYSICS_DT
                    if index < len(prediction) and abs(prediction[index][0] - timestamp) <= 1.0e-6:
                        positions.append((prediction[index][1], prediction[index][2]))
                    else:
                        positions.append(self._prediction_position(prediction, timestamp))
                tracks[vehicle.id] = tuple(positions)
            else:
                tracks[vehicle.id] = tuple(
                    (
                        vehicle.position[0] + vehicle.velocity[0] * index * PHYSICS_DT,
                        vehicle.position[1] + vehicle.velocity[1] * index * PHYSICS_DT,
                    )
                    for index in range(steps + 1)
                )
        return tracks

    def _static_first_hit(
        self,
        shooter: DroneSnapshot,
        target: DroneSnapshot,
        aim_direction: Vec2,
        state: GameState,
        tracks: dict[int, tuple[Vec2, ...]],
        steps: int,
    ) -> tuple[float, int, int, str, int | None] | None:
        projectile_position = shooter.position
        projectile_velocity = _scale(aim_direction, self.weapon.projectile_speed)
        vehicles = tuple(state.own_drones) + tuple(state.opponent_drones)
        for index in range(steps):
            elapsed = index * PHYSICS_DT
            next_projectile = (
                projectile_position[0] + projectile_velocity[0] * PHYSICS_DT,
                projectile_position[1] + projectile_velocity[1] * PHYSICS_DT,
            )
            candidates: list[tuple[float, int, int, str, int | None]] = []
            for obstacle_index, obstacle in enumerate(self.obstacles):
                contact = swept_obstacle_contact(projectile_position, next_projectile, obstacle, 0.0)
                if contact is not None:
                    candidates.append((elapsed + contact * PHYSICS_DT, 0, obstacle_index, "terrain", None))
            for vehicle in vehicles:
                if vehicle.id in {shooter.id, target.id} or vehicle.status is not DroneStatus.ACTIVE:
                    continue
                vehicle_track = tracks[vehicle.id]
                contact = swept_points_contact(
                    projectile_position,
                    next_projectile,
                    vehicle_track[index],
                    vehicle_track[index + 1],
                    PROJECTILE_CONTACT_RADIUS,
                )
                if contact is not None:
                    candidates.append(
                        (
                            elapsed + contact * PHYSICS_DT,
                            2,
                            vehicle.id,
                            "friendly" if vehicle.team is self.team else "enemy",
                            vehicle.id,
                        )
                    )
            exit_contact = swept_arena_exit(projectile_position, next_projectile, self.width, self.height)
            if exit_contact is not None:
                candidates.append((elapsed + exit_contact * PHYSICS_DT, 5, -1, "exit", None))
            if candidates:
                return min(candidates)
            projectile_position = next_projectile
        return None

    def _first_hit(
        self,
        shooter: DroneSnapshot,
        target: DroneSnapshot,
        target_samples: Sequence[Vec2],
        aim_direction: Vec2,
        state: GameState,
        static_first: tuple[float, int, int, str, int | None] | None,
    ) -> _ShotOutcome:
        steps = len(target_samples) - 1
        projectile_position = shooter.position
        projectile_velocity = _scale(aim_direction, self.weapon.projectile_speed)
        closest_target_distance = float("inf")
        for index in range(steps):
            elapsed = index * PHYSICS_DT
            next_elapsed = elapsed + PHYSICS_DT
            next_projectile = (
                projectile_position[0] + projectile_velocity[0] * PHYSICS_DT,
                projectile_position[1] + projectile_velocity[1] * PHYSICS_DT,
            )
            target_start, target_end = target_samples[index], target_samples[index + 1]
            closest_target_distance = min(
                closest_target_distance,
                _minimum_relative_distance(projectile_position, next_projectile, target_start, target_end),
            )
            target_contact = swept_points_contact(
                projectile_position,
                next_projectile,
                target_start,
                target_end,
                PROJECTILE_CONTACT_RADIUS,
            )
            target_first: tuple[float, int, int, str, int | None] | None = None
            if target_contact is not None:
                target_first = (elapsed + target_contact * PHYSICS_DT, 2, target.id, "target", target.id)
            if static_first is not None and static_first[0] <= next_elapsed + EPS:
                first = min((candidate for candidate in (target_first, static_first) if candidate is not None))
                return _ShotOutcome(first[3], first[4], state.time + first[0], PROJECTILE_CONTACT_RADIUS - closest_target_distance)
            if target_first is not None:
                return _ShotOutcome("target", target.id, state.time + target_first[0], PROJECTILE_CONTACT_RADIUS - closest_target_distance)
            projectile_position = next_projectile
        if static_first is not None:
            return _ShotOutcome(static_first[3], static_first[4], state.time + static_first[0], PROJECTILE_CONTACT_RADIUS - closest_target_distance)
        return _ShotOutcome(None, None, None, PROJECTILE_CONTACT_RADIUS - closest_target_distance)

    def evaluate_target(
        self,
        shooter: DroneSnapshot,
        target: DroneSnapshot,
        state: GameState,
        predictions: dict[int, tuple[tuple[float, float, float], ...]],
        tracks: dict[int, tuple[Vec2, ...]] | None = None,
    ) -> ShotEvaluation | None:
        lead = self._lead_point(shooter, target, self.weapon.projectile_speed)
        if lead is None:
            return None
        base_direction = _unit((lead[0] - shooter.position[0], lead[1] - shooter.position[1]))
        if base_direction == (0.0, 0.0):
            return None
        flight_time = min(
            self.MAX_FLIGHT_TIME,
            max(0.25, _distance(shooter.position, lead) / max(EPS, self.weapon.projectile_speed) + 1.0),
        )
        modes = (
            "continue",
            "zero",
            "brake",
            "forward",
            "left",
            "right",
            "half_left",
            "half_right",
            "forward_left",
            "forward_right",
            "brake_left",
            "brake_right",
        )
        best: ShotEvaluation | None = None
        steps = round(flight_time / PHYSICS_DT)
        tracks = tracks or self._build_vehicle_tracks(state, predictions, steps)
        for offset_degrees in self.AIM_OFFSETS_DEGREES:
            offset = offset_degrees * pi / 180.0
            angle = atan2(base_direction[1], base_direction[0]) + offset
            aim_direction = (cos(angle), sin(angle))
            static_first = self._static_first_hit(shooter, target, aim_direction, state, tracks, steps)
            outcomes = []
            for mode in modes:
                target_samples = self._target_trajectory(shooter, target, aim_direction, mode, flight_time)
                outcomes.append(self._first_hit(shooter, target, target_samples, aim_direction, state, static_first))
            hits = [outcome for outcome in outcomes if outcome.first_kind == "target" and outcome.first_id == target.id]
            hard_outcomes = [outcome for mode, outcome in zip(modes, outcomes) if mode in self.HARD_MODES]
            hard_hits = [outcome for outcome in hard_outcomes if outcome.first_kind == "target" and outcome.first_id == target.id]
            hit_fraction = len(hits) / len(outcomes)
            hard_fraction = len(hard_hits) / len(hard_outcomes)
            mean_margin = sum(outcome.margin for outcome in outcomes) / len(outcomes)
            evaluation = ShotEvaluation(
                target.id,
                aim_direction,
                hit_fraction,
                hard_fraction,
                mean_margin,
                min((outcome.hit_time for outcome in hits if outcome.hit_time is not None), default=None),
                sum(outcome.first_kind == "friendly" for outcome in outcomes) / len(outcomes),
                sum(outcome.first_kind == "terrain" for outcome in outcomes) / len(outcomes),
            )
            if best is None or (
                evaluation.hit_fraction,
                evaluation.hard_evasion_hit_fraction,
                evaluation.mean_margin,
                -abs(offset_degrees),
            ) > (
                best.hit_fraction,
                best.hard_evasion_hit_fraction,
                best.mean_margin,
                0.0,
            ):
                best = evaluation
        return best

    def _urgent(self, target: DroneSnapshot, state: GameState) -> bool:
        distance_to_goal = _distance(target.position, self.own_goal.center)
        speed = max(0.1, self.specs[target.drone_type].max_speed)
        return target.drone_type is DroneType.TRANSPORT and distance_to_goal / speed < 8.0

    def _utility(self, target: DroneSnapshot, evaluation: ShotEvaluation, tank: DroneSnapshot, state: GameState) -> float:
        score_value = self.specs[target.drone_type].point_value
        goal_progress = max(0.0, 1.0 - _distance(target.position, self.own_goal.center) / 70.0)
        score_denial = score_value * (1.0 + 2.0 * goal_progress)
        urgency = 3.0 if self._urgent(target, state) else 0.0
        threat = 0.0
        if target.drone_type is DroneType.TANK:
            threat = 0.35 * float(target.shots_remaining or 0)
            nearest_transport = min(
                (
                    _distance(target.position, friend.position)
                    for friend in state.own_drones
                    if friend.status is DroneStatus.ACTIVE and friend.drone_type is DroneType.TRANSPORT
                ),
                default=100.0,
            )
            threat += max(0.0, 2.0 - nearest_transport / 8.0)
        finish = 1.0 if evaluation.target_hit_time is None else max(0.0, 1.0 - (evaluation.target_hit_time - state.time) / self.MAX_FLIGHT_TIME)
        ammo_cost = 0.25 + (0.45 if (tank.shots_remaining or 0) <= 2 else 0.0)
        return score_denial + urgency + threat + finish + 0.5 * evaluation.mean_margin - ammo_cost

    def choose_fire(
        self,
        tank: DroneSnapshot,
        state: GameState,
        predictions: dict[int, tuple[tuple[float, float, float], ...]],
        tracks: dict[int, tuple[Vec2, ...]] | None = None,
    ) -> Vec2 | None:
        if (
            tank.drone_type is not DroneType.TANK
            or tank.status is not DroneStatus.ACTIVE
            or not tank.shots_remaining
            or tank.next_fire_time is None
            or state.time + EPS < tank.next_fire_time
        ):
            return None
        enemies = [enemy for enemy in state.opponent_drones if enemy.status is DroneStatus.ACTIVE]
        rough = sorted(
            enemies,
            key=lambda enemy: (
                _distance(enemy.position, self.own_goal.center) - 6.0 * self.specs[enemy.drone_type].point_value,
                enemy.id,
            ),
        )[: self.MAX_TARGETS_TO_EVALUATE]
        if tracks is None:
            tracks = self._build_vehicle_tracks(state, predictions, round(self.MAX_FLIGHT_TIME / PHYSICS_DT))
        accepted: list[tuple[float, ShotEvaluation]] = []
        for target in rough:
            evaluation = self.evaluate_target(tank, target, state, predictions, tracks)
            if evaluation is None:
                continue
            urgent = self._urgent(target, state)
            hit_threshold = self.URGENT_HIT_THRESHOLD if urgent else self.NORMAL_HIT_THRESHOLD
            hard_threshold = self.URGENT_HARD_THRESHOLD if urgent else self.NORMAL_HARD_THRESHOLD
            if evaluation.hit_fraction + EPS < hit_threshold or evaluation.hard_evasion_hit_fraction + EPS < hard_threshold:
                continue
            if evaluation.friendly_first_fraction > EPS or evaluation.terrain_first_fraction > EPS:
                continue
            accepted.append((self._utility(target, evaluation, tank, state), evaluation))
        self.last_evaluations = tuple(evaluation for _, evaluation in accepted)
        if not accepted:
            return None
        _, selected = max(accepted, key=lambda item: (item[0], item[1].hit_fraction, item[1].hard_evasion_hit_fraction, -item[1].target_id))
        return selected.aim_direction


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
        self.weapon = game_info.weapon_spec
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
        self.fire_control = ConservativeFireControl(
            self.team,
            self.width,
            self.height,
            self.obstacles,
            self.specs,
            self.weapon,
            self.own_goal,
        )

    def _destination(self, drone: DroneSnapshot, state: GameState) -> Vec2:
        enemies = [enemy for enemy in state.opponent_drones if enemy.status is DroneStatus.ACTIVE]
        if drone.drone_type is DroneType.SCOUT and enemies:
            scouts = sorted(
                (item for item in state.own_drones if item.status is DroneStatus.ACTIVE and item.drone_type is DroneType.SCOUT),
                key=lambda item: item.id,
            )
            rank = next((index for index, item in enumerate(scouts) if item.id == drone.id), 0)
            enemy_transports = [enemy for enemy in enemies if enemy.drone_type is DroneType.TRANSPORT]
            if rank % 4 == 0 and enemy_transports:
                target = min(
                    enemy_transports,
                    key=lambda enemy: (_distance(enemy.position, self.own_goal.center), enemy.id),
                )
                lead = min(1.0, _distance(drone.position, target.position) / max(1.0, self.specs[DroneType.SCOUT].max_speed))
                return (target.position[0] + target.velocity[0] * lead, target.position[1] + target.velocity[1] * lead)
            transports = [item for item in state.own_drones if item.status is DroneStatus.ACTIVE and item.drone_type is DroneType.TRANSPORT]
            if rank % 4 == 1 and transports:
                transport = min(transports, key=lambda item: (_distance(drone.position, item.position), item.id))
                offset = ((rank % 5) - 2) * 1.1
                return (
                    transport.position[0] + 2.8 * self.direction,
                    min(self.height - 0.7, max(0.7, transport.position[1] + offset)),
                )
        if drone.drone_type is DroneType.TANK:
            transports = [item for item in state.own_drones if item.status is DroneStatus.ACTIVE and item.drone_type is DroneType.TRANSPORT]
            midfield = self.width * (0.46 if self.team is Team.A else 0.54)
            if transports:
                lead_transport = max(transports, key=lambda item: (self.direction * item.position[0], -item.id))
                desired_x = lead_transport.position[0] - self.direction * 5.5
                if self.direction * (desired_x - midfield) > 0.0:
                    return (desired_x, lead_transport.position[1])
            return (midfield, self.lanes.get(drone.id, self.height * 0.5))
        return _goal_target(self.goal, drone, self.lanes.get(drone.id, self.goal.center[1]))

    def _priority(self, drone: DroneSnapshot, state: GameState) -> float:
        remaining = _distance(drone.position, self.goal.center)
        time_left = max(1.0, 90.0 - state.time)
        urgency = remaining / max(0.1, self.specs[drone.drone_type].max_speed) / time_left
        value = self.specs[drone.drone_type].point_value
        class_bonus = 20.0 if drone.drone_type is DroneType.TRANSPORT else 10.0 if drone.drone_type is DroneType.TANK else 0.0
        emergency = 0.0
        for friend in state.own_drones:
            if friend.id == drone.id or friend.status is not DroneStatus.ACTIVE:
                continue
            dx, dy = friend.position[0] - drone.position[0], friend.position[1] - drone.position[1]
            separation = hypot(dx, dy)
            if separation >= 2.0:
                continue
            closing = ((friend.velocity[0] - drone.velocity[0]) * dx + (friend.velocity[1] - drone.velocity[1]) * dy) / max(EPS, separation)
            if closing < 0.0:
                emergency = max(emergency, (2.0 - separation) / 2.0)
        passage = 0.0
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                center, radius = obstacle.center, obstacle.radius
            else:
                center = ((obstacle.x_min + obstacle.x_max) * 0.5, (obstacle.y_min + obstacle.y_max) * 0.5)
                radius = hypot(obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min) * 0.5
            surface = _distance(drone.position, center) - radius
            if 0.0 < surface < 3.0:
                passage = max(passage, (3.0 - surface) / 3.0)
        aging = min(10.0, 2.0 * self.aging.get(drone.id, 0))
        return 1000.0 * emergency + 35.0 * passage + 100.0 * urgency + class_bonus + aging - 0.01 * drone.id + value

    def _steer(
        self,
        drone: DroneSnapshot,
        target: Vec2,
        *,
        hold: bool = False,
        timestamp: float | None = None,
        arrival_time: float | None = None,
    ) -> Vec2:
        spec = self.specs[drone.drone_type]
        dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
        distance = hypot(dx, dy)
        if hold or distance < 1.0e-6:
            desired_velocity = (0.0, 0.0)
            forward = (self.direction, 0.0)
        else:
            forward = (dx / distance, dy / distance)
            desired_speed = min(spec.max_speed, sqrt(max(0.0, 2.0 * spec.max_acceleration * distance)))
            if timestamp is not None and arrival_time is not None:
                remaining = max(CONTROLLER_PERIOD, arrival_time - timestamp)
                # Track the SIPP schedule, rather than racing to the next
                # waypoint and then braking after the reservation window.
                desired_speed = min(desired_speed, distance / remaining * 1.10)
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
            static_route = _static_route(self.graph, self.obstacles, drone.position, target)
            planning_target = _receding_goal(
                static_route,
                self.specs[drone.drone_type].max_speed,
                PLANNING_HORIZON,
            )
            planner = SippPlanner(
                self.graph,
                self.obstacles,
                reservations,
                state.time,
                speed=self.specs[drone.drone_type].max_speed,
                reservation_radius=TRACKING_RESERVATION_RADIUS,
            )
            plan = planner.plan(drone.position, planning_target)
            if plan is None:
                fallback = static_route
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
            prediction = self._predict_route(drone, plan, state.time, PREDICTION_HORIZON)
            self.predictions[drone.id] = prediction
            reservations.extend(
                reservation_segments(
                    _coarsen_trajectory(prediction),
                    radius=TRACKING_RESERVATION_RADIUS,
                )
            )
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
            target, hold, arrival_time = self._plan_target_details(plan, timestamp)
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
                timestamp=timestamp,
                arrival_time=arrival_time,
            )
            dynamic = advance_dynamics(dynamic, command, self.specs[drone.drone_type], 0.05)
            samples.append((timestamp + 0.05, dynamic.position[0], dynamic.position[1]))
        return tuple(samples)

    @staticmethod
    def _plan_target(plan: SippPlan, timestamp: float) -> tuple[Vec2, bool]:
        target, hold, _ = SwarmController._plan_target_details(plan, timestamp)
        return target, hold

    @staticmethod
    def _plan_target_details(plan: SippPlan, timestamp: float) -> tuple[Vec2, bool, float | None]:
        points = plan.points
        for index in range(1, len(points)):
            if timestamp <= points[index].time + EPS:
                previous = points[index - 1]
                if points[index].time - previous.time > MERGE_EPS and _distance(previous.position, points[index].position) > EPS:
                    if timestamp < points[index].time - MERGE_EPS:
                        return points[index].position, False, points[index].time
                return previous.position, True, None
        return points[-1].position, True, None

    def _emergency_guard(self, drone: DroneSnapshot, command: Vec2, state: GameState, priority: float) -> Vec2:
        for friend in state.own_drones:
            if friend.id == drone.id or friend.status is not DroneStatus.ACTIVE:
                continue
            if self._priority(friend, state) <= priority:
                continue
            own_prediction = self.predictions.get(drone.id)
            friend_prediction = self.predictions.get(friend.id)
            if own_prediction and friend_prediction:
                samples = min(
                    len(own_prediction),
                    len(friend_prediction),
                    round(EMERGENCY_GUARD_HORIZON / PHYSICS_DT) + 1,
                )
                predicted_miss = min(
                    hypot(
                        own_prediction[index][1] - friend_prediction[index][1],
                        own_prediction[index][2] - friend_prediction[index][2],
                    )
                    for index in range(samples)
                )
                if predicted_miss < 0.82:
                    spec = self.specs[drone.drone_type]
                    relative = (friend.position[0] - drone.position[0], friend.position[1] - drone.position[1])
                    away = hypot(relative[0], relative[1])
                    if away < EPS:
                        return clip_vector((-drone.velocity[0] * 3.0, -drone.velocity[1] * 3.0), spec.max_acceleration)
                    return clip_vector(
                        (
                            -relative[0] / away * spec.max_acceleration - drone.velocity[0] * 3.0,
                            -relative[1] / away * spec.max_acceleration - drone.velocity[1] * 3.0,
                        ),
                        spec.max_acceleration,
                    )
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
        for drone in active:
            plan = self.plans.get(drone.id)
            if plan is not None:
                self.predictions[drone.id] = self._predict_route(drone, plan, state.time, PREDICTION_HORIZON)
        self.fire_control.observe(state)
        vehicle_tracks = None
        if any(
            drone.drone_type is DroneType.TANK
            and drone.shots_remaining
            and drone.next_fire_time is not None
            and state.time + EPS >= drone.next_fire_time
            for drone in active
        ):
            vehicle_tracks = self.fire_control._build_vehicle_tracks(
                state,
                self.predictions,
                round(self.fire_control.MAX_FLIGHT_TIME / PHYSICS_DT),
            )
        ready_tanks = [
            drone
            for drone in active
            if drone.drone_type is DroneType.TANK
            and drone.shots_remaining
            and drone.next_fire_time is not None
            and state.time + EPS >= drone.next_fire_time
        ]
        fire_tank_id = None
        if ready_tanks:
            ready_tanks.sort(key=lambda drone: drone.id)
            fire_tank_id = ready_tanks[round(state.time / CONTROLLER_PERIOD) % len(ready_tanks)].id
        actions = {}
        for drone in active:
            plan = self.plans.get(drone.id)
            if plan is None:
                command = self._steer(drone, self._destination(drone, state))
            else:
                target, hold, arrival_time = self._plan_target_details(plan, state.time)
                command = self._steer(
                    drone,
                    target,
                    hold=hold,
                    timestamp=state.time,
                    arrival_time=arrival_time,
                )
            command = self._emergency_guard(drone, command, state, self._priority(drone, state))
            if drone.drone_type is DroneType.TANK and drone.id == fire_tank_id:
                fire_direction = self.fire_control.choose_fire(drone, state, self.predictions, vehicle_tracks)
                if fire_direction is not None:
                    actions[drone.id] = {"acceleration": command, "fire_direction": fire_direction}
                    continue
            actions[drone.id] = command
        return actions
