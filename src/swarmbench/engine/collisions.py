"""Continuous collision helpers over one piecewise-linear physics step."""

from __future__ import annotations

from math import sqrt

from swarmbench.api import CircleObstacle, GoalZone, Obstacle, RectangleObstacle, Vec2


def _lerp(start: Vec2, end: Vec2, t: float) -> Vec2:
    return (start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)


def swept_points_contact(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2, radius: float) -> float | None:
    """Earliest normalized time two swept points are within ``radius``."""
    rx, ry = a0[0] - b0[0], a0[1] - b0[1]
    vx, vy = (a1[0] - a0[0]) - (b1[0] - b0[0]), (a1[1] - a0[1]) - (b1[1] - b0[1])
    c = rx * rx + ry * ry - radius * radius
    if c <= 0:
        return 0.0
    a = vx * vx + vy * vy
    if a == 0:
        return None
    b = 2 * (rx * vx + ry * vy)
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    contact = (-b - sqrt(discriminant)) / (2 * a)
    return contact if 0.0 <= contact <= 1.0 else None


def swept_circle_contact(start: Vec2, end: Vec2, center: Vec2, radius: float) -> float | None:
    return swept_points_contact(start, end, center, center, radius)


def swept_aabb_contact(start: Vec2, end: Vec2, bounds: tuple[float, float, float, float]) -> float | None:
    x_min, x_max, y_min, y_max = bounds
    enter, leave = 0.0, 1.0
    for origin, delta, low, high in (
        (start[0], end[0] - start[0], x_min, x_max),
        (start[1], end[1] - start[1], y_min, y_max),
    ):
        if delta == 0:
            if origin < low or origin > high:
                return None
            continue
        t0, t1 = (low - origin) / delta, (high - origin) / delta
        if t0 > t1:
            t0, t1 = t1, t0
        enter, leave = max(enter, t0), min(leave, t1)
        if enter > leave:
            return None
    return enter if 0.0 <= enter <= 1.0 else None


def swept_obstacle_contact(start: Vec2, end: Vec2, obstacle: Obstacle, drone_radius: float) -> float | None:
    if isinstance(obstacle, CircleObstacle):
        return swept_circle_contact(start, end, obstacle.center, obstacle.radius + drone_radius)
    return swept_aabb_contact(
        start,
        end,
        (
            obstacle.x_min - drone_radius,
            obstacle.x_max + drone_radius,
            obstacle.y_min - drone_radius,
            obstacle.y_max + drone_radius,
        ),
    )


def swept_goal_contact(start: Vec2, end: Vec2, goal: GoalZone) -> float | None:
    return swept_aabb_contact(start, end, (goal.x_min, goal.x_max, goal.y_min, goal.y_max))


def swept_arena_exit(start: Vec2, end: Vec2, width: float, height: float) -> float | None:
    """Return the first normalized time a segment leaves the closed arena."""
    candidates: list[float] = []
    for origin, destination, low, high in (
        (start[0], end[0], 0.0, width),
        (start[1], end[1], 0.0, height),
    ):
        delta = destination - origin
        if delta < 0 and destination < low:
            candidates.append((low - origin) / delta)
        elif delta > 0 and destination > high:
            candidates.append((high - origin) / delta)
    valid = [value for value in candidates if 0.0 <= value <= 1.0]
    return min(valid) if valid else None


def contact_position(start: Vec2, end: Vec2, time: float) -> Vec2:
    return _lerp(start, end, time)

