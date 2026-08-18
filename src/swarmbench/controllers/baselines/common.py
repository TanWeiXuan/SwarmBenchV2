"""Small deterministic steering helpers shared by the readable baselines."""

from __future__ import annotations

from math import hypot, sqrt

from swarmbench import CircleObstacle, DroneSnapshot, DroneSpec, DroneStatus, DroneType, GameState, GoalZone, RectangleObstacle
from swarmbench.engine.collisions import swept_obstacle_contact, swept_points_contact


def goal_target(goal: GoalZone, drone: DroneSnapshot) -> tuple[float, float]:
    y = min(goal.y_max - 0.75, max(goal.y_min + 0.75, drone.position[1]))
    return (goal.center[0], y)


def distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def _obstacle_center_radius(obstacle: CircleObstacle | RectangleObstacle) -> tuple[tuple[float, float], float]:
    if isinstance(obstacle, CircleObstacle):
        return obstacle.center, obstacle.radius
    center = ((obstacle.x_min + obstacle.x_max) / 2, (obstacle.y_min + obstacle.y_max) / 2)
    radius = hypot(obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min) / 2
    return center, radius


def steer(
    drone: DroneSnapshot,
    target: tuple[float, float],
    spec: DroneSpec,
    obstacles: tuple[CircleObstacle | RectangleObstacle, ...],
    *,
    repulsion: float = 1.0,
) -> tuple[float, float]:
    """Velocity tracking plus short-range deterministic obstacle steering."""
    dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
    remaining = hypot(dx, dy)
    if remaining < 1e-9:
        desired_velocity = (0.0, 0.0)
    else:
        desired_speed = min(spec.max_speed, (2.0 * spec.max_acceleration * remaining) ** 0.5)
        desired_velocity = (dx / remaining * desired_speed, dy / remaining * desired_speed)

    ax = 2.2 * (desired_velocity[0] - drone.velocity[0])
    ay = 2.2 * (desired_velocity[1] - drone.velocity[1])
    if remaining > 0:
        forward = (dx / remaining, dy / remaining)
        for obstacle in obstacles:
            center, radius = _obstacle_center_radius(obstacle)
            ox, oy = center[0] - drone.position[0], center[1] - drone.position[1]
            projection = ox * forward[0] + oy * forward[1]
            lateral = ox * (-forward[1]) + oy * forward[0]
            safe_radius = radius + 1.1
            if -0.5 < projection < 8.0 and abs(lateral) < safe_radius:
                side = -1.0 if lateral > 0 else 1.0
                if abs(lateral) < 1e-9:
                    side = 1.0 if drone.id % 2 == 0 else -1.0
                strength = repulsion * spec.max_acceleration * (1.0 - max(0.0, projection) / 8.0)
                ax += -forward[1] * side * strength
                ay += forward[0] * side * strength
            center_distance = hypot(ox, oy)
            surface_distance = center_distance - radius
            if 0 < surface_distance < 3.0:
                strength = repulsion * spec.max_acceleration * (3.0 - surface_distance) / 3.0
                ax -= ox / center_distance * strength
                ay -= oy / center_distance * strength
    return (ax, ay)


def intercept_point(shooter: DroneSnapshot, target: DroneSnapshot, projectile_speed: float = 20.0) -> tuple[float, float]:
    """Constant-velocity lead point for a constant-speed projectile."""
    rx = target.position[0] - shooter.position[0]
    ry = target.position[1] - shooter.position[1]
    vx, vy = target.velocity
    a = vx * vx + vy * vy - projectile_speed * projectile_speed
    b = 2.0 * (rx * vx + ry * vy)
    c = rx * rx + ry * ry
    times: list[float] = []
    if abs(a) < 1e-12:
        if abs(b) > 1e-12:
            times.append(-c / b)
    else:
        discriminant = b * b - 4 * a * c
        if discriminant >= 0:
            root = sqrt(discriminant)
            times.extend(((-b - root) / (2 * a), (-b + root) / (2 * a)))
    positive = [value for value in times if value > 0]
    time = min(positive, default=0.0)
    return (target.position[0] + vx * time, target.position[1] + vy * time)


def _safety_acceleration(
    drone: DroneSnapshot,
    acceleration: tuple[float, float],
    state: GameState,
    spec: DroneSpec,
) -> tuple[float, float]:
    ax, ay = acceleration
    for friend in state.own_drones:
        if friend.id == drone.id or friend.status is not DroneStatus.ACTIVE:
            continue
        dx = drone.position[0] - friend.position[0]
        dy = drone.position[1] - friend.position[1]
        separation = hypot(dx, dy)
        if 0 < separation < 2.0:
            strength = spec.max_acceleration * 1.5 * (2.0 - separation) / 2.0
            ax += dx / separation * strength
            ay += dy / separation * strength
    for projectile in state.projectiles:
        if projectile.source_drone_id == drone.id:
            continue
        rx = projectile.position[0] - drone.position[0]
        ry = projectile.position[1] - drone.position[1]
        rvx = projectile.velocity[0] - drone.velocity[0]
        rvy = projectile.velocity[1] - drone.velocity[1]
        speed_squared = rvx * rvx + rvy * rvy
        if speed_squared == 0:
            continue
        closest_time = max(0.0, min(1.2, -(rx * rvx + ry * rvy) / speed_squared))
        closest_x = rx + rvx * closest_time
        closest_y = ry + rvy * closest_time
        closest = hypot(closest_x, closest_y)
        if 0 < closest < 1.1:
            side = 1.0 if drone.id % 2 == 0 else -1.0
            length = hypot(rvx, rvy)
            ax += -rvy / length * side * spec.max_acceleration * 1.5
            ay += rvx / length * side * spec.max_acceleration * 1.5
    return (ax, ay)


def _clear_shot(
    shooter: DroneSnapshot,
    aim: tuple[float, float],
    state: GameState,
    obstacles: tuple[CircleObstacle | RectangleObstacle, ...],
) -> bool:
    if any(swept_obstacle_contact(shooter.position, aim, obstacle, 0.0) is not None for obstacle in obstacles):
        return False
    return not any(
        friend.id != shooter.id
        and friend.status is DroneStatus.ACTIVE
        and swept_points_contact(shooter.position, aim, friend.position, friend.position, 0.30) is not None
        for friend in state.own_drones
    )


def finalize_action(
    drone: DroneSnapshot,
    acceleration: tuple[float, float],
    state: GameState,
    spec: DroneSpec,
    obstacles: tuple[CircleObstacle | RectangleObstacle, ...],
    fire_target: DroneSnapshot | None = None,
):
    acceleration = _safety_acceleration(drone, acceleration, state, spec)
    if (
        drone.drone_type is DroneType.TANK
        and fire_target is not None
        and drone.shots_remaining
        and drone.next_fire_time is not None
        and state.time + 1e-12 >= drone.next_fire_time
    ):
        aim = intercept_point(drone, fire_target)
        if _clear_shot(drone, aim, state, obstacles):
            return {
                "acceleration": acceleration,
                "fire_direction": (aim[0] - drone.position[0], aim[1] - drone.position[1]),
            }
    return acceleration


def preferred_fire_target(drone: DroneSnapshot, state: GameState, specs: dict[DroneType, DroneSpec]) -> DroneSnapshot | None:
    enemies = [enemy for enemy in state.opponent_drones if enemy.status is DroneStatus.ACTIVE]
    return min(
        enemies,
        key=lambda enemy: (
            distance(drone.position, enemy.position) - 5.0 * specs[enemy.drone_type].point_value,
            enemy.id,
        ),
        default=None,
    )

