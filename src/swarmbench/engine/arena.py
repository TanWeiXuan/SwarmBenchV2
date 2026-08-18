"""Seeded v2 symmetric arena generation and deterministic validation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot
from random import Random

from swarmbench.api import (
    ARENA_HEIGHT,
    ARENA_WIDTH,
    DRONE_COUNT_RANGES,
    DRONE_RADIUS,
    DRONE_SPECS,
    DYNAMICS_VARIATION,
    CircleObstacle,
    DroneSnapshot,
    DroneSpec,
    DroneType,
    GoalZone,
    Obstacle,
    RectangleObstacle,
    TANK_WEAPON_SPEC,
    Team,
    Vec2,
)
from swarmbench.version import SCENARIO_GENERATOR_VERSION

GRID_RESOLUTION = 0.5
PLANNING_CLEARANCE = 0.35
SPAWN_MIN_SPACING = 0.8
MAX_GENERATION_ATTEMPTS = 100
MIN_OBSTACLES = 8
MAX_OBSTACLES = 15


@dataclass(frozen=True, slots=True)
class Scenario:
    seed: int
    generator_version: int
    width: float
    height: float
    goal_for_a: GoalZone
    goal_for_b: GoalZone
    obstacles: tuple[Obstacle, ...]
    drones: tuple[DroneSnapshot, ...]
    drone_specs: tuple[tuple[DroneType, DroneSpec], ...] = field(
        default_factory=lambda: tuple(DRONE_SPECS.items())
    )

    def target_goal(self, team: Team) -> GoalZone:
        return self.goal_for_a if team is Team.A else self.goal_for_b

    def own_goal(self, team: Team) -> GoalZone:
        return self.goal_for_b if team is Team.A else self.goal_for_a

    def team_drones(self, team: Team) -> tuple[DroneSnapshot, ...]:
        return tuple(drone for drone in self.drones if drone.team is team)

    def spec_for(self, drone_type: DroneType) -> DroneSpec:
        return dict(self.drone_specs)[drone_type]


def _point_blocked(point: Vec2, obstacle: Obstacle, clearance: float) -> bool:
    x, y = point
    if isinstance(obstacle, CircleObstacle):
        return hypot(x - obstacle.center[0], y - obstacle.center[1]) <= obstacle.radius + clearance
    return (
        obstacle.x_min - clearance <= x <= obstacle.x_max + clearance
        and obstacle.y_min - clearance <= y <= obstacle.y_max + clearance
    )


def point_blocked(point: Vec2, obstacles: tuple[Obstacle, ...], clearance: float = DRONE_RADIUS) -> bool:
    return any(_point_blocked(point, obstacle, clearance) for obstacle in obstacles)


def _obstacle_bounds(obstacle: Obstacle, padding: float = 0.0) -> tuple[float, float, float, float]:
    if isinstance(obstacle, CircleObstacle):
        x, y = obstacle.center
        radius = obstacle.radius + padding
        return (x - radius, x + radius, y - radius, y + radius)
    return (
        obstacle.x_min - padding,
        obstacle.x_max + padding,
        obstacle.y_min - padding,
        obstacle.y_max + padding,
    )


def _obstacles_overlap(left: Obstacle, right: Obstacle, padding: float = 0.5) -> bool:
    lx0, lx1, ly0, ly1 = _obstacle_bounds(left, padding)
    rx0, rx1, ry0, ry1 = _obstacle_bounds(right, padding)
    return lx0 <= rx1 and rx0 <= lx1 and ly0 <= ry1 and ry0 <= ly1


def mirror_obstacle(obstacle: Obstacle, width: float = ARENA_WIDTH) -> Obstacle:
    if isinstance(obstacle, CircleObstacle):
        return CircleObstacle((width - obstacle.center[0], obstacle.center[1]), obstacle.radius)
    return RectangleObstacle(width - obstacle.x_max, width - obstacle.x_min, obstacle.y_min, obstacle.y_max)


def _same_obstacle(left: Obstacle, right: Obstacle, tolerance: float = 1e-9) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, CircleObstacle) and isinstance(right, CircleObstacle):
        values = (*left.center, left.radius, *right.center, right.radius)
        return all(abs(values[index] - values[index + 3]) <= tolerance for index in range(3))
    if isinstance(left, RectangleObstacle) and isinstance(right, RectangleObstacle):
        left_values = (left.x_min, left.x_max, left.y_min, left.y_max)
        right_values = (right.x_min, right.x_max, right.y_min, right.y_max)
        return all(abs(a - b) <= tolerance for a, b in zip(left_values, right_values, strict=True))
    return False


def _random_left_obstacle(rng: Random) -> Obstacle:
    if rng.random() < 0.5:
        radius = rng.uniform(1.5, 3.5)
        return CircleObstacle(
            (rng.uniform(20.0 + radius, 49.5 - radius), rng.uniform(2.0 + radius, 58.0 - radius)),
            radius,
        )
    width = rng.uniform(2.0, 7.0)
    height = rng.uniform(2.0, 7.0)
    center_x = rng.uniform(20.0 + width / 2, 49.5 - width / 2)
    center_y = rng.uniform(2.0 + height / 2, 58.0 - height / 2)
    return RectangleObstacle(
        center_x - width / 2,
        center_x + width / 2,
        center_y - height / 2,
        center_y + height / 2,
    )


def _random_center_obstacle(rng: Random) -> Obstacle:
    if rng.random() < 0.5:
        radius = rng.uniform(1.5, 3.5)
        return CircleObstacle((ARENA_WIDTH / 2, rng.uniform(2.0 + radius, 58.0 - radius)), radius)
    width = rng.uniform(2.0, 7.0)
    height = rng.uniform(2.0, 7.0)
    center_y = rng.uniform(2.0 + height / 2, 58.0 - height / 2)
    return RectangleObstacle(50.0 - width / 2, 50.0 + width / 2, center_y - height / 2, center_y + height / 2)


def _generate_obstacles(rng: Random) -> tuple[Obstacle, ...]:
    target_count = rng.randint(MIN_OBSTACLES, MAX_OBSTACLES)
    obstacles: list[Obstacle] = []
    draws = 0
    for _ in range(target_count // 2):
        while draws < 1_000:
            draws += 1
            left = _random_left_obstacle(rng)
            right = mirror_obstacle(left)
            if not any(_obstacles_overlap(candidate, existing) for candidate in (left, right) for existing in obstacles):
                obstacles.extend((left, right))
                break
        else:
            raise RuntimeError("could not place requested mirrored obstacles")
    if target_count % 2:
        while draws < 1_000:
            draws += 1
            center = _random_center_obstacle(rng)
            if not any(_obstacles_overlap(center, existing) for existing in obstacles):
                obstacles.append(center)
                break
        else:
            raise RuntimeError("could not place centerline obstacle")
    return tuple(obstacles)


def _sample_specs(rng: Random) -> tuple[tuple[DroneType, DroneSpec], ...]:
    sampled = []
    for drone_type, nominal in DRONE_SPECS.items():
        def vary(value: float) -> float:
            return round(rng.uniform(value * (1 - DYNAMICS_VARIATION), value * (1 + DYNAMICS_VARIATION)), 9)

        sampled.append(
            (drone_type, DroneSpec(vary(nominal.max_speed), vary(nominal.max_acceleration), vary(nominal.max_jerk), nominal.point_value))
        )
    return tuple(sampled)


def _sample_counts(rng: Random) -> dict[DroneType, int]:
    return {drone_type: rng.randint(low, high) for drone_type, (low, high) in DRONE_COUNT_RANGES.items()}


def _generate_mirrored_spawns(
    rng: Random,
    obstacles: tuple[Obstacle, ...],
    counts: dict[DroneType, int],
) -> tuple[DroneSnapshot, ...]:
    team_size = sum(counts.values())
    positions: list[Vec2] = []
    for _ in range(team_size):
        for _attempt in range(1_000):
            point = (rng.uniform(5.0, 14.0), rng.uniform(3.0, 57.0))
            if point_blocked(point, obstacles, DRONE_RADIUS):
                continue
            if all(hypot(point[0] - other[0], point[1] - other[1]) >= SPAWN_MIN_SPACING for other in positions):
                positions.append(point)
                break
        else:
            raise RuntimeError("could not place team spawn")

    types = [drone_type for drone_type in DroneType for _ in range(counts[drone_type])]
    rng.shuffle(types)
    def initial_snapshot(drone_id: int, team: Team, drone_type: DroneType, point: Vec2) -> DroneSnapshot:
        if drone_type is DroneType.TANK:
            return DroneSnapshot(
                drone_id,
                team,
                drone_type,
                point,
                shots_remaining=TANK_WEAPON_SPEC.magazine_size,
                next_fire_time=TANK_WEAPON_SPEC.initial_lockout,
            )
        return DroneSnapshot(drone_id, team, drone_type, point)

    team_a = tuple(initial_snapshot(index, Team.A, types[index], point) for index, point in enumerate(positions))
    team_b = tuple(
        initial_snapshot(team_size + index, Team.B, types[index], (ARENA_WIDTH - point[0], point[1]))
        for index, point in enumerate(positions)
    )
    return team_a + team_b


def _reachable_from_goal(scenario: Scenario, goal: GoalZone) -> set[tuple[int, int]]:
    resolution = GRID_RESOLUTION
    nx = round(scenario.width / resolution) + 1
    ny = round(scenario.height / resolution) + 1
    clearance = DRONE_RADIUS + PLANNING_CLEARANCE

    def open_cell(cell: tuple[int, int]) -> bool:
        i, j = cell
        return not point_blocked((i * resolution, j * resolution), scenario.obstacles, clearance)

    starts = [
        (i, j)
        for i in range(nx)
        for j in range(ny)
        if goal.contains((i * resolution, j * resolution)) and open_cell((i, j))
    ]
    reached = set(starts)
    queue = deque(starts)
    while queue:
        i, j = queue.popleft()
        for neighbor in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            ni, nj = neighbor
            if 0 <= ni < nx and 0 <= nj < ny and neighbor not in reached and open_cell(neighbor):
                reached.add(neighbor)
                queue.append(neighbor)
    return reached


def scenario_is_traversable(scenario: Scenario) -> bool:
    reachable_a = _reachable_from_goal(scenario, scenario.goal_for_a)
    reachable_b = _reachable_from_goal(scenario, scenario.goal_for_b)
    for drone in scenario.drones:
        cell = (round(drone.position[0] / GRID_RESOLUTION), round(drone.position[1] / GRID_RESOLUTION))
        if cell not in (reachable_a if drone.team is Team.A else reachable_b):
            return False
    return True


def validate_scenario(scenario: Scenario, *, check_traversability: bool = True) -> None:
    if scenario.generator_version != SCENARIO_GENERATOR_VERSION:
        raise ValueError("unsupported scenario generator version")
    if not MIN_OBSTACLES <= len(scenario.obstacles) <= MAX_OBSTACLES:
        raise ValueError("scenario must contain 8 to 15 obstacles")
    if scenario.goal_for_b != GoalZone(0.0, 3.0, scenario.goal_for_a.y_min, scenario.goal_for_a.y_max):
        raise ValueError("goals are not x-mirrored")
    specs = dict(scenario.drone_specs)
    if set(specs) != set(DroneType):
        raise ValueError("scenario must define every vehicle class")
    for drone_type, spec in specs.items():
        nominal = DRONE_SPECS[drone_type]
        for sampled, base in ((spec.max_speed, nominal.max_speed), (spec.max_acceleration, nominal.max_acceleration), (spec.max_jerk, nominal.max_jerk)):
            if not base * 0.9 <= sampled <= base * 1.1:
                raise ValueError("sampled dynamics lie outside the documented range")
        if spec.point_value != nominal.point_value:
            raise ValueError("point value must not vary")

    team_a = scenario.team_drones(Team.A)
    team_b = scenario.team_drones(Team.B)
    if len(team_a) != len(team_b):
        raise ValueError("teams must have equal size")
    for drone_type, (low, high) in DRONE_COUNT_RANGES.items():
        count_a = sum(drone.drone_type is drone_type for drone in team_a)
        count_b = sum(drone.drone_type is drone_type for drone in team_b)
        if count_a != count_b or not low <= count_a <= high:
            raise ValueError("vehicle count is outside the documented range")
    for left, right in zip(team_a, team_b, strict=True):
        if left.drone_type is not right.drone_type or right.position != (scenario.width - left.position[0], left.position[1]):
            raise ValueError("spawns are not x-mirrored")
    for team in Team:
        drones = scenario.team_drones(team)
        for index, drone in enumerate(drones):
            x, y = drone.position
            if not (DRONE_RADIUS <= x <= scenario.width - DRONE_RADIUS and DRONE_RADIUS <= y <= scenario.height - DRONE_RADIUS):
                raise ValueError("spawn lies outside arena")
            if point_blocked(drone.position, scenario.obstacles, DRONE_RADIUS):
                raise ValueError("spawn overlaps obstacle")
            if any(hypot(x - other.position[0], y - other.position[1]) < SPAWN_MIN_SPACING for other in drones[index + 1 :]):
                raise ValueError("spawn spacing is too small")
    for obstacle in scenario.obstacles:
        x0, x1, y0, y1 = _obstacle_bounds(obstacle)
        if x0 < 18.0 or x1 > 82.0 or y0 < 0.0 or y1 > scenario.height:
            raise ValueError("obstacle overlaps a protected region")
        if not any(_same_obstacle(mirror_obstacle(obstacle, scenario.width), candidate) for candidate in scenario.obstacles):
            raise ValueError("obstacles are not x-mirrored")
    if any(_obstacles_overlap(left, right, 0.0) for index, left in enumerate(scenario.obstacles) for right in scenario.obstacles[index + 1 :]):
        raise ValueError("obstacles overlap")
    if check_traversability and not scenario_is_traversable(scenario):
        raise ValueError("scenario is not traversable")


def generate_scenario(seed: int, generator_version: int = SCENARIO_GENERATOR_VERSION) -> Scenario:
    if generator_version != SCENARIO_GENERATOR_VERSION:
        raise ValueError(f"unsupported generator version: {generator_version}")
    rng = Random(seed)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        goal_center = rng.uniform(10.0, 50.0)
        goal_for_a = GoalZone(97.0, 100.0, goal_center - 7.0, goal_center + 7.0)
        goal_for_b = GoalZone(0.0, 3.0, goal_center - 7.0, goal_center + 7.0)
        specs = _sample_specs(rng)
        counts = _sample_counts(rng)
        try:
            obstacles = _generate_obstacles(rng)
            drones = _generate_mirrored_spawns(rng, obstacles, counts)
        except RuntimeError:
            continue
        scenario = Scenario(seed, generator_version, ARENA_WIDTH, ARENA_HEIGHT, goal_for_a, goal_for_b, obstacles, drones, specs)
        if scenario_is_traversable(scenario):
            return scenario
    raise RuntimeError(f"could not generate traversable arena after {MAX_GENERATION_ATTEMPTS} attempts")
