from math import hypot

from swarmbench.api import DRONE_COUNT_RANGES, DRONE_RADIUS, DRONE_SPECS, DroneType, GoalZone, Team
from swarmbench.engine.arena import (
    SPAWN_MIN_SPACING,
    generate_scenario,
    point_blocked,
    scenario_is_traversable,
    validate_scenario,
)


def test_generation_is_deterministic_and_varies_by_seed() -> None:
    assert generate_scenario(12345) == generate_scenario(12345)
    assert generate_scenario(12345) != generate_scenario(12346)


def test_generated_scenario_has_valid_goals_spawns_and_obstacles() -> None:
    scenario = generate_scenario(7)
    validate_scenario(scenario)
    assert 8 <= len(scenario.obstacles) <= 15
    assert scenario.goal_for_a.x_max == scenario.width
    assert scenario.goal_for_b.x_min == 0.0
    assert scenario.goal_for_b == GoalZone(0.0, 3.0, scenario.goal_for_a.y_min, scenario.goal_for_a.y_max)
    for team in Team:
        drones = scenario.team_drones(team)
        for kind, (low, high) in DRONE_COUNT_RANGES.items():
            assert low <= sum(drone.drone_type is kind for drone in drones) <= high
        for index, drone in enumerate(drones):
            assert not point_blocked(drone.position, scenario.obstacles, DRONE_RADIUS)
            assert all(
                hypot(drone.position[0] - other.position[0], drone.position[1] - other.position[1]) >= SPAWN_MIN_SPACING
                for other in drones[index + 1 :]
            )
    team_a = scenario.team_drones(Team.A)
    team_b = scenario.team_drones(Team.B)
    assert len(team_a) == len(team_b)
    assert all(
        left.drone_type is right.drone_type
        and right.position == (scenario.width - left.position[0], left.position[1])
        for left, right in zip(team_a, team_b, strict=True)
    )
    for kind, spec in scenario.drone_specs:
        nominal = DRONE_SPECS[kind]
        assert nominal.max_speed * 0.9 <= spec.max_speed <= nominal.max_speed * 1.1
        assert nominal.max_acceleration * 0.9 <= spec.max_acceleration <= nominal.max_acceleration * 1.1
        assert nominal.max_jerk * 0.9 <= spec.max_jerk <= nominal.max_jerk * 1.1


def test_hundreds_of_seeded_arenas_are_valid_and_reachable() -> None:
    for seed in range(200):
        scenario = generate_scenario(seed)
        validate_scenario(scenario, check_traversability=False)
        assert scenario_is_traversable(scenario)

