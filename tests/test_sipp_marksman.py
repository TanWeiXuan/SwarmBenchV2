from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from swarmbench import CircleObstacle, DroneSnapshot, DroneType, GameState, GoalZone, RectangleObstacle, Team
from swarmbench.api import DRONE_SPECS, TANK_WEAPON_SPEC

MODULE_PATH = Path(__file__).parents[1] / "submissions" / "TanWeiXuan" / "sipp_marksman_v1.py"
SPEC = importlib.util.spec_from_file_location("sipp_marksman_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
sipp = importlib.util.module_from_spec(SPEC)
sys.modules["sipp_marksman_v1"] = sipp
SPEC.loader.exec_module(sipp)


def graph(points, edges):
    adjacency = [[] for _ in points]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    return sipp.SpatialGraph(points, adjacency)


def planner(graph_value, reservations=(), *, now=0.0, horizon=10.0, speed=5.0):
    return sipp.SippPlanner(graph_value, (), reservations, now, horizon, speed)


def test_forbidden_interval_solver_handles_stationary_passing_tangent_and_miss():
    stationary = sipp.reservation_segments(((2.0, 0.0, 0.0), (5.0, 0.0, 0.0)))[0]
    assert sipp.forbidden_interval_at_point((0.0, 0.0), stationary) == pytest.approx((2.0, 5.0))

    passing = sipp.reservation_segments(((0.0, -2.0, 0.0), (4.0, 2.0, 0.0)))[0]
    interval = sipp.forbidden_interval_at_point((0.0, 0.0), passing)
    assert interval is not None
    assert interval[0] < 2.0 < interval[1]

    tangent = sipp.reservation_segments(((0.0, -1.0, 0.0), (2.0, 1.0, 0.0)))[0]
    tangent_interval = sipp.forbidden_interval_at_point((0.0, 0.0), tangent, radius=0.0)
    assert tangent_interval == pytest.approx((1.0, 1.0))

    miss = sipp.reservation_segments(((0.0, 2.0, 0.0), (2.0, 2.0, 0.0)))[0]
    assert sipp.forbidden_interval_at_point((1.0, 0.0), miss) is None


def test_safe_intervals_merge_overlapping_reservations_and_clip_horizon():
    first = sipp.reservation_segments(((2.0, 0.0, 0.0), (3.0, 0.0, 0.0)))[0]
    second = sipp.reservation_segments(((2.9, 0.0, 0.0), (4.0, 0.0, 0.0)))[0]
    intervals = sipp.safe_intervals((0.0, 0.0), (first, second), 0.0, 5.0)
    assert len(intervals) == 2
    assert intervals[0] == pytest.approx((0.0, 2.0))
    assert intervals[1] == pytest.approx((4.0, 5.0))


def test_empty_sipp_uses_direct_shortest_path():
    road = graph(((0.0, 0.0), (2.0, 0.0)), ((0, 1),))
    result = planner(road).plan((0.0, 0.0), (2.0, 0.0))
    assert result is not None
    assert [point.position for point in result.points] == [(0.0, 0.0), (2.0, 0.0)]
    assert result.arrival_time == pytest.approx(2.0 / 4.0 + 0.35)


def test_sipp_waits_for_a_reserved_destination_interval():
    road = graph(((0.0, 0.0), (1.0, 0.0)), ((0, 1),))
    blocked = sipp.reservation_segments(((2.0, 1.0, 0.0), (5.0, 1.0, 0.0)))
    result = planner(road, blocked, now=2.0, horizon=8.0, speed=1.0).plan((0.0, 0.0), (1.0, 0.0))
    assert result is not None
    assert result.points[-1].time >= 5.0
    assert result.points[-2].position == (0.0, 0.0)
    assert result.points[-2].time >= 4.8


def test_edge_closest_approach_rejects_head_on_swap_and_allows_late_passage():
    reservation = sipp.reservation_segments(((0.0, 10.0, 0.0), (2.0, 0.0, 0.0)))[0]
    assert sipp.edge_conflicts_reservation((0.0, 0.0), (10.0, 0.0), 0.0, 2.0, reservation)
    assert not sipp.edge_conflicts_reservation((0.0, 0.0), (10.0, 0.0), 3.0, 5.0, reservation)


def test_sipp_uses_detour_for_perpendicular_crossing():
    road = graph(
        ((-5.0, 0.0), (5.0, 0.0), (-5.0, 3.0), (5.0, 3.0)),
        ((0, 1), (0, 2), (2, 3), (3, 1)),
    )
    reservation = sipp.reservation_segments(((0.0, 0.0, -5.0), (2.85, 0.0, 5.0)))
    result = planner(road, reservation, horizon=8.0).plan((-5.0, 0.0), (5.0, 0.0))
    assert result is not None
    assert len(result.points) >= 2
    assert result.points[-1].time > 2.85
    assert all(
        not sipp.edge_conflicts_reservation(
            left.position,
            right.position,
            left.time,
            right.time,
            reservation[0],
        )
        for left, right in zip(result.points, result.points[1:])
        if right.position != left.position
    )


def test_sipp_never_uses_a_fast_catch_up_edge_through_a_slower_reservation():
    road = graph(
        ((0.0, 0.0), (10.0, 0.0), (0.0, 3.0), (10.0, 3.0)),
        ((0, 1), (0, 2), (2, 3), (3, 1)),
    )
    reservation = sipp.reservation_segments(((0.0, 1.0, 0.0), (3.0, 10.0, 0.0)))
    result = planner(road, reservation, horizon=10.0).plan((0.0, 0.0), (10.0, 0.0))
    assert result is not None
    assert result.points[-1].time > 3.0


def test_planning_is_deterministic():
    road = graph(
        ((0.0, 0.0), (10.0, 0.0), (0.0, 2.0), (10.0, 2.0)),
        ((0, 1), (0, 2), (2, 3), (3, 1)),
    )
    reservation = sipp.reservation_segments(((5.0, -1.0, 1.0), (5.0, 3.0, 3.0)))
    first = planner(road, reservation, horizon=8.0).plan((0.0, 0.0), (10.0, 0.0))
    second = planner(road, reservation, horizon=8.0).plan((0.0, 0.0), (10.0, 0.0))
    assert first == second


def make_fire_control(obstacles=()):
    return sipp.ConservativeFireControl(
        Team.A,
        100.0,
        60.0,
        tuple(obstacles),
        dict(DRONE_SPECS),
        TANK_WEAPON_SPEC,
        GoalZone(0.0, 3.0, 20.0, 40.0),
    )


def make_tank(*, time=5.0, next_fire_time=5.0, shots=5):
    return DroneSnapshot(
        0,
        Team.A,
        DroneType.TANK,
        (10.0, 30.0),
        shots_remaining=shots,
        next_fire_time=next_fire_time,
    )


def test_fire_legality_gate_enforces_lockout_cooldown_and_ammo():
    fire = make_fire_control()
    target = DroneSnapshot(20, Team.B, DroneType.TRANSPORT, (20.0, 30.0))
    tank = make_tank(time=4.9)
    early = GameState(4.9, (tank,), (target,), 0, 0)
    assert fire.choose_fire(tank, early, {}) is None

    legal = GameState(5.0, (make_tank(),), (target,), 0, 0)
    assert fire.choose_fire(legal.own_drones[0], legal, {}) is not None

    cooldown_tank = make_tank(time=5.0, next_fire_time=9.0)
    cooldown = GameState(5.0, (cooldown_tank,), (target,), 0, 0)
    assert fire.choose_fire(cooldown_tank, cooldown, {}) is None

    empty_tank = make_tank(shots=0)
    empty = GameState(9.0, (empty_tank,), (target,), 0, 0)
    assert fire.choose_fire(empty_tank, empty, {}) is None


def test_stationary_short_range_transport_is_high_confidence():
    fire = make_fire_control()
    tank = make_tank()
    target = DroneSnapshot(20, Team.B, DroneType.TRANSPORT, (20.0, 30.0))
    state = GameState(5.0, (tank,), (target,), 0, 0)
    evaluation = fire.evaluate_target(tank, target, state, {})
    assert evaluation is not None
    assert evaluation.hit_fraction >= 0.90
    assert evaluation.hard_evasion_hit_fraction >= 0.75
    assert fire.choose_fire(tank, state, {}) is not None


def test_long_range_lateral_scout_fails_conservative_gate():
    fire = make_fire_control()
    tank = make_tank()
    target = DroneSnapshot(20, Team.B, DroneType.SCOUT, (80.0, 30.0), velocity=(0.0, 5.0))
    state = GameState(5.0, (tank,), (target,), 0, 0)
    evaluation = fire.evaluate_target(tank, target, state, {})
    assert evaluation is not None
    assert evaluation.hit_fraction < fire.NORMAL_HIT_THRESHOLD
    assert fire.choose_fire(tank, state, {}) is None


def test_obstacle_and_friendly_first_hit_are_hard_vetoes():
    wall = RectangleObstacle(14.0, 16.0, 28.0, 32.0)
    fire = make_fire_control((wall,))
    tank = make_tank()
    target = DroneSnapshot(20, Team.B, DroneType.TRANSPORT, (20.0, 30.0))
    state = GameState(5.0, (tank,), (target,), 0, 0)
    blocked = fire.evaluate_target(tank, target, state, {})
    assert blocked is not None
    assert blocked.terrain_first_fraction > 0.0
    assert fire.choose_fire(tank, state, {}) is None

    friend = DroneSnapshot(1, Team.A, DroneType.SCOUT, (15.0, 30.0))
    fire = make_fire_control()
    friendly_state = GameState(5.0, (tank, friend), (target,), 0, 0)
    friendly = fire.evaluate_target(tank, target, friendly_state, {})
    assert friendly is not None
    assert friendly.friendly_first_fraction > 0.0
    assert fire.choose_fire(tank, friendly_state, {}) is None


def test_non_piercing_first_enemy_blocks_intended_target():
    fire = make_fire_control()
    tank = make_tank()
    nearer = DroneSnapshot(19, Team.B, DroneType.SCOUT, (15.0, 30.0))
    target = DroneSnapshot(20, Team.B, DroneType.TRANSPORT, (20.0, 30.0))
    state = GameState(5.0, (tank,), (nearer, target), 0, 0)
    evaluation = fire.evaluate_target(tank, target, state, {})
    assert evaluation is not None
    assert evaluation.hit_fraction == 0.0
    assert fire.choose_fire(tank, state, {}) is not None
    assert all(item.target_id != target.id for item in fire.last_evaluations)


def test_target_histories_are_bounded_and_deterministic():
    fire = make_fire_control()
    tank = make_tank()
    target = DroneSnapshot(20, Team.B, DroneType.TRANSPORT, (20.0, 30.0))
    for index in range(20):
        fire.observe(GameState(5.0 + index * 0.1, (tank,), (target,), 0, 0))
    assert len(fire.histories[target.id]) <= 11
