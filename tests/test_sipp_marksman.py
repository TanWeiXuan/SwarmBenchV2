from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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
