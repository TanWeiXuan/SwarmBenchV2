import pytest

from swarmbench.api import CircleObstacle, GoalZone, RectangleObstacle
from swarmbench.engine.collisions import swept_goal_contact, swept_obstacle_contact, swept_points_contact


def test_head_on_swept_interception() -> None:
    assert swept_points_contact((0.0, 0.0), (10.0, 0.0), (10.0, 0.0), (0.0, 0.0), 0.75) == pytest.approx(0.4625)


def test_parallel_points_do_not_collide() -> None:
    assert swept_points_contact((0.0, 0.0), (10.0, 0.0), (0.0, 2.0), (10.0, 2.0), 0.75) is None


def test_high_speed_circle_collision_does_not_tunnel() -> None:
    obstacle = CircleObstacle((5.0, 0.0), 1.0)
    assert swept_obstacle_contact((0.0, 0.0), (10.0, 0.0), obstacle, 0.25) == pytest.approx(0.375)


def test_high_speed_rectangle_collision_does_not_tunnel() -> None:
    obstacle = RectangleObstacle(4.0, 6.0, -1.0, 1.0)
    assert swept_obstacle_contact((0.0, 0.0), (10.0, 0.0), obstacle, 0.25) == pytest.approx(0.375)


def test_swept_goal_entry() -> None:
    goal = GoalZone(97.0, 100.0, 20.0, 34.0)
    assert swept_goal_contact((95.0, 25.0), (99.0, 25.0), goal) == pytest.approx(0.5)

