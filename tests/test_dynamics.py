from math import hypot

import pytest

from swarmbench.api import DRONE_SPECS, PHYSICS_DT, DroneSpec, DroneType
from swarmbench.engine.dynamics import DynamicState, advance_dynamics, clip_vector, rk4_constant_acceleration


FAST = DRONE_SPECS[DroneType.SCOUT]


def test_zero_command_preserves_rest() -> None:
    state = advance_dynamics(DynamicState((4.0, 7.0)), (0.0, 0.0), FAST, PHYSICS_DT)
    assert state == DynamicState((4.0, 7.0))


def test_rk4_constant_acceleration_matches_analytic_solution() -> None:
    position, velocity = rk4_constant_acceleration((1.0, 2.0), (3.0, -1.0), (2.0, 4.0), 0.5)
    assert position == pytest.approx((2.75, 2.0))
    assert velocity == pytest.approx((4.0, 1.0))


def test_jerk_limited_rise_reaches_fast_limit_in_quarter_second() -> None:
    state = DynamicState((0.0, 0.0))
    accelerations = []
    for _ in range(5):
        state = advance_dynamics(state, (4.0, 0.0), FAST, PHYSICS_DT)
        accelerations.append(state.acceleration[0])
    assert accelerations == pytest.approx([0.8, 1.6, 2.4, 3.2, 4.0])


def test_velocity_saturation_uses_vector_norm() -> None:
    state = DynamicState((0.0, 0.0), (4.9, 4.9), (4.0, 4.0))
    advanced = advance_dynamics(state, (4.0, 4.0), FAST, PHYSICS_DT)
    assert hypot(*advanced.velocity) == pytest.approx(FAST.max_speed)
    assert advanced.velocity[0] == pytest.approx(advanced.velocity[1])


def test_acceleration_command_saturation_uses_vector_norm() -> None:
    no_jerk_limit = DroneSpec(max_speed=100.0, max_acceleration=4.0, max_jerk=1000.0, point_value=1)
    state = advance_dynamics(DynamicState((0.0, 0.0)), (100.0, 100.0), no_jerk_limit, PHYSICS_DT)
    assert hypot(*state.acceleration) == pytest.approx(4.0)
    assert state.acceleration[0] == pytest.approx(state.acceleration[1])


def test_clip_vector_does_not_apply_axis_limits() -> None:
    clipped = clip_vector((3.0, 4.0), 2.5)
    assert clipped == pytest.approx((1.5, 2.0))


def test_jerk_change_does_not_overshoot_request() -> None:
    state = DynamicState((0.0, 0.0), acceleration=(2.0, 0.0))
    advanced = advance_dynamics(state, (1.8, 0.0), FAST, PHYSICS_DT)
    assert advanced.acceleration == pytest.approx((1.8, 0.0))


def test_nonfinite_vectors_are_rejected() -> None:
    with pytest.raises(ValueError):
        clip_vector((float("nan"), 0.0), 1.0)


def test_repeatability() -> None:
    initial = DynamicState((1.25, 2.5), (-0.2, 0.4), (0.1, -0.3))
    left = initial
    right = initial
    for _ in range(100):
        left = advance_dynamics(left, (3.0, -2.0), FAST, PHYSICS_DT)
        right = advance_dynamics(right, (3.0, -2.0), FAST, PHYSICS_DT)
    assert left == right

