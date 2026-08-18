"""Deterministic jerk-limited holonomic dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from swarmbench.api import DroneSpec, Vec2


@dataclass(frozen=True, slots=True)
class DynamicState:
    position: Vec2
    velocity: Vec2 = (0.0, 0.0)
    acceleration: Vec2 = (0.0, 0.0)


def clip_vector(vector: Vec2, maximum: float) -> Vec2:
    """Clip a 2-D vector by Euclidean norm, preserving its direction."""
    x, y = float(vector[0]), float(vector[1])
    if not isfinite(x) or not isfinite(y):
        raise ValueError("vector components must be finite")
    magnitude = hypot(x, y)
    if magnitude <= maximum or magnitude == 0.0:
        return (x, y)
    scale = maximum / magnitude
    return (x * scale, y * scale)


def _add(left: Vec2, right: Vec2) -> Vec2:
    return (left[0] + right[0], left[1] + right[1])


def _scale(vector: Vec2, factor: float) -> Vec2:
    return (vector[0] * factor, vector[1] * factor)


def rk4_constant_acceleration(position: Vec2, velocity: Vec2, acceleration: Vec2, dt: float) -> tuple[Vec2, Vec2]:
    """RK4 integration for dp/dt=v, dv/dt=a with constant acceleration."""
    k1_p, k1_v = velocity, acceleration
    k2_p, k2_v = _add(velocity, _scale(k1_v, dt / 2)), acceleration
    k3_p, k3_v = _add(velocity, _scale(k2_v, dt / 2)), acceleration
    k4_p, k4_v = _add(velocity, _scale(k3_v, dt)), acceleration

    weighted_p = (
        k1_p[0] + 2 * k2_p[0] + 2 * k3_p[0] + k4_p[0],
        k1_p[1] + 2 * k2_p[1] + 2 * k3_p[1] + k4_p[1],
    )
    weighted_v = (
        k1_v[0] + 2 * k2_v[0] + 2 * k3_v[0] + k4_v[0],
        k1_v[1] + 2 * k2_v[1] + 2 * k3_v[1] + k4_v[1],
    )
    return _add(position, _scale(weighted_p, dt / 6)), _add(velocity, _scale(weighted_v, dt / 6))


def advance_dynamics(state: DynamicState, desired_acceleration: Vec2, spec: DroneSpec, dt: float) -> DynamicState:
    """Advance one physics tick after applying acceleration and jerk limits."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    desired = clip_vector(desired_acceleration, spec.max_acceleration)
    delta = (desired[0] - state.acceleration[0], desired[1] - state.acceleration[1])
    limited_delta = clip_vector(delta, spec.max_jerk * dt)
    acceleration = clip_vector(_add(state.acceleration, limited_delta), spec.max_acceleration)
    position, velocity = rk4_constant_acceleration(state.position, state.velocity, acceleration, dt)
    velocity = clip_vector(velocity, spec.max_speed)
    return DynamicState(position, velocity, acceleration)

