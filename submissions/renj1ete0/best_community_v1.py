"""GPT-5 mini High — Best Community Controller v1

Controller: GPT-5 mini High
Author: renj1ete0

A compact, robust single-file controller that combines goal-directed
steering, lightweight obstacle avoidance, projectile dodging, friend
separation, and simple tank gunnery. Designed to be deterministic,
lightweight, and safe for submission validation smoke tests.
"""

from __future__ import annotations

from math import hypot, sqrt
from typing import Optional, Tuple

from swarmbench import BaseSwarmController, DroneStatus, DroneType, Team

# Tuning
TRACK_GAIN = 2.6
FRIEND_REPULSE = 1.6
OBSTACLE_REPULSE = 2.2
PROJECTILE_DODGE_TRIGGER = 1.2
PROJECTILE_DODGE_GAIN = 2.0
LANE_MARGIN = 1.0
TINY = 1e-9


def _clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _closest_approach(rx, ry, rvx, rvy, horizon):
    speed_sq = rvx * rvx + rvy * rvy
    if speed_sq < TINY:
        return hypot(rx, ry), 0.0
    when = - (rx * rvx + ry * rvy) / speed_sq
    if when < 0.0:
        when = 0.0
    if when > horizon:
        when = horizon
    return hypot(rx + rvx * when, ry + rvy * when), when


def _normalize(vx: float, vy: float) -> Tuple[float, float, float]:
    s = hypot(vx, vy)
    if s < TINY:
        return 0.0, 0.0, 0.0
    return vx / s, vy / s, s


class SwarmController(BaseSwarmController):
    CONTROLLER_NAME = "GPT-5 mini High"

    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.home = game_info.own_goal
        self.specs = dict(game_info.drone_specs)
        # guard for older/newer runner payload shapes; fallback to documented 20 m/s
        self.weapon = getattr(game_info, "weapon_spec", None)
        if self.weapon is None:
            class _W:
                def __init__(self):
                    self.projectile_speed = 20.0

            self.weapon = _W()

        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.forward = 1.0 if self.team is Team.A else -1.0
        # keep obstacles split for cheap checks
        self.circles = []
        self.boxes = []
        for obs in game_info.obstacles:
            # CircleObstacle has .center, .radius; rectangles use x_min etc.
            if hasattr(obs, "center"):
                cx, cy = obs.center
                self.circles.append((cx, cy, obs.radius))
            else:
                self.boxes.append((obs.x_min, obs.x_max, obs.y_min, obs.y_max))

        own = sorted(game_info.own_initial_drones, key=lambda d: d.id)
        # assign lanes across the goal mouth to reduce collisions
        totals = {}
        for drone in own:
            totals[drone.drone_type] = totals.get(drone.drone_type, 0) + 1
        seen = {}
        self.lane = {}
        span = max(0.1, self.goal.y_max - self.goal.y_min - 2.0)
        for drone in own:
            rank = seen.get(drone.drone_type, 0)
            seen[drone.drone_type] = rank + 1
            count = max(1, totals.get(drone.drone_type, 1))
            self.lane[drone.id] = self.goal.y_min + 1.0 + span * (rank + 0.5) / count

        self._last_command = {}

    def step(self, state):
        try:
            return self._decide(state)
        except Exception:
            # fail-safe: keep previous commands or push gently forward
            return {
                d.id: self._last_command.get(d.id, (self.forward * 1.0, 0.0))
                for d in state.own_drones
                if d.status is DroneStatus.ACTIVE
            }

    def _decide(self, state):
        own = [d for d in state.own_drones if d.status is DroneStatus.ACTIVE]
        foes = [d for d in state.opponent_drones if d.status is DroneStatus.ACTIVE]
        actions = {}
        for drone in own:
            spec = self.specs[drone.drone_type]
            # base goal-directed acceleration
            acc = self._goal_accel(drone, spec)
            # avoid obstacles and friends
            acc = (acc[0] + self._obstacle_repulse(drone, spec)[0], acc[1] + self._obstacle_repulse(drone, spec)[1])
            acc = (acc[0] + self._friend_repulse(drone, own, spec)[0], acc[1] + self._friend_repulse(drone, own, spec)[1])
            # dodge incoming projectiles
            acc = (acc[0] + self._projectile_dodge(drone, state)[0], acc[1] + self._projectile_dodge(drone, state)[1])
            # clip to class max acceleration
            ax, ay = acc
            mag = hypot(ax, ay)
            if mag > spec.max_acceleration and mag > TINY:
                scale = spec.max_acceleration / mag
                ax, ay = ax * scale, ay * scale
            self._last_command[drone.id] = (ax, ay)

            # TANK gunnery: simple lead on nearest visible foe
            if drone.drone_type is DroneType.TANK and drone.shots_remaining:
                aim = self._tank_aim(drone, foes)
                if aim is not None:
                    actions[drone.id] = {"acceleration": (ax, ay), "fire_direction": aim}
                    continue

            actions[drone.id] = (ax, ay)
        return actions

    # ---------------- movement helpers ----------------

    def _goal_accel(self, drone, spec):
        # If near the goal face, aim for the assigned lane to reduce collisions
        goal_x = self.goal.x_min if self.team is Team.A else self.goal.x_max
        lane_y = self.lane.get(drone.id, (self.goal.y_min + self.goal.y_max) * 0.5)
        if abs(drone.position[0] - goal_x) < 14.0:
            target = (goal_x + self.forward * 2.0, _clamp(lane_y, self.goal.y_min + 1.0, self.goal.y_max - 1.0))
        else:
            target = (goal_x + self.forward * 6.0, lane_y)
        dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
        nx, ny, _ = _normalize(dx, dy)
        desired_v = (nx * spec.max_speed, ny * spec.max_speed)
        ax = TRACK_GAIN * (desired_v[0] - drone.velocity[0])
        ay = TRACK_GAIN * (desired_v[1] - drone.velocity[1])
        return ax, ay

    def _obstacle_repulse(self, drone, spec):
        # Lightweight repulsion from obstacles
        px, py = drone.position
        ax, ay = 0.0, 0.0
        for cx, cy, radius in self.circles:
            rx, ry = px - cx, py - cy
            d = hypot(rx, ry)
            influence = radius + 6.0
            if d < influence and d > TINY:
                strength = OBSTACLE_REPULSE * (influence - d) / influence * spec.max_acceleration
                ax += rx / d * strength
                ay += ry / d * strength
        for x0, x1, y0, y1 in self.boxes:
            # closest point on box
            cx = _clamp(px, x0, x1)
            cy = _clamp(py, y0, y1)
            rx, ry = px - cx, py - cy
            d = hypot(rx, ry)
            influence = max(3.0, max(x1 - x0, y1 - y0) * 0.75)
            if d < influence and d > TINY:
                strength = OBSTACLE_REPULSE * (influence - d) / influence * spec.max_acceleration
                ax += rx / d * strength
                ay += ry / d * strength
        return ax, ay

    def _friend_repulse(self, drone, own, spec):
        px, py = drone.position
        ax, ay = 0.0, 0.0
        for friend in own:
            if friend.id == drone.id:
                continue
            rx, ry = px - friend.position[0], py - friend.position[1]
            d = hypot(rx, ry)
            if d < 0.01:
                continue
            if d < 6.0:
                push = FRIEND_REPULSE * (6.0 - d) / 6.0 * spec.max_acceleration
                ax += rx / d * push
                ay += ry / d * push
        return ax, ay

    # ---------------- projectile handling ----------------

    def _projectile_dodge(self, drone, state):
        px, py = drone.position
        vx, vy = drone.velocity
        ax, ay = 0.0, 0.0
        for shot in state.projectiles:
            if shot.source_drone_id == drone.id:
                continue
            rx, ry = shot.position[0] - px, shot.position[1] - py
            rvx, rvy = shot.velocity[0] - vx, shot.velocity[1] - vy
            miss, when = _closest_approach(rx, ry, rvx, rvy, 3.0)
            if miss >= PROJECTILE_DODGE_TRIGGER or when <= TINY:
                continue
            span = hypot(rvx, rvy)
            if span < TINY:
                continue
            cross = rx * rvy - ry * rvx
            side = 1.0 if cross >= 0.0 else -1.0
            push = PROJECTILE_DODGE_GAIN * (1.0 - miss / PROJECTILE_DODGE_TRIGGER) * (spec_max := self.specs[drone.drone_type].max_acceleration)
            ax += -rvy / span * side * push
            ay += rvx / span * side * push
        return ax, ay

    # ---------------- simple gunnery ----------------

    def _tank_aim(self, tank, foes):
        if not foes:
            return None
        best = None
        best_dist = float("inf")
        for foe in foes:
            # prefer closest (cheap heuristic)
            d = _dist(tank.position, foe.position)
            if d < best_dist:
                best_dist = d
                best = foe
        if best is None:
            return None
        aim, flight = self._lead(tank.position, best)
        if flight <= 0.0:
            return None
        dx, dy = aim[0] - tank.position[0], aim[1] - tank.position[1]
        nx, ny, s = _normalize(dx, dy)
        if s < TINY:
            return None
        return (nx, ny)

    def _lead(self, origin, target):
        # closed-form intercept for constant-velocity target and projectile speed
        speed = self.weapon.projectile_speed
        rx = target.position[0] - origin[0]
        ry = target.position[1] - origin[1]
        vx, vy = target.velocity
        a = vx * vx + vy * vy - speed * speed
        b = 2.0 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry
        times = []
        if abs(a) < 1e-12:
            if abs(b) > 1e-12:
                t = -c / b
                if t > 0:
                    times.append(t)
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                root = sqrt(disc)
                t1 = (-b - root) / (2.0 * a)
                t2 = (-b + root) / (2.0 * a)
                if t1 > 0:
                    times.append(t1)
                if t2 > 0:
                    times.append(t2)
        if not times:
            return ((0.0, 0.0), 0.0)
        flight = min(times)
        return ((target.position[0] + vx * flight, target.position[1] + vy * flight), flight)
