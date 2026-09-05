"""Qwen 3.8 27B (INT4, local) controller for SwarmBench v2.

Design
------
The match is decided by which vehicles are still alive at t=20s. Both swarms
meet in midfield around t=8-16s and largely annihilate each other, and the
single biggest source of self-destruction is friendly contact: any two of our
own vehicles within 0.75 m destroy BOTH. So the controller is organised around
four ideas:

1. A grid flow-field planner (Dijkstra cost-to-go + chamfer clearance) with
   line-of-sight shortcutting gives every vehicle a near-optimal, obstacle-free
   route. A safety filter replays the engine's jerk-limited dynamics and rejects
   any command that would crash, so obstacle deaths are essentially eliminated.

2. A value calculus prices every body by the points it is still likely to
   convert (score chance from cost-to-go vs time left) and prices every contact
   as a trade (enemy points at risk minus our points at risk). Cheap SCOUTs only
   welcome contact when the trade pays; expensive TRANSPORTs avoid it.

3. Roles are assigned centrally each tick: TANKs hold a firing station,
   TRANSPORTs run the goal (staged behind a line while the opening clash is
   still dangerous, each in its own lateral lane), and SCOUTs are allocated to
   guard the goal, raid high-value enemies, escort TRANSPORTs, or stand on an
   enemy TANK's firing line.

4. A one-step lookahead rollout re-decides each vehicle's command by pricing a
   fan of candidate turns against the live enemy/friendly/projectile field, so
   the summed reflex pushes never cancel out into a vehicle holding its line
   into attackers.

Gunnery: TANKs pick the highest expected-value target they can hit (weighted by
score chance and hit probability), with lead prediction, a friendly-fire check,
and target reservation so two TANKs never shoot the same body.
"""

from __future__ import annotations

from heapq import heappop, heappush
from math import atan2, cos, hypot, sin, sqrt

import numpy as np

from swarmbench import (
    BaseSwarmController,
    CircleObstacle,
    DroneStatus,
    DroneType,
    RectangleObstacle,
    Team,
)

# Vec2 in the public API is a plain (x, y) tuple; we alias it for annotations.
Vec2 = tuple[float, float]

TINY = 1e-9
SQ2 = 1.4142135623730951

# Physical radii (the public API does not export them, so pin them here).
RADIUS = 0.25
VEHICLE_TOUCH = 0.75

# Planner geometry. The generator guarantees reachability at
# DRONE_RADIUS + PLANNING_CLEARANCE = 0.25 + 0.35 = 0.6 m, so a 0.5 m grid with
# that clearance is always solvable.
GRID = 0.5
PLAN_CLEARANCE = 0.6
LOS_MARGIN = 0.45
COMFORT = 1.8
COMFORT_WEIGHT = 2.2
CHAIN = 44

# Safety filter: replay the engine dynamics this far ahead.
HORIZON = 2.0
SUB_DT = 0.1
CRASH_MARGIN = 0.12

# Rollout: candidate turns (radians) around the proposed acceleration.
# Symmetric so a vehicle can peel off in either direction.
FAN = (0.35, -0.35, 0.7, -0.7, 1.05, -1.05, 1.4, -1.4, 1.9, -1.9, 2.5, -2.5, 3.14159265)
# Rollout candidate turns (radians) around the proposed acceleration.
# Includes 0.0 (keep the proposal) plus symmetric peels in both directions.
LOOK = (0.0, 0.25, -0.25, 0.55, -0.55, 0.9, -0.9, 1.4, -1.4, 2.2, -2.2)
LOOK_STEPS = 6
LOOK_DT = 0.28

# Roles.
RUN, HUNT, KEEP, GUN, BLOCK, ESCORT = 0, 1, 2, 3, 4, 5

MATCH_DURATION = 90.0

# Tunable parameters (hand-set from the mechanics, then refined by matches).
TUNE = {
    # -- value calculus -------------------------------------------------
    "chance_bias": 1.15,     # score chance at zero cost-to-go, before clamping
    "chance_span": 70.0,     # metres of cost-to-go that exhaust that chance
    "chance_floor": 0.12,    # nothing is ever completely written off
    "magazine_worth": 0.6708,  # points denied per unfired round on a killed TANK

    # -- collision policy -----------------------------------------------
    "shy_accept": 0.1908,      # trade gain above which a SCOUT welcomes contact
    "shy_reach_scout": 15.5579, # a 10 m/s head-on closure needs this much warning
    "shy_reach_heavy": 17.0,
    "shy_gain_scout": 0.85,
    "shy_gain_heavy": 1.5,
    "shy_keeper": 0.55,      # keepers are meant to be in the way
    "shy_horizon": 3.0,
    "shy_trigger": 1.3615,      # metres of predicted miss that still count as a hit
    "shy_patience": 0.45,    # how much a distant-in-time threat is discounted

    # -- roles ----------------------------------------------------------
    "raiders": 1,            # SCOUTs sent after high-value enemy vehicles
    "raid_floor": 0.6643,       # expected points denied before a raid is worth it
    "raid_horizon": 22.0,    # seconds we are willing to spend catching a mark
    "raid_lag": 3.0,         # softens the value-per-second ranking
    "guard_cap": 2,          # SCOUTs that may peel off to kill a ward's pursuer
    "guard_horizon": 7.0,
    "guard_miss": 4.0,
    "guard_slack": 1.0,
    "escort_cap": 2,         # SCOUTs shadowing TRANSPORTs through open ground
    "escort_watch": 20.0,
    "escort_stand": 3.5675,     # metres up the threat bearing the escort sits
    "escort_lead": 0.6,
    "escort_done": 16.0,     # a ward this close to scoring needs no escort
    "keeper_cap": 3,
    "keeper_per_value": 2.2703, # enemy value per keeper held back
    "keeper_post": 10.1857,
    "keeper_reach": 20.092,
    "keeper_watch": 55.0,
    "keeper_greed": 5.7898,
    "block_cap": 2,
    "block_range": 34.0,
    "block_stand": 2.6,
    "chase_lead": 4.0,

    # -- TRANSPORT staging ----------------------------------------------
    "stage_watch": 1.0,      # share of the crossing an interceptor must beat
    "stage_threat": 4.0,     # interceptors still alive that justify waiting
    "stage_depth": 15.8868,     # metres in front of our own goal line to wait at
    "stage_reserve": 1.25,   # safety factor on the remaining crossing time
    "stage_margin": 8.3911,     # plus this many seconds of slack
    "lane_samples": 9,
    "lane_berth": 0.9,       # pull toward the band this vehicle was given
    "lane_mate": 7.0,        # cost of sharing a band with another TRANSPORT
    "lane_width": 11.0,
    "lane_crowd": 9.0,
    "lane_travel": 0.55,     # cost of the lateral move itself

    # -- opportunism ----------------------------------------------------
    "ram_radius": 13.937,
    "ram_floor": 0.8578,        # trade gain required to leave the scoring line
    "ram_detour": 0.06,      # gain charged per metre of detour
    "ram_lead": 3.0,

    # -- reflexes -------------------------------------------------------
    "berth_edge": 7.0,       # metres of arena edge left free when staging
    "friend_reach": 8.0,
    "friend_floor": 2.45,     # separation held regardless of closing speed
    "friend_press": 1.5,
    "friend_trigger": 2.4,
    "friend_horizon": 1.6,
    "friend_gain": 1.7733,
    "dodge_gain": 5.7293,
    "dodge_trigger": 1.6814,
    "dodge_reach": 41.6529,
    "dodge_horizon": 2.3522,
    "dodge_lag": 0.22,
    "gun_fear": 0.75,
    "gun_fear_range": 26.0,

    # -- gunnery --------------------------------------------------------
    "fire_floor": 0.3936,      # expected points denied required to spend a round
    "shot_base": 0.0853,        # worth of a round the target can still dodge
    "shot_span": 3.2964,        # seconds of flight time a shot may take
    "shot_envelope": 1.2,    # miss distance a live round is credited with covering
    "shot_idle": 0.1599,       # worth of killing something that cannot score
    "escape_weight": 0.92,
    "patience": 33.4536,
    "tank_station": 14.4701,    # the guns stand in our own goal mouth
    "tank_seek": 29.5102,
    "friendly_line": 0.95,
    "look_near": 1.7506,        # separation at which a contact stops being priced
    "look_enemy": 0.8696,       # weight on what a contact with them would trade
    "look_friend": 0.5687,     # weight on what a contact with one of ours throws away
    "look_shot": 0.4009,       # weight on walking into a round already in flight
    "look_progress": 0.3061,   # points per metre of the proposed heading kept
    "look_shy": 0.3298,        # what our own body is worth beyond what it converts
}


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def _scale(a: Vec2, s: float) -> Vec2:
    return (a[0] * s, a[1] * s)


def _dist(a: Vec2, b: Vec2) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> float:
    abx, aby = b[0] - a[0], b[1] - a[1]
    if abx == 0.0 and aby == 0.0:
        return _dist(p, a)
    t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / (abx * abx + aby * aby)
    t = _clamp(t, 0.0, 1.0)
    return hypot(p[0] - (a[0] + abx * t), p[1] - (a[1] + aby * t))


def _segment_hits_bounds(
    a: Vec2, b: Vec2, x_min: float, x_max: float, y_min: float, y_max: float
) -> bool:
    if a[0] > b[0] + TINY:
        lo, hi = a[0], b[0]
    else:
        lo, hi = b[0], a[0]
    if a[1] > b[1] + TINY:
        lo_y, hi_y = a[1], b[1]
    else:
        lo_y, hi_y = b[1], a[1]
    if hi < x_min or lo > x_max or hi_y < y_min or lo_y > y_max:
        return False
    if (x_min <= a[0] <= x_max and y_min <= a[1] <= y_max) or (
        x_min <= b[0] <= x_max and y_min <= b[1] <= y_max
    ):
        return True
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0.0:
        if a[1] < y_min or a[1] > y_max:
            return False
        return x_min <= a[0] <= x_max
    if dy == 0.0:
        if a[0] < x_min or a[0] > x_max:
            return False
        return y_min <= a[1] <= y_max
    t1 = (x_min - a[0]) / dx
    t2 = (x_max - a[0]) / dx
    t3 = (y_min - a[1]) / dy
    t4 = (y_max - a[1]) / dy
    tmin = max(0.0, min(t1, t2, t3, t4))
    tmax = min(1.0, max(t1, t2, t3, t4))
    if tmin > tmax:
        return False
    return (x_min <= a[0] + dx * tmin <= x_max and y_min <= a[1] + dy * tmin <= y_max)


def _segment_hits_box(a: Vec2, b: Vec2, box: RectangleObstacle) -> bool:
    return _segment_hits_bounds(a, b, box.x_min, box.x_max, box.y_min, box.y_max)


def _closest_approach(
    origin: Vec2, velocity: Vec2, target: Vec2, target_velocity: Vec2, horizon: float
) -> tuple[float, float]:
    rvx, rvy = origin[0] - target[0], origin[1] - target[1]
    rel_vx, rel_vy = velocity[0] - target_velocity[0], velocity[1] - target_velocity[1]
    a = rel_vx * rel_vx + rel_vy * rel_vy
    if a < TINY:
        return _dist(origin, target), 0.0
    t = -(rvx * rel_vx + rvy * rel_vy) / a
    t = _clamp(t, 0.0, horizon)
    gap = hypot(rvx + rel_vx * t, rvy + rel_vy * t)
    return gap, t


def _pursuit_time(origin: Vec2, target: Vec2, own_speed: float, target_speed: float) -> float:
    gap = _dist(origin, target)
    if gap < TINY:
        return 0.0
    if own_speed <= target_speed:
        return float("inf")
    return gap / (own_speed - target_speed)


def _closest_approach_rel(
    rx: float, ry: float, rvx: float, rvy: float, horizon: float
) -> tuple[float, float]:
    """Smallest separation of two constant-velocity points over [0, horizon]."""
    speed_sq = rvx * rvx + rvy * rvy
    if speed_sq < TINY:
        return hypot(rx, ry), 0.0
    when = _clamp(-(rx * rvx + ry * rvy) / speed_sq, 0.0, horizon)
    return hypot(rx + rvx * when, ry + rvy * when), when


def _pursuit_time_rel(
    gap: float, target_velocity: Vec2, offset: Vec2, speed: float
) -> float | None:
    """Time for a `speed` pursuer to reach a constant-velocity target."""
    vx, vy = target_velocity
    a = vx * vx + vy * vy - speed * speed
    b = 2.0 * (offset[0] * vx + offset[1] * vy)
    c = gap * gap
    if -TINY < a < TINY:
        return -c / b if b < -TINY else None
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    root = sqrt(disc)
    options = [value for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if value > 0.0]
    return min(options) if options else None


class SwarmController(BaseSwarmController):
    def initialize(self, game_info) -> None:
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.home = game_info.own_goal
        self.specs = dict(game_info.drone_specs)
        self.weapon = game_info.weapon_spec
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.forward = 1.0 if self.team is Team.A else -1.0
        self.duration = MATCH_DURATION
        self.goal_face = self.goal.x_min if self.team is Team.A else self.goal.x_max
        self.home_face = self.home.x_max if self.team is Team.A else self.home.x_min

        self._prepare_obstacles(game_info.obstacles)
        self._build_grid()
        self.attack = self._flow_field(self.goal)
        self.defend = self._flow_field(self.home)
        self._chain_cache: dict[tuple[int, int], list[Vec2]] = {}
        self._cost_cache: dict[tuple[int, int], float] = {}

        own = sorted(game_info.own_initial_drones, key=lambda d: d.id)
        self.side = {d.id: 1.0 if d.id % 2 == 0 else -1.0 for d in own}
        self.lane = self._assign_lanes(own)
        self.berth = self._assign_berths(own)
        self.now = 0.0
        self.left = self.duration
        self._chance_cache: dict[int, float] = {}
        self._lanes: dict[int, float] = {}
        self._last_command: dict[int, Vec2] = {}
        self._claims: set[int] = set()
        self._own_active: list = []
        self._foes_active: list = []
        self._reserve_live = sum(
            drone.drone_type is DroneType.TRANSPORT
            for drone in game_info.opponent_initial_drones
        ) >= 7

    # ------------------------------------------------------------------ setup

    def _sd(self, x: float) -> float:
        # Signed distance from the own-goal face along the attack axis.
        # sd == 0 at the own goal, sd grows toward the target goal.
        return self.forward * (x - self.home_face)

    def _prepare_obstacles(self, obstacles) -> None:
        self.circles = []
        self.boxes = []
        for obstacle in obstacles:
            if isinstance(obstacle, CircleObstacle):
                self.circles.append(obstacle)
            elif isinstance(obstacle, RectangleObstacle):
                self.boxes.append(obstacle)
        self.box_bounds = [
            (
                box.x_min - PLAN_CLEARANCE,
                box.y_min - PLAN_CLEARANCE,
                box.x_max + PLAN_CLEARANCE,
                box.y_max + PLAN_CLEARANCE,
            )
            for box in self.boxes
        ]

    def _build_grid(self) -> None:
        self.cols = max(1, int(self.width / GRID) + 1)
        self.rows = max(1, int(self.height / GRID) + 1)
        self.blocked = np.zeros((self.rows, self.cols), dtype=bool)
        for circle in self.circles:
            radius = circle.radius + PLAN_CLEARANCE
            cx, cy = circle.center[0], circle.center[1]
            x0 = max(0, int((cx - radius) / GRID))
            x1 = min(self.cols - 1, int((cx + radius) / GRID))
            y0 = max(0, int((cy - radius) / GRID))
            y1 = min(self.rows - 1, int((cy + radius) / GRID))
            for row in range(y0, y1 + 1):
                py = (row + 0.5) * GRID
                for col in range(x0, x1 + 1):
                    px = (col + 0.5) * GRID
                    if (px - cx) ** 2 + (py - cy) ** 2 <= radius * radius:
                        self.blocked[row, col] = True
        for box, (bx0, by0, bx1, by1) in zip(self.boxes, self.box_bounds):
            x0 = max(0, int(bx0 / GRID))
            x1 = min(self.cols - 1, int(bx1 / GRID))
            y0 = max(0, int(by0 / GRID))
            y1 = min(self.rows - 1, int(by1 / GRID))
            self.blocked[y0 : y1 + 1, x0 : x1 + 1] = True

    def _chamfer(self, blocked: np.ndarray) -> np.ndarray:
        # Distance from every cell to the nearest *blocked* cell. The blocked
        # cells are the sources (0); free cells start at infinity and are
        # filled in by the two-pass chamfer relaxation below. (Seeding the
        # free cells at 0 instead would measure distance to the nearest free
        # cell, which is 0 everywhere and defeats the comfort penalty.)
        distance = np.full((self.rows, self.cols), 1e9, dtype=np.float32)
        distance[blocked] = 0.0
        for row in range(1, self.rows):
            distance[row] = np.minimum(distance[row], distance[row - 1] + 1.0)
        for row in range(self.rows - 2, -1, -1):
            distance[row] = np.minimum(distance[row], distance[row + 1] + 1.0)
        for col in range(1, self.cols):
            distance[:, col] = np.minimum(distance[:, col], distance[:, col - 1] + 1.0)
        for col in range(self.cols - 2, -1, -1):
            distance[:, col] = np.minimum(distance[:, col], distance[:, col + 1] + 1.0)
        for row in range(1, self.rows):
            for col in range(self.cols):
                best = distance[row, col]
                if col > 0:
                    best = min(best, distance[row - 1, col - 1] + SQ2)
                if col < self.cols - 1:
                    best = min(best, distance[row - 1, col + 1] + SQ2)
                distance[row, col] = best
        for row in range(self.rows - 2, -1, -1):
            for col in range(self.cols - 1, -1, -1):
                best = distance[row, col]
                if col > 0:
                    best = min(best, distance[row + 1, col - 1] + SQ2)
                if col < self.cols - 1:
                    best = min(best, distance[row + 1, col + 1] + SQ2)
                distance[row, col] = best
        return distance

    def _flow_field(self, zone) -> np.ndarray:
        # Base per-step cost (1.0 per grid cell) so the field has a genuine
        # gradient toward the goal even in obstacle-free open space; the
        # comfort term on top makes hugging obstacles expensive. Without the
        # base cost the field is flat (0) everywhere and the chain walker has
        # no downhill direction to follow.
        cost = np.ones((self.rows, self.cols), dtype=np.float32)
        cost[self.blocked] = 1e9
        clearance = self._chamfer(self.blocked)
        for row in range(self.rows):
            y = (row + 0.5) * GRID
            if zone.y_min <= y <= zone.y_max:
                col0 = max(0, int(min(zone.x_min, zone.x_max) / GRID))
                col1 = min(self.cols - 1, int(max(zone.x_min, zone.x_max) / GRID))
                cost[row, col0 : col1 + 1] = 0.0
        cost += COMFORT_WEIGHT * np.clip(PLAN_CLEARANCE - clearance, 0.0, None)
        cost[self.blocked] = 1e9
        distance = np.full((self.rows, self.cols), 1e9, dtype=np.float32)
        queue: list[tuple[float, int, int]] = []
        for row in range(self.rows):
            for col in range(self.cols):
                if cost[row, col] == 0.0:
                    distance[row, col] = 0.0
                    heappush(queue, (0.0, row, col))
        while queue:
            value, row, col = heappop(queue)
            if value > distance[row, col] + 1e-6:
                continue
            for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nrow, ncol = row + drow, col + dcol
                if 0 <= nrow < self.rows and 0 <= ncol < self.cols:
                    step = SQ2 if drow and dcol else 1.0
                    candidate = value + cost[nrow, ncol] * step
                    if candidate < distance[nrow, ncol] - 1e-6:
                        distance[nrow, ncol] = candidate
                        heappush(queue, (candidate, nrow, ncol))
        return distance

    def _cell(self, point: Vec2) -> tuple[int, int]:
        col = int(_clamp(point[0] / GRID, 0.0, self.cols - 1))
        row = int(_clamp(point[1] / GRID, 0.0, self.rows - 1))
        return row, col

    def _open_cell(self, point: Vec2) -> tuple[int, int]:
        row, col = self._cell(point)
        if not self.blocked[row, col]:
            return row, col
        for radius in range(1, 6):
            for drow in range(-radius, radius + 1):
                for dcol in range(-radius, radius + 1):
                    nrow, ncol = row + drow, col + dcol
                    if 0 <= nrow < self.rows and 0 <= ncol < self.cols and not self.blocked[nrow, ncol]:
                        return nrow, ncol
        return row, col

    def _cost_to_go(self, drone, field: np.ndarray) -> float:
        key = (drone.id, id(field))
        cached = self._cost_cache.get(key)
        if cached is not None:
            return cached
        row, col = self._open_cell(drone.position)
        value = float(field[row, col])
        self._cost_cache[key] = value
        return value

    def _chain(self, drone, field: np.ndarray) -> list[Vec2]:
        key = (drone.id, id(field))
        cached = self._chain_cache.get(key)
        if cached is not None:
            return cached
        row, col = self._open_cell(drone.position)
        if field[row, col] >= 1e8:
            chain: list[Vec2] = []
        else:
            chain = []
            guard = 0
            while guard < CHAIN:
                guard += 1
                if field[row, col] <= 1e-6:
                    chain.append(((col + 0.5) * GRID, (row + 0.5) * GRID))
                    break
                best = (field[row, col], row, col)
                for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    nrow, ncol = row + drow, col + dcol
                    if 0 <= nrow < self.rows and 0 <= ncol < self.cols:
                        value = field[nrow, ncol]
                        if value < best[0]:
                            best = (value, nrow, ncol)
                if best[1] == row and best[2] == col:
                    break
                row, col = best[1], best[2]
                chain.append(((col + 0.5) * GRID, (row + 0.5) * GRID))
        self._chain_cache[key] = chain
        return chain

    def _visible(self, a: Vec2, b: Vec2, margin: float = LOS_MARGIN) -> bool:
        lo_x, hi_x = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
        lo_y, hi_y = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
        for circle in self.circles:
            cx, cy = circle.center
            reach = circle.radius + margin
            if cx + reach < lo_x or cx - reach > hi_x or cy + reach < lo_y or cy - reach > hi_y:
                continue
            if _point_segment_distance(circle.center, a, b) <= reach:
                return False
        for box in self.boxes:
            x0, x1, y0, y1 = box.x_min, box.x_max, box.y_min, box.y_max
            if x1 + margin < lo_x or x0 - margin > hi_x or y1 + margin < lo_y or y0 - margin > hi_y:
                continue
            if _segment_hits_bounds(a, b, x0 - margin, x1 + margin, y0 - margin, y1 + margin):
                return False
        return True

    def _waypoint(self, drone, field: np.ndarray) -> Vec2:
        chain = self._chain(drone, field)
        if not chain:
            return (self.width / 2.0, self.height / 2.0)
        for point in chain:
            if self._visible(drone.position, point):
                return point
        return chain[0]

    def _route(self, drone, field: np.ndarray) -> Vec2:
        chain = self._chain(drone, field)
        if not chain:
            return (self.width / 2.0, self.height / 2.0)
        # Pick the first chain point that is both visible and at least 1.2
        # units away, so a drone that has reached its immediate waypoint keeps
        # advancing toward the goal instead of being sent to the arena center.
        for point in chain:
            if _dist(drone.position, point) >= 1.2:
                if self._visible(drone.position, point):
                    return point
        # No far-enough visible point (drone is near the goal): fall back to
        # the nearest visible point, then to the farthest chain point.
        for point in chain:
            if self._visible(drone.position, point):
                return point
        return chain[-1]

    # ------------------------------------------------------------- value model

    def _score_chance(self, drone) -> float:
        cached = self._chance_cache.get(drone.id)
        if cached is not None:
            return cached
        if drone.status is not DroneStatus.ACTIVE:
            value = 0.0
        else:
            # Own bodies are priced by how close they are to the target goal;
            # enemy bodies by how close they are to scoring on our goal.
            field = self.attack if drone.team is self.team else self.defend
            cost = self._cost_to_go(drone, field)
            spec = self.specs[drone.drone_type]
            if cost >= 1e8 or cost > spec.max_speed * self.left:
                value = 0.0
            else:
                value = _clamp(
                    TUNE["chance_bias"] - cost / TUNE["chance_span"],
                    TUNE["chance_floor"],
                    1.0,
                )
        self._chance_cache[drone.id] = value
        return value

    def _trade_gain(self, me, foe) -> float:
        gain = self.specs[foe.drone_type].point_value * self._score_chance(foe)
        gain -= self.specs[me.drone_type].point_value * self._score_chance(me)
        if foe.drone_type is DroneType.TANK and foe.shots_remaining:
            gain += TUNE["magazine_worth"] * foe.shots_remaining
        return gain

    # ------------------------------------------------------------------ planning

    def _assign_duties(self, state, own, foes) -> dict[int, tuple]:
        duties: dict[int, tuple] = {}
        scouts = []
        for drone in own:
            if drone.drone_type is DroneType.TANK:
                duties[drone.id] = (GUN if drone.shots_remaining else RUN, None)
            elif drone.drone_type is DroneType.TRANSPORT:
                duties[drone.id] = (RUN, None)
            else:
                scouts.append(drone)
        if not scouts:
            return duties
        wards = [drone for drone in own if drone.drone_type is DroneType.TRANSPORT]
        wards.sort(key=lambda drone: self._cost_to_go(drone, self.attack))
        free = list(scouts)
        want_keep = min(self._keepers_wanted(foes), len(free))
        if want_keep:
            free.sort(key=lambda scout: (self._cost_to_go(scout, self.defend), scout.id))
            for scout in free[:want_keep]:
                duties[scout.id] = (KEEP, None)
            del free[:want_keep]
        for mark in self._raid_targets(foes, free)[: TUNE["raiders"]]:
            if not free:
                break
            best = min(free, key=lambda scout: (self._pursuit_reach(scout, mark), scout.id))
            free.remove(best)
            duties[best.id] = (HUNT, mark)
        for foe, _when in self._pursuers(wards, foes, free)[: TUNE["guard_cap"]]:
            if not free:
                break
            best = min(free, key=lambda scout: (self._pursuit_reach(scout, foe), scout.id))
            free.remove(best)
            duties[best.id] = (HUNT, foe)
        for ward, gun in self._gun_lines(wards, foes)[: TUNE["block_cap"]]:
            if not free:
                break
            best = min(free, key=lambda scout: (_dist(scout.position, ward.position), scout.id))
            free.remove(best)
            duties[best.id] = (BLOCK, (ward, gun))
        for ward in self._escort_wants(wards, foes)[: TUNE["escort_cap"]]:
            if not free:
                break
            best = min(free, key=lambda scout: (_dist(scout.position, ward.position), scout.id))
            free.remove(best)
            duties[best.id] = (ESCORT, ward)
        for scout in free:
            duties[scout.id] = (RUN, None)
        return duties

    def _reach_time(self, drone) -> float:
        cost = self._cost_to_go(drone, self.attack)
        if cost >= 1e8:
            return float("inf")
        return cost / self.specs[drone.drone_type].max_speed

    def _pursuit_reach(self, scout, mark) -> float:
        speed = self.specs[scout.drone_type].max_speed
        offset = (mark.position[0] - scout.position[0], mark.position[1] - scout.position[1])
        gap = hypot(offset[0], offset[1])
        when = _pursuit_time_rel(gap, mark.velocity, offset, speed)
        return when if when is not None else gap / max(speed, TINY)

    def _keepers_wanted(self, foes) -> int:
        watch = TUNE["keeper_watch"]
        threat = 0.0
        for foe in foes:
            cost = self._cost_to_go(foe, self.defend)
            if cost >= watch:
                continue
            threat += (1.0 - cost / watch) * self.specs[foe.drone_type].point_value * self._score_chance(foe)
        return int(_clamp(threat / TUNE["keeper_per_value"], 0.0, TUNE["keeper_cap"]))

    def _raid_targets(self, foes, free) -> list:
        if not free:
            return []
        picks = []
        for foe in foes:
            if foe.drone_type is DroneType.SCOUT:
                continue
            worth = self.specs[foe.drone_type].point_value * self._score_chance(foe)
            if foe.drone_type is DroneType.TANK and foe.shots_remaining:
                worth += TUNE["magazine_worth"] * foe.shots_remaining
            if worth < TUNE["raid_floor"]:
                continue
            when = min(self._pursuit_reach(scout, foe) for scout in free)
            if when > TUNE["raid_horizon"]:
                continue
            picks.append((-worth / (when + TUNE["raid_lag"]), foe.id, foe))
        picks.sort()
        return [foe for _rate, _id, foe in picks]

    def _pursuers(self, wards, foes, free) -> list:
        if not wards or not free:
            return []
        horizon = TUNE["guard_horizon"]
        threats = {}
        for ward in wards:
            for foe in foes:
                offset = (ward.position[0] - foe.position[0], ward.position[1] - foe.position[1])
                gap = hypot(offset[0], offset[1])
                if gap > horizon * self.specs[foe.drone_type].max_speed:
                    continue
                catch = _pursuit_time_rel(gap, ward.velocity, offset, self.specs[foe.drone_type].max_speed)
                if catch is None or catch > horizon:
                    continue
                miss, _when = _closest_approach_rel(
                    -offset[0], -offset[1],
                    foe.velocity[0] - ward.velocity[0], foe.velocity[1] - ward.velocity[1],
                    horizon,
                )
                if miss > TUNE["guard_miss"]:
                    continue
                reach = min(self._pursuit_reach(scout, foe) for scout in free)
                if reach > catch + TUNE["guard_slack"]:
                    continue
                if catch < threats.get(foe.id, (None, 1.0e6))[1]:
                    threats[foe.id] = (foe, catch)
        return sorted(threats.values(), key=lambda item: (item[1], item[0].id))

    def _escort_wants(self, wards, foes) -> list:
        picks = []
        for ward in wards:
            if self._cost_to_go(ward, self.attack) < TUNE["escort_done"]:
                continue
            exposure = 0.0
            for foe in foes:
                gap = _dist(foe.position, ward.position)
                if gap < TUNE["escort_watch"]:
                    exposure += 1.0 - gap / TUNE["escort_watch"]
            if exposure <= 0.0:
                continue
            picks.append((-exposure, ward.id, ward))
        picks.sort()
        return [ward for _e, _i, ward in picks]

    def _gun_lines(self, wards, foes) -> list:
        guns = [foe for foe in foes if foe.drone_type is DroneType.TANK and foe.shots_remaining]
        if not guns:
            return []
        reach = TUNE["block_range"]
        lines = []
        for ward in wards:
            best, best_gap = None, reach
            for gun in guns:
                gap = _dist(gun.position, ward.position)
                if gap < best_gap and self._visible(gun.position, ward.position, margin=0.0):
                    best, best_gap = gun, gap
            if best is not None:
                lines.append((ward, best))
        return lines

    # ------------------------------------------------------------- lane / berth

    def _assign_lanes(self, own) -> dict[int, float]:
        lanes: dict[int, float] = {}
        drones = sorted(own, key=lambda d: d.id)
        count = max(1, len(drones))
        for index, drone in enumerate(drones):
            y = TUNE["lane_berth"] + (self.height - 2.0 * TUNE["lane_berth"]) * (index + 0.5) / count
            lanes[drone.id] = y
        return lanes

    def _assign_berths(self, own) -> dict[int, float]:
        berths: dict[int, float] = {}
        for drone in own:
            berths[drone.id] = drone.position[1]
        return berths

    # ----------------------------------------------------------------- behaviors

    def _goal_run(self, state, drone, foes) -> Vec2:
        if drone.drone_type is DroneType.TRANSPORT and self._staged(drone):
            return self._stage(drone)
        waypoint = self._route(drone, self.attack)
        return self._steer_to(drone, waypoint)

    def _staged(self, drone) -> bool:
        if self.now < TUNE["stage_watch"]:
            return False
        crossing = self._reach_time(drone)
        if not crossing < float("inf"):
            if self.left < TUNE["stage_reserve"] * crossing + TUNE["stage_margin"]:
                return False
        threats = 0
        for foe in self._foes_active:
            if foe.status is not DroneStatus.ACTIVE:
                continue
            if foe.drone_type is DroneType.TANK:
                continue
            # Only enemies already past the stage line (in the run to the goal)
            # can intercept a releasing transport.
            if self._sd(foe.position[0]) < TUNE["stage_depth"]:
                continue
            if _pursuit_time(foe.position, drone.position, self.specs[foe.drone_type].max_speed, self.specs[drone.drone_type].max_speed) < TUNE["stage_reserve"] * crossing:
                threats += 1
        return threats >= TUNE["stage_threat"]

    def _stage(self, drone) -> Vec2:
        post_x = self.home_face + self.forward * TUNE["stage_depth"]
        lane = self._lanes.get(drone.id, self.lane.get(drone.id, self.height / 2.0))
        return self._steer_to(drone, (post_x, lane))

    def _stage_lanes(self, own) -> None:
        self._lanes = {}
        staged = [
            d for d in own
            if d.status is DroneStatus.ACTIVE
            and d.drone_type is DroneType.TRANSPORT
            and self._staged(d)
        ]
        if not staged:
            return
        staged.sort(key=lambda d: self.berth.get(d.id, self.height / 2.0))
        for index, drone in enumerate(staged):
            samples = [
                TUNE["lane_berth"] + (self.height - 2.0 * TUNE["lane_berth"]) * (i + 0.5) / TUNE["lane_samples"]
                for i in range(TUNE["lane_samples"])
            ]
            best = min(
                samples,
                key=lambda y: (
                    min((_dist((self.home_face, y), o.position) for o in own if o.id != drone.id and o.drone_type is DroneType.TRANSPORT), default=1e9)
                    + TUNE["lane_crowd"] * abs(y - self.berth.get(drone.id, self.height / 2.0)) / max(1.0, TUNE["lane_width"])
                    + TUNE["lane_mate"] * abs(y - self.lane.get(drone.id, self.height / 2.0)) / max(1.0, TUNE["lane_width"])
                ),
            )
            self._lanes[drone.id] = best

    def _free_kill(self, drone, foes) -> Vec2 | None:
        best: Vec2 | None = None
        best_gain = TUNE["ram_floor"]
        for foe in foes:
            if foe.status is not DroneStatus.ACTIVE:
                continue
            if _dist(drone.position, foe.position) > TUNE["ram_radius"]:
                continue
            gain = self._trade_gain(drone, foe)
            if gain < best_gain:
                continue
            cost = self._cost_to_go(drone, self.attack)
            detour = _dist(drone.position, foe.position) - self._sd(drone.position[0])
            if cost < 1e8 and detour > TUNE["ram_detour"] * cost:
                continue
            best_gain = gain
            best = (foe.position[0] + self.forward * TUNE["ram_lead"], foe.position[1])
        return best

    def _chase(self, drone, target) -> Vec2:
        if target is None:
            return self._goal_run(None, drone, self._foes_active)
        lead = (
            target.position[0] + self.forward * TUNE["chase_lead"] * (target.velocity[0] / max(1.0, self.specs[target.drone_type].max_speed)),
            target.position[1],
        )
        return self._steer_to(drone, lead)

    def _escort(self, drone, ward) -> Vec2:
        if ward is None:
            return self._goal_run(None, drone, self._foes_active)
        offset = self.side.get(drone.id, 1.0) * TUNE["escort_stand"]
        post = (
            ward.position[0] - self.forward * TUNE["escort_lead"],
            ward.position[1] + offset,
        )
        return self._steer_to(drone, post)

    def _keep(self, drone, foes) -> Vec2:
        mouth = (self.home_face + self.forward * TUNE["keeper_post"], self.lane.get(drone.id, self.height / 2.0))
        threats = [
            f for f in foes
            if f.status is DroneStatus.ACTIVE
            and f.drone_type is not DroneType.TANK
            and self._sd(f.position[0]) < TUNE["keeper_watch"]
        ]
        threats.sort(key=lambda f: self._sd(f.position[0]))
        if threats:
            threat = threats[0]
            if _dist(drone.position, threat.position) < TUNE["keeper_reach"]:
                return self._steer_to(drone, (threat.position[0] + self.forward * TUNE["keeper_greed"], threat.position[1]))
        return self._steer_to(drone, mouth)

    def _block(self, drone, pair) -> Vec2:
        if pair is None:
            return self._goal_run(None, drone, self._foes_active)
        tank_id, ward_id = pair
        tank = next((f for f in self._foes_active if f.id == tank_id), None)
        ward = next((d for d in self._own_active if d.id == ward_id), None)
        if tank is None or ward is None:
            return self._goal_run(None, drone, self._foes_active)
        mid = ((tank.position[0] + ward.position[0]) / 2.0, (tank.position[1] + ward.position[1]) / 2.0)
        return self._steer_to(drone, mid)

    def _tank_move(self, state, drone, foes) -> Vec2:
        station_x = self.home_face + self.forward * TUNE["tank_station"]
        threats = [
            f for f in foes
            if f.status is DroneStatus.ACTIVE
            and f.drone_type is not DroneType.TANK
            and self._sd(f.position[0]) < TUNE["tank_seek"]
        ]
        if threats:
            threats.sort(key=lambda f: self._sd(f.position[0]))
            threat = threats[0]
            if _dist(drone.position, threat.position) < TUNE["tank_seek"]:
                return self._steer_to(drone, (station_x, threat.position[1]))
        return self._steer_to(drone, (station_x, self.lane.get(drone.id, self.height / 2.0)))

    def _steer_to(self, drone, target: Vec2) -> Vec2:
        spec = self.specs[drone.drone_type]
        dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
        distance = hypot(dx, dy)
        if distance < 0.35:
            return (0.0, 0.0)
        desired_speed = min(spec.max_speed, spec.max_acceleration * max(0.5, distance / spec.max_speed))
        desired_vx = desired_speed * dx / distance
        desired_vy = desired_speed * dy / distance
        desired_ax = _clamp((desired_vx - drone.velocity[0]) / 0.1, -spec.max_acceleration, spec.max_acceleration)
        desired_ay = _clamp((desired_vy - drone.velocity[1]) / 0.1, -spec.max_acceleration, spec.max_acceleration)
        return (desired_ax, desired_ay)

    # ------------------------------------------------------------------ reflexes

    def _engage(self, drone, foes, role) -> Vec2:
        spec = self.specs[drone.drone_type]
        reach = TUNE["shy_reach_scout"] if drone.drone_type is DroneType.SCOUT else TUNE["shy_reach_heavy"]
        gain_scale = TUNE["shy_gain_scout"] if drone.drone_type is DroneType.SCOUT else TUNE["shy_gain_heavy"]
        if role is KEEP:
            gain_scale *= TUNE["shy_keeper"]
        best = None
        best_score = TUNE["shy_accept"]
        for foe in foes:
            if foe.status is not DroneStatus.ACTIVE:
                continue
            gap = _dist(drone.position, foe.position)
            if gap > reach:
                continue
            gain = self._trade_gain(drone, foe)
            if gain < best_score:
                continue
            best_score = gain
            best = foe
        if best is None:
            return (0.0, 0.0)
        gap, t_close = _closest_approach(drone.position, drone.velocity, best.position, best.velocity, TUNE["shy_horizon"])
        if gap < TINY:
            return (0.0, 0.0)
        if t_close > TUNE["shy_horizon"] or gap > TUNE["shy_trigger"]:
            return (0.0, 0.0)
        if t_close < TINY:
            t_close = TUNE["shy_patience"]
        ux, uy = (drone.position[0] - best.position[0]) / gap, (drone.position[1] - best.position[1]) / gap
        strength = _clamp(gain_scale * (best_score - TUNE["shy_accept"]), 0.0, spec.max_acceleration)
        return (ux * strength, uy * strength)

    def _keep_off_guns(self, drone, foes) -> Vec2:
        if drone.drone_type is not DroneType.TRANSPORT:
            return (0.0, 0.0)
        spec = self.specs[drone.drone_type]
        best = None
        best_score = 0.0
        for foe in foes:
            if foe.status is not DroneStatus.ACTIVE or foe.drone_type is not DroneType.TANK or (foe.shots_remaining or 0) <= 0:
                continue
            gap = _dist(drone.position, foe.position)
            if gap > TUNE["gun_fear_range"]:
                continue
            score = TUNE["gun_fear"] * (1.0 - gap / TUNE["gun_fear_range"])
            if score < best_score:
                continue
            best_score = score
            best = foe
        if best is None:
            return (0.0, 0.0)
        gap, t_close = _closest_approach(drone.position, drone.velocity, best.position, best.velocity, TUNE["shy_horizon"])
        if gap < TINY or t_close > TUNE["shy_horizon"] or gap > TUNE["shy_trigger"]:
            return (0.0, 0.0)
        ux, uy = (drone.position[0] - best.position[0]) / gap, (drone.position[1] - best.position[1]) / gap
        strength = _clamp(best_score * spec.max_acceleration, 0.0, spec.max_acceleration)
        return (ux * strength, uy * strength)

    def _avoid_friends(self, drone, own) -> Vec2:
        spec = self.specs[drone.drone_type]
        my_value = self.specs[drone.drone_type].point_value
        best = None
        best_score = TUNE["friend_floor"]
        for other in own:
            if other.id == drone.id or other.status is not DroneStatus.ACTIVE:
                continue
            gap = _dist(drone.position, other.position)
            if gap > TUNE["friend_reach"]:
                continue
            other_value = self.specs[other.drone_type].point_value
            weight = 1.0 + 0.5 * (other_value - my_value)
            gap2, t_close = _closest_approach(drone.position, drone.velocity, other.position, other.velocity, TUNE["friend_horizon"])
            if gap2 > TUNE["friend_trigger"]:
                continue
            score = weight * TUNE["friend_gain"]
            if score < best_score:
                continue
            best_score = score
            best = (other, gap, t_close)
        if best is None:
            return (0.0, 0.0)
        other, gap, t_close = best
        if gap < TINY:
            # Stacked on a friend: the away-direction is undefined, so part
            # ways along a perpendicular to our velocity; the id-based side
            # breaks symmetry so a paired pair pushes in opposite directions.
            speed = hypot(drone.velocity[0], drone.velocity[1])
            s = self.side.get(drone.id, 1.0)
            if speed < TINY:
                ux, uy = s, 0.0
            else:
                ux, uy = -drone.velocity[1] / speed * s, drone.velocity[0] / speed * s
            return (ux * spec.max_acceleration, uy * spec.max_acceleration)
        if t_close < TINY:
            t_close = TINY
        ux, uy = (drone.position[0] - other.position[0]) / gap, (drone.position[1] - other.position[1]) / gap
        strength = _clamp(best_score * spec.max_acceleration, 0.0, spec.max_acceleration)
        return (ux * strength, uy * strength)

    def _dodge(self, drone, state) -> Vec2:
        spec = self.specs[drone.drone_type]
        best = None
        best_score = -1.0
        for projectile in state.projectiles:
            if projectile.team == self.team:
                continue
            gap = _dist(drone.position, projectile.position)
            if gap > TUNE["dodge_reach"]:
                continue
            gap2, t_close = _closest_approach(
                drone.position, drone.velocity, projectile.position, projectile.velocity, TUNE["dodge_horizon"]
            )
            if gap2 > TUNE["dodge_trigger"]:
                continue
            if t_close < TINY:
                t_close = TUNE["dodge_lag"]
            ux, uy = (drone.position[0] - projectile.position[0]) / max(gap, TINY), (drone.position[1] - projectile.position[1]) / max(gap, TINY)
            score = TUNE["dodge_gain"] * (1.0 - gap / TUNE["dodge_reach"])
            if score < best_score:
                continue
            best_score = score
            best = (ux, uy)
        if best is None:
            return (0.0, 0.0)
        strength = _clamp(best_score * spec.max_acceleration, 0.0, spec.max_acceleration)
        return (best[0] * strength, best[1] * strength)

    # ------------------------------------------------------------------ gunnery

    def _lead(self, origin: Vec2, target) -> tuple[Vec2, float]:
        speed = self.weapon.projectile_speed
        rx = target.position[0] - origin[0]
        ry = target.position[1] - origin[1]
        vx, vy = target.velocity
        a = vx * vx + vy * vy - speed * speed
        b = 2.0 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry
        times = []
        if -TINY < a < TINY:
            if abs(b) > TINY:
                times.append(-c / b)
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                root = sqrt(disc)
                times.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        flight = min((value for value in times if value > 0.0), default=0.0)
        return (target.position[0] + vx * flight, target.position[1] + vy * flight), flight

    def _hit_chance(self, foe, flight: float) -> float:
        spec = self.specs[foe.drone_type]
        lag = max(0.0, flight - TUNE["dodge_lag"])
        escape = 0.5 * spec.max_acceleration * lag * lag
        certain = _clamp(1.0 - TUNE["escape_weight"] * escape / 0.75, 0.0, 1.0)
        base = TUNE["shot_base"]
        return base + (1.0 - base) * certain

    def _line_value(self, tank, foes, pvx: float, pvy: float, span: float) -> float:
        total = 0.0
        for foe in foes:
            rx = tank.position[0] - foe.position[0]
            ry = tank.position[1] - foe.position[1]
            miss, when = _closest_approach_rel(rx, ry, pvx - foe.velocity[0], pvy - foe.velocity[1], span)
            if miss >= 0.75 or when <= TINY:
                continue
            worth = self.specs[foe.drone_type].point_value
            worth *= TUNE["shot_idle"] + (1.0 - TUNE["shot_idle"]) * self._score_chance(foe)
            total += worth * self._hit_chance(foe, when)
        return total

    def _fire_floor(self, tank, state) -> float:
        slack = self.left - tank.shots_remaining * self.weapon.cooldown
        if slack > TUNE["patience"]:
            return TUNE["fire_floor"]
        if slack > TUNE["patience"] * 0.4:
            return TUNE["fire_floor"] * 0.4
        return 0.05

    def _friendly_in_line(self, tank, own, pvx: float, pvy: float, flight: float) -> bool:
        for friend in own:
            if friend.id == tank.id:
                continue
            rx = tank.position[0] - friend.position[0]
            ry = tank.position[1] - friend.position[1]
            miss, _ = _closest_approach_rel(rx, ry, pvx - friend.velocity[0], pvy - friend.velocity[1], flight)
            if miss < TUNE["friendly_line"]:
                return True
        return False

    def _gunnery(self, tank, own, foes, state) -> Vec2 | None:
        if not tank.shots_remaining or tank.next_fire_time is None or state.time + TINY < tank.next_fire_time:
            return None
        speed = self.weapon.projectile_speed
        span = TUNE["shot_span"]
        if self._reserve_live:
            for shot in state.projectiles:
                if shot.team is not self.team:
                    continue
                covered = None
                for foe in foes:
                    miss, when = _closest_approach_rel(
                        shot.position[0] - foe.position[0],
                        shot.position[1] - foe.position[1],
                        shot.velocity[0] - foe.velocity[0],
                        shot.velocity[1] - foe.velocity[1],
                        span,
                    )
                    if (miss < TUNE["shot_envelope"] and when > TINY
                            and (covered is None or when < covered[0])):
                        covered = (when, foe.id)
                if covered is not None:
                    self._claims.add(covered[1])
        best, best_target = None, None
        best_score = self._fire_floor(tank, state)
        for foe in foes:
            if foe.id in self._claims:
                continue
            aim, flight = self._lead(tank.position, foe)
            if flight <= 0.0 or flight > span:
                continue
            reach = _dist(tank.position, aim)
            if reach < TINY:
                continue
            if not self._visible(tank.position, aim, margin=0.0):
                continue
            dirx = (aim[0] - tank.position[0]) / reach
            diry = (aim[1] - tank.position[1]) / reach
            if self._friendly_in_line(tank, own, dirx * speed, diry * speed, span):
                continue
            score = self._line_value(tank, (foe,), dirx * speed, diry * speed, span)
            if score > best_score:
                best, best_target, best_score = (dirx, diry), foe.id, score
        if best_target is not None:
            self._claims.add(best_target)
        return best

    # ------------------------------------------------------------------ rollout

    def _rollout(self, drone, base: Vec2, own, foes, state) -> Vec2:
        spec = self.specs[drone.drone_type]
        ax, ay = base
        magnitude = hypot(ax, ay)
        if magnitude < TINY:
            ax, ay, magnitude = self.forward * spec.max_acceleration, 0.0, spec.max_acceleration
        ux, uy = ax / magnitude, ay / magnitude

        marks = []
        mine = self.specs[drone.drone_type].point_value
        near = TUNE["look_near"]
        px, py = drone.position
        span = near + spec.max_speed * (LOOK_STEPS * LOOK_DT)
        for foe in foes:
            if abs(foe.position[0] - px) > span or abs(foe.position[1] - py) > span:
                continue
            marks.append((foe.position, foe.velocity,
                          (self._trade_gain(drone, foe) - TUNE["look_shy"]) * TUNE["look_enemy"]))
        for friend in own:
            if friend.id == drone.id:
                continue
            if abs(friend.position[0] - px) > span or abs(friend.position[1] - py) > span:
                continue
            theirs = self.specs[friend.drone_type].point_value
            marks.append((friend.position, friend.velocity,
                          -(mine + theirs) * TUNE["look_friend"]))
        for shot in state.projectiles:
            if shot.source_drone_id == drone.id:
                continue
            if abs(shot.position[0] - px) > span + 24.0 or abs(shot.position[1] - py) > span + 24.0:
                continue
            marks.append((shot.position, shot.velocity, -mine * TUNE["look_shot"]))
        if not marks:
            return base

        turns = np.asarray(LOOK)
        cs, sn = np.cos(turns), np.sin(turns)
        cand = np.empty((turns.size, 2))
        cand[:, 0] = (ux * cs - uy * sn) * spec.max_acceleration
        cand[:, 1] = (ux * sn + uy * cs) * spec.max_acceleration

        pos = np.empty((turns.size, LOOK_STEPS, 2))
        acc = np.tile(np.asarray(drone.acceleration, dtype=float), (turns.size, 1))
        here = np.tile(np.asarray(drone.position, dtype=float), (turns.size, 1))
        vel = np.tile(np.asarray(drone.velocity, dtype=float), (turns.size, 1))
        jerk = spec.max_jerk * LOOK_DT
        for step in range(LOOK_STEPS):
            delta = cand - acc
            reach = np.hypot(delta[:, 0], delta[:, 1])
            acc = acc + delta * np.minimum(1.0, jerk / np.maximum(reach, TINY))[:, None]
            here = here + vel * LOOK_DT + 0.5 * acc * (LOOK_DT * LOOK_DT)
            vel = vel + acc * LOOK_DT
            fast = np.hypot(vel[:, 0], vel[:, 1])
            vel = vel * np.minimum(1.0, spec.max_speed / np.maximum(fast, TINY))[:, None]
            pos[:, step, :] = here

        clock = (np.arange(1, LOOK_STEPS + 1) * LOOK_DT)[:, None]
        track = np.empty((len(marks), LOOK_STEPS, 2))
        stake = np.empty(len(marks))
        for i, (mp, mv, worth) in enumerate(marks):
            track[i] = np.asarray(mp) + np.asarray(mv) * clock
            stake[i] = worth
        gap = np.sqrt(((pos[:, None, :, :] - track[None, :, :, :]) ** 2).sum(-1)).min(2)
        touch = np.clip((near - gap) / max(near - VEHICLE_TOUCH, TINY), 0.0, 1.0)
        score = touch @ stake

        forward = (pos[:, -1, 0] - px) * ux + (pos[:, -1, 1] - py) * uy
        score = score + forward * TUNE["look_progress"]
        for pick in np.argsort(-score, kind="stable"):
            if pick == 0:
                return base
            choice = (float(cand[pick, 0]), float(cand[pick, 1]))
            if not self._crashes(drone, choice):
                return choice
        return base

    # ------------------------------------------------------------- safety filter

    def _near_obstacles(self, point: Vec2, radius: float):
        nearby = []
        for box, (bx0, by0, bx1, by1) in zip(self.boxes, self.box_bounds):
            if point[0] < bx0 - radius or point[0] > bx1 + radius or point[1] < by0 - radius or point[1] > by1 + radius:
                continue
            nearby.append(box)
        for circle in self.circles:
            if _dist(point, circle.center) > circle.radius + radius:
                continue
            nearby.append(circle)
        return nearby

    def _crashes(self, drone, command: Vec2) -> bool:
        # Replay the engine's jerk-limited dynamics and test the *swept* path
        # (segment per sub-step) against the obstacles, exactly like the
        # authoritative continuous collision check. Endpoint-only tests let
        # fast drones tunnel through thin obstacles.
        spec = self.specs[drone.drone_type]
        px, py = drone.position[0], drone.position[1]
        vx, vy = drone.velocity[0], drone.velocity[1]
        cax, cay = drone.acceleration[0], drone.acceleration[1]
        dax, day = command[0], command[1]
        magnitude = hypot(dax, day)
        if magnitude > spec.max_acceleration:
            scale = spec.max_acceleration / magnitude
            dax, day = dax * scale, day * scale
        radius = RADIUS + CRASH_MARGIN
        reach = hypot(vx, vy) * HORIZON + 0.5 * spec.max_acceleration * HORIZON * HORIZON + 1.0
        nearby = self._near_obstacles((px, py), reach)
        circles = [o for o in nearby if isinstance(o, CircleObstacle)]
        boxes = [o for o in nearby if isinstance(o, RectangleObstacle)]
        steps = int(HORIZON / SUB_DT)
        for _ in range(steps):
            jx, jy = dax - cax, day - cay
            span = hypot(jx, jy)
            if span > spec.max_jerk * SUB_DT:
                scale = spec.max_jerk * SUB_DT / span
                jx, jy = jx * scale, jy * scale
            cax, cay = cax + jx, cay + jy
            accel_mag = hypot(cax, cay)
            if accel_mag > spec.max_acceleration:
                scale = spec.max_acceleration / accel_mag
                cax, cay = cax * scale, cay * scale
            ex = px + vx * SUB_DT + 0.5 * cax * SUB_DT * SUB_DT
            ey = py + vy * SUB_DT + 0.5 * cay * SUB_DT * SUB_DT
            vx, vy = vx + cax * SUB_DT, vy + cay * SUB_DT
            speed = hypot(vx, vy)
            if speed > spec.max_speed:
                scale = spec.max_speed / speed
                vx, vy = vx * scale, vy * scale
            if (
                px < radius
                or px > self.width - radius
                or py < radius
                or py > self.height - radius
                or ex < radius
                or ex > self.width - radius
                or ey < radius
                or ey > self.height - radius
            ):
                return True
            for circle in circles:
                if _point_segment_distance(circle.center, (px, py), (ex, ey)) <= circle.radius + radius:
                    return True
            for box in boxes:
                if _segment_hits_bounds(
                    (px, py),
                    (ex, ey),
                    box.x_min - radius,
                    box.x_max + radius,
                    box.y_min - radius,
                    box.y_max + radius,
                ):
                    return True
            px, py = ex, ey
        return False

    def _safe_acceleration(self, drone, command: Vec2) -> Vec2:
        if not self._crashes(drone, command):
            return command
        angle = atan2(command[1], command[0])
        magnitude = hypot(command[0], command[1])
        for turn in FAN:
            candidate = (magnitude * cos(angle + turn), magnitude * sin(angle + turn))
            if not self._crashes(drone, candidate):
                return candidate
        return (0.0, 0.0)

    # --------------------------------------------------------------------- step

    def step(self, state) -> dict[int, Vec2 | dict]:
        try:
            return self._decide(state)
        except Exception:
            return {
                drone.id: self._last_command.get(drone.id, (0.0, 0.0))
                for drone in state.own_drones
                if drone.status is DroneStatus.ACTIVE
            }

    def _decide(self, state) -> dict[int, Vec2 | dict]:
        self.now = state.time
        self.left = max(0.0, self.duration - state.time)
        own = [d for d in state.own_drones if d.status is DroneStatus.ACTIVE]
        foes = [d for d in state.opponent_drones if d.status is DroneStatus.ACTIVE]
        self._own_active = own
        self._foes_active = foes
        self._chance_cache.clear()
        self._cost_cache.clear()
        self._chain_cache.clear()
        self._claims.clear()

        duties = self._assign_duties(state, own, foes)
        self._stage_lanes(own)

        result: dict[int, Vec2 | dict] = {}
        for drone in own:
            role, mark = duties.get(drone.id, (RUN, None))
            if role is KEEP:
                command = self._keep(drone, foes)
            elif role is HUNT:
                command = self._chase(drone, mark)
            elif role is ESCORT:
                command = self._escort(drone, mark)
            elif role is BLOCK:
                command = self._block(drone, (mark[1].id, mark[0].id))
            elif role is GUN:
                command = self._tank_move(state, drone, foes)
            else:
                command = self._goal_run(state, drone, foes)

            command = _add(command, self._engage(drone, foes, role))
            command = _add(command, self._keep_off_guns(drone, foes))
            command = _add(command, self._avoid_friends(drone, own))
            command = _add(command, self._dodge(drone, state))
            free = self._free_kill(drone, foes)
            if free is not None and role in (RUN, HUNT):
                command = _add(command, _scale(self._steer_to(drone, free), 0.5))
            command = self._rollout(drone, command, own, foes, state)
            command = self._safe_acceleration(drone, command)
            self._last_command[drone.id] = command

            fire = None
            if drone.drone_type is DroneType.TANK:
                fire = self._gunnery(drone, own, foes, state)
            if fire is not None:
                result[drone.id] = {"acceleration": command, "fire_direction": fire}
            else:
                result[drone.id] = command
        return result
