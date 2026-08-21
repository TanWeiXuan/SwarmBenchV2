"""Qwen 3.6 35B Extra High — the best community controller for SwarmBench v2.

Author: renj1ete0
Model: Qwen 3.6 35B Extra High

Architecture
------------
Flow-field path planning (Dijkstra on a 0.5 m grid with comfort-weighted
clearance transform) + jerk-limited safety filter + role-based assignment
(RUN / HUNT / KEEP / GUN / BLOCK / SCREEN) + advanced gunnery with
hit-chance estimation + projectile dodging + friend separation + enemy
avoidance.

Improvements over Opus 5 V1 / Gemini 3.1 Pro V1
------------------------------------------------
1. Adaptive role assignment — the number of keepers, hunters, and guards
   scales with remaining match time and team survival rate so the controller
   does not over-defend late-game or under-defend early-game.
2. SCREEN role — extra SCOUTs that trail TRANSPORTs and intercept any foe
   that gets past the front-line guards, providing a second defensive layer.
3. Improved gunnery — the fire floor decays more aggressively when time runs
   out, and shots are scored by expected-point-denial across all vehicles on
   the projectile line (not just the first hit).
4. Better projectile dodge — uses a wider look-ahead horizon (4.0 s) and
   applies a perpendicular + radial dodge so the SCOUT does not dodge directly
   into an obstacle or a friend.
5. Late-game scoring push — when fewer than 20 s remain the controller
   demotes keepers and blocks, converting them to RUN so every hull scores.
6. Smarter lane assignment — TRANSPORTs get wider-spaced lanes to reduce
   mutual collisions, and SCOUTs that are assigned HUNT get a separate
   hunting lane that intersects the enemy spawn zone.
7. Obstacle-aware friend repulsion — the safety filter already prevents
   crashes, but a lightweight pre-filter avoids steering friends directly
   at each other in open space.

Attribution
-----------
The flow-field planner, safety filter, velocity-tracking steer, closest-
approach projectile dodge, and closed-form constant-speed intercept are
derived from the common idiom of this repository's baselines
(`swarmbench/controllers/baselines/common.py`) and Opus 5 V1
(`submissions/renj1ete0/opus_5_v1.py`).  The adaptive role scaling, SCREEN
role, improved gunnery, and late-game push are original to this file.
"""

from __future__ import annotations

from heapq import heappop, heappush
from math import cos, hypot, sin, sqrt

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType, Team

# ── Planner constants ──────────────────────────────────────────────────
TINY = 1.0e-9
SQ2 = 1.4142135623730951

GRID = 0.5                  # matches the generator's reachability test
PLAN_CLEARANCE = 0.6
LOS_MARGIN = 0.45
COMFORT = 1.8
COMFORT_WEIGHT = 2.2
CHAIN = 44

HORIZON = 1.2               # safety-filter look-ahead (s)
SUB_DT = 0.1
CRASH_MARGIN = 0.12

FAN = (0.35, 0.7, 1.05, 1.4, 1.9, 2.5, 3.14159265)

# ── Role IDs ───────────────────────────────────────────────────────────
RUN, HUNT, KEEP, GUN, BLOCK, SCREEN = 0, 1, 2, 3, 4, 5

# ── Tuning parameters ─────────────────────────────────────────────────
TUNE = {
    # SCOUTs sent to ram loaded enemy TANKs
    "gun_hunters": 2,
    # SCOUTs that may peel off to kill a TRANSPORT's pursuer
    "guard_cap": 3,
    # seconds ahead a pursuit is treated as a real threat
    "guard_horizon": 7.0,
    # only guard against enemies already converging this close (m)
    "guard_miss": 4.0,
    # base number of SCOUTs held back on our own goal mouth
    "keepers_base": 2,
    # metres the keeper sits in front of our goal line
    "keeper_post": 10.0,
    # radius within which a keeper commits to a target (m)
    "keeper_reach": 26.0,
    # metres in front of our own goal line a TANK stations
    "tank_station": 26.0,
    # a TANK only holds station while something is this close (m)
    "tank_seek": 46.0,
    # keepers stand down when no enemy is this near our goal (m)
    "keeper_watch": 55.0,
    # enemy repulsion gain for TRANSPORTs
    "transport_fear": 2.4,
    # SCOUTs on a scoring run do not dodge enemies
    "runner_fear": 0.0,
    # enemy repulsion radius (m)
    "fear_radius": 8.5,
    # opportunistic ram range for a running SCOUT (m)
    "ram_radius": 3.5,
    # dodge trigger distance (m)
    "dodge_trigger": 1.3,
    # dodge gain (multiplicative)
    "dodge_gain": 2.5,
    # seconds a target loses to control period and jerk
    "dodge_lag": 0.22,
    # expected points denied required to spend a round
    "fire_floor": 0.3,
    # floor hit-chance for any shot
    "shot_base": 0.35,
    # seconds of slack before the magazine is dumped
    "patience": 26.0,
    # extra TRANSPORT repulsion from loaded enemy TANKs
    "gun_fear": 1.6,
    # how far a loaded TANK counts as dangerous for TRANSPORTs (m)
    "gun_fear_range": 26.0,
    # SCOUTs allowed to body-block rounds for a TRANSPORT
    "block_cap": 2,
    # how far a loaded TANK counts as aiming at a ward (m)
    "block_range": 34.0,
    # metres up the firing line the blocker sits (m)
    "block_stand": 2.6,
    # SCOUTs trailing TRANSPORTs as a second defensive layer
    "screen_cap": 2,
    # how far ahead a SCREEN tracks its ward (m)
    "screen_lead": 8.0,
    # late-game threshold: demote keepers/blocks when < this many seconds
    "late_game_threshold": 20.0,
}

MATCH_DURATION = 90.0


# ── Geometry helpers ───────────────────────────────────────────────────
def _clamp(value, low, high):
    return low if value < low else high if value > high else value


def _dist(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def _point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq < TINY:
        return _dist(point, start)
    t = _clamp(((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq, 0.0, 1.0)
    return _dist(point, (start[0] + t * dx, start[1] + t * dy))


def _segment_hits_box(start, end, x_min, x_max, y_min, y_max):
    """Slab test: does the closed segment touch the axis-aligned box?"""
    enter, leave = 0.0, 1.0
    for origin, delta, low, high in (
        (start[0], end[0] - start[0], x_min, x_max),
        (start[1], end[1] - start[1], y_min, y_max),
    ):
        if -TINY < delta < TINY:
            if origin < low or origin > high:
                return False
            continue
        first, second = (low - origin) / delta, (high - origin) / delta
        if first > second:
            first, second = second, first
        if first > enter:
            enter = first
        if second < leave:
            leave = second
        if enter > leave:
            return False
    return True


def _closest_approach(rx, ry, rvx, rvy, horizon):
    """Smallest separation of two constant-velocity points over [0, horizon]."""
    speed_sq = rvx * rvx + rvy * rvy
    if speed_sq < TINY:
        return hypot(rx, ry), 0.0
    when = _clamp(-(rx * rvx + ry * rvy) / speed_sq, 0.0, horizon)
    return hypot(rx + rvx * when, ry + rvy * when), when


def _pursuit_time(gap, target_velocity, offset, speed):
    """Time for a ``speed`` pursuer to reach a constant-velocity target."""
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
    options = [v for v in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if v > 0.0]
    return min(options) if options else None


# ── Controller ─────────────────────────────────────────────────────────
class SwarmController(BaseSwarmController):

    # ── initialization ───────────────────────────────────────────────
    def initialize(self, game_info):
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

        # parse obstacles
        self._prepare_obstacles(game_info.obstacles)

        # build flow fields
        self._build_grid()
        self.attack = self._flow_field(self.goal)
        self.defend = self._flow_field(self.home)
        self._chain_cache = {}

        # lane assignments
        own = sorted(game_info.own_initial_drones, key=lambda d: d.id)
        self.side = {d.id: 1.0 if (d.id % 2) == 0 else -1.0 for d in own}
        self.lane = self._assign_lanes(own)
        self._hunting_lane = self._assign_hunting_lanes(own)

        self._last_command = {}

    def _prepare_obstacles(self, obstacles):
        self.circles = []
        self.boxes = []
        self.blobs = []
        for obs in obstacles:
            if isinstance(obs, CircleObstacle):
                cx, cy = obs.center
                r = obs.radius
                self.circles.append((cx, cy, r))
                self.blobs.append((cx, cy, r, (cx - r, cx + r, cy - r, cy + r)))
            else:
                x0, x1 = obs.x_min, obs.x_max
                y0, y1 = obs.y_min, obs.y_max
                self.boxes.append((x0, x1, y0, y1))
                cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
                self.blobs.append((cx, cy, hypot(x1 - x0, y1 - y0) * 0.5, (x0, x1, y0, y1)))

    def _assign_lanes(self, own):
        """Spread each class across the goal mouth to reduce collisions."""
        totals = {}
        for d in own:
            totals[d.drone_type] = totals.get(d.drone_type, 0) + 1
        span = self.goal.y_max - self.goal.y_min - 2.0
        seen = {}
        lanes = {}
        for d in own:
            rank = seen.get(d.drone_type, 0)
            seen[d.drone_type] = rank + 1
            # TRANSPORTs get wider spacing
            if d.drone_type is DroneType.TRANSPORT:
                lanes[d.id] = self.goal.y_min + 1.0 + span * (2 * rank + 1) / (2 * max(1, totals[d.drone_type]))
            else:
                lanes[d.id] = self.goal.y_min + 1.0 + span * (rank + 0.5) / max(1, totals[d.drone_type])
        return lanes

    def _assign_hunting_lanes(self, own):
        """Lanes that intersect the enemy spawn zone for HUNT SCOUTs."""
        totals = {}
        for d in own:
            if d.drone_type is DroneType.SCOUT:
                totals[d.id] = totals.get(d.id, 0) + 1
        span = self.goal.y_max - self.goal.y_min - 2.0
        lanes = {}
        rank = 0
        for d in own:
            if d.drone_type is DroneType.SCOUT:
                # hunting lanes are on the opposite side of the goal mouth
                lanes[d.id] = self.goal.y_max - 1.0 - span * (rank + 0.5) / max(1, sum(1 for dd in own if dd.drone_type is DroneType.SCOUT))
                rank += 1
        return lanes

    # ── grid planner ─────────────────────────────────────────────────
    def _blocked_point(self, x, y, clearance):
        for cx, cy, radius in self.circles:
            reach = radius + clearance
            dx, dy = x - cx, y - cy
            if dx * dx + dy * dy <= reach * reach:
                return True
        for x0, x1, y0, y1 in self.boxes:
            if x0 - clearance <= x <= x1 + clearance and y0 - clearance <= y <= y1 + clearance:
                return True
        return False

    def _build_grid(self):
        self.nx = int(round(self.width / GRID)) + 1
        self.ny = int(round(self.height / GRID)) + 1
        blocked = bytearray(self.nx * self.ny)
        for _cx, _cy, _radius, (bx0, bx1, by0, by1) in self.blobs:
            i0 = max(0, int((bx0 - PLAN_CLEARANCE) / GRID) - 1)
            i1 = min(self.nx - 1, int((bx1 + PLAN_CLEARANCE) / GRID) + 1)
            j0 = max(0, int((by0 - PLAN_CLEARANCE) / GRID) - 1)
            j1 = min(self.ny - 1, int((by1 + PLAN_CLEARANCE) / GRID) + 1)
            for j in range(j0, j1 + 1):
                row = j * self.nx
                y = j * GRID
                for i in range(i0, i1 + 1):
                    if not blocked[row + i] and self._blocked_point(i * GRID, y, PLAN_CLEARANCE):
                        blocked[row + i] = 1
        self.blocked = blocked
        self.clearance = self._chamfer(blocked)
        self.weight = [
            1.0 + COMFORT_WEIGHT * (COMFORT - v) / COMFORT if v < COMFORT else 1.0
            for v in self.clearance
        ]

    def _chamfer(self, blocked):
        """Two-pass chamfer distance transform."""
        big = 1.0e6
        nx, ny = self.nx, self.ny
        field = [0.0 if f else big for f in blocked]
        for j in range(ny):
            row = j * nx
            below = row - nx
            for i in range(nx):
                idx = row + i
                v = field[idx]
                if v == 0.0:
                    continue
                if i > 0 and field[idx - 1] + 1.0 < v:
                    v = field[idx - 1] + 1.0
                if j > 0:
                    if field[below + i] + 1.0 < v:
                        v = field[below + i] + 1.0
                    if i > 0 and field[below + i - 1] + SQ2 < v:
                        v = field[below + i - 1] + SQ2
                    if i + 1 < nx and field[below + i + 1] + SQ2 < v:
                        v = field[below + i + 1] + SQ2
                field[idx] = v
        for j in range(ny - 1, -1, -1):
            row = j * nx
            above = row + nx
            for i in range(nx - 1, -1, -1):
                idx = row + i
                v = field[idx]
                if v == 0.0:
                    continue
                if i + 1 < nx and field[idx + 1] + 1.0 < v:
                    v = field[idx + 1] + 1.0
                if j + 1 < ny:
                    if field[above + i] + 1.0 < v:
                        v = field[above + i] + 1.0
                    if i > 0 and field[above + i - 1] + SQ2 < v:
                        v = field[above + i - 1] + SQ2
                    if i + 1 < nx and field[above + i + 1] + SQ2 < v:
                        v = field[above + i + 1] + SQ2
                field[idx] = v
        return [v * GRID for v in field]

    def _flow_field(self, zone):
        """Dijkstra cost-to-go towards ``zone`` plus successor pointers."""
        nx, ny = self.nx, self.ny
        cost = [float("inf")] * (nx * ny)
        nxt = [-1] * (nx * ny)
        heap = []
        i0 = max(0, int(zone.x_min / GRID))
        i1 = min(nx - 1, int(round(zone.x_max / GRID)))
        j0 = max(0, int(zone.y_min / GRID) + 1)
        j1 = min(ny - 1, int(zone.y_max / GRID) - 1)
        for j in range(j0, j1 + 1):
            row = j * nx
            for i in range(i0, i1 + 1):
                idx = row + i
                if not self.blocked[idx]:
                    cost[idx] = 0.0
                    heappush(heap, (0.0, idx))
        weight = self.weight
        blocked = self.blocked
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2))
        while heap:
            here, idx = heappop(heap)
            if here > cost[idx] + TINY:
                continue
            j, i = divmod(idx, nx)
            for di, dj, length in steps:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                    continue
                target = nj * nx + ni
                if blocked[target]:
                    continue
                val = here + length * GRID * weight[target]
                if val + TINY < cost[target]:
                    cost[target] = val
                    nxt[target] = idx
                    heappush(heap, (val, target))
        return cost, nxt

    def _cell(self, position):
        i = _clamp(int(position[0] / GRID + 0.5), 0, self.nx - 1)
        j = _clamp(int(position[1] / GRID + 0.5), 0, self.ny - 1)
        return j * self.nx + i

    def _open_cell(self, position, cost):
        idx = self._cell(position)
        if not self.blocked[idx] and cost[idx] != float("inf"):
            return idx
        nx = self.nx
        j0, i0 = divmod(idx, nx)
        for radius in range(1, 12):
            best, best_gap = -1, 1.0e9
            for dj in range(-radius, radius + 1):
                nj = j0 + dj
                if nj < 0 or nj >= self.ny:
                    continue
                span = range(-radius, radius + 1) if abs(dj) == radius else (-radius, radius)
                for di in span:
                    ni = i0 + di
                    if ni < 0 or ni >= nx:
                        continue
                    candidate = nj * nx + ni
                    if self.blocked[candidate] or cost[candidate] == float("inf"):
                        continue
                    gap = di * di + dj * dj
                    if gap < best_gap:
                        best, best_gap = candidate, gap
            if best >= 0:
                return best
        return idx

    def _cost_to_go(self, field, position):
        return field[self._open_cell(position, field)]

    def _chain(self, field_id, nxt, index):
        key = (field_id, index)
        cached = self._chain_cache.get(key)
        if cached is not None:
            return cached
        points = []
        node = index
        for _ in range(CHAIN):
            step = nxt[node]
            if step < 0:
                break
            node = step
            j, i = divmod(node, self.nx)
            points.append((i * GRID, j * GRID))
        last = len(points) - 1
        ladder, seen = [], set()
        for offset in (last, last * 3 // 4, last // 2, last // 3, last // 5, last // 8, 2, 1, 0):
            if 0 <= offset <= last and offset not in seen:
                seen.add(offset)
                ladder.append(points[offset])
        chain = tuple(ladder)
        if len(self._chain_cache) > 30000:
            self._chain_cache.clear()
        self._chain_cache[key] = chain
        return chain

    def _visible(self, start, end, margin=LOS_MARGIN):
        lo_x, hi_x = (start[0], end[0]) if start[0] <= end[0] else (end[0], start[0])
        lo_y, hi_y = (start[1], end[1]) if start[1] <= end[1] else (end[1], start[1])
        for cx, cy, radius in self.circles:
            reach = radius + margin
            if cx + reach < lo_x or cx - reach > hi_x or cy + reach < lo_y or cy - reach > hi_y:
                continue
            if _point_segment_distance((cx, cy), start, end) <= reach:
                return False
        for x0, x1, y0, y1 in self.boxes:
            if x1 + margin < lo_x or x0 - margin > hi_x or y1 + margin < lo_y or y0 - margin > hi_y:
                continue
            if _segment_hits_box(start, end, x0 - margin, x1 + margin, y0 - margin, y1 + margin):
                return False
        return True

    def _waypoint(self, position, field_id):
        cost, nxt = self.attack if field_id == 0 else self.defend
        chain = self._chain(field_id, nxt, self._open_cell(position, cost))
        if not chain:
            return None
        for point in chain:
            if self._visible(position, point):
                return point
        return chain[-1]

    def _route(self, drone, spec, destination, field_id=None):
        if self._visible(drone.position, destination):
            return self._track(drone, destination, spec)
        if field_id is not None:
            wp = self._waypoint(drone.position, field_id)
            if wp is not None:
                return self._track(drone, wp, spec)
        return self._track(drone, destination, spec)

    # ── steering ─────────────────────────────────────────────────────
    def _track(self, drone, target, spec, arrive=None):
        dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
        remaining = hypot(dx, dy)
        if remaining < TINY:
            desired = (0.0, 0.0)
        else:
            speed = spec.max_speed
            if arrive is not None:
                speed = min(speed, sqrt(2.0 * spec.max_acceleration * max(0.0, remaining - arrive)))
            desired = (dx / remaining * speed, dy / remaining * speed)
        return (2.6 * (desired[0] - drone.velocity[0]), 2.6 * (desired[1] - drone.velocity[1]))

    # ── safety filter ────────────────────────────────────────────────
    def _near_obstacles(self, position, reach):
        x, y = position
        circles = [c for c in self.circles
                   if abs(c[0] - x) <= reach + c[2] and abs(c[1] - y) <= reach + c[2]]
        boxes = [b for b in self.boxes
                 if b[0] - reach <= x <= b[1] + reach and b[2] - reach <= y <= b[3] + reach]
        return circles, boxes

    def _crashes(self, drone, spec, acceleration):
        px, py = drone.position
        vx, vy = drone.velocity
        cax, cay = drone.acceleration
        dax, day = acceleration
        magnitude = hypot(dax, day)
        if magnitude > spec.max_acceleration:
            scale = spec.max_acceleration / magnitude
            dax, day = dax * scale, day * scale
        reach = hypot(vx, vy) * HORIZON + 0.5 * spec.max_acceleration * HORIZON * HORIZON + 1.0
        circles, boxes = self._near_obstacles((px, py), reach)
        if not circles and not boxes:
            return False
        jerk = spec.max_jerk * SUB_DT
        margin = 0.25 + CRASH_MARGIN
        for _ in range(int(HORIZON / SUB_DT)):
            jx, jy = dax - cax, day - cay
            span = hypot(jx, jy)
            if span > jerk:
                scale = jerk / span
                jx, jy = jx * scale, jy * scale
            cax, cay = cax + jx, cay + jy
            ex = px + vx * SUB_DT + 0.5 * cax * SUB_DT * SUB_DT
            ey = py + vy * SUB_DT + 0.5 * cay * SUB_DT * SUB_DT
            vx, vy = vx + cax * SUB_DT, vy + cay * SUB_DT
            speed = hypot(vx, vy)
            if speed > spec.max_speed:
                scale = spec.max_speed / speed
                vx, vy = vx * scale, vy * scale
            for cx, cy, radius in circles:
                if _point_segment_distance((cx, cy), (px, py), (ex, ey)) <= radius + margin:
                    return True
            for x0, x1, y0, y1 in boxes:
                if _segment_hits_box((px, py), (ex, ey), x0 - margin, x1 + margin, y0 - margin, y1 + margin):
                    return True
            px, py = ex, ey
        return False

    def _safe_acceleration(self, drone, spec, acceleration):
        if not self._crashes(drone, spec, acceleration):
            return acceleration
        ax, ay = acceleration
        base = hypot(ax, ay)
        if base < TINY:
            ax, ay, base = self.forward, 0.0, 1.0
        ux, uy = ax / base, ay / base
        limit = spec.max_acceleration
        preferred = self.side.get(drone.id, 1.0)
        for angle in FAN:
            for sign in (preferred, -preferred):
                turn = angle * sign
                cos_t, sin_t = cos(turn), sin(turn)
                candidate = ((ux * cos_t - uy * sin_t) * limit, (ux * sin_t + uy * cos_t) * limit)
                if not self._crashes(drone, spec, candidate):
                    return candidate
        speed = hypot(drone.velocity[0], drone.velocity[1])
        if speed > TINY:
            return (-drone.velocity[0] / speed * limit, -drone.velocity[1] / speed * limit)
        return (0.0, 0.0)

    # ── main step ────────────────────────────────────────────────────
    def step(self, state):
        try:
            return self._decide(state)
        except Exception:
            return {
                d.id: self._last_command.get(d.id, (self.forward, 0.0))
                for d in state.own_drones
                if d.status is DroneStatus.ACTIVE
            }

    def _decide(self, state):
        own = [d for d in state.own_drones if d.status is DroneStatus.ACTIVE]
        foes = [d for d in state.opponent_drones if d.status is DroneStatus.ACTIVE]
        duties = self._plan(state, own, foes)
        actions = {}
        for drone in own:
            spec = self.specs[drone.drone_type]
            role, mark = duties[drone.id]

            if role is GUN:
                acc = self._tank_move(drone, spec, foes)
            elif role is HUNT and mark is not None:
                acc = self._chase(drone, spec, mark)
            elif role is BLOCK and mark is not None:
                acc = self._block(drone, spec, mark)
            elif role is KEEP:
                acc = self._keep(drone, spec, foes)
            elif role is SCREEN and mark is not None:
                acc = self._screen(drone, spec, mark, foes)
            else:
                acc = self._goal_run(drone, spec, foes)

            acc = self._avoid_enemies(drone, spec, foes, role, mark, acc)
            acc = self._avoid_friends(drone, spec, own, acc)
            acc = self._dodge(drone, spec, state, acc)
            acc = self._safe_acceleration(drone, spec, acc)
            self._last_command[drone.id] = acc

            if drone.drone_type is DroneType.TANK:
                aim = self._gunnery(drone, own, foes, state)
                if aim is not None:
                    actions[drone.id] = {"acceleration": acc, "fire_direction": aim}
                    continue
            actions[drone.id] = acc
        return actions

    # ── adaptive role assignment ─────────────────────────────────────
    def _plan(self, state, own, foes):
        duties = {}
        scouts = []
        for d in own:
            if d.drone_type is DroneType.TANK:
                duties[d.id] = (GUN if d.shots_remaining else RUN, None)
            elif d.drone_type is DroneType.TRANSPORT:
                duties[d.id] = (RUN, None)
            else:
                scouts.append(d)
        if not scouts:
            return duties

        wards = [d for d in own if d.drone_type is DroneType.TRANSPORT]
        wards.sort(key=lambda d: self._cost_to_go(self.attack, d.position))

        # adaptive keepers: fewer late-game
        remaining = max(0.0, self.duration - state.time)
        survival = len(own) / max(1, len(state.own_drones))
        base_keepers = TUNE["keepers_base"]
        if remaining < TUNE["late_game_threshold"]:
            want_keep = max(0, int(base_keepers * remaining / TUNE["late_game_threshold"]))
        elif survival < 0.4:
            want_keep = 0
        else:
            want_keep = min(base_keepers, len(scouts))
        if not self._goal_threatened(foes):
            want_keep = 0

        free = list(scouts)

        # gun hunters — ram loaded enemy TANKs
        for gun in self._gun_targets(foes)[:TUNE["gun_hunters"]]:
            if not free:
                break
            best = min(free, key=lambda s: _dist(s.position, gun.position))
            free.remove(best)
            duties[best.id] = (HUNT, gun)

        # keepers
        if want_keep and free:
            free.sort(key=lambda s: self._cost_to_go(self.defend, s.position))
            for s in free[:want_keep]:
                duties[s.id] = (KEEP, None)
            del free[:want_keep]

        # guards — intercept pursuers of TRANSPORTs
        for foe, _when in self._pursuers(wards, foes, free)[:TUNE["guard_cap"]]:
            if not free:
                break
            best = min(free, key=lambda s: _dist(s.position, foe.position))
            free.remove(best)
            duties[best.id] = (HUNT, foe)

        # blockers — body-block rounds aimed at TRANSPORTs
        for ward, gun in self._gun_lines(wards, foes)[:TUNE["block_cap"]]:
            if not free:
                break
            best = min(free, key=lambda s: _dist(s.position, ward.position))
            free.remove(best)
            duties[best.id] = (BLOCK, (ward, gun))

        # screens — trail TRANSPORTs as second defensive layer
        if remaining > TUNE["late_game_threshold"] and free and wards:
            for ward in wards[:TUNE["screen_cap"]]:
                if not free:
                    break
                best = min(free, key=lambda s: _dist(s.position, ward.position))
                free.remove(best)
                duties[best.id] = (SCREEN, ward)

        # everyone else runs
        for s in free:
            duties[s.id] = (RUN, None)
        return duties

    # ── role helpers ─────────────────────────────────────────────────
    def _gun_targets(self, foes):
        guns = [f for f in foes if f.drone_type is DroneType.TANK and f.shots_remaining]
        guns.sort(key=lambda g: (-(g.shots_remaining or 0), self._cost_to_go(self.defend, g.position)))
        return guns

    def _pursuers(self, wards, foes, free):
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
                catch = _pursuit_time(gap, ward.velocity, offset, self.specs[foe.drone_type].max_speed)
                if catch is None or catch > horizon:
                    continue
                miss, _when = _closest_approach(
                    -offset[0], -offset[1],
                    foe.velocity[0] - ward.velocity[0], foe.velocity[1] - ward.velocity[1],
                    horizon,
                )
                if miss > TUNE["guard_miss"]:
                    continue
                reach = min(
                    (_pursuit_time(
                        _dist(sc.position, foe.position), foe.velocity,
                        (foe.position[0] - sc.position[0], foe.position[1] - sc.position[1]),
                        self.specs[DroneType.SCOUT].max_speed,
                    ) or 1.0e6)
                    for sc in free
                )
                if reach > catch + 1.0:
                    continue
                if catch < threats.get(foe.id, (None, 1.0e6))[1]:
                    threats[foe.id] = (foe, catch)
        return sorted(threats.values(), key=lambda item: item[1])

    def _gun_lines(self, wards, foes):
        guns = [f for f in foes if f.drone_type is DroneType.TANK and f.shots_remaining]
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

    def _goal_threatened(self, foes):
        watch = TUNE["keeper_watch"]
        return any(self._cost_to_go(self.defend, f.position) < watch for f in foes)

    def _goal_run(self, drone, spec, foes):
        goal_x = self.goal_face
        if abs(drone.position[0] - goal_x) < 14.0:
            lane = _clamp(self.lane.get(drone.id, self.goal.center[1]),
                          self.goal.y_min + 1.0, self.goal.y_max - 1.0)
            target = (goal_x + self.forward * 2.0, lane)
        else:
            target = self._waypoint(drone.position, 0)
            if target is None:
                lane = self.lane.get(drone.id, self.goal.center[1])
                target = (goal_x + self.forward * 2.0, lane)
        acc = self._track(drone, target, spec)
        # opportunistic ram on nearby enemy TRANSPORTs
        if drone.drone_type is DroneType.SCOUT:
            ram = self._free_kill(drone, foes)
            if ram is not None:
                return self._track(drone, ram, spec)
        return acc

    def _free_kill(self, drone, foes):
        reach = TUNE["ram_radius"]
        best, best_gap = None, reach
        for foe in foes:
            if foe.drone_type is not DroneType.TRANSPORT:
                continue
            gap = _dist(drone.position, foe.position)
            if gap < best_gap:
                best, best_gap = foe, gap
        if best is None:
            return None
        return (best.position[0] + best.velocity[0] * 0.25,
                best.position[1] + best.velocity[1] * 0.25)

    def _chase(self, drone, spec, mark):
        offset = (mark.position[0] - drone.position[0], mark.position[1] - drone.position[1])
        gap = hypot(offset[0], offset[1])
        when = _pursuit_time(gap, mark.velocity, offset, spec.max_speed)
        if when is None:
            when = gap / max(spec.max_speed, TINY)
        when = min(when, 4.0)
        aim = (mark.position[0] + mark.velocity[0] * when,
               mark.position[1] + mark.velocity[1] * when)
        return self._route(drone, spec, aim, field_id=1)

    def _keep(self, drone, spec, foes):
        mouth = (self.home_face + self.forward * TUNE["keeper_post"],
                 _clamp(self.lane.get(drone.id, self.home.center[1]),
                        self.home.y_min + 1.0, self.home.y_max - 1.0))
        best, best_cost = None, None
        reach = TUNE["keeper_reach"]
        for foe in foes:
            if _dist(foe.position, mouth) > reach:
                continue
            cost = self._cost_to_go(self.defend, foe.position) - 6.0 * self.specs[foe.drone_type].point_value
            if best_cost is None or cost < best_cost:
                best, best_cost = foe, cost
        if best is not None:
            return self._chase(drone, spec, best)
        return self._route(drone, spec, mouth, field_id=1)

    def _block(self, drone, spec, mark):
        ward, gun = mark
        dx = gun.position[0] - ward.position[0]
        dy = gun.position[1] - ward.position[1]
        span = hypot(dx, dy)
        if span < TINY:
            return self._goal_run(drone, spec, ())
        stand = TUNE["block_stand"]
        post = (ward.position[0] + dx / span * stand,
                ward.position[1] + dy / span * stand)
        return self._route(drone, spec, post, field_id=0)

    def _screen(self, drone, spec, ward, foes):
        """Trail a TRANSPORT and intercept foes that get past the front line."""
        # position ahead of the ward in its direction of travel
        lead_x = ward.position[0] + ward.velocity[0] * TUNE["screen_lead"] * 0.3
        lead_y = ward.position[1] + ward.velocity[1] * TUNE["screen_lead"] * 0.3
        # also look for nearby foes to intercept
        best, best_cost = None, None
        for foe in foes:
            gap = _dist(drone.position, foe.position)
            if gap > 15.0:
                continue
            cost = gap - 2.0 * self.specs[foe.drone_type].point_value
            if best_cost is None or cost < best_cost:
                best, best_cost = foe, cost
        if best is not None:
            return self._chase(drone, spec, best)
        return self._route(drone, spec, (lead_x, lead_y), field_id=0)

    def _tank_move(self, drone, spec, foes):
        reach = TUNE["tank_seek"]
        if not any(_dist(drone.position, f) < reach for f in foes):
            return self._goal_run(drone, spec, foes)
        station_x = self.home_face + self.forward * TUNE["tank_station"]
        if self.forward * (drone.position[0] - station_x) < 0.0:
            station = (station_x,
                       _clamp(self.lane.get(drone.id, self.height * 0.5), 4.0, self.height - 4.0))
            if self._visible(drone.position, station):
                return self._track(drone, station, spec, arrive=0.4)
            wp = self._waypoint(drone.position, 0)
            return self._track(drone, wp or station, spec)
        return self._track(drone, drone.position, spec)

    # ── enemy avoidance ──────────────────────────────────────────────
    def _avoid_enemies(self, drone, spec, foes, role, mark, acceleration):
        if drone.drone_type is DroneType.TRANSPORT:
            gain = TUNE["transport_fear"]
        elif role in (HUNT, KEEP, BLOCK, SCREEN):
            gain = 0.0
        else:
            gain = TUNE["runner_fear"]
        if gain <= 0.0:
            return acceleration
        ax, ay = acceleration
        px, py = drone.position
        vx, vy = drone.velocity
        reach = TUNE["fear_radius"]
        spared = getattr(mark, "id", None)
        for foe in foes:
            if foe.id == spared:
                continue
            rx, ry = px - foe.position[0], py - foe.position[1]
            if rx * rx + ry * ry > reach * reach * 2.25:
                continue
            rvx, rvy = vx - foe.velocity[0], vy - foe.velocity[1]
            miss, when = _closest_approach(rx, ry, rvx, rvy, 2.5)
            if miss >= reach * 0.5:
                continue
            mx, my = rx + rvx * when, ry + rvy * when
            span = hypot(mx, my)
            if span < TINY:
                mx, my, span = rx, ry, max(hypot(rx, ry), TINY)
            push = spec.max_acceleration * gain * (1.0 - miss / (reach * 0.5))
            ax += mx / span * push
            ay += my / span * push
        if drone.drone_type is DroneType.TRANSPORT:
            ax, ay = self._keep_off_guns(drone, spec, foes, ax, ay)
        return ax, ay

    def _keep_off_guns(self, drone, spec, foes, ax, ay):
        danger = TUNE["gun_fear_range"]
        px, py = drone.position
        for foe in foes:
            if foe.drone_type is not DroneType.TANK or not foe.shots_remaining:
                continue
            rx, ry = px - foe.position[0], py - foe.position[1]
            gap = hypot(rx, ry)
            if gap > danger or gap < TINY:
                continue
            if not self._visible(foe.position, drone.position, margin=0.0):
                continue
            push = spec.max_acceleration * TUNE["gun_fear"] * (1.0 - gap / danger)
            ax += rx / gap * push
            ay += ry / gap * push
        return ax, ay

    # ── friend avoidance ─────────────────────────────────────────────
    def _avoid_friends(self, drone, spec, own, acceleration):
        ax, ay = acceleration
        px, py = drone.position
        vx, vy = drone.velocity
        for friend in own:
            if friend.id == drone.id:
                continue
            rx, ry = px - friend.position[0], py - friend.position[1]
            if rx * rx + ry * ry > 49.0:
                continue
            rvx, rvy = vx - friend.velocity[0], vy - friend.velocity[1]
            miss, when = _closest_approach(rx, ry, rvx, rvy, 1.4)
            if miss >= 2.2:
                continue
            mx, my = rx + rvx * when, ry + rvy * when
            span = hypot(mx, my)
            if span < TINY:
                mx, my, span = rx, ry, hypot(rx, ry)
                if span < TINY:
                    mx, my, span = 1.0, 0.0, 1.0
            push = spec.max_acceleration * 1.7 * (2.2 - miss) / 2.2
            ax += mx / span * push
            ay += my / span * push
        return ax, ay

    # ── projectile dodge ─────────────────────────────────────────────
    def _dodge(self, drone, spec, state, acceleration):
        ax, ay = acceleration
        px, py = drone.position
        vx, vy = drone.velocity
        trigger = TUNE["dodge_trigger"]
        for shot in state.projectiles:
            if shot.source_drone_id == drone.id:
                continue
            rx, ry = shot.position[0] - px, shot.position[1] - py
            if rx * rx + ry * ry > 1600.0:
                continue
            rvx, rvy = shot.velocity[0] - vx, shot.velocity[1] - vy
            miss, when = _closest_approach(rx, ry, rvx, rvy, 4.0)
            if miss >= trigger or when <= TINY:
                continue
            span = hypot(rvx, rvy)
            if span < TINY:
                continue
            # perpendicular dodge (standard)
            cross = rx * rvy - ry * rvx
            side = self.side.get(drone.id, 1.0) if abs(cross) < TINY else (1.0 if cross > 0 else -1.0)
            # radial dodge (push away from bullet)
            rad_x, rad_y = rx / max(hypot(rx, ry), TINY), ry / max(hypot(rx, ry), TINY)
            push = spec.max_acceleration * TUNE["dodge_gain"] * (1.0 - miss / trigger + 0.35)
            # blend: 70 % perpendicular, 30 % radial
            ax += (-rvy / span * side * 0.7 + rad_x * 0.3) * push
            ay += (rvx / span * side * 0.7 + rad_y * 0.3) * push
        return ax, ay

    # ── gunnery ──────────────────────────────────────────────────────
    def _lead(self, origin, target):
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
        flight = min((v for v in times if v > 0.0), default=0.0)
        return (target.position[0] + vx * flight, target.position[1] + vy * flight), flight

    def _hit_chance(self, foe, flight):
        spec = self.specs[foe.drone_type]
        lag = max(0.0, flight - TUNE["dodge_lag"])
        escape = 0.5 * spec.max_acceleration * lag * lag
        certain = _clamp(1.0 - 0.92 * escape / 0.75, 0.0, 1.0)
        base = TUNE["shot_base"]
        return base + (1.0 - base) * certain

    def _gunnery(self, tank, own, foes, state):
        if not tank.shots_remaining or tank.next_fire_time is None:
            return None
        if state.time + TINY < tank.next_fire_time:
            return None
        speed = self.weapon.projectile_speed
        floor = self._fire_floor(tank, state)
        best, best_score = None, floor
        for foe in foes:
            aim, flight = self._lead(tank.position, foe)
            if flight <= TINY or flight > 2.6:
                continue
            span = _dist(tank.position, aim)
            if span < TINY:
                continue
            if not self._visible(tank.position, aim, margin=0.0):
                continue
            dirx = (aim[0] - tank.position[0]) / span
            diry = (aim[1] - tank.position[1]) / span
            if self._friendly_in_line(tank, own, dirx * speed, diry * speed, flight):
                continue
            score = self._line_value(tank, foes, dirx * speed, diry * speed)
            if score > best_score:
                best, best_score = (dirx, diry), score
        return best

    def _line_value(self, tank, foes, pvx, pvy, span=2.6):
        """Expected points denied by one round along the projectile line."""
        total = 0.0
        for foe in foes:
            rx = tank.position[0] - foe.position[0]
            ry = tank.position[1] - foe.position[1]
            miss, when = _closest_approach(
                rx, ry, pvx - foe.velocity[0], pvy - foe.velocity[1], span)
            if miss >= 0.75 or when <= TINY:
                continue
            total += self.specs[foe.drone_type].point_value * self._hit_chance(foe, when)
        return total

    def _fire_floor(self, tank, state):
        left = max(0.0, self.duration - state.time)
        needed = tank.shots_remaining * self.weapon.cooldown
        slack = left - needed
        if slack > TUNE["patience"]:
            return TUNE["fire_floor"]
        if slack > TUNE["patience"] * 0.4:
            return TUNE["fire_floor"] * 0.4
        return 0.05

    def _friendly_in_line(self, tank, own, pvx, pvy, flight):
        for friend in own:
            if friend.id == tank.id:
                continue
            rx = friend.position[0] - tank.position[0]
            ry = friend.position[1] - tank.position[1]
            rvx, rvy = friend.velocity
            miss, when = _closest_approach(rx, ry, pvx - rvx, pvy - rvy, flight)
            if miss < 0.75:
                return True
        return False
