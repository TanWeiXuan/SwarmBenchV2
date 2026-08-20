"""Sonnet 5 V3: engine-exact geometry, flow-field planning, phase-aware tactics.

Written by Claude Sonnet 5 (Anthropic) for SwarmBench.

Design rationale
-----------------
Two structurally different families of controller already compete on this
leaderboard: reactive potential-field steerers (Wayfinder V2, BigPickle V1,
Aegis Apex V2, Phalanx V2, and this author's own ``sonnet_5_v2``), and this
author's own ``opus_5_v1`` - a global flow-field planner with a forward-
simulated safety filter that reimplements the engine's own jerk-limited
dynamics and swept collision tests by hand.

That reimplementation is real engineering risk: a subtly wrong constant in a
hand-rolled RK4/jerk integrator or swept-segment test silently degrades every
downstream decision. This controller keeps ``opus_5_v1``'s two central ideas
- a Dijkstra flow field over a clearance-weighted grid so vehicles route
around obstacle clusters instead of getting stuck in local potential-field
minima, and a forward-simulated safety filter that vetoes any acceleration
about to cause an ``OBSTACLE_CRASH`` - but obtains both from the engine
itself. ``swarmbench.engine.dynamics.advance_dynamics`` is the exact function
the authoritative simulator calls every physics tick, and
``swarmbench.engine.collisions.swept_obstacle_contact`` /
``swept_points_contact`` are the exact continuous contact tests it resolves
events with. Importing them (both live under the ``swarmbench`` package, so
the submission sandbox's import allow-list accepts them) means the safety
filter, line-of-sight shortcut, and tank line-of-fire check all agree with
the simulator by construction rather than by careful reimplementation, and
the file stays a fraction of the size a hand-rolled version would need.

Tactically, vehicle contact of any kind (including friendly) destroys both
participants, so trading a 1-point SCOUT for a 5-point enemy TRANSPORT is a
strongly favourable swing and losing a TRANSPORT to an enemy SCOUT is the
worst trade on the board:

* SCOUTs are assigned one of five duties every step by a small central
  planner: HUNT a loaded enemy TANK (worth one body point but holding five
  rounds worth several TRANSPORTs, and it is the slowest thing on the
  board); HUNT a genuine pursuer that is on a converging course to catch a
  friendly TRANSPORT before it can reach goal (turns a five-point loss into
  an even one-for-one trade); BLOCK a loaded enemy TANK's firing line to a
  threatened TRANSPORT (a round stopped by a SCOUT costs one point instead
  of five); KEEP our own goal mouth while an enemy is still close enough to
  threaten it; or RUN for goal, opportunistically ramming any enemy
  TRANSPORT that comes within free-kill range along the way.
* Role quotas (how many SCOUTs keep, how many hunt loaded TANKs, how many
  guard against pursuers) shift with the live score differential and time
  remaining: behind, more SCOUTs raid; comfortably ahead late, more SCOUTs
  sit on defense.
* TRANSPORTs always push their lane toward goal, repelled from nearby
  enemies and additionally kept off the short-range envelope of any loaded
  enemy TANK they cannot out-accelerate away from.
* TANKs hold a firing station in front of our own goal while anything is
  within range, otherwise advance to bank their own point once their sector
  is clear. Fire control is planned centrally across all of our TANKs in one
  pass so two TANKs do not waste rounds on the same single-hit-point target:
  candidate shots are scored by expected points denied (value times a
  dodge-probability model driven by the target's own acceleration budget
  and the shot's flight time), a target already claimed by an earlier TANK
  this tick is discounted so magazines spread across threats, and a
  firing-floor threshold loosens late in the match so unused rounds get
  spent before the clock runs out.

Attribution: the flow-field/safety-filter architecture, the duty-based SCOUT
role split (HUNT/BLOCK/KEEP/RUN), the pursuit-intercept and gunnery
value/dodge-probability models are carried over and restructured from this
author's own ``submissions/renj1ete0/opus_5_v1.py``, replacing its
hand-rolled dynamics/collision math with direct calls into
``swarmbench.engine``. The phase-aware (score/time) role quotas extend an
idea introduced by BigPickle V1. The goal-seeking PD velocity tracker is the
common idiom shared by this repository's built-in baselines.
"""

from __future__ import annotations

from heapq import heappop, heappush
from math import cos, hypot, sin, sqrt

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType, Team
from swarmbench.api import DRONE_RADIUS, PROJECTILE_CONTACT_RADIUS
from swarmbench.engine.collisions import swept_obstacle_contact, swept_points_contact
from swarmbench.engine.dynamics import DynamicState, advance_dynamics

TINY = 1.0e-9
SQ2 = 1.4142135623730951

GRID = 0.5                 # planner cell size, matches the generator's own reachability test
PLAN_CLEARANCE = 0.6       # the generator guarantees reachability at this clearance
LOS_MARGIN = 0.45          # half-width required for a straight-line shortcut
COMFORT = 1.8              # metres of clearance below which routing is penalised
COMFORT_WEIGHT = 2.2
CHAIN = 44                 # cells of flow-field path cached per source cell

HORIZON_STEPS = 10         # forward-simulation steps in the safety filter
SUB_DT = 0.1                # seconds per forward-simulation step
CRASH_MARGIN = 0.05         # extra metres demanded on top of the engine's own drone radius
FAN = (0.35, 0.7, 1.05, 1.4, 1.9, 2.5, 3.14159265)

RUN, HUNT, KEEP, GUN, BLOCK = 0, 1, 2, 3, 4

BASE = {
    "gun_hunters": 1,        # SCOUTs sent to ram loaded enemy TANKs
    "guard_cap": 2,          # SCOUTs that may peel off to kill a TRANSPORT's pursuer
    "guard_horizon": 7.0,    # seconds ahead a pursuit is treated as a real threat
    "guard_miss": 4.0,       # only guard against enemies already converging this close
    "keepers": 2,            # SCOUTs held back on our own goal mouth
    "keeper_post": 10.0,     # metres the keeper sits in front of our goal line
    "keeper_reach": 26.0,    # radius within which a keeper commits to a target
    "tank_station": 26.0,    # metres in front of our own goal line
    "tank_seek": 46.0,       # a TANK only holds station while something is this close
    "keeper_watch": 55.0,    # keepers stand down when no enemy is this near our goal
    "transport_fear": 2.4,   # enemy repulsion gain for TRANSPORTs
    "runner_fear": 0.0,      # SCOUTs on a scoring run do not dodge enemies: an even
                              # SCOUT-for-SCOUT trade removes a threat to a TRANSPORT
    "fear_radius": 7.0,
    "ram_radius": 3.0,       # opportunistic ram range for a running SCOUT
    "dodge_gain": 2.2,
    "dodge_trigger": 1.2,
    "dodge_lag": 0.22,       # seconds a target loses to control period and jerk
    "fire_floor": 0.3,       # expected points denied required to spend a round
    "patience": 26.0,        # seconds of slack before the magazine gets dumped
    "gun_fear": 1.6,         # extra TRANSPORT repulsion from loaded enemy TANKs
    "gun_fear_range": 26.0,
    "block_cap": 2,          # SCOUTs allowed to body-block rounds for a TRANSPORT
    "block_range": 34.0,     # how far a loaded TANK counts as aiming at a ward
    "block_stand": 2.6,      # metres up the firing line the blocker sits
    "claim_discount": 0.35,  # residual value of a target another TANK claimed this tick
}

MATCH_DURATION = 90.0


def _clamp(value, low, high):
    return low if value < low else high if value > high else value


def _dist(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def _closest_approach(rx, ry, rvx, rvy, horizon):
    """Smallest separation of two constant-velocity points over ``[0, horizon]``."""
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
    options = [value for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if value > 0.0]
    return min(options) if options else None


class SwarmController(BaseSwarmController):
    # ------------------------------------------------------------------ setup

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

        self._prepare_obstacles(game_info.obstacles)
        self._build_grid()
        self.attack = self._flow_field(self.goal)
        self.defend = self._flow_field(self.home)
        self._chain_cache = {}

        own = sorted(game_info.own_initial_drones, key=lambda drone: drone.id)
        self.side = {drone.id: 1.0 if (drone.id % 2) == 0 else -1.0 for drone in own}
        self.lane = self._assign_lanes(own)
        self._last_command = {}

    def _assign_lanes(self, own):
        """Spread each class across the 14 m goal mouth so arrivals do not collide."""
        totals = {}
        for drone in own:
            totals[drone.drone_type] = totals.get(drone.drone_type, 0) + 1
        span = self.goal.y_max - self.goal.y_min - 2.0
        seen = {}
        lanes = {}
        for drone in own:
            rank = seen.get(drone.drone_type, 0)
            seen[drone.drone_type] = rank + 1
            lanes[drone.id] = self.goal.y_min + 1.0 + span * (rank + 0.5) / max(1, totals[drone.drone_type])
        return lanes

    def _prepare_obstacles(self, obstacles):
        """Cheap (center, bounding-radius) blobs for fast rejection before an
        exact ``swept_obstacle_contact`` test against the real shape."""
        self.obstacles = obstacles
        self.blobs = []
        for obstacle in obstacles:
            if isinstance(obstacle, CircleObstacle):
                cx, cy = obstacle.center
                bound = obstacle.radius
            else:
                cx = (obstacle.x_min + obstacle.x_max) * 0.5
                cy = (obstacle.y_min + obstacle.y_max) * 0.5
                bound = hypot(obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min) * 0.5
            self.blobs.append((obstacle, cx, cy, bound))

    def _blocked_point(self, x, y, clearance):
        point = (x, y)
        for obstacle, cx, cy, bound in self.blobs:
            reach = bound + clearance
            if abs(x - cx) > reach or abs(y - cy) > reach:
                continue
            if swept_obstacle_contact(point, point, obstacle, clearance) is not None:
                return True
        return False

    def _near_blobs(self, position, reach):
        x, y = position
        return [
            obstacle
            for obstacle, cx, cy, bound in self.blobs
            if abs(x - cx) <= reach + bound and abs(y - cy) <= reach + bound
        ]

    def _clear(self, start, end, margin):
        """Exact, engine-identical test: does the segment touch any obstacle?"""
        x0, y0 = start
        x1, y1 = end
        lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)
        lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
        for obstacle, cx, cy, bound in self.blobs:
            reach = bound + margin
            if cx + reach < lo_x or cx - reach > hi_x or cy + reach < lo_y or cy - reach > hi_y:
                continue
            if swept_obstacle_contact(start, end, obstacle, margin) is not None:
                return False
        return True

    # -------------------------------------------------------------- planning

    def _build_grid(self):
        self.nx = int(round(self.width / GRID)) + 1
        self.ny = int(round(self.height / GRID)) + 1
        blocked = bytearray(self.nx * self.ny)
        for obstacle, cx, cy, bound in self.blobs:
            reach = bound + PLAN_CLEARANCE
            i0 = max(0, int((cx - reach) / GRID) - 1)
            i1 = min(self.nx - 1, int((cx + reach) / GRID) + 1)
            j0 = max(0, int((cy - reach) / GRID) - 1)
            j1 = min(self.ny - 1, int((cy + reach) / GRID) + 1)
            for j in range(j0, j1 + 1):
                row = j * self.nx
                y = j * GRID
                for i in range(i0, i1 + 1):
                    if not blocked[row + i] and self._blocked_point(i * GRID, y, PLAN_CLEARANCE):
                        blocked[row + i] = 1
        self.blocked = blocked
        self.clearance = self._chamfer(blocked)
        self.weight = [
            1.0 + COMFORT_WEIGHT * (COMFORT - value) / COMFORT if value < COMFORT else 1.0
            for value in self.clearance
        ]

    def _chamfer(self, blocked):
        """Two-pass approximate distance (metres) from each cell to blocked space."""
        big = 1.0e6
        nx, ny = self.nx, self.ny
        field = [0.0 if flag else big for flag in blocked]
        for j in range(ny):
            row = j * nx
            below = row - nx
            for i in range(nx):
                index = row + i
                value = field[index]
                if value == 0.0:
                    continue
                if i > 0 and field[index - 1] + 1.0 < value:
                    value = field[index - 1] + 1.0
                if j > 0:
                    if field[below + i] + 1.0 < value:
                        value = field[below + i] + 1.0
                    if i > 0 and field[below + i - 1] + SQ2 < value:
                        value = field[below + i - 1] + SQ2
                    if i + 1 < nx and field[below + i + 1] + SQ2 < value:
                        value = field[below + i + 1] + SQ2
                field[index] = value
        for j in range(ny - 1, -1, -1):
            row = j * nx
            above = row + nx
            for i in range(nx - 1, -1, -1):
                index = row + i
                value = field[index]
                if value == 0.0:
                    continue
                if i + 1 < nx and field[index + 1] + 1.0 < value:
                    value = field[index + 1] + 1.0
                if j + 1 < ny:
                    if field[above + i] + 1.0 < value:
                        value = field[above + i] + 1.0
                    if i > 0 and field[above + i - 1] + SQ2 < value:
                        value = field[above + i - 1] + SQ2
                    if i + 1 < nx and field[above + i + 1] + SQ2 < value:
                        value = field[above + i + 1] + SQ2
                field[index] = value
        return [value * GRID for value in field]

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
                index = row + i
                if not self.blocked[index]:
                    cost[index] = 0.0
                    heappush(heap, (0.0, index))
        weight = self.weight
        blocked = self.blocked
        steps = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2))
        while heap:
            here, index = heappop(heap)
            if here > cost[index] + TINY:
                continue
            j, i = divmod(index, nx)
            for di, dj, length in steps:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                    continue
                target = nj * nx + ni
                if blocked[target]:
                    continue
                value = here + length * GRID * weight[target]
                if value + TINY < cost[target]:
                    cost[target] = value
                    nxt[target] = index
                    heappush(heap, (value, target))
        return cost, nxt

    def _cell(self, position):
        i = _clamp(int(position[0] / GRID + 0.5), 0, self.nx - 1)
        j = _clamp(int(position[1] / GRID + 0.5), 0, self.ny - 1)
        return j * self.nx + i

    def _open_cell(self, position, cost):
        """Nearest cell that is both free and connected to the destination."""
        index = self._cell(position)
        if not self.blocked[index] and cost[index] != float("inf"):
            return index
        nx = self.nx
        j0, i0 = divmod(index, nx)
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
        return index

    def _cost_to_go(self, field, position):
        cost = field[0]
        return cost[self._open_cell(position, cost)]

    def _chain(self, field_id, nxt, index):
        """Cached ladder of candidate waypoints along the flow path from a cell."""
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
        return self._clear(start, end, margin)

    def _waypoint(self, position, field_id):
        """Furthest visible point along the flow-field path from ``position``."""
        cost, nxt = self.attack if field_id == 0 else self.defend
        chain = self._chain(field_id, nxt, self._open_cell(position, cost))
        if not chain:
            return None
        for point in chain:
            if self._visible(position, point):
                return point
        return chain[-1]

    def _route(self, drone, spec, destination, field_id=None):
        """Steer straight at ``destination`` when visible, otherwise via the field."""
        if self._visible(drone.position, destination):
            return self._track(drone, destination, spec)
        if field_id is not None:
            waypoint = self._waypoint(drone.position, field_id)
            if waypoint is not None:
                return self._track(drone, waypoint, spec)
        return self._track(drone, destination, spec)

    # -------------------------------------------------------------- steering

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

    def _crashes(self, drone, spec, acceleration):
        """Forward-simulate the engine's own dynamics and test the swept path."""
        vx, vy = drone.velocity
        magnitude = hypot(acceleration[0], acceleration[1])
        capped = acceleration
        if magnitude > spec.max_acceleration:
            scale = spec.max_acceleration / magnitude
            capped = (acceleration[0] * scale, acceleration[1] * scale)
        travel = hypot(vx, vy) * (HORIZON_STEPS * SUB_DT) + 0.5 * spec.max_acceleration * (HORIZON_STEPS * SUB_DT) ** 2 + 1.0
        near = self._near_blobs(drone.position, travel)
        if not near:
            return False
        margin = DRONE_RADIUS + CRASH_MARGIN
        state = DynamicState(drone.position, drone.velocity, drone.acceleration)
        for _ in range(HORIZON_STEPS):
            nxt = advance_dynamics(state, capped, spec, SUB_DT)
            for obstacle in near:
                if swept_obstacle_contact(state.position, nxt.position, obstacle, margin) is not None:
                    return True
            state = nxt
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

    # ------------------------------------------------------------------ step

    def step(self, state):
        try:
            return self._decide(state)
        except Exception:
            return {
                drone.id: self._last_command.get(drone.id, (self.forward, 0.0))
                for drone in state.own_drones
                if drone.status is DroneStatus.ACTIVE
            }

    def _decide(self, state):
        own = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        foes = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        duties = self._plan(state, own, foes)
        tanks = [drone for drone in own if drone.drone_type is DroneType.TANK]
        fire_plan = self._plan_fire(tanks, own, foes, state)
        actions = {}
        for drone in own:
            spec = self.specs[drone.drone_type]
            role, mark = duties[drone.id]
            if role is GUN:
                acceleration = self._tank_move(drone, spec, foes)
            elif role is HUNT and mark is not None:
                acceleration = self._chase(drone, spec, mark)
            elif role is BLOCK and mark is not None:
                acceleration = self._block(drone, spec, mark)
            elif role is KEEP:
                acceleration = self._keep(drone, spec, foes)
            else:
                acceleration = self._goal_run(drone, spec, foes)
            acceleration = self._avoid_enemies(drone, spec, foes, role, mark, acceleration)
            acceleration = self._avoid_friends(drone, spec, own, acceleration)
            acceleration = self._dodge(drone, spec, state, acceleration)
            acceleration = self._safe_acceleration(drone, spec, acceleration)
            self._last_command[drone.id] = acceleration
            aim = fire_plan.get(drone.id)
            if aim is not None:
                actions[drone.id] = {"acceleration": acceleration, "fire_direction": aim}
            else:
                actions[drone.id] = acceleration
        return actions

    # ------------------------------------------------------------------ plan

    def _phase_quotas(self, num_scouts, state):
        """Scale role quotas with the live score differential and time left."""
        score_diff = state.own_score - state.opponent_score
        time_left = max(0.0, self.duration - state.time)
        keepers = BASE["keepers"]
        gun_hunters = BASE["gun_hunters"]
        guard_cap = BASE["guard_cap"]
        block_cap = BASE["block_cap"]
        if score_diff < 0 and time_left > 12.0:
            keepers = max(0, keepers - 1)
            gun_hunters = min(num_scouts, gun_hunters + 1)
        if score_diff > 0 and time_left < 25.0:
            keepers = min(num_scouts, keepers + 1)
            guard_cap = min(num_scouts, guard_cap + 1)
        if score_diff > 0 and time_left < 12.0:
            gun_hunters = 0
        return keepers, gun_hunters, guard_cap, block_cap

    def _plan(self, state, own, foes):
        """Assign one duty per vehicle: the value game decided centrally."""
        duties = {}
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
        wards.sort(key=lambda drone: self._cost_to_go(self.attack, drone.position))

        keepers, gun_hunters, guard_cap, block_cap = self._phase_quotas(len(scouts), state)
        want_keep = min(keepers, len(scouts))
        if not self._goal_threatened(foes):
            want_keep = 0

        free = list(scouts)
        for gun in self._gun_targets(foes)[:gun_hunters]:
            if not free:
                break
            best = min(free, key=lambda scout: _dist(scout.position, gun.position))
            free.remove(best)
            duties[best.id] = (HUNT, gun)

        if want_keep and free:
            free.sort(key=lambda scout: self._cost_to_go(self.defend, scout.position))
            for scout in free[:want_keep]:
                duties[scout.id] = (KEEP, None)
            del free[:want_keep]

        for foe, _when in self._pursuers(wards, foes, free)[:guard_cap]:
            if not free:
                break
            best = min(free, key=lambda scout: _dist(scout.position, foe.position))
            free.remove(best)
            duties[best.id] = (HUNT, foe)

        for ward, gun in self._gun_lines(wards, foes)[:block_cap]:
            if not free:
                break
            best = min(free, key=lambda scout: _dist(scout.position, ward.position))
            free.remove(best)
            duties[best.id] = (BLOCK, (ward, gun))

        for scout in free:
            duties[scout.id] = (RUN, None)
        return duties

    def _gun_targets(self, foes):
        """Loaded enemy TANKs, nearest to our own goal first."""
        guns = [foe for foe in foes if foe.drone_type is DroneType.TANK and foe.shots_remaining]
        guns.sort(key=lambda gun: (-(gun.shots_remaining or 0), self._cost_to_go(self.defend, gun.position)))
        return guns

    def _pursuers(self, wards, foes, free):
        """Enemies that will reach one of our TRANSPORTs before we reach the goal."""
        if not wards or not free:
            return []
        horizon = BASE["guard_horizon"]
        scout_speed = self.specs[DroneType.SCOUT].max_speed
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
                if miss > BASE["guard_miss"]:
                    continue
                reach = min(
                    (_pursuit_time(
                        _dist(scout.position, foe.position),
                        foe.velocity,
                        (foe.position[0] - scout.position[0], foe.position[1] - scout.position[1]),
                        scout_speed,
                    ) or 1.0e6)
                    for scout in free
                )
                if reach > catch + 1.0:
                    continue
                if catch < threats.get(foe.id, (None, 1.0e6))[1]:
                    threats[foe.id] = (foe, catch)
        return sorted(threats.values(), key=lambda item: item[1])

    def _gun_lines(self, wards, foes):
        """(ward, tank) pairs where a loaded enemy TANK has a clear firing line."""
        guns = [foe for foe in foes if foe.drone_type is DroneType.TANK and foe.shots_remaining]
        if not guns:
            return []
        reach = BASE["block_range"]
        lines = []
        for ward in wards:
            best, best_gap = None, reach
            for gun in guns:
                gap = _dist(gun.position, ward.position)
                if gap < best_gap and self._clear(gun.position, ward.position, 0.0):
                    best, best_gap = gun, gap
            if best is not None:
                lines.append((ward, best))
        return lines

    def _goal_threatened(self, foes):
        watch = BASE["keeper_watch"]
        return any(self._cost_to_go(self.defend, foe.position) < watch for foe in foes)

    def _goal_run(self, drone, spec, foes):
        if abs(drone.position[0] - self.goal_face) < 14.0:
            lane = _clamp(self.lane.get(drone.id, self.goal.center[1]), self.goal.y_min + 1.0, self.goal.y_max - 1.0)
            target = (self.goal_face + self.forward * 2.0, lane)
        else:
            target = self._waypoint(drone.position, 0)
            if target is None:
                target = (self.goal_face + self.forward * 2.0, self.goal.center[1])
        acceleration = self._track(drone, target, spec)
        if drone.drone_type is DroneType.SCOUT:
            ram = self._free_kill(drone, foes)
            if ram is not None:
                return self._track(drone, ram, spec)
        return acceleration

    def _free_kill(self, drone, foes):
        """A high-value enemy close enough that ramming barely costs progress."""
        reach = BASE["ram_radius"]
        best, best_gap = None, reach
        for foe in foes:
            if foe.drone_type is not DroneType.TRANSPORT:
                continue
            gap = _dist(drone.position, foe.position)
            if gap < best_gap:
                best, best_gap = foe, gap
        if best is None:
            return None
        return (best.position[0] + best.velocity[0] * 0.25, best.position[1] + best.velocity[1] * 0.25)

    def _chase(self, drone, spec, mark):
        offset = (mark.position[0] - drone.position[0], mark.position[1] - drone.position[1])
        gap = hypot(offset[0], offset[1])
        when = _pursuit_time(gap, mark.velocity, offset, spec.max_speed)
        if when is None:
            when = gap / max(spec.max_speed, TINY)
        when = min(when, 4.0)
        aim = (mark.position[0] + mark.velocity[0] * when, mark.position[1] + mark.velocity[1] * when)
        return self._route(drone, spec, aim, field_id=1)

    def _keep(self, drone, spec, foes):
        """Guard the mouth of our own goal and body-block anything that arrives."""
        mouth = (self.home_face + self.forward * BASE["keeper_post"],
                 _clamp(self.lane.get(drone.id, self.home.center[1]), self.home.y_min + 1.0, self.home.y_max - 1.0))
        best, best_cost = None, None
        reach = BASE["keeper_reach"]
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
        """Stand on the firing line: a round stopped by a SCOUT costs one point."""
        ward, gun = mark
        dx = gun.position[0] - ward.position[0]
        dy = gun.position[1] - ward.position[1]
        span = hypot(dx, dy)
        if span < TINY:
            return self._goal_run(drone, spec, ())
        stand = BASE["block_stand"]
        post = (ward.position[0] + dx / span * stand, ward.position[1] + dy / span * stand)
        return self._route(drone, spec, post, field_id=0)

    def _tank_move(self, drone, spec, foes):
        """Hold a firing station while there is anything to shoot, else advance."""
        reach = BASE["tank_seek"]
        if not any(_dist(drone.position, foe.position) < reach for foe in foes):
            return self._goal_run(drone, spec, foes)
        station_x = self.home_face + self.forward * BASE["tank_station"]
        if self.forward * (drone.position[0] - station_x) < 0.0:
            station = (station_x, _clamp(self.lane.get(drone.id, self.height * 0.5), 4.0, self.height - 4.0))
            if self._visible(drone.position, station):
                return self._track(drone, station, spec, arrive=0.4)
            waypoint = self._waypoint(drone.position, 0)
            return self._track(drone, waypoint or station, spec)
        return self._track(drone, drone.position, spec)

    # ------------------------------------------------------------ reflexes

    def _avoid_enemies(self, drone, spec, foes, role, mark, acceleration):
        if drone.drone_type is DroneType.TRANSPORT:
            gain = BASE["transport_fear"]
        elif role in (HUNT, KEEP, BLOCK):
            gain = 0.0
        else:
            gain = BASE["runner_fear"]
        if gain <= 0.0:
            ax, ay = acceleration
        else:
            ax, ay = acceleration
            px, py = drone.position
            vx, vy = drone.velocity
            reach = BASE["fear_radius"]
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
        """A TRANSPORT inside a loaded TANK's short range cannot dodge; back off."""
        danger = BASE["gun_fear_range"]
        px, py = drone.position
        for foe in foes:
            if foe.drone_type is not DroneType.TANK or not foe.shots_remaining:
                continue
            rx, ry = px - foe.position[0], py - foe.position[1]
            gap = hypot(rx, ry)
            if gap > danger or gap < TINY:
                continue
            if not self._clear(foe.position, drone.position, 0.0):
                continue
            push = spec.max_acceleration * BASE["gun_fear"] * (1.0 - gap / danger)
            ax += rx / gap * push
            ay += ry / gap * push
        return ax, ay

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

    def _dodge(self, drone, spec, state, acceleration):
        ax, ay = acceleration
        px, py = drone.position
        vx, vy = drone.velocity
        trigger = BASE["dodge_trigger"]
        for shot in state.projectiles:
            if shot.source_drone_id == drone.id:
                continue
            rx, ry = shot.position[0] - px, shot.position[1] - py
            if rx * rx + ry * ry > 1600.0:
                continue
            rvx, rvy = shot.velocity[0] - vx, shot.velocity[1] - vy
            miss, when = _closest_approach(rx, ry, rvx, rvy, 3.0)
            if miss >= trigger or when <= TINY:
                continue
            span = hypot(rvx, rvy)
            if span < TINY:
                continue
            cross = rx * rvy - ry * rvx
            side = self.side.get(drone.id, 1.0) if abs(cross) < TINY else (1.0 if cross > 0 else -1.0)
            push = spec.max_acceleration * BASE["dodge_gain"] * (1.0 - miss / trigger + 0.35)
            ax += -rvy / span * side * push
            ay += rvx / span * side * push
        return ax, ay

    # --------------------------------------------------------------- gunnery

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
        flight = min((value for value in times if value > 0.0), default=0.0)
        return (target.position[0] + vx * flight, target.position[1] + vy * flight), flight

    def _hit_chance(self, foe, flight):
        """How much lateral room the target has to leave the 0.75 m contact disc."""
        spec = self.specs[foe.drone_type]
        lag = max(0.0, flight - BASE["dodge_lag"])
        escape = 0.5 * spec.max_acceleration * lag * lag
        certain = _clamp(1.0 - 0.92 * escape / PROJECTILE_CONTACT_RADIUS, 0.0, 1.0)
        base = 0.35
        return base + (1.0 - base) * certain

    def _plan_fire(self, tanks, own, foes, state):
        """Choose one target per TANK in a single pass so magazines spread
        across threats instead of two TANKs wasting rounds on one target."""
        result = {}
        claimed = set()
        speed = self.weapon.projectile_speed
        for tank in sorted(tanks, key=lambda drone: drone.id):
            if not tank.shots_remaining or tank.next_fire_time is None or state.time + TINY < tank.next_fire_time:
                continue
            floor = self._fire_floor(tank, state)
            best_dir, best_score, best_hits = None, floor, ()
            for foe in foes:
                aim, flight = self._lead(tank.position, foe)
                if flight <= 0.0 or flight > 2.6:
                    continue
                span = _dist(tank.position, aim)
                if span < TINY:
                    continue
                if not self._clear(tank.position, aim, 0.0):
                    continue
                dirx, diry = (aim[0] - tank.position[0]) / span, (aim[1] - tank.position[1]) / span
                pvx, pvy = dirx * speed, diry * speed
                if self._friendly_in_line(tank, own, pvx, pvy, flight):
                    continue
                score, hits = self._line_value(tank, foes, pvx, pvy, claimed)
                if score > best_score:
                    best_dir, best_score, best_hits = (dirx, diry), score, hits
            if best_dir is not None:
                result[tank.id] = best_dir
                claimed.update(best_hits)
        return result

    def _line_value(self, tank, foes, pvx, pvy, claimed, span=2.6):
        """Expected points denied by one round, counting everything on the line."""
        total = 0.0
        hits = []
        for foe in foes:
            rx = tank.position[0] - foe.position[0]
            ry = tank.position[1] - foe.position[1]
            miss, when = _closest_approach(rx, ry, pvx - foe.velocity[0], pvy - foe.velocity[1], span)
            if miss >= PROJECTILE_CONTACT_RADIUS or when <= TINY:
                continue
            weight = BASE["claim_discount"] if foe.id in claimed else 1.0
            total += self.specs[foe.drone_type].point_value * self._hit_chance(foe, when) * weight
            hits.append(foe.id)
        return total, tuple(hits)

    def _fire_floor(self, tank, state):
        """Be choosy while there is time to wait, then dump the magazine."""
        left = max(0.0, self.duration - state.time)
        needed = tank.shots_remaining * self.weapon.cooldown
        slack = left - needed
        if slack > BASE["patience"]:
            return BASE["fire_floor"]
        if slack > BASE["patience"] * 0.4:
            return BASE["fire_floor"] * 0.4
        return 0.05

    def _friendly_in_line(self, tank, own, pvx, pvy, flight):
        proj_end = (tank.position[0] + pvx * flight, tank.position[1] + pvy * flight)
        for friend in own:
            if friend.id == tank.id or friend.status is not DroneStatus.ACTIVE:
                continue
            friend_end = (friend.position[0] + friend.velocity[0] * flight, friend.position[1] + friend.velocity[1] * flight)
            if swept_points_contact(tank.position, proj_end, friend.position, friend_end, PROJECTILE_CONTACT_RADIUS) is not None:
                return True
        return False
