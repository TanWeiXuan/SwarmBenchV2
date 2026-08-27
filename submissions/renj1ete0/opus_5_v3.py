"""Opus 5 V3 - a denial-first controller for SwarmBench v2.

Written by Claude Opus 5 (Anthropic).

Lineage and attribution
-----------------------
The planner is my own ``opus_5_v1``: a 0.5 m grid at the clearance the
generator guarantees, a chamfer clearance transform, one Dijkstra sweep per
goal for a cost-to-go field and successor pointers, line-of-sight shortcutting
along that flow field, and a safety filter that replays the engine's own
jerk-limited dynamics before accepting a command.  Instrumented matches still
show 0.05 obstacle deaths per game.

The tactical layer - the value calculus, the collision policy, TRANSPORT
staging and lane assignment, the role plan and the reflexes - is my own
``opus_5_v2``, whose design notes are kept below.

Three tactical rules and one constant are taken unchanged from TanWeiXuan's
``Codex_5_6_Crossfire_v1``, which forked Opus 5 V2 and beat it: a single
proactive raider rather than three, distinct target reservation between ready
TANKs inside one control tick, reservation of the targets that live friendly
rounds already cover when the enemy fields seven or more TRANSPORTs, and a
2.8 s ceiling on flight time.  Crossfire also narrowed the valuation of a shot
to its intended target rather than to everything on the line, which is what
makes reservation mean anything; that is kept too.

What V3 changes
---------------
V2 and Crossfire mirror at 17.6-17.6 points a game out of about 45 points of
vehicles a side.  Instrumenting that mirror says where the game actually is,
per side per game:

    TRANSPORT   3.03 scored   1.75 shot   0.84 rammed   0.22 friendly
    SCOUT       1.78 scored   3.91 shot   5.91 rammed   0.59 friendly
    TANK        0.69 scored   0.22 shot   1.69 rammed   0.34 alive at 90 s

TRANSPORTs are two thirds of the value on the board and 86% of the points
actually scored, and the thing that stops them is a 20 m/s round.  So the
whole margin is in the gunnery duel, and V3 is four measured changes to it.

*A round is worth spending only on a shot the target cannot leave.*
``_hit_chance`` already modelled evasion - a vehicle sees the round, loses the
control period and its own jerk limit, and is then bounded by its class
acceleration - but ``shot_base`` credited a target that can comfortably dodge
with 0.3 of a kill anyway, and ``shot_idle`` credited a quarter of a kill for
destroying something that had already lost its race with the clock.  Both were
too generous, and the magazine drained into 44 m shots at SCOUTs.  At 0.20 and
0.10 the same five rounds go into targets that are close enough to be certain,
which is very nearly the definition of an arriving TRANSPORT.

*The guns belong at our own mouth, not in the middle.*  Once a round is only
worth firing at short flight times, the place to stand is the fourteen metres
of frontage every enemy TRANSPORT has to funnel into, so ``tank_station`` comes
back from 26 m to 12 m in front of our own goal line.  It also stops the other
leak in the table above: a 1.5 m/s vehicle parked 26 m out is on ground a
5 m/s SCOUT reaches at t=12 s, and 1.75 of three TANKs a game were being rammed
before t=30 s with two thirds of the magazine still in them.

*Dodge earlier and harder.*  ``dodge_trigger`` 1.2 m was barely wider than the
0.75 m contact disc, and ``dodge_gain`` 2.2 was diluted by whatever steer it
was added to.  A TRANSPORT accelerates at 2 m/s and needs about a second to
clear the disc, so the round that kills it was fired from 20-30 m - the exact
band the census showed - and the answer is to commit sooner.

*Rather more keepers.*  A vehicle contact resolves before a goal entry, so a
SCOUT in the mouth trades one point for whatever walks into it.
``keeper_per_value`` 2.2 to 1.7 holds one back per 1.7 points of enemy value
closing on us instead of per 2.2.

Against the thirteen rated controllers in this repository, over both sides of
fresh seeds 301-340, this scores 0.954 of the match points where Crossfire
scores 0.892; on a second disjoint set, seeds 501-540, it scores 0.940.
Head to head against Crossfire itself it takes 0.713 of the match points on
the first set and 0.831 on the second.  Head to head
the per-game census moves the way the reasoning says it should: enemy
TRANSPORTs killed by our rounds go from 1.75 to 2.77, our own TRANSPORTs lost
to theirs from 1.75 to 1.18, our TANKs alive at the whistle from 0.34 to 1.40,
and the score from 17.6-17.6 to 19.8-11.5.

Things that were measured and thrown away
-----------------------------------------
Holding the SCOUT wave out of the opening clash - the largest single leak in
the census, 7.9 of twelve SCOUTs destroyed between t=10 s and t=20 s and 3.55
of those contacts SCOUT against SCOUT for no points at all - loses badly at
every depth tried (0.60 to 0.74 against 0.81).  Forward pressure is evidently
paying for itself in ways the fate census does not show.  Re-deciding the
evasion by search instead of by summed repulsion, carrying the target's
present acceleration into the firing solution, spending the magazine early
when the gun is about to be rammed, and giving TRANSPORTs a conversion chance
calibrated separately from SCOUTs were all neutral or worse.

The Opus 5 V2 design notes follow.

What V1 got right
-----------------
V1's premise was that the biggest leak in this game is ``OBSTACLE_CRASH``, and
that a real planner removes it: a 0.5 m grid at the clearance the generator
guarantees, a chamfer clearance transform, one Dijkstra sweep per goal for a
cost-to-go field and successor pointers, line-of-sight shortcutting along that
flow field, and a safety filter that replays the engine's own jerk-limited
dynamics before accepting a command.  That was right, and it is kept here
unchanged - instrumented matches still show 0.00 obstacle deaths per game.

What V1 got wrong
-----------------
V1 also reasoned that a SCOUT-for-SCOUT contact is an even trade and therefore
never worth avoiding, so it sent everything at the goal at once.  Instrumenting
the V1 mirror shows what that produces per game, out of roughly 44 points of
vehicles on each side:

    scored                 7.2      lost to vehicle contact   24.7
    lost to projectiles   12.6      lost to the scenery        0.0

Both swarms meet near midfield at about t=8 s and annihilate: 11.0 of the 12.6
cross-team contacts happen before t=20 s, while the goals are not scored until
t=35 s onwards.  Only 0.5 of ~12 SCOUTs and 1.1 of ~6 TRANSPORTs ever reach a
goal, and 0.97 contacts per game are friendly - both vehicles lost for nothing.

The tell is that raising V1's opportunistic ram radius from 3 m to 10 m changes
its results *not at all*, on any seed.  Enemy TRANSPORTs only reach midfield at
t≈16 s, and by then every SCOUT that might have rammed one is already dead.  So
the trade was even in material and worthless in fact: a dead SCOUT denies
nothing and converts nothing.  The vehicles that decide the match are the ones
still alive at t=20 s.

V2's three changes
------------------
*Trade on value, not on class.*  ``_score_chance`` reads cost-to-go off the same
Dijkstra field the vehicles steer on and asks what a vehicle would still have
converted in the time left; ``_trade_gain`` differences that across a contact,
which kills both participants.  A SCOUT now seeks contact with an enemy
TRANSPORT (five points for one), with a loaded TANK (a body plus the magazine it
never fires), and with anything about to score on us - and side-steps a mutual
SCOUT rush, which pays nothing.  Evasion starts 15 m out, because two SCOUTs
close at 10 m/s and a 1.5 m sidestep costs the better part of a second.

*Stage the TRANSPORTs.*  Two thirds of a team's value is TRANSPORTs, and one
that crosses with the first wave is just another body in the wreck.  ``_staged``
holds a TRANSPORT back while enemies fast enough to run it down could still
intercept it mid-crossing, and releases it as the ground clears.  The wait is
bounded by the crossing itself: as soon as the remaining time only just covers
the distance still to go, it leaves regardless, so staging never trades five
certain points for a safer nothing - it only ever spends slack.

Where they wait matters as much as whether they wait.  Letting each TRANSPORT
pick the emptiest lateral band on its own walks several of them into the same
band, and since both goals sit at one latitude - they are mirrored in x only -
that band is usually the one the enemy is coming down too.  Four of seven
TRANSPORTs died in a single friendly pile-up on the staging line before
``_stage_lanes`` began claiming bands for the whole group in one pass, each
costed against the bands already taken.

*Spend the survivors.*  With SCOUTs alive past the clash, raiding an enemy
TRANSPORT is finally a move that can be executed rather than merely intended, so
raid targets are ranked by points denied per second to reach them.  Escorts
shadow our own TRANSPORTs on the bearing the threat is actually coming from,
keepers scale with the enemy value bearing down on our mouth instead of sitting
at a fixed two, and friendly separation is weighted by relative worth - a SCOUT
yields to a TRANSPORT rather than both splitting the difference - with a floor
that does not depend on closing speed, since station-keeping vehicles drift
together at almost no relative velocity at all.

Gunnery is V1's, tightened: a shot is valued by what the target would still have
scored, so the magazine is not emptied into vehicles that have already lost
their race with the clock.

Attribution: the proportional velocity-tracking steer, the closest-approach
projectile dodge and the closed-form constant-speed intercept lead are the
common idiom of this repository's built-in baselines
(``swarmbench/controllers/baselines/common.py``).  The planner, clearance
transform, safety filter and gunnery are carried over from my own ``opus_5_v1``
submission in this repository; the value calculus, collision policy, TRANSPORT
staging and role assignment are new here.

That paragraph describes V2.  V3 borrows nothing further from anyone else
beyond the four Crossfire rules named at the top of this file.
"""
from __future__ import annotations

from heapq import heappop, heappush
from math import cos, hypot, sin, sqrt

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType, Team

TINY = 1.0e-9
SQ2 = 1.4142135623730951

GRID = 0.5                 # planner cell size, matches the generator's own test
PLAN_CLEARANCE = 0.6       # generator guarantees reachability at this clearance
LOS_MARGIN = 0.45          # half-width required for a straight-line shortcut
COMFORT = 1.8              # metres of clearance below which routing is penalised
COMFORT_WEIGHT = 2.2
CHAIN = 44                 # cells of flow-field path cached per source cell

HORIZON = 1.2              # seconds of forward simulation in the safety filter
SUB_DT = 0.1
CRASH_MARGIN = 0.12        # extra metres demanded on top of the lethal radius

FAN = (0.35, 0.7, 1.05, 1.4, 1.9, 2.5, 3.14159265)

RUN, HUNT, KEEP, GUN, BLOCK, ESCORT = 0, 1, 2, 3, 4, 5

TUNE = {
    # -- value calculus -------------------------------------------------
    "chance_bias": 1.15,     # score chance at zero cost-to-go, before clamping
    "chance_span": 70.0,     # metres of cost-to-go that exhaust that chance
    "chance_floor": 0.12,    # nothing is ever completely written off
    "magazine_worth": 0.55,  # points denied per unfired round on a killed TANK

    # -- collision policy -----------------------------------------------
    "shy_accept": 0.35,      # trade gain above which a SCOUT welcomes contact
    "shy_reach_scout": 15.0, # a 10 m/s head-on closure needs this much warning
    "shy_reach_heavy": 17.0,
    "shy_gain_scout": 1.5,
    "shy_gain_heavy": 2.4,
    "shy_keeper": 0.35,      # keepers are meant to be in the way
    "shy_horizon": 3.0,
    "shy_trigger": 2.0,      # metres of predicted miss that still count as a hit
    "shy_patience": 0.45,    # how much a distant-in-time threat is discounted

    # -- roles ----------------------------------------------------------
    "raiders": 1,            # SCOUTs sent after high-value enemy vehicles
    "raid_floor": 1.2,       # expected points denied before a raid is worth it
    "raid_horizon": 22.0,    # seconds we are willing to spend catching a mark
    "raid_lag": 3.0,         # softens the value-per-second ranking
    "guard_cap": 2,          # SCOUTs that may peel off to kill a ward's pursuer
    "guard_horizon": 7.0,
    "guard_miss": 4.0,
    "guard_slack": 1.0,
    "escort_cap": 2,         # SCOUTs shadowing TRANSPORTs through open ground
    "escort_watch": 20.0,
    "escort_stand": 3.2,     # metres up the threat bearing the escort sits
    "escort_lead": 0.6,
    "escort_done": 16.0,     # a ward this close to scoring needs no escort
    "keeper_cap": 3,
    "keeper_per_value": 1.7, # enemy value per keeper held back
    "keeper_post": 10.0,
    "keeper_reach": 26.0,
    "keeper_watch": 55.0,
    "keeper_greed": 6.0,
    "block_cap": 2,
    "block_range": 34.0,
    "block_stand": 2.6,
    "chase_lead": 4.0,

    # -- TRANSPORT staging ----------------------------------------------
    "stage_watch": 1.0,      # share of the crossing an interceptor must beat
    "stage_threat": 2.0,     # interceptors still alive that justify waiting
    "stage_depth": 22.0,     # metres in front of our own goal line to wait at
    "stage_reserve": 1.25,   # safety factor on the remaining crossing time
    "stage_margin": 8.0,     # plus this many seconds of slack
    "lane_samples": 9,
    "lane_berth": 0.9,       # pull toward the band this vehicle was given
    "lane_mate": 7.0,        # cost of sharing a band with another TRANSPORT
    "lane_width": 11.0,
    "lane_crowd": 9.0,
    "lane_travel": 0.55,     # cost of the lateral move itself

    # -- opportunism ----------------------------------------------------
    "ram_radius": 14.0,
    "ram_floor": 0.9,        # trade gain required to leave the scoring line
    "ram_detour": 0.06,      # gain charged per metre of detour
    "ram_lead": 3.0,

    # -- reflexes -------------------------------------------------------
    "berth_edge": 7.0,       # metres of arena edge left free when staging
    "friend_reach": 8.0,
    "friend_floor": 3.0,     # separation held regardless of closing speed
    "friend_press": 1.5,
    "friend_trigger": 2.4,
    "friend_horizon": 1.6,
    "friend_gain": 1.7,
    "dodge_gain": 4.2,
    "dodge_trigger": 1.6,
    "dodge_reach": 42.0,
    "dodge_horizon": 3.0,
    "dodge_lag": 0.22,
    "gun_fear": 1.6,
    "gun_fear_range": 26.0,

    # -- gunnery --------------------------------------------------------
    "fire_floor": 0.55,      # expected points denied required to spend a round
    "shot_base": 0.2,        # worth of a round the target can still dodge
    "shot_span": 2.8,        # seconds of flight time a shot may take
    "shot_envelope": 1.2,    # miss distance a live round is credited with covering
    "shot_idle": 0.10,       # worth of killing something that cannot score
    "escape_weight": 0.92,
    "patience": 26.0,
    "tank_station": 12.0,    # the guns stand in our own goal mouth
    "tank_seek": 46.0,
    "friendly_line": 0.95,
}

MATCH_DURATION = 90.0          # the documented default; only used to pace play


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
        self.berth = self._assign_berths(own)
        self.now = 0.0
        self.left = self.duration
        self._chance_cache = {}
        self._lanes = {}
        self._last_command = {}
        self._claims = set()
        self._reserve_live = sum(
            drone.drone_type is DroneType.TRANSPORT
            for drone in game_info.opponent_initial_drones
        ) >= 7

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

    def _assign_berths(self, own):
        """Spread the TRANSPORTs across the arena height while they wait.

        Staged vehicles hover, so a lane preference they all shared would walk
        them into each other - and a friendly contact destroys both just as
        thoroughly as an enemy one does.
        """
        wards = [drone for drone in own if drone.drone_type is DroneType.TRANSPORT]
        span = self.height - 2.0 * TUNE["berth_edge"]
        return {
            drone.id: TUNE["berth_edge"] + span * (rank + 0.5) / max(1, len(wards))
            for rank, drone in enumerate(wards)
        }

    def _prepare_obstacles(self, obstacles):
        self.circles = []
        self.boxes = []
        self.blobs = []          # (cx, cy, bounding radius, aabb) for coarse rejection
        for obstacle in obstacles:
            if isinstance(obstacle, CircleObstacle):
                cx, cy = obstacle.center
                radius = obstacle.radius
                self.circles.append((cx, cy, radius))
                self.blobs.append((cx, cy, radius, (cx - radius, cx + radius, cy - radius, cy + radius)))
            else:
                x0, x1 = obstacle.x_min, obstacle.x_max
                y0, y1 = obstacle.y_min, obstacle.y_max
                self.boxes.append((x0, x1, y0, y1))
                cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
                self.blobs.append((cx, cy, hypot(x1 - x0, y1 - y0) * 0.5, (x0, x1, y0, y1)))

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

    # -------------------------------------------------------------- planning

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
        """Cached ladder of candidate waypoints along the flow path from a cell.

        Only the handful of points the line-of-sight probe actually tries are
        kept, which keeps the cache small enough to be irrelevant next to the
        sandbox memory limit however long the match runs.
        """
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

    def _near_obstacles(self, position, reach):
        """Obstacles whose inflated bounds sit within ``reach`` of ``position``."""
        x, y = position
        circles = [item for item in self.circles
                   if abs(item[0] - x) <= reach + item[2] and abs(item[1] - y) <= reach + item[2]]
        boxes = [item for item in self.boxes
                 if item[0] - reach <= x <= item[1] + reach and item[2] - reach <= y <= item[3] + reach]
        return circles, boxes

    def _crashes(self, drone, spec, acceleration):
        """Replay the engine's jerk-limited dynamics and test the swept path."""
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
        self._claims = set()
        self.now = state.time
        self.left = max(0.0, self.duration - self.now)
        own = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        foes = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        self._chance_cache = {}
        self._lanes = self._stage_lanes(own, foes)
        duties = self._plan(state, own, foes)
        actions = {}
        for drone in own:
            spec = self.specs[drone.drone_type]
            role, mark = duties[drone.id]
            if role is GUN:
                acceleration = self._tank_move(drone, spec, foes)
            elif role is HUNT and mark is not None:
                acceleration = self._chase(drone, spec, mark)
            elif role is ESCORT and mark is not None:
                acceleration = self._escort(drone, spec, mark, foes)
            elif role is BLOCK and mark is not None:
                acceleration = self._block(drone, spec, mark)
            elif role is KEEP:
                acceleration = self._keep(drone, spec, foes)
            else:
                acceleration = self._goal_run(drone, spec, foes, own)
            acceleration = self._engage(drone, spec, foes, role, mark, acceleration)
            acceleration = self._avoid_friends(drone, spec, own, acceleration)
            acceleration = self._dodge(drone, spec, state, acceleration)
            acceleration = self._safe_acceleration(drone, spec, acceleration)
            self._last_command[drone.id] = acceleration
            if drone.drone_type is DroneType.TANK:
                aim = self._gunnery(drone, own, foes, state)
                if aim is not None:
                    actions[drone.id] = {"acceleration": acceleration, "fire_direction": aim}
                    continue
            actions[drone.id] = acceleration
        return actions

    # -------------------------------------------------------- value calculus

    def _score_chance(self, drone, field):
        """Rough probability this vehicle still converts itself into points.

        Cost-to-go comes from the same Dijkstra field the vehicles steer on, so
        a vehicle behind a wall is correctly treated as further away than its
        straight-line distance suggests.  Anything that cannot physically reach
        the goal in the time that is left is worth nothing as a scorer, which
        is what stops SCOUTs from spending themselves on a TANK that has
        already lost its race with the clock.
        """
        key = (drone.id, field is self.attack)
        cached = self._chance_cache.get(key)
        if cached is not None:
            return cached
        spec = self.specs[drone.drone_type]
        cost = self._cost_to_go(field, drone.position)
        if cost == float("inf") or cost > spec.max_speed * self.left:
            value = 0.0
        else:
            value = _clamp(TUNE["chance_bias"] - cost / TUNE["chance_span"], TUNE["chance_floor"], 1.0)
        self._chance_cache[key] = value
        return value

    def _trade_gain(self, drone, foe):
        """Points won by destroying ``foe`` at the cost of ``drone``.

        Every vehicle contact kills both participants, so a trade is only worth
        taking when what the enemy still stood to score exceeds what we did.
        A loaded TANK carries the extra worth of the rounds it will never fire.
        """
        gain = self.specs[foe.drone_type].point_value * self._score_chance(foe, self.defend)
        gain -= self.specs[drone.drone_type].point_value * self._score_chance(drone, self.attack)
        if foe.drone_type is DroneType.TANK and foe.shots_remaining:
            gain += TUNE["magazine_worth"] * foe.shots_remaining
        return gain

    # ------------------------------------------------------------------ plan

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
        free = list(scouts)

        want_keep = min(self._keepers_wanted(foes), len(free))
        if want_keep:
            free.sort(key=lambda scout: (self._cost_to_go(self.defend, scout.position), scout.id))
            for scout in free[:want_keep]:
                duties[scout.id] = (KEEP, None)
            del free[:want_keep]

        for mark in self._raid_targets(foes, free)[:TUNE["raiders"]]:
            if not free:
                break
            best = min(free, key=lambda scout: (self._reach_time(scout, mark), scout.id))
            free.remove(best)
            duties[best.id] = (HUNT, mark)

        for foe, _when in self._pursuers(wards, foes, free)[:TUNE["guard_cap"]]:
            if not free:
                break
            best = min(free, key=lambda scout: (self._reach_time(scout, foe), scout.id))
            free.remove(best)
            duties[best.id] = (HUNT, foe)

        for ward, gun in self._gun_lines(wards, foes)[:TUNE["block_cap"]]:
            if not free:
                break
            best = min(free, key=lambda scout: (_dist(scout.position, ward.position), scout.id))
            free.remove(best)
            duties[best.id] = (BLOCK, (ward, gun))

        for ward in self._escort_wants(wards, foes)[:TUNE["escort_cap"]]:
            if not free:
                break
            best = min(free, key=lambda scout: (_dist(scout.position, ward.position), scout.id))
            free.remove(best)
            duties[best.id] = (ESCORT, ward)

        for scout in free:
            duties[scout.id] = (RUN, None)
        return duties

    def _reach_time(self, scout, mark):
        speed = self.specs[scout.drone_type].max_speed
        offset = (mark.position[0] - scout.position[0], mark.position[1] - scout.position[1])
        gap = hypot(offset[0], offset[1])
        when = _pursuit_time(gap, mark.velocity, offset, speed)
        return when if when is not None else gap / max(speed, TINY)

    def _keepers_wanted(self, foes):
        """One keeper per unit of enemy value that is genuinely closing on us.

        A vehicle contact resolves before a goal entry, so a SCOUT sitting in
        the mouth trades one point for whatever walks into it - the cheapest
        five points in the game when a TRANSPORT is the thing arriving.
        """
        watch = TUNE["keeper_watch"]
        threat = 0.0
        for foe in foes:
            cost = self._cost_to_go(self.defend, foe.position)
            if cost >= watch:
                continue
            threat += (1.0 - cost / watch) * self.specs[foe.drone_type].point_value * self._score_chance(foe, self.defend)
        return int(_clamp(threat / TUNE["keeper_per_value"], 0.0, TUNE["keeper_cap"]))

    def _raid_targets(self, foes, free):
        """Enemy vehicles worth spending a SCOUT on, richest trade first.

        This is where a surviving SCOUT earns its keep: an enemy TRANSPORT is
        five points that a one-point body can simply stand in front of.
        """
        if not free:
            return []
        picks = []
        for foe in foes:
            if foe.drone_type is DroneType.SCOUT:
                continue
            worth = self.specs[foe.drone_type].point_value * self._score_chance(foe, self.defend)
            if foe.drone_type is DroneType.TANK and foe.shots_remaining:
                worth += TUNE["magazine_worth"] * foe.shots_remaining
            if worth < TUNE["raid_floor"]:
                continue
            when = min(self._reach_time(scout, foe) for scout in free)
            if when > TUNE["raid_horizon"]:
                continue
            picks.append((-worth / (when + TUNE["raid_lag"]), foe.id, foe))
        picks.sort()
        return [foe for _rate, _id, foe in picks]

    def _pursuers(self, wards, foes, free):
        """Enemies that will reach one of our TRANSPORTs before it can score.

        A TRANSPORT cannot outrun a SCOUT, so fleeing only delays the trade; the
        answer is to spend a SCOUT of our own on the pursuer, which turns a
        five-for-one loss into an even one.
        """
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
                reach = min(self._reach_time(scout, foe) for scout in free)
                if reach > catch + TUNE["guard_slack"]:
                    continue
                if catch < threats.get(foe.id, (None, 1.0e6))[1]:
                    threats[foe.id] = (foe, catch)
        return sorted(threats.values(), key=lambda item: (item[1], item[0].id))

    def _escort_wants(self, wards, foes):
        """TRANSPORTs still deep in contested ground, most exposed first."""
        picks = []
        for ward in wards:
            if self._cost_to_go(self.attack, ward.position) < TUNE["escort_done"]:
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

    def _gun_lines(self, wards, foes):
        """(ward, tank) pairs where a loaded enemy TANK has a clear firing line."""
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

    # ------------------------------------------------------------- behaviours

    def _goal_run(self, drone, spec, foes, own=()):
        if drone.drone_type is DroneType.TRANSPORT and self._staged(drone, spec, foes):
            return self._stage(drone, spec, own, foes)
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

    def _staged(self, drone, spec, foes):
        """Should this TRANSPORT wait rather than walk into the opening clash?

        Both swarms meet in the middle inside about ten seconds and largely
        destroy each other; a TRANSPORT that arrives with them is simply one of
        the bodies.  Setting off a few seconds later crosses the same ground
        once it is empty.  The wait is always bounded by the time the crossing
        actually needs, so a staged TRANSPORT never trades five certain points
        for a safer nothing.
        """
        cost = self._cost_to_go(self.attack, drone.position)
        if cost == float("inf"):
            return False
        travel = cost / max(spec.max_speed, TINY)
        if self.left < travel * TUNE["stage_reserve"] + TUNE["stage_margin"]:
            return False
        threat = 0.0
        for foe in foes:
            if foe.drone_type is DroneType.TANK:
                continue
            if self.specs[foe.drone_type].max_speed <= spec.max_speed:
                continue                      # cannot actually run us down
            if self._reach_time(foe, drone) < travel * TUNE["stage_watch"]:
                threat += 1.0
        return threat >= TUNE["stage_threat"]

    def _stage(self, drone, spec, own, foes):
        """Hold a loose station behind the clash, in the band reserved for us."""
        post_x = self.home_face + self.forward * TUNE["stage_depth"]
        lane = self._lanes.get(drone.id, self.berth.get(drone.id, drone.position[1]))
        if self.forward * (drone.position[0] - post_x) < 0.0:
            return self._route(drone, spec, (post_x, lane), field_id=0)
        return self._track(drone, (drone.position[0], lane), spec, arrive=0.5)

    def _stage_lanes(self, own, foes):
        """One lateral band per waiting TRANSPORT, deconflicted in a single pass.

        Letting each TRANSPORT pick the emptiest band on its own walks several
        of them into the same band, and a friendly contact destroys both just
        as thoroughly as an enemy one.  Both goals sit at the same latitude -
        they are mirrored in x only - so the band they all agreed on was
        usually the one the enemy was heading down as well.  Bands are
        therefore claimed in a fixed order, each one costed against the bands
        already taken.
        """
        wards = [drone for drone in own if drone.drone_type is DroneType.TRANSPORT]
        if not wards:
            return {}
        samples = TUNE["lane_samples"]
        width = TUNE["lane_width"]
        bands = [3.0 + (self.height - 6.0) * (step + 0.5) / samples for step in range(samples)]
        crowd = []
        for y in bands:
            cost = 0.0
            for foe in foes:
                gap = abs(foe.position[1] - y)
                if gap < width:
                    cost += (1.0 - gap / width) * TUNE["lane_crowd"]
            crowd.append(cost)
        lanes = {}
        taken = []
        for ward in sorted(wards, key=lambda drone: (self.berth.get(drone.id, 0.0), drone.id)):
            berth = self.berth.get(ward.id, ward.position[1])
            best, best_cost = ward.position[1], None
            for y, base in zip(bands, crowd):
                cost = base + abs(y - ward.position[1]) * TUNE["lane_travel"]
                cost += abs(y - berth) * TUNE["lane_berth"]
                for other in taken:
                    gap = abs(other - y)
                    if gap < width:
                        cost += (1.0 - gap / width) * TUNE["lane_mate"]
                if best_cost is None or cost < best_cost:
                    best, best_cost = y, cost
            taken.append(best)
            lanes[ward.id] = best
        return lanes

    def _free_kill(self, drone, foes):
        """The richest trade a running SCOUT can take without leaving its lane."""
        reach = TUNE["ram_radius"]
        best, best_gain = None, TUNE["ram_floor"]
        for foe in foes:
            gap = _dist(drone.position, foe.position)
            if gap > reach:
                continue
            gain = self._trade_gain(drone, foe) - gap * TUNE["ram_detour"]
            if gain > best_gain:
                best, best_gain = foe, gain
        if best is None:
            return None
        when = min(self._reach_time(drone, best), TUNE["ram_lead"])
        return (best.position[0] + best.velocity[0] * when, best.position[1] + best.velocity[1] * when)

    def _chase(self, drone, spec, mark):
        offset = (mark.position[0] - drone.position[0], mark.position[1] - drone.position[1])
        gap = hypot(offset[0], offset[1])
        when = _pursuit_time(gap, mark.velocity, offset, spec.max_speed)
        if when is None:
            when = gap / max(spec.max_speed, TINY)
        when = min(when, TUNE["chase_lead"])
        aim = (mark.position[0] + mark.velocity[0] * when, mark.position[1] + mark.velocity[1] * when)
        return self._route(drone, spec, aim, field_id=1)

    def _escort(self, drone, spec, ward, foes):
        """Shadow a TRANSPORT on the side the trouble is coming from.

        Standing between the ward and the threat means an attacker meets a
        one-point body first, which is the whole difference between losing five
        points and losing one.
        """
        wx, wy = ward.position
        dx, dy = 0.0, 0.0
        watch = TUNE["escort_watch"]
        for foe in foes:
            gap = _dist(foe.position, ward.position)
            if gap > watch or gap < TINY:
                continue
            weight = (1.0 - gap / watch) * self.specs[foe.drone_type].point_value
            dx += (foe.position[0] - wx) / gap * weight
            dy += (foe.position[1] - wy) / gap * weight
        span = hypot(dx, dy)
        if span < TINY:
            dx, dy, span = self.forward, 0.0, 1.0
        stand = TUNE["escort_stand"]
        lead = TUNE["escort_lead"]
        post = (wx + dx / span * stand + ward.velocity[0] * lead,
                wy + dy / span * stand + ward.velocity[1] * lead)
        return self._route(drone, spec, post, field_id=0)

    def _keep(self, drone, spec, foes):
        """Guard the mouth of our own goal and body-block anything that arrives."""
        mouth = (self.home_face + self.forward * TUNE["keeper_post"],
                 _clamp(self.lane.get(drone.id, self.home.center[1]), self.home.y_min + 1.0, self.home.y_max - 1.0))
        best, best_cost = None, None
        reach = TUNE["keeper_reach"]
        for foe in foes:
            if _dist(foe.position, mouth) > reach:
                continue
            cost = self._cost_to_go(self.defend, foe.position)
            cost -= TUNE["keeper_greed"] * self.specs[foe.drone_type].point_value
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
        stand = TUNE["block_stand"]
        post = (ward.position[0] + dx / span * stand, ward.position[1] + dy / span * stand)
        return self._route(drone, spec, post, field_id=0)

    def _tank_move(self, drone, spec, foes):
        """Hold a firing station while there is anything to shoot, else advance."""
        reach = TUNE["tank_seek"]
        if not any(_dist(drone.position, foe.position) < reach for foe in foes):
            return self._goal_run(drone, spec, foes)
        station_x = self.home_face + self.forward * TUNE["tank_station"]
        if self.forward * (drone.position[0] - station_x) < 0.0:
            station = (station_x, _clamp(self.lane.get(drone.id, self.height * 0.5), 4.0, self.height - 4.0))
            if self._visible(drone.position, station):
                return self._track(drone, station, spec, arrive=0.4)
            waypoint = self._waypoint(drone.position, 0)
            return self._track(drone, waypoint or station, spec)
        return self._track(drone, drone.position, spec)

    # ------------------------------------------------------------- reflexes

    def _engage(self, drone, spec, foes, role, mark, acceleration):
        """Take the trades that pay and slip past the ones that do not.

        V1 ran its SCOUTs straight through the opening clash on the grounds that
        a SCOUT-for-SCOUT swap is even.  It is even in material and worthless in
        practice: both swarms annihilate inside ten seconds, and the survivors -
        not the corpses - are what convert into points afterwards.  A contact is
        now sought only when ``_trade_gain`` says it pays, and side-stepped
        otherwise, which costs a metre or so of lateral drift.
        """
        ax, ay = acceleration
        px, py = drone.position
        vx, vy = drone.velocity
        heavy = drone.drone_type is not DroneType.SCOUT
        spared = getattr(mark, "id", None)
        reach = TUNE["shy_reach_heavy"] if heavy else TUNE["shy_reach_scout"]
        gain = TUNE["shy_gain_heavy"] if heavy else TUNE["shy_gain_scout"]
        if role is KEEP:
            gain *= TUNE["shy_keeper"]
        horizon = TUNE["shy_horizon"]
        trigger = TUNE["shy_trigger"]
        for foe in foes:
            if foe.id == spared:
                continue
            rx, ry = px - foe.position[0], py - foe.position[1]
            if rx * rx + ry * ry > reach * reach:
                continue
            if not heavy and self._trade_gain(drone, foe) > TUNE["shy_accept"]:
                continue
            rvx, rvy = vx - foe.velocity[0], vy - foe.velocity[1]
            miss, when = _closest_approach(rx, ry, rvx, rvy, horizon)
            if miss >= trigger:
                continue
            mx, my = rx + rvx * when, ry + rvy * when
            span = hypot(mx, my)
            if span < TINY:
                mx, my = -rvy, rvx
                span = hypot(mx, my)
                if span < TINY:
                    mx, my, span = rx, ry, max(hypot(rx, ry), TINY)
            urgency = (1.0 - miss / trigger) * (1.0 - TUNE["shy_patience"] * when / horizon)
            push = spec.max_acceleration * gain * max(urgency, 0.0)
            ax += mx / span * push
            ay += my / span * push
        if drone.drone_type is DroneType.TRANSPORT:
            ax, ay = self._keep_off_guns(drone, spec, foes, ax, ay)
        return ax, ay

    def _keep_off_guns(self, drone, spec, foes, ax, ay):
        """A TRANSPORT inside a loaded TANK's short range cannot dodge; back off."""
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

    def _avoid_friends(self, drone, spec, own, acceleration):
        """Friendly contacts destroy both vehicles too, and buy nothing at all."""
        ax, ay = acceleration
        px, py = drone.position
        vx, vy = drone.velocity
        mine = self.specs[drone.drone_type].point_value
        reach = TUNE["friend_reach"]
        trigger = TUNE["friend_trigger"]
        for friend in own:
            if friend.id == drone.id:
                continue
            rx, ry = px - friend.position[0], py - friend.position[1]
            if rx * rx + ry * ry > reach * reach:
                continue
            rvx, rvy = vx - friend.velocity[0], vy - friend.velocity[1]
            gap_sq = rx * rx + ry * ry
            miss, when = _closest_approach(rx, ry, rvx, rvy, TUNE["friend_horizon"])
            if miss >= trigger and gap_sq >= TUNE["friend_floor"] ** 2:
                continue
            mx, my = rx + rvx * when, ry + rvy * when
            span = hypot(mx, my)
            if span < TINY:
                mx, my = -rvy, rvx
                span = hypot(mx, my)
                if span < TINY:
                    mx, my, span = 1.0, 0.0, 1.0
            # The cheaper vehicle does most of the yielding, so a TRANSPORT is
            # not shoved off its line by a SCOUT that costs a fifth as much.
            theirs = self.specs[friend.drone_type].point_value
            share = 2.0 * theirs / (mine + theirs)
            push = spec.max_acceleration * TUNE["friend_gain"] * max(0.0, 1.0 - miss / trigger) * share
            floor = TUNE["friend_floor"]
            if gap_sq < floor * floor:
                # Station-keeping vehicles drift together at almost no relative
                # speed, which the closest-approach test barely notices.
                gap = max(sqrt(gap_sq), TINY)
                push += spec.max_acceleration * TUNE["friend_press"] * (1.0 - gap / floor) * share
                if miss >= trigger:
                    mx, my, span = rx, ry, gap
            ax += mx / span * push
            ay += my / span * push
        return ax, ay

    def _dodge(self, drone, spec, state, acceleration):
        ax, ay = acceleration
        px, py = drone.position
        vx, vy = drone.velocity
        trigger = TUNE["dodge_trigger"]
        reach = TUNE["dodge_reach"]
        for shot in state.projectiles:
            if shot.source_drone_id == drone.id:
                continue
            rx, ry = shot.position[0] - px, shot.position[1] - py
            if rx * rx + ry * ry > reach * reach:
                continue
            rvx, rvy = shot.velocity[0] - vx, shot.velocity[1] - vy
            miss, when = _closest_approach(rx, ry, rvx, rvy, TUNE["dodge_horizon"])
            if miss >= trigger or when <= TINY:
                continue
            span = hypot(rvx, rvy)
            if span < TINY:
                continue
            cross = rx * rvy - ry * rvx
            side = self.side.get(drone.id, 1.0) if abs(cross) < TINY else (1.0 if cross > 0 else -1.0)
            push = spec.max_acceleration * TUNE["dodge_gain"] * (1.0 - miss / trigger + 0.35)
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
        """How much lateral room the target has to leave the 0.75 m contact disc.

        A vehicle only starts evading once it can see the round, loses about a
        fifth of a second to the control period and its own jerk limit, and is
        then bounded by its class acceleration.  TRANSPORTs are so sluggish that
        anything under roughly a second of flight is effectively unavoidable,
        which is exactly the shot worth waiting for.
        """
        spec = self.specs[foe.drone_type]
        lag = max(0.0, flight - TUNE["dodge_lag"])
        escape = 0.5 * spec.max_acceleration * lag * lag
        certain = _clamp(1.0 - TUNE["escape_weight"] * escape / 0.75, 0.0, 1.0)
        base = TUNE["shot_base"]
        return base + (1.0 - base) * certain

    def _gunnery(self, tank, own, foes, state):
        """Pick this TANK's shot, or none, without banking a body twice.

        Fifteen rounds a side decide most of the scoreboard, so the whole
        problem is not wasting them.  Targets already covered - by a round of
        ours still in the air, or by another ready TANK earlier in this same
        control tick - are struck off first (both rules are Crossfire's), and
        what is left is valued one target at a time by ``_line_value``, which
        pays only for what the target would still have scored and only in
        proportion to its inability to leave the contact disc in time.

        Between them that means the magazine goes into arriving TRANSPORTs at
        ranges they cannot dodge out of, which is what ``tank_station`` puts
        this vehicle twelve metres in front of our own goal line to arrange.
        """
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
                    miss, when = _closest_approach(
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

    def _line_value(self, tank, foes, pvx, pvy, span):
        """Expected points denied by one round along this line.

        Value is weighted by whether the target was going to score at all - a
        round spent on a TRANSPORT that has already lost its race with the clock
        denies nothing - and by whether the target can leave the contact disc
        before the round arrives.  ``foes`` is the one target the shot is
        reserved against, so two ready TANKs cannot both bank the same body.
        """
        total = 0.0
        for foe in foes:
            rx = tank.position[0] - foe.position[0]
            ry = tank.position[1] - foe.position[1]
            miss, when = _closest_approach(rx, ry, pvx - foe.velocity[0], pvy - foe.velocity[1], span)
            if miss >= 0.75 or when <= TINY:
                continue
            worth = self.specs[foe.drone_type].point_value
            worth *= TUNE["shot_idle"] + (1.0 - TUNE["shot_idle"]) * self._score_chance(foe, self.defend)
            total += worth * self._hit_chance(foe, when)
        return total

    def _fire_floor(self, tank, state):
        """Be choosy while there is time to wait, then dump the magazine."""
        slack = self.left - tank.shots_remaining * self.weapon.cooldown
        if slack > TUNE["patience"]:
            return TUNE["fire_floor"]
        if slack > TUNE["patience"] * 0.4:
            return TUNE["fire_floor"] * 0.4
        return 0.05

    def _friendly_in_line(self, tank, own, pvx, pvy, flight):
        for friend in own:
            if friend.id == tank.id:
                continue
            rx = tank.position[0] - friend.position[0]
            ry = tank.position[1] - friend.position[1]
            miss, _ = _closest_approach(rx, ry, pvx - friend.velocity[0], pvy - friend.velocity[1], flight)
            if miss < TUNE["friendly_line"]:
                return True
        return False
