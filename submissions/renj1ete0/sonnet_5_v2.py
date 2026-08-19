"""Claude Sonnet 5 V2: phase-aware roles, global defensive assignment, exact
obstacle/line-of-fire geometry, and value-weighted ramming.

Strategy summary
-----------------
Vehicle-vehicle contact of any kind destroys both participants, so a cheap
SCOUT (1 pt) trading into an enemy TRANSPORT (5 pt) is a strongly favorable
swing, while losing our own TRANSPORT to an enemy SCOUT is the worst trade on
the board. This controller leans on that asymmetry:

* SCOUT roles are recomputed every ~0.5 s into three groups: "defenders" are
  matched one-to-one against the enemies most threatening our own goal via a
  Hungarian (min-cost bipartite) assignment; "raiders" hunt the enemy
  TRANSPORT that is furthest advanced toward our goal on a predictive
  intercept course; the remainder run goal lanes but divert onto any nearby
  high-value target that presents a cheap trade. Role quotas shift with score
  differential and time remaining: behind, more scouts raid; comfortably
  ahead late, more scouts defend.
* TRANSPORT always pushes its lane toward the goal while actively fleeing any
  nearby enemy, since a lost transport is a 5-point swing.
* TANK holds a defensive line on our own half (or trails just behind our
  lead transport once it has pushed ahead), tracks the local center of mass
  of nearby enemies for sightlines, and fires with a closed-form intercept
  lead the instant a valuable target is in a clear, friendly-fire-free line.

All drones share obstacle avoidance (exact nearest-point repulsion plus a
bounding-circle lookahead deflection), friendly separation, and incoming
projectile dodging.

Attribution: the goal-seeking PD steer, bounding-circle lookahead deflection,
friendly-separation, and projectile-dodge terms follow the shared steering
idiom used across this repository's built-in baselines and its top-rated
community controllers (e.g. Wayfinder V2, BigPickle V1); this file
reimplements that idiom independently and adds exact nearest-point obstacle
repulsion plus exact segment/rectangle line-of-fire checks (in the spirit of
BigPickle V1's rectangle-aware blocking). The phase/score-aware scout quotas
extend an idea introduced by BigPickle V1. The Hungarian-assignment
defensive screen and opportunistic ramming carry over from this author's own
Sonnet 5 V1 (submissions/TanWeiXuan/sonnet_5_v1.py).
"""

from __future__ import annotations

from math import hypot, sqrt

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType, Team

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy is a required dependency, but stay defensive
    linear_sum_assignment = None


EPSILON = 1.0e-9


def _distance(left, right):
    return hypot(left[0] - right[0], left[1] - right[1])


def _clamp(value, low, high):
    return low if value < low else high if value > high else value


def _obstacle_bounds(obstacle):
    """Bounding (center, radius) used for the longer-range lookahead deflection."""
    if isinstance(obstacle, CircleObstacle):
        return obstacle.center, obstacle.radius
    center = ((obstacle.x_min + obstacle.x_max) * 0.5, (obstacle.y_min + obstacle.y_max) * 0.5)
    radius = hypot(obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min) * 0.5
    return center, radius


def _nearest_point_on_obstacle(point, obstacle):
    """Exact closest point on the true obstacle shape, for short-range repulsion."""
    if isinstance(obstacle, CircleObstacle):
        cx, cy = obstacle.center
        dx, dy = point[0] - cx, point[1] - cy
        d = hypot(dx, dy)
        if d < EPSILON:
            return (cx + obstacle.radius, cy)
        return (cx + dx / d * obstacle.radius, cy + dy / d * obstacle.radius)
    return (
        _clamp(point[0], obstacle.x_min, obstacle.x_max),
        _clamp(point[1], obstacle.y_min, obstacle.y_max),
    )


def _point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq < EPSILON:
        return _distance(point, start)
    t = _clamp(((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq, 0.0, 1.0)
    return _distance(point, (start[0] + t * dx, start[1] + t * dy))


def _segment_box_intersects(start, end, x_min, x_max, y_min, y_max):
    enter, leave = 0.0, 1.0
    for origin, change, low, high in (
        (start[0], end[0] - start[0], x_min, x_max),
        (start[1], end[1] - start[1], y_min, y_max),
    ):
        if abs(change) < EPSILON:
            if origin < low or origin > high:
                return False
            continue
        first, second = (low - origin) / change, (high - origin) / change
        if first > second:
            first, second = second, first
        enter, leave = max(enter, first), min(leave, second)
        if enter > leave:
            return False
    return True


class SwarmController(BaseSwarmController):
    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = game_info.obstacles
        self.obstacle_bounds = [_obstacle_bounds(obstacle) for obstacle in self.obstacles]
        self.specs = dict(game_info.drone_specs)
        self.weapon = game_info.weapon_spec
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.direction = 1.0 if self.team is Team.A else -1.0

        ordered = sorted(game_info.own_initial_drones, key=lambda d: (d.drone_type.value, d.id))
        totals = {}
        for drone in ordered:
            totals[drone.drone_type] = totals.get(drone.drone_type, 0) + 1
        counts = {}
        span = self.goal.y_max - self.goal.y_min
        self.lanes = {}
        for drone in ordered:
            index = counts.get(drone.drone_type, 0)
            counts[drone.drone_type] = index + 1
            n = totals[drone.drone_type]
            self.lanes[drone.id] = self.goal.y_min + 0.8 + (span - 1.6) * (index + 0.5) / max(1, n)

        self.steps = 0
        self.defenders = {}  # scout id -> assigned enemy id
        self.raiders = set()  # scout ids hunting enemy transports
        self.escorts = {}  # scout id -> assigned own transport id

    # ------------------------------------------------------------------
    # steering primitives
    # ------------------------------------------------------------------
    def _goal_target(self, drone):
        y = self.lanes.get(drone.id, self.goal.center[1])
        y = _clamp(y, self.goal.y_min + 0.7, self.goal.y_max - 0.7)
        return (self.goal.center[0], y)

    def _steer(self, drone, target, state, caution=1.0):
        spec = self.specs[drone.drone_type]
        dx = target[0] - drone.position[0]
        dy = target[1] - drone.position[1]
        remaining = hypot(dx, dy)
        if remaining < EPSILON:
            forward = (self.direction, 0.0)
            desired = (0.0, 0.0)
        else:
            forward = (dx / remaining, dy / remaining)
            speed = min(spec.max_speed, sqrt(max(0.0, 2.0 * spec.max_acceleration * remaining)))
            desired = (forward[0] * speed, forward[1] * speed)
        ax = 2.2 * (desired[0] - drone.velocity[0])
        ay = 2.2 * (desired[1] - drone.velocity[1])

        for (center, radius), obstacle in zip(self.obstacle_bounds, self.obstacles):
            ox = center[0] - drone.position[0]
            oy = center[1] - drone.position[1]
            along = ox * forward[0] + oy * forward[1]
            lateral = ox * -forward[1] + oy * forward[0]
            clearance = radius + 1.1
            if -0.5 < along < 8.5 and abs(lateral) < clearance:
                side = -1.0 if lateral > 0.0 else 1.0
                if abs(lateral) < 1e-6:
                    side = 1.0 if drone.id % 2 == 0 else -1.0
                force = caution * spec.max_acceleration * 1.7 * (1.0 - max(0.0, along) / 8.5)
                ax += -forward[1] * side * force
                ay += forward[0] * side * force

            nearest = _nearest_point_on_obstacle(drone.position, obstacle)
            gap = _distance(drone.position, nearest)
            if 0 < gap < 2.4:
                force = caution * spec.max_acceleration * 1.6 * (2.4 - gap) / 2.4
                rx = drone.position[0] - nearest[0]
                ry = drone.position[1] - nearest[1]
                ax += rx / gap * force
                ay += ry / gap * force

        for friend in state.own_drones:
            if friend.id == drone.id or friend.status is not DroneStatus.ACTIVE:
                continue
            rx = drone.position[0] - friend.position[0]
            ry = drone.position[1] - friend.position[1]
            rvx = drone.velocity[0] - friend.velocity[0]
            rvy = drone.velocity[1] - friend.velocity[1]
            speed_sq = rvx * rvx + rvy * rvy
            t = _clamp(-(rx * rvx + ry * rvy) / speed_sq, 0.0, 0.8) if speed_sq > EPSILON else 0.0
            mx, my = rx + rvx * t, ry + rvy * t
            miss = hypot(mx, my)
            if EPSILON < miss < 2.0:
                force = spec.max_acceleration * 1.8 * (2.0 - miss) / 2.0
                ax += mx / miss * force
                ay += my / miss * force

        for projectile in state.projectiles:
            if projectile.source_drone_id == drone.id:
                continue
            rx = projectile.position[0] - drone.position[0]
            ry = projectile.position[1] - drone.position[1]
            rvx = projectile.velocity[0] - drone.velocity[0]
            rvy = projectile.velocity[1] - drone.velocity[1]
            speed_sq = rvx * rvx + rvy * rvy
            if speed_sq < EPSILON:
                continue
            # TRANSPORT has weak acceleration relative to the 20 m/s round, so it
            # needs to start dodging earlier and harder to build up enough lateral
            # offset before the shot arrives; other classes keep the tighter margin.
            is_transport = drone.drone_type is DroneType.TRANSPORT
            lookahead = 1.8 if is_transport else 1.1
            trigger = 2.3 if is_transport else 1.3
            dodge_mult = 2.8 if is_transport else 2.0
            t = _clamp(-(rx * rvx + ry * rvy) / speed_sq, 0.0, lookahead)
            cx, cy = rx + rvx * t, ry + rvy * t
            close = hypot(cx, cy)
            if 0 < close < trigger:
                length = sqrt(speed_sq)
                lateral0 = rx * -rvy + ry * rvx
                if abs(lateral0) < 1e-6:
                    side = 1.0 if drone.id % 2 == 0 else -1.0
                else:
                    side = 1.0 if lateral0 > 0.0 else -1.0
                ax += -rvy / length * side * spec.max_acceleration * dodge_mult
                ay += rvx / length * side * spec.max_acceleration * dodge_mult
        return (ax, ay)

    def _flee_enemies(self, drone, enemies, spec):
        ax = ay = 0.0
        for enemy in enemies:
            dx = drone.position[0] - enemy.position[0]
            dy = drone.position[1] - enemy.position[1]
            d = hypot(dx, dy)
            if 0 < d < 4.5:
                force = spec.max_acceleration * 1.3 * (4.5 - d) / 4.5
                ax += dx / d * force
                ay += dy / d * force
        return (ax, ay)

    # ------------------------------------------------------------------
    # tank weapon logic
    # ------------------------------------------------------------------
    def _lead(self, shooter_position, target):
        rx = target.position[0] - shooter_position[0]
        ry = target.position[1] - shooter_position[1]
        vx, vy = target.velocity
        s = self.weapon.projectile_speed
        a = vx * vx + vy * vy - s * s
        b = 2.0 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry
        times = []
        if abs(a) < EPSILON:
            if abs(b) > EPSILON:
                times.append(-c / b)
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                root = sqrt(disc)
                times.append((-b - root) / (2.0 * a))
                times.append((-b + root) / (2.0 * a))
        t = min((value for value in times if value > 0.0), default=0.0)
        return (target.position[0] + vx * t, target.position[1] + vy * t)

    def _blocked(self, start, end, padding=0.0):
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                if _point_segment_distance(obstacle.center, start, end) <= obstacle.radius + padding:
                    return True
            elif _segment_box_intersects(
                start, end,
                obstacle.x_min - padding, obstacle.x_max + padding,
                obstacle.y_min - padding, obstacle.y_max + padding,
            ):
                return True
        return False

    def _fire(self, tank, enemies, own_active, state):
        if not tank.shots_remaining or tank.next_fire_time is None or state.time + EPSILON < tank.next_fire_time:
            return None
        candidates = sorted(
            enemies,
            key=lambda enemy: (
                _distance(tank.position, enemy.position) - 6.0 * self.specs[enemy.drone_type].point_value,
                enemy.id,
            ),
        )
        for target in candidates:
            aim = self._lead(tank.position, target)
            fx, fy = aim[0] - tank.position[0], aim[1] - tank.position[1]
            if hypot(fx, fy) < EPSILON:
                continue
            if self._blocked(tank.position, aim, padding=0.35):
                continue
            unsafe = any(
                friend.id != tank.id
                and friend.status is DroneStatus.ACTIVE
                and _point_segment_distance(friend.position, tank.position, aim) < 0.85
                for friend in own_active
            )
            if unsafe:
                continue
            return (fx, fy)
        return None

    def _tank_target(self, drone, transports, enemies):
        midfield_x = self.width * (0.46 if self.direction > 0 else 0.54)
        lane_y = self.lanes.get(drone.id, self.height * 0.5)
        if enemies:
            nearest = sorted(enemies, key=lambda enemy: _distance(drone.position, enemy.position))[:3]
            cluster_y = sum(enemy.position[1] for enemy in nearest) / len(nearest)
            # Blend with each tank's own static lane so multiple tanks tracking the
            # same enemy cluster don't converge on an identical point and collide,
            # and so the tank isn't dragged into exposed sightlines chasing enemies.
            target_y = 0.25 * cluster_y + 0.75 * lane_y
        else:
            target_y = lane_y
        target_y = _clamp(target_y, 2.0, self.height - 2.0)
        target_x = midfield_x
        if transports:
            lead = max(transports, key=lambda t: self.direction * t.position[0])
            desired_x = lead.position[0] - self.direction * 5.5
            if self.direction * (desired_x - midfield_x) > 0.0:
                target_x = desired_x
        return (target_x, target_y)

    # ------------------------------------------------------------------
    # scout role assignment (recomputed every ~0.5 s)
    # ------------------------------------------------------------------
    def _update_roles(self, own_active, enemies, state):
        scouts = sorted((d for d in own_active if d.drone_type is DroneType.SCOUT), key=lambda d: d.id)
        transports = [d for d in own_active if d.drone_type is DroneType.TRANSPORT]
        if not scouts:
            self.defenders, self.raiders, self.escorts = {}, set(), {}
            return

        n = len(scouts)
        time_left = 90.0 - state.time
        score_diff = state.own_score - state.opponent_score

        n_defend = max(2, n // 4)
        n_raid = max(2, n // 5)
        n_escort = min(len(transports), max(2, n // 4)) if transports else 0

        if score_diff < 0 and time_left > 20.0:
            n_raid = max(n_raid, n * 2 // 5)
        elif score_diff > 2 and time_left < 25.0:
            n_defend = max(n_defend, n * 2 // 5)
            n_raid = max(1, n // 6)

        n_defend = min(n_defend, n)
        n_raid = min(n_raid, max(0, n - n_defend))
        n_escort = min(n_escort, max(0, n - n_defend - n_raid))

        self.defenders = {}
        if enemies and n_defend > 0:
            threats = sorted(
                enemies,
                key=lambda e: _distance(e.position, self.own_goal.center) - 6.0 * self.specs[e.drone_type].point_value,
            )[: min(n_defend, len(enemies))]
            if threats:
                if linear_sum_assignment is not None:
                    cost = [[_distance(d.position, t.position) for t in threats] for d in scouts]
                    rows, cols = linear_sum_assignment(cost)
                    for r, c in zip(rows, cols):
                        self.defenders[scouts[r].id] = threats[c].id
                else:
                    remaining = scouts[:]
                    for threat in threats:
                        if not remaining:
                            break
                        nearest = min(remaining, key=lambda d: _distance(d.position, threat.position))
                        self.defenders[nearest.id] = threat.id
                        remaining.remove(nearest)

        used = set(self.defenders.keys())
        remaining_scouts = [d for d in scouts if d.id not in used]

        self.raiders = {d.id for d in remaining_scouts[:n_raid]}
        used.update(self.raiders)
        remaining_scouts = [d for d in remaining_scouts if d.id not in self.raiders]

        self.escorts = {}
        if transports and n_escort > 0:
            # Cover the most-advanced (most exposed) transports first.
            exposed = sorted(transports, key=lambda t: -self.direction * t.position[0])
            for index, drone in enumerate(remaining_scouts[:n_escort]):
                self.escorts[drone.id] = exposed[index % len(exposed)].id

    def _opportunistic_ram(self, drone, enemies, spec):
        best = None
        best_score = None
        for enemy in enemies:
            d = _distance(drone.position, enemy.position)
            if enemy.drone_type is DroneType.TANK and d > 4.0:
                continue
            if d > 10.0:
                continue
            value = self.specs[enemy.drone_type].point_value
            score = value * 4.0 - d
            if best_score is None or score > best_score:
                best_score = score
                best = enemy
        if best is None:
            return None
        lead_t = min(1.2, _distance(drone.position, best.position) / max(spec.max_speed, 0.1))
        return (best.position[0] + best.velocity[0] * lead_t, best.position[1] + best.velocity[1] * lead_t)

    def _raid_target(self, drone, enemies, enemy_transports):
        """Pick the highest-leverage kill: an exposed enemy transport, or a
        live enemy tank close enough to catch. Killing a tank before it
        empties its magazine prevents further transport losses to gunfire,
        which is usually worth more than the tank's own 1-point value."""
        best = None
        best_score = None
        for enemy in enemy_transports:
            score = _distance(enemy.position, self.own_goal.center) - 3.0 * self.specs[enemy.drone_type].point_value
            if best_score is None or score < best_score:
                best_score, best = score, enemy
        for enemy in enemies:
            if enemy.drone_type is not DroneType.TANK:
                continue
            if enemy.shots_remaining is not None and enemy.shots_remaining <= 0:
                continue
            d = _distance(drone.position, enemy.position)
            if d > 13.0:
                continue
            score = d - 15.0
            if best_score is None or score < best_score:
                best_score, best = score, enemy
        return best

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def step(self, state):
        self.steps += 1
        own_active = [d for d in state.own_drones if d.status is DroneStatus.ACTIVE]
        enemies = [d for d in state.opponent_drones if d.status is DroneStatus.ACTIVE]
        enemy_by_id = {e.id: e for e in enemies}
        transports = [d for d in own_active if d.drone_type is DroneType.TRANSPORT]
        enemy_transports = [d for d in enemies if d.drone_type is DroneType.TRANSPORT]

        if self.steps == 1 or self.steps % 5 == 0:
            self._update_roles(own_active, enemies, state)
        else:
            self.defenders = {k: v for k, v in self.defenders.items() if v in enemy_by_id}

        actions = {}
        for drone in own_active:
            spec = self.specs[drone.drone_type]
            caution = 1.0

            if drone.drone_type is DroneType.TANK:
                target = self._tank_target(drone, transports, enemies)
                caution = 1.15

            elif drone.drone_type is DroneType.TRANSPORT:
                target = self._goal_target(drone)

            else:  # SCOUT
                target = None
                if drone.id in self.defenders and enemy_by_id.get(self.defenders[drone.id]) is not None:
                    enemy = enemy_by_id[self.defenders[drone.id]]
                    lead_t = min(1.5, _distance(drone.position, enemy.position) / max(spec.max_speed, 0.1))
                    target = (enemy.position[0] + enemy.velocity[0] * lead_t, enemy.position[1] + enemy.velocity[1] * lead_t)
                    caution = 0.75
                elif drone.id in self.escorts:
                    charge = next((t for t in transports if t.id == self.escorts[drone.id]), None)
                    if charge is not None:
                        nearby_threats = [e for e in enemies if _distance(e.position, charge.position) < 7.0]
                        gunners = [
                            e for e in enemies
                            if e.drone_type is DroneType.TANK
                            and (e.shots_remaining is None or e.shots_remaining > 0)
                            and _distance(e.position, charge.position) < 22.0
                        ]
                        if nearby_threats:
                            threat = min(nearby_threats, key=lambda e: _distance(e.position, charge.position))
                            target = ((threat.position[0] + charge.position[0]) * 0.5, (threat.position[1] + charge.position[1]) * 0.5)
                            caution = 0.75
                        elif gunners:
                            # Screen the transport from the nearest armed enemy tank by
                            # standing on the tank-transport bearing just ahead of it, so
                            # a sniping shot meant for the transport hits the escort first.
                            gunner = min(gunners, key=lambda e: _distance(e.position, charge.position))
                            gx, gy = gunner.position[0] - charge.position[0], gunner.position[1] - charge.position[1]
                            gd = hypot(gx, gy) or 1.0
                            target = (charge.position[0] + gx / gd * 2.2, charge.position[1] + gy / gd * 2.2)
                            caution = 0.8
                        else:
                            offset = 1.3 if drone.id % 2 == 0 else -1.3
                            target = (
                                charge.position[0] + self.direction * 2.6,
                                _clamp(charge.position[1] + offset, 0.7, self.height - 0.7),
                            )
                elif drone.id in self.raiders:
                    victim = self._raid_target(drone, enemies, enemy_transports)
                    if victim is not None:
                        lead_t = min(1.3, _distance(drone.position, victim.position) / max(spec.max_speed, 0.1))
                        target = (victim.position[0] + victim.velocity[0] * lead_t, victim.position[1] + victim.velocity[1] * lead_t)
                        caution = 0.75

                if target is None:
                    target = self._goal_target(drone)
                    if enemies:
                        ram = self._opportunistic_ram(drone, enemies, spec)
                        if ram is not None:
                            target = ram
                            caution = 0.7

            ax, ay = self._steer(drone, target, state, caution)
            if drone.drone_type is DroneType.TRANSPORT and enemies:
                fx, fy = self._flee_enemies(drone, enemies, spec)
                ax += fx
                ay += fy

            command = {"acceleration": (ax, ay)}
            if drone.drone_type is DroneType.TANK and enemies:
                fire = self._fire(drone, enemies, own_active, state)
                if fire is not None:
                    command["fire_direction"] = fire

            actions[drone.id] = command
        return actions
