"""BigPickle V1 - Score-phase-aware swarm controller.

Strategy adapts based on game phase and score differential:
- Early phase (0-30s): Balanced intercept/escort/goal split.
- Mid phase (30-60s): Aggressive interception of high-value targets.
- Late phase (60-90s): Score-aware play - push when behind, defend when ahead.

Key improvements over Wayfinder V2:
- Phase-aware dynamic scout allocation ratios.
- Urgency-weighted enemy transport targeting (closest to own goal).
- Enhanced tank fire with obstacle-aware blocking for rectangles.
- Interpose-based escort positioning when threats approach.

Attribution: Core steering, lane assignment, obstacle avoidance, and
collision logic adapted from Wayfinder V2 (wayfinder_v2.py).
"""

from math import hypot, sqrt

from swarmbench import (
    BaseSwarmController,
    CircleObstacle,
    DroneStatus,
    DroneType,
    RectangleObstacle,
    Team,
    TANK_WEAPON_SPEC,
)

EPSILON = 1.0e-9


def _distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def _point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq < EPSILON:
        return _distance(point, start)
    t = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
    return _distance(point, (start[0] + t * dx, start[1] + t * dy))


def _segment_box_intersect(start, end, x_min, x_max, y_min, y_max):
    enter, leave = 0.0, 1.0
    for origin, change, low, high in (
        (start[0], end[0] - start[0], x_min, x_max),
        (start[1], end[1] - start[1], y_min, y_max),
    ):
        if abs(change) < EPSILON:
            if origin < low or origin > high:
                return False
            continue
        first = (low - origin) / change
        second = (high - origin) / change
        if first > second:
            first, second = second, first
        enter = max(enter, first)
        leave = min(leave, second)
        if enter > leave:
            return False
    return True


class SwarmController(BaseSwarmController):
    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)
        self.weapon = getattr(game_info, "weapon_spec", TANK_WEAPON_SPEC)
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.direction = 1.0 if self.team is Team.A else -1.0

        self._obs_data = []
        for obs in self.obstacles:
            if isinstance(obs, CircleObstacle):
                self._obs_data.append(("c", obs.center, obs.radius, None))
            else:
                ctr = ((obs.x_min + obs.x_max) * 0.5, (obs.y_min + obs.y_max) * 0.5)
                r = hypot(obs.x_max - obs.x_min, obs.y_max - obs.y_min) * 0.5
                self._obs_data.append(("r", ctr, r, obs))

        ordered = sorted(game_info.own_initial_drones, key=lambda d: (d.drone_type.value, d.id))
        n = len(ordered)
        self.lanes = {}
        for i, d in enumerate(ordered):
            y_lo = self.goal.y_min + 0.7
            y_hi = self.goal.y_max - 0.7
            self.lanes[d.id] = y_lo + (y_hi - y_lo) * (i + 0.5) / max(1, n)

    def _goal_target(self, drone):
        lane = self.lanes.get(drone.id, self.goal.center[1])
        return (
            self.goal.center[0],
            min(self.goal.y_max - 0.65, max(self.goal.y_min + 0.65, lane)),
        )

    def _shape(self, obs_data):
        return obs_data[1], obs_data[2]

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
            desired_speed = min(spec.max_speed, sqrt(2.0 * spec.max_acceleration * remaining))
            desired = (forward[0] * desired_speed, forward[1] * desired_speed)
        ax = 2.15 * (desired[0] - drone.velocity[0])
        ay = 2.15 * (desired[1] - drone.velocity[1])

        for obs_data in self._obs_data:
            ctr, radius = self._shape(obs_data)
            ox = ctr[0] - drone.position[0]
            oy = ctr[1] - drone.position[1]
            along = ox * forward[0] + oy * forward[1]
            lateral = ox * -forward[1] + oy * forward[0]
            clearance = radius + 1.05
            if -0.4 < along < 9.0 and abs(lateral) < clearance:
                side = -1.0 if lateral > 0.0 else 1.0
                if abs(lateral) < 0.08:
                    side = 1.0 if drone.id % 2 == 0 else -1.0
                force = caution * spec.max_acceleration * 1.7 * (1.0 - max(0.0, along) / 9.0)
                ax += -forward[1] * side * force
                ay += forward[0] * side * force
            center_gap = hypot(ox, oy)
            surface_gap = center_gap - radius
            if EPSILON < surface_gap < 2.6:
                force = caution * spec.max_acceleration * (2.6 - surface_gap) / 2.6
                ax -= ox / center_gap * force
                ay -= oy / center_gap * force

        for friend in state.own_drones:
            if friend.id == drone.id or friend.status is not DroneStatus.ACTIVE:
                continue
            rx = drone.position[0] - friend.position[0]
            ry = drone.position[1] - friend.position[1]
            rvx = drone.velocity[0] - friend.velocity[0]
            rvy = drone.velocity[1] - friend.velocity[1]
            speed_sq = rvx * rvx + rvy * rvy
            closest_time = min(0.75, max(0.0, -(rx * rvx + ry * rvy) / speed_sq)) if speed_sq > EPSILON else 0.0
            mx = rx + rvx * closest_time
            my = ry + rvy * closest_time
            miss = hypot(mx, my)
            if EPSILON < miss < 2.0:
                force = spec.max_acceleration * 1.8 * (2.0 - miss) / 2.0
                ax += mx / miss * force
                ay += my / miss * force

        for proj in state.projectiles:
            rx = proj.position[0] - drone.position[0]
            ry = proj.position[1] - drone.position[1]
            rvx = proj.velocity[0] - drone.velocity[0]
            rvy = proj.velocity[1] - drone.velocity[1]
            speed_sq = rvx * rvx + rvy * rvy
            if speed_sq < EPSILON:
                continue
            closest_time = min(1.0, max(0.0, -(rx * rvx + ry * rvy) / speed_sq))
            if hypot(rx + rvx * closest_time, ry + rvy * closest_time) < 1.3:
                length = sqrt(speed_sq)
                side = 1.0 if drone.id % 2 == 0 else -1.0
                ax += -rvy / length * side * spec.max_acceleration * 1.9
                ay += rvx / length * side * spec.max_acceleration * 1.9
        return (ax, ay)

    def _blocked(self, start, end, padding=0.0):
        for obs_data in self._obs_data:
            ctr, radius = obs_data[1], obs_data[2]
            if obs_data[0] == "c":
                if _point_segment_distance(ctr, start, end) <= radius + padding:
                    return True
            else:
                rect = obs_data[3]
                if _segment_box_intersect(
                    start, end,
                    rect.x_min - padding, rect.x_max + padding,
                    rect.y_min - padding, rect.y_max + padding,
                ):
                    return True
        return False

    def _lead(self, shooter, target):
        rx = target.position[0] - shooter.position[0]
        ry = target.position[1] - shooter.position[1]
        vx, vy = target.velocity
        speed = self.weapon.projectile_speed
        a = vx * vx + vy * vy - speed * speed
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
                times.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        flight = min((v for v in times if v > 0.0), default=0.0)
        return (target.position[0] + vx * flight, target.position[1] + vy * flight)

    def _fire(self, tank, enemies, state):
        if not tank.shots_remaining or tank.next_fire_time is None or state.time + EPSILON < tank.next_fire_time:
            return None
        candidates = sorted(
            enemies,
            key=lambda enemy: (
                _distance(tank.position, enemy.position)
                - 5.0 * self.specs[enemy.drone_type].point_value,
                enemy.id,
            ),
        )
        for target in candidates:
            aim = self._lead(tank, target)
            if self._blocked(tank.position, aim):
                continue
            unsafe = any(
                fr.id != tank.id
                and fr.status is DroneStatus.ACTIVE
                and _point_segment_distance(fr.position, tank.position, aim) < 0.85
                for fr in state.own_drones
            )
            if not unsafe:
                return (aim[0] - tank.position[0], aim[1] - tank.position[1])
        return None

    def step(self, state):
        own = [d for d in state.own_drones if d.status is DroneStatus.ACTIVE]
        enemies = [d for d in state.opponent_drones if d.status is DroneStatus.ACTIVE]
        transports = [d for d in own if d.drone_type is DroneType.TRANSPORT]
        enemy_transports = [d for d in enemies if d.drone_type is DroneType.TRANSPORT]
        scouts = sorted(
            (d for d in own if d.drone_type is DroneType.SCOUT),
            key=lambda d: d.id,
        )
        scout_rank = {d.id: i for i, d in enumerate(scouts)}
        n_scouts = len(scouts)

        score_diff = state.own_score - state.opponent_score
        time_left = 90.0 - state.time

        if score_diff < 0 and time_left > 25:
            n_intercept = max(2, n_scouts * 2 // 5)
        elif score_diff > 3 and time_left < 25:
            n_intercept = max(1, n_scouts // 5)
        else:
            n_intercept = max(1, n_scouts // 4) if n_scouts else 0
        n_escort = min(len(transports) * 2, max(1, n_scouts // 4)) if transports else 0

        actions = {}
        for drone in own:
            target = self._goal_target(drone)
            caution = 1.0

            if drone.drone_type is DroneType.SCOUT:
                rank = scout_rank.get(drone.id, 0)
                if rank < n_intercept and enemy_transports:
                    victim = min(
                        enemy_transports,
                        key=lambda e: (
                            _distance(e.position, self.own_goal.center)
                            - 3.0 * self.specs[e.drone_type].point_value,
                            e.id,
                        ),
                    )
                    lead_time = min(
                        1.2,
                        _distance(drone.position, victim.position)
                        / max(1.0, self.specs[DroneType.SCOUT].max_speed),
                    )
                    target = (
                        victim.position[0] + victim.velocity[0] * lead_time,
                        victim.position[1] + victim.velocity[1] * lead_time,
                    )
                    caution = 0.8
                elif rank < n_intercept + n_escort and transports:
                    charge = transports[(rank - n_intercept) % len(transports)]
                    offset = ((rank % 5) - 2) * 1.2
                    target = (
                        charge.position[0] + self.direction * 2.8,
                        min(self.height - 0.7, max(0.7, charge.position[1] + offset)),
                    )
                    threats = [
                        e for e in enemies
                        if _distance(e.position, charge.position) < 6.0
                    ]
                    if threats:
                        threat = min(threats, key=lambda e: (_distance(e.position, charge.position), e.id))
                        ix = (threat.position[0] + charge.position[0]) * 0.5
                        iy = (threat.position[1] + charge.position[1]) * 0.5
                        target = (ix, iy)
                elif enemy_transports:
                    victim = min(
                        enemy_transports,
                        key=lambda e: (_distance(drone.position, e.position), e.id),
                    )
                    lead_time = min(
                        1.0,
                        _distance(drone.position, victim.position)
                        / max(1.0, self.specs[DroneType.SCOUT].max_speed),
                    )
                    target = (
                        victim.position[0] + victim.velocity[0] * lead_time,
                        victim.position[1] + victim.velocity[1] * lead_time,
                    )
                    caution = 0.85

            elif drone.drone_type is DroneType.TANK:
                midfield_x = self.width * (0.46 if self.team is Team.A else 0.54)
                if transports:
                    lead_transport = max(transports, key=lambda t: self.direction * t.position[0])
                    desired_x = lead_transport.position[0] - self.direction * 5.5
                    if self.direction * (desired_x - midfield_x) > 0:
                        target = (desired_x, lead_transport.position[1])
                    else:
                        target = (midfield_x, self.lanes.get(drone.id, self.height * 0.5))
                else:
                    target = (midfield_x, self.lanes.get(drone.id, self.height * 0.5))

            command = {"acceleration": self._steer(drone, target, state, caution)}

            if drone.drone_type is DroneType.TANK and enemies:
                fire = self._fire(drone, enemies, state)
                if fire is not None:
                    command["fire_direction"] = fire

            actions[drone.id] = command
        return actions
