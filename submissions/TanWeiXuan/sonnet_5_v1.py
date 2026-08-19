"""Claude Sonnet 5: role-based swarm control with predictive ramming and lead fire.

Strategy summary
-----------------
Vehicle-vehicle contact of any kind destroys both parties, so a cheap SCOUT
(1 pt) that collides with an enemy TRANSPORT (5 pt) is a strongly favorable
trade. This controller leans on that mechanic:

* TANK: holds a defensive line on our own side of the arena, drifts toward
  the local center of mass of nearby enemies for sightlines, and fires with
  a closed-form intercept lead the moment a target is in a clear line of
  fire (no obstacle or friendly drone in the way).
* SCOUT: split every half-second into "defenders" (assigned via a min-cost
  bipartite match to the enemies most threatening our goal, so they can
  intercept and ram) and "attackers" (push down an assigned lane toward the
  goal, but opportunistically divert onto a collision course with any
  nearby high-value enemy, i.e. a TRANSPORT, or a TANK caught at close
  range).
* TRANSPORT: always pushes its lane toward the goal while actively steering
  away from nearby enemies, since losing a TRANSPORT is the worst trade on
  the board.

All drones share obstacle avoidance, friendly separation, and incoming
projectile dodging so the higher-level intent above does not get anyone
killed by the terrain, a teammate, or a stray round.
"""

from __future__ import annotations

from math import hypot, sqrt

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType, Team

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - scipy is a required dependency, but stay defensive
    linear_sum_assignment = None


TINY = 1.0e-9


def _dist(left, right):
    return hypot(left[0] - right[0], left[1] - right[1])


def _obstacle_shape(obstacle):
    if isinstance(obstacle, CircleObstacle):
        return obstacle.center, obstacle.radius
    center = ((obstacle.x_min + obstacle.x_max) * 0.5, (obstacle.y_min + obstacle.y_max) * 0.5)
    radius = hypot(obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min) * 0.5
    return center, radius


class SwarmController(BaseSwarmController):
    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = game_info.obstacles
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
            self.lanes[drone.id] = self.goal.y_min + 0.9 + (span - 1.8) * (index + 0.5) / max(1, n)

        self.steps = 0
        self.defenders = {}  # own scout id -> assigned enemy id

    # ------------------------------------------------------------------
    # steering primitives
    # ------------------------------------------------------------------
    def _lane_target(self, drone):
        y = self.lanes.get(drone.id, self.goal.center[1])
        y = min(self.goal.y_max - 0.7, max(self.goal.y_min + 0.7, y))
        return (self.goal.center[0], y)

    def _steer(self, drone, target, spec, caution=1.0):
        dx = target[0] - drone.position[0]
        dy = target[1] - drone.position[1]
        remaining = hypot(dx, dy)
        if remaining < TINY:
            desired = (0.0, 0.0)
            forward = (self.direction, 0.0)
        else:
            forward = (dx / remaining, dy / remaining)
            speed = min(spec.max_speed, sqrt(max(0.0, 2.0 * spec.max_acceleration * remaining)))
            desired = (forward[0] * speed, forward[1] * speed)
        ax = 2.2 * (desired[0] - drone.velocity[0])
        ay = 2.2 * (desired[1] - drone.velocity[1])

        for obstacle in self.obstacles:
            center, radius = _obstacle_shape(obstacle)
            ox = center[0] - drone.position[0]
            oy = center[1] - drone.position[1]
            along = ox * forward[0] + oy * forward[1]
            lateral = ox * -forward[1] + oy * forward[0]
            clearance = radius + 1.1
            if -0.5 < along < 8.5 and abs(lateral) < clearance:
                side = -1.0 if lateral > 0 else 1.0
                if abs(lateral) < 1e-6:
                    side = 1.0 if drone.id % 2 == 0 else -1.0
                force = caution * spec.max_acceleration * 1.6 * (1.0 - max(0.0, along) / 8.5)
                ax += -forward[1] * side * force
                ay += forward[0] * side * force
            center_dist = hypot(ox, oy)
            surface = center_dist - radius
            if 0 < surface < 2.6:
                force = caution * spec.max_acceleration * (2.6 - surface) / 2.6
                ax -= ox / center_dist * force
                ay -= oy / center_dist * force
        return ax, ay

    def _separation(self, drone, own_active, spec):
        ax = ay = 0.0
        for friend in own_active:
            if friend.id == drone.id:
                continue
            dx = drone.position[0] - friend.position[0]
            dy = drone.position[1] - friend.position[1]
            sep = hypot(dx, dy)
            if 0 < sep < 1.8:
                force = spec.max_acceleration * 1.4 * (1.8 - sep) / 1.8
                ax += dx / sep * force
                ay += dy / sep * force
        return ax, ay

    def _dodge_projectiles(self, drone, state, spec):
        ax = ay = 0.0
        for projectile in state.projectiles:
            if projectile.source_drone_id == drone.id:
                continue
            rx = projectile.position[0] - drone.position[0]
            ry = projectile.position[1] - drone.position[1]
            rvx = projectile.velocity[0] - drone.velocity[0]
            rvy = projectile.velocity[1] - drone.velocity[1]
            speed_sq = rvx * rvx + rvy * rvy
            if speed_sq < TINY:
                continue
            t = max(0.0, min(1.2, -(rx * rvx + ry * rvy) / speed_sq))
            cx = rx + rvx * t
            cy = ry + rvy * t
            close = hypot(cx, cy)
            if 0 < close < 1.3:
                side = 1.0 if drone.id % 2 == 0 else -1.0
                length = hypot(rvx, rvy)
                ax += -rvy / length * side * spec.max_acceleration * 1.8
                ay += rvx / length * side * spec.max_acceleration * 1.8
        return ax, ay

    # ------------------------------------------------------------------
    # tank weapon logic
    # ------------------------------------------------------------------
    def _intercept_point(self, shooter_pos, target):
        rx = target.position[0] - shooter_pos[0]
        ry = target.position[1] - shooter_pos[1]
        vx, vy = target.velocity
        s = self.weapon.projectile_speed
        a = vx * vx + vy * vy - s * s
        b = 2.0 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry
        times = []
        if abs(a) < 1e-9:
            if abs(b) > 1e-9:
                times.append(-c / b)
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0:
                root = sqrt(disc)
                times.append((-b - root) / (2.0 * a))
                times.append((-b + root) / (2.0 * a))
        positive = [t for t in times if t > 0]
        t = min(positive) if positive else 0.0
        return (target.position[0] + vx * t, target.position[1] + vy * t)

    def _clear_shot(self, shooter, aim, own_active):
        sx, sy = shooter.position
        dx, dy = aim[0] - sx, aim[1] - sy
        length_sq = dx * dx + dy * dy
        if length_sq < TINY:
            return False
        for obstacle in self.obstacles:
            center, radius = _obstacle_shape(obstacle)
            t = min(1.0, max(0.0, ((center[0] - sx) * dx + (center[1] - sy) * dy) / length_sq))
            px, py = sx + dx * t, sy + dy * t
            if hypot(center[0] - px, center[1] - py) < radius + 0.4:
                return False
        for friend in own_active:
            if friend.id == shooter.id:
                continue
            t = min(1.0, max(0.0, ((friend.position[0] - sx) * dx + (friend.position[1] - sy) * dy) / length_sq))
            px, py = sx + dx * t, sy + dy * t
            if hypot(friend.position[0] - px, friend.position[1] - py) < 0.6:
                return False
        return True

    def _best_target(self, drone, enemies, own_active):
        best = None
        best_score = None
        for enemy in enemies:
            aim = self._intercept_point(drone.position, enemy)
            if not self._clear_shot(drone, aim, own_active):
                continue
            d = _dist(drone.position, enemy.position)
            value = self.specs[enemy.drone_type].point_value
            score = value * 10.0 - d
            if best_score is None or score > best_score:
                best_score = score
                best = (enemy, aim)
        return best

    # ------------------------------------------------------------------
    # defender / attacker role assignment (recomputed every ~0.5s)
    # ------------------------------------------------------------------
    def _update_assignments(self, own_active, enemies):
        scouts = [d for d in own_active if d.drone_type is DroneType.SCOUT]
        if not scouts or not enemies:
            self.defenders = {}
            return

        threats = sorted(
            enemies,
            key=lambda e: _dist(e.position, self.own_goal.center) - 6.0 * self.specs[e.drone_type].point_value,
        )
        n_defenders = min(len(scouts) // 2 + 1, len(scouts), len(threats))
        chosen = threats[:n_defenders]

        if linear_sum_assignment is not None and chosen:
            cost = [[_dist(d.position, t.position) for t in chosen] for d in scouts]
            rows, cols = linear_sum_assignment(cost)
            pairs = sorted(zip(rows, cols), key=lambda rc: cost[rc[0]][rc[1]])[:n_defenders]
            self.defenders = {scouts[r].id: chosen[c].id for r, c in pairs}
        else:
            assigned = {}
            remaining = scouts[:]
            for threat in chosen:
                if not remaining:
                    break
                nearest = min(remaining, key=lambda d: _dist(d.position, threat.position))
                assigned[nearest.id] = threat.id
                remaining.remove(nearest)
            self.defenders = assigned

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def step(self, state):
        self.steps += 1
        own_active = [d for d in state.own_drones if d.status is DroneStatus.ACTIVE]
        enemies = [d for d in state.opponent_drones if d.status is DroneStatus.ACTIVE]
        enemy_by_id = {e.id: e for e in enemies}

        if enemies and (self.steps == 1 or self.steps % 5 == 0):
            self._update_assignments(own_active, enemies)
        else:
            self.defenders = {k: v for k, v in self.defenders.items() if v in enemy_by_id}

        actions = {}
        for drone in own_active:
            spec = self.specs[drone.drone_type]
            ax = ay = 0.0

            if drone.drone_type is DroneType.TANK:
                target_x = 50.0 - self.direction * 5.0
                if enemies:
                    nearest = sorted(enemies, key=lambda e: _dist(drone.position, e.position))[:3]
                    target_y = sum(e.position[1] for e in nearest) / len(nearest)
                else:
                    target_y = self.goal.center[1]
                target_y = min(self.height - 2.0, max(2.0, target_y))
                tax, tay = self._steer(drone, (target_x, target_y), spec, caution=1.2)
                ax += tax
                ay += tay

            elif drone.id in self.defenders and enemy_by_id.get(self.defenders[drone.id]) is not None:
                enemy = enemy_by_id[self.defenders[drone.id]]
                lead_t = min(2.0, _dist(drone.position, enemy.position) / max(spec.max_speed, 0.1))
                aim = (enemy.position[0] + enemy.velocity[0] * lead_t, enemy.position[1] + enemy.velocity[1] * lead_t)
                tax, tay = self._steer(drone, aim, spec, caution=0.8)
                ax += tax
                ay += tay

            else:
                ram_target = None
                if drone.drone_type is DroneType.SCOUT and enemies:
                    best_score = None
                    for enemy in enemies:
                        d = _dist(drone.position, enemy.position)
                        if enemy.drone_type is DroneType.TANK and d > 4.0:
                            continue
                        if d > 11.0:
                            continue
                        value = self.specs[enemy.drone_type].point_value
                        score = value * 4.0 - d
                        if best_score is None or score > best_score:
                            best_score = score
                            ram_target = enemy
                if ram_target is not None:
                    lead_t = min(1.5, _dist(drone.position, ram_target.position) / max(spec.max_speed, 0.1))
                    aim = (
                        ram_target.position[0] + ram_target.velocity[0] * lead_t,
                        ram_target.position[1] + ram_target.velocity[1] * lead_t,
                    )
                    tax, tay = self._steer(drone, aim, spec, caution=0.6)
                else:
                    tax, tay = self._steer(drone, self._lane_target(drone), spec)
                ax += tax
                ay += tay

            sx, sy = self._separation(drone, own_active, spec)
            ax += sx
            ay += sy
            dxp, dyp = self._dodge_projectiles(drone, state, spec)
            ax += dxp
            ay += dyp

            if drone.drone_type is DroneType.TRANSPORT and enemies:
                for enemy in enemies:
                    ddx = drone.position[0] - enemy.position[0]
                    ddy = drone.position[1] - enemy.position[1]
                    sep = hypot(ddx, ddy)
                    if 0 < sep < 4.0:
                        force = spec.max_acceleration * 1.3 * (4.0 - sep) / 4.0
                        ax += ddx / sep * force
                        ay += ddy / sep * force

            command = {"acceleration": (ax, ay)}
            if (
                drone.drone_type is DroneType.TANK
                and drone.shots_remaining
                and drone.next_fire_time is not None
                and state.time + 1e-9 >= drone.next_fire_time
                and enemies
            ):
                found = self._best_target(drone, enemies, own_active)
                if found is not None:
                    _, aim = found
                    fx = aim[0] - drone.position[0]
                    fy = aim[1] - drone.position[1]
                    if hypot(fx, fy) > TINY:
                        command["fire_direction"] = (fx, fy)

            actions[drone.id] = command
        return actions
