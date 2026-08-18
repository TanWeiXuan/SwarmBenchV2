"""A modest v2 controller inspired by Aegis Weave and Opus Apex.

Transports advance in lanes, Scouts either screen the convoy or intercept
valuable enemies, and Tanks trail the formation while taking clear lead shots.
The controller also avoids friendly collisions, obstacles, and known bullets.
"""

from math import hypot, sqrt

from swarmbench import (
    BaseSwarmController,
    CircleObstacle,
    DroneStatus,
    DroneType,
    RectangleObstacle,
    Team,
)


EPSILON = 1.0e-9


def distance(left, right):
    return hypot(left[0] - right[0], left[1] - right[1])


def segment_point_distance(left, right, point):
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    denominator = dx * dx + dy * dy
    if denominator < EPSILON:
        return distance(left, point)
    amount = ((point[0] - left[0]) * dx + (point[1] - left[1]) * dy) / denominator
    amount = min(1.0, max(0.0, amount))
    closest = (left[0] + amount * dx, left[1] + amount * dy)
    return distance(closest, point)


def segment_box_intersects(left, right, x_min, x_max, y_min, y_max):
    enter, leave = 0.0, 1.0
    for origin, change, low, high in (
        (left[0], right[0] - left[0], x_min, x_max),
        (left[1], right[1] - left[1], y_min, y_max),
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
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)
        self.weapon = game_info.weapon_spec
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.forward = 1.0 if self.team is Team.A else -1.0

        ordered = sorted(game_info.own_initial_drones, key=lambda drone: (drone.drone_type.value, drone.position[1], drone.id))
        self.lanes = {}
        for index, drone in enumerate(ordered):
            span = max(1.0, self.goal.y_max - self.goal.y_min - 1.5)
            fraction = ((index * 7) % max(1, len(ordered)) + 0.5) / max(1, len(ordered))
            self.lanes[drone.id] = self.goal.y_min + 0.75 + span * fraction

    def _goal_target(self, drone):
        y = self.lanes.get(drone.id, self.goal.center[1])
        return (self.goal.center[0], min(self.goal.y_max - 0.7, max(self.goal.y_min + 0.7, y)))

    def _obstacle_center_radius(self, obstacle):
        if isinstance(obstacle, CircleObstacle):
            return obstacle.center, obstacle.radius
        center = ((obstacle.x_min + obstacle.x_max) * 0.5, (obstacle.y_min + obstacle.y_max) * 0.5)
        radius = hypot(obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min) * 0.5
        return center, radius

    def _blocked(self, left, right, padding=0.0):
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                if segment_point_distance(left, right, obstacle.center) <= obstacle.radius + padding:
                    return True
            elif segment_box_intersects(
                left,
                right,
                obstacle.x_min - padding,
                obstacle.x_max + padding,
                obstacle.y_min - padding,
                obstacle.y_max + padding,
            ):
                return True
        return False

    def _steer(self, drone, target, state):
        spec = self.specs[drone.drone_type]
        dx = target[0] - drone.position[0]
        dy = target[1] - drone.position[1]
        remaining = hypot(dx, dy)
        if remaining < EPSILON:
            desired_x = desired_y = 0.0
            forward_x, forward_y = self.forward, 0.0
        else:
            speed = min(spec.max_speed, sqrt(2.0 * spec.max_acceleration * remaining))
            forward_x, forward_y = dx / remaining, dy / remaining
            desired_x, desired_y = forward_x * speed, forward_y * speed
        ax = 2.25 * (desired_x - drone.velocity[0])
        ay = 2.25 * (desired_y - drone.velocity[1])

        # A deterministic side-step around obstacles directly ahead.
        for obstacle in self.obstacles:
            center, radius = self._obstacle_center_radius(obstacle)
            offset_x = center[0] - drone.position[0]
            offset_y = center[1] - drone.position[1]
            along = offset_x * forward_x + offset_y * forward_y
            across = offset_x * -forward_y + offset_y * forward_x
            safe = radius + 1.25
            if -0.4 < along < 9.0 and abs(across) < safe:
                side = -1.0 if across > 0.0 else 1.0
                if abs(across) < 0.05:
                    side = 1.0 if drone.id % 2 == 0 else -1.0
                strength = spec.max_acceleration * 1.7 * (1.0 - max(0.0, along) / 9.0)
                ax += -forward_y * side * strength
                ay += forward_x * side * strength
            center_gap = hypot(offset_x, offset_y)
            surface_gap = center_gap - radius
            if EPSILON < surface_gap < 2.5:
                strength = spec.max_acceleration * 1.5 * (2.5 - surface_gap) / 2.5
                ax -= offset_x / center_gap * strength
                ay -= offset_y / center_gap * strength

        # Friendly collision avoidance considers current and short-term spacing.
        for friend in state.own_drones:
            if friend.id == drone.id or friend.status is not DroneStatus.ACTIVE:
                continue
            rx = drone.position[0] - friend.position[0]
            ry = drone.position[1] - friend.position[1]
            rvx = drone.velocity[0] - friend.velocity[0]
            rvy = drone.velocity[1] - friend.velocity[1]
            speed_squared = rvx * rvx + rvy * rvy
            closest_time = min(0.8, max(0.0, -(rx * rvx + ry * rvy) / speed_squared)) if speed_squared else 0.0
            future_x = rx + rvx * closest_time
            future_y = ry + rvy * closest_time
            gap = hypot(future_x, future_y)
            if EPSILON < gap < 2.0:
                strength = spec.max_acceleration * 1.8 * (2.0 - gap) / 2.0
                ax += future_x / gap * strength
                ay += future_y / gap * strength

        # Perfect information makes a short closest-approach bullet dodge cheap.
        for projectile in state.projectiles:
            if projectile.source_drone_id == drone.id:
                continue
            rx = projectile.position[0] - drone.position[0]
            ry = projectile.position[1] - drone.position[1]
            rvx = projectile.velocity[0] - drone.velocity[0]
            rvy = projectile.velocity[1] - drone.velocity[1]
            speed_squared = rvx * rvx + rvy * rvy
            if speed_squared < EPSILON:
                continue
            closest_time = min(1.0, max(0.0, -(rx * rvx + ry * rvy) / speed_squared))
            miss_x = rx + rvx * closest_time
            miss_y = ry + rvy * closest_time
            miss = hypot(miss_x, miss_y)
            if miss < 1.35:
                length = sqrt(speed_squared)
                side = 1.0 if drone.id % 2 == 0 else -1.0
                ax += -rvy / length * side * spec.max_acceleration * 2.0
                ay += rvx / length * side * spec.max_acceleration * 2.0
        return (ax, ay)

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
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                root = sqrt(discriminant)
                times.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        time = min((value for value in times if value > 0.0), default=0.0)
        return (target.position[0] + vx * time, target.position[1] + vy * time)

    def _clear_shot(self, shooter, aim, state):
        if self._blocked(shooter.position, aim):
            return False
        for friend in state.own_drones:
            if friend.id != shooter.id and friend.status is DroneStatus.ACTIVE:
                if segment_point_distance(shooter.position, aim, friend.position) <= 0.8:
                    return False
        return True

    def _fire_target(self, tank, enemies):
        candidates = []
        for enemy in enemies:
            value = self.specs[enemy.drone_type].point_value
            candidates.append((distance(tank.position, enemy.position) - 6.0 * value, enemy.id, enemy))
        return min(candidates, default=(0.0, 0, None))[2]

    def step(self, state):
        own = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        transports = [drone for drone in own if drone.drone_type is DroneType.TRANSPORT]
        enemy_transports = [drone for drone in enemies if drone.drone_type is DroneType.TRANSPORT]
        actions = {}

        for drone in own:
            target = self._goal_target(drone)
            if drone.drone_type is DroneType.SCOUT:
                if drone.id % 2 == 0 and transports:
                    charge = min(transports, key=lambda item: (distance(drone.position, item.position), item.id))
                    offset = ((drone.id % 5) - 2) * 1.25
                    target = (
                        charge.position[0] + self.forward * 2.8,
                        min(self.height - 0.6, max(0.6, charge.position[1] + offset)),
                    )
                    threats = [enemy for enemy in enemies if distance(enemy.position, charge.position) < 8.0]
                    if threats:
                        target = min(threats, key=lambda item: (distance(item.position, charge.position), item.id)).position
                elif enemy_transports:
                    victim = min(enemy_transports, key=lambda item: (distance(drone.position, item.position), item.id))
                    lead = min(1.5, distance(drone.position, victim.position) / max(1.0, self.specs[drone.drone_type].max_speed))
                    target = (victim.position[0] + victim.velocity[0] * lead, victim.position[1] + victim.velocity[1] * lead)
            elif drone.drone_type is DroneType.TANK:
                if transports:
                    charge = min(transports, key=lambda item: (distance(drone.position, item.position), item.id))
                    target = (charge.position[0] - self.forward * 4.5, charge.position[1])
                else:
                    target = (self.width * 0.42 if self.team is Team.A else self.width * 0.58, self.lanes.get(drone.id, 30.0))

            command = {"acceleration": self._steer(drone, target, state)}
            if (
                drone.drone_type is DroneType.TANK
                and drone.shots_remaining
                and drone.next_fire_time is not None
                and state.time + EPSILON >= drone.next_fire_time
            ):
                victim = self._fire_target(drone, enemies)
                if victim is not None:
                    aim = self._lead(drone, victim)
                    if self._clear_shot(drone, aim, state):
                        command["fire_direction"] = (aim[0] - drone.position[0], aim[1] - drone.position[1])
            actions[drone.id] = command
        return actions
