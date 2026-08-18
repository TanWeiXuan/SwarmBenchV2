"""A lane-running controller with light raiding and midfield tank support."""

from math import hypot, sqrt

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType, Team


TINY = 1.0e-9


def distance(left, right):
    return hypot(left[0] - right[0], left[1] - right[1])


def point_segment_distance(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared < TINY:
        return distance(point, start)
    amount = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    amount = min(1.0, max(0.0, amount))
    closest = (start[0] + amount * dx, start[1] + amount * dy)
    return distance(point, closest)


class SwarmController(BaseSwarmController):
    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)
        self.projectile_speed = game_info.weapon_spec.projectile_speed
        self.direction = 1.0 if self.team is Team.A else -1.0
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        ordered = sorted(game_info.own_initial_drones, key=lambda drone: (drone.drone_type.value, drone.id))
        self.lanes = {
            drone.id: self.goal.y_min
            + 0.7
            + (self.goal.y_max - self.goal.y_min - 1.4) * (index + 0.5) / max(1, len(ordered))
            for index, drone in enumerate(ordered)
        }

    def _shape(self, obstacle):
        if isinstance(obstacle, CircleObstacle):
            return obstacle.center, obstacle.radius
        center = ((obstacle.x_min + obstacle.x_max) * 0.5, (obstacle.y_min + obstacle.y_max) * 0.5)
        radius = hypot(obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min) * 0.5
        return center, radius

    def _goal_target(self, drone):
        lane = self.lanes.get(drone.id, self.goal.center[1])
        return (self.goal.center[0], min(self.goal.y_max - 0.65, max(self.goal.y_min + 0.65, lane)))

    def _steer(self, drone, target, state, caution=1.0):
        spec = self.specs[drone.drone_type]
        dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
        remaining = hypot(dx, dy)
        if remaining < TINY:
            forward = (self.direction, 0.0)
            desired = (0.0, 0.0)
        else:
            forward = (dx / remaining, dy / remaining)
            desired_speed = min(spec.max_speed, sqrt(2.0 * spec.max_acceleration * remaining))
            desired = (forward[0] * desired_speed, forward[1] * desired_speed)
        ax = 2.15 * (desired[0] - drone.velocity[0])
        ay = 2.15 * (desired[1] - drone.velocity[1])

        for obstacle in self.obstacles:
            center, radius = self._shape(obstacle)
            ox, oy = center[0] - drone.position[0], center[1] - drone.position[1]
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
            if TINY < surface_gap < 2.6:
                force = caution * spec.max_acceleration * (2.6 - surface_gap) / 2.6
                ax -= ox / center_gap * force
                ay -= oy / center_gap * force

        for friend in state.own_drones:
            if friend.id == drone.id or friend.status is not DroneStatus.ACTIVE:
                continue
            rx, ry = drone.position[0] - friend.position[0], drone.position[1] - friend.position[1]
            rvx = drone.velocity[0] - friend.velocity[0]
            rvy = drone.velocity[1] - friend.velocity[1]
            speed_squared = rvx * rvx + rvy * rvy
            closest_time = min(0.75, max(0.0, -(rx * rvx + ry * rvy) / speed_squared)) if speed_squared else 0.0
            miss_x, miss_y = rx + rvx * closest_time, ry + rvy * closest_time
            miss = hypot(miss_x, miss_y)
            if TINY < miss < 2.0:
                force = spec.max_acceleration * 1.8 * (2.0 - miss) / 2.0
                ax += miss_x / miss * force
                ay += miss_y / miss * force

        for projectile in state.projectiles:
            rx = projectile.position[0] - drone.position[0]
            ry = projectile.position[1] - drone.position[1]
            rvx = projectile.velocity[0] - drone.velocity[0]
            rvy = projectile.velocity[1] - drone.velocity[1]
            speed_squared = rvx * rvx + rvy * rvy
            if speed_squared < TINY:
                continue
            closest_time = min(1.0, max(0.0, -(rx * rvx + ry * rvy) / speed_squared))
            if hypot(rx + rvx * closest_time, ry + rvy * closest_time) < 1.3:
                speed = sqrt(speed_squared)
                side = 1.0 if drone.id % 2 == 0 else -1.0
                ax += -rvy / speed * side * spec.max_acceleration * 1.9
                ay += rvx / speed * side * spec.max_acceleration * 1.9
        return (ax, ay)

    def _blocked(self, start, end, padding=0.0):
        for obstacle in self.obstacles:
            center, radius = self._shape(obstacle)
            if point_segment_distance(center, start, end) <= radius + padding:
                return True
        return False

    def _lead(self, shooter, target):
        rx = target.position[0] - shooter.position[0]
        ry = target.position[1] - shooter.position[1]
        vx, vy = target.velocity
        a = vx * vx + vy * vy - self.projectile_speed * self.projectile_speed
        b = 2.0 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry
        times = []
        if abs(a) < TINY:
            if abs(b) > TINY:
                times.append(-c / b)
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                root = sqrt(discriminant)
                times.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        flight_time = min((value for value in times if value > 0.0), default=0.0)
        return (target.position[0] + vx * flight_time, target.position[1] + vy * flight_time)

    def _fire(self, tank, enemies, state):
        if not tank.shots_remaining or tank.next_fire_time is None or state.time + TINY < tank.next_fire_time:
            return None
        candidates = sorted(
            enemies,
            key=lambda enemy: (
                distance(tank.position, enemy.position) - 4.0 * self.specs[enemy.drone_type].point_value,
                enemy.id,
            ),
        )
        for target in candidates:
            aim = self._lead(tank, target)
            if self._blocked(tank.position, aim):
                continue
            unsafe = any(
                friend.id != tank.id
                and friend.status is DroneStatus.ACTIVE
                and point_segment_distance(friend.position, tank.position, aim) < 0.85
                for friend in state.own_drones
            )
            if not unsafe:
                return (aim[0] - tank.position[0], aim[1] - tank.position[1])
        return None

    def step(self, state):
        own = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        transports = [drone for drone in own if drone.drone_type is DroneType.TRANSPORT]
        enemy_transports = [drone for drone in enemies if drone.drone_type is DroneType.TRANSPORT]
        scouts = sorted((drone for drone in own if drone.drone_type is DroneType.SCOUT), key=lambda drone: drone.id)
        scout_rank = {drone.id: rank for rank, drone in enumerate(scouts)}
        actions = {}

        for drone in own:
            target = self._goal_target(drone)
            caution = 1.0
            if drone.drone_type is DroneType.SCOUT:
                rank = scout_rank[drone.id]
                if rank % 4 == 0 and enemy_transports:
                    victim = min(enemy_transports, key=lambda enemy: (distance(drone.position, enemy.position), enemy.id))
                    lead_time = min(1.0, distance(drone.position, victim.position) / max(1.0, self.specs[DroneType.SCOUT].max_speed))
                    target = (
                        victim.position[0] + victim.velocity[0] * lead_time,
                        victim.position[1] + victim.velocity[1] * lead_time,
                    )
                    caution = 0.8
                elif transports and rank % 4 == 1:
                    charge = transports[rank % len(transports)]
                    target = (
                        charge.position[0] + self.direction * 2.8,
                        min(self.height - 0.7, max(0.7, charge.position[1] + (1.2 if rank % 2 else -1.2))),
                    )
            elif drone.drone_type is DroneType.TANK:
                midfield_x = self.width * (0.46 if self.team is Team.A else 0.54)
                target = (midfield_x, self.lanes.get(drone.id, self.height * 0.5))
                if transports:
                    lead_transport = max(transports, key=lambda item: (self.direction * item.position[0], -item.id))
                    desired_x = lead_transport.position[0] - self.direction * 5.5
                    if self.direction * (desired_x - midfield_x) > 0.0:
                        target = (desired_x, lead_transport.position[1])

            command = {"acceleration": self._steer(drone, target, state, caution)}
            if drone.drone_type is DroneType.TANK and enemies:
                fire_direction = self._fire(drone, enemies, state)
                if fire_direction is not None:
                    command["fire_direction"] = fire_direction
            actions[drone.id] = command
        return actions
