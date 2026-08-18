"""A compact defensive convoy controller for SwarmBench v2."""

from math import hypot, sqrt

from swarmbench import BaseSwarmController, CircleObstacle, DroneStatus, DroneType, Team


EPSILON = 1.0e-9


def gap(left, right):
    return hypot(left[0] - right[0], left[1] - right[1])


def point_segment_gap(point, start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared < EPSILON:
        return gap(point, start)
    amount = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    amount = min(1.0, max(0.0, amount))
    return gap(point, (start[0] + amount * dx, start[1] + amount * dy))


def segment_box(start, end, obstacle, padding=0.0):
    enter, leave = 0.0, 1.0
    for origin, change, low, high in (
        (start[0], end[0] - start[0], obstacle.x_min - padding, obstacle.x_max + padding),
        (start[1], end[1] - start[1], obstacle.y_min - padding, obstacle.y_max + padding),
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
        self.specs = dict(game_info.drone_specs)
        self.shot_speed = game_info.weapon_spec.projectile_speed
        self.forward = 1.0 if self.team is Team.A else -1.0
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        ordered = sorted(game_info.own_initial_drones, key=lambda item: (item.position[1], item.id))
        self.lanes = {
            drone.id: self.goal.y_min + 0.8 + (self.goal.y_max - self.goal.y_min - 1.6) * (index + 0.5) / len(ordered)
            for index, drone in enumerate(ordered)
        }

    def _goal(self, drone):
        return (self.goal.center[0], self.lanes.get(drone.id, self.goal.center[1]))

    def _obstacle_shape(self, obstacle):
        if isinstance(obstacle, CircleObstacle):
            return obstacle.center, obstacle.radius
        center = ((obstacle.x_min + obstacle.x_max) * 0.5, (obstacle.y_min + obstacle.y_max) * 0.5)
        radius = hypot(obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min) * 0.5
        return center, radius

    def _line_blocked(self, start, end, padding=0.0):
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                if point_segment_gap(obstacle.center, start, end) <= obstacle.radius + padding:
                    return True
            elif segment_box(start, end, obstacle, padding):
                return True
        return False

    def _move(self, drone, target, state):
        spec = self.specs[drone.drone_type]
        dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
        distance = hypot(dx, dy)
        if distance < EPSILON:
            direction = (self.forward, 0.0)
            wanted = (0.0, 0.0)
        else:
            direction = (dx / distance, dy / distance)
            speed = min(spec.max_speed, sqrt(2.0 * spec.max_acceleration * distance))
            wanted = (direction[0] * speed, direction[1] * speed)
        ax = 2.3 * (wanted[0] - drone.velocity[0])
        ay = 2.3 * (wanted[1] - drone.velocity[1])

        for obstacle in self.obstacles:
            center, radius = self._obstacle_shape(obstacle)
            ox, oy = center[0] - drone.position[0], center[1] - drone.position[1]
            along = ox * direction[0] + oy * direction[1]
            lateral = ox * -direction[1] + oy * direction[0]
            if -0.3 < along < 8.0 and abs(lateral) < radius + 1.2:
                side = -1.0 if lateral > 0.0 else 1.0
                if abs(lateral) < 0.05:
                    side = 1.0 if drone.id % 2 else -1.0
                strength = spec.max_acceleration * 1.8 * (1.0 - max(0.0, along) / 8.0)
                ax += -direction[1] * side * strength
                ay += direction[0] * side * strength
            center_distance = hypot(ox, oy)
            surface = center_distance - radius
            if EPSILON < surface < 2.2:
                force = spec.max_acceleration * (2.2 - surface) / 2.2
                ax -= ox / center_distance * force
                ay -= oy / center_distance * force

        for friend in state.own_drones:
            if friend.id == drone.id or friend.status is not DroneStatus.ACTIVE:
                continue
            rx, ry = drone.position[0] - friend.position[0], drone.position[1] - friend.position[1]
            rvx, rvy = drone.velocity[0] - friend.velocity[0], drone.velocity[1] - friend.velocity[1]
            velocity_squared = rvx * rvx + rvy * rvy
            time = min(0.8, max(0.0, -(rx * rvx + ry * rvy) / velocity_squared)) if velocity_squared else 0.0
            fx, fy = rx + rvx * time, ry + rvy * time
            separation = hypot(fx, fy)
            if EPSILON < separation < 2.1:
                force = spec.max_acceleration * 1.9 * (2.1 - separation) / 2.1
                ax += fx / separation * force
                ay += fy / separation * force

        for projectile in state.projectiles:
            if projectile.source_drone_id == drone.id:
                continue
            rx, ry = projectile.position[0] - drone.position[0], projectile.position[1] - drone.position[1]
            rvx, rvy = projectile.velocity[0] - drone.velocity[0], projectile.velocity[1] - drone.velocity[1]
            velocity_squared = rvx * rvx + rvy * rvy
            if velocity_squared < EPSILON:
                continue
            time = min(1.1, max(0.0, -(rx * rvx + ry * rvy) / velocity_squared))
            miss = hypot(rx + rvx * time, ry + rvy * time)
            if miss < 1.4:
                length = sqrt(velocity_squared)
                side = 1.0 if drone.id % 2 else -1.0
                ax += -rvy / length * side * spec.max_acceleration * 2.1
                ay += rvx / length * side * spec.max_acceleration * 2.1
        return (ax, ay)

    def _lead(self, shooter, target):
        rx, ry = target.position[0] - shooter.position[0], target.position[1] - shooter.position[1]
        vx, vy = target.velocity
        a = vx * vx + vy * vy - self.shot_speed * self.shot_speed
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

    def _shot(self, tank, enemies, state):
        if not enemies or not tank.shots_remaining or tank.next_fire_time is None or state.time < tank.next_fire_time:
            return None
        victim = min(
            enemies,
            key=lambda enemy: (
                gap(tank.position, enemy.position) - 5.0 * self.specs[enemy.drone_type].point_value,
                enemy.id,
            ),
        )
        aim = self._lead(tank, victim)
        if self._line_blocked(tank.position, aim):
            return None
        if any(
            friend.id != tank.id
            and friend.status is DroneStatus.ACTIVE
            and point_segment_gap(friend.position, tank.position, aim) < 0.82
            for friend in state.own_drones
        ):
            return None
        return (aim[0] - tank.position[0], aim[1] - tank.position[1])

    def step(self, state):
        own = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        transports = [drone for drone in own if drone.drone_type is DroneType.TRANSPORT]
        scouts = [drone for drone in own if drone.drone_type is DroneType.SCOUT]
        enemy_transports = [drone for drone in enemies if drone.drone_type is DroneType.TRANSPORT]
        scout_rank = {drone.id: index for index, drone in enumerate(sorted(scouts, key=lambda item: item.id))}
        actions = {}

        for drone in own:
            target = self._goal(drone)
            if drone.drone_type is DroneType.SCOUT:
                rank = scout_rank[drone.id]
                if rank % 3 == 0 and enemies:
                    danger = min(
                        enemies,
                        key=lambda enemy: (gap(enemy.position, self.own_goal.center) - 3.0 * self.specs[enemy.drone_type].point_value, enemy.id),
                    )
                    lead = min(1.2, gap(drone.position, danger.position) / max(1.0, self.specs[DroneType.SCOUT].max_speed))
                    target = (danger.position[0] + danger.velocity[0] * lead, danger.position[1] + danger.velocity[1] * lead)
                elif transports:
                    charge = transports[rank % len(transports)]
                    target = (
                        charge.position[0] + self.forward * (2.4 + 0.4 * (rank % 2)),
                        min(self.height - 0.6, max(0.6, charge.position[1] + ((rank % 4) - 1.5) * 1.1)),
                    )
                    close = [enemy for enemy in enemies if gap(enemy.position, charge.position) < 7.0]
                    if close:
                        target = min(close, key=lambda enemy: (gap(enemy.position, charge.position), enemy.id)).position
                elif enemy_transports:
                    target = min(enemy_transports, key=lambda enemy: (gap(drone.position, enemy.position), enemy.id)).position
            elif drone.drone_type is DroneType.TANK:
                if transports:
                    charge = min(transports, key=lambda item: (gap(drone.position, item.position), item.id))
                    target = (charge.position[0] - self.forward * 5.0, charge.position[1])
                else:
                    target = (self.width * (0.40 if self.team is Team.A else 0.60), self.lanes.get(drone.id, self.height * 0.5))

            command = {"acceleration": self._move(drone, target, state)}
            if drone.drone_type is DroneType.TANK:
                fire = self._shot(drone, enemies, state)
                if fire is not None:
                    command["fire_direction"] = fire
            actions[drone.id] = command
        return actions
