"""Luna_xHigh MPC: centralized tactical rollouts and global tank fire control.

The controller deliberately separates tactical decisions from continuous steering.
Every half second it builds a small beam of *joint* plans.  Each plan assigns all
vehicles a role (advance, flank, intercept, screen, cover, or retreat), rolls the
formation forward for four seconds with a lightweight deterministic model, and is
scored on progress, expected scoring, trades, exposure, interception risk, and
ammunition.  The first action of the best plan is then executed until the next
receding-horizon decision.

Tanks are assigned targets by one global scheduler rather than independently.  A
target can normally receive at most one scheduled shot; assignment value combines
predicted hit probability, target value, threat, line of fire, and ammunition cost.
"""

from __future__ import annotations

from math import exp, hypot, isfinite, sqrt

from swarmbench import (
    BaseSwarmController,
    CircleObstacle,
    DroneStatus,
    DroneType,
    RectangleObstacle,
    Team,
)


EPS = 1.0e-9
CONTACT_RADIUS = 0.75
DRONE_RADIUS = 0.25
PHYSICS_DT = 0.20
ROLLOUT_HORIZON = 4.0
REPLAN_STEPS = 5


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _distance(left, right) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def _clip(vector, maximum: float):
    x, y = float(vector[0]), float(vector[1])
    magnitude = hypot(x, y)
    if magnitude <= maximum or magnitude < EPS:
        return (x, y)
    scale = maximum / magnitude
    return (x * scale, y * scale)


def _unit(vector):
    magnitude = hypot(vector[0], vector[1])
    if magnitude < EPS:
        return (0.0, 0.0)
    return (vector[0] / magnitude, vector[1] / magnitude)


def _lerp(start, end, amount: float):
    return (
        start[0] + (end[0] - start[0]) * amount,
        start[1] + (end[1] - start[1]) * amount,
    )


def _segment_point_distance(start, end, point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared < EPS:
        return _distance(start, point)
    amount = _clamp(
        ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared,
        0.0,
        1.0,
    )
    return _distance(
        (start[0] + amount * dx, start[1] + amount * dy),
        point,
    )


def _segment_box_intersects(start, end, x_min, x_max, y_min, y_max) -> bool:
    enter, leave = 0.0, 1.0
    for origin, change, low, high in (
        (start[0], end[0] - start[0], x_min, x_max),
        (start[1], end[1] - start[1], y_min, y_max),
    ):
        if abs(change) < EPS:
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


def _swept_contact(a0, a1, b0, b1, radius: float) -> float | None:
    """Earliest normalized contact of two linearly swept points."""
    rx, ry = a0[0] - b0[0], a0[1] - b0[1]
    vx = (a1[0] - a0[0]) - (b1[0] - b0[0])
    vy = (a1[1] - a0[1]) - (b1[1] - b0[1])
    c = rx * rx + ry * ry - radius * radius
    if c <= 0.0:
        return 0.0
    quadratic = vx * vx + vy * vy
    if quadratic < EPS:
        return None
    linear = 2.0 * (rx * vx + ry * vy)
    discriminant = linear * linear - 4.0 * quadratic * c
    if discriminant < 0.0:
        return None
    contact = (-linear - sqrt(discriminant)) / (2.0 * quadratic)
    return contact if 0.0 <= contact <= 1.0 else None


class SwarmController(BaseSwarmController):
    """A deterministic receding-horizon controller for the full formation."""

    def initialize(self, game_info):
        self.team = game_info.team
        self.direction = 1.0 if self.team is Team.A else -1.0
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.width = game_info.arena_width
        self.height = game_info.arena_height
        self.obstacles = tuple(game_info.obstacles)
        self.specs = dict(game_info.drone_specs)
        self.weapon = game_info.weapon_spec

        ordered = sorted(
            game_info.own_initial_drones,
            key=lambda drone: (drone.drone_type.value, drone.position[1], drone.id),
        )
        self.lanes = {}
        for index, drone in enumerate(ordered):
            # Initial lanes remain stable across replans, which keeps the joint
            # formation from oscillating when two profiles score similarly.
            fraction = (index + 0.5) / max(1, len(ordered))
            self.lanes[drone.id] = 3.0 + (self.height - 6.0) * fraction

        self.steps = 0
        self.plan = None
        self.plan_score = float("-inf")
        self.last_plan_name = "uninitialized"

    # ------------------------------------------------------------------
    # Geometry and prediction helpers
    # ------------------------------------------------------------------
    def _obstacle_hit(self, start, end, obstacle, padding: float = 0.0) -> bool:
        if isinstance(obstacle, CircleObstacle):
            return _segment_point_distance(start, end, obstacle.center) <= obstacle.radius + padding
        return _segment_box_intersects(
            start,
            end,
            obstacle.x_min - padding,
            obstacle.x_max + padding,
            obstacle.y_min - padding,
            obstacle.y_max + padding,
        )

    def _blocked(self, start, end, padding: float = 0.0) -> bool:
        return any(self._obstacle_hit(start, end, obstacle, padding) for obstacle in self.obstacles)

    def _point_blocked(self, point, padding: float = DRONE_RADIUS) -> bool:
        return self._blocked(point, point, padding)

    def _goal_lane(self, drone_id: int, flank: int = 0, lane_bias: float = 0.0) -> float:
        base = self.lanes.get(drone_id, self.goal.center[1])
        if flank > 0:
            base = 0.55 * base + 0.45 * (self.goal.y_max - 1.0)
        elif flank < 0:
            base = 0.55 * base + 0.45 * (self.goal.y_min + 1.0)
        return _clamp(base + lane_bias, self.goal.y_min + 0.75, self.goal.y_max - 0.75)

    def _goal_target(self, drone, plan):
        return (self.goal.center[0], self._goal_lane(drone.id, plan["flank"]))

    def _lead_point(self, shooter, target):
        rx = target.position[0] - shooter.position[0]
        ry = target.position[1] - shooter.position[1]
        vx, vy = target.velocity
        speed = self.weapon.projectile_speed
        a = vx * vx + vy * vy - speed * speed
        b = 2.0 * (rx * vx + ry * vy)
        c = rx * rx + ry * ry
        roots = []
        if abs(a) < EPS:
            if abs(b) > EPS:
                roots.append(-c / b)
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                root = sqrt(discriminant)
                roots.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
        flight = min((value for value in roots if value > 0.0), default=0.0)
        flight = min(flight, 2.5)
        return (target.position[0] + vx * flight, target.position[1] + vy * flight)

    def _clear_shot(self, shooter, target, state) -> bool:
        aim = self._lead_point(shooter, target)
        if self._blocked(shooter.position, aim, 0.0):
            return False
        for friend in state.own_drones:
            if friend.id == shooter.id or friend.status is not DroneStatus.ACTIVE:
                continue
            if _segment_point_distance(shooter.position, aim, friend.position) < 0.82:
                return False
        return True

    def _hit_probability(self, source, target, source_position=None, target_position=None) -> float:
        start = source_position if source_position is not None else source.position
        end_target = target_position if target_position is not None else target.position
        distance = _distance(start, end_target)
        if distance < EPS:
            return 0.0
        blocked = self._blocked(start, end_target, 0.0)
        # A clear 20 m/s shot is very reliable at the arena's normal ranges;
        # a blocked shot is retained as a tiny probability for score smoothness
        # but is never selected for execution by _clear_shot.
        base = 0.93 * exp(-distance / 115.0)
        if blocked:
            base *= 0.04
        if target.drone_type is DroneType.SCOUT:
            base *= 0.94
        return _clamp(base, 0.0, 0.96)

    def _threat(self, enemy) -> float:
        # Progress is measured toward our side, so larger values mean more
        # urgent interception.  Transport value is intentionally dominant.
        progress = (self.width - enemy.position[0]) if self.team is Team.A else enemy.position[0]
        value = self.specs[enemy.drone_type].point_value
        goal_distance = _distance(enemy.position, self.own_goal.center)
        return 1.8 * value + 0.055 * progress + max(0.0, 18.0 - goal_distance) * 0.15

    # ------------------------------------------------------------------
    # Centralized tactical plan generation
    # ------------------------------------------------------------------
    def _profiles(self, state):
        behind = state.own_score < state.opponent_score
        late = state.time > 65.0
        profiles = [
            {"name": "surge", "flank": 0, "intercept": 0.15, "escort": 0.15, "tank": "cover", "bias": 0.4},
            {"name": "flank_high", "flank": 1, "intercept": 0.20, "escort": 0.20, "tank": "cover", "bias": 0.1},
            {"name": "flank_low", "flank": -1, "intercept": 0.20, "escort": 0.20, "tank": "cover", "bias": 0.1},
            {"name": "convoy_screen", "flank": 0, "intercept": 0.20, "escort": 0.50, "tank": "trail", "bias": 0.3},
            {"name": "hunter", "flank": 0, "intercept": 0.55, "escort": 0.10, "tank": "forward", "bias": 0.0},
            {"name": "defensive_intercept", "flank": 0, "intercept": 0.70, "escort": 0.25, "tank": "hold", "bias": -0.1},
            {"name": "retreat_cover", "flank": 0, "intercept": 0.80, "escort": 0.20, "tank": "retreat", "bias": -0.2},
        ]
        if behind and not late:
            profiles.extend(
                [
                    {"name": "all_in_high", "flank": 1, "intercept": 0.35, "escort": 0.10, "tank": "forward", "bias": 0.7},
                    {"name": "all_in_low", "flank": -1, "intercept": 0.35, "escort": 0.10, "tank": "forward", "bias": 0.7},
                ]
            )
        if not behind and late:
            profiles.append(
                {"name": "lockdown", "flank": 0, "intercept": 0.90, "escort": 0.35, "tank": "hold", "bias": 0.8}
            )
        return profiles

    def _assign_threats(self, scouts, enemies, count: int):
        if count <= 0 or not scouts or not enemies:
            return {}
        threats = sorted(enemies, key=lambda enemy: (-self._threat(enemy), enemy.id))[:count]
        remaining = list(scouts)
        assignments = {}
        # Greedy minimum-cost matching is sufficient for the small swarm and
        # keeps the planner deterministic without a heavy dependency.
        for enemy in threats:
            if not remaining:
                break
            scout = min(
                remaining,
                key=lambda drone: (_distance(drone.position, enemy.position) - 0.25 * self._threat(enemy), drone.id),
            )
            assignments[scout.id] = enemy.id
            remaining.remove(scout)
        return assignments

    def _build_plan(self, state, profile):
        own = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        transports = sorted((drone for drone in own if drone.drone_type is DroneType.TRANSPORT), key=lambda drone: drone.id)
        scouts = sorted((drone for drone in own if drone.drone_type is DroneType.SCOUT), key=lambda drone: drone.id)
        tanks = sorted((drone for drone in own if drone.drone_type is DroneType.TANK), key=lambda drone: drone.id)

        plan = {
            "name": profile["name"],
            "flank": profile["flank"],
            "roles": {},
            "targets": {},
            "supports": {},
            "offsets": {},
            "waypoints": {},
            "profile": profile,
            "shot_targets": {},
        }

        for rank, drone in enumerate(transports):
            plan["roles"][drone.id] = "advance"
            lane_bias = ((rank % 3) - 1) * 1.0
            plan["waypoints"][drone.id] = (self.goal.center[0], self._goal_lane(drone.id, profile["flank"], lane_bias))

        threat_assignments = self._assign_threats(
            scouts,
            enemies,
            min(len(scouts), int(round(len(scouts) * profile["intercept"]))),
        )
        enemy_by_id = {enemy.id: enemy for enemy in enemies}
        for scout in scouts:
            plan["roles"][scout.id] = "advance"
            plan["waypoints"][scout.id] = (self.goal.center[0], self._goal_lane(scout.id, profile["flank"]))

        for scout_id, enemy_id in threat_assignments.items():
            plan["roles"][scout_id] = "intercept"
            plan["targets"][scout_id] = enemy_id

        remaining_scouts = [scout for scout in scouts if scout.id not in threat_assignments]
        escort_count = min(len(remaining_scouts), int(round(len(scouts) * profile["escort"])))
        for index, scout in enumerate(remaining_scouts[:escort_count]):
            if transports:
                transport = transports[index % len(transports)]
                plan["roles"][scout.id] = "screen"
                plan["supports"][scout.id] = transport.id
                plan["offsets"][scout.id] = ((index % 3) - 1) * 1.5

        # In hunter/all-in profiles, unassigned Scouts raid the most valuable
        # available transport instead of independently picking random targets.
        enemy_transports = [enemy for enemy in enemies if enemy.drone_type is DroneType.TRANSPORT]
        if profile["intercept"] >= 0.5 and enemy_transports:
            raiders = [scout for scout in remaining_scouts[escort_count:] if scout.id not in threat_assignments]
            for scout, enemy in zip(
                raiders,
                sorted(enemy_transports, key=lambda item: (-self._threat(item), item.id)),
                strict=False,
            ):
                plan["roles"][scout.id] = "raid"
                plan["targets"][scout.id] = enemy.id

        for rank, tank in enumerate(tanks):
            tank_lane = self.goal.center[1] + (rank - (len(tanks) - 1) / 2.0) * 8.0
            tank_lane = _clamp(tank_lane, 5.0, self.height - 5.0)
            style = profile["tank"]
            if style == "retreat":
                x = self.width * (0.31 if self.team is Team.A else 0.69)
                role = "retreat"
            elif style == "forward":
                x = self.width * (0.54 if self.team is Team.A else 0.46)
                role = "cover"
            elif style == "trail" and transports:
                lead = max(transports, key=lambda item: self.direction * item.position[0])
                x = lead.position[0] - self.direction * 5.5
                role = "trail"
            else:
                x = self.width * (0.43 if self.team is Team.A else 0.57)
                role = "cover"
            plan["roles"][tank.id] = role
            plan["waypoints"][tank.id] = (_clamp(x, 8.0, self.width - 8.0), tank_lane)

        plan["shot_targets"] = self._schedule_weapons(state, plan)
        return plan

    def _schedule_weapons(self, state, plan):
        """Solve a tiny global assignment problem for currently known Tanks."""
        tanks = [
            drone
            for drone in state.own_drones
            if drone.status is DroneStatus.ACTIVE
            and drone.drone_type is DroneType.TANK
            and drone.shots_remaining
        ]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        if not tanks or not enemies:
            return {}

        tank_positions = {
            tank.id: plan["waypoints"].get(tank.id, tank.position)
            for tank in tanks
        }
        choices = {}
        for tank in tanks:
            scored = []
            for enemy in enemies:
                if self._blocked(tank_positions[tank.id], enemy.position, 0.0):
                    continue
                probability = self._hit_probability(
                    tank,
                    enemy,
                    source_position=tank_positions[tank.id],
                    target_position=enemy.position,
                )
                value = self.specs[enemy.drone_type].point_value
                score = probability * (2.4 * value + self._threat(enemy))
                score -= 0.20  # ammunition opportunity cost
                if score > 0.05:
                    scored.append((score, enemy.id))
            choices[tank.id] = sorted(scored, key=lambda item: (-item[0], item[1]))

        order = sorted(tanks, key=lambda tank: (tank.next_fire_time or 0.0, -tank.shots_remaining, tank.id))
        best_score = float("-inf")
        best_assignment = {}

        def search(index: int, used_targets: set[int], score: float, assignment: dict[int, int]):
            nonlocal best_score, best_assignment
            if index >= len(order):
                if score > best_score:
                    best_score = score
                    best_assignment = dict(assignment)
                return
            tank = order[index]
            # Holding fire is an explicit action in the assignment search.
            search(index + 1, used_targets, score, assignment)
            for option_score, enemy_id in choices[tank.id]:
                duplicate_penalty = 3.0 if enemy_id in used_targets else 0.0
                # Prefer unique targets.  The duplicate branch is retained only
                # when there are fewer useful enemies than Tanks.
                if enemy_id in used_targets and len(used_targets) < len(enemies):
                    continue
                assignment[tank.id] = enemy_id
                search(index + 1, used_targets | {enemy_id}, score + option_score - duplicate_penalty, assignment)
                assignment.pop(tank.id, None)

        search(0, set(), 0.0, {})
        return best_assignment

    # ------------------------------------------------------------------
    # Lightweight tactical MPC rollout
    # ------------------------------------------------------------------
    def _model_from_state(self, state):
        model = {}
        for drone in state.own_drones:
            if drone.status is DroneStatus.ACTIVE:
                spec = self.specs[drone.drone_type]
                model[drone.id] = {
                    "own": True,
                    "id": drone.id,
                    "type": drone.drone_type,
                    "position": tuple(drone.position),
                    "velocity": tuple(drone.velocity),
                    "max_speed": spec.max_speed,
                    "max_acceleration": spec.max_acceleration,
                    "value": spec.point_value,
                    "alive": True,
                }
        for drone in state.opponent_drones:
            if drone.status is DroneStatus.ACTIVE:
                spec = self.specs[drone.drone_type]
                model[drone.id] = {
                    "own": False,
                    "id": drone.id,
                    "type": drone.drone_type,
                    "position": tuple(drone.position),
                    "velocity": tuple(drone.velocity),
                    "max_speed": spec.max_speed,
                    "max_acceleration": spec.max_acceleration,
                    "value": spec.point_value,
                    "alive": True,
                }
        return model

    def _model_target(self, agent, model, plan):
        role = plan["roles"].get(agent["id"], "advance")
        if role in {"intercept", "raid"}:
            target = model.get(plan["targets"].get(agent["id"]))
            if target is not None and target["alive"]:
                lead = min(1.5, _distance(agent["position"], target["position"]) / max(1.0, agent["max_speed"]))
                return (
                    target["position"][0] + target["velocity"][0] * lead,
                    target["position"][1] + target["velocity"][1] * lead,
                )
        if role == "screen":
            support = model.get(plan["supports"].get(agent["id"]))
            if support is not None and support["alive"]:
                return (
                    support["position"][0] + self.direction * 2.7,
                    support["position"][1] + plan["offsets"].get(agent["id"], 0.0),
                )
        if role == "trail":
            support = min(
                (
                    item
                    for item in model.values()
                    if item["own"] and item["alive"] and item["type"] is DroneType.TRANSPORT
                ),
                key=lambda item: (-self.direction * item["position"][0], item["id"]),
                default=None,
            )
            if support is not None:
                return (support["position"][0] - self.direction * 5.5, support["position"][1])
        if role == "retreat":
            return (self.own_goal.center[0], agent["position"][1])
        return plan["waypoints"].get(agent["id"], (self.goal.center[0], self.goal.center[1]))

    def _advance_model_agent(self, agent, target, dt: float):
        old_position = agent["position"]
        dx, dy = target[0] - old_position[0], target[1] - old_position[1]
        remaining = hypot(dx, dy)
        if remaining < EPS:
            desired_velocity = (0.0, 0.0)
        else:
            desired_speed = min(agent["max_speed"], sqrt(2.0 * agent["max_acceleration"] * remaining))
            desired_velocity = (dx / remaining * desired_speed, dy / remaining * desired_speed)
        acceleration = _clip(
            (
                2.0 * (desired_velocity[0] - agent["velocity"][0]),
                2.0 * (desired_velocity[1] - agent["velocity"][1]),
            ),
            agent["max_acceleration"],
        )
        velocity = _clip(
            (
                agent["velocity"][0] + acceleration[0] * dt,
                agent["velocity"][1] + acceleration[1] * dt,
            ),
            agent["max_speed"],
        )
        position = (old_position[0] + velocity[0] * dt, old_position[1] + velocity[1] * dt)
        agent["velocity"] = velocity
        agent["position"] = position
        return old_position, position

    def _rollout(self, state, plan) -> float:
        model = self._model_from_state(state)
        initial_positions = {agent_id: agent["position"] for agent_id, agent in model.items()}
        projectiles = [
            {
                "position": tuple(projectile.position),
                "velocity": tuple(projectile.velocity),
                "source": projectile.source_drone_id,
            }
            for projectile in state.projectiles
        ]
        own_lost = 0.0
        enemy_destroyed = 0.0
        own_scored = 0.0
        enemy_scored = 0.0
        exposure = 0.0
        interception_risk = 0.0
        expected_shot_value = 0.0
        shots_used = 0
        shot_next = {}
        shot_left = {}
        for drone in state.own_drones:
            if drone.status is DroneStatus.ACTIVE and drone.drone_type is DroneType.TANK:
                shot_left[drone.id] = drone.shots_remaining or 0
                shot_next[drone.id] = max(0.0, (drone.next_fire_time or state.time) - state.time)

        steps = round(ROLLOUT_HORIZON / PHYSICS_DT)
        elapsed = 0.0
        for _ in range(steps):
            old_positions = {agent_id: agent["position"] for agent_id, agent in model.items() if agent["alive"]}
            segments = {}
            for agent in sorted(model.values(), key=lambda item: item["id"]):
                if not agent["alive"]:
                    continue
                if agent["own"]:
                    target = self._model_target(agent, model, plan)
                else:
                    y = _clamp(agent["position"][1], self.own_goal.y_min + 0.75, self.own_goal.y_max - 0.75)
                    target = (self.own_goal.center[0], y)
                segments[agent["id"]] = self._advance_model_agent(agent, target, PHYSICS_DT)

            # Existing projectiles are advanced continuously.  Obstacle shielding
            # is resolved before projectile/vehicle contact in the real engine.
            next_projectiles = []
            for projectile in projectiles:
                start = projectile["position"]
                end = (
                    start[0] + projectile["velocity"][0] * PHYSICS_DT,
                    start[1] + projectile["velocity"][1] * PHYSICS_DT,
                )
                if self._blocked(start, end, 0.0):
                    continue
                hit = False
                for agent_id, agent in sorted(model.items()):
                    if not agent["alive"] or agent_id == projectile["source"] or not agent["own"]:
                        continue
                    segment = segments.get(agent_id)
                    if segment is not None and _swept_contact(start, end, segment[0], segment[1], CONTACT_RADIUS) is not None:
                        agent["alive"] = False
                        own_lost += agent["value"]
                        hit = True
                        break
                    if agent["alive"] and _segment_point_distance(start, end, agent["position"]) < 1.8:
                        exposure += 0.35
                if not hit:
                    next_projectiles.append({"position": end, "velocity": projectile["velocity"], "source": projectile["source"]})
            projectiles = next_projectiles

            # Contacts and obstacle crashes are deliberately modeled as one-for-
            # one eliminations, matching the authoritative engine's trade rule.
            active_ids = [agent_id for agent_id, agent in sorted(model.items()) if agent["alive"]]
            for index, left_id in enumerate(active_ids):
                left = model[left_id]
                left_segment = segments.get(left_id, (left["position"], left["position"]))
                if any(
                    self._obstacle_hit(left_segment[0], left_segment[1], obstacle, DRONE_RADIUS)
                    for obstacle in self.obstacles
                ):
                    left["alive"] = False
                    if left["own"]:
                        own_lost += left["value"]
                    else:
                        enemy_destroyed += left["value"]
                    continue
                for right_id in active_ids[index + 1 :]:
                    right = model[right_id]
                    if not right["alive"]:
                        continue
                    right_segment = segments.get(right_id, (right["position"], right["position"]))
                    if _swept_contact(left_segment[0], left_segment[1], right_segment[0], right_segment[1], CONTACT_RADIUS) is not None:
                        left["alive"] = False
                        right["alive"] = False
                        if left["own"] and right["own"]:
                            own_lost += left["value"] + right["value"]
                        elif not left["own"] and not right["own"]:
                            enemy_destroyed += left["value"] + right["value"]
                        elif left["own"]:
                            own_lost += left["value"]
                            enemy_destroyed += right["value"]
                        else:
                            own_lost += right["value"]
                            enemy_destroyed += left["value"]
                        break

            for agent in model.values():
                if not agent["alive"]:
                    continue
                position = agent["position"]
                if agent["own"] and self.goal.contains(position):
                    agent["alive"] = False
                    own_scored += agent["value"]
                elif not agent["own"] and self.own_goal.contains(position):
                    agent["alive"] = False
                    enemy_scored += agent["value"]

            # Evaluate scheduled Tank shots at their predicted time.  This is an
            # expected-value term, not a claim that the approximate rollout can
            # replace the authoritative projectile resolver.
            elapsed += PHYSICS_DT
            for tank_id, target_id in plan["shot_targets"].items():
                if tank_id not in shot_left or shot_left[tank_id] <= 0 or elapsed + EPS < shot_next[tank_id]:
                    continue
                tank = model.get(tank_id)
                target = model.get(target_id)
                if tank is None or target is None or not tank["alive"] or not target["alive"]:
                    continue
                source_position = tank["position"]
                target_position = target["position"]
                probability = self._hit_probability(
                    self._snapshot_like(tank),
                    self._snapshot_like(target),
                    source_position=source_position,
                    target_position=target_position,
                )
                expected_shot_value += probability * target["value"]
                shots_used += 1
                shot_left[tank_id] -= 1
                shot_next[tank_id] += self.weapon.cooldown

            for own in model.values():
                if not own["own"] or not own["alive"]:
                    continue
                for enemy in model.values():
                    if enemy["own"] or not enemy["alive"]:
                        continue
                    distance = _distance(own["position"], enemy["position"])
                    if own["type"] is DroneType.TRANSPORT:
                        interception_risk += max(0.0, 10.0 - distance) / 10.0 * (1.0 + enemy["value"] * 0.25)
                    if _distance(enemy["position"], self.own_goal.center) < 18.0:
                        interception_risk += 0.035 * enemy["value"]

        own_transports = [
            agent
            for agent in model.values()
            if agent["own"] and agent["type"] is DroneType.TRANSPORT
        ]
        transport_progress = 0.0
        for transport in own_transports:
            start = initial_positions[transport["id"]]
            transport_progress += max(0.0, self.direction * (transport["position"][0] - start[0]))

        surviving_own_value = sum(agent["value"] for agent in model.values() if agent["own"] and agent["alive"])
        remaining_ammo = sum(shot_left.values())
        # The score intentionally uses separate terms so the tactical decision
        # remains interpretable when a new profile is added.
        return (
            18.0 * own_scored
            - 9.0 * enemy_scored
            - 2.7 * own_lost
            + 4.2 * enemy_destroyed
            + 2.8 * expected_shot_value
            + 0.72 * transport_progress
            + 0.06 * surviving_own_value
            - 1.15 * exposure
            - 1.05 * interception_risk
            + 0.12 * remaining_ammo
            - 0.10 * shots_used
        )

    def _snapshot_like(self, agent):
        # _hit_probability only reads position, velocity, and drone_type.
        return type(
            "PredictedDrone",
            (),
            {
                "position": agent["position"],
                "velocity": agent["velocity"],
                "drone_type": agent["type"],
            },
        )()

    def _choose_plan(self, state):
        candidates = []
        for profile in self._profiles(state):
            plan = self._build_plan(state, profile)
            utility = self._rollout(state, plan)
            score_pressure = state.own_score - state.opponent_score
            if score_pressure < 0:
                utility += profile["bias"] * 1.5
            elif state.time > 65.0:
                utility -= profile["bias"] * 0.5
            candidates.append((utility, profile["name"], plan))
        if not candidates:
            return {"roles": {}, "targets": {}, "supports": {}, "offsets": {}, "waypoints": {}, "shot_targets": {}, "flank": 0}
        candidates.sort(key=lambda item: (-item[0], item[1]))
        self.plan_score = candidates[0][0]
        self.last_plan_name = candidates[0][1]
        return candidates[0][2]

    # ------------------------------------------------------------------
    # First-action execution from the selected joint plan
    # ------------------------------------------------------------------
    def _runtime_target(self, drone, state, plan):
        role = plan["roles"].get(drone.id, "advance")
        active_enemies = {enemy.id: enemy for enemy in state.opponent_drones if enemy.status is DroneStatus.ACTIVE}
        if role in {"intercept", "raid"}:
            target = active_enemies.get(plan["targets"].get(drone.id))
            if target is not None:
                lead = min(1.5, _distance(drone.position, target.position) / max(1.0, self.specs[drone.drone_type].max_speed))
                return (
                    target.position[0] + target.velocity[0] * lead,
                    target.position[1] + target.velocity[1] * lead,
                )
        if role == "screen":
            support_id = plan["supports"].get(drone.id)
            support = next(
                (item for item in state.own_drones if item.id == support_id and item.status is DroneStatus.ACTIVE),
                None,
            )
            if support is not None:
                return (
                    support.position[0] + self.direction * 2.7,
                    support.position[1] + plan["offsets"].get(drone.id, 0.0),
                )
        if role == "trail":
            transports = [item for item in state.own_drones if item.status is DroneStatus.ACTIVE and item.drone_type is DroneType.TRANSPORT]
            if transports:
                lead = max(transports, key=lambda item: (self.direction * item.position[0], -item.id))
                return (lead.position[0] - self.direction * 5.5, lead.position[1])
        if role == "retreat":
            return (self.own_goal.center[0], drone.position[1])
        if drone.id in plan["waypoints"]:
            return plan["waypoints"][drone.id]
        return self._goal_target(drone, plan)

    def _steer(self, drone, target, state, role):
        spec = self.specs[drone.drone_type]
        dx, dy = target[0] - drone.position[0], target[1] - drone.position[1]
        remaining = hypot(dx, dy)
        if remaining < EPS:
            desired_velocity = (0.0, 0.0)
            forward = (self.direction, 0.0)
        else:
            forward = (dx / remaining, dy / remaining)
            desired_speed = min(spec.max_speed, sqrt(2.0 * spec.max_acceleration * remaining))
            desired_velocity = (forward[0] * desired_speed, forward[1] * desired_speed)
        acceleration = [
            2.25 * (desired_velocity[0] - drone.velocity[0]),
            2.25 * (desired_velocity[1] - drone.velocity[1]),
        ]

        # Deterministic obstacle side-stepping makes the coarse MPC target
        # executable in the continuous engine rather than only in the rollout.
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                center, radius = obstacle.center, obstacle.radius
            else:
                center = ((obstacle.x_min + obstacle.x_max) / 2.0, (obstacle.y_min + obstacle.y_max) / 2.0)
                radius = hypot(obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min) / 2.0
            ox, oy = center[0] - drone.position[0], center[1] - drone.position[1]
            along = ox * forward[0] + oy * forward[1]
            lateral = ox * -forward[1] + oy * forward[0]
            safe = radius + 1.15
            if -0.5 < along < 9.0 and abs(lateral) < safe:
                side = -1.0 if lateral > 0.0 else 1.0
                if abs(lateral) < 0.05:
                    side = 1.0 if drone.id % 2 == 0 else -1.0
                force = spec.max_acceleration * 1.65 * (1.0 - max(0.0, along) / 9.0)
                acceleration[0] += -forward[1] * side * force
                acceleration[1] += forward[0] * side * force
            gap = hypot(ox, oy) - radius
            if EPS < gap < 2.6:
                force = spec.max_acceleration * (2.6 - gap) / 2.6
                length = max(EPS, hypot(ox, oy))
                acceleration[0] -= ox / length * force
                acceleration[1] -= oy / length * force

        # Transports and Tanks avoid enemy contacts; intercepting Scouts are
        # allowed to seek a collision with their assigned target.
        for friend in state.own_drones:
            if friend.id == drone.id or friend.status is not DroneStatus.ACTIVE:
                continue
            dx, dy = drone.position[0] - friend.position[0], drone.position[1] - friend.position[1]
            rvx, rvy = drone.velocity[0] - friend.velocity[0], drone.velocity[1] - friend.velocity[1]
            relative_speed = rvx * rvx + rvy * rvy
            closest_time = _clamp(-(dx * rvx + dy * rvy) / relative_speed, 0.0, 0.8) if relative_speed > EPS else 0.0
            future_dx, future_dy = dx + rvx * closest_time, dy + rvy * closest_time
            separation = hypot(future_dx, future_dy)
            if 0.0 < separation < 2.15:
                force = spec.max_acceleration * 1.9 * (2.15 - separation) / 2.15
                acceleration[0] += future_dx / separation * force
                acceleration[1] += future_dy / separation * force
        if role not in {"intercept", "raid"}:
            for enemy in state.opponent_drones:
                if enemy.status is not DroneStatus.ACTIVE:
                    continue
                dx, dy = drone.position[0] - enemy.position[0], drone.position[1] - enemy.position[1]
                rvx, rvy = drone.velocity[0] - enemy.velocity[0], drone.velocity[1] - enemy.velocity[1]
                relative_speed = rvx * rvx + rvy * rvy
                closest_time = _clamp(-(dx * rvx + dy * rvy) / relative_speed, 0.0, 0.75) if relative_speed > EPS else 0.0
                future_dx, future_dy = dx + rvx * closest_time, dy + rvy * closest_time
                separation = hypot(future_dx, future_dy)
                if 0.0 < separation < 2.8:
                    force = spec.max_acceleration * 1.4 * (2.8 - separation) / 2.8
                    acceleration[0] += future_dx / separation * force
                    acceleration[1] += future_dy / separation * force

        for projectile in state.projectiles:
            if projectile.source_drone_id == drone.id:
                continue
            rx, ry = projectile.position[0] - drone.position[0], projectile.position[1] - drone.position[1]
            rvx, rvy = projectile.velocity[0] - drone.velocity[0], projectile.velocity[1] - drone.velocity[1]
            speed_squared = rvx * rvx + rvy * rvy
            if speed_squared < EPS:
                continue
            closest_time = _clamp(-(rx * rvx + ry * rvy) / speed_squared, 0.0, 1.15)
            miss_x, miss_y = rx + rvx * closest_time, ry + rvy * closest_time
            miss = hypot(miss_x, miss_y)
            if miss < 1.35:
                length = max(EPS, hypot(rvx, rvy))
                side = 1.0 if drone.id % 2 == 0 else -1.0
                acceleration[0] += -rvy / length * side * spec.max_acceleration * 1.8
                acceleration[1] += rvx / length * side * spec.max_acceleration * 1.8
        return _clip(acceleration, spec.max_acceleration)

    def _fire_command(self, drone, state, plan):
        if (
            drone.drone_type is not DroneType.TANK
            or not drone.shots_remaining
            or drone.next_fire_time is None
            or state.time + EPS < drone.next_fire_time
        ):
            return None
        target_id = plan.get("shot_targets", {}).get(drone.id)
        target = next(
            (enemy for enemy in state.opponent_drones if enemy.id == target_id and enemy.status is DroneStatus.ACTIVE),
            None,
        )
        if target is None or not self._clear_shot(drone, target, state):
            return None
        aim = self._lead_point(drone, target)
        direction = (aim[0] - drone.position[0], aim[1] - drone.position[1])
        if not all(isfinite(component) for component in direction) or hypot(*direction) < EPS:
            return None
        return direction

    def step(self, state):
        self.steps += 1
        if self.plan is None or self.steps % REPLAN_STEPS == 1:
            self.plan = self._choose_plan(state)

        actions = {}
        for drone in state.own_drones:
            if drone.status is not DroneStatus.ACTIVE:
                continue
            role = self.plan["roles"].get(drone.id, "advance")
            target = self._runtime_target(drone, state, self.plan)
            command = {"acceleration": self._steer(drone, target, state, role)}
            fire_direction = self._fire_command(drone, state, self.plan)
            if fire_direction is not None:
                command["fire_direction"] = fire_direction
            actions[drone.id] = command
        return actions
