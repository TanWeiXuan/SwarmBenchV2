"""Nemotron_3_Ultra: Competitive controller with adaptive flanking, predictive targeting, and dynamic role assignment."""

from math import hypot

from swarmbench import BaseSwarmController, DroneStatus, DroneType, Team

from swarmbench.controllers.baselines.common import distance, finalize_action, goal_target, intercept_point, preferred_fire_target, steer


class SwarmController(BaseSwarmController):
    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)
        self.direction = 1.0 if self.team is Team.A else -1.0
        self.roles = {}
        self.last_assignments = {}

    def _assign_roles(self, active, enemies):
        scouts = [d for d in active if d.drone_type is DroneType.SCOUT]
        tanks = [d for d in active if d.drone_type is DroneType.TANK]
        transports = [d for d in active if d.drone_type is DroneType.TRANSPORT]

        enemy_transports = [d for d in enemies if d.drone_type is DroneType.TRANSPORT]
        enemy_tanks = [d for d in enemies if d.drone_type is DroneType.TANK]
        enemy_scouts = [d for d in enemies if d.drone_type is DroneType.SCOUT]

        roles = {}
        for scout in scouts:
            if enemy_transports:
                target = min(enemy_transports, key=lambda e: distance(scout.position, e.position))
                if distance(scout.position, target.position) < 25.0:
                    roles[scout.id] = ("intercept_transport", target.id)
                    continue
            if enemy_scouts:
                target = min(enemy_scouts, key=lambda e: distance(scout.position, e.position))
                if distance(scout.position, target.position) < 20.0:
                    roles[scout.id] = ("intercept_scout", target.id)
                    continue
            if enemy_tanks:
                target = min(enemy_tanks, key=lambda e: distance(scout.position, e.position))
                if distance(scout.position, target.position) < 30.0:
                    roles[scout.id] = ("harass_tank", target.id)
                    continue
            roles[scout.id] = ("advance", None)

        for tank in tanks:
            if enemy_tanks:
                target = min(enemy_tanks, key=lambda e: distance(tank.position, e.position))
                if distance(tank.position, target.position) < 40.0:
                    roles[tank.id] = ("duel_tank", target.id)
                    continue
            if enemy_transports:
                target = min(enemy_transports, key=lambda e: distance(tank.position, e.position))
                roles[tank.id] = ("snipe_transport", target.id)
                continue
            roles[tank.id] = ("anchor", None)

        for transport in transports:
            roles[transport.id] = ("push", None)

        return roles

    def _compute_target(self, drone, role, target_id, active, enemies, state):
        spec = self.specs[drone.drone_type]

        if role == "intercept_transport":
            target_enemy = next((e for e in enemies if e.id == target_id), None)
            if target_enemy:
                flank = 4.0 * self.direction * ((drone.id % 3) - 1) * 0.7
                ahead = -3.0 * self.direction if target_enemy.velocity[0] * self.direction > 0 else 0.0
                return (
                    target_enemy.position[0] + ahead,
                    target_enemy.position[1] + flank,
                )

        elif role == "intercept_scout":
            target_enemy = next((e for e in enemies if e.id == target_id), None)
            if target_enemy:
                return intercept_point(drone, target_enemy)

        elif role == "harass_tank":
            target_enemy = next((e for e in enemies if e.id == target_id), None)
            if target_enemy:
                return (
                    target_enemy.position[0] - 2.5 * self.direction,
                    target_enemy.position[1] + ((drone.id % 2) * 2.0 - 1.0) * 3.5,
                )

        elif role == "duel_tank":
            target_enemy = next((e for e in enemies if e.id == target_id), None)
            if target_enemy:
                firing_x = 38.0 if self.team is Team.A else 62.0
                return (firing_x, target_enemy.position[1])

        elif role == "snipe_transport":
            target_enemy = next((e for e in enemies if e.id == target_id), None)
            if target_enemy:
                firing_x = 38.0 if self.team is Team.A else 62.0
                return (firing_x, target_enemy.position[1])

        elif role == "anchor":
            return goal_target(self.goal, drone)

        elif role == "push":
            tanks = [d for d in active if d.drone_type is DroneType.TANK]
            if tanks:
                anchor = min(tanks, key=lambda t: distance(drone.position, t.position))
                return (anchor.position[0] + 2.5 * self.direction, anchor.position[1])

        return goal_target(self.goal, drone)

    def step(self, state):
        active = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]

        self.roles = self._assign_roles(active, enemies)

        actions = {}
        for drone in active:
            role, target_id = self.roles.get(drone.id, ("advance", None))
            target = self._compute_target(drone, role, target_id, active, enemies, state)
            spec = self.specs[drone.drone_type]

            repulsion = 1.3 if drone.drone_type is DroneType.SCOUT else 1.2
            acceleration = steer(drone, target, spec, self.obstacles, repulsion=repulsion)

            fire_target = preferred_fire_target(drone, state, self.specs)
            if role in ("duel_tank", "snipe_transport") and fire_target:
                actions[drone.id] = finalize_action(
                    drone, acceleration, state, spec, self.obstacles, fire_target
                )
            else:
                actions[drone.id] = finalize_action(
                    drone, acceleration, state, spec, self.obstacles, fire_target
                )
        return actions