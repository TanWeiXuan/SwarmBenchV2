"""Periodic linear-sum assignment of Scout defenders to threats."""

from scipy.optimize import linear_sum_assignment

from swarmbench import BaseSwarmController, DroneStatus, DroneType, Team

from swarmbench.controllers.baselines.common import distance, finalize_action, goal_target, preferred_fire_target, steer


class AssignmentController(BaseSwarmController):
    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)
        self.assignments = {}
        self.steps = 0

    def _assign(self, defenders, enemies):
        if not defenders or not enemies:
            return {}
        costs = []
        for defender in defenders:
            row = []
            for enemy in enemies:
                threat_progress = 100.0 - enemy.position[0] if self.team is Team.B else enemy.position[0]
                row.append(
                    distance(defender.position, enemy.position)
                    - 3.5 * self.specs[enemy.drone_type].point_value
                    - 0.08 * threat_progress
                )
            costs.append(row)
        rows, columns = linear_sum_assignment(costs)
        return {defenders[row].id: enemies[column].id for row, column in zip(rows, columns, strict=True)}

    def step(self, state):
        self.steps += 1
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        enemy_by_id = {drone.id: drone for drone in enemies}
        defenders = [
            drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE and drone.drone_type is DroneType.SCOUT
        ]
        if self.steps == 1 or self.steps % 5 == 0:
            self.assignments = self._assign(defenders, enemies)
        else:
            self.assignments = {
                defender_id: enemy_id
                for defender_id, enemy_id in self.assignments.items()
                if enemy_id in enemy_by_id
            }

        actions = {}
        for drone in state.own_drones:
            if drone.status is not DroneStatus.ACTIVE:
                continue
            target = goal_target(self.goal, drone)
            if drone.id in self.assignments:
                target = enemy_by_id[self.assignments[drone.id]].position
            spec = self.specs[drone.drone_type]
            acceleration = steer(drone, target, spec, self.obstacles)
            actions[drone.id] = finalize_action(
                drone, acceleration, state, spec, self.obstacles, preferred_fire_target(drone, state, self.specs)
            )
        return actions


SwarmController = AssignmentController
