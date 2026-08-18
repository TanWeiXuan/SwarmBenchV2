"""Value-aware unique greedy interception assignments."""

from swarmbench import BaseSwarmController, DroneStatus, DroneType

from swarmbench.controllers.baselines.common import distance, finalize_action, goal_target, preferred_fire_target, steer


class GreedyValueController(BaseSwarmController):
    def initialize(self, game_info):
        self.goal = game_info.target_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)

    def step(self, state):
        active_enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        defenders = [
            drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE and drone.drone_type is DroneType.SCOUT
        ]
        candidates = []
        for defender in defenders:
            for enemy in active_enemies:
                value = self.specs[enemy.drone_type].point_value
                exchange_bonus = 5.0 if enemy.drone_type is DroneType.TRANSPORT else 0.0
                cost = distance(defender.position, enemy.position) - 4.0 * value - exchange_bonus
                candidates.append((cost, defender.id, enemy.id))
        assigned_defenders: dict[int, int] = {}
        assigned_enemies: set[int] = set()
        for _, defender_id, enemy_id in sorted(candidates):
            if defender_id not in assigned_defenders and enemy_id not in assigned_enemies:
                assigned_defenders[defender_id] = enemy_id
                assigned_enemies.add(enemy_id)
        enemy_by_id = {enemy.id: enemy for enemy in active_enemies}

        actions = {}
        for drone in state.own_drones:
            if drone.status is not DroneStatus.ACTIVE:
                continue
            target = goal_target(self.goal, drone)
            if drone.id in assigned_defenders:
                target = enemy_by_id[assigned_defenders[drone.id]].position
            spec = self.specs[drone.drone_type]
            acceleration = steer(drone, target, spec, self.obstacles)
            actions[drone.id] = finalize_action(
                drone, acceleration, state, spec, self.obstacles, preferred_fire_target(drone, state, self.specs)
            )
        return actions


SwarmController = GreedyValueController
