"""Predictive Tank fire with Scout screening and Transport scoring."""

from swarmbench import BaseSwarmController, DroneStatus, DroneType, Team

from swarmbench.controllers.baselines.common import distance, finalize_action, goal_target, preferred_fire_target, steer


class MarksmanController(BaseSwarmController):
    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)

    def step(self, state):
        active = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        tanks = [drone for drone in active if drone.drone_type is DroneType.TANK]
        actions = {}
        for drone in active:
            target = goal_target(self.goal, drone)
            if drone.drone_type is DroneType.TANK:
                firing_x = 38.0 if self.team is Team.A else 62.0
                target = (firing_x, min(52.0, max(8.0, drone.position[1])))
            elif drone.drone_type is DroneType.SCOUT and tanks:
                tank = min(tanks, key=lambda item: (distance(drone.position, item.position), item.id))
                direction = 1.0 if self.team is Team.A else -1.0
                lane = (tank.position[0] + 4.0 * direction, tank.position[1] + ((drone.id % 3) - 1) * 1.8)
                target = lane
                threats = [
                    enemy for enemy in state.opponent_drones
                    if enemy.status is DroneStatus.ACTIVE and distance(enemy.position, tank.position) < 12.0
                ]
                if threats:
                    target = min(threats, key=lambda item: (distance(item.position, tank.position), item.id)).position
            spec = self.specs[drone.drone_type]
            acceleration = steer(drone, target, spec, self.obstacles, repulsion=1.35)
            actions[drone.id] = finalize_action(
                drone, acceleration, state, spec, self.obstacles, preferred_fire_target(drone, state, self.specs)
            )
        return actions


SwarmController = MarksmanController
