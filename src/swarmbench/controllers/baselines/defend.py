"""Transport attackers supported by Scout defenders and Tanks."""

from swarmbench import BaseSwarmController, DroneStatus, DroneType, Team

from swarmbench.controllers.baselines.common import distance, finalize_action, goal_target, preferred_fire_target, steer


class DefendController(BaseSwarmController):
    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.own_goal = game_info.own_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)

    def step(self, state):
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        actions = {}
        for drone in state.own_drones:
            if drone.status is not DroneStatus.ACTIVE:
                continue
            target = goal_target(self.goal, drone)
            if drone.drone_type is DroneType.SCOUT and enemies:
                threats = sorted(
                    enemies,
                    key=lambda enemy: (
                        distance(enemy.position, self.own_goal.center) + 0.35 * distance(drone.position, enemy.position),
                        enemy.id,
                    ),
                )
                threat = threats[0]
                in_defensive_half = threat.position[0] < 55.0 if self.team is Team.A else threat.position[0] > 45.0
                if in_defensive_half:
                    target = threat.position
            spec = self.specs[drone.drone_type]
            acceleration = steer(drone, target, spec, self.obstacles)
            actions[drone.id] = finalize_action(
                drone, acceleration, state, spec, self.obstacles, preferred_fire_target(drone, state, self.specs)
            )
        return actions


SwarmController = DefendController
