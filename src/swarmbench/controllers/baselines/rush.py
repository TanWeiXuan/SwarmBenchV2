"""Direct goal-seeking baseline with local obstacle steering."""

from swarmbench import BaseSwarmController, DroneStatus

from swarmbench.controllers.baselines.common import finalize_action, goal_target, preferred_fire_target, steer


class RushController(BaseSwarmController):
    def initialize(self, game_info):
        self.goal = game_info.target_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)

    def step(self, state):
        actions = {}
        for drone in state.own_drones:
            if drone.status is not DroneStatus.ACTIVE:
                continue
            spec = self.specs[drone.drone_type]
            acceleration = steer(drone, goal_target(self.goal, drone), spec, self.obstacles)
            actions[drone.id] = finalize_action(
                drone, acceleration, state, spec, self.obstacles, preferred_fire_target(drone, state, self.specs)
            )
        return actions


SwarmController = RushController
