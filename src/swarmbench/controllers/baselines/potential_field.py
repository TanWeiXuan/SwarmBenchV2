"""Goal attraction combined with obstacle, enemy, and friendly fields."""

from math import hypot

from swarmbench import BaseSwarmController, DroneStatus, DroneType

from swarmbench.controllers.baselines.common import finalize_action, goal_target, preferred_fire_target, steer


class PotentialFieldController(BaseSwarmController):
    def initialize(self, game_info):
        self.goal = game_info.target_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)

    def step(self, state):
        own_active = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        enemies = [drone for drone in state.opponent_drones if drone.status is DroneStatus.ACTIVE]
        actions = {}
        for drone in own_active:
            ax, ay = steer(
                drone,
                goal_target(self.goal, drone),
                self.specs[drone.drone_type],
                self.obstacles,
                repulsion=1.25,
            )
            for friend in own_active:
                if friend.id == drone.id:
                    continue
                dx, dy = drone.position[0] - friend.position[0], drone.position[1] - friend.position[1]
                separation = hypot(dx, dy)
                if 0 < separation < 1.5:
                    ax += dx / separation * (1.5 - separation)
                    ay += dy / separation * (1.5 - separation)
            if drone.drone_type is DroneType.SCOUT and enemies:
                valuable = min(
                    enemies,
                    key=lambda enemy: (
                        hypot(drone.position[0] - enemy.position[0], drone.position[1] - enemy.position[1])
                        - 3.0 * self.specs[enemy.drone_type].point_value,
                        enemy.id,
                    ),
                )
                dx, dy = valuable.position[0] - drone.position[0], valuable.position[1] - drone.position[1]
                separation = hypot(dx, dy)
                if 0 < separation < 18.0:
                    ax += dx / separation * 1.5
                    ay += dy / separation * 1.5
            actions[drone.id] = finalize_action(
                drone,
                (ax, ay),
                state,
                self.specs[drone.drone_type],
                self.obstacles,
                preferred_fire_target(drone, state, self.specs),
            )
        return actions


SwarmController = PotentialFieldController
