"""Transport convoys with Scout escorts and trailing Tank support."""

from swarmbench import BaseSwarmController, DroneStatus, DroneType, Team

from swarmbench.controllers.baselines.common import distance, finalize_action, goal_target, preferred_fire_target, steer


class ConvoyController(BaseSwarmController):
    def initialize(self, game_info):
        self.team = game_info.team
        self.goal = game_info.target_goal
        self.obstacles = game_info.obstacles
        self.specs = dict(game_info.drone_specs)

    def step(self, state):
        active = [drone for drone in state.own_drones if drone.status is DroneStatus.ACTIVE]
        transports = [drone for drone in active if drone.drone_type is DroneType.TRANSPORT]
        direction = 1.0 if self.team is Team.A else -1.0
        actions = {}
        for drone in active:
            target = goal_target(self.goal, drone)
            if transports and drone.drone_type is DroneType.SCOUT:
                transport = min(transports, key=lambda item: (distance(drone.position, item.position), item.id))
                slot = ((drone.id % 5) - 2) * 1.1
                target = (transport.position[0] + 2.5 * direction, transport.position[1] + slot)
                nearby = [
                    enemy for enemy in state.opponent_drones
                    if enemy.status is DroneStatus.ACTIVE and distance(enemy.position, transport.position) < 8.0
                ]
                if nearby:
                    target = min(nearby, key=lambda item: (distance(item.position, transport.position), item.id)).position
            elif transports and drone.drone_type is DroneType.TANK:
                transport = min(transports, key=lambda item: (distance(drone.position, item.position), item.id))
                target = (transport.position[0] - 4.0 * direction, transport.position[1])
            spec = self.specs[drone.drone_type]
            acceleration = steer(drone, target, spec, self.obstacles, repulsion=1.4)
            actions[drone.id] = finalize_action(
                drone, acceleration, state, spec, self.obstacles, preferred_fire_target(drone, state, self.specs)
            )
        return actions


SwarmController = ConvoyController
