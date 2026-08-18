# Controller API v2

Controllers import the public surface from `swarmbench` and define exactly one `SwarmController(BaseSwarmController)`.

```python
from swarmbench import BaseSwarmController, DroneStatus, DroneType


class SwarmController(BaseSwarmController):
    def initialize(self, game_info):
        self.goal = game_info.target_goal
        self.specs = dict(game_info.drone_specs)

    def step(self, state):
        enemy = next((d for d in state.opponent_drones if d.status is DroneStatus.ACTIVE), None)
        result = {}
        for vehicle in state.own_drones:
            if vehicle.status is not DroneStatus.ACTIVE:
                continue
            command = {"acceleration": (self.specs[vehicle.drone_type].max_acceleration, 0.0)}
            if vehicle.drone_type is DroneType.TANK and enemy is not None:
                command["fire_direction"] = (
                    enemy.position[0] - vehicle.position[0],
                    enemy.position[1] - vehicle.position[1],
                )
            result[vehicle.id] = command
        return result
```

`initialize(game_info)` is called once with team/side, arena size, mirrored goals, immutable obstacles, exact sampled class specifications, weapon specification, both initial teams, scenario identity, controller RNG seed, and API version. Initialization has a 10 s watchdog.

`step(state)` runs at 10 Hz simulation time. State contains the timestamp, complete immutable own/opponent vehicle snapshots, all active projectile snapshots, and both scores. A vehicle snapshot includes ID, team, class, kinematics, status/reason, and—on Tanks—remaining shots and next legal fire time. A projectile snapshot includes ID, team, source Tank ID, position, and constant velocity.

Return a dictionary keyed by own integer IDs. Values may be movement-only `(ax, ay)` or dictionaries with optional `acceleration` and `fire_direction` vectors. Finite acceleration is clipped by the scenario-specific class limit; omitted or malformed acceleration retains the previous command. Fire is a transient request, never retained or queued. It must be a finite nonzero direction from an active Tank; the engine normalizes it and enforces lockout, ammo, and cooldown. Unknown IDs are ignored and counted.

Both sides run concurrently. The 500 ms soft limit discards the complete update—including fire requests—while preserving controller state and earlier movement. The 5 s hard limit forfeits. CI warns when p95 exceeds 400 ms or maximum exceeds 450 ms.

The same object handles every call in one match; a fresh process/object is created for the next match. Python, NumPy, and loaded PyTorch CPU RNGs are deterministically seeded from the controller seed.
