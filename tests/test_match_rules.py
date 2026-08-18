from dataclasses import replace

from swarmbench.api import CircleObstacle, DroneSnapshot, DroneStatus, DroneType, GoalZone, RectangleObstacle, Team
from swarmbench.engine.arena import Scenario
from swarmbench.engine.events import EventType
from swarmbench.engine.match import Simulator


def scenario_with(*drones: DroneSnapshot, obstacles=()) -> Scenario:
    return Scenario(1, 1, 100.0, 60.0, GoalZone(97.0, 100.0, 20.0, 40.0), GoalZone(0.0, 3.0, 20.0, 40.0), obstacles, drones)


def moving(drone_id: int, team: Team, position: tuple[float, float], velocity: tuple[float, float], kind: DroneType = DroneType.SCOUT) -> DroneSnapshot:
    return DroneSnapshot(drone_id, team, kind, position, velocity)


def test_friendly_drones_collide_and_both_are_destroyed() -> None:
    simulator = Simulator(scenario_with(moving(0, Team.A, (50.0, 30.0), (5.0, 0.0)), moving(1, Team.A, (50.5, 30.0), (-5.0, 0.0))))
    events = simulator.step()
    assert events[0].event_type is EventType.VEHICLE_COLLISION
    assert all(drone.status is DroneStatus.ELIMINATED for drone in simulator.snapshots())


def test_one_for_one_rule_prevents_multi_elimination() -> None:
    simulator = Simulator(
        scenario_with(
            moving(0, Team.A, (50.0, 30.0), (0.0, 0.0)),
            moving(20, Team.B, (50.5, 30.0), (0.0, 0.0)),
            moving(21, Team.B, (50.6, 30.0), (0.0, 0.0)),
        )
    )
    events = simulator.step()
    assert len(events) == 1
    assert events[0].drone_ids == (0, 20)
    assert simulator.drones[21].snapshot.status is DroneStatus.ACTIVE


def test_goal_scores_and_removes_drone() -> None:
    drone = moving(0, Team.A, (96.9, 30.0), (5.0, 0.0), DroneType.TRANSPORT)
    simulator = Simulator(scenario_with(drone))
    events = simulator.step()
    assert events[0].event_type is EventType.GOAL
    assert simulator.scores[Team.A] == 5
    assert simulator.drones[0].snapshot.status is DroneStatus.SCORED


def test_swept_obstacle_crash_eliminates_drone() -> None:
    drone = moving(0, Team.A, (49.0, 30.0), (30.0, 0.0))
    simulator = Simulator(scenario_with(drone, obstacles=(CircleObstacle((50.0, 30.0), 0.5),)), dt=0.1)
    events = simulator.step()
    assert events[0].event_type is EventType.OBSTACLE_CRASH
    assert simulator.drones[0].snapshot.status is DroneStatus.ELIMINATED


def test_exact_tie_interception_precedes_goal() -> None:
    a = moving(0, Team.A, (97.0, 30.0), (0.0, 0.0))
    b = moving(20, Team.B, (97.5, 30.0), (0.0, 0.0))
    simulator = Simulator(scenario_with(a, b))
    events = simulator.step()
    assert events[0].event_type is EventType.VEHICLE_COLLISION
    assert simulator.scores[Team.A] == 0
    assert simulator.drones[0].snapshot.status is DroneStatus.ELIMINATED


def test_simultaneous_tie_break_uses_drone_ids() -> None:
    a = moving(0, Team.A, (50.0, 30.0), (0.0, 0.0))
    b_low = moving(20, Team.B, (50.5, 30.0), (0.0, 0.0))
    b_high = replace(b_low, id=21)
    simulator = Simulator(scenario_with(a, b_high, b_low))
    event = simulator.step()[0]
    assert event.drone_ids == (0, 20)


def test_tank_lockout_magazine_and_cooldown_are_per_vehicle() -> None:
    tank = moving(0, Team.A, (10.0, 10.0), (0.0, 0.0), DroneType.TANK)
    simulator = Simulator(scenario_with(tank))
    assert simulator.fire(Team.A, {0: (1.0, 0.0)}) == ()
    simulator.time = 5.0
    fired = simulator.fire(Team.A, {0: (1.0, 0.0)})
    assert fired[0].event_type is EventType.PROJECTILE_FIRED
    assert simulator.drones[0].snapshot.shots_remaining == 4
    assert simulator.drones[0].snapshot.next_fire_time == 9.0
    assert simulator.fire(Team.A, {0: (1.0, 0.0)}) == ()
    simulator.time = 9.0
    assert simulator.fire(Team.A, {0: (1.0, 0.0)})


def test_projectile_friendly_fire_is_continuous_and_non_piercing() -> None:
    tank = moving(0, Team.A, (10.0, 10.0), (0.0, 0.0), DroneType.TANK)
    first = moving(1, Team.A, (11.0, 10.0), (0.0, 0.0))
    second = moving(20, Team.B, (13.0, 10.0), (0.0, 0.0))
    simulator = Simulator(scenario_with(tank, first, second))
    simulator.time = 5.0
    simulator.fire(Team.A, {0: (1.0, 0.0)})
    events = simulator.step()
    assert next(event for event in events if event.event_type is EventType.PROJECTILE_HIT).drone_ids == (1,)
    assert simulator.drones[1].snapshot.status is DroneStatus.ELIMINATED
    assert simulator.drones[20].snapshot.status is DroneStatus.ACTIVE
    assert simulator.projectile_snapshots() == ()


def test_projectile_uses_point_seven_five_metre_contact_radius() -> None:
    tank = moving(0, Team.A, (10.0, 10.0), (0.0, 0.0), DroneType.TANK)
    target = moving(20, Team.B, (10.5, 10.7), (0.0, 0.0))
    simulator = Simulator(scenario_with(tank, target))
    simulator.time = 5.0
    simulator.fire(Team.A, {0: (1.0, 0.0)})

    events = simulator.step()

    assert next(event for event in events if event.event_type is EventType.PROJECTILE_HIT).drone_ids == (20,)
    assert simulator.drones[20].snapshot.status is DroneStatus.ELIMINATED


def test_obstacle_blocks_projectile_before_vehicle() -> None:
    tank = moving(0, Team.A, (10.0, 10.0), (0.0, 0.0), DroneType.TANK)
    target = moving(20, Team.B, (12.0, 10.0), (0.0, 0.0))
    wall = RectangleObstacle(10.4, 10.6, 9.0, 11.0)
    simulator = Simulator(scenario_with(tank, target, obstacles=(wall,)))
    simulator.time = 5.0
    simulator.fire(Team.A, {0: (1.0, 0.0)})
    events = simulator.step()
    assert any(event.event_type is EventType.PROJECTILE_BLOCKED for event in events)
    assert simulator.drones[20].snapshot.status is DroneStatus.ACTIVE
