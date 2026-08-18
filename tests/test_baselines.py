from __future__ import annotations

from math import isfinite

import pytest

from swarmbench.api import CONTROLLER_PERIOD, DRONE_SPECS, PHYSICS_DT, DroneStatus, GameInfo, GameState, Team
from swarmbench.controllers.baselines import (
    AssignmentController,
    ConvoyController,
    DefendController,
    GreedyValueController,
    MarksmanController,
    PotentialFieldController,
    RushController,
    baseline_path,
)
from swarmbench.controller_runner import ControllerProcess
from swarmbench.engine import Scenario, Simulator, generate_scenario
from swarmbench.version import CONTROLLER_API_VERSION


CONTROLLERS = [RushController, DefendController, GreedyValueController, AssignmentController, PotentialFieldController, MarksmanController, ConvoyController]


def info_for(scenario: Scenario, team: Team) -> GameInfo:
    return GameInfo(
        team,
        scenario.width,
        scenario.height,
        scenario.target_goal(team),
        scenario.own_goal(team),
        scenario.obstacles,
        scenario.drone_specs,
        scenario.team_drones(team),
        scenario.team_drones(team.opponent),
        scenario.seed,
        scenario.generator_version,
        scenario.seed * 2 + (0 if team is Team.A else 1),
        CONTROLLER_API_VERSION,
    )


def state_for(simulator: Simulator, team: Team) -> GameState:
    return GameState(
        simulator.time,
        simulator.snapshots(team),
        simulator.snapshots(team.opponent),
        simulator.scores[team],
        simulator.scores[team.opponent],
        simulator.projectile_snapshots(),
    )


@pytest.mark.parametrize("controller_type", CONTROLLERS)
def test_baseline_actions_are_complete_finite_and_deterministic(controller_type) -> None:
    scenario = generate_scenario(17)
    first = controller_type()
    second = controller_type()
    info = info_for(scenario, Team.A)
    state = GameState(0.0, info.own_initial_drones, info.opponent_initial_drones, 0, 0)
    first.initialize(info)
    second.initialize(info)
    actions = first.step(state)
    assert actions == second.step(state)
    assert set(actions) == {drone.id for drone in state.own_drones}
    for action in actions.values():
        acceleration = action.get("acceleration") if isinstance(action, dict) else action
        assert len(acceleration) == 2 and all(isfinite(value) for value in acceleration)


@pytest.mark.parametrize("controller_type", CONTROLLERS)
def test_each_baseline_scores_in_empty_arena(controller_type) -> None:
    generated = generate_scenario(5)
    scenario = Scenario(
        generated.seed,
        generated.generator_version,
        generated.width,
        generated.height,
        generated.goal_for_a,
        generated.goal_for_b,
        (),
        generated.team_drones(Team.A),
        generated.drone_specs,
    )
    simulator = Simulator(scenario)
    controller = controller_type()
    controller.initialize(info_for(scenario, Team.A))
    for tick in range(round(90.0 / PHYSICS_DT)):
        if tick % round(CONTROLLER_PERIOD / PHYSICS_DT) == 0:
            simulator.set_commands(Team.A, controller.step(state_for(simulator, Team.A)))
        simulator.step()
    assert simulator.scores[Team.A] > 0
    assert all(drone.status is not DroneStatus.ACTIVE for drone in simulator.snapshots(Team.A)) or simulator.scores[Team.A] >= 10


def test_deterministic_head_to_head_smoke() -> None:
    def play() -> tuple[int, int, tuple]:
        scenario = generate_scenario(31)
        simulator = Simulator(scenario)
        left, right = RushController(), PotentialFieldController()
        left.initialize(info_for(scenario, Team.A))
        right.initialize(info_for(scenario, Team.B))
        for tick in range(round(12.0 / PHYSICS_DT)):
            if tick % 2 == 0:
                simulator.set_commands(Team.A, left.step(state_for(simulator, Team.A)))
                simulator.set_commands(Team.B, right.step(state_for(simulator, Team.B)))
            simulator.step()
        return simulator.scores[Team.A], simulator.scores[Team.B], tuple(simulator.events)

    assert play() == play()


def test_baseline_uses_the_same_subprocess_api_as_submissions() -> None:
    scenario = generate_scenario(9)
    info = info_for(scenario, Team.A)
    state = GameState(0.0, info.own_initial_drones, info.opponent_initial_drones, 0, 0)
    with ControllerProcess(baseline_path("rush")) as controller:
        controller.initialize(info)
        result = controller.step(state)
    assert len(result.accepted_changes) == len(info.own_initial_drones)
