from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from swarmbench.api import (
    DRONE_SPECS,
    CircleObstacle,
    DroneSnapshot,
    DroneType,
    GameInfo,
    GameState,
    GoalZone,
    Team,
)
from swarmbench.controller_runner import ControllerError, ControllerProcess, ControllerTimeout, step_concurrently
from swarmbench.controller_runner.protocol import PROTOCOL_VERSION, ProtocolError, validate_response


def write_controller(path: Path, body: str) -> Path:
    path.write_text("from swarmbench import BaseSwarmController\n" + body, encoding="utf-8")
    return path


def game_info(team: Team = Team.A) -> GameInfo:
    own = (DroneSnapshot(0 if team is Team.A else 20, team, DroneType.SCOUT, (5.0, 5.0)),)
    opponent = (DroneSnapshot(20 if team is Team.A else 0, team.opponent, DroneType.TRANSPORT, (95.0, 5.0)),)
    return GameInfo(
        team,
        100.0,
        60.0,
        GoalZone(97.0, 100.0, 20.0, 34.0) if team is Team.A else GoalZone(0.0, 3.0, 20.0, 34.0),
        GoalZone(0.0, 3.0, 20.0, 34.0) if team is Team.A else GoalZone(97.0, 100.0, 20.0, 34.0),
        (CircleObstacle((50.0, 30.0), 2.0),),
        tuple(DRONE_SPECS.items()),
        own,
        opponent,
        42,
        1,
        123,
        1,
    )


def game_state(team: Team = Team.A) -> GameState:
    info = game_info(team)
    return GameState(0.0, info.own_initial_drones, info.opponent_initial_drones, 0, 0)


def test_initialize_stateful_calls_and_print_isolation(tmp_path: Path) -> None:
    path = write_controller(
        tmp_path / "stateful.py",
        """
class SwarmController(BaseSwarmController):
    def initialize(self, game_info):
        print('initializing normally')
        self.count = 0
    def step(self, state):
        print('ordinary controller output')
        self.count += 1
        return {state.own_drones[0].id: (self.count, 0)}
""",
    )
    with ControllerProcess(path) as controller:
        controller.initialize(game_info())
        assert controller.step(game_state()).commands[0] == (1.0, 0.0)
        assert controller.step(game_state()).commands[0] == (2.0, 0.0)
        assert "ordinary controller output" in controller.logs


def test_partial_malformed_unknown_and_clipped_actions(tmp_path: Path) -> None:
    path = write_controller(
        tmp_path / "validation.py",
        """
class SwarmController(BaseSwarmController):
    def step(self, state):
        return {0: (30, 40), 20: (1, 2), 99: (0, 0), 'bad': (1, 2)}
""",
    )
    info = game_info()
    with ControllerProcess(path) as controller:
        controller.initialize(info)
        result = controller.step(game_state())
        assert result.commands[0] == pytest.approx((2.4, 3.2))
        assert result.unknown_drone_ids == 3


def test_structured_tank_command_keeps_movement_and_transient_fire(tmp_path: Path) -> None:
    path = write_controller(
        tmp_path / "tank.py",
        """
class SwarmController(BaseSwarmController):
    def step(self, state):
        return {0: {'acceleration': (3, 4), 'fire_direction': (10, 0)}}
""",
    )
    info = game_info()
    tank = replace(info.own_initial_drones[0], drone_type=DroneType.TANK, shots_remaining=5, next_fire_time=5.0)
    info = replace(info, own_initial_drones=(tank,))
    state = replace(game_state(), own_drones=(tank,))
    with ControllerProcess(path) as controller:
        controller.initialize(info)
        result = controller.step(state)
    assert result.commands[0] == pytest.approx((0.72, 0.96))
    assert result.fire_requests == {0: (10.0, 0.0)}


def test_nan_wrong_dimensions_and_nonnumeric_retain_previous(tmp_path: Path) -> None:
    path = write_controller(
        tmp_path / "invalid.py",
        """
class SwarmController(BaseSwarmController):
    def __init__(self): self.count = 0
    def step(self, state):
        self.count += 1
        if self.count == 1: return {0: (1, 0)}
        return {0: (float('nan'), 0), '0.0': [1], '00': ['x', 1]}
""",
    )
    with ControllerProcess(path) as controller:
        controller.initialize(game_info())
        controller.step(game_state())
        result = controller.step(game_state())
        assert result.commands[0] == (1.0, 0.0)
        assert result.invalid_actions == 2
        assert result.unknown_drone_ids == 1


def test_controller_exception_forfeits(tmp_path: Path) -> None:
    path = write_controller(
        tmp_path / "exception.py",
        """
class SwarmController(BaseSwarmController):
    def step(self, state): raise RuntimeError('boom')
""",
    )
    with ControllerProcess(path) as controller:
        controller.initialize(game_info())
        with pytest.raises(ControllerError, match="boom"):
            controller.step(game_state())
        assert controller.stats.exceptions == 1


def test_soft_deadline_discards_late_actions_but_preserves_state(tmp_path: Path) -> None:
    path = write_controller(
        tmp_path / "late.py",
        """
import time
class SwarmController(BaseSwarmController):
    def __init__(self): self.count = 0
    def step(self, state):
        self.count += 1
        if self.count == 1: time.sleep(0.05)
        return {0: (self.count, 0)}
""",
    )
    with ControllerProcess(path, soft_deadline=0.01, hard_timeout=0.5) as controller:
        controller.initialize(game_info())
        late = controller.step(game_state())
        assert late.missed_soft_deadline
        assert late.commands[0] == (0.0, 0.0)
        on_time = controller.step(game_state())
        assert on_time.commands[0] == (2.0, 0.0)


def test_hard_timeout_terminates_process(tmp_path: Path) -> None:
    path = write_controller(
        tmp_path / "timeout.py",
        """
import time
class SwarmController(BaseSwarmController):
    def step(self, state): time.sleep(2); return {}
""",
    )
    with ControllerProcess(path, hard_timeout=0.05) as controller:
        controller.initialize(game_info())
        with pytest.raises(ControllerTimeout):
            controller.step(game_state())
        assert not controller.alive
        assert controller.stats.hard_timeouts == 1


def test_wrong_sequence_is_rejected() -> None:
    with pytest.raises(ProtocolError):
        validate_response({"protocol_version": PROTOCOL_VERSION, "sequence": 8, "status": "ok"}, 7)


def test_both_controllers_are_evaluated_concurrently(tmp_path: Path) -> None:
    body = """
import time
class SwarmController(BaseSwarmController):
    def step(self, state): time.sleep(0.15); return {}
"""
    left = ControllerProcess(write_controller(tmp_path / "left.py", body), hard_timeout=1.0)
    right = ControllerProcess(write_controller(tmp_path / "right.py", body), hard_timeout=1.0)
    try:
        left.initialize(game_info(Team.A))
        right.initialize(game_info(Team.B))
        started = time.perf_counter()
        step_concurrently(left, game_state(Team.A), right, game_state(Team.B))
        elapsed = time.perf_counter() - started
        assert elapsed < 0.27
    finally:
        left.close()
        right.close()
