from dataclasses import FrozenInstanceError

import pytest

from swarmbench import BaseSwarmController, DroneSnapshot, DroneType, GameState, Team


class MinimalController(BaseSwarmController):
    def step(self, state: GameState) -> dict[int, tuple[float, float]]:
        return {drone.id: (0.0, 0.0) for drone in state.own_drones}


def test_public_controller_api_is_small_and_usable() -> None:
    drone = DroneSnapshot(0, Team.A, DroneType.SCOUT, (1.0, 2.0))
    state = GameState(0.0, (drone,), (), 0, 0)
    assert MinimalController().step(state) == {0: (0.0, 0.0)}


def test_snapshots_are_immutable() -> None:
    drone = DroneSnapshot(0, Team.A, DroneType.SCOUT, (1.0, 2.0))
    with pytest.raises(FrozenInstanceError):
        drone.position = (2.0, 3.0)  # type: ignore[misc]
