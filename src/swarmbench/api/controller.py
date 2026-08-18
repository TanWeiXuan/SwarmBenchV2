"""The complete public interface implemented by contestants."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .state import GameInfo, GameState
from .types import DroneCommand


class BaseSwarmController(ABC):
    """Base class for every built-in and submitted controller."""

    def initialize(self, game_info: GameInfo) -> None:
        """Called exactly once before the first control step."""

    @abstractmethod
    def step(self, state: GameState) -> dict[int, DroneCommand]:
        """Return movement-only or structured commands keyed by own vehicle ID."""
        raise NotImplementedError

