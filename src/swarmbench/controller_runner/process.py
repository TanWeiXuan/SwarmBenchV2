"""Persistent local controller processes with watchdogs and action validation."""

from __future__ import annotations

import json
import os
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import hypot, isfinite
from pathlib import Path
from typing import Any

from swarmbench.api import DroneSpec, DroneType, GameInfo, GameState, Vec2
from swarmbench.engine.dynamics import clip_vector

from .protocol import ProtocolError, game_info_to_dict, game_state_to_dict, request, validate_response
from .sandbox import docker_run_command


class ControllerError(RuntimeError):
    pass


class ControllerTimeout(ControllerError):
    pass


@dataclass(frozen=True, slots=True)
class StepResult:
    commands: dict[int, Vec2]
    accepted_changes: dict[int, Vec2]
    fire_requests: dict[int, Vec2]
    elapsed: float
    missed_soft_deadline: bool = False
    invalid_actions: int = 0
    unknown_drone_ids: int = 0


@dataclass(slots=True)
class ControllerStats:
    step_times: list[float] = field(default_factory=list)
    missed_updates: int = 0
    hard_timeouts: int = 0
    exceptions: int = 0
    invalid_actions: int = 0
    unknown_drone_ids: int = 0

    def summary(self) -> dict[str, float | int]:
        values = sorted(self.step_times)

        def percentile(fraction: float) -> float:
            if not values:
                return 0.0
            return values[min(len(values) - 1, max(0, int((len(values) - 1) * fraction)))]

        return {
            "mean": statistics.fmean(values) if values else 0.0,
            "median": statistics.median(values) if values else 0.0,
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(values, default=0.0),
            "missed_updates": self.missed_updates,
            "hard_timeouts": self.hard_timeouts,
            "exceptions": self.exceptions,
            "invalid_actions": self.invalid_actions,
            "unknown_drone_ids": self.unknown_drone_ids,
        }


class ControllerProcess:
    """One stateful controller process, fresh for each match."""

    def __init__(
        self,
        controller_path: str | Path,
        *,
        soft_deadline: float = 0.5,
        hard_timeout: float = 5.0,
        backend: str = "local",
        docker_image: str = "swarmbench-controller",
    ) -> None:
        self.controller_path = Path(controller_path).resolve()
        self.soft_deadline = soft_deadline
        self.hard_timeout = hard_timeout
        if backend not in {"local", "docker"}:
            raise ValueError("backend must be 'local' or 'docker'")
        self.backend = backend
        self.docker_image = docker_image
        self.stats = ControllerStats()
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._logs: deque[str] = deque(maxlen=200)
        self._sequence = 0
        self._scratch: tempfile.TemporaryDirectory[str] | None = None
        self._commands: dict[int, Vec2] = {}
        self._drone_types: dict[int, DroneType] = {}
        self._drone_specs: dict[DroneType, DroneSpec] = {}

    @property
    def logs(self) -> str:
        return "".join(self._logs)

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _clean_environment(self, seed: int) -> dict[str, str]:
        keep = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "USERNAME", "USER"}
        environment = {key: value for key, value in os.environ.items() if key.upper() in keep}
        package_root = str(Path(__file__).resolve().parents[2])
        environment.update(
            {
                "PYTHONPATH": package_root,
                "PYTHONHASHSEED": str(seed % (2**32)),
                "SWARMBENCH_CONTROLLER_SEED": str(seed),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "SWARMBENCH_TORCH_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        return environment

    def _read_stream(self, stream: Any, output: queue.Queue[str | None] | deque[str], responses: bool) -> None:
        try:
            for line in stream:
                if responses:
                    output.put(line)  # type: ignore[union-attr]
                else:
                    output.append(line[:2000])  # type: ignore[union-attr]
        finally:
            if responses:
                output.put(None)  # type: ignore[union-attr]

    def _start(self, seed: int) -> None:
        if not self.controller_path.is_file():
            raise FileNotFoundError(self.controller_path)
        self._scratch = tempfile.TemporaryDirectory(prefix="swarmbench-controller-") if self.backend == "local" else None
        command = (
            [sys.executable, "-u", "-m", "swarmbench.controller_runner.worker", str(self.controller_path)]
            if self.backend == "local"
            else docker_run_command(self.docker_image, self.controller_path, seed)
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=self._scratch.name if self._scratch else None,
            env=self._clean_environment(seed) if self.backend == "local" else None,
        )
        threading.Thread(target=self._read_stream, args=(self._process.stdout, self._responses, True), daemon=True).start()
        threading.Thread(target=self._read_stream, args=(self._process.stderr, self._logs, False), daemon=True).start()

    def _call(self, command: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
        if not self.alive or self._process is None or self._process.stdin is None:
            raise ControllerError("controller process is not running")
        sequence = self._sequence
        self._sequence += 1
        started = time.perf_counter()
        try:
            self._process.stdin.write(json.dumps(request(sequence, command, payload), allow_nan=False) + "\n")
            self._process.stdin.flush()
            line = self._responses.get(timeout=timeout)
        except BrokenPipeError as error:
            raise ControllerError(f"controller exited before responding: {self.logs[-2000:]}") from error
        except queue.Empty as error:
            self.stats.hard_timeouts += 1
            self.terminate()
            raise ControllerTimeout(f"controller exceeded {timeout:.3f}s hard timeout") from error
        elapsed = time.perf_counter() - started
        if line is None:
            raise ControllerError(f"controller exited unexpectedly: {self.logs[-2000:]}")
        try:
            response = validate_response(json.loads(line), sequence)
        except (json.JSONDecodeError, ProtocolError) as error:
            self.terminate()
            raise ControllerError(str(error)) from error
        if response["status"] == "error":
            self.stats.exceptions += 1
            raise ControllerError(str(response.get("error", "controller exception")))
        return response, elapsed

    def initialize(self, info: GameInfo, timeout: float = 10.0) -> None:
        self._start(info.controller_seed)
        self._commands = {drone.id: (0.0, 0.0) for drone in info.own_initial_drones}
        self._drone_types = {drone.id: drone.drone_type for drone in info.own_initial_drones}
        self._drone_specs = dict(info.drone_specs)
        self._call("initialize", game_info_to_dict(info), timeout)

    def step(self, state: GameState) -> StepResult:
        response, elapsed = self._call("step", game_state_to_dict(state), self.hard_timeout)
        self.stats.step_times.append(elapsed)
        if elapsed > self.soft_deadline:
            self.stats.missed_updates += 1
            return StepResult(dict(self._commands), {}, {}, elapsed, missed_soft_deadline=True)

        raw_actions = response.get("result")
        invalid = 0
        unknown = 0
        accepted: dict[int, Vec2] = {}
        fires: dict[int, Vec2] = {}
        if not isinstance(raw_actions, dict):
            invalid += 1
        else:
            for raw_id, raw_command in raw_actions.items():
                try:
                    drone_id = int(raw_id)
                except (TypeError, ValueError):
                    unknown += 1
                    continue
                if drone_id not in self._commands:
                    unknown += 1
                    continue
                raw_acceleration: Any = raw_command
                raw_fire: Any = None
                if isinstance(raw_command, dict):
                    raw_acceleration = raw_command.get("acceleration")
                    raw_fire = raw_command.get("fire_direction")
                    unknown_keys = set(raw_command) - {"acceleration", "fire_direction"}
                    invalid += len(unknown_keys)
                if raw_acceleration is not None and (
                    not isinstance(raw_acceleration, (list, tuple))
                    or len(raw_acceleration) != 2
                    or isinstance(raw_acceleration[0], bool)
                    or isinstance(raw_acceleration[1], bool)
                ):
                    invalid += 1
                    raw_acceleration = None
                if raw_acceleration is not None:
                    try:
                        command = (float(raw_acceleration[0]), float(raw_acceleration[1]))
                    except (TypeError, ValueError):
                        invalid += 1
                    else:
                        if not all(isfinite(component) for component in command):
                            invalid += 1
                        else:
                            command = clip_vector(command, self._drone_specs[self._drone_types[drone_id]].max_acceleration)
                            self._commands[drone_id] = command
                            accepted[drone_id] = command
                if raw_fire is not None:
                    if (
                        self._drone_types[drone_id] is not DroneType.TANK
                        or not isinstance(raw_fire, (list, tuple))
                        or len(raw_fire) != 2
                        or isinstance(raw_fire[0], bool)
                        or isinstance(raw_fire[1], bool)
                    ):
                        invalid += 1
                    else:
                        try:
                            direction = (float(raw_fire[0]), float(raw_fire[1]))
                        except (TypeError, ValueError):
                            invalid += 1
                        else:
                            magnitude = hypot(*direction)
                            if magnitude == 0.0 or not isfinite(magnitude):
                                invalid += 1
                            else:
                                fires[drone_id] = direction
        self.stats.invalid_actions += invalid
        self.stats.unknown_drone_ids += unknown
        return StepResult(dict(self._commands), accepted, fires, elapsed, invalid_actions=invalid, unknown_drone_ids=unknown)

    def terminate(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        if self._scratch is not None:
            self._scratch.cleanup()
            self._scratch = None

    def close(self) -> None:
        if self.alive:
            try:
                self._call("shutdown", {}, 1.0)
            except ControllerError:
                pass
        self.terminate()

    def __enter__(self) -> "ControllerProcess":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def step_concurrently(
    controller_a: ControllerProcess,
    state_a: GameState,
    controller_b: ControllerProcess,
    state_b: GameState,
) -> tuple[StepResult, StepResult]:
    """Evaluate both controllers concurrently at one simulation instant."""
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="swarmbench-step") as executor:
        future_a = executor.submit(controller_a.step, state_a)
        future_b = executor.submit(controller_b.step, state_b)
        return future_a.result(), future_b.result()
