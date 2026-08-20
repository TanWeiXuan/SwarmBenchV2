"""Experiment 3: effective tactical-action instrumentation and dynamic PPO.

The first milestone in this module records the duties that the simulator
actually executes.  Later Experiment 3 policies use the same representation so
nominal neural actions cannot be mistaken for consequential assignments.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from swarmbench import DroneStatus, DroneType, Team
from swarmbench.api import CONTROLLER_PERIOD, DEFAULT_MATCH_DURATION, PHYSICS_DT
from swarmbench.engine import Simulator, generate_scenario
from swarmbench.match import game_info, game_state

try:
    from experiments.opus_rl_plan_ppo import OPPONENTS, ROOT, SUBJECT_PATH, _commands, _configure_worker, _load_controller
except ModuleNotFoundError:  # Direct ``python experiments/...py`` execution.
    from opus_rl_plan_ppo import OPPONENTS, ROOT, SUBJECT_PATH, _commands, _configure_worker, _load_controller


EXPERIMENT_DIR = ROOT / ".rl_local" / "dynamic"
TACTICAL_INTERVAL = 1.0
FIXED_MODE_4 = (0, 0, 0, 1, 0)
POINT_VALUE = {DroneType.SCOUT: 1, DroneType.TRANSPORT: 5, DroneType.TANK: 1}


def _target_signature(mark: Any) -> tuple[Any, ...] | None:
    if mark is None:
        return None
    if isinstance(mark, tuple):
        ward, gun = mark
        return ("BLOCK_LINE", int(ward.id), int(gun.id))
    return (mark.drone_type.value, int(mark.id))


def _duty_signature(role: int, mark: Any, roles: dict[str, int]) -> tuple[Any, ...]:
    if role == roles["HUNT"] and mark is not None:
        label = f"HUNT_{mark.drone_type.value}"
    else:
        label = next((name for name, value in roles.items() if value == role), f"ROLE_{role}")
    return (label, _target_signature(mark))


def _instrumented_controller(base_type, *, force_mode_4: bool):
    globals_ = base_type._plan.__globals__
    roles = {name: globals_[name] for name in ("RUN", "HUNT", "KEEP", "GUN", "BLOCK")}

    class InstrumentedController(base_type):
        def initialize(self, info):
            super().initialize(info)
            self.effective_records: list[dict[str, Any]] = []
            self._effective_previous: dict[int, tuple[Any, ...]] = {}
            self._role_since: dict[int, float] = {}
            self._next_effective_record = 0.0

        def _allocation(self, state, own, foes):
            if force_mode_4:
                return FIXED_MODE_4
            return super()._allocation(state, own, foes)

        def _plan(self, state, own, foes):
            duties = super()._plan(state, own, foes)
            if state.time + 1.0e-9 >= self._next_effective_record:
                scouts = sorted((drone for drone in own if drone.drone_type is DroneType.SCOUT), key=lambda drone: drone.id)
                current = {
                    int(scout.id): _duty_signature(*duties[scout.id], roles)
                    for scout in scouts
                }
                changed = 0
                target_changed = 0
                durations = {}
                for scout_id, signature in current.items():
                    previous = self._effective_previous.get(scout_id)
                    if previous != signature:
                        changed += 1
                        if previous is not None and previous[1] != signature[1]:
                            target_changed += 1
                        self._role_since[scout_id] = state.time
                    durations[str(scout_id)] = state.time - self._role_since.get(scout_id, state.time)
                counts = Counter(signature[0] for signature in current.values())
                self.effective_records.append(
                    {
                        "time": state.time,
                        "duties": {str(key): value for key, value in current.items()},
                        "role_counts": dict(sorted(counts.items())),
                        "duty_changes": changed,
                        "target_changes": target_changed,
                        "role_durations": durations,
                        "commands_different_from_mode_4": 0,
                    }
                )
                self._effective_previous = current
                self._next_effective_record = state.time + TACTICAL_INTERVAL
            return duties

        def note_shadow_commands(self, state, commands, shadow_commands):
            if not self.effective_records or abs(self.effective_records[-1]["time"] - state.time) > 1.0e-9:
                return
            scout_ids = {
                drone.id
                for drone in state.own_drones
                if drone.status is DroneStatus.ACTIVE and drone.drone_type is DroneType.SCOUT
            }
            self.effective_records[-1]["commands_different_from_mode_4"] = sum(
                commands.get(drone_id) != shadow_commands.get(drone_id)
                for drone_id in scout_ids
            )

    return InstrumentedController


def _event_dict(event) -> dict[str, Any]:
    return {
        "time": event.time,
        "type": event.event_type.value,
        "drone_ids": list(event.drone_ids),
        "team": event.team.value if event.team else None,
        "points": event.points,
    }


def run_instrumented_match(
    *,
    opponent: str,
    seed: int,
    side: str,
    subject: str = "experiment2",
    duration: float = DEFAULT_MATCH_DURATION,
) -> dict[str, Any]:
    """Run one match with an inert fixed-mode-4 shadow on the same observations."""
    _configure_worker()
    started = time.perf_counter()
    scenario = generate_scenario(seed)
    simulator = Simulator(scenario)
    subject_team = Team(side)
    base_type = _load_controller(SUBJECT_PATH, f"dynamic_subject_{os.getpid()}_{seed}")
    subject_type = _instrumented_controller(base_type, force_mode_4=subject == "fixed_mode_4")
    shadow_type = _instrumented_controller(base_type, force_mode_4=True)
    opponent_type = _load_controller(OPPONENTS[opponent], f"dynamic_opponent_{os.getpid()}_{seed}")
    controller = subject_type()
    shadow = shadow_type()
    foe = opponent_type()
    own_info = game_info(scenario, subject_team)
    controller.initialize(own_info)
    shadow.initialize(own_info)
    foe.initialize(game_info(scenario, subject_team.opponent))
    controllers = {subject_team: controller, subject_team.opponent: foe}
    events = []
    score_timeline = [{"time": 0.0, "for": 0, "against": 0}]
    previous_scores = (0, 0)
    control_stride = round(CONTROLLER_PERIOD / PHYSICS_DT)
    for tick in range(round(duration / PHYSICS_DT)):
        if tick % control_stride == 0:
            states = {team: game_state(simulator, team) for team in Team}
            commands = {team: _commands(controllers[team], states[team]) for team in Team}
            shadow_commands = _commands(shadow, states[subject_team])
            controller.note_shadow_commands(states[subject_team], commands[subject_team], shadow_commands)
            for team in Team:
                simulator.set_commands(team, commands[team])
        events.extend(_event_dict(event) for event in simulator.step())
        scores = (simulator.scores[subject_team], simulator.scores[subject_team.opponent])
        if scores != previous_scores:
            score_timeline.append({"time": simulator.time, "for": scores[0], "against": scores[1]})
            previous_scores = scores
        if not any(drone.status is DroneStatus.ACTIVE for drone in simulator.snapshots()):
            break

    snapshots = simulator.snapshots()
    surviving_value = {
        team.value: sum(
            POINT_VALUE[drone.drone_type]
            for drone in snapshots
            if drone.team is team and drone.status is DroneStatus.ACTIVE
        )
        for team in Team
    }
    event_counts = Counter(event["type"] for event in events)
    records = controller.effective_records
    role_counts = Counter()
    for record in records:
        role_counts.update(record["role_counts"])
    score_for, score_against = previous_scores
    return {
        "schema": "opus-rl-plan-effective-actions-v1",
        "seed": seed,
        "side": side,
        "opponent": opponent,
        "subject": subject,
        "score_for": score_for,
        "score_against": score_against,
        "outcome": 1 if score_for > score_against else -1 if score_for < score_against else 0,
        "score_timeline": score_timeline,
        "event_counts": dict(sorted(event_counts.items())),
        "events": events,
        "surviving_value": surviving_value,
        "tank_ammunition_used": sum(
            5 - (drone.shots_remaining or 0)
            for drone in snapshots
            if drone.team is subject_team and drone.drone_type is DroneType.TANK
        ),
        "effective_decisions": len(records),
        "effective_role_counts": dict(sorted(role_counts.items())),
        "duty_changes": sum(record["duty_changes"] for record in records[1:]),
        "target_changes": sum(record["target_changes"] for record in records[1:]),
        "scout_command_differences_from_mode_4": sum(
            record["commands_different_from_mode_4"] for record in records
        ),
        "role_timeline": records,
        "wall_seconds": time.perf_counter() - started,
    }


def instrument_command(args) -> None:
    result = run_instrumented_match(
        opponent=args.opponent,
        seed=args.seed,
        side=args.side,
        subject=args.subject,
        duration=args.duration,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "score_for",
                    "score_against",
                    "effective_decisions",
                    "duty_changes",
                    "target_changes",
                    "scout_command_differences_from_mode_4",
                )
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    instrument = subparsers.add_parser("instrument")
    instrument.add_argument("--opponent", choices=OPPONENTS, default="opus")
    instrument.add_argument("--seed", type=int, default=3_100_003)
    instrument.add_argument("--side", choices=("A", "B"), default="A")
    instrument.add_argument("--subject", choices=("experiment2", "fixed_mode_4"), default="experiment2")
    instrument.add_argument("--duration", type=float, default=DEFAULT_MATCH_DURATION)
    instrument.add_argument("--output")
    instrument.set_defaults(function=instrument_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
