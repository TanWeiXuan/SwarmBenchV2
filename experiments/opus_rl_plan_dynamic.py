"""Experiment 3: effective tactical-action instrumentation and dynamic PPO.

The first milestone in this module records the duties that the simulator
actually executes.  Later Experiment 3 policies use the same representation so
nominal neural actions cannot be mistaken for consequential assignments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

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

ROLE_NAMES = ("RUN", "HUNT_TRANSPORT", "HUNT_TANK", "GUARD_TRANSPORT", "KEEP", "BLOCK")
TACTICAL_RUN, HUNT_TRANSPORT, HUNT_TANK, GUARD_TRANSPORT, TACTICAL_KEEP, TACTICAL_BLOCK = range(len(ROLE_NAMES))
ROLE_COUNT = len(ROLE_NAMES)
GLOBAL_FEATURES = 32
ENTITY_FEATURES = 14
PAIR_FEATURES = 32
ENTITY_EMBEDDING = 16
CONTEXT_EMBEDDING = 48


@dataclass(frozen=True)
class CandidateObservation:
    """A target key used only for masking plus ID-free numeric pair features."""

    key: tuple[Any, ...]
    features: tuple[float, ...]


@dataclass(frozen=True)
class ScoutObservation:
    entity: tuple[float, ...]
    previous_role: int
    role_duration: float
    base_role_mask: tuple[bool, ...]
    candidates: tuple[tuple[CandidateObservation, ...], ...]


@dataclass(frozen=True)
class TacticalObservation:
    global_features: tuple[float, ...]
    own_entities: tuple[tuple[float, ...], ...]
    foe_entities: tuple[tuple[float, ...], ...]
    scouts: tuple[ScoutObservation, ...]


@dataclass(frozen=True)
class ScoutAction:
    role: int
    target_index: int = -1


@dataclass
class PolicyDecision:
    actions: tuple[ScoutAction, ...]
    log_probability: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    factor_count: int


class DynamicActorCritic(nn.Module):
    """Permutation-invariant context with shared role and pairwise target heads."""

    def __init__(self, run_bias: float = 1.0) -> None:
        super().__init__()
        self.entity_encoder = nn.Sequential(
            nn.Linear(ENTITY_FEATURES, 24),
            nn.Tanh(),
            nn.Linear(24, ENTITY_EMBEDDING),
            nn.Tanh(),
        )
        # Each team contributes mean and count-normalized sum embeddings.
        self.context_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURES + 4 * ENTITY_EMBEDDING, 64),
            nn.Tanh(),
            nn.Linear(64, CONTEXT_EMBEDDING),
            nn.Tanh(),
        )
        role_input = CONTEXT_EMBEDDING + ENTITY_EMBEDDING + ROLE_COUNT + 1 + ROLE_COUNT
        self.role_head = nn.Sequential(nn.Linear(role_input, 64), nn.Tanh(), nn.Linear(64, ROLE_COUNT))
        target_input = CONTEXT_EMBEDDING + ENTITY_EMBEDDING + PAIR_FEATURES
        self.target_head = nn.Sequential(nn.Linear(target_input, 64), nn.Tanh(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(CONTEXT_EMBEDDING, 32), nn.Tanh(), nn.Linear(32, 1))
        self.reset_parameters(run_bias)

    def reset_parameters(self, run_bias: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                gain = 0.01 if module is self.role_head[-1] or module is self.target_head[-1] else math.sqrt(2.0)
                nn.init.orthogonal_(module.weight, gain)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            # A modest safe initialization, not the collapse-inducing bias from
            # Experiment 2: RUN starts at only e/(e+5) ~= 35% probability.
            self.role_head[-1].bias[TACTICAL_RUN] = run_bias
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _encoded_set(self, entities: tuple[tuple[float, ...], ...]) -> tuple[torch.Tensor, torch.Tensor]:
        if not entities:
            zero = torch.zeros(ENTITY_EMBEDDING, dtype=torch.float32, device=self.device)
            return zero, zero
        values = torch.tensor(entities, dtype=torch.float32, device=self.device)
        encoded = self.entity_encoder(values)
        return encoded.mean(dim=0), encoded.sum(dim=0) / 26.0

    def encode_context(self, observation: TacticalObservation) -> torch.Tensor:
        own_mean, own_sum = self._encoded_set(observation.own_entities)
        foe_mean, foe_sum = self._encoded_set(observation.foe_entities)
        global_features = torch.tensor(observation.global_features, dtype=torch.float32, device=self.device)
        return self.context_encoder(torch.cat((global_features, own_mean, own_sum, foe_mean, foe_sum)))

    def decide(
        self,
        observation: TacticalObservation,
        *,
        stochastic: bool,
        generator: torch.Generator | None = None,
        actions: tuple[ScoutAction, ...] | None = None,
    ) -> PolicyDecision:
        """Sample or re-evaluate one full autoregressive swarm assignment."""
        if actions is not None and len(actions) != len(observation.scouts):
            raise ValueError("recorded action count does not match live scouts")
        context = self.encode_context(observation)
        value = self.critic(context).squeeze(-1)
        used_targets: set[tuple[Any, ...]] = set()
        assigned_counts = torch.zeros(ROLE_COUNT, dtype=torch.float32, device=self.device)
        chosen: list[ScoutAction] = []
        log_probability = torch.zeros((), dtype=torch.float32, device=self.device)
        entropy = torch.zeros((), dtype=torch.float32, device=self.device)
        factor_count = 0
        scout_total = max(1, len(observation.scouts))

        for scout_index, scout in enumerate(observation.scouts):
            scout_entity = self.entity_encoder(
                torch.tensor(scout.entity, dtype=torch.float32, device=self.device)
            )
            previous_role = torch.zeros(ROLE_COUNT, dtype=torch.float32, device=self.device)
            if 0 <= scout.previous_role < ROLE_COUNT:
                previous_role[scout.previous_role] = 1.0
            duration = torch.tensor([scout.role_duration], dtype=torch.float32, device=self.device)
            role_input = torch.cat(
                (context, scout_entity, previous_role, duration, assigned_counts / scout_total)
            )
            role_logits = self.role_head(role_input)
            available_by_role: dict[int, list[int]] = {}
            role_mask = list(scout.base_role_mask)
            for role in (HUNT_TRANSPORT, HUNT_TANK, GUARD_TRANSPORT, TACTICAL_BLOCK):
                available = [
                    index
                    for index, candidate in enumerate(scout.candidates[role])
                    if candidate.key not in used_targets
                ]
                available_by_role[role] = available
                role_mask[role] = role_mask[role] and bool(available)
            masked_role_logits = role_logits.masked_fill(
                ~torch.tensor(role_mask, dtype=torch.bool, device=self.device), -1.0e9
            )
            role_distribution = torch.distributions.Categorical(logits=masked_role_logits)
            if actions is not None:
                role = actions[scout_index].role
                if not role_mask[role]:
                    raise ValueError("recorded role is masked in autoregressive replay")
            elif stochastic:
                role = int(torch.multinomial(role_distribution.probs, 1, generator=generator).item())
            else:
                role = int(masked_role_logits.argmax().item())
            role_tensor = torch.tensor(role, dtype=torch.long, device=self.device)
            log_probability = log_probability + role_distribution.log_prob(role_tensor)
            entropy = entropy + role_distribution.entropy()
            factor_count += 1

            target_index = -1
            if role in available_by_role:
                available = available_by_role[role]
                target_inputs = []
                for index in available:
                    pair = torch.tensor(
                        scout.candidates[role][index].features,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    target_inputs.append(torch.cat((context, scout_entity, pair)))
                target_logits = self.target_head(torch.stack(target_inputs)).squeeze(-1)
                target_distribution = torch.distributions.Categorical(logits=target_logits)
                if actions is not None:
                    target_index = actions[scout_index].target_index
                    if target_index not in available:
                        raise ValueError("recorded target is masked in autoregressive replay")
                    local_index = available.index(target_index)
                elif stochastic:
                    local_index = int(
                        torch.multinomial(target_distribution.probs, 1, generator=generator).item()
                    )
                    target_index = available[local_index]
                else:
                    local_index = int(target_logits.argmax().item())
                    target_index = available[local_index]
                local_tensor = torch.tensor(local_index, dtype=torch.long, device=self.device)
                log_probability = log_probability + target_distribution.log_prob(local_tensor)
                entropy = entropy + target_distribution.entropy()
                factor_count += 1
                used_targets.add(scout.candidates[role][target_index].key)

            chosen.append(ScoutAction(role, target_index))
            assigned_counts[role] += 1.0

        if factor_count:
            entropy = entropy / factor_count
        return PolicyDecision(tuple(chosen), log_probability, value, entropy, factor_count)


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def entity_features(controller, drone, *, friendly: bool) -> tuple[float, ...]:
    """ID-free, team-relative physical features shared across variable sets."""
    kind = drone.drone_type
    spec = controller.specs[kind]
    field = controller.attack if friendly else controller.defend
    cost = controller._cost_to_go(field, drone.position)
    speed = math.hypot(*drone.velocity)
    max_speed = max(spec.max_speed, 1.0e-9)
    max_acceleration = max(spec.max_acceleration, 1.0e-9)
    return (
        float(kind is DroneType.SCOUT),
        float(kind is DroneType.TRANSPORT),
        float(kind is DroneType.TANK),
        _clamp(controller.forward * (drone.position[0] - controller.width / 2.0) / (controller.width / 2.0), -1.0, 1.0),
        _clamp((drone.position[1] - controller.height / 2.0) / (controller.height / 2.0), -1.0, 1.0),
        _clamp(controller.forward * drone.velocity[0] / max_speed, -1.0, 1.0),
        _clamp(drone.velocity[1] / max_speed, -1.0, 1.0),
        _clamp(controller.forward * drone.acceleration[0] / max_acceleration, -1.0, 1.0),
        _clamp(drone.acceleration[1] / max_acceleration, -1.0, 1.0),
        1.0 - _clamp(cost / controller.width, 0.0, 1.0),
        (drone.shots_remaining or 0) / 5.0,
        POINT_VALUE[kind] / 5.0,
        _clamp(speed / max_speed, 0.0, 1.5),
        _clamp(cost / max_speed / controller.duration, 0.0, 1.0),
    )


@dataclass(frozen=True)
class LiveCandidate:
    key: tuple[Any, ...]
    mark: Any
    target: Any
    ward: Any = None


def _pair_features(controller, scout, role: int, candidate: LiveCandidate) -> tuple[float, ...]:
    target = candidate.target
    target_entity = entity_features(controller, target, friendly=False)
    dx = controller.forward * (target.position[0] - scout.position[0]) / controller.width
    dy = (target.position[1] - scout.position[1]) / controller.height
    dvx = controller.forward * (target.velocity[0] - scout.velocity[0]) / max(
        controller.specs[DroneType.SCOUT].max_speed, 1.0e-9
    )
    dvy = (target.velocity[1] - scout.velocity[1]) / max(
        controller.specs[DroneType.SCOUT].max_speed, 1.0e-9
    )
    distance = math.hypot(target.position[0] - scout.position[0], target.position[1] - scout.position[1])
    scout_speed = max(controller.specs[DroneType.SCOUT].max_speed, 1.0e-9)
    ward = candidate.ward
    if ward is None:
        ward_dx = ward_dy = ward_progress = 0.0
    else:
        ward_dx = controller.forward * (ward.position[0] - scout.position[0]) / controller.width
        ward_dy = (ward.position[1] - scout.position[1]) / controller.height
        ward_progress = 1.0 - _clamp(controller._cost_to_go(controller.attack, ward.position) / controller.width, 0.0, 1.0)
    role_one_hot = tuple(float(index == role) for index in range(ROLE_COUNT))
    return role_one_hot + target_entity + (
        _clamp(dx, -1.0, 1.0),
        _clamp(dy, -1.0, 1.0),
        _clamp(dvx, -2.0, 2.0),
        _clamp(dvy, -2.0, 2.0),
        _clamp(distance / controller.width, 0.0, 1.0),
        _clamp(distance / scout_speed / controller.duration, 0.0, 1.0),
        target_entity[-1],
        POINT_VALUE[target.drone_type] / 5.0,
        (target.shots_remaining or 0) / 5.0,
        _clamp(ward_dx, -1.0, 1.0),
        _clamp(ward_dy, -1.0, 1.0),
        ward_progress,
    )


def tactical_observation(controller, state, own, foes, assignment_state):
    """Build model tensors and parallel live objects for deterministic execution."""
    scouts = sorted((drone for drone in own if drone.drone_type is DroneType.SCOUT), key=lambda drone: drone.id)
    wards = [drone for drone in own if drone.drone_type is DroneType.TRANSPORT]
    transports = sorted(
        (drone for drone in foes if drone.drone_type is DroneType.TRANSPORT), key=lambda drone: drone.id
    )
    tanks = sorted(
        (drone for drone in foes if drone.drone_type is DroneType.TANK and drone.shots_remaining), key=lambda drone: drone.id
    )
    guard_targets = controller._pursuers(wards, foes, scouts)
    guard_candidates = []
    for target, _catch_time in guard_targets:
        ward = min(wards, key=lambda item: (math.hypot(item.position[0] - target.position[0], item.position[1] - target.position[1]), item.id))
        guard_candidates.append(LiveCandidate(("DRONE", int(target.id)), target, target, ward))
    block_candidates = [
        LiveCandidate(("BLOCK_LINE", int(ward.id), int(gun.id)), (ward, gun), gun, ward)
        for ward, gun in controller._gun_lines(wards, foes)
    ]
    shared = {
        HUNT_TRANSPORT: [LiveCandidate(("DRONE", int(target.id)), target, target) for target in transports],
        HUNT_TANK: [LiveCandidate(("DRONE", int(target.id)), target, target) for target in tanks],
        GUARD_TRANSPORT: guard_candidates,
        TACTICAL_BLOCK: block_candidates,
    }
    scout_observations = []
    live_candidates = []
    for scout in scouts:
        previous_role, _previous_target, since = assignment_state.get(
            int(scout.id), (TACTICAL_RUN, None, state.time)
        )
        candidates_by_role = []
        live_by_role = []
        for role in range(ROLE_COUNT):
            role_candidates = shared.get(role, [])
            candidates_by_role.append(
                tuple(CandidateObservation(candidate.key, _pair_features(controller, scout, role, candidate)) for candidate in role_candidates)
            )
            live_by_role.append(tuple(role_candidates))
        mask = (
            True,
            bool(transports),
            bool(tanks),
            bool(guard_candidates),
            controller._goal_threatened(foes),
            bool(block_candidates),
        )
        scout_observations.append(
            ScoutObservation(
                entity_features(controller, scout, friendly=True),
                previous_role,
                _clamp((state.time - since) / 10.0, 0.0, 1.0),
                mask,
                tuple(candidates_by_role),
            )
        )
        live_candidates.append(tuple(live_by_role))
    observation = TacticalObservation(
        tuple(controller._policy_features(state, own, foes)),
        tuple(entity_features(controller, drone, friendly=True) for drone in own),
        tuple(entity_features(controller, drone, friendly=False) for drone in foes),
        tuple(scout_observations),
    )
    return observation, tuple(scouts), tuple(live_candidates)


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
