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
import random
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
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

COMMUNITY_OPPONENTS = {
    f"{path.parent.name}/{path.stem}": path
    for path in (ROOT / "submissions").glob("*/*.py")
    if path.resolve() != SUBJECT_PATH.resolve()
}
BASELINE_OPPONENTS = {
    name: OPPONENTS[name]
    for name in ("potential", "greedy", "assignment", "rush", "defend", "convoy", "marksman")
}
LEAGUE_OPPONENTS = {**COMMUNITY_OPPONENTS, **BASELINE_OPPONENTS}
HARD_OPPONENTS = (
    "renj1ete0/opus_5_v1",
    "TanWeiXuan/Luna_xHigh_opus_breaker",
    "renj1ete0/GPT-5.3-Codex",
    "renj1ete0/gemini_3_1_pro_v1",
    "renj1ete0/sonnet_5_v3",
    "TanWeiXuan/Luna_xHigh_sipp_marksman_v1",
)


def _opponent_path(name: str) -> Path:
    if name in LEAGUE_OPPONENTS:
        return LEAGUE_OPPONENTS[name]
    if name in OPPONENTS:
        return OPPONENTS[name]
    raise KeyError(f"unknown opponent: {name}")

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

    def encode_context_batch(
        self, observations: list[TacticalObservation]
    ) -> torch.Tensor:
        """Pack variable live sets without padding or hard-coded drone counts."""
        batch_size = len(observations)

        def packed_team(attribute: str) -> tuple[torch.Tensor, torch.Tensor]:
            entities = []
            owners = []
            for observation_index, observation in enumerate(observations):
                values = getattr(observation, attribute)
                entities.extend(values)
                owners.extend([observation_index] * len(values))
            if not entities:
                zero = torch.zeros(
                    (batch_size, ENTITY_EMBEDDING),
                    dtype=torch.float32,
                    device=self.device,
                )
                return zero, zero
            encoded = self.entity_encoder(
                torch.tensor(entities, dtype=torch.float32, device=self.device)
            )
            owner_tensor = torch.tensor(owners, dtype=torch.long, device=self.device)
            sums = torch.zeros(
                (batch_size, ENTITY_EMBEDDING),
                dtype=torch.float32,
                device=self.device,
            ).index_add(0, owner_tensor, encoded)
            counts = torch.zeros(
                batch_size, dtype=torch.float32, device=self.device
            ).index_add(
                0,
                owner_tensor,
                torch.ones(len(owners), dtype=torch.float32, device=self.device),
            )
            means = sums / counts.clamp_min(1.0).unsqueeze(1)
            return means, sums / 26.0

        own_mean, own_sum = packed_team("own_entities")
        foe_mean, foe_sum = packed_team("foe_entities")
        global_features = torch.tensor(
            [observation.global_features for observation in observations],
            dtype=torch.float32,
            device=self.device,
        )
        return self.context_encoder(
            torch.cat(
                (global_features, own_mean, own_sum, foe_mean, foe_sum), dim=1
            )
        )

    def evaluate_batch(
        self,
        observations: list[TacticalObservation],
        actions: list[tuple[ScoutAction, ...]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized exact replay of recorded autoregressive assignments."""
        if len(observations) != len(actions):
            raise ValueError("observation/action batch mismatch")
        contexts = self.encode_context_batch(observations)
        values = self.critic(contexts).squeeze(-1)
        scout_features = []
        scout_observation_indices = []
        scout_inputs = []
        role_masks = []
        chosen_roles = []
        target_factors = []

        for observation_index, (observation, assignment) in enumerate(
            zip(observations, actions)
        ):
            if len(observation.scouts) != len(assignment):
                raise ValueError("recorded action count does not match live scouts")
            used_targets: set[tuple[Any, ...]] = set()
            assigned_counts = [0.0] * ROLE_COUNT
            scout_total = max(1, len(observation.scouts))
            for scout, action in zip(observation.scouts, assignment):
                flat_scout_index = len(scout_features)
                scout_features.append(scout.entity)
                scout_observation_indices.append(observation_index)
                previous = [0.0] * ROLE_COUNT
                if 0 <= scout.previous_role < ROLE_COUNT:
                    previous[scout.previous_role] = 1.0
                scout_inputs.append(
                    previous
                    + [scout.role_duration]
                    + [count / scout_total for count in assigned_counts]
                )
                mask = list(scout.base_role_mask)
                available_by_role = {}
                for role in (
                    HUNT_TRANSPORT,
                    HUNT_TANK,
                    GUARD_TRANSPORT,
                    TACTICAL_BLOCK,
                ):
                    available = [
                        index
                        for index, candidate in enumerate(scout.candidates[role])
                        if candidate.key not in used_targets
                    ]
                    available_by_role[role] = available
                    mask[role] = mask[role] and bool(available)
                if not mask[action.role]:
                    raise ValueError("recorded role is masked in batch replay")
                role_masks.append(mask)
                chosen_roles.append(action.role)
                if action.role in available_by_role:
                    available = available_by_role[action.role]
                    if action.target_index not in available:
                        raise ValueError("recorded target is masked in batch replay")
                    target_factors.append(
                        (
                            observation_index,
                            flat_scout_index,
                            scout.candidates[action.role],
                            available,
                            available.index(action.target_index),
                        )
                    )
                    used_targets.add(
                        scout.candidates[action.role][action.target_index].key
                    )
                assigned_counts[action.role] += 1.0

        batch_size = len(observations)
        log_probability = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        entropy_sum = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        factor_counts = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        if scout_features:
            scout_encoded = self.entity_encoder(
                torch.tensor(
                    scout_features, dtype=torch.float32, device=self.device
                )
            )
            observation_indices = torch.tensor(
                scout_observation_indices, dtype=torch.long, device=self.device
            )
            role_input = torch.cat(
                (
                    contexts[observation_indices],
                    scout_encoded,
                    torch.tensor(
                        scout_inputs, dtype=torch.float32, device=self.device
                    ),
                ),
                dim=1,
            )
            role_logits = self.role_head(role_input).masked_fill(
                ~torch.tensor(role_masks, dtype=torch.bool, device=self.device),
                -1.0e9,
            )
            role_distribution = torch.distributions.Categorical(logits=role_logits)
            selected = torch.tensor(
                chosen_roles, dtype=torch.long, device=self.device
            )
            log_probability = log_probability.index_add(
                0, observation_indices, role_distribution.log_prob(selected)
            )
            entropy_sum = entropy_sum.index_add(
                0, observation_indices, role_distribution.entropy()
            )
            factor_counts = factor_counts.index_add(
                0,
                observation_indices,
                torch.ones(
                    len(scout_features), dtype=torch.float32, device=self.device
                ),
            )

            packed_targets = []
            target_slices = []
            for (
                observation_index,
                flat_scout_index,
                candidates,
                available,
                chosen_local,
            ) in target_factors:
                start = len(packed_targets)
                for candidate_index in available:
                    pair = torch.tensor(
                        candidates[candidate_index].features,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    packed_targets.append(
                        torch.cat(
                            (
                                contexts[observation_index],
                                scout_encoded[flat_scout_index],
                                pair,
                            )
                        )
                    )
                target_slices.append(
                    (observation_index, start, len(packed_targets), chosen_local)
                )
            if packed_targets:
                packed_logits = self.target_head(
                    torch.stack(packed_targets)
                ).squeeze(-1)
                target_log_probabilities = []
                target_entropies = []
                target_observation_indices = []
                for observation_index, start, end, chosen_local in target_slices:
                    distribution = torch.distributions.Categorical(
                        logits=packed_logits[start:end]
                    )
                    target_log_probabilities.append(
                        distribution.log_prob(
                            torch.tensor(
                                chosen_local, dtype=torch.long, device=self.device
                            )
                        )
                    )
                    target_entropies.append(distribution.entropy())
                    target_observation_indices.append(observation_index)
                target_indices = torch.tensor(
                    target_observation_indices,
                    dtype=torch.long,
                    device=self.device,
                )
                log_probability = log_probability.index_add(
                    0, target_indices, torch.stack(target_log_probabilities)
                )
                entropy_sum = entropy_sum.index_add(
                    0, target_indices, torch.stack(target_entropies)
                )
                factor_counts = factor_counts.index_add(
                    0,
                    target_indices,
                    torch.ones(
                        len(target_slices),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
        entropy = entropy_sum / factor_counts.clamp_min(1.0)
        return log_probability, values, entropy

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


def update_assignments(state, scouts, live_candidates, actions, previous):
    """Convert sampled indices into stable execution tokens and persistence state."""
    updated = {}
    for scout, candidates_by_role, action in zip(scouts, live_candidates, actions):
        target_key = None
        if action.target_index >= 0:
            target_key = candidates_by_role[action.role][action.target_index].key
        old_role, old_target, old_since = previous.get(
            int(scout.id), (TACTICAL_RUN, None, state.time)
        )
        since = old_since if (old_role, old_target) == (action.role, target_key) else state.time
        updated[int(scout.id)] = (action.role, target_key, since)
    return updated


def resolve_tactical_duties(controller, state, own, foes, assignment_state):
    """Resolve stable ID tokens to current snapshots on every control update."""
    run, hunt, keep, gun, block = controller.BASE_ROLES
    own_by_id = {int(drone.id): drone for drone in own}
    foes_by_id = {int(drone.id): drone for drone in foes}
    duties = {}
    for drone in own:
        if drone.drone_type is DroneType.TANK:
            duties[drone.id] = (gun if drone.shots_remaining else run, None)
        elif drone.drone_type is DroneType.TRANSPORT:
            duties[drone.id] = (run, None)
        else:
            role, target_key, _since = assignment_state.get(
                int(drone.id), (TACTICAL_RUN, None, state.time)
            )
            if role == TACTICAL_RUN:
                duties[drone.id] = (run, None)
            elif role == TACTICAL_KEEP:
                duties[drone.id] = (keep, None)
            elif role == TACTICAL_BLOCK and target_key and target_key[0] == "BLOCK_LINE":
                ward = own_by_id.get(int(target_key[1]))
                tank = foes_by_id.get(int(target_key[2]))
                duties[drone.id] = (block, (ward, tank)) if ward is not None and tank is not None else (run, None)
            elif target_key and target_key[0] == "DRONE":
                target = foes_by_id.get(int(target_key[1]))
                duties[drone.id] = (hunt, target) if target is not None else (run, None)
            else:
                duties[drone.id] = (run, None)
    return duties


def _dynamic_controller(
    base_type,
    model: DynamicActorCritic,
    generator: torch.Generator,
    *,
    stochastic: bool,
    decision_interval: float = TACTICAL_INTERVAL,
    ablation: str = "full",
):
    """Inject learned scout duties while retaining every deterministic skill."""
    if ablation not in {"full", "freeze_start"}:
        raise ValueError(f"unsupported dynamic ablation: {ablation}")

    class DynamicController(base_type):
        def initialize(self, info):
            super().initialize(info)
            self.dynamic_assignments: dict[int, tuple[int, tuple[Any, ...] | None, float]] = {}
            self.dynamic_records: list[dict[str, Any]] = []
            self.dynamic_inference_seconds = 0.0
            self._next_dynamic_time = 0.0
            self._previous_effective: dict[int, tuple[int, tuple[Any, ...] | None]] = {}

        def _plan(self, state, own, foes):
            should_decide = state.time + 1.0e-9 >= self._next_dynamic_time
            if ablation == "freeze_start" and self.dynamic_records:
                should_decide = False
            if should_decide:
                observation, scouts, live_candidates = tactical_observation(
                    self, state, own, foes, self.dynamic_assignments
                )
                started = time.perf_counter()
                with torch.no_grad():
                    decision = model.decide(
                        observation,
                        stochastic=stochastic,
                        generator=generator,
                    )
                self.dynamic_inference_seconds += time.perf_counter() - started
                updated = update_assignments(
                    state,
                    scouts,
                    live_candidates,
                    decision.actions,
                    self.dynamic_assignments,
                )
                effective = {
                    scout_id: (role, target)
                    for scout_id, (role, target, _since) in updated.items()
                }
                duty_changes = sum(
                    self._previous_effective.get(scout_id) != assignment
                    for scout_id, assignment in effective.items()
                )
                target_changes = sum(
                    scout_id in self._previous_effective
                    and self._previous_effective[scout_id][1] != assignment[1]
                    for scout_id, assignment in effective.items()
                )
                self.dynamic_assignments = updated
                self._previous_effective = effective
                self._next_dynamic_time = state.time + decision_interval
                self.dynamic_records.append(
                    {
                        "observation": observation,
                        "actions": decision.actions,
                        "log_probability": float(decision.log_probability.item()),
                        "value": float(decision.value.item()),
                        "entropy": float(decision.entropy.item()),
                        "factor_count": decision.factor_count,
                        "time": state.time,
                        "score_difference": state.own_score - state.opponent_score,
                        "selected_roles": [ROLE_NAMES[action.role] for action in decision.actions],
                        "selected_targets": [
                            updated[int(scout.id)][1] for scout in scouts
                        ],
                        "duty_changes": duty_changes,
                        "target_changes": target_changes,
                        "commands_different_from_mode_4": 0,
                    }
                )
            return resolve_tactical_duties(
                self, state, own, foes, self.dynamic_assignments
            )

        def note_shadow_commands(self, state, commands, shadow_commands):
            if not self.dynamic_records or abs(self.dynamic_records[-1]["time"] - state.time) > 1.0e-9:
                return
            scout_ids = {
                drone.id
                for drone in state.own_drones
                if drone.status is DroneStatus.ACTIVE and drone.drone_type is DroneType.SCOUT
            }
            self.dynamic_records[-1]["commands_different_from_mode_4"] = sum(
                commands.get(drone_id) != shadow_commands.get(drone_id)
                for drone_id in scout_ids
            )

    globals_ = base_type._plan.__globals__
    DynamicController.BASE_ROLES = tuple(
        globals_[name] for name in ("RUN", "HUNT", "KEEP", "GUN", "BLOCK")
    )
    return DynamicController


def _mode4_teacher_controller(base_type):
    """Collect safe mode-4 duties as a brief initialization target."""
    globals_ = base_type._plan.__globals__
    base_run, base_hunt, base_keep, _base_gun, base_block = (
        globals_[name] for name in ("RUN", "HUNT", "KEEP", "GUN", "BLOCK")
    )

    def teacher_action(base_role, mark):
        if base_role == base_run:
            return TACTICAL_RUN, None
        if base_role == base_keep:
            return TACTICAL_KEEP, None
        if base_role == base_block and mark is not None:
            return TACTICAL_BLOCK, (
                "BLOCK_LINE",
                int(mark[0].id),
                int(mark[1].id),
            )
        if base_role == base_hunt and mark is not None:
            # Fixed mode 4 has zero proactive transport/tank hunters; every
            # HUNT it emits originates from the pursuer guard pass.
            return GUARD_TRANSPORT, ("DRONE", int(mark.id))
        return TACTICAL_RUN, None

    class TeacherController(base_type):
        def initialize(self, info):
            super().initialize(info)
            self.teacher_assignments = {}
            self.teacher_records = []
            self._next_teacher_time = 0.0

        def _allocation(self, state, own, foes):
            return FIXED_MODE_4

        def _plan(self, state, own, foes):
            duties = super()._plan(state, own, foes)
            if state.time + 1.0e-9 < self._next_teacher_time:
                return duties
            observation, scouts, live_candidates = tactical_observation(
                self, state, own, foes, self.teacher_assignments
            )
            actions = []
            for scout, candidates_by_role in zip(scouts, live_candidates):
                base_role, mark = duties[scout.id]
                role, key = teacher_action(base_role, mark)
                target_index = -1
                if key is not None:
                    target_index = next(
                        (
                            index
                            for index, candidate in enumerate(candidates_by_role[role])
                            if candidate.key == key
                        ),
                        -1,
                    )
                    if target_index < 0:
                        role, key = TACTICAL_RUN, None
                actions.append(ScoutAction(role, target_index))
            action_tuple = tuple(actions)
            self.teacher_assignments = update_assignments(
                state,
                scouts,
                live_candidates,
                action_tuple,
                self.teacher_assignments,
            )
            self.teacher_records.append((observation, action_tuple))
            self._next_teacher_time = state.time + TACTICAL_INTERVAL
            return duties

    TeacherController.teacher_action = staticmethod(teacher_action)
    return TeacherController


@dataclass(frozen=True)
class BehaviorCloneJob:
    seed: int
    side: str
    opponent: str
    duration: float = DEFAULT_MATCH_DURATION


def run_behavior_clone_job(
    job: BehaviorCloneJob,
) -> list[tuple[TacticalObservation, tuple[ScoutAction, ...]]]:
    _configure_worker()
    scenario = generate_scenario(job.seed)
    simulator = Simulator(scenario)
    subject_team = Team(job.side)
    base_type = _load_controller(
        SUBJECT_PATH, f"bc_base_{os.getpid()}_{job.seed}"
    )
    teacher_type = _mode4_teacher_controller(base_type)
    if job.opponent == "fixed_mode_4":
        opponent_type = _instrumented_controller(base_type, force_mode_4=True)
    else:
        opponent_type = _load_controller(
            _opponent_path(job.opponent), f"bc_foe_{os.getpid()}_{job.seed}"
        )
    teacher = teacher_type()
    opponent = opponent_type()
    teacher.initialize(game_info(scenario, subject_team))
    opponent.initialize(game_info(scenario, subject_team.opponent))
    controllers = {subject_team: teacher, subject_team.opponent: opponent}
    control_stride = round(CONTROLLER_PERIOD / PHYSICS_DT)
    for tick in range(round(job.duration / PHYSICS_DT)):
        if tick % control_stride == 0:
            commands = {
                team: _commands(controllers[team], game_state(simulator, team))
                for team in Team
            }
            for team in Team:
                simulator.set_commands(team, commands[team])
        simulator.step()
        if not any(
            drone.status is DroneStatus.ACTIVE for drone in simulator.snapshots()
        ):
            break
    return teacher.teacher_records


def collect_behavior_clone_jobs(
    jobs: list[BehaviorCloneJob], workers: int
) -> tuple[list[tuple[TacticalObservation, tuple[ScoutAction, ...]]], float]:
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_behavior_clone_job, job) for job in jobs]
        for future in as_completed(futures):
            records.extend(future.result())
    return records, time.perf_counter() - started


@dataclass(frozen=True)
class DynamicRolloutJob:
    seed: int
    side: str
    opponent: str
    policy_version: int
    policy_state: dict[str, torch.Tensor]
    stochastic: bool = True
    duration: float = DEFAULT_MATCH_DURATION
    decision_interval: float = TACTICAL_INTERVAL
    reward_kind: str = "terminal"
    ablation: str = "full"
    diagnostics: bool = False


@dataclass
class DynamicEpisodeResult:
    seed: int
    side: str
    opponent: str
    policy_version: int
    observations: list[TacticalObservation]
    actions: list[tuple[ScoutAction, ...]]
    old_log_probabilities: list[float]
    old_values: list[float]
    rewards: list[float]
    dones: list[bool]
    score_for: int
    score_against: int
    outcome: float
    selected_role_counts: dict[str, int]
    duty_changes: int
    target_changes: int
    command_differences: int
    inference_seconds: float
    events: list[dict[str, Any]]
    score_timeline: list[dict[str, Any]]
    tactical_records: list[dict[str, Any]]
    wall_seconds: float

    @property
    def transitions(self) -> int:
        return len(self.observations)


def _frozen_dynamic_state(model: DynamicActorCritic) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def run_dynamic_rollout_job(job: DynamicRolloutJob) -> DynamicEpisodeResult:
    """Execute one complete factorized-policy episode with frozen weights."""
    _configure_worker()
    started = time.perf_counter()
    random.seed(job.seed ^ (0 if job.side == "A" else 0x6C8E9CF5))
    torch.manual_seed(job.seed % (2**31))
    model = DynamicActorCritic()
    model.load_state_dict(job.policy_state)
    model.eval()
    generator = torch.Generator().manual_seed(
        (job.seed * 6364136223846793005 + job.policy_version * 1447 + ord(job.side))
        % (2**63)
    )
    scenario = generate_scenario(job.seed)
    simulator = Simulator(scenario)
    subject_team = Team(job.side)
    base_type = _load_controller(
        SUBJECT_PATH, f"dynamic_base_{os.getpid()}_{job.seed}"
    )
    subject_type = _dynamic_controller(
        base_type,
        model,
        generator,
        stochastic=job.stochastic,
        decision_interval=job.decision_interval,
        ablation=job.ablation,
    )
    if job.opponent == "fixed_mode_4":
        opponent_type = _instrumented_controller(base_type, force_mode_4=True)
    else:
        opponent_type = _load_controller(
            _opponent_path(job.opponent),
            f"dynamic_foe_{os.getpid()}_{job.seed}",
        )
    subject = subject_type()
    opponent = opponent_type()
    own_info = game_info(scenario, subject_team)
    subject.initialize(own_info)
    opponent.initialize(game_info(scenario, subject_team.opponent))
    controllers = {subject_team: subject, subject_team.opponent: opponent}
    shadow = None
    if job.diagnostics:
        shadow_type = _instrumented_controller(base_type, force_mode_4=True)
        shadow = shadow_type()
        shadow.initialize(own_info)
    events = []
    score_timeline = [{"time": 0.0, "for": 0, "against": 0}]
    previous_scores = (0, 0)
    control_stride = round(CONTROLLER_PERIOD / PHYSICS_DT)
    for tick in range(round(job.duration / PHYSICS_DT)):
        if tick % control_stride == 0:
            states = {team: game_state(simulator, team) for team in Team}
            commands = {
                team: _commands(controllers[team], states[team])
                for team in Team
            }
            if shadow is not None:
                shadow_commands = _commands(shadow, states[subject_team])
                subject.note_shadow_commands(
                    states[subject_team], commands[subject_team], shadow_commands
                )
            for team in Team:
                simulator.set_commands(team, commands[team])
        resolved = simulator.step()
        if job.diagnostics:
            events.extend(_event_dict(event) for event in resolved)
        scores = (
            simulator.scores[subject_team],
            simulator.scores[subject_team.opponent],
        )
        if scores != previous_scores:
            score_timeline.append(
                {"time": simulator.time, "for": scores[0], "against": scores[1]}
            )
            previous_scores = scores
        if not any(
            drone.status is DroneStatus.ACTIVE for drone in simulator.snapshots()
        ):
            break
    score_for = simulator.scores[subject_team]
    score_against = simulator.scores[subject_team.opponent]
    outcome = (
        1.0
        if score_for > score_against
        else -1.0
        if score_for < score_against
        else 0.0
    )
    records = subject.dynamic_records
    rewards = dynamic_rewards(
        records,
        score_for - score_against,
        outcome,
        subject._initial_value,
        job.reward_kind,
    )
    role_counts = Counter(
        role
        for record in records
        for role in record["selected_roles"]
    )
    return DynamicEpisodeResult(
        seed=job.seed,
        side=job.side,
        opponent=job.opponent,
        policy_version=job.policy_version,
        observations=[record["observation"] for record in records],
        actions=[record["actions"] for record in records],
        old_log_probabilities=[record["log_probability"] for record in records],
        old_values=[record["value"] for record in records],
        rewards=rewards,
        dones=[False] * max(0, len(records) - 1) + ([True] if records else []),
        score_for=score_for,
        score_against=score_against,
        outcome=outcome,
        selected_role_counts=dict(role_counts),
        duty_changes=sum(record["duty_changes"] for record in records[1:]),
        target_changes=sum(record["target_changes"] for record in records[1:]),
        command_differences=sum(
            record["commands_different_from_mode_4"] for record in records
        ),
        inference_seconds=subject.dynamic_inference_seconds,
        events=events,
        score_timeline=score_timeline,
        tactical_records=[
            {
                key: record[key]
                for key in (
                    "time",
                    "score_difference",
                    "selected_roles",
                    "selected_targets",
                    "duty_changes",
                    "target_changes",
                    "commands_different_from_mode_4",
                )
            }
            for record in records
        ],
        wall_seconds=time.perf_counter() - started,
    )


def collect_dynamic_jobs(
    jobs: list[DynamicRolloutJob], workers: int
) -> tuple[list[DynamicEpisodeResult], float]:
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_dynamic_rollout_job, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return results, time.perf_counter() - started


def collect_dynamic_with_pool(
    pool: ProcessPoolExecutor, jobs: list[DynamicRolloutJob]
) -> tuple[list[DynamicEpisodeResult], float]:
    started = time.perf_counter()
    futures = [pool.submit(run_dynamic_rollout_job, job) for job in jobs]
    results = [future.result() for future in as_completed(futures)]
    return results, time.perf_counter() - started


def dynamic_gae(
    episodes: list[DynamicEpisodeResult], gamma: float, gae_lambda: float
) -> dict[str, Any]:
    observations = []
    actions = []
    old_log_probabilities = []
    old_values = []
    advantages = []
    returns = []
    for episode in episodes:
        episode_advantages = [0.0] * episode.transitions
        gae = 0.0
        next_value = 0.0
        for index in reversed(range(episode.transitions)):
            nonterminal = 0.0 if episode.dones[index] else 1.0
            delta = (
                episode.rewards[index]
                + gamma * next_value * nonterminal
                - episode.old_values[index]
            )
            gae = delta + gamma * gae_lambda * nonterminal * gae
            episode_advantages[index] = gae
            next_value = episode.old_values[index]
        observations.extend(episode.observations)
        actions.extend(episode.actions)
        old_log_probabilities.extend(episode.old_log_probabilities)
        old_values.extend(episode.old_values)
        advantages.extend(episode_advantages)
        returns.extend(
            advantage + value
            for advantage, value in zip(episode_advantages, episode.old_values)
        )
    return {
        "observations": observations,
        "actions": actions,
        "old_log_probabilities": torch.tensor(
            old_log_probabilities, dtype=torch.float32
        ),
        "old_values": torch.tensor(old_values, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "returns": torch.tensor(returns, dtype=torch.float32),
    }


def dynamic_rewards(
    records: list[dict[str, Any]],
    final_difference: int,
    outcome: float,
    maximum_score: int,
    kind: str,
) -> list[float]:
    """Winning dominates; optional score potential has total magnitude <= 0.1."""
    if kind not in {"terminal", "score_potential"}:
        raise ValueError(f"unknown dynamic reward: {kind}")
    rewards = [0.0] * len(records)
    if not rewards:
        return rewards
    if kind == "score_potential":
        differences = [
            int(record["score_difference"]) for record in records
        ] + [final_difference]
        for index in range(len(records)):
            rewards[index] = (
                0.1
                * (differences[index + 1] - differences[index])
                / max(1, maximum_score)
            )
    rewards[-1] += outcome
    return rewards


@dataclass(frozen=True)
class DynamicPPOConfig:
    seed: int = 93_001
    workers: int = 6
    episodes_per_iteration: int = 18
    iterations: int = 10
    duration: float = DEFAULT_MATCH_DURATION
    decision_interval: float = TACTICAL_INTERVAL
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 1.0e-4
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    target_kl: float = 0.02
    validation_interval: int = 5
    run_bias: float = 1.0
    reward_kind: str = "terminal"


class AdaptiveOpponentLeague:
    """Small evidence-driven mixture: 25% uniform, 75% useful difficulty."""

    def __init__(self) -> None:
        names = sorted(LEAGUE_OPPONENTS)
        self.names = tuple(names + ["fixed_mode_4"])
        self.stats = {
            name: {"games": 0, "points": 0.0, "wins": 0, "draws": 0, "losses": 0}
            for name in self.names
        }
        rating_data = json.loads(
            (ROOT / "leaderboard" / "ratings.json").read_text(encoding="utf-8")
        )
        rating_by_id = {
            controller["controller_id"]: float(controller["rating"])
            for controller in rating_data["controllers"]
        }
        aliases = {"potential": "potential_field", "greedy": "greedy_value"}
        self.ratings = {
            name: (
                2000.0
                if name == "fixed_mode_4"
                else rating_by_id.get(aliases.get(name, name), 1500.0)
            )
            for name in self.names
        }

    def weights(self) -> dict[str, float]:
        ratings = list(self.ratings.values())
        rating_low, rating_high = min(ratings), max(ratings)
        usefulness = {}
        for name in self.names:
            stats = self.stats[name]
            if stats["games"]:
                point_rate = stats["points"] / stats["games"]
                learner_loss_rate = stats["losses"] / stats["games"]
                near_parity = 1.0 - min(1.0, abs(point_rate - 0.5) * 2.0)
            else:
                learner_loss_rate = 0.5
                near_parity = 1.0
            strength = (self.ratings[name] - rating_low) / max(
                1.0, rating_high - rating_low
            )
            hard_bonus = 0.5 if name in HARD_OPPONENTS or name == "fixed_mode_4" else 0.0
            usefulness[name] = (
                0.25
                + 1.5 * learner_loss_rate
                + near_parity
                + 0.75 * strength
                + hard_bonus
            )
        total = sum(usefulness.values())
        uniform = 0.25 / len(self.names)
        return {
            name: uniform + 0.75 * usefulness[name] / total
            for name in self.names
        }

    def sample(self, rng: random.Random) -> str:
        weights = self.weights()
        return rng.choices(self.names, [weights[name] for name in self.names], k=1)[0]

    def record(self, episodes: list[DynamicEpisodeResult]) -> None:
        for episode in episodes:
            stats = self.stats[episode.opponent]
            stats["games"] += 1
            stats["points"] += 1.0 if episode.outcome > 0 else 0.5 if episode.outcome == 0 else 0.0
            if episode.outcome > 0:
                stats["wins"] += 1
            elif episode.outcome < 0:
                stats["losses"] += 1
            else:
                stats["draws"] += 1

    def state_dict(self) -> dict[str, Any]:
        return {"stats": self.stats}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name, values in state.get("stats", {}).items():
            if name in self.stats:
                self.stats[name].update(values)


def _explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    variance = torch.var(targets, unbiased=False)
    if float(variance) < 1.0e-12:
        return 0.0
    return float(
        1.0 - torch.var(targets - predictions, unbiased=False) / variance
    )


def dynamic_ppo_update(
    model: DynamicActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any],
    config: DynamicPPOConfig,
    generator: torch.Generator,
) -> dict[str, float]:
    advantages = batch["advantages"]
    advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1.0e-8)
    count = len(advantages)
    summaries = []
    epochs_completed = 0
    started = time.perf_counter()
    for epoch in range(config.update_epochs):
        permutation = torch.randperm(count, generator=generator)
        epoch_kls = []
        for start in range(0, count, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            minibatch_observations = [
                batch["observations"][int(index)] for index in indices
            ]
            minibatch_actions = [
                batch["actions"][int(index)] for index in indices
            ]
            new_log_probabilities, values, entropies = model.evaluate_batch(
                minibatch_observations, minibatch_actions
            )
            old_log_probabilities = batch["old_log_probabilities"][indices]
            log_ratio = new_log_probabilities - old_log_probabilities
            ratio = log_ratio.exp()
            minibatch_advantages = advantages[indices]
            unclipped = ratio * minibatch_advantages
            clipped = torch.clamp(
                ratio,
                1.0 - config.clip_epsilon,
                1.0 + config.clip_epsilon,
            ) * minibatch_advantages
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            critic_loss = 0.5 * (
                values - batch["returns"][indices]
            ).square().mean()
            entropy = entropies.mean()
            loss = (
                actor_loss
                + config.value_coefficient * critic_loss
                - config.entropy_coefficient * entropy
            )
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_gradient_norm
            )
            optimizer.step()
            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > config.clip_epsilon
                ).float().mean()
            kl = float(approximate_kl)
            epoch_kls.append(kl)
            summaries.append(
                {
                    "actor_loss": float(actor_loss.detach()),
                    "critic_loss": float(critic_loss.detach()),
                    "entropy": float(entropy.detach()),
                    "total_loss": float(loss.detach()),
                    "approximate_kl": kl,
                    "clip_fraction": float(clip_fraction),
                    "gradient_norm": float(gradient_norm),
                }
            )
        epochs_completed = epoch + 1
        if statistics.fmean(epoch_kls) > config.target_kl:
            break
    metrics = {
        key: statistics.fmean(summary[key] for summary in summaries)
        for key in summaries[0]
    }
    metrics.update(
        {
            "epochs_completed": epochs_completed,
            "optimization_seconds": time.perf_counter() - started,
            "explained_variance_before": _explained_variance(
                batch["old_values"], batch["returns"]
            ),
        }
    )
    return metrics


def summarize_dynamic_episodes(
    episodes: list[DynamicEpisodeResult], elapsed: float | None = None
) -> dict[str, Any]:
    outcomes = [episode.outcome for episode in episodes]
    differences = [episode.score_for - episode.score_against for episode in episodes]
    role_counts = Counter()
    event_counts = Counter()
    for episode in episodes:
        role_counts.update(episode.selected_role_counts)
        event_counts.update(event["type"] for event in episode.events)
    role_total = sum(role_counts.values())
    transitions = sum(episode.transitions for episode in episodes)
    inference = sum(episode.inference_seconds for episode in episodes)
    summary = {
        "matches": len(episodes),
        "wins": sum(outcome > 0 for outcome in outcomes),
        "draws": sum(outcome == 0 for outcome in outcomes),
        "losses": sum(outcome < 0 for outcome in outcomes),
        "match_points": sum(
            1.0 if outcome > 0 else 0.5 if outcome == 0 else 0.0
            for outcome in outcomes
        ),
        "mean_score_difference": statistics.fmean(differences),
        "median_score_difference": statistics.median(differences),
        "transitions": transitions,
        "role_frequencies": {
            role: role_counts[role] / max(1, role_total) for role in ROLE_NAMES
        },
        "duty_changes_per_match": statistics.fmean(
            episode.duty_changes for episode in episodes
        ),
        "target_changes_per_match": statistics.fmean(
            episode.target_changes for episode in episodes
        ),
        "command_differences_per_match": statistics.fmean(
            episode.command_differences for episode in episodes
        ),
        "mean_inference_microseconds": 1.0e6 * inference / max(1, transitions),
        "event_counts": dict(event_counts),
        "by_opponent": {},
        "by_side": {},
    }
    for opponent in sorted({episode.opponent for episode in episodes}):
        subset = [episode for episode in episodes if episode.opponent == opponent]
        summary["by_opponent"][opponent] = {
            "wins": sum(episode.outcome > 0 for episode in subset),
            "draws": sum(episode.outcome == 0 for episode in subset),
            "losses": sum(episode.outcome < 0 for episode in subset),
            "mean_score_difference": statistics.fmean(
                episode.score_for - episode.score_against for episode in subset
            ),
        }
    for side in ("A", "B"):
        subset = [episode for episode in episodes if episode.side == side]
        summary["by_side"][side] = {
            "wins": sum(episode.outcome > 0 for episode in subset),
            "draws": sum(episode.outcome == 0 for episode in subset),
            "losses": sum(episode.outcome < 0 for episode in subset),
            "mean_score_difference": statistics.fmean(
                episode.score_for - episode.score_against for episode in subset
            ),
        }
    if elapsed is not None:
        summary["wall_seconds"] = elapsed
        summary["matches_per_second"] = len(episodes) / elapsed
        summary["decisions_per_second"] = transitions / elapsed
    return summary


DYNAMIC_VALIDATION_SEEDS = (93_100_019, 93_100_057)
DYNAMIC_VALIDATION_OPPONENTS = HARD_OPPONENTS + ("fixed_mode_4",)


def _dynamic_training_jobs(
    config: DynamicPPOConfig,
    model: DynamicActorCritic,
    version: int,
    rng: random.Random,
    league: AdaptiveOpponentLeague,
) -> list[DynamicRolloutJob]:
    policy_state = _frozen_dynamic_state(model)
    return [
        DynamicRolloutJob(
            seed=rng.randrange(10_000_000, 90_000_000),
            side="A" if (index + version) % 2 == 0 else "B",
            opponent=league.sample(rng),
            policy_version=version,
            policy_state=policy_state,
            stochastic=True,
            duration=config.duration,
            decision_interval=config.decision_interval,
            reward_kind=config.reward_kind,
        )
        for index in range(config.episodes_per_iteration)
    ]


def _dynamic_validation_jobs(
    model: DynamicActorCritic,
    version: int,
    duration: float,
    *,
    seeds: tuple[int, ...] = DYNAMIC_VALIDATION_SEEDS,
    opponents: tuple[str, ...] = DYNAMIC_VALIDATION_OPPONENTS,
    ablation: str = "full",
    diagnostics: bool = True,
) -> list[DynamicRolloutJob]:
    policy_state = _frozen_dynamic_state(model)
    return [
        DynamicRolloutJob(
            seed=seed,
            side=side,
            opponent=opponent,
            policy_version=version,
            policy_state=policy_state,
            stochastic=False,
            duration=duration,
            ablation=ablation,
            diagnostics=diagnostics,
        )
        for opponent in opponents
        for seed in seeds
        for side in ("A", "B")
    ]


def _atomic_dynamic_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _dynamic_checkpoint_payload(
    model,
    optimizer,
    config,
    iteration,
    total_matches,
    total_decisions,
    rng,
    update_generator,
    league,
    best_validation,
):
    return {
        "schema": "opus-rl-plan-dynamic-ppo-v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(config),
        "iteration": iteration,
        "total_matches": total_matches,
        "total_decisions": total_decisions,
        "python_rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "update_generator_state": update_generator.get_state(),
        "league": league.state_dict(),
        "best_validation": best_validation,
        "observation_normalization": "fixed_physical_scaling",
    }


def _append_json_line(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def _validation_key(summary: dict[str, Any]) -> tuple[float, float]:
    return float(summary["match_points"]), float(summary["mean_score_difference"])


def train_dynamic_ppo(args) -> None:
    config = DynamicPPOConfig(
        seed=args.seed,
        workers=args.workers,
        episodes_per_iteration=args.episodes,
        iterations=args.iterations,
        duration=args.duration,
        decision_interval=args.decision_interval,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip,
        update_epochs=args.epochs,
        minibatch_size=args.minibatch,
        learning_rate=args.learning_rate,
        entropy_coefficient=args.entropy,
        value_coefficient=args.value_coefficient,
        max_gradient_norm=args.max_gradient_norm,
        target_kl=args.target_kl,
        validation_interval=args.validation_interval,
        run_bias=args.run_bias,
        reward_kind=args.reward,
    )
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    model = DynamicActorCritic(config.run_bias)
    if args.initialize and args.resume:
        raise ValueError("--initialize and --resume are mutually exclusive")
    if args.initialize:
        initialization = torch.load(
            Path(args.initialize), map_location="cpu", weights_only=False
        )
        model.load_state_dict(initialization["model_state"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, eps=1.0e-5
    )
    rng = random.Random(config.seed)
    update_generator = torch.Generator().manual_seed(config.seed + 1)
    league = AdaptiveOpponentLeague()
    start_iteration = total_matches = total_decisions = 0
    best_validation = None
    run_dir = EXPERIMENT_DIR / args.run_name
    latest_path = run_dir / "latest.pt"
    metrics_path = run_dir / "metrics.jsonl"
    if args.resume:
        checkpoint = torch.load(Path(args.resume), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_iteration = int(checkpoint["iteration"])
        total_matches = int(checkpoint["total_matches"])
        total_decisions = int(checkpoint["total_decisions"])
        rng.setstate(checkpoint["python_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        update_generator.set_state(checkpoint["update_generator_state"])
        league.load_state_dict(checkpoint.get("league", {}))
        best_validation = checkpoint.get("best_validation")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        for iteration in range(start_iteration, config.iterations):
            iteration_started = time.perf_counter()
            sampling_weights = league.weights()
            jobs = _dynamic_training_jobs(config, model, iteration, rng, league)
            episodes, rollout_seconds = collect_dynamic_with_pool(pool, jobs)
            if any(episode.policy_version != iteration for episode in episodes):
                raise RuntimeError("stale factorized-policy trajectory detected")
            league.record(episodes)
            batch = dynamic_gae(episodes, config.gamma, config.gae_lambda)
            ppo_metrics = dynamic_ppo_update(
                model, optimizer, batch, config, update_generator
            )
            total_matches += len(episodes)
            total_decisions += sum(episode.transitions for episode in episodes)
            elapsed = time.perf_counter() - iteration_started
            metric = {
                "iteration": iteration + 1,
                "policy_version": iteration,
                "total_matches": total_matches,
                "total_decisions": total_decisions,
                "workers": config.workers,
                "rollout_seconds": rollout_seconds,
                "matches_per_second": len(episodes) / rollout_seconds,
                "decisions_per_second": sum(episode.transitions for episode in episodes)
                / rollout_seconds,
                "collection_fraction": rollout_seconds / elapsed,
                "optimization_fraction": ppo_metrics["optimization_seconds"] / elapsed,
                "training": summarize_dynamic_episodes(episodes),
                "ppo": ppo_metrics,
                "league_weights": sampling_weights,
                "league_stats": league.state_dict()["stats"],
                "jobs": [
                    {"seed": job.seed, "side": job.side, "opponent": job.opponent}
                    for job in jobs
                ],
            }
            if (
                (iteration + 1) % config.validation_interval == 0
                or iteration + 1 == config.iterations
            ):
                validation_episodes, validation_seconds = collect_dynamic_with_pool(
                    pool,
                    _dynamic_validation_jobs(
                        model, iteration + 1, config.duration
                    ),
                )
                validation = summarize_dynamic_episodes(
                    validation_episodes, validation_seconds
                )
                metric["validation"] = validation
                if best_validation is None or _validation_key(validation) > tuple(
                    best_validation["key"]
                ):
                    best_validation = {
                        "iteration": iteration + 1,
                        "key": list(_validation_key(validation)),
                        "summary": validation,
                    }
                    _atomic_dynamic_checkpoint(
                        _dynamic_checkpoint_payload(
                            model,
                            optimizer,
                            config,
                            iteration + 1,
                            total_matches,
                            total_decisions,
                            rng,
                            update_generator,
                            league,
                            best_validation,
                        ),
                        run_dir / "best.pt",
                    )
                _atomic_dynamic_checkpoint(
                    _dynamic_checkpoint_payload(
                        model,
                        optimizer,
                        config,
                        iteration + 1,
                        total_matches,
                        total_decisions,
                        rng,
                        update_generator,
                        league,
                        best_validation,
                    ),
                    run_dir / f"validation-{iteration + 1:04d}.pt",
                )
            checkpoint_started = time.perf_counter()
            _atomic_dynamic_checkpoint(
                _dynamic_checkpoint_payload(
                    model,
                    optimizer,
                    config,
                    iteration + 1,
                    total_matches,
                    total_decisions,
                    rng,
                    update_generator,
                    league,
                    best_validation,
                ),
                latest_path,
            )
            metric["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
            _append_json_line(metrics_path, metric)
            compact = {
                "iteration": iteration + 1,
                "WDL": [
                    metric["training"][key]
                    for key in ("wins", "draws", "losses")
                ],
                "mps": round(metric["matches_per_second"], 3),
                "dps": round(metric["decisions_per_second"], 2),
                "entropy": round(ppo_metrics["entropy"], 3),
                "kl": round(ppo_metrics["approximate_kl"], 5),
                "validation": metric.get("validation", {}).get("match_points"),
            }
            print(json.dumps(compact, sort_keys=True), flush=True)


def evaluate_dynamic_checkpoint(args) -> None:
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    model = DynamicActorCritic()
    model.load_state_dict(checkpoint["model_state"])
    jobs = _dynamic_validation_jobs(
        model,
        int(checkpoint["iteration"]),
        args.duration,
        seeds=tuple(args.seeds),
        opponents=tuple(args.opponents),
        ablation=args.ablation,
        diagnostics=True,
    )
    episodes, elapsed = collect_dynamic_jobs(jobs, args.workers)
    summary = summarize_dynamic_episodes(episodes, elapsed)
    summary["checkpoint"] = args.checkpoint
    summary["ablation"] = args.ablation
    summary["failures"] = [
        {
            "seed": episode.seed,
            "side": episode.side,
            "opponent": episode.opponent,
            "score": [episode.score_for, episode.score_against],
            "score_timeline": episode.score_timeline,
            "event_counts": dict(Counter(event["type"] for event in episode.events)),
            "role_counts": episode.selected_role_counts,
            "duty_changes": episode.duty_changes,
            "target_changes": episode.target_changes,
            "command_differences": episode.command_differences,
            "tactical_records": episode.tactical_records,
        }
        for episode in episodes
        if episode.outcome <= 0
    ]
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "wins",
                    "draws",
                    "losses",
                    "match_points",
                    "mean_score_difference",
                    "duty_changes_per_match",
                    "target_changes_per_match",
                    "command_differences_per_match",
                    "mean_inference_microseconds",
                )
            },
            sort_keys=True,
        )
    )


def behavior_clone_mode4(args) -> None:
    """Briefly clone safe mode 4, then leave PPO entirely unconstrained."""
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    league = AdaptiveOpponentLeague()
    weights = league.weights()
    jobs = [
        BehaviorCloneJob(
            seed=rng.randrange(10_000_000, 90_000_000),
            side="A" if index % 2 == 0 else "B",
            opponent=rng.choices(
                league.names, [weights[name] for name in league.names], k=1
            )[0],
            duration=args.duration,
        )
        for index in range(args.matches)
    ]
    records, collection_seconds = collect_behavior_clone_jobs(jobs, args.workers)
    model = DynamicActorCritic(run_bias=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    update_generator = torch.Generator().manual_seed(args.seed + 1)
    losses = []
    started = time.perf_counter()
    for _epoch in range(args.epochs):
        permutation = torch.randperm(len(records), generator=update_generator)
        for start in range(0, len(records), args.minibatch):
            indices = permutation[start : start + args.minibatch]
            observations = [records[int(index)][0] for index in indices]
            actions = [records[int(index)][1] for index in indices]
            log_probabilities, _values, _entropies = model.evaluate_batch(
                observations, actions
            )
            factor_counts = torch.tensor(
                [
                    sum(1 + int(action.target_index >= 0) for action in assignment)
                    for assignment in actions
                ],
                dtype=torch.float32,
            )
            loss = -(log_probabilities / factor_counts.clamp_min(1.0)).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
    role_counts = Counter(
        ROLE_NAMES[action.role]
        for _observation, actions in records
        for action in actions
    )
    run_dir = EXPERIMENT_DIR / args.run_name
    checkpoint = {
        "schema": "opus-rl-plan-dynamic-bc-v1",
        "model_state": model.state_dict(),
        "iteration": 0,
        "seed": args.seed,
        "matches": args.matches,
        "records": len(records),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "teacher": "fixed_mode_4",
        "league_weights": weights,
    }
    _atomic_dynamic_checkpoint(checkpoint, run_dir / "bc.pt")
    report = {
        key: checkpoint[key]
        for key in (
            "schema",
            "seed",
            "matches",
            "records",
            "epochs",
            "learning_rate",
            "teacher",
        )
    }
    report.update(
        {
            "collection_seconds": collection_seconds,
            "optimization_seconds": time.perf_counter() - started,
            "final_loss": losses[-1],
            "mean_loss": statistics.fmean(losses),
            "teacher_role_counts": dict(role_counts),
            "checkpoint": str(run_dir / "bc.pt"),
        }
    )
    (run_dir / "bc-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


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


def benchmark_dynamic_workers(args) -> None:
    torch.manual_seed(args.seed)
    model = DynamicActorCritic()
    state = _frozen_dynamic_state(model)
    output = []
    for workers in args.levels:
        jobs = [
            DynamicRolloutJob(
                seed=args.seed + index,
                side="A" if index % 2 == 0 else "B",
                opponent=args.opponent,
                policy_version=0,
                policy_state=state,
                stochastic=False,
                duration=args.duration,
            )
            for index in range(args.matches)
        ]
        episodes, elapsed = collect_dynamic_jobs(jobs, workers)
        row = {
            "workers": workers,
            "matches": len(episodes),
            "duration": args.duration,
            "wall_seconds": elapsed,
            "matches_per_second": len(episodes) / elapsed,
            "decisions_per_second": sum(
                episode.transitions for episode in episodes
            )
            / elapsed,
            "mean_inference_microseconds": 1.0e6
            * sum(episode.inference_seconds for episode in episodes)
            / max(1, sum(episode.transitions for episode in episodes)),
            "controller_subprocesses": 0,
        }
        output.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")


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

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--levels", nargs="+", type=int, default=[1, 2, 4, 6, 8, 12])
    benchmark.add_argument("--matches", type=int, default=12)
    benchmark.add_argument("--duration", type=float, default=30.0)
    benchmark.add_argument("--opponent", choices=tuple(sorted(LEAGUE_OPPONENTS)), default="renj1ete0/opus_5_v1")
    benchmark.add_argument("--seed", type=int, default=93_000_001)
    benchmark.add_argument("--output")
    benchmark.set_defaults(function=benchmark_dynamic_workers)

    train = subparsers.add_parser("train")
    train.add_argument("--run-name", required=True)
    train.add_argument("--seed", type=int, default=DynamicPPOConfig.seed)
    train.add_argument("--workers", type=int, default=DynamicPPOConfig.workers)
    train.add_argument("--episodes", type=int, default=DynamicPPOConfig.episodes_per_iteration)
    train.add_argument("--iterations", type=int, default=DynamicPPOConfig.iterations)
    train.add_argument("--duration", type=float, default=DynamicPPOConfig.duration)
    train.add_argument("--decision-interval", type=float, default=DynamicPPOConfig.decision_interval)
    train.add_argument("--gamma", type=float, default=DynamicPPOConfig.gamma)
    train.add_argument("--gae-lambda", type=float, default=DynamicPPOConfig.gae_lambda)
    train.add_argument("--clip", type=float, default=DynamicPPOConfig.clip_epsilon)
    train.add_argument("--epochs", type=int, default=DynamicPPOConfig.update_epochs)
    train.add_argument("--minibatch", type=int, default=DynamicPPOConfig.minibatch_size)
    train.add_argument("--learning-rate", type=float, default=DynamicPPOConfig.learning_rate)
    train.add_argument("--entropy", type=float, default=DynamicPPOConfig.entropy_coefficient)
    train.add_argument("--value-coefficient", type=float, default=DynamicPPOConfig.value_coefficient)
    train.add_argument("--max-gradient-norm", type=float, default=DynamicPPOConfig.max_gradient_norm)
    train.add_argument("--target-kl", type=float, default=DynamicPPOConfig.target_kl)
    train.add_argument("--validation-interval", type=int, default=DynamicPPOConfig.validation_interval)
    train.add_argument("--run-bias", type=float, default=DynamicPPOConfig.run_bias)
    train.add_argument(
        "--reward",
        choices=("terminal", "score_potential"),
        default=DynamicPPOConfig.reward_kind,
    )
    train.add_argument("--initialize")
    train.add_argument("--resume")
    train.set_defaults(function=train_dynamic_ppo)

    clone = subparsers.add_parser("clone")
    clone.add_argument("--run-name", required=True)
    clone.add_argument("--seed", type=int, default=93_001)
    clone.add_argument("--workers", type=int, default=6)
    clone.add_argument("--matches", type=int, default=18)
    clone.add_argument("--duration", type=float, default=DEFAULT_MATCH_DURATION)
    clone.add_argument("--epochs", type=int, default=3)
    clone.add_argument("--minibatch", type=int, default=64)
    clone.add_argument("--learning-rate", type=float, default=3.0e-4)
    clone.set_defaults(function=behavior_clone_mode4)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--workers", type=int, default=DynamicPPOConfig.workers)
    evaluate.add_argument("--duration", type=float, default=DEFAULT_MATCH_DURATION)
    evaluate.add_argument("--seeds", nargs="+", type=int, required=True)
    evaluate.add_argument(
        "--opponents",
        nargs="+",
        choices=tuple(sorted((*LEAGUE_OPPONENTS, "fixed_mode_4"))),
        required=True,
    )
    evaluate.add_argument("--ablation", choices=("full", "freeze_start"), default="full")
    evaluate.add_argument("--output")
    evaluate.set_defaults(function=evaluate_dynamic_checkpoint)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
