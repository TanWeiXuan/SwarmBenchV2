from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from swarmbench import DroneType

from experiments.opus_rl_plan_dynamic import (
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    HUNT_TRANSPORT,
    PAIR_FEATURES,
    ROLE_COUNT,
    TACTICAL_RUN,
    AdaptiveOpponentLeague,
    CandidateObservation,
    DynamicActorCritic,
    ScoutObservation,
    TacticalObservation,
    _target_signature,
    run_instrumented_match,
)


def test_target_signature_uses_type_and_id_without_semantic_id_features() -> None:
    target = SimpleNamespace(id=17, drone_type=DroneType.TRANSPORT)
    assert _target_signature(target) == ("TRANSPORT", 17)
    ward = SimpleNamespace(id=3, drone_type=DroneType.TRANSPORT)
    gun = SimpleNamespace(id=9, drone_type=DroneType.TANK)
    assert _target_signature((ward, gun)) == ("BLOCK_LINE", 3, 9)


def test_fixed_mode_4_effective_instrumentation_is_self_consistent() -> None:
    result = run_instrumented_match(
        opponent="potential",
        seed=3_100_003,
        side="A",
        subject="fixed_mode_4",
        duration=2.0,
    )
    assert result["effective_decisions"] == 2
    assert result["scout_command_differences_from_mode_4"] == 0
    assert result["role_timeline"][0]["duties"]


def _observation(*, scouts: int, candidate: bool) -> TacticalObservation:
    target = CandidateObservation(("DRONE", 11), (0.0,) * PAIR_FEATURES)
    scout_values = []
    for _ in range(scouts):
        candidates = [tuple() for _ in range(ROLE_COUNT)]
        if candidate:
            candidates[HUNT_TRANSPORT] = (target,)
        scout_values.append(
            ScoutObservation(
                (0.0,) * ENTITY_FEATURES,
                TACTICAL_RUN,
                0.0,
                (True, candidate, False, False, False, False),
                tuple(candidates),
            )
        )
    return TacticalObservation(
        (0.0,) * GLOBAL_FEATURES,
        ((0.0,) * ENTITY_FEATURES, (1.0,) * ENTITY_FEATURES),
        ((0.25,) * ENTITY_FEATURES, (-0.5,) * ENTITY_FEATURES),
        tuple(scout_values),
    )


def test_dynamic_actor_is_small_and_set_context_is_permutation_invariant() -> None:
    model = DynamicActorCritic()
    observation = _observation(scouts=1, candidate=False)
    reversed_sets = TacticalObservation(
        observation.global_features,
        tuple(reversed(observation.own_entities)),
        tuple(reversed(observation.foe_entities)),
        observation.scouts,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000
    assert torch.allclose(model.encode_context(observation), model.encode_context(reversed_sets))


def test_autoregressive_mask_prevents_duplicate_target_assignment() -> None:
    model = DynamicActorCritic(run_bias=0.0)
    with torch.no_grad():
        model.role_head[-1].weight.zero_()
        model.role_head[-1].bias.zero_()
        model.role_head[-1].bias[HUNT_TRANSPORT] = 10.0
    decision = model.decide(_observation(scouts=2, candidate=True), stochastic=False)
    assert decision.actions[0].role == HUNT_TRANSPORT
    assert decision.actions[0].target_index == 0
    assert decision.actions[1].role == TACTICAL_RUN
    assert decision.factor_count == 3
    replay = model.decide(
        _observation(scouts=2, candidate=True),
        stochastic=False,
        actions=decision.actions,
    )
    assert torch.allclose(replay.log_probability, decision.log_probability)


def test_adaptive_league_is_normalized_and_contains_current_hard_field() -> None:
    league = AdaptiveOpponentLeague()
    weights = league.weights()
    assert sum(weights.values()) == pytest.approx(1.0)
    assert "renj1ete0/opus_5_v1" in weights
    assert "renj1ete0/GPT-5.3-Codex" in weights
    assert "renj1ete0/gemini_3_1_pro_v1" in weights
    assert "renj1ete0/sonnet_5_v3" in weights
    assert "fixed_mode_4" in weights
