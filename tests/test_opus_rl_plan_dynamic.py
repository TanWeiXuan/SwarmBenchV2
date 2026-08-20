from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from swarmbench import DroneType

from experiments.opus_rl_plan_dynamic import (
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    GUARD_TRANSPORT,
    HUNT_TRANSPORT,
    PAIR_FEATURES,
    ROLE_COUNT,
    ROLE_NAMES,
    ScoutAction,
    TACTICAL_RUN,
    AdaptiveOpponentLeague,
    CandidateObservation,
    DynamicActorCritic,
    ScoutObservation,
    TacticalObservation,
    SUBJECT_PATH,
    _mode4_teacher_controller,
    _target_signature,
    dynamic_rewards,
    run_instrumented_match,
)
from experiments.opus_rl_plan_ppo import _load_controller


def test_target_signature_uses_type_and_id_without_semantic_id_features() -> None:
    target = SimpleNamespace(id=17, drone_type=DroneType.TRANSPORT)
    assert _target_signature(target) == ("TRANSPORT", 17)
    ward = SimpleNamespace(id=3, drone_type=DroneType.TRANSPORT)
    gun = SimpleNamespace(id=9, drone_type=DroneType.TANK)
    assert _target_signature((ward, gun)) == ("BLOCK_LINE", 3, 9)


def test_mode4_hunt_is_labeled_by_guard_origin_not_target_type() -> None:
    base = _load_controller(SUBJECT_PATH, "teacher_role_test")
    teacher = _mode4_teacher_controller(base)
    hunt = base._plan.__globals__["HUNT"]
    tank = SimpleNamespace(id=9, drone_type=DroneType.TANK)
    role, key = teacher.teacher_action(hunt, tank)
    assert ROLE_NAMES[role] == "GUARD_TRANSPORT"
    assert key == ("DRONE", 9)


def test_score_potential_remains_subordinate_to_match_outcome() -> None:
    records = [{"score_difference": 0}, {"score_difference": 5}]
    assert dynamic_rewards(records, 10, 1.0, 50, "terminal") == [0.0, 1.0]
    shaped = dynamic_rewards(records, 10, 1.0, 50, "score_potential")
    assert shaped == pytest.approx([0.01, 1.01])
    assert sum(shaped) == pytest.approx(1.02)


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
            candidates[GUARD_TRANSPORT] = (target,)
        scout_values.append(
            ScoutObservation(
                (0.0,) * ENTITY_FEATURES,
                TACTICAL_RUN,
                0.0,
                (True, candidate, False, candidate, False, False),
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
    batch_log_probability, batch_value, batch_entropy, imitation_loss = (
        model.evaluate_batch(
            [_observation(scouts=2, candidate=True)], [decision.actions]
        )
    )
    assert imitation_loss is None
    assert batch_log_probability[0].item() == pytest.approx(
        decision.log_probability.item(), abs=1.0e-6
    )
    assert batch_value[0].item() == pytest.approx(decision.value.item(), abs=1.0e-6)
    assert batch_entropy[0].item() == pytest.approx(decision.entropy.item(), abs=1.0e-6)


def test_weighted_imitation_loss_emphasizes_guard_roles() -> None:
    model = DynamicActorCritic(run_bias=0.0)
    observation = _observation(scouts=1, candidate=True)
    run = ((ScoutAction(TACTICAL_RUN, -1)),)
    guard = ((ScoutAction(GUARD_TRANSPORT, 0)),)
    weights = torch.ones(ROLE_COUNT)
    weights[GUARD_TRANSPORT] = 30.0
    _log_probability, _value, _entropy, loss = model.evaluate_batch(
        [observation, observation], [run, guard], weights
    )
    assert loss is not None
    role_logits = model.role_head(
        torch.cat(
            (
                model.encode_context(observation),
                model.entity_encoder(torch.tensor(observation.scouts[0].entity)),
                    torch.tensor(
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        + [0.0] * ROLE_COUNT
                    ),
            )
        )
    )
    role_logits = role_logits.masked_fill(
        ~torch.tensor(observation.scouts[0].base_role_mask), -1.0e9
    )
    expected_role_loss = -(
        torch.log_softmax(role_logits, dim=0)[TACTICAL_RUN]
        + 30.0 * torch.log_softmax(role_logits, dim=0)[GUARD_TRANSPORT]
    ) / 31.0
    # A single available target has zero target NLL, so this isolates role weighting.
    assert loss.item() == pytest.approx(expected_role_loss.item(), abs=1.0e-6)


def test_adaptive_league_is_normalized_and_contains_current_hard_field() -> None:
    league = AdaptiveOpponentLeague()
    weights = league.weights()
    assert sum(weights.values()) == pytest.approx(1.0)
    assert "renj1ete0/opus_5_v1" in weights
    assert "renj1ete0/GPT-5.3-Codex" in weights
    assert "renj1ete0/gemini_3_1_pro_v1" in weights
    assert "renj1ete0/sonnet_5_v3" in weights
    assert "fixed_mode_4" in weights
