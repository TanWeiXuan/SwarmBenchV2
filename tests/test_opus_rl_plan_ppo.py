from __future__ import annotations

import pytest
import torch

from experiments.opus_rl_plan_ppo import (
    ACTION_COUNT,
    OBSERVATION_SIZE,
    ActorCritic,
    EpisodeResult,
    _episode_rewards,
    compute_gae,
)


def test_actor_critic_is_small_and_has_expected_heads() -> None:
    model = ActorCritic()
    logits, values = model(torch.zeros((3, OBSERVATION_SIZE)))
    assert logits.shape == (3, ACTION_COUNT)
    assert values.shape == (3,)
    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000
    assert logits[0].argmax().item() == 0


def test_reward_keeps_score_shaping_below_win_loss_scale() -> None:
    records = [{"score_difference": 0}, {"score_difference": 5}]
    assert _episode_rewards(records, 10, 1.0, "terminal", 50) == [0.0, 1.0]
    shaped = _episode_rewards(records, 10, 1.0, "score_potential", 50)
    assert shaped == pytest.approx([0.01, 1.01])
    assert sum(shaped) == pytest.approx(1.02)


def test_gae_does_not_bootstrap_past_episode_end() -> None:
    episode = EpisodeResult(
        seed=1,
        side="A",
        opponent="opus",
        policy_version=0,
        observations=[[0.0] * OBSERVATION_SIZE, [0.0] * OBSERVATION_SIZE],
        actions=[0, 1],
        old_log_probs=[0.0, 0.0],
        old_values=[0.2, 0.4],
        rewards=[0.0, 1.0],
        dones=[False, True],
        decision_times=[0.0, 2.0],
        score_for=1,
        score_against=0,
        outcome=1.0,
        transitions=2,
        action_counts={0: 1, 1: 1},
        mode_switches=1,
        inference_seconds=0.0,
        wall_seconds=0.0,
    )
    batch = compute_gae([episode], gamma=1.0, gae_lambda=1.0)
    assert batch["advantages"].tolist() == pytest.approx([0.8, 0.6])
    assert batch["returns"].tolist() == pytest.approx([1.0, 1.0])
