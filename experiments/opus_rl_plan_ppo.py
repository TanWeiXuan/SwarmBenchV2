"""Local Experiment 2: sequential six-mode PPO for Opus_RL_Plan.

This module is research/training code, not a controller submission.  It runs
trusted repository controllers directly inside process-isolated match workers
while using the authoritative Scenario/Simulator implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
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

from swarmbench import DroneStatus, Team
from swarmbench.api import CONTROLLER_PERIOD, DEFAULT_MATCH_DURATION, PHYSICS_DT
from swarmbench.controllers.baselines import baseline_path
from swarmbench.engine import Simulator, generate_scenario
from swarmbench.match import game_info, game_state


ROOT = Path(__file__).resolve().parents[1]
SUBJECT_PATH = ROOT / "submissions" / "TanWeiXuan" / "Opus_RL_Plan.py"
EXPERIMENT_DIR = ROOT / ".rl_local" / "ppo"
OBSERVATION_SIZE = 38
ACTION_COUNT = 6
DECISION_INTERVAL = 2.0
VALIDATION_SEEDS = (9_100_019, 9_100_057)
VALIDATION_OPPONENTS = ("opus", "breaker", "sipp", "mpc", "wayfinder", "potential", "greedy", "bigpickle")

OPPONENTS = {
    "opus": ROOT / "submissions" / "renj1ete0" / "opus_5_v1.py",
    "breaker": ROOT / "submissions" / "TanWeiXuan" / "Luna_xHigh_opus_breaker.py",
    "sipp": ROOT / "submissions" / "TanWeiXuan" / "Luna_xHigh_sipp_marksman_v1.py",
    "mpc": ROOT / "submissions" / "TanWeiXuan" / "Luna_xHigh_mpc.py",
    "wayfinder": ROOT / "submissions" / "TanWeiXuan" / "wayfinder_v2.py",
    "bigpickle": ROOT / "submissions" / "TanWeiXuan" / "BigPickle_V1.py",
    "aegis": ROOT / "submissions" / "TanWeiXuan" / "aegis_apex_v2.py",
    "phalanx": ROOT / "submissions" / "TanWeiXuan" / "phalanx_v2.py",
    "sonnet": ROOT / "submissions" / "renj1ete0" / "sonnet_5_v2.py",
    "potential": baseline_path("potential_field"),
    "greedy": baseline_path("greedy_value"),
    "assignment": baseline_path("assignment"),
    "rush": baseline_path("rush"),
    "defend": baseline_path("defend"),
    "convoy": baseline_path("convoy"),
    "marksman": baseline_path("marksman"),
}


class ActorCritic(nn.Module):
    """Small shared trunk with categorical actor and scalar critic heads."""

    def __init__(self, opus_bias: float = 2.0) -> None:
        super().__init__()
        self.opus_bias = opus_bias
        self.trunk = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
            nn.Linear(48, 24),
            nn.Tanh(),
        )
        self.actor = nn.Linear(24, ACTION_COUNT)
        self.critic = nn.Linear(24, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.zeros_(self.actor.bias)
        # Positive bias starts near Opus; zero gives uniform stochastic
        # exploration while deterministic tie-breaking still selects Opus.
        with torch.no_grad():
            self.actor.bias[0] = self.opus_bias
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(observations)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


@dataclass(frozen=True)
class RolloutJob:
    seed: int
    side: str
    opponent: str
    policy_version: int
    policy_state: dict[str, torch.Tensor]
    stochastic: bool = True
    reward_kind: str = "terminal"
    switch_penalty: float = 0.0
    duration: float = DEFAULT_MATCH_DURATION


@dataclass
class EpisodeResult:
    seed: int
    side: str
    opponent: str
    policy_version: int
    observations: list[list[float]]
    actions: list[int]
    old_log_probs: list[float]
    old_values: list[float]
    rewards: list[float]
    dones: list[bool]
    decision_times: list[float]
    score_for: int
    score_against: int
    outcome: float
    transitions: int
    action_counts: dict[int, int]
    mode_switches: int
    inference_seconds: float
    wall_seconds: float


@dataclass(frozen=True)
class ReferenceJob:
    subject: str
    seed: int
    side: str
    opponent: str
    duration: float = DEFAULT_MATCH_DURATION


def _load_controller(path: Path, module_tag: str):
    spec = importlib.util.spec_from_file_location(module_tag, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load controller: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwarmController


def _configure_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _ppo_controller(base_type, model: ActorCritic, generator: torch.Generator, stochastic: bool):
    class PPOController(base_type):
        def initialize(self, info):
            super().initialize(info)
            self._plan_action = 0
            self._next_policy_time = 0.0
            self.ppo_records: list[dict[str, Any]] = []
            self.ppo_inference_seconds = 0.0

        def _allocation(self, state, own, foes):
            if state.time + 1.0e-9 >= self._next_policy_time:
                previous = self._plan_action
                observation = tuple(self._policy_features(state, own, foes)) + tuple(
                    1.0 if index == previous else 0.0 for index in range(ACTION_COUNT)
                )
                started = time.perf_counter()
                with torch.no_grad():
                    values = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    logits, critic = model(values)
                    distribution = torch.distributions.Categorical(logits=logits[0])
                    if stochastic:
                        action = int(torch.multinomial(distribution.probs, 1, generator=generator).item())
                    else:
                        action = int(logits[0].argmax().item())
                    log_probability = float(distribution.log_prob(torch.tensor(action)).item())
                self.ppo_inference_seconds += time.perf_counter() - started
                self._plan_action = action
                self._next_policy_time = state.time + DECISION_INTERVAL
                self.ppo_records.append(
                    {
                        "observation": list(observation),
                        "action": action,
                        "log_probability": log_probability,
                        "value": float(critic[0].item()),
                        "time": state.time,
                        "score_difference": state.own_score - state.opponent_score,
                        "previous_action": previous,
                    }
                )
            return self.PLAN_ACTIONS[self._plan_action] if hasattr(self, "PLAN_ACTIONS") else base_type.__module__

    # PLAN_ACTIONS is module-level in the submission, not a class attribute.
    PPOController.PLAN_ACTIONS = base_type._allocation.__globals__["PLAN_ACTIONS"]
    return PPOController


def _commands(controller, state):
    commands = controller.step(state)
    if not isinstance(commands, dict):
        raise TypeError("trusted controller returned a non-dict command set")
    return commands


def run_direct_match(
    controller_a_path: Path,
    controller_b_path: Path,
    *,
    seed: int,
    duration: float = DEFAULT_MATCH_DURATION,
) -> tuple[int, int]:
    """Run trusted deterministic controllers without nested subprocesses."""
    _configure_worker()
    scenario = generate_scenario(seed)
    simulator = Simulator(scenario)
    type_a = _load_controller(controller_a_path, f"direct_a_{seed}")
    type_b = _load_controller(controller_b_path, f"direct_b_{seed}")
    controllers = {Team.A: type_a(), Team.B: type_b()}
    for team in Team:
        controllers[team].initialize(game_info(scenario, team))
    control_stride = round(CONTROLLER_PERIOD / PHYSICS_DT)
    for tick in range(round(duration / PHYSICS_DT)):
        if tick % control_stride == 0:
            commands = {
                team: _commands(controllers[team], game_state(simulator, team))
                for team in Team
            }
            for team in Team:
                simulator.set_commands(team, commands[team])
        simulator.step()
        if not any(drone.status is DroneStatus.ACTIVE for drone in simulator.snapshots()):
            break
    return simulator.scores[Team.A], simulator.scores[Team.B]


def _episode_rewards(
    records: list[dict[str, Any]],
    final_difference: int,
    outcome: float,
    kind: str,
    maximum_score: int,
    switch_penalty: float = 0.0,
) -> list[float]:
    if kind not in {"terminal", "score_potential"}:
        raise ValueError(f"unknown reward kind: {kind}")
    rewards = [0.0] * len(records)
    if not rewards:
        return rewards
    if kind == "score_potential":
        differences = [int(record["score_difference"]) for record in records] + [final_difference]
        for index in range(len(records)):
            rewards[index] += 0.10 * (differences[index + 1] - differences[index]) / max(1, maximum_score)
    for index, record in enumerate(records):
        if index > 0 and record["action"] != record["previous_action"]:
            rewards[index] -= switch_penalty
    rewards[-1] += outcome
    return rewards


def run_rollout_job(job: RolloutJob) -> EpisodeResult:
    """Execute one complete PPO episode using a frozen policy snapshot."""
    _configure_worker()
    started = time.perf_counter()
    random.seed(job.seed ^ (0 if job.side == "A" else 0x5A5A5A5A))
    torch.manual_seed(job.seed % (2**31))
    model = ActorCritic()
    model.load_state_dict(job.policy_state)
    model.eval()
    generator = torch.Generator().manual_seed((job.seed * 6364136223846793005 + job.policy_version * 1447 + ord(job.side)) % (2**63))

    scenario = generate_scenario(job.seed)
    simulator = Simulator(scenario)
    subject_team = Team(job.side)
    opponent_path = OPPONENTS[job.opponent]
    base_type = _load_controller(SUBJECT_PATH, f"ppo_subject_{os.getpid()}_{job.seed}")
    subject_type = _ppo_controller(base_type, model, generator, job.stochastic)
    opponent_type = _load_controller(opponent_path, f"ppo_opponent_{os.getpid()}_{job.seed}")
    subject = subject_type()
    opponent = opponent_type()
    subject.initialize(game_info(scenario, subject_team))
    opponent.initialize(game_info(scenario, subject_team.opponent))
    controllers = {subject_team: subject, subject_team.opponent: opponent}

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
        if not any(drone.status is DroneStatus.ACTIVE for drone in simulator.snapshots()):
            break

    score_for = simulator.scores[subject_team]
    score_against = simulator.scores[subject_team.opponent]
    outcome = 1.0 if score_for > score_against else -1.0 if score_for < score_against else 0.0
    records = subject.ppo_records
    rewards = _episode_rewards(
        records,
        score_for - score_against,
        outcome,
        job.reward_kind,
        subject._initial_value,
        job.switch_penalty,
    )
    actions = [int(record["action"]) for record in records]
    switches = sum(record["action"] != record["previous_action"] for record in records)
    return EpisodeResult(
        seed=job.seed,
        side=job.side,
        opponent=job.opponent,
        policy_version=job.policy_version,
        observations=[record["observation"] for record in records],
        actions=actions,
        old_log_probs=[record["log_probability"] for record in records],
        old_values=[record["value"] for record in records],
        rewards=rewards,
        dones=[False] * max(0, len(records) - 1) + ([True] if records else []),
        decision_times=[record["time"] for record in records],
        score_for=score_for,
        score_against=score_against,
        outcome=outcome,
        transitions=len(records),
        action_counts=dict(Counter(actions)),
        mode_switches=switches,
        inference_seconds=subject.ppo_inference_seconds,
        wall_seconds=time.perf_counter() - started,
    )


def run_reference_job(job: ReferenceJob) -> EpisodeResult:
    subject_paths = {
        "controller": SUBJECT_PATH,
        "opus": OPPONENTS["opus"],
        "breaker": OPPONENTS["breaker"],
    }
    subject_path = subject_paths[job.subject]
    opponent_path = OPPONENTS[job.opponent]
    started = time.perf_counter()
    if job.side == "A":
        score_for, score_against = run_direct_match(subject_path, opponent_path, seed=job.seed, duration=job.duration)
    else:
        score_against, score_for = run_direct_match(opponent_path, subject_path, seed=job.seed, duration=job.duration)
    outcome = 1.0 if score_for > score_against else -1.0 if score_for < score_against else 0.0
    return EpisodeResult(
        seed=job.seed,
        side=job.side,
        opponent=job.opponent,
        policy_version=-1,
        observations=[],
        actions=[],
        old_log_probs=[],
        old_values=[],
        rewards=[],
        dones=[],
        decision_times=[],
        score_for=score_for,
        score_against=score_against,
        outcome=outcome,
        transitions=0,
        action_counts={},
        mode_switches=0,
        inference_seconds=0.0,
        wall_seconds=time.perf_counter() - started,
    )


def collect_jobs(jobs: list[RolloutJob], workers: int) -> tuple[list[EpisodeResult], float]:
    started = time.perf_counter()
    results: list[EpisodeResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_rollout_job, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return results, time.perf_counter() - started


def collect_with_pool(pool: ProcessPoolExecutor, jobs: list[RolloutJob]) -> tuple[list[EpisodeResult], float]:
    started = time.perf_counter()
    futures = [pool.submit(run_rollout_job, job) for job in jobs]
    results = [future.result() for future in as_completed(futures)]
    return results, time.perf_counter() - started


def _frozen_state(model: ActorCritic) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 82_002
    workers: int = 6
    episodes_per_iteration: int = 30
    iterations: int = 20
    duration: float = DEFAULT_MATCH_DURATION
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 6
    minibatch_size: int = 256
    learning_rate: float = 2.5e-4
    entropy_coefficient: float = 0.02
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    target_kl: float = 0.03
    reward_kind: str = "terminal"
    validation_interval: int = 5
    initial_opus_bias: float = 2.0
    switch_penalty: float = 0.0


TRAINING_OPPONENT_WEIGHTS = (
    ("opus", 4),
    ("breaker", 4),
    ("sipp", 2),
    ("mpc", 2),
    ("wayfinder", 2),
    ("potential", 2),
    ("greedy", 2),
    ("bigpickle", 1),
    ("aegis", 1),
    ("assignment", 1),
    ("rush", 1),
    ("defend", 1),
    ("convoy", 1),
    ("marksman", 1),
)


def _training_opponents() -> list[str]:
    return [name for name, weight in TRAINING_OPPONENT_WEIGHTS for _ in range(weight)]


def _training_jobs(config: PPOConfig, model: ActorCritic, iteration: int, rng: random.Random) -> list[RolloutJob]:
    state = _frozen_state(model)
    opponents = _training_opponents()
    jobs = []
    for index in range(config.episodes_per_iteration):
        # Reserve the compact, published validation range completely.
        seed = rng.randrange(1_000_000, 9_000_000)
        jobs.append(
            RolloutJob(
                seed=seed,
                side="A" if (index + iteration) % 2 == 0 else "B",
                opponent=rng.choice(opponents),
                policy_version=iteration,
                policy_state=state,
                stochastic=True,
                reward_kind=config.reward_kind,
                switch_penalty=config.switch_penalty,
                duration=config.duration,
            )
        )
    return jobs


def _validation_jobs(model: ActorCritic, policy_version: int, duration: float, seeds=VALIDATION_SEEDS, opponents=VALIDATION_OPPONENTS):
    state = _frozen_state(model)
    return [
        RolloutJob(
            seed=seed,
            side=side,
            opponent=opponent,
            policy_version=policy_version,
            policy_state=state,
            stochastic=False,
            reward_kind="terminal",
            duration=duration,
        )
        for opponent in opponents
        for seed in seeds
        for side in ("A", "B")
    ]


def compute_gae(episodes: list[EpisodeResult], gamma: float, gae_lambda: float):
    observations: list[list[float]] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    old_values: list[float] = []
    advantages: list[float] = []
    returns: list[float] = []
    for episode in episodes:
        episode_advantages = [0.0] * episode.transitions
        gae = 0.0
        next_value = 0.0
        for index in range(episode.transitions - 1, -1, -1):
            nonterminal = 0.0 if episode.dones[index] else 1.0
            delta = episode.rewards[index] + gamma * next_value * nonterminal - episode.old_values[index]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            episode_advantages[index] = gae
            next_value = episode.old_values[index]
        observations.extend(episode.observations)
        actions.extend(episode.actions)
        old_log_probs.extend(episode.old_log_probs)
        old_values.extend(episode.old_values)
        advantages.extend(episode_advantages)
        returns.extend(advantage + value for advantage, value in zip(episode_advantages, episode.old_values))
    return {
        "observations": torch.tensor(observations, dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.long),
        "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
        "old_values": torch.tensor(old_values, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "returns": torch.tensor(returns, dtype=torch.float32),
    }


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    variance = torch.var(targets, unbiased=False)
    if float(variance) < 1.0e-12:
        return 0.0
    return float(1.0 - torch.var(targets - predictions, unbiased=False) / variance)


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer, batch, config: PPOConfig, generator: torch.Generator):
    advantages = batch["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)
    count = len(advantages)
    summaries = []
    epochs_completed = 0
    started = time.perf_counter()
    for epoch in range(config.update_epochs):
        permutation = torch.randperm(count, generator=generator)
        epoch_kls = []
        for start in range(0, count, config.minibatch_size):
            indices = permutation[start:start + config.minibatch_size]
            logits, values = model(batch["observations"][indices])
            distribution = torch.distributions.Categorical(logits=logits)
            new_log_probs = distribution.log_prob(batch["actions"][indices])
            log_ratio = new_log_probs - batch["old_log_probs"][indices]
            ratio = log_ratio.exp()
            mb_advantages = advantages[indices]
            unclipped = ratio * mb_advantages
            clipped = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * mb_advantages
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            critic_loss = 0.5 * (values - batch["returns"][indices]).square().mean()
            entropy = distribution.entropy().mean()
            loss = actor_loss + config.value_coefficient * critic_loss - config.entropy_coefficient * entropy

            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
            optimizer.step()
            with torch.no_grad():
                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > config.clip_epsilon).float().mean()
            value = float(approximate_kl)
            epoch_kls.append(value)
            summaries.append(
                {
                    "actor_loss": float(actor_loss.detach()),
                    "critic_loss": float(critic_loss.detach()),
                    "entropy": float(entropy.detach()),
                    "total_loss": float(loss.detach()),
                    "approximate_kl": value,
                    "clip_fraction": float(clip_fraction),
                    "gradient_norm": float(gradient_norm),
                }
            )
        epochs_completed = epoch + 1
        if statistics.fmean(epoch_kls) > config.target_kl:
            break
    keys = summaries[0]
    metrics = {key: statistics.fmean(item[key] for item in summaries) for key in keys}
    metrics.update(
        {
            "epochs_completed": epochs_completed,
            "optimization_seconds": time.perf_counter() - started,
            "explained_variance_before": explained_variance(batch["old_values"], batch["returns"]),
        }
    )
    return metrics


def summarize_episodes(episodes: list[EpisodeResult]) -> dict[str, Any]:
    outcomes = [episode.outcome for episode in episodes]
    differences = [episode.score_for - episode.score_against for episode in episodes]
    action_counts = Counter()
    for episode in episodes:
        action_counts.update(episode.action_counts)
    total_actions = sum(action_counts.values())
    inference_seconds = sum(episode.inference_seconds for episode in episodes)
    return {
        "matches": len(episodes),
        "wins": sum(value > 0 for value in outcomes),
        "draws": sum(value == 0 for value in outcomes),
        "losses": sum(value < 0 for value in outcomes),
        "match_points": sum(1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in outcomes),
        "mean_score_difference": statistics.fmean(differences),
        "median_score_difference": statistics.median(differences),
        "decisions": total_actions,
        "action_frequencies": {str(index): action_counts[index] / max(1, total_actions) for index in range(ACTION_COUNT)},
        "mode_switches": sum(episode.mode_switches for episode in episodes),
        "switches_per_match": statistics.fmean(episode.mode_switches for episode in episodes),
        "policy_inference_seconds": inference_seconds,
        "mean_policy_inference_microseconds": 1.0e6 * inference_seconds / max(1, total_actions),
        # The trusted runner raises immediately on malformed commands or a
        # failed update, so every successfully returned episode has zero here.
        "invalid_actions": 0,
        "missed_updates": 0,
        "timeouts": 0,
    }


def validation_details(episodes: list[EpisodeResult]) -> dict[str, Any]:
    summary = summarize_episodes(episodes)
    transition_counts = Counter()
    for episode in episodes:
        transition_counts.update(zip(episode.actions, episode.actions[1:]))
    summary["mode_transitions"] = {
        f"{left}->{right}": transition_counts[(left, right)]
        for left in range(ACTION_COUNT)
        for right in range(ACTION_COUNT)
        if transition_counts[(left, right)]
    }
    summary["action_sequence_samples"] = [episode.actions for episode in episodes[:5] if episode.actions]
    summary["by_opponent"] = {
        opponent: summarize_episodes([episode for episode in episodes if episode.opponent == opponent])
        for opponent in sorted({episode.opponent for episode in episodes})
    }
    summary["by_side"] = {
        side: summarize_episodes([episode for episode in episodes if episode.side == side])
        for side in ("A", "B")
    }
    return summary


def _checkpoint_payload(model, optimizer, config, iteration, total_decisions, total_matches, rng, update_generator, best_validation):
    return {
        "schema": "opus-rl-plan-ppo-v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(config),
        "iteration": iteration,
        "total_decisions": total_decisions,
        "total_matches": total_matches,
        "python_rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "update_generator_state": update_generator.get_state(),
        "best_validation": best_validation,
        "observation_normalization": "fixed_physical_scaling",
    }


def atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _append_metric(path: Path, metric: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metric, sort_keys=True) + "\n")


def _validation_key(summary: dict[str, Any]) -> tuple[float, float]:
    return float(summary["match_points"]), float(summary["mean_score_difference"])


def train_ppo(args) -> None:
    config = PPOConfig(
        seed=args.seed,
        workers=args.workers,
        episodes_per_iteration=args.episodes,
        iterations=args.iterations,
        duration=args.duration,
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
        reward_kind=args.reward,
        validation_interval=args.validation_interval,
        initial_opus_bias=args.initial_opus_bias,
        switch_penalty=args.switch_penalty,
    )
    run_dir = EXPERIMENT_DIR / args.run_name
    checkpoint_path = run_dir / "latest.pt"
    metrics_path = run_dir / "metrics.jsonl"
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    model = ActorCritic(config.initial_opus_bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1.0e-5)
    rng = random.Random(config.seed)
    update_generator = torch.Generator().manual_seed(config.seed + 1)
    start_iteration = 0
    total_decisions = 0
    total_matches = 0
    best_validation = None
    if args.resume:
        checkpoint = torch.load(Path(args.resume), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_iteration = int(checkpoint["iteration"])
        total_decisions = int(checkpoint["total_decisions"])
        total_matches = int(checkpoint["total_matches"])
        rng.setstate(checkpoint["python_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        best_validation = checkpoint.get("best_validation")
        if "update_generator_state" in checkpoint:
            update_generator.set_state(checkpoint["update_generator_state"])

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        for iteration in range(start_iteration, config.iterations):
            iteration_started = time.perf_counter()
            jobs = _training_jobs(config, model, iteration, rng)
            episodes, rollout_seconds = collect_with_pool(pool, jobs)
            if any(episode.policy_version != iteration for episode in episodes):
                raise RuntimeError("stale policy trajectory detected")
            batch = compute_gae(episodes, config.gamma, config.gae_lambda)
            update_metrics = ppo_update(model, optimizer, batch, config, update_generator)
            total_decisions += sum(episode.transitions for episode in episodes)
            total_matches += len(episodes)
            total_seconds = time.perf_counter() - iteration_started
            metric = {
                "iteration": iteration + 1,
                "policy_version": iteration,
                "total_decisions": total_decisions,
                "total_matches": total_matches,
                "rollout_seconds": rollout_seconds,
                "matches_per_second": len(episodes) / rollout_seconds,
                "decisions_per_second": sum(episode.transitions for episode in episodes) / rollout_seconds,
                "optimization_fraction": update_metrics["optimization_seconds"] / total_seconds,
                "collection_fraction": rollout_seconds / total_seconds,
                "workers": config.workers,
                "controller_subprocesses": 0,
                "serialization_checkpoint_seconds": 0.0,
                "training": summarize_episodes(episodes),
                "ppo": update_metrics,
                "jobs": [{"seed": job.seed, "side": job.side, "opponent": job.opponent} for job in jobs],
            }
            if (iteration + 1) % config.validation_interval == 0 or iteration + 1 == config.iterations:
                validation_episodes, validation_seconds = collect_with_pool(
                    pool,
                    _validation_jobs(model, iteration + 1, config.duration),
                )
                validation = validation_details(validation_episodes)
                validation["wall_seconds"] = validation_seconds
                metric["validation"] = validation
                if best_validation is None or _validation_key(validation) > tuple(best_validation["key"]):
                    best_validation = {"iteration": iteration + 1, "key": list(_validation_key(validation)), "summary": validation}
                    atomic_torch_save(
                        _checkpoint_payload(model, optimizer, config, iteration + 1, total_decisions, total_matches, rng, update_generator, best_validation),
                        run_dir / "best.pt",
                    )
                atomic_torch_save(
                    _checkpoint_payload(model, optimizer, config, iteration + 1, total_decisions, total_matches, rng, update_generator, best_validation),
                    run_dir / f"validation-{iteration + 1:04d}.pt",
                )
            checkpoint_started = time.perf_counter()
            atomic_torch_save(
                _checkpoint_payload(model, optimizer, config, iteration + 1, total_decisions, total_matches, rng, update_generator, best_validation),
                checkpoint_path,
            )
            metric["serialization_checkpoint_seconds"] = time.perf_counter() - checkpoint_started
            _append_metric(metrics_path, metric)
            compact = {
                "iteration": metric["iteration"],
                "WDL": [metric["training"][key] for key in ("wins", "draws", "losses")],
                "reward": config.reward_kind,
                "mps": round(metric["matches_per_second"], 3),
                "dps": round(metric["decisions_per_second"], 2),
                "entropy": round(metric["ppo"]["entropy"], 3),
                "kl": round(metric["ppo"]["approximate_kl"], 5),
                "value_ev": round(metric["ppo"]["explained_variance_before"], 3),
                "validation": metric.get("validation", {}).get("match_points"),
            }
            print(json.dumps(compact, sort_keys=True), flush=True)


def evaluate_checkpoint(args) -> None:
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    model = ActorCritic()
    model.load_state_dict(checkpoint["model_state"])
    jobs = _validation_jobs(
        model,
        int(checkpoint["iteration"]),
        args.duration,
        seeds=tuple(args.seeds),
        opponents=tuple(args.opponents),
    )
    episodes, elapsed = collect_jobs(jobs, args.workers)
    summary = validation_details(episodes)
    summary.update(
        {
            "checkpoint": str(Path(args.checkpoint)),
            "policy_version": int(checkpoint["iteration"]),
            "wall_seconds": elapsed,
            "matches_per_second": len(episodes) / elapsed,
            "decisions_per_second": sum(episode.transitions for episode in episodes) / elapsed,
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _python_tuple(value: Any) -> str:
    """Format nested float lists as compact, deterministic Python tuples."""
    if isinstance(value, list):
        items = ", ".join(_python_tuple(item) for item in value)
        return f"({items}{',' if len(value) == 1 else ''})"
    return f"{float(value):.9g}"


def exported_actor_constants(model: ActorCritic, checkpoint_name: str) -> str:
    """Return the actor-only parameters; the critic is intentionally omitted."""
    state = model.state_dict()
    tensors = (
        ("POLICY_W1", state["trunk.0.weight"]),
        ("POLICY_B1", state["trunk.0.bias"]),
        ("POLICY_W2", state["trunk.2.weight"]),
        ("POLICY_B2", state["trunk.2.bias"]),
        ("POLICY_W3", state["trunk.4.weight"]),
        ("POLICY_B3", state["trunk.4.bias"]),
        ("POLICY_W4", state["actor.weight"]),
        ("POLICY_B4", state["actor.bias"]),
    )
    lines = [
        "# BEGIN EXPORTED PPO ACTOR",
        f"POLICY_SOURCE = {checkpoint_name!r}",
        "POLICY_PARAMETER_COUNT = 5550",
    ]
    lines.extend(f"{name} = {_python_tuple(tensor.detach().cpu().tolist())}" for name, tensor in tensors)
    lines.append("# END EXPORTED PPO ACTOR")
    return "\n".join(lines)


def export_actor(args) -> None:
    """Replace only the bounded weight block in the dependency-free controller."""
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ActorCritic()
    model.load_state_dict(checkpoint["model_state"])
    block = exported_actor_constants(model, checkpoint_path.name)
    controller_path = Path(args.controller)
    source = controller_path.read_text(encoding="utf-8")
    start_marker = "# BEGIN EXPORTED PPO ACTOR"
    end_marker = "# END EXPORTED PPO ACTOR"
    if start_marker in source:
        start = source.index(start_marker)
        end = source.index(end_marker, start) + len(end_marker)
    else:
        start = source.index("POLICY_W1 =")
        final_bias = source.index("POLICY_B4 =", start)
        end = source.index("\n", final_bias)
    controller_path.write_text(source[:start] + block + source[end:], encoding="utf-8")
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "controller": str(controller_path),
                "actor_parameters": 5550,
                "critic_exported": False,
            },
            sort_keys=True,
        )
    )


def _fixed_actor(action: int) -> ActorCritic:
    model = ActorCritic(0.0)
    with torch.no_grad():
        model.actor.weight.zero_()
        model.actor.bias.zero_()
        model.actor.bias[action] = 1.0
    return model


def compare_policies(args) -> None:
    policy_models: dict[str, tuple[ActorCritic, int]] = {}
    if "ppo" in args.subjects:
        checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
        model = ActorCritic()
        model.load_state_dict(checkpoint["model_state"])
        policy_models["ppo"] = (model, int(checkpoint["iteration"]))
    if "fixed_opus" in args.subjects:
        policy_models["fixed_opus"] = (_fixed_actor(0), -1)
    if "fixed_scoring" in args.subjects:
        policy_models["fixed_scoring"] = (_fixed_actor(4), -1)

    all_results = {}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for subject in args.subjects:
            if subject in policy_models:
                model, version = policy_models[subject]
                jobs = _validation_jobs(
                    model,
                    version,
                    args.duration,
                    seeds=tuple(args.seeds),
                    opponents=tuple(args.opponents),
                )
                episodes, elapsed = collect_with_pool(pool, jobs)
            else:
                jobs = [
                    ReferenceJob(subject, seed, side, opponent, args.duration)
                    for opponent in args.opponents
                    for seed in args.seeds
                    for side in ("A", "B")
                ]
                batch_started = time.perf_counter()
                futures = [pool.submit(run_reference_job, job) for job in jobs]
                episodes = [future.result() for future in as_completed(futures)]
                elapsed = time.perf_counter() - batch_started
            summary = validation_details(episodes)
            summary["wall_seconds"] = elapsed
            summary["matches_per_second"] = len(episodes) / elapsed
            all_results[subject] = summary
            print(
                subject,
                summary["wins"],
                summary["draws"],
                summary["losses"],
                round(summary["match_points"], 1),
                round(summary["mean_score_difference"], 3),
                flush=True,
            )
    report = {
        "schema": "opus-rl-plan-ppo-comparison-v1",
        "checkpoint": args.checkpoint,
        "seeds": args.seeds,
        "opponents": args.opponents,
        "sides": ["A", "B"],
        "workers": args.workers,
        "wall_seconds": time.perf_counter() - started,
        "subjects": all_results,
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def benchmark_workers(args) -> None:
    torch.manual_seed(args.seed)
    model = ActorCritic()
    state = _frozen_state(model)
    levels = [value for value in args.levels if value <= max(1, os.cpu_count() or 1)]
    output = []
    for workers in levels:
        jobs = [
            RolloutJob(
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
        results, elapsed = collect_jobs(jobs, workers)
        decisions = sum(result.transitions for result in results)
        row = {
            "workers": workers,
            "matches": len(results),
            "duration": args.duration,
            "wall_seconds": elapsed,
            "matches_per_second": len(results) / elapsed,
            "decisions_per_second": decisions / elapsed,
            "controller_subprocesses": 0,
        }
        output.append(row)
        print(json.dumps(row, sort_keys=True))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def verify_direct(args) -> None:
    from swarmbench.match import run_match

    left = OPPONENTS[args.left]
    right = OPPONENTS[args.right]
    direct = run_direct_match(left, right, seed=args.seed, duration=args.duration)
    isolated = run_match(left, right, seed=args.seed, duration=args.duration)
    expected = (isolated.score_a, isolated.score_b)
    print({"direct": direct, "isolated": expected, "equal": direct == expected})
    if direct != expected:
        raise SystemExit("direct trusted-controller runner diverged")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--levels", nargs="+", type=int, default=[1, 2, 4, 6, 8, 12])
    benchmark.add_argument("--matches", type=int, default=12)
    benchmark.add_argument("--duration", type=float, default=30.0)
    benchmark.add_argument("--opponent", choices=OPPONENTS, default="opus")
    benchmark.add_argument("--seed", type=int, default=820_001)
    benchmark.add_argument("--output")
    benchmark.set_defaults(function=benchmark_workers)

    verify = subparsers.add_parser("verify-direct")
    verify.add_argument("--left", choices=OPPONENTS, default="opus")
    verify.add_argument("--right", choices=OPPONENTS, default="breaker")
    verify.add_argument("--seed", type=int, default=42)
    verify.add_argument("--duration", type=float, default=90.0)
    verify.set_defaults(function=verify_direct)

    train = subparsers.add_parser("train")
    train.add_argument("--run-name", required=True)
    train.add_argument("--seed", type=int, default=PPOConfig.seed)
    train.add_argument("--workers", type=int, default=PPOConfig.workers)
    train.add_argument("--episodes", type=int, default=PPOConfig.episodes_per_iteration)
    train.add_argument("--iterations", type=int, default=PPOConfig.iterations)
    train.add_argument("--duration", type=float, default=PPOConfig.duration)
    train.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    train.add_argument("--gae-lambda", type=float, default=PPOConfig.gae_lambda)
    train.add_argument("--clip", type=float, default=PPOConfig.clip_epsilon)
    train.add_argument("--epochs", type=int, default=PPOConfig.update_epochs)
    train.add_argument("--minibatch", type=int, default=PPOConfig.minibatch_size)
    train.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    train.add_argument("--entropy", type=float, default=PPOConfig.entropy_coefficient)
    train.add_argument("--value-coefficient", type=float, default=PPOConfig.value_coefficient)
    train.add_argument("--max-gradient-norm", type=float, default=PPOConfig.max_gradient_norm)
    train.add_argument("--target-kl", type=float, default=PPOConfig.target_kl)
    train.add_argument("--reward", choices=("terminal", "score_potential"), default=PPOConfig.reward_kind)
    train.add_argument("--validation-interval", type=int, default=PPOConfig.validation_interval)
    train.add_argument("--initial-opus-bias", type=float, default=PPOConfig.initial_opus_bias)
    train.add_argument("--switch-penalty", type=float, default=PPOConfig.switch_penalty)
    train.add_argument("--resume")
    train.set_defaults(function=train_ppo)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--workers", type=int, default=PPOConfig.workers)
    evaluate.add_argument("--duration", type=float, default=DEFAULT_MATCH_DURATION)
    evaluate.add_argument("--seeds", nargs="+", type=int, required=True)
    evaluate.add_argument("--opponents", nargs="+", choices=OPPONENTS, required=True)
    evaluate.add_argument("--output")
    evaluate.set_defaults(function=evaluate_checkpoint)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--checkpoint", required=True)
    compare.add_argument("--subjects", nargs="+", choices=("ppo", "fixed_opus", "fixed_scoring", "controller", "opus", "breaker"), required=True)
    compare.add_argument("--workers", type=int, default=PPOConfig.workers)
    compare.add_argument("--duration", type=float, default=DEFAULT_MATCH_DURATION)
    compare.add_argument("--seeds", nargs="+", type=int, required=True)
    compare.add_argument("--opponents", nargs="+", choices=OPPONENTS, required=True)
    compare.add_argument("--output")
    compare.set_defaults(function=compare_policies)

    export = subparsers.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--controller", default=str(SUBJECT_PATH))
    export.set_defaults(function=export_actor)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
