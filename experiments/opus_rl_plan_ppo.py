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

    def __init__(self) -> None:
        super().__init__()
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
        # Start from deterministic Opus under argmax while stochastic training
        # still samples every safe mode (p(Opus) is approximately 0.60).
        with torch.no_grad():
            self.actor.bias[0] = 2.0
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


def _episode_rewards(records: list[dict[str, Any]], final_difference: int, outcome: float, kind: str, maximum_score: int) -> list[float]:
    if kind not in {"terminal", "score_potential"}:
        raise ValueError(f"unknown reward kind: {kind}")
    rewards = [0.0] * len(records)
    if not rewards:
        return rewards
    if kind == "score_potential":
        differences = [int(record["score_difference"]) for record in records] + [final_difference]
        for index in range(len(records)):
            rewards[index] += 0.10 * (differences[index + 1] - differences[index]) / max(1, maximum_score)
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


def collect_jobs(jobs: list[RolloutJob], workers: int) -> tuple[list[EpisodeResult], float]:
    started = time.perf_counter()
    results: list[EpisodeResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_rollout_job, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return results, time.perf_counter() - started


def _frozen_state(model: ActorCritic) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
