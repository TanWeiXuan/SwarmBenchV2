from __future__ import annotations

import argparse
import time
from pathlib import Path

from swarmbench.controllers.baselines import BASELINE_NAMES, baseline_path
from swarmbench.engine import generate_scenario
from swarmbench.match import run_match
from swarmbench.replay import load_replay, save_replay
from swarmbench.replay.renderer import render_arena, render_replay


def _controller_path(value: str) -> Path:
    return baseline_path(value) if value in BASELINE_NAMES else Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swarmbench", description="Deterministic swarm-control benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    match = subparsers.add_parser("match", help="run one isolated match")
    match.add_argument("--controller-a", required=True)
    match.add_argument("--controller-b", required=True)
    match.add_argument("--seed", type=int, required=True)
    match.add_argument("--duration", type=float, default=90.0)
    match.add_argument("--replay", type=Path)
    match.add_argument("--render", type=Path)
    match.add_argument("--render-fps", type=int, choices=(5, 10, 20), default=10)
    match.add_argument("--render-quality", choices=("low", "high"), default="low")

    render = subparsers.add_parser("render", help="render an existing replay")
    render.add_argument("replay", type=Path)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--render-fps", type=int, choices=(5, 10, 20), default=10)
    render.add_argument("--render-quality", choices=("low", "high"), default="low")

    arena = subparsers.add_parser("arena", help="generate and optionally render an arena")
    arena.add_argument("--seed", type=int, required=True)
    arena.add_argument("--render", type=Path)

    validate = subparsers.add_parser("validate", help="validate one submission")
    validate.add_argument("controller", type=Path)

    tournament = subparsers.add_parser("tournament", help="run a local tournament")
    tournament.add_argument("--seed", type=int, required=True)
    tournament.add_argument("--size", choices=("small", "default", "large"), default="small")
    tournament.add_argument("--mode", choices=("official", "exhibition"), default="exhibition")

    benchmark = subparsers.add_parser("benchmark", help="time a short headless isolated match")
    benchmark.add_argument("--duration", type=float, default=10.0)
    benchmark.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "match":
        result = run_match(_controller_path(args.controller_a), _controller_path(args.controller_b), seed=args.seed, duration=args.duration)
        print(f"A {result.score_a} - {result.score_b} B | winner: {result.winner.value if result.winner else 'DRAW'}")
        if args.replay:
            save_replay(result.replay, args.replay)
        if args.render:
            actual = render_replay(result.replay, args.render, fps=args.render_fps, quality=args.render_quality)
            print(f"rendered {actual}")
        return 0
    if args.command == "render":
        print(f"Loading replay from {args.replay}...", flush=True)
        actual = render_replay(load_replay(args.replay), args.output, fps=args.render_fps, quality=args.render_quality)
        print(f"rendered {actual}")
        return 0
    if args.command == "arena":
        scenario = generate_scenario(args.seed)
        print(f"seed={scenario.seed} obstacles={len(scenario.obstacles)} drones={len(scenario.drones)}")
        if args.render:
            print(f"rendered {render_arena(scenario, args.render)}")
        return 0
    if args.command == "validate":
        from swarmbench.competition.submission import validate_controller_cli

        return validate_controller_cli(args.controller)
    if args.command == "tournament":
        from swarmbench.competition.tournament import tournament_cli

        return tournament_cli(seed=args.seed, size=args.size, mode=args.mode)
    if args.command == "benchmark":
        started = time.perf_counter()
        result = run_match(baseline_path("rush"), baseline_path("rush"), seed=args.seed, duration=args.duration)
        elapsed = time.perf_counter() - started
        print(f"simulated {result.replay.final_time:.2f}s in {elapsed:.3f}s ({result.replay.final_time / elapsed:.1f}x real-time)")
        return 0
    return 2
