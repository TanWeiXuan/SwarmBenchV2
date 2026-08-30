# SwarmBench

SwarmBench is a deterministic, open-source benchmark and Kaggle-style competition for multi-agent swarm control and combat. Two Python controllers command mirrored mixed-class teams through a randomized 100 m × 60 m arena. The authoritative simulator is headless; replays and rendering are separate.

![Sol 5.6 RL versus Sonnet 5 V3 match replay](match.gif)

Each team has 10–14 SCOUT vehicles, 4–8 TRANSPORT vehicles, and 2–4 TANK vehicles. A vehicle scores by reaching the opposite goal. All vehicle contacts—including friendly contacts—destroy both participants within 0.75 m. Tanks fire visible, non-piercing 20 m/s projectiles that obstacles or either team's vehicles can intercept. Each Tank carries five shots, cannot fire during the first five seconds, and can fire at most once every four seconds thereafter. The higher score after 90 simulation seconds wins.

| Type | Count | Nominal speed | Nominal acceleration | Nominal jerk | Value |
| --- | ---: | ---: | ---: | ---: | ---: |
| SCOUT | 10–14 | 5.0 m/s | 4.0 m/s² | 16.0 m/s³ | 1 |
| TRANSPORT | 4–8 | 2.5 m/s | 2.0 m/s² | 8.0 m/s³ | 5 |
| TANK | 2–4 | 1.5 m/s | 1.2 m/s² | 4.8 m/s³ | 1 |

Speed, acceleration, and jerk are sampled independently within ±20% of the nominal values once per game and shared by both sides. Counts, dynamics, goals, obstacles, and spawns are reproducible from the one game seed.

## Quick start

SwarmBench targets Python 3.12.

```bash
python -m pip install -e ".[dev,competition,render]"
python -m pytest
python -m swarmbench arena --seed 42 --render arena.png
python -m swarmbench match --controller-a marksman --controller-b convoy --seed 42 --replay match.json --render match.mp4
python -m swarmbench render match.json --output match.gif --killfeed
```

Rendering defaults to 10 FPS at 640×384. Use `--render-fps 20 --render-quality high` when fidelity matters more than rendering speed. `--killfeed` adds a compact translucent icon-only feed at the bottom centre with the latest five eliminations, collisions, crashes, and scores for five seconds; adjust it with `--killfeed-lines` and `--killfeed-duration`. Header counts use `S`, `Tr`, and `Tk` for Scouts, Transports, and Tanks. MP4 output uses ffmpeg when available and otherwise falls back to GIF.

Built-in controller names are `rush`, `defend`, `greedy_value`, `assignment`, `potential_field`, `marksman`, and `convoy`.

## Write a controller

A controller is one class with two methods:

```python
from swarmbench import BaseSwarmController, DroneStatus, DroneType


class SwarmController(BaseSwarmController):
    def initialize(self, game_info):
        self.goal = game_info.target_goal

    def step(self, state):
        direction = 1.0 if self.goal.center[0] > 50.0 else -1.0
        target = next((d for d in state.opponent_drones if d.status is DroneStatus.ACTIVE), None)
        actions = {}
        for vehicle in state.own_drones:
            if vehicle.status is not DroneStatus.ACTIVE:
                continue
            command = {"acceleration": (4.0 * direction, 0.0)}
            if vehicle.drone_type is DroneType.TANK and target is not None:
                command["fire_direction"] = (
                    target.position[0] - vehicle.position[0],
                    target.position[1] - vehicle.position[1],
                )
            actions[vehicle.id] = command
        return actions
```

The object persists for one match, so assignments, caches, recurrent state, and loaded models may live on `self`. `step()` receives immutable perfect-information vehicle and projectile snapshots. Movement-only `(ax, ay)` remains valid; structured commands add transient Tank fire requests. See the [Controller API](docs/CONTROLLER_API.md) for validation, command retention, and deadlines.

## Generate a controller with a coding agent

Replace `<OUTPUT_PATH>` with `submissions/<github-login>/<controller-name>.py` before using this prompt. Identify the coding agent or model in the controller name or a top-of-file comment.

```text
You are competing in SwarmBench. Inspect the repository to understand the game
rules, controller API, physics, scoring, projectiles, obstacles, execution
limits, validation tools, and built-in baselines. Implement the strongest valid
and robust controller you can as one Python file at `<OUTPUT_PATH>`.

You may validate it, play matches, inspect results, and iterate, but do not
modify any other repository file, exploit bugs, or violate documented limits.
Optimize for unseen opponents and hidden deterministic seeds rather than known
controllers or test cases. Clearly attribute non-trivial strategy or code drawn
from another controller.
```

## Submit a controller

Open a PR containing exactly one file at `submissions/<github-login>/<controller-name>.py`. Validation checks structure, imports/API, an empty-arena run, timing, and deterministic side-swapped calibration. A trusted reporter maintains a sticky progress comment and enables squash auto-merge after the required `Submission Gate` passes. See the [submission guide](docs/SUBMISSION_GUIDE.md).

## Community leaderboard

Only the latest rating state is committed; permanent tournament history lives in [Tournament Results Discussions](https://github.com/TanWeiXuan/SwarmBenchV2/discussions/categories/tournament-results).

<!-- LEADERBOARD_START -->
| Rank | Controller | Author | Rating | RD | W | D | L | Games |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Opus 5 V3 | renj1ete0 | 2394 | 30 | 588 | 22 | 30 | 640 |
| 2 | Codex 5 6 Crossfire V1 | TanWeiXuan | 2253 | 26 | 1618 | 66 | 236 | 1920 |
| 3 | Opus 5 V4 | renj1ete0 | 2203 | 29 | 124 | 3 | 1 | 128 |
| 4 | Opus 5 V2 | renj1ete0 | 2185 | 26 | 1750 | 72 | 362 | 2184 |
| 5 | Sonnet 5 V3 | renj1ete0 | 1976 | 21 | 1357 | 211 | 1080 | 2648 |
| 6 | Sol 5 6 Rl | TanWeiXuan | 1970 | 22 | 1345 | 140 | 1091 | 2576 |
| 7 | Gpt-5.3-Codex | renj1ete0 | 1965 | 21 | 1358 | 277 | 1045 | 2680 |
| 8 | Gemini 3 1 Pro V1 | renj1ete0 | 1958 | 22 | 1367 | 158 | 1099 | 2624 |
| 9 | Opus 5 V1 | renj1ete0 | 1946 | 22 | 1670 | 350 | 1040 | 3060 |
| 10 | Luna Xhigh Opus Breaker | TanWeiXuan | 1863 | 22 | 1390 | 232 | 1218 | 2840 |
<!-- LEADERBOARD_END -->

## Reproducibility and security

Scenario identity is `(generator_version, seed)`. Match replays record the sampled scenario dynamics, controller hashes, accepted movement changes, actual shots, events, and final result. SwarmBench seeds Python, NumPy, and PyTorch CPU RNGs where applicable; arbitrary native third-party behavior cannot always be bit-for-bit portable, but the engine is deterministic.

Running third-party controllers locally executes untrusted Python. Official CI uses persistent Docker workers with no network, a read-only filesystem, bounded scratch space, CPU/memory/process limits, and no write credentials or secrets. Static source checks are usability checks, not a security boundary. Read [SECURITY.md](docs/SECURITY.md) before running community code locally.

Detailed rules are in [GAME_SPEC.md](docs/GAME_SPEC.md); tournament design and manual commands are in [TOURNAMENTS.md](docs/TOURNAMENTS.md). SwarmBench is available under the [MIT License](LICENSE).
