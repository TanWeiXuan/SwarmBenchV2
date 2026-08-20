# Opus_RL_Plan experiments

This is the continuation record for the learned high-level planner. The
competition controller retains Opus/Opus Breaker's deterministic path planning,
safety, movement, interception, collision/projectile avoidance, assignment, and
gunnery. Learning selects only one of six safe scout-allocation profiles every
two simulated seconds.

## Observation, actions, and variable swarm size

The base observation has 32 fixed, physically scaled global features: time and
score; live friendly/enemy counts by type; transport/scout progress; transport
time-to-goal; forward speed; ammunition and loaded-tank fractions; goal
pressure; threatened/pursued transport fractions; projectiles; and survival
fractions by type. Experiment 2 appends a six-way one-hot encoding of the
current tactical mode, for 38 inputs total.

There is no fixed-drone assumption. Per-drone data is pooled with counts,
means, minima, and fractions. Initial counts are read from each generated arena
and survival is normalized against those counts. Each control step filters both
teams to `DroneStatus.ACTIVE`, so a death, collision, or scored transport is
absent from the next observation. The deterministic planner then assigns the
selected profile only to the current live scout list; caps larger than the
available list naturally stop when it is empty. Zero-count types use explicit
safe defaults, so arenas with different initial counts and late-game empty
groups do not change tensor shape or cause division by zero.

The six actions are tuples of `transport_hunters`, `gun_hunters`, `keepers`,
`guard_cap`, and `block_cap`. They do not identify individual drones or issue
movement commands. Existing deterministic scoring and conflict resolution pick
the actual scouts and targets.

## Experiment 1: retrospective profile classification

Experiment 1 is commit `3531afe`. For each whole match, all six constant
profiles were evaluated and the best whole-match profile supplied labels for a
supervised classifier:

`whole-match fixed profile -> retrospective label -> classifier`

The retained 32-48-48-24-6 model used 917 sampled states. On 96 held-out games
it scored 84-0-12/+14.07 versus Opus at 81-4-11/+15.96 and Breaker at
68-11-17/+10.08. It went only 2-0-10 directly against Opus. An expanded 2,020
sample checkpoint had 96.2% training accuracy but regressed to 25-1-6/+8.75 on
its 32-game validation and was rejected. The method was not sequential RL and
its whole-trajectory labels were noisy at individual decision states.

## Experiment 2: sequential PPO

The actor-critic is `38 -> 48 -> 48 -> 24`, with Tanh activations, a six-logit
actor, and scalar critic. It has 5,575 parameters during training; deployment
exports only the 5,550 actor parameters. PyTorch is not imported by the
controller.

At every decision the rollout stores observation, sampled categorical action,
old log probability, critic value, terminal-aware reward, done flag, policy
version, time, score, and previous mode. Complete match trajectories use GAE
with normalized advantages and clipped PPO updates:

- terminal reward: win +1, draw 0, loss -1;
- gamma 0.995, GAE lambda 0.95, clip epsilon 0.2;
- Adam learning rate 2.5e-4, six epochs, minibatch 256;
- value coefficient 0.5, gradient norm cap 0.5, target KL 0.03;
- entropy coefficient 0.005 in the retained uniform-initialized run;
- 24 full 90-second matches per update, six process workers;
- deterministic validation uses actor argmax.

Training sampled both sides and fresh seeds from 1,000,000-8,999,999. Opus and
Breaker each have weight four; SIPP, MPC, Wayfinder, Potential Field, and
Greedy each have weight two; BigPickle, Aegis, Assignment, Rush, Defend,
Convoy, and Marksman each have weight one. Validation seeds and all later
comparison seeds were excluded from updates. Workers run the authoritative
simulator directly with frozen policy snapshots and compute both controllers'
commands from the same pre-update state.

## Throughput

A 12-match, 30-second benchmark measured the following on the development
machine. Six workers were retained because they had the best observed rate.

| Workers | Matches/s | Decisions/s |
| ---: | ---: | ---: |
| 1 | 0.567 | 8.50 |
| 2 | 0.779 | 11.69 |
| 4 | 1.097 | 16.46 |
| 6 | 1.266 | 18.98 |
| 8 | 1.232 | 18.48 |
| 12 | 1.067 | 16.01 |

There are no nested controller subprocesses in trusted local training workers,
and Torch/OMP/MKL/OpenBLAS threads are capped at one. Policy serialization is
once per rollout batch. Metrics record collection and optimization wall time,
throughput, inference time, checkpoint time, entropy, KL, clip fraction, losses,
gradient norm, explained variance, action frequencies, and transition counts.

## Experiments and failures

Terminal-only and bounded score-potential rewards produced the same 27-4-1,
29-point, +17.06 validation result after five 18-match updates. Shaping was
therefore rejected as unnecessary.

An Opus-biased initialization ran 15 updates/270 matches. Critic explained
variance rose from 0.16 to 0.65, but deterministic action 0 remained at 100%
and validation stayed 27-4-1. The bias protected the baseline too strongly and
prevented useful policy departure.

Uniform initialization with seed 82002 and terminal reward produced:

| Update | Validation W-D-L | Points | Mean diff | Deterministic modes |
| ---: | ---: | ---: | ---: | --- |
| 5 | 30-0-2 | 30 | +17.03 | mode 4: 100% |
| 10 | 30-0-2 | 30 | +17.03 | mode 4: 89.3%, mode 5: 10.7% |
| 15 | 28-0-4 | 28 | +15.66 | mode 5: 98.0% |

The best state was replayed from its resumable checkpoint through update 10;
`uniform_terminal_replay/validation-0010.pt` is the retained actor source. A
second initialization seed was unstable: update 5 chose mode 2 for 89% of
decisions and scored 24-0-8/+8.78, then recovered to fixed Opus at update 10.
This establishes material seed sensitivity.

Observed stochastic training trajectories switched roughly 30 times per match.
A 0.002 switching penalty was tried after this evidence appeared. It did not
change held-out outcomes or eliminate training-time oscillation and was
rejected. Deterministic evaluation was much more stable (roughly 1-2 changes).

Atomic `latest.pt`, `best.pt`, and validation checkpoints contain actor-critic,
optimizer, iteration, match/decision totals, configuration, Python/Torch update
RNG state, and best-validation metadata. Resume was exercised by the retained
replay. Checkpoints and raw metrics remain under ignored `.rl_local/` paths.

## Held-out results

The final schedule used six unseen seeds (930001, 930007, 930011, 930019,
930037, 930049), twelve opponents, and both sides: 144 games per subject.

| Subject | W-D-L | Points | Mean diff |
| --- | ---: | ---: | ---: |
| Experiment 2 PPO | 136-2-6 | 137.0 | +17.826 |
| Fixed aggressive scoring (mode 4) | 136-2-6 | 137.0 | +17.826 |
| Original Opus | 129-10-5 | 134.0 | +17.243 |
| Experiment 1 | 121-7-16 | 124.5 | +13.514 |
| Opus Breaker | 118-12-14 | 124.0 | +10.924 |

PPO was balanced by side (67-2-3 as A, 69-0-3 as B), went 8-1-3 directly
against Opus and 10-0-2 against Breaker, and selected mode 4 for 92.0% and mode
5 for 8.0% of decisions with 1.85 changes per match. It beat Opus by three
match points and +0.58 mean differential, but had one more loss.

Most importantly, fixed mode 4 produced identical per-opponent and aggregate
scores in every final match. The clean answer is therefore mixed: PPO found a
stronger allocation than fixed Opus and learned genuine within-match changes,
but there is no evidence in this schedule that its state-dependent switching
caused any improvement. A prior 32-game screening also tied fixed mode 4
exactly. Do not claim the tactical switching hypothesis is validated.

The pure-Python export matched PyTorch logits to under 9e-9 on a numerical
check and exactly matched its checkpoint's per-opponent outcomes on a separate
16-game unseen equivalence schedule. No critic or training runtime is shipped.

## Recommended next experiment

Before expanding the action space or learning targets, evaluate mode 4 against
PPO on a much larger direct Opus/SIPP-heavy schedule and inspect states where
the actor chooses mode 5. If those decisions still never change outcomes, ship
the simpler constant allocation or use state-restorable short-horizon
counterfactuals to label only high-leverage decision states. Pairwise target
scoring, recurrence, and end-to-end control remain unjustified.
