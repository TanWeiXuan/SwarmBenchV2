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

## Experiment 2 recommended next experiment

Before expanding the action space or learning targets, evaluate mode 4 against
PPO on a much larger direct Opus/SIPP-heavy schedule and inspect states where
the actor chooses mode 5. If those decisions still never change outcomes, ship
the simpler constant allocation or use state-restorable short-horizon
counterfactuals to label only high-leverage decision states. Pairwise target
scoring, recurrence, and end-to-end control remain unjustified.

## Experiment 3: dynamic scout roles and learned targets

Experiment 3 replaces the six-profile action with effective per-scout duties.
It preserves the shipped controller and all Experiment 1/2 evidence; the
research implementation is `experiments/opus_rl_plan_dynamic.py`. No
Experiment 3 actor was exported into the submission because no checkpoint met
the acceptance criteria below.

### Representation and action

The v2 actor-critic has 28,432 parameters. It uses:

- the original 32 global features plus maximum friendly/enemy transport
  progress, nearest enemy-scout and loaded-tank distance to a friendly
  transport, and normalized guard/block opportunity counts (38 total);
- a shared `14 -> 24 -> 16` live-drone encoder;
- separate friendly/enemy mean and count-normalized-sum pooling, with no
  padding;
- a `102 -> 64 -> 48 -> 48 -> 48` global context encoder (the last two
  48-wide layers are the larger-model experiment);
- a shared per-scout role head receiving context, the scout embedding,
  previous role, role duration, and current assignment counts;
- a shared pairwise target head receiving context, scout embedding, target
  entity features, relative geometry/velocity, distance/intercept proxy,
  route/goal progress, value, ammunition, and ward geometry;
- a centralized critic used only during training.

The six roles are RUN, HUNT_TRANSPORT, HUNT_TANK, GUARD_TRANSPORT, KEEP, and
BLOCK. Roles and targets are chosen autoregressively in sorted live-scout
order. Occupied targets are removed from later masks. Impossible roles are
masked. Raw IDs are used only as stable ordering/re-resolution keys and never
as neural features. The whole assignment log probability is the sum of its
sampled role and learned-target factors.

Only active drones enter the set encoder, so random initial counts and deaths
do not change tensor dimensions. Empty team sets pool to zero. The actor loops
over the current live scout tuple; a dead/scored scout simply disappears at
the next decision. Survival features continue to use the arena's sampled
initial counts. If a selected target dies, the stable key cannot resolve and
the deterministic skill safely falls back to RUN until reassignment. A valid
zero-scout behavior-clone minibatch has no actor factor and is skipped rather
than treated as an error.

The actor runs every 1.0 simulated second. Low-level Opus path planning,
interception, obstacle/collision/projectile avoidance, jerk limiting, tank
movement, and gunnery remain deterministic.

### PPO and league

PPO uses terminal win/draw/loss reward, gamma 0.995, GAE lambda 0.95, clip 0.2,
four epochs, minibatch 64, Adam 1e-4, entropy coefficient 0.01, value
coefficient 0.5, gradient cap 0.5, and target KL 0.02. Each update contains 18
complete matches from a frozen actor snapshot. Checkpoints include optimizer,
RNG, opponent-league, and best-validation state and support resume.

The adaptive league discovers every current submission plus built-ins. It
retains 25% uniform sampling and weights the rest by learner losses, parity,
rating, and a hard-field bonus. After early runs wasted matches on solved
opponents, the initial hard-field mass was raised to 45.1%, split approximately
equally across Opus, Breaker, GPT-5.3-Codex, Gemini 3.1 Pro, Sonnet 5 V3,
SIPP/Marksman, and fixed mode 4. This change followed measured matchup results;
no opponent identity is an actor input.

### Effective-action instrumentation and ablations

Every diagnostic rollout records selected roles/targets, resolved duties,
duty and target changes, role duration, commands differing from a simultaneous
fixed-mode-4 shadow, score timelines, simulator events, surviving value, and
tank ammunition. Exact recorded-action replay is tested.

The implemented evaluation ablations are:

- full dynamic roles and learned targets;
- freeze the first assignments for the match;
- explicit all-RUN (the learned most-common static action);
- dynamic learned roles with the old deterministic intercept-time target
  ranking;
- direct fixed mode 4 as an opponent/baseline.

### Throughput

The v2 12-match, 30-second benchmark retained six workers:

| Workers | Matches/s | Decisions/s | Mean inference |
| ---: | ---: | ---: | ---: |
| 1 | 0.766 | 22.98 | 2.15 ms |
| 2 | 0.837 | 25.11 | 3.22 ms |
| 4 | 1.293 | 38.80 | 4.35 ms |
| 6 | **1.404** | **42.12** | 4.95 ms |
| 8 | 1.355 | 40.66 | 4.14 ms |
| 12 | 1.313 | 39.38 | 5.80 ms |

Optimization was about 3-6% of wall time in representative runs. Greedy
evaluation of the trained checkpoint averaged about 2.1-2.3 ms per tactical
decision, with no invalid actions or controller subprocesses.

### Initialization and failed variants

Fixed mode 4 produces highly imbalanced labels. Twelve-match clones contained
only about 3% GUARD labels. Unweighted, weight-5, weight-10, weight-14, and
weight-17 clones selected RUN for every scout. Weight 20/30 crossed an abrupt
threshold, guarded 19% of decisions, changed duties about 21-31 times per
match, and lost badly. Class weighting alone was therefore not a useful
12-match solution.

The retained bootstrap uses 60 teacher matches, eight epochs, GUARD weight 5,
and independent seeds. It learned conditional guards instead of a global
prior: representative training confusion was 220/384 true guards with 112
false guards among 10,502 RUN labels. The three independent clones had
94.5-95.2% exact full-assignment accuracy.

Three terminal-PPO seeds from the larger clones were screened on ten fresh
seeds, both sides, against Opus, Breaker, and fixed mode 4:

| PPO seed | Points / 60 | Guard frequency | Duty changes/match |
| ---: | ---: | ---: | ---: |
| 1 | 31.5 | 4.23% | 5.45 |
| 2 | 29.5 | 1.34% | 1.57 |
| 3 | 29.0 | 2.71% | 4.02 |

All deterministic actors remained RUN/GUARD-only. A training-only temperature
of 1.5 roughly doubled entropy and sampled HUNT/KEEP/BLOCK about 2.4% of the
time, but terminal validation stayed at 20/28. Bounded score-potential shaping
with that temperature reached 27/28 on the small internal validation yet only
29/60 on the fresh screen and reverted to 98.6% RUN. Both were rejected.

The missing target initialization was then corrected. Behavior cloning kept
the fixed-mode role teacher but added a target-only auxiliary loss that teaches
the pairwise head the proven deterministic intercept ranking for every
multi-candidate HUNT_TRANSPORT, HUNT_TANK, GUARD, and BLOCK set. It does not
supply role labels, although its gradients also adapt the shared entity/context
features. Runs performed 527-560 such minibatch updates.

Target-pretrained PPO results on the same fresh screen were:

| Seed | All | vs Opus | vs Breaker | vs fixed mode 4 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 35.5/60 | 12.0/20 | 13.0/20 | 10.5/20 |
| 2 | 30.0/60 | 10.0/20 | 11.0/20 | 9.0/20 |
| 3 | 32.0/60 | 11.0/20 | 13.5/20 | 7.5/20 |
| Combined | 97.5/180 | 33.0/60 | 37.5/60 | 27.0/60 |

The method therefore replicated an Opus gain on this small screen (55% across
seeds), but not a fixed-mode-4 gain. Seed 1 was the only checkpoint to beat all
three. Its same-seed ablations were full 35.5/60, old targets 34.0/60, and
freeze/all-RUN 29.0/60. Dynamic timing added 6.5 points and learned targeting
added 1.5 points on that schedule, establishing consequential actions rather
than nominal switching.

### Larger evidence gate and conclusion

The best seed-1 checkpoint was then tested on 50 new seeds, both sides (100
games per opponent, 300 total):

| Opponent | W-D-L | Points | Point rate | Paired-seed 95% CI | Mean diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| Opus | 37-10-53 | 42.0 | 42.0% | 29.3-54.7% | -0.57 |
| Opus Breaker | 59-12-29 | 65.0 | 65.0% | 53.1-76.9% | +1.62 |
| fixed mode 4 | 32-47-21 | 55.5 | 55.5% | 49.0-62.0% | +0.65 |
| Total | 128-69-103 | 162.5/300 | 54.2% | - | +0.57 |

Side results were balanced (61-35-54 as A and 67-34-49 as B). The actor used
RUN 98.05% and GUARD 1.95%, with 2.56 duty/target changes and 6.76 scout-command
differences from mode 4 per match. Among the 63 Opus losses/draws, there were
only three obstacle crashes; the dominant events were ordinary vehicle
collisions and projectile trades, not invalid actions or path-planning failure.

On the identical 100 Opus games, freeze/all-RUN scored 45.0 points, old targets
43.5, and the full policy 42.0. Both learned guard timing and target choice hurt
the Opus matchup on the larger schedule. The earlier 12-8 screen was sampling
noise.

Experiment 3 therefore does **not** satisfy the controller acceptance criteria.
It proves that the variable-set/factorized implementation can learn and execute
state-dependent assignments, and one checkpoint beats Breaker and fixed mode 4
with useful dynamic/target ablations. It does not convincingly beat Opus, and
the best result does not replicate against fixed mode 4 across training seeds.
No 250-seed claim, pure-Python export, or submission replacement was made.

### Experiment 3b: state-restorable guard counterfactuals

The proposed counterfactual experiment was implemented rather than left as a
suggestion. At a feasible guard state, the authoritative simulator and both
stateful controllers are deep-copied before commands execute. Branches force
one scout to RUN or one of the two best feasible GUARD targets for five
seconds; all other learned decisions continue normally, the override expires,
and every branch runs to the real match end. Selection is lexicographic on
win/draw/loss and then score differential. Temporary assignments expire when
the scout/target dies and reserve their target exactly like normal
autoregressive assignment.

The first dataset branched the first guard opportunity on ten seeds, both
sides, against Opus, Breaker, and fixed mode 4. All 60 jobs found a state; 40
had a unique terminal preference, split 31 GUARD / 9 RUN, with mean score
spread 6.82 and maximum 42. A second batch sampled feasible opportunity
indices 0-5. It found 47/60 states at mean time 9.26 seconds; 16 were decisive
and exactly balanced (8/8), with 31 ties. Artifacts contain plain nested data,
not process-local dataclass pickle identities.

Fine-tuning ranks only the alternative full assignments, so unchanged scout
factors cancel. It uses a seed-modulus holdout and per-epoch early stopping.
Unconstrained fitting demonstrated that the labels are learnable (ranking
accuracy rose from 36.4% to 66.7% train and 14.3% to 57.1% holdout), but it
changed shared features globally, activated KEEP/HUNT, guarded 19-21%, and
collapsed to 12.5-16.5/60. Restricting gradients to the final RUN/GUARD and
target output rows removed unrelated roles but still over-guarded.

Because the sampled data is conditioned on guard feasibility, its 31:9 class
ratio cannot safely become a global role prior. Inverse-frequency weighting
kept the actor near its source behavior (2.81% GUARD). On the first paired
screen it improved from 28.0/60 to 30.5/60. That gain did not generalize:

| Policy on 30 new seeds × both sides | All | Opus | Breaker | fixed mode 4 |
| --- | ---: | ---: | ---: | ---: |
| Counterfactual fine-tune | 78.5/180 | 26.0/60 | 27.0/60 | 25.5/60 |
| Untouched source | 80.5/180 | 25.0/60 | 29.0/60 | 26.5/60 |

Adding the later-state batch produced 56 decisive labels (39 GUARD / 17 RUN),
but its combined fit regressed to 25.5/60 versus the same source's 28.0/60 on
the smaller paired screen. Counterfactual branching therefore found genuinely
high-leverage states, but direct small-sample policy ranking did not generalize
and is rejected.

More PPO epochs, higher entropy, score shaping, recurrence, Transformers, and
end-to-end control remain unjustified. A future continuation should require a
substantially larger counterfactual dataset and train a calibrated action-value
model with an uncertainty/fallback gate, rather than directly shifting the
deployed role logits from a few dozen labels.
