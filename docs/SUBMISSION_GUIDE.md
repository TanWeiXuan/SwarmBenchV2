# Submission guide

## File contract

A normal controller PR changes exactly one regular, non-symlink Python file:

```text
submissions/<your-github-login>/<controller-name>.py
```

The 5 MiB maximum includes embedded learned weights. External assets, configuration, downloads, and installation are unavailable. The file must define `SwarmController(BaseSwarmController)`. Allowed third-party imports are NumPy 2.2.6, SciPy 1.15.3, PyTorch CPU 2.12.0, and NetworkX 3.4.2; standard-library modules are available subject to sandbox restrictions. Exact pins live in `requirements-controller.txt`.

## Automated path

The required workflow performs:

1. PR shape/path/size and source/API/import validation.
2. One obstacle-free 90 s smoke test with no opposing team. A nonzero score is required; defenders need not score.
3. Four deterministic scenario seeds against all seven baseline anchors and up to eight existing community controllers, with a side swap for every opponent/seed. Small community pools are used in full; larger pools use a deterministic mix of nearby, low-rated, and high-rated opponents.
4. Strict calibration artifact validation and a provisional Glicko-2 rating.
5. The required `Submission Gate` check.

The sticky PR comment reports job/calibration-batch progress. During calibration, every opponent keeps its existing rating; only the submitted controller receives the provisional rating. Successful validation enables squash auto-merge. After merge, a trusted job downloads the SHA/path-bound calibration artifact, validates only primitive JSON, and opens a current-rating bot PR. No privileged job checks out a PR head or imports controller code.

## Timing and state

The soft/hard step deadlines are 500 ms and 5 s. A soft miss discards that update but preserves state changes; hard timeout or exception forfeits. Design for ample margin. Initialization has 10 s. Controllers persist within a match and reset between matches.

## Local checks

```bash
python -m swarmbench validate submissions/you/controller.py
python -m swarmbench match --controller-a submissions/you/controller.py --controller-b rush --seed 42 --replay local.json
```

These commands use the simpler local subprocess backend. Local execution is not a security sandbox; run only code you trust.
