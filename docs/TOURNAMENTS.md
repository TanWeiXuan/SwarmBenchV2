# Tournaments and ratings

Official rating periods run at minute 17 every six hours. Manual runs use the same workflow and may be `official` or non-rating `exhibition`, with `small`, `default`, or `large` presets.

```bash
gh workflow run tournament.yml -f mode=exhibition -f size=small -f seed=demo-2026
gh run list --workflow tournament.yml
```

The scheduler targets about eight distinct opponents per controller at default size: roughly three quarters are nearby in current rating and one quarter explores more broadly. Unordered pairings are deduplicated. Default pairings use four deterministic scenarios and swap sides for each, yielding eight games.

SwarmBench implements Glicko-2 with new-player defaults 1500 rating, 350 RD, and 0.06 volatility. Individual games provide win/draw/loss observations. Every participant, including baseline controllers, is updated simultaneously from pre-period ratings; score differential does not affect Glicko.

## Five-stage trust boundary

One long-lived trusted reporter creates the permanent Tournament Results Discussion while five sequential Docker compute jobs execute controllers with `contents: read`, no secrets/write permissions, no network, and a read-only filesystem. As each artifact arrives, the reporter validates its primitive JSON and adds one provisional progress comment at approximately 20%, 40%, 60%, 80%, and 100%. GitHub Actions installation tokens can create Discussions and comments but cannot edit a Discussion opener, so the immutable opener records the initial RUNNING state and the latest bot reply is authoritative for progress, failure, or the final COMPLETE report.

Only the final trusted job can update current ratings. It requires all expected batch IDs exactly once, matching engine/format versions, pairings, sides, seeds, scores, and result ranges. A missing/duplicate/tampered batch aborts the entire period; ratings remain unchanged and the Discussion is marked failed. Exhibition runs always leave ratings/README untouched.

Successful official runs open a small bot PR changing only `leaderboard/ratings.json` and the README top ten. Permanent history and detailed pair results remain in Discussions, not Git. Raw results and a small set of compressed representative/closest replay candidates are workflow artifacts.

Local tournament CLI:

```bash
python -m swarmbench tournament --mode exhibition --size small --seed 42
```
