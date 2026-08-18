# Built-in controllers

SwarmBench includes seven deterministic reference controllers using the same API as submissions:

- **Rush** sends every class toward the goal and takes clear Tank shots.
- **Defend** advances Transports while Scouts intercept threats and Tanks provide defensive fire.
- **Greedy value** assigns Scouts to valuable nearby targets and uses value-aware Tank fire.
- **Assignment** periodically solves a global Scout-to-threat assignment with SciPy.
- **Potential field** combines goal attraction, obstacle/friendly separation, projectile evasion, and enemy attraction.
- **Marksman** demonstrates predictive Tank leading, clear shot lanes, and Scout screening.
- **Convoy** demonstrates Transport formations, Scout escorts/body-blocking, and trailing Tank support.

Shared steering, friendly-contact avoidance, projectile evasion, predictive aiming, and line-of-fire checks live in `baselines/common.py`. All controllers are deterministic, consume the sampled class specifications, persist state for one match, and are recreated for the next.

```bash
swarmbench match --controller-a marksman --controller-b convoy --seed 42
```

See [`docs/CONTROLLER_API.md`](../../../docs/CONTROLLER_API.md) for the public contract.
