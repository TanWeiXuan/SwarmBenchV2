# SwarmBench v2 game specification

## Versions, arena, and teams

The v2 constants are `ENGINE_VERSION=2.0.0`, controller API 2, scenario generator 2, replay format 2, and tournament format 2. Coordinates and dynamics use SI units; x increases left-to-right and y bottom-to-top.

The arena is 100 m × 60 m. Team A starts left and targets the right goal; Team B is its exact reflection across x=50. One seeded count is shared by both sides: 10–14 SCOUT, 4–8 TRANSPORT, and 2–4 TANK vehicles. SCOUT and TANK score 1 point; TRANSPORT scores 5. Every vehicle has physical radius 0.25 m and one hit point.

Nominal SCOUT dynamics are 5 m/s speed, 4 m/s² acceleration, and 16 m/s³ jerk. TRANSPORT uses 2.5, 2, and 8; TANK uses 1.5, 1.2, and 4.8. Each value is sampled independently and uniformly within ±20% once per scenario and shared by both teams. Controllers receive the exact values before play.

Goals are 3 m deep and 14 m high. The generator creates 8–15 non-overlapping circle/axis-aligned-rectangle obstacles: mirrored pairs plus one self-mirrored centerline obstacle for odd totals. Goals, obstacles, spawn coordinates, class assignment, counts, and dynamics are x-mirrored. Candidate arenas retain protected spawn/goal strips and must pass the existing 0.5 m-grid reachability test after clearance inflation.

## Movement and vehicle contacts

Physics runs at 20 Hz and controls at 10 Hz. Desired acceleration is norm-clipped to the sampled class limit. Actual acceleration approaches it under the sampled jerk limit, then position and velocity integrate with constant-acceleration RK4 and sampled speed clipping. A retained movement command applies until replaced.

All vehicle pairs, friendly or opposing, use a continuous relative swept-point test at 0.75 m. Contact destroys both vehicles. Obstacle tests inflate geometry by 0.25 m. Goal entry uses swept segment/AABB entry. Resolved vehicles leave active play immediately.

## Tank projectiles

Each TANK has five rounds. Firing is disabled before `t=5.0 s`; each Tank then has its own four-second cooldown. A valid fire direction is normalized and produces a point projectile moving at a constant absolute velocity of 20 m/s. Projectile velocity does not inherit Tank velocity.

Projectiles originate at the source Tank's center and cannot hit that source. They can hit friendly or opposing vehicles using a continuous relative sweep with a 0.75 m contact radius. The first hit destroys the vehicle and consumes the projectile. Obstacles consume projectiles, arena exit removes them, and projectiles neither pierce nor collide with each other. Every active projectile's ID, team, source, position, and velocity is perfect information.

Candidates sort by normalized contact time, then priority and stable IDs. Exact-time priority is projectile/obstacle, vehicle/obstacle, projectile/vehicle, vehicle/vehicle, goal, then projectile exit. This preserves obstacle shielding, one-for-one vehicle contact, and deterministic outcomes.

## Result, controller timing, and determinism

The default duration is 90 simulation seconds. Higher score wins; equal scores draw. A controller exception or hard timeout forfeits. Both controllers run concurrently while simulation time is paused. A response after 500 ms but before 5 s is discarded without rolling back controller state; at 5 s the process is terminated.

One game seed drives every scenario variable through the versioned generator. Stable ordering and tie-breaks make identical seeds and controller outputs repeat exactly. A replay stores the sampled scenario, controller hashes, accepted movement changes, actual shots, important events, and final result; rendering is non-authoritative.
