# Behavior specification (proposed, round one)

Status: **proposed targets for review**. They are set before any confirmation testing so that
later results can pass or fail against them. Nothing here has been demonstrated. No controller was
trained in round one: per the brief, controller development follows the design review. The
targets are written in size-normalized form, so they apply to whichever height class is selected.
`H` is hip height. Speeds use the Froude number Fr = v²/(gH), so a 60 cm and a 120 cm robot are
held to dynamically similar gaits.

## 1. Locomotion (selected controller, hard floor and declared carpet)

| Quantity | Proposed target | Why this value | Measured on |
|---|---|---|---|
| Nominal forward speed | Fr ≥ 0.012: 0.21 m/s at H = 0.39 m (60 cm class), 0.28 m/s at H = 0.66 m (100 cm), 0.30 m/s at H = 0.79 m (120 cm) | The reference walk used for sizing. About 2× the R3 median (0.104 m/s, Fr ≈ 0.003), which read as slow, rigid stepping. | Mean over 10 s of steady walking, 20 trials per surface |
| Brisk speed (stretch) | Fr 0.025 | Peak-load sizing case | Same |
| Cadence and stride | 1.6–2.2 steps/s at nominal; stride 0.5–0.65 H | Avoids both shuffling and lunging | Foot-contact events |
| Duty factor | 0.55–0.65 | Walking, not running | Foot-contact events |
| Turning | ≥ 0.35 rad/s while walking at nominal speed; ≥ 0.5 rad/s turning in place | Museum spaces need tight turns | Yaw rate over a 180° turn |
| Start / stop | Reach 90 % of commanded speed within 2 strides; stop within 1.5 strides; peak horizontal trunk acceleration ≤ 1.5 m/s² | Smooth starts and stops (R3 missed its acceleration limit in 5/20 trials) | Trunk IMU, 100 Hz |
| Body motion | Trunk roll and pitch ≤ 4° RMS, ≤ 8° peak while walking; lateral sway ≤ 0.12 H peak-to-peak | A stable, unhurried look | Trunk orientation and CoM track |
| Swing clearance | Minimum toe clearance ≥ 0.04 H on hard floor and ≥ 0.06 H on carpet (≥ 15 mm absolute) | Carpet pile and thresholds | Lowest pad point during swing |
| Slip | Stance-foot slip ≤ 5 mm per step (median), ≤ 15 mm worst | Traction margin | Foot site vs contact point |
| Impact | Peak vertical ground reaction ≤ 1.6 × body weight at nominal speed | Joint, bearing and structure loads | Contact forces |
| Head and tail during walking | Neck pitch tracks a ±0.05–0.15 rad motion and tail yaw ±0.1–0.2 rad without breaking any limit above | "Modest powered head and tail movement" | Joint tracking error and the limits above |
| Real-time visual quality | Independent viewers rate gait naturalness ≥ 3.5/5 on blind real-time clips (rubric: foot placement, knee/hock motion, body bob, head/tail coordination, no shuffling or jitter) | The goal is natural-looking motion, not slow rigid stepping | Rated video at 1× speed |
| Actuator duty | Every drive's RMS torque ≤ sustained planning torque; no peak over the planning peak; winding-heat estimate within the declared thermal budget | Connects behavior to the hardware ledger | Logged torques through the drive model |

**Carpet.** A friction change alone is not a carpet model. The carpet condition must include
compliant pile depth (contact stiffness and damping, 8–15 mm), higher rolling resistance on the
swing toe, and a randomized friction band. The repository's existing `robot_floor` carpet
(friction 0.95, solref 0.024 s) is a starting point, not a validated model.

## 2. Controller provenance (to be reported for every result)

* **Learned:** the PPO policy acting on deployable inputs only (encoders, IMU-estimated gravity and gyro, foot-load sensors, command, previous action).
* **Reference trajectory:** any phase-based or ZMP-based reference the policy adds residuals to.
* **Fitted corrections:** any hand-fitted feedback or transition terms.

Required ablations: reference-only, policy-only (no reference), and the full controller. Each is run
on the same fresh seeds. Privileged state may be used by a critic or for evaluation, never by the
deployed actor.

## 3. Autonomy (onboard Jetson; the external page selects and supervises only)

| Behavior | Proposed target |
|---|---|
| Selected-person following | Hold 1.5–2.5 m following distance (median), never closer than 1.0 m. Target kept in view ≥ 90 % of the time on scripted routes. |
| Multiple people and crossing | No silent identity switch: on any ambiguity the robot stops and reports "re-select" within 0.5 s. Reacquire the same person after ≤ 2 s occlusion in ≥ 80 % of crossings. |
| Target loss | Stop within 1.0 s of loss, report status, never pick a new person on its own. |
| Obstacles and blocked paths | No contact with static obstacles ≥ 0.10 m tall or with people. Detour when a path exists. Stop and report when blocked. |
| Stop / resume | Operator stop is honored within 0.3 s. Resume only on explicit operator start. |
| Interface disconnection | Heartbeat loss → controlled stop within 1.0 s, then stand. No motion until reconnection plus an explicit start. |
| Behavior transitions | Walk ↔ stop ↔ turn ↔ follow transitions meet the start/stop and body-motion limits of §1. |
| Perception inputs | Markerless ML person detection and tracking on rendered RGB-D (simulation) or the real camera. Ranging from ToF/lidar. Never simulator ground-truth positions. |

Integrated pass criterion: a single controller and interface must pass §1 and §3 together, in the
same runs, across locomotion, obstacle avoidance, person following and transitions. Walking
results from one controller cannot be combined with navigation results from another.

## 4. Runtime and power

* Removable battery: 15–20 min of the nominal reference walk at 75 % usable energy (65–85 % range reported), measured on hardware before it is claimed.
* External power: the robot must operate tethered with the battery cassette fitted (or an equal-mass dummy), the same CoM, and regenerated energy absorbed on board.
