# Stage 1: bounded closed-loop probe (plan)

**Status (8 Nov 2026):** approved by the owner (`DECISION_LOG.md` P23); plan written, implementation starting.
Bodies (P16): `60-N-T-10` and `60-N-H-10` (`round2/refine/narrow_carpet_60`) and `75-M-H-10`
(`round2/refine/feet_75`). The probe checks whether a controller can use the margins the screens found
(`ROUND2_REVIEW.md` §13). It is not the verification training. Nothing here changes the round-two records.

**Summary.**

1. **Spine: a reference tracker.** It tracks the designlab reference walks and turns with inverse-dynamics
   feedforward and divergent-component-of-motion (DCM) balance feedback, with about 14 gains per body fitted by
   CMA-ES. A prototype, untuned and fed the true centre of mass, already walked all three bodies 40 strides at the
   nominal speed during the design study.
2. **On top: a residual PPO policy** (deployable actor inputs, privileged critic), plus a policy-only ablation.
3. **Output per body:** a verdict on each behavior of `ROUND2_REVIEW.md` §13 from 20 fresh paired seeds, and each
   Stage 0 margin bisected to failure, naming the limit that binds it (body-limited or controller-limited).
4. **Equal budget:** per body 2 × 720 CMA-ES evaluations, 2 × 5M PPO steps, one policy-only seed and identical
   test panels: about 10 core-hours, about 10 h of wall clock for all three on this 4-core machine, capped at 12 h.

The plan came from a design study: three independent designs (tracking first; RL in the repository's existing
stack; reference-guided RL), judged and merged. Throughput and the prototype were measured on this machine.

## 1. Judgment

- **Design 1** is the most informative and the cheapest:
  - It bisects each margin to failure and labels each failure body-limited or controller-limited, at 1.2 h per body.
  - Pilot (a prototype tracker run during the design study; true CoM, untuned): all three bodies walked 40 strides, and 20 at μ 0.54.
  - The bodies fell at Fr 0.05 (60 cm), under a 2 N·s push and on the piles.
  - It has no learned controller.
- **Design 2** reuses the most RL code. But it learns six skills from the bare reference in 7.5M steps. It also goes through private `train_base` functions and has no privileged critic. A failure can't be traced to either the body or the learner.
- **Design 3** is the closest to BEHAVIOR_SPEC §2. It also shows that the reference itself breaks the sway and acceleration limits. It costs about 8 h, and its failures can't be traced either.

## 2. What is built

- Paths are relative to `/home/user/mesozoic-labs/hardware/trex_form_study`. `environments/` and `configs/` are relative to `/home/user/mesozoic-labs`.
- Only new files are added; `round2/` is not touched. The code lives in `designlab/stage1/` with tests in `tests/test_stage1.py`; run outputs go to `stage1/runs/<run>/` (large traces and checkpoints are git-ignored; JSON summaries and the report are committed).

| File | Content |
|---|---|
| `designlab/stage1/plant.py` | Checks `xml_sha256` (993c5ac5…, 96ef84c5…, 0517e4d3…). MjSpec adds an IMU at the ledger's `imu` row (gyro, accelerometer, xbody framequat), encoders, foot load from pad–floor contacts, a home ctrl and `carpet.hold_servos` gains. Joint ranges are cut to `summary.joint_range_usable` (`review/screening/<id>_clearance.json`). The home pose is re-settled for 1 s, because 75-M-H-10 has no contact at home. Floors come from `carpet.floor_variants`, `apply_floor` and `pile_model`, plus `pile_floor(grid=0.008)` for training. |
| `drive.py` | Each ankle becomes two tendons, q_p ± q_r, with one ±peak motor each. That gives \|τp\| + \|τr\| ≤ 2·peak, as in `sizing._motor_space`. A torque–speed envelope from `catalog.Actuator`, updated every substep, holds `.peak` up to `.speed_at(24.0)`. It then falls linearly to zero at the no-load speed on 24 V. |
| `refs.py`, `transitions.py` | Call `gait.set_accuracy("r2")`. Then `gait.generate` and `inverse_dynamics` for `sizing.NOMINAL`, `sizing.BRISK`, `envelope.FAST` and the carpet lifts (0.19 / 0.195 / 0.18 H), and `turning.generate` for `turning.CASES` in both directions. Starts, stops, speed changes and turn entries use ZMP preview control and `statics.leg_ik`. |
| `control.py`, `estimator.py`, `tune.py` | The servo target is ctrl = q_ref + Δq + (kv·q̇_ref + τ_ff + Δτ)/kp, set every 2 ms. DCM feedback at 100 Hz drives the ankle CoP (5 mm inside the pad) and the swing foot (inward by at most the turn gap minus 1 mm). It also drives trunk PD and heading and phase-rate PI (±5 %). A capture step handles pushes while standing. The estimator uses encoders, IMU and stance-leg kinematics. Tuning is numpy CMA-ES. |
| `env.py`, `policy.py`, `train.py` | A standalone env, because BaseDinoEnv holds one ctrl per step. The residual is 14 joint offsets (±0.2 rad on the legs, ±0.1 rad on neck and tail) plus ±25 % phase rate, at 50 Hz with a 10 Hz low-pass. The actor reads encoders, IMU gravity and gyro, foot loads, phase, command and its previous action. The critic also reads root velocity, height, errors, friction, floor and push (pattern: `critic_phase_ppo.CriticPhaseActorCriticPolicy`). Reward: 0.55 imitation plus 0.35 `gaussian_tracking_reward`, minus duty, slip, flight, impact, clearance and tilt penalties. SB3 PPO uses Design 2's settings. Friction, mass, kp, latency and IMU noise are randomized. |
| `episodes.py`, `metrics.py`, `sweeps.py`, `report.py`, `tests/test_stage1.py` | Metrics from per-substep contacts, `carpet.swing_profiles`, `toe_catch`, `clearance.screen`, `recovery_gate.binomial_lcb`. |

Controllers, all on the same seeds:
- **R0:** the reference with feedforward and PD only.
- **R1:** the tracker on the true CoM, as a diagnostic.
- **R2:** the tracker on the estimated CoM.
- **R3:** R2 plus the residual. This is the one controller judged on every behavior (BEHAVIOR_SPEC §3).
- **R4:** policy only (Q3).

## 3. Order and fail-fast checks

1. Decisions: taken as listed in §9 (the owner may override any).
2. **Plant. G1:**
   - the hashes match;
   - with zero command the body stands 20 s on hard floor and on low pile;
   - the tendon map equals `sizing._motor_space` to 1e-9;
   - seeded runs repeat exactly;
   - the sim runs at least 20k steps/s per core.

   If any check fails, stop.
3. **References. G2:**
   - they match the records to 6 digits (60-N-T-10 nominal: 0.218266 m/s, CoP margin 0.030316 m);
   - feedforward replay with the envelope walks at least 10 strides (13.6 / 16.6 / 60 without the envelope).
4. **Transitions,** within one day. **G3:** the steady part is within 1 mm of the periodic walk's CoM and 2 mm of its ZMP. Otherwise fall back to Design 3's amplitude ramps and record that.
5. **Tracker. G4:** it reproduces the pilot (40 strides; 20 at μ 0.54) and walks at least 20 strides on the estimated CoM.
6. **Tuning. G5:** R2 passes the nominal walk and both turns on the validation seeds (at least 17/20). A body that fails is labelled controller-limited and keeps its budget.
7. **PPO,** seed 42, then 43.
   - **G6a:** a zero residual reproduces R2, and each job runs at least 600 env steps/s. Otherwise cut the steps equally for every body.
   - **G6b, at 1M steps:** R3 survives at least as long as R2. Otherwise make one fix, then stop.
   - **G6c, at 2.5M steps:** R3 beats R2 on at least one condition. Otherwise skip seed 43 and judge with R2.
8. **Panels and sweeps. G7:** run on the references themselves, the metrics reproduce Stage 0: sway 0.168 / 0.169 / 0.164 H; cadence 1.96 / 1.97 / 1.76 steps/s.
9. **Report.**

## 4. Behaviors and gates

Per body: 20 episodes per condition, seeds 3042–3061, each starting from the settled stance.

| Behavior | Conditions | Success |
|---|---|---|
| Stand | 20 s; hard floor, robot_floor_carpet, low pile | tilt < 12° |
| Walk | nominal 0.218 / 0.217 / 0.243, brisk 0.315 / 0.314 / 0.351, Fr 0.05 0.446 / 0.444 / 0.496 m/s, commanded 5 % above; μ 0.9, and also 0.54 at nominal | 10 s mean ≥ target; duty 0.55–0.65, no flight; at nominal, cadence 1.6–2.2 steps/s, stride 0.5–0.65 H, GRF ≤ 1.6 BW; roll and pitch about home ≤ 4° RMS and ≤ 8° peak; toe clearance ≥ 0.04 H; slip ≤ 5 mm median, ≤ 15 mm worst |
| Turn | ±0.35 rad/s walking, ±0.5 rad/s in place, through 180° | yaw rate ≥ target; walk limits hold; in-place drift ≤ 0.1 m; no leg interference |
| Start and stop | stand, nominal, stand; twice | 90 % of speed within 2 strides; stopped within 1.5 strides (double support or a step allowed); acceleration per Q10 |
| Push | standing and walking; 4 directions; a 0.1 s pulse at 0.5 × the Stage 0 forward bound (Q7) | no fall; back to the command within 3 s |
| Carpet | nominal walk with the carpet swing at μ 0.54, on robot_floor_carpet and low pile | walk limits hold; nominal speed; toe ≥ `carpet_target(H)` (24.3 / 24.1 / 30.1 mm) above the pile; no `toe_catch` |

Every success also needs, on each drive, motor-space RMS ≤ sustained torque and peak speed ≤ `speed_at(24 V)`. Every joint must also keep at least 0.10 rad of usable range.

An episode counts as a fall if the trunk tilts more than 0.6 rad, the root drops below 0.7 × home height, or a `cl_*` proxy goes below the floor. Only the foot pads touch the floor, as in the screened model.

**Pass:** 20/20 survive (LCB 0.861) and at least 17/20 succeed (LCB 0.656). The thresholds are 0.80 and 0.60, from `configs/behavior_certification.toml`.

**Reported, not gated:** medium and high pile, sway, head and tail motion, brisk and Fr 0.05 walks at μ 0.54, and the 0.5 rad/s walking turn.

**Verdict:**
- *shown*: both seeds pass;
- *provisional*: one seed passes;
- *body-limited*: in the 0.5 s before most failures, a hardware limit binds: a drive, the ankle linkage, slip, the CoP at a pad edge with the ankle saturated, step reach, the turn-gap clamp or joint range;
- *controller-limited*: any other failure.

## 5. Margin ledger (bisection on R2 and R3; 7 halvings; seeds 3042–3044)

| Margin | Swept | Stage 0: 60-N-T-10 / 60-N-H-10 / 75-M-H-10 |
|---|---|---|
| Grip | lowest friction μ that still works: nominal walk; 0.35 rad/s turn; brisk walk | per body, walk / turn / brisk: `60-N-T-10` 0.48 / 0.54 / 0.75; `60-N-H-10` 0.47 / 0.53 / 0.73; `75-M-H-10` 0.36 / 0.41 / 0.56 (this cell groups by body, not by condition; decision 33) |
| Push | Δv per direction, standing and walking | forward bound 0.487 / 0.478 / 0.589 m/s |
| Stop | deceleration held | one foot 1.25 / 1.25 / 1.41 m/s² |
| Drive speed | envelope speed scale at Fr 0.05 | knee 0.93 / 0.95 / 0.93, including the screens' 15 % speed margin; without it 0.81 / 0.83 / 0.81 (decision 33) |
| Drive torque | torque scale | worst sustained ratio 0.76 / 0.76 / 0.77, the larger of the nominal walk's RMS and the one-foot hold; the nominal walk's RMS alone 0.76 / 0.76 / 0.66 (decision 33) |
| Model doubts | kp × 0.25–1, latency 0–20 ms, mass +10 %, pile damping 0.5–2 | no Stage 0 value |
| Pack voltage | bus 25.2 V (full), 22.2 V (nominal), 21 V (near empty) in the drive envelope | Stage 0 used 24 V; at 22.2 V the knee's rated speed (8.04 rad/s) equals the Fr 0.05 requirement including the screens' 15 % speed margin (8.08–8.12 rad/s; the peak swing speed itself is about 7.0 rad/s) |

## 6. Budget and seeds

| Per body | Amount | Core-hours |
|---|---|---|
| Tuning | 2 × 720 evaluations of 55 simulated s each | 2.2 |
| PPO | 2 × 5M steps, planned at 750 steps/s | 3.7 |
| Panels and sweeps | R2 and R3 on 2 seeds each; R0 and R1 on hard floor | 2.3 |

- R4, if approved, adds 1.9 core-hours per body.
- Total wall-clock is about 8 h: 10 h with R4, about 3 h for the tracker alone.
- The cap is 10 h, or 12 h with R4. If it is hit, every body is judged at the last checkpoint all of them reached.
- Wall-clock is reported per body, because 75-M-H-10 runs slower on pile.

Seeds:
- CMA-ES: 42 and 43, trained on episodes 1000–1999 and selected on 2000–2019.
- PPO: 42 and 43.
- Panels: 3042–3061 for every body and controller, with pushes paired by seed.

## 7. Commands and outputs

Each step runs as `python -m designlab.stage1 {prepare|tune|train|eval|sweep|report} --body <id> --seed <k> --out stage1/<run>`. `all --cap-hours 10` runs everything.

Each body's folder `stage1/<run>/<body>/` holds:
- `plant.json`: gains, envelopes, hashes, versions, commit;
- `refs.json`;
- `tune/<seed>/cma.json`;
- `ppo/<seed>/run.json`;
- `eval/<row>/<behavior>/<condition>.json`: per-episode metrics, LCB, failure label;
- `sweeps/*.json`;
- 50 Hz traces;
- `summary.json`.

The report is `stage1/<run>/STAGE1_PROBE.md`.

## 8. Report to the owner

The report gives:
- the verdict grid, with the provenance of each controller;
- the margin ledger, naming each binding limit;
- the R0–R4 ablations and how sensitive the results are to the plant;
- the budget used;
- what verification training should carry over;
- draft decision-log rows (Stage 1, A15, A16) for the owner's approval.

It does not show hardware effects, carpet dynamics (A16) or real self-collision. A controller-limited result says nothing about the body.

For the owner's follow-up (P25), the report also gives a gap map for the 60 cm build: each behavior or look it falls short on, whether the body or the controller limits it, and the options to close it (feet, pads, knee transmission, supply voltage, a new reference walk, or the 75 cm body) with their mass, cost and risk.

## 9. Decisions taken

The design study listed ten decisions. None blocks on the owner, so the defaults below are taken; the owner may
override any of them.

1. **Location:** the code is in the repository on the working branch (`designlab/stage1/`, `tests/`, `stage1/`),
   so nothing is lost if the cloud container is recycled.
2. **Scope:** the tracker plus the residual PPO policy, on the repository's validated Stable-Baselines3 and
   Gymnasium versions where they install (the exact versions used are recorded in each run's `plant.json`).
3. **Policy-only ablation:** yes, one seed per body (BEHAVIOR_SPEC §2 asks for it).
4. **Drive model:** the holding-servo gains as hard limits, with the torque–speed envelope and the two-motor ankle
   linkage; the servo gain is swept in the margin ledger. (The MJCF's own gains are about 7× softer, and the 60 cm
   bodies fall over on them.)
5. **Budget:** about 10 core-hours per body, the same for each, with a 12 h cap in all.
6. **Pass rule:** 20/20 episodes survive and at least 17/20 succeed (the repository's certification thresholds);
   "shown" needs both training seeds.
7. **Push size:** a 0.1 s pulse at 0.5 × the Stage 0 forward push bound (0.24 / 0.24 / 0.29 m/s) in four directions;
   the push margin itself is bisected to failure.
8. **Carpet:** gated on `robot_floor`'s carpet and low pile at μ 0.54; medium and high pile are reported, not gated,
   because carpet damping is not sourced (A16).
9. **Sway:** reported against 0.12 H, not gated. The reference walk itself sways 0.164–0.169 H; lowering the gait's
   sway gain to 0.7 only brings it to 0.162–0.166 H, because the walk's ZMP correction restores it. Gating on it would
   need a new reference walk (finding for the owner).
10. **Start and stop acceleration:** gated on the forward component low-passed at 1 Hz. In steady walking the
    reference already peaks at 2.1–2.2 m/s² (centre of mass, horizontal), above the spec's 1.5 m/s² (finding for the
    owner).

### Decisions taken during the build (step 3, 8 Nov)

The step-3 build found that the plant's two-motor ankle needs three modelling choices the plan did not settle, and
that one of the plan's range rules cannot hold for the turn references. They are decided on the reasons below, which
do not depend on the gate's outcome; the results with the other choices stay recorded in `gates.json` (G2's 2 × 2
plant matrix and the 20-start panels).

11. **Ankle servo gains: joint space.** `carpet.hold_servos` defines one hold servo per joint for the screened model,
    which drives each ankle axis with one motor. Applying that rule to each of the linkage's two motors doubles the
    ankle's stiffness in joint space; giving each motor half the gain reproduces the screened stiffness exactly.
    (Per-motor damping then comes out equal on pitch and roll: with the linkage armature of decision 12 it is
    0.94 N·m·s/rad on the 60 cm bodies and 4.16 on `75-M-H-10`, against `hold_servos`' 0.80 / 0.65 and 3.28 / 2.90 on
    pitch / roll, so 17–27 % more damping on pitch and 44–45 % more on roll; recorded in `plant.json`.)
12. **Ankle armature: the linkage's.** Both motors' rotors are geared to both ankle axes, so each axis carries twice
    the screened single-motor armature. This is the physical two-motor ankle the plan specifies.
13. **G2 replay rule: a 20-start panel.** An open-loop replay has no feedback, so its fall time is chaotic: start
    offsets of 1 mm/s move one body's run from 25.8 to 5 strides. G2 is judged on the median of 20 replays (root
    velocity offset by at most 1 mm/s, seed 0) reaching 10 strides, and the single unperturbed run is reported.
14. **Joint range in the turns.** The 0.5 rad/s turn references themselves leave only 0.046–0.095 rad of hip-yaw
    range (and 0.1004 rad in `75-M-H-10`'s 0.35 rad/s walking turn), so plan §4's 0.10 rad rule cannot hold in
    the turns. Walks keep 0.10 rad. In turns a success must reach no joint limit and keep at least the reference's own
    smallest margin less 0.02 rad. (Finding for the owner: the fastest target turns use nearly all of the thigh
    twist's range.)

With choices 11–13, G2's replays reach 60.0 / 22.8 / 60.0 strides unperturbed and panel medians of 33.6 / 24.2 / 60
(`60-N-T-10` / `60-N-H-10` / `75-M-H-10`).

### Decisions taken during the build (steps 4–5, 9 Nov)

The transitions pass G3 on all three bodies. The tracker passes G4 on `75-M-H-10` but fails it on both 60 cm bodies
with its untuned defaults (R1 hard / R1 at μ 0.54 / R2: 14.1 / 5.0 / 2.2 and 30.5 / 12.0 / 7.5 strides against
40 / 20 / 20), and every fall is the controller's: lifting every hardware limit does not change panel survival
significantly. Two things separate it from the prototype that walked: the default heading and phase-rate terms, and
the two-motor ankle, which lets one axis take up to twice a motor's peak torque while a 60 cm foot's pads reach only
15 mm behind the ankle axis. With the prototype's terms and the screened per-axis torque box, R1 walks 60 strides in
19 and 17 of 20 starts. These decisions follow; none changes a body (P24).

15. **Ankle torque in the tracker.** The tracker commands each ankle axis at most one motor's peak torque (the
    screened per-axis box) and no more than the stance pad can carry with its centre of pressure 5 mm inside the pad,
    as §2 already asks of the DCM feedback. More torque only rolls the foot. This limits the controller, not the plant.
16. **Tuning start.** CMA-ES starts from the prototype's terms (ankle and step feedback and trunk PD at its values;
    heading and phase-rate gains at zero) and may move every gain.
17. **G4 is judged at that start**, with the panel rule of decision 13. G4 checks that the tracker reproduces the
    prototype; the untuned defaults' result stays recorded.
18. **"Stopped within 1.5 strides"** means forward speed below 5 % of the command within 1.5 strides of the stop
    command. The stop reference ends forward motion after 0.76–0.79 strides and reaches full rest after 1.55–1.57
    strides; the time to rest is reported.
19. **Estimator stance point.** The stance-leg odometry uses the loaded pad's contact point, not the foot body's
    centre of mass (about 11 mm above the floor, which moves while the contact does not).
20. **Fall attribution and the inward step.** A condition is body-limited by a hardware limit only if lifting that
    limit from the start, over the condition's 20 starts, improves survival significantly (paired test, p < 0.05);
    otherwise its falls are controller-limited. The turning screen's "turn gap" is the pelvis plate's clearance to the
    same-side thigh in the turning pose, not a gap between the legs, so it is not a hardware limit and does not bound
    the swing foot. The inward swing-foot bound is the leg-to-leg clearance: the swing pad stays at least 5 mm from
    the stance pad (the legs are 32–58 mm apart at the reference stance). This supersedes §2's "turn gap minus
    1 mm" and §4's turn-gap entry in the body-limited list.

### Budget re-planned (9 Nov, before the fit)

21. **Budget.** Measured on the built code, the Python tracker runs at 0.24 wall-s per simulated s per core, about
    2.4× the plan's assumption (MuJoCo's step is only about 16 % of the cost). The CMA-ES fit as planned (2 × 720
    evaluations per body) takes about 16 core-hours, about 6.5 h of wall clock for the three bodies in parallel, and a
    residual-PPO environment would run at about 200–235 env steps/s against G6a's 600. Plan §6's 10 h cap (12 h with
    the policy-only ablation) cannot hold. The stage cap becomes **24 h of wall clock in all**, still equal per body:
    the fit runs as planned; a faster tracker is built meanwhile in an isolated copy and merged only after the fit,
    once it reproduces recorded evaluations bit for bit; PPO (two seeds per body) and the policy-only ablation then
    get an equal per-body share of the time left (5M steps per seed if the faster tracker lands, otherwise cut
    equally); if the cap is reached, every body is judged at the last checkpoint all of them reached (§6). The owner
    may set a different cap.

22. **The fit's rules** (objective `stage1-tune-v2`, `designlab/stage1/tune.py`). CMA-ES fits 13 gains; the capture
    step's margin is held at its decision-16 start value, 0.0, because the fit's behavior mix has no standing push and
    the objective cannot see it (the gated push panels later use 0.0). G5 selects the candidate with the fewest
    validation falls, then the lower mean survival penalty, then the lower mean performance penalty, so survival
    dominates as §4 requires. Drive RMS is judged over the same window as the other walk and turn limits. During the
    fit and until G5 has run, the ten modules the tuner hashes (`__init__`, `control`, `drive`, `estimator`, `legs`,
    `metrics`, `plant`, `refs`, `transitions`, `tune`), the run directory's reference and transition files, the
    library versions, `designlab/*.py` and `round2/` stay unchanged; the faster tracker of decision 21 is built in a
    separate working tree.

### Decisions taken before training (9 Nov, after G5)

G5 passes on all three bodies with both CMA-ES seeds. Training uses each body's selection by `env.GAINS_RULE`:
`60-N-T-10` seed42_gen20, `60-N-H-10` seed43_gen60, `75-M-H-10` seed43_gen40. The faster tracker of decision 21 was
merged only after this. It reproduces recorded evaluations exactly: two fit evaluations, checked at every physics step,
and two G5 validation episodes of each body's selection, checked on every recorded value. A zero residual reproduces R2
at every physics step, also with the trunk's mass changed (G6a). The checks before training found the following; they
change no body (P24).

23. **Carpet and pile.** The fit and G5 ran on hard floor only. On robot_floor_carpet and on the low pile, R2 fell in
    each of the 3 starts tried per floor:
    - `60-N-T-10` at the tuning start: 3.3–3.5 s on carpet, 3.0–3.1 s on low pile;
    - `75-M-H-10` at its G5 selection: 4.9–5.0 s on carpet, 4.4 s on low pile.

    Sliding friction alone is not the cause: at μ 0.54 on hard floor, R2 walks the whole carpet walk.

    A refit with carpet in the fit mix and in G5 would take about as long as the fit did (about 6.5 h on the 4 cores)
    and cannot fit within decision 21's cap beside PPO. So:
    - R3 and R4 train on the planned floor spread: hard, hard, robot_floor_carpet, low pile (`env.FLOOR_RULE`).
    - The carpet condition is reported for R0–R2 with its fall attribution (decision 20). Its verdict for R3 and R4
      rests on what the policy learned.
    - The report states that the tracker was never fitted on carpet.
    - A carpet-aware fit is the first follow-up if R3 does not walk on carpet.
    - G6's monitor includes the carpet condition (decision 27).
24. **Learning rate** (`policy.LR_RULE`). The learning rate is 1e-5 → 3e-6 (linear), which is design 2's schedule
    × 0.1; target_kl stays 0.03. The measured action noise, σ 0.02 (`policy.LOG_STD_RULE`), is 18× smaller than the
    build's earlier e^-1, and PPO's KL per step grows as 1/σ²:
    - At design 2's 1e-4, each update stopped after 1–9 of its 160 minibatch steps. Without the stop, the update
      diverged (KL 3.5).
    - At 1e-5, every minibatch step of 6 updates ran (KL 0.019–0.023 against the 0.045 stop; clip fraction 0.23–0.27),
      and the critic's explained variance turned positive by the fourth update.
    - At 3e-6, every step also ran, but the explained variance stayed negative.

    Measured on R3 `75-M-H-10` (`stage1/runs/2026-11-08/ppo_lr_probe.json`). R4 takes the same settings.
25. **Rules from the PPO build**, each recorded in the code and in every run.json:
    - **Trunk mass** (`env.MASS_RULE`). The trunk mass factor 0.9–1.1 scales the trunk's mass and inertia, and MuJoCo's
      derived constants are recomputed. The controller and R2's estimator keep the nominal model.
    - **Servo stiffness** (`env.KP_RULE`). Training draws kp × 0.8–1.2. The ledger's kp × 0.25–1 is an evaluation sweep.
    - **Floors** (`env.FLOOR_RULE`). Env i runs on hard, hard, robot_floor_carpet and low pile (8 mm grid), in turn.
    - **Seeds** (`train.SEED_RULE`, `env.EPISODE_SEED_RULE`). Env i of session k of PPO seed s is reset with
      10,000 s + 100 k + i, and its episodes draw seeds from that. No two runs, sessions or envs share an episode.
    - **Gains** (`env.GAINS_RULE`). Both PPO seeds train on the body-level G5 selection. A resume is refused if the
      settings, env configs, gains, libraries or Stage 1 sources changed. A change of sources is allowed only when
      flagged.
    - **Action noise** (`policy.LOG_STD_RULE`). σ 0.02 was measured on `75-M-H-10` only, the one body with a G5 record
      at the time; R4 takes the same value.
    - **Judging** (`train.JUDGED_RULE`). There is no best-model selection: each mode is judged over its own runs at the
      budget's checkpoint, or at the last checkpoint every run reached if the cap stops training.
26. **G6a and the budget** (`train.BUDGET_RULE`, `train.CHECKPOINT_RULE`). G6a's rate is judged per wall second of the
    job, with its process count. One R3 job runs about 200 env steps/s, against G6a's 600:
    - 4 envs in one process: 204/s at load 4.2 and 199/s on an idle machine;
    - one env alone: 358/s on hard floor, 234/s on carpet and 135/s on low pile.

    The plan's consequence applies: the steps are cut equally for every body. Nine jobs (6 R3, 3 R4) share the 4 cores
    and must leave the panels time before the cap (08:00 UTC, 10 Nov). Every run of R3 and R4 therefore trains 4M env
    steps instead of 5M (4,005,888 in whole rollouts).

    Checkpoints are saved at rollout starts, after the previous update. The last one is saved after the last update, so
    the judged policy has trained on every step.
27. **G6b and G6c** (`train.G6_RULE`).
    - **Conditions.** The monitor runs the six hard-floor tasks plus the carpet walk on robot_floor_carpet and on low
      pile, with 3 episode seeds each.
    - **G6b** compares the mean survival over every condition.
    - **G6c** needs, on some condition, fewer falls than R2. Failing that, it needs as many falls and, paired by
      episode seed, a higher mean reward per step on every seed, with a mean gain of at least 0.01 per step and at
      least R2's own spread across the seeds. A 0.004 gain after one Adam step no longer passes.
    - **When.** Both run at the first rollout start at or past the plan's 1M and 2.5M steps.
28. **R4.** One seed (42) per body, with R3's settings, floors and budget. R4 is judged over its own runs. G6 does not
    apply to R4, because G6 compares a policy with R2.
29. **Training order, priority and stop** (decided during training). Plan §3 trains seed 42, then 43, but the cap
    does not allow that order. All nine runs (R3 seeds 42 and 43 and R4 seed 42, on each body) therefore started
    together at 13:51 UTC on the 4 cores. G6c's consequence ("skip seed 43 and judge with R2") then stops seed 43's
    run if seed 42 fails G6c.

    At 15:00 the work left exceeded the time left before the cap: PPO at the measured rates (R3 76–87 env steps/s per
    job and R4 105–121, with nine jobs sharing), the panels, and the margin ledger. So:
    - R4, the ablation, runs at a lower CPU priority (nice 4) than R3, so that R3 reaches more of its budget.
    - Training stops at 03:00 UTC at the latest, leaving the policy rows' panels and the report time before 08:00.

    Each mode is then judged at the last checkpoint all of its runs reached (`train.JUDGED_RULE`, decision 21). R4's
    may be lower than R3's, and the report says so.

### Decisions taken for the panels (9 Nov)

30. **Back to the command after a push** (`panels.RECOVERY_RULE`). §4 asks that a pushed body be "back to the command
    within 3 s". The panels read this from the end of the pulse:
    - **Standing:** the true CoM's horizontal speed is below 5 % of the nominal command (decision 18's "stopped") at
      every feedback update from some time at most 3 s after the pulse's end to the end of the episode.
    - **Walking:** from some update at most 3 s after the pulse's end, the CoM's mean velocity over every stride-long
      window that starts at an update and ends within the episode is 90–110 % of the commanded speed along the walk's
      starting heading and at most 10 % of it across. The 90 % is the start's reading (§4 Start and stop); 110 % and
      10 % bound a push that leaves the body faster or moving sideways by the same margin. Unpushed R2 walks read
      96.1–98.6 % along and at most 2.8 % across (R2/42 on all three bodies, about 7 s of steady walking each).
    - **The window looks ahead from the pulse's end.** A trailing window averages the stride before each update, so
      just after the pulse it mostly measures the walking before it. It read the body back on its command 0.001 s and
      0.008 s after the pulse in two episodes that then fell about 0.9 s later.
    - **The reading must hold.** Each push episode runs 2 s past the 3 s, so the reading holds for at least 2 s. A fall
      before the end means the body has not recovered. An update at the pulse's end itself is recorded as "never left
      the command".
31. **How the panels read §4** (`designlab/stage1/panels.py`; each rule is copied into every record).
    - **Rows.**
      - R0 is the tracker with every feedback term and the ankle cap off. It commands R0's targets on the 1.05 clock,
        and at rate 1 it equals R0 exactly.
      - R1 runs on the true CoM with the body-level G5 gains.
      - R2/42 and R2/43 each use their own CMA-ES seed's G5 selection.
      - R0 and R1 run every hard-floor condition, the μ 0.54 walks included. The R2 rows run all of them.
    - **Floors and friction.**
      - μ 0.9 is the plant as compiled, so G5's episodes reproduce exactly; μ 0.54 is set as G4 set it.
      - The carpet walk sets μ 0.54, as the env does.
      - Piles are §2's 4 mm evaluation piles, not the 8 mm training grid.
      - Medium and high pile are not run on the 60 cm bodies: no 60 cm body has a settled stance on them (round two:
        none stands passively there).
    - **Walks.** Brisk and Fr 0.05 have no stop reference of their own, so each ends in its gait and is judged over its
      last 10 s. Cadence, stride and peak GRF are judged at nominal speed only, as §4 says.
    - **Turns** have a fixed direction per condition.
    - **Start and stop.** One nominal stride separates the start from the stop, with 2 s standing between the two
      cycles and at the end. Each stop is read by decision 18, from its own command until the next start.
    - **Pushes.** Each of the 4 directions, standing and walking, is its own 20-episode condition.
      - The force is the body's mass × 0.5 × the Stage 0 forward bound / 0.1 s, at the trunk's CoM.
      - The onset is drawn from the episode's seed and is the same for every row; walking pushes land at a uniform phase
        within the second stride.
      - Recovery follows decision 30.
    - **Carpet.** Judged as a nominal walk. "Above the pile" means above the uncompressed pile top, and a toe catch is
      the lowest point at or below the pile top in mid-swing.
    - **Limits.** Stands, pushes and start/stop judge the drive and range limits over the whole episode; walks and turns
      judge them over tune's windows.
    - **Attribution** (decision 20) runs on a condition's falls only when there are at least 5 of them, because the
      sign test cannot reach p < 0.05 with fewer. It runs with no null run.
      - A fall whose only candidate cannot be lifted (the CoP at a pad edge with the ankle saturated) is labelled
        controller-limited with that candidate named.
      - A success failure without a fall is labelled by its terms' class. Hardware terms: drive RMS, drive speed, joint
        range, leg interference. Controller terms: all the others.
    - **Verdicts per controller.** R2's come from R2/42 and R2/43, R3's from R3/42 and R3/43; R4 is at most provisional.
      The carpet condition is reported for R0–R2 and gated for R3 and R4 (decision 23).
    - **G7 gate.** Stage 0's sway (2 × the reference's sway amplitude / H) and cadence (2 / period) are read from the
      references themselves. This is a kinematic check, not a physics run.
    - **Failed episodes.** An episode that raised, or whose worker was lost, is counted apart from falls and is re-run on
      resume. Every episode keeps a 50 Hz trace (git-ignored).

### Decisions taken during training (9 Nov)

32. **G6b and its one fix** (`train.FIX_RULE`). At 1,007,616 steps, G6b failed on five of the six R3 runs: R3's mean
    survival over the monitor was below R2's.

    | Run | R3 (s) | R2 (s) | Cause |
    |---|---|---|---|
    | `60-N-T-10` seed 42 | 14.1 | 16.0 | falls on the brisk walk |
    | `60-N-T-10` seed 43 | 15.4 | 16.0 | falls on the brisk walk |
    | `60-N-H-10` seed 42 | 15.0 | 15.5 | falls on the brisk walk |
    | `75-M-H-10` seed 42 | 17.99 | 18.0 | carpet falls 0.1 s earlier than R2's |
    | `75-M-H-10` seed 43 | 16.9 | 18.0 | falls on the brisk walk |

    `60-N-H-10` seed 43 passed (15.9 s against 15.5). R3 falls on the brisk walk in 1–3 of 3 episodes, where R2 falls
    in at most one. On the hard-floor tasks R3 gains up to 0.02 reward per step. On carpet and pile it falls as R2 does,
    at nearly the same times.

    The policy's actions are small. On `60-N-T-10` seed 42 the joint residual is 0.012 of its range RMS (about
    2.4 mrad) and the clock rate −0.2 %. So the brisk falls show how close R2's brisk walk on the 60 cm bodies is to its
    stability edge. The panels agree: on `60-N-T-10`, 17/20 and 0/20 survive with its two fit seeds; on `60-N-H-10`,
    5/20 and 20/20.

    The plan's consequence is to make one fix, then stop:
    - **The fix.** The brisk and Fr 0.05 walks are weighted ×2 in the task mix (weights 1, 1, 1, 1, 1, 2, 2, 1, 1 over
      the walk, the four turns, brisk, Fr 0.05, the stand and the carpet walk).
    - **Where it applies.** Every R3 run, from its 1,007,616-step checkpoint, so every body and seed keeps one recipe.
      The steps the 60 cm runs had trained past that checkpoint are discarded.
    - **The check after it.** G6b runs again 1M steps later (`G6b_after_fix`). A run that fails it stops there, and its
      body's R3 is judged with R2. G6c stays at 2.5M steps.

    Other fixes were considered and rejected:
    - A fall penalty: a fall already forfeits the value of the episode's remaining reward.
    - Fewer carpet and pile episodes: this would give up decision 23's carpet experiment.
    - A smaller residual: it is already small.

### Decisions taken for the margin ledger and the report (9 Nov)

These were settled before any sweep ran and before the R3 and R4 panels; they change no body (P24). Each is recorded in
the code as a rule that every record carries.

33. **How the margin ledger reads §5** (`designlab/stage1/ledger.py`).
    - **Rows and seeds** (`ledger.ROWS_RULE`). R2 is the CMA-ES seed of the body-level G5 selection and R3 is R3/42:
      one seed per controller, not §6's "R2 and R3 on 2 seeds each", within decision 21's cap. A body whose R3 is
      judged with R2 (decision 34) has no R3 row. The episodes are the panels' first three (3042–3044).
    - **What "works" means** (`ledger.WORKS_RULE`). A value works if none of the 3 episodes falls and none fails a
      success term that its seed passed at the search's nominal value. Read outright (all 3 succeed), a term one seed
      already fails at nominal ends every search there: `60-N-T-10`'s R2/42 nominal walk passes the panels (20/20
      survive, 18/20 succeed), and seed 3042 fails only the peak GRF, so the grip, drive-torque, kp, latency and mass
      margins would all read "fails at nominal". At the nominal value itself, works is survives. Each evaluation also
      records the outright reading.
    - **The binding limit** (`ledger.LABEL_RULE`, `ledger.LIFT_RULE`). Decision 20's sign test needs at least 5 falls,
      which 3 seeds cannot give. So at each fails end (and point) with falls, the value is run again on the same 3
      seeds once for each relaxable hardware limit that is a candidate of one of its falls. The falls are body-limited
      by the limits whose lifting makes all 3 seeds survive, else controller-limited. Success failures are labelled by
      their new terms' classes (`panels.SUCCESS_LABEL_RULE`). When both occur, the label is that of the kind with more
      failed episodes, the falls' on a tie, as `panels.FAILURE_LABEL_RULE` breaks it. A search that fails at nominal
      also cites the panels' 20-seed label of the same row and condition, and is flagged if that record passes.
    - **Stop: not run, a deviation from §5 for the owner** (`ledger.STOP_RULE`). The deceleration a stop holds needs
      stop references that decelerate at other rates. The run holds one stop transition, and another would be a new
      reference, not this plan's. Sweeping the stop transition's own clock rate instead would change its stepping as
      well as its deceleration, and needs a reference clock whose rate changes by segment, which `control.Track` does
      not have. Each stop record keeps the existing stop reference's deceleration (1 Hz low-pass) and the row's
      start-and-stop panel readings. The report labels the row "not run (plan deviation, decision 33)".
    - **Stage 0 values** (`ledger.STAGE0_RULE`). §5's grip cell groups by body (";") and by condition ("/"): walk / turn
      / brisk per body, as `ROUND2_REVIEW.md` §7.4 and §7.6 give it and the turning screens' single-support friction
      need shows (runs `straight`, `walk_0.35_left`/`_right`, `straight_brisk`). The drive-speed value includes the
      screens' 15 % speed margin (`sizing.SPEED_MARGIN`); without it, which is what the ledger's limit compares, it is
      0.81 / 0.83 / 0.81. The drive-torque value is the larger of the nominal walk's RMS and the one-foot hold (on
      `75-M-H-10`, the hip yaw's hold); the nominal walk's RMS alone is 0.76 / 0.76 / 0.66. The report shows both
      numbers.
    - **Trunk mass "+10 %"** (`ledger.MASS_RULE`). Read both ways, as two points: the trunk × 1.1 (`env.MASS_RULE`'s
      factor: +4.15–4.40 % of the robot's mass) and the trunk factor that adds 10 % of the robot's mass (× 1.230 /
      1.227 / 1.241). The search for the largest factor that works goes up to × 1.3 (+12–13 % of the robot), beyond
      the plan's +10 %, to find the margin; it is extra compute within the cap. Records and the report give the whole
      robot's mass change beside each factor.
    - **Pile damping** (`ledger.PILE_DAMPING_RULE`). Both directions on the 20 s stand on low pile, which R2 passes on
      every body, and on the carpet walk on low pile, where decision 23 found R2 falling in every start tried (its
      carpet searches then end "fails at nominal" for one evaluation). Both run for every row that runs the pile.
    - **Where it runs.** Every record key holds the sha256 of the round-two model files the plant reads. A checkout
      without the git-ignored `round2/refine/<case>/models` rebuilds them elsewhere with another model-ledger JSON, so
      its keys differ from the main checkout's. The ledger, the R3 and R4 panels and any eval run from the main
      checkout; `eval` and `sweep` refuse a checkout without those files unless `--allow-rebuilt-models`.
34. **Judging after G6 failures, and the report** (`train.R2_JUDGED_RULE`, `train.JUDGED_RULE`,
    `designlab/stage1/policy_rows.py`, `designlab/stage1/report.py`).
    - **A body judged with R2.** When a body's PPO seed 42 fails G6c (§3 step 7, decision 29) or one of its R3 runs is
      stopped by G6b's second failure (decision 32), its R3 is judged with R2. Its R3 runs are then left out of the
      judged checkpoint, so a run stopped early does not cap the other bodies' checkpoint. Its R3 rows have no judged
      checkpoint (an explicit one can still be run for the record), the ledger leaves its R3 row out, and the report
      shows R2's verdicts, flagged, with R3's column marked "not judged". Until seed 42 reaches G6c the report says
      "G6c not reached".
    - **A policy row's checkpoint in the keys.** The R3 and R4 records' keys name the checkpoint by its step count and
      its files' sha256, with the run's env configs as they were for that checkpoint (before or after G6b's fix). They
      do not name how it was selected or whether it was final, so records made before `judge --final` stay valid.
      The report reads final from run.json's judged checkpoint at the same step count.
    - **R4's falls.** Decision 20's attribution is applied to R2 and R3 only, so a condition R4 fails by falls reads
      "fails by falls (R4: not attributed)", a settled verdict; the report does not flag the by-design case as interim.
    - **The body summary.** §7's `<body>/summary.json` is the report's `stage1_tables.json` (every body's tables from
      every step); the panels' part is `<body>/eval/summary.json`.
    - **Wall clock.** The PPO wall clock per body is the union of its runs' sessions (elapsed); their sum is given as
      job-hours, since the runs share the cores.
35. **Training stops at 01:00 UTC**, not 03:00 (decision 29). After training, the policy rows still need:
    - their panels: 3 rows × 28 conditions × 20 episodes per body;
    - R3's fall attribution;
    - R3's margin ledger.

    Together that is about 12 core-hours at the measured costs. The report and its check need about 2.5 h more, all
    before the 08:00 cap. At the measured rates, every R3 run passes G6c's 2.5M steps by about 00:45 (the 75 cm runs
    last).

    A run that has not reached 2.5M steps by 01:00 trains on alone until it does, but not past 01:30. Every mode is then
    judged at the last checkpoint all of its runs reached (`train.JUDGED_RULE`, `train.R2_JUDGED_RULE`). R3 thus
    trains about 2.5M of the 4M steps planned and R4 fewer (decision 29). The report states both.
