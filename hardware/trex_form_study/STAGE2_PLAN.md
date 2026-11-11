# Stage 2: the agility round, with character (simulation)

**Status (11 Nov 2026): the owner has taken decisions 3–6 below (P29); the look build (Track A) has started; nothing else has run.** Simulation only; it changes no body and adopts nothing
without the owner (P24). Paths are relative to `hardware/trex_form_study/` unless they begin `environments/`,
`notebooks/` or `docs/`. S1 and S1b are the Stage 1 and 1b reports (`stage1/runs/2026-11-08/STAGE1_PROBE.md`,
`stage1/runs/2026-11-10-speed/STAGE1B_SPEED.md`); "decision n" is `STAGE1_PLAN.md`'s, "S2-n" this plan's; [look] and
[pipeline] are this round's research notes (paths at the end). "a / b" figures are `60-N-T-10` / `75-M-H-10`.

## 1. Owner request

- **P28** (10 Nov): character (the powered head and tail) and being somewhat agile, not speed; the 60 and 75 cm builds
  pushed to their upper end in simulation; larger sizes much later.
- **Compute:** this CPU for quick validation; long training in a Colab notebook the owner runs, saving to Google Drive,
  from `notebooks/sb3_training.ipynb`.
- **Head and tail:** is the extra expressiveness worth a slight speed hit? Two heads are measured, today's and one with
  a neck turn and a jaw; a tail lift waits for the tail servo's bench test.
- **The agility bar** (agreed): walk reliably at 0.4 m/s or more; turn 90° while walking in about 2 s; take a nudge
  while walking without falling; stop within a step or two; all with a realistic servo delay. Character first:
  lifelike, not mechanical.

## 2. Questions

1. **Upper end.** With a lifelike walk and a realistic delay, which bar conditions does each build show, and what are
   its fastest walk, quickest turn and largest walking nudge?
2. **Expressive head.** What do neck yaw and a jaw cost on the bar, and add to the look?
3. **Next step.** Which hardware change, in the model only, would turn a failing condition or raise the upper end?

## 3. Bodies and variants

**S2-1 (owner decision 3): one 60 cm body on Colab, `60-N-T-10`.** The evidence cuts both ways (S1 §2–§3; S1b §2).
For the Tapered: 16 of 20 animal-pace walks up after Stage 1b's one-seed refit (the Hybrid 1), and the fast walking
turn, whose thigh-twist load the quick turn shares, meeting every limit in 14–18 of 20 (8–9). For the Hybrid: the best
brisk controller yet (20 up, 19 meeting every limit; the Tapered's best 19 and 17), 22 of 160 walking shoves survived
(6), and its nominal walk staying up to 8 ms of delay (6). Two fits of one body differed by 15–17 of 20 (S1b Answers).
The Hybrid would add 4 Colab runs, up to 80 runtime-hours (2 runs with the current head only). **75 cm: `75-M-H-10`**
as picked (P16, P18, P20).

**S2-2: two heads per body.**

| | Current | Expressive (`-X`) |
|---|---|---|
| Joints | neck pitch ±0.35, tail yaw ±0.45 rad (`round2/refine/*/models/*.xml`) | + neck yaw at the neck base, ±0.6 rad, cut to the clearance screen's range; + jaw, 0–0.6 rad ([look] §5.1) |
| Servos | neck STS3250 / XM540-W270, tail STS3215 (`stage1/runs/2026-11-08/<body>/plant.json`) | + 2 STS3215 (`designlab/data/actuators.json`: 0.055 kg, $22; planning sustained 0.491, peak 0.862 N·m) |
| Mass | none added | 0.075 kg per servo with bracket; yaw at the neck base, jaw servo 0.10 / 0.12 m ahead of the neck-pitch axis ([look] §5.1–5.2); the yaw stage's own bearing (mass, cost) not modelled |

- **`-X` is its own build** (about a day, and 1 core-h per body): neck-yaw and jaw roles in `mjcf.joint_role`, the
  sizing's motor map and the records' `selection` (`designlab/sizing.py:68-69`, `designlab/stage1/drive.py:126-127`);
  the yaw stage and hinged jaw in `designlab/build.py`; a new refine case with records and the screens behind the push
  bound and usable ranges. With the new joints locked it must equal today's body plus the added parts; it passes G1–G3
  (G4, the tracker's gate, does not apply). The env's 14-joint widths (`env.py:138-142, 284-286`) are generalised;
  new joints take the MJCF's default damping and friction, unsourced ([look] §5.1).
- **Expected costs, to be measured:** CoM 5–6 mm forward; hip-pitch hold +0.61–0.71 / +0.75–0.88 N·m (low end
  [look] §5.2, high end the lead's), from 0.55 to 0.65–0.67 and from 0.76 to 0.83–0.84 of the planning limit (the
  picks' recorded holds, `round2/refine/{narrow_carpet_60,feet_75}/candidates/*.json`). The 60 cm neck pitch, at
  0.83–0.84 of A1 in the closed-loop nominal walk (S1 §3), would reach about 0.92–0.93 with the jaw servo ([look] §5.2)
  before any look-at or roar, so the 60 cm `-X` may miss "every limit". At 75 cm, gaze shifts of 0.8 s or slower stay
  within the STS3215's peak ([look] §5.1). P2 deferred the jaw; `-X` models it in simulation only.
- **Research view, not a decision:** neck yaw is the most valuable joint per gram at a slow pace, the jaw second; the
  speed cost is probably small ([look] §5.3).

## 4. Tracks and method

### A. The look (CPU, first)

1. **Look generator:** new `GaitSpec` fields in `designlab/gait.py`, defaulting to today's walk so that Stage 1's
   references rebuild bit for bit (G2): [look] §3.1's F1–F6, F9–F12, F14, and F13 at 75 cm; no trunk pitch (F8). Other
   F rows below are `DECISION_LOG.md`'s.
2. **Tail-phase check first:** [look] reads today's tail as adding to the legs' yaw momentum; `designlab/turning.py`'s
   straight runs settle it.
3. **Variant ladder** V0 (today) to V6 (all) per body ([look] §4.1) at the nominal walk, 0.40 m/s and Froude 0.05,
   with a 90° turn from 0.40 m/s, a two-step stop, and a look-at and roar on `-X`, each priced by the Stage 0 screen
   before rendering, turns with their thigh-twist range and grip need (the Hybrid's too). Over a drive limit, a variant
   keeps its feature and loses that speed rung (character first) unless the owner decides otherwise.
4. **Renders:** kinematic playback (perfect tracking, no physics), labelled so, in real time from [look] §4.1's two
   views with its plots and anchors; a clip writer joins `designlab/render.py`.
5. **Look gate L1:** the owner reviews a page per variant (clips, unrevised cost, any use of F25's tuned walk, and
   cadence against a dynamically similar T. rex's, [look] Bottom line 5) with rubric scores from 3–5 viewers new to the
   project on V0 and the proposed variant, and picks the features; that reference is frozen by hash.

### B. The learned controller

1. **S2-3: R5, an imitation policy under commands, is the judged controller, chosen now** (a residual on R2L would
   leave `-X` without a base, R2L being fitted on the current heads only).
   - R5 is `control.Tracker` in R2 mode with no feedback terms or ankle cap (`terms=()`, `ankle_cap=()`): its
     estimator gives the inputs, and a zero action sends the panels' R0 targets (`panels.ROWS_RULE`) at the commanded
     clock, which defines **R0L**. At 50 Hz R5 adds joint offsets to the look reference's servo targets, feedforward
     kept: ±0.35 rad on the legs, ±0.15 on head and tail, ±25 % clock rate. No tracker: Stage 1's residual on it used
     about 1 % of its authority at σ 0.02 (S1 N3), and the tracker falls in nearly every 60 cm walk at 10 ms (S1b §4).
   - Exploration: initial σ 0.1 of each action's range, as Stage 1's rule (`policy.LOG_STD_RULE`) needs a zero action
     that keeps 18 of 20 walks up (R0: 3 on `60-N-T-10`, S1 §4); the learning rate, the largest of 1e-4, 3e-5 and 1e-5
     that runs every minibatch step in decision 24's probe.
   - Inputs: Stage 1's deployable ones (`env.py:138`), the last three actions and, on `-X`, the head command.
   - References (the approved look): walks at nominal and 0.30–0.50 m/s, walking turns at 0.35–1.0 rad/s, the
     0.5 rad/s turn in place, their starts, two-step stops and changes, commanded by speed and yaw rate. Tasks: Stage
     1's plus section 5's; half start mid-task (`Plant.set_state`, `Estimator.reset`, the delay queue filled from the
     reference).
   - Reward: Stage 1's (`REWARD_RULE`) plus foot and contact imitation, head attitude in the world, action rate,
     impact, and neck and tail torque over their limit; weights recorded in the env config.
   - Randomised: delay 0–25 ms; stiffness ×0.6–1.2; friction 0.4–1.2 (F22); trunk mass ×0.9–1.1; joint damping and
     friction ×0.5–1.5 ([look] §5.2); IMU noise; pushes to 0.5 × the Stage 0 bound walking and 1.0 × standing; gaze
     and jaw events on `-X`. The 10 Hz action filter (`env.py:64`) is re-chosen in validation.
   - Floors: hard on three envs in four, robot_floor_carpet on one; no low pile (P26; [pipeline] §3.3).
2. **Baselines and ablation:** R0L; R2L, the tracker refitted as Stage 1b's speed fit (0–10 ms; `STAGE1B_PLAN.md` §3)
   on the bar's tasks, current heads, one seed per body, here, without F5's toe-off, which its flat-pad ankle cap and
   estimator cannot follow ([look] §3.3); R6, policy only (Stage 1's R4 with R5's reward and starts), one seed per body
   (BEHAVIOR_SPEC §2).
3. **Validation on this CPU** ([pipeline] §5): G8a (R5's zero action reproduces the panels' R0 exactly, at 0 and
   10 ms); G9, as Stage 1's G7 (the new measures read as designed on the references, R0L and, for toe clearance,
   Stage 1b's 10 ms turn traces, and each reference passes its own bar with margin); rates and memory per floor and
   process count; kill and resume; a fresh-clone dry run of the bundle; the σ and learning-rate probes; the notebook on
   a tiny budget. **S2-4:** then 2M steps here on `75-M-H-10`, where R0 walks (S1 §4), with training's delay draw;
   Colab starts only if R5's 10 ms monitor shows a gain over R0L (section 6), otherwise the owner is told first.
4. **S2-5: a Stage 2 notebook** ([pipeline] option b, §2.1) at `stage2/stage2_training.ipynb`, outside `notebooks/`,
   which main-repository tests scan (`environments/shared/tests/test_record_fields.py:41`).
   - It reuses the scaffold's setup cell, also pinning numpy, scipy and torch by version, and its Drive mount, flushed
     disconnect, checkpoint-pair check and safe loader ([pipeline] §2.2).
   - It runs `designlab/stage1/train.py`, extended with R5 and R6, a picklable schedule, the pair check on resume, a
     flagged resume across library versions (as `--allow-source-change`), the G8b fix inside the running job, and
     fine-tunes from another run's checkpoint and normalisation statistics in a new run directory.
   - Inputs: the lead's sha256-manifested Drive bundle, as rebuilt models change every record key
     (`stage1/runs/2026-11-10-speed/ARM_K.md` §2); the smoke rebuilds them once, as a check.
   - 8 envs in every run and session, so PPO's batch and floor mix never depend on the runtime (`train.py` refuses a
     resume with other settings); the vCPUs set only the worker processes; a queue starts each next job.
   - Clips render in a guarded subprocess (the scaffold forces NVIDIA's EGL; the form study uses OSMesa).
   - Checkpoints: each 2.5M check's and the newest two, written locally, copied to Drive in pairs.

### C. Agility tests

Section 5's panels (`designlab/stage1/panels.py`) run here, from the main checkout, on judged checkpoints from Drive,
keeping Colab's floating point out of the verdicts ([pipeline] §4.3): the full set on judged R5 policies, the bar and
no-loss conditions at 10 ms on other rows, the bar and speed ladder for Track D; decision 20's attribution on gated
conditions with 5 or more falls.

### D. Hardware sensitivity (model only)

Levers: (1) hip-pitch and knee speed ×1.25 at the same torque and mass, then a real part (60 cm: the faster knee,
`60-N-T-10-K`); (2) an STS3250 tail servo (sustained 0.819 against 0.491 N·m; +0.02 kg, +$67); (3) bigger 60 cm feet
(F23) on a new narrow model, built as `-X` is; (4) joint damping and friction ×0.25 on the tail, neck yaw and jaw
([look] §5.2); (5) on `-X`, an STS3250 neck yaw at 75 cm and a neck-pitch spring at 60 cm; (6) thigh-twist range beyond
±0.30 rad, if no quick turn fits. Each gets the Stage 0 screen, then the judged R5 on the bar and speed ladder,
labelled untrained transfer (a lower bound). **S2-6:** a 5M-step fine-tune from the judged checkpoint, beside one on
the unchanged body, only where the screen raises a ceiling 10 % or more or the lever's limit is body-limiting by
decision 20's lift test (for the neck and tail, never lifted: causing most of a failing condition's misses).

## 5. Pass criteria (set before the runs)

- **Episodes:** 20 per condition, seeds 3042–3061, settled stance, nominal plant, commanded 5 % above (`panels.py`
  `EPISODE_RULE`), 10 ms (S2-7), hard floor, unless stated.
- **Readings,** each 20 of 20 up and at least 17 successes under both training seeds (provisional with one; two fits
  differed by 15–17 of 20, S1b Answers), in order: (1) **shown with the look,** the headline for walks and turns:
  successes also keep [look] §4.2's bands (bob within ±30 % of the reference's, head pitch in the world at most 1.5°
  RMS, tail phase within ±0.1 cycle, toe-off at least 70 % of the reference's); (2) **shown:** every limit
  (`panels.py:75-77`); (3) **shown but for A1:** every limit but a neck or tail servo's sustained planning limit (for
  the tail bench test and lever 4).
- **Every limit:** Stage 1's walk, drive and range limits (`LIMITS_RULE`; decision 14), with toe clearance (0.04 H)
  read over the middle half of each foot's own swing, lift-off to touchdown, Stage 1's reading beside it: the
  reference's clock (`metrics.py:283-291`) misreads a foot lifting 10 ms late (S1b N5). Duty 0.55–0.65 from 0.30 m/s
  up (0.40 m/s: about 0.57–0.62 with F3); below, 0.55–0.72 if the owner widens the band (owner decision 2), else F3
  starts at brisk.
- **S2-7: the realistic delay is 10 ms** until a bench value (draft A17; Stage 1b's test value; [look] §4.2). The bar
  also runs at 0, 5, 15 and 20 ms, reported, and re-runs here at the bench value, never read between delays (the 75 cm
  walk falls at 10 ms, not 20; S1 N4).
- **Floors:** every bar item at μ 0.9 and μ 0.54, each verdict naming its floors by the treads' unmeasured estimates
  (S1b §9): μ 0.9, tile and the grippiest wood; μ 0.54, every demo floor but the slipperiest vinyl and laminate (to
  about 0.42). robot_floor_carpet reported; no pile (P26).

**The bar: 16 gated conditions per body and head, at 10 ms,** each item at both grips.

| Item | Conditions | Measured |
|---|---|---|
| Walk at 0.4 m/s or more | 0.40 m/s | Start, the change to 0.40 m/s, then the gait, judged over its last 10 s (`WALK_RULE`): the CoM covers at least 4.0 m along the commanded heading, within 0.25 m of that line (new; drift went unjudged, S1b N3) |
| 90° turn while walking, about 2 s | from 0.40 m/s, left; right | The reference turns at π/4 rad/s (5 % fast) between its ramps to the start heading + 90°, itself reaching 80° by 2.0 s (G9; most 10 ms walking-turn falls came in the ramp, S1b N5); the trunk's heading is within 10° of start + 90° 2.0 s after the command and for 3 s of walking after; time to within 10° reported; walk limits, decision 14's range rule, no leg interference |
| Nudge while walking | at 0.40 m/s: forward, back, left, right | Stage 1's 0.1 s pulse at the trunk's CoM (`PUSH_RULE`) at 0.25 × the Stage 0 forward bound, the edge Stage 1's tracker recovered (S1 §3): Δv 0.12 / 0.15 m/s, about 13 / 22 N, at a uniform phase; no fall, back on the command within 3 s (decision 30) |
| Stop within a step or two | from 0.40 m/s | at most two touchdowns (`metrics.py`'s 2 % load rule) after the stop command before the forward speed stays under 5 % of the command (decision 18); deceleration (1 Hz) at most 1.5 m/s² (decision 10) |

- **Ladders (Q1),** each read as its highest rung shown at 10 ms under both seeds, at μ 0.9: speed 0.30–0.50 m/s in
  0.05 steps, each rung whose reference passes the Stage 0 screen (0.45 / 0.50 m/s stand in for Froude 0.05), also at
  0 and 5 ms; turn rate from 0.40 m/s at 0.35, 0.5, π/4 and 1.0 rad/s, within the thigh twist's range (A.3); walking
  nudges at 0.25, 0.375 and 0.5 × the bound.
- **No loss:** Stage 1's 12 gated hard-floor conditions (S1 §2) at 0 ms, where a loss is a trade (`STAGE1B_PLAN.md`
  §4); at 10 ms, reported beside R2L's (Stage 1b ran none there, S1b §9).
- **Reported:** the turn and nudges from the nominal walk (the quick turn there needs about 1.1–1.4 of the thigh
  twist's range, linearly from F11, so its reference takes quicker steps); carpet walks; stiffness ×0.7; R0L, R2L, R6;
  falls attributed (decision 20).
- **Look gates:** L1 (A.5); L1b, the owner's review of closed-loop clips at 10 ms before the report; L2, a rating
  nothing waits on: the spec's blind rating (BEHAVIOR_SPEC §1) of such clips by at least 12 raters new to the project,
  with [look] §4.1's rubric and Godspeed items. A clip's score is its rubric mean over raters (95 % interval); the pass
  is a walking-clip mean of at least 3.5/5 and, per clip, an upper 95 % bound under 0.5 on its shortfall from its
  reference render (paired by rater).
- **The head's cost** (set now), per body, paired by episode seed over both training seeds: verdicts in each reading,
  achieved speed per speed rung, closed-loop hip-pitch, knee and neck-pitch loads; `-X` runs the bar head neutral (the
  hardware's cost) and with gaze events and head-first turns (the cost of using it). A difference inside the current
  head's seed-to-seed spread reads "below what this round resolves".

## 6. Compute and budget

- **This CPU** (4 cores, 15 GB): about 80 core-hours, estimated: panels and attribution 45 (about 3 per judged R5
  policy, from Stage 1's 0.75 per 28 conditions; 0.8 per other row), validation 10, R2L 8 (each fit twice Stage 1b's
  1.6, S1b N10), references and transitions 6, Track A 5, models 3, Track D 1. That is a day of all four cores over the
  phases, after about 4–5 days of development (look generator, `-X`, R5, the bar's measures, the notebook) that
  overlap L1.
- **Day-one Colab smoke** (the owner, about an hour, on the chosen runtime): vCPUs and memory per worker, the rate at
  8 envs, a test clip, a kill and resume on Drive, an archive that loads here, the inputs rebuilt once against the
  bundle, a launch's setup time.
- **The notebook** runs a queue of jobs (body, head, seed, mode) per runtime. The owner opens it from the branch, sets
  REPO_REF, the queue and RUN_ID, runs all and checks four printed lines (commit, inputs' hashes, G8a, first rate);
  after a disconnect, the same with RESUME on; a steps sheet guides it, as for the stance runs
  (`docs/investigations/stance_2026_10/stance3/colab_steps.txt`). It saves to
  `MyDrive/mesozoic-labs/stage2/runs/<run_id>/`, outside `logs/`: SESSION files (runtime type, vCPUs, memory,
  versions, commit), STATUS.json, `run.json`, kept checkpoints, logs, a clip per check, DONE or HALT ([pipeline] §4.2),
  which the lead reads with the Drive tools.
- **Runtime:** stepping is CPU-bound (an L4 runtime trained the T. rex stance no faster than a CPU one, [pipeline]
  §3.2): High-RAM CPU first (8 vCPUs by a third-party table of October 2024, unverified; standard CPU 2, L4 12), L4
  only if High-RAM is not offered.
- **S2-8, the budget,** fixed once after the smoke for every run: 40M env steps (16 times Stage 1's 2.5M), or fewer if
  the smoke's rate cannot reach 40M in 20 h (about 560 env steps/s), then rounded down to 5M steps; never under 15M
  without the owner. Runtime-hours per run (steps over rate, plus setup) are then tabled per runtime type; Stage 1's
  rates do not carry over ([pipeline] §3.3).
- **Runs:** R5, 2 bodies × 2 heads × seeds 42 and 43 = 8; R6, 2; at most 8 fine-tunes of 5M; at most about 230
  runtime-hours if each run takes 20 h. Phases: R5 seed 42, R6 in spare slots; seed 43; the fine-tunes. About 13
  launches over about 6 days with 3 runtimes at once (5 with 4); the 8-day cap assumes at least 3.
- **Cost** (the same table; 100 compute units, CU, cost about $10): High-RAM CPU 0.14 CU per runtime-hour, L4 3.0, so
  the 220-hour cap is about 31 CU (about $3) or 660 (about $66). The owner confirms that the plan allows 24 h sessions,
  background execution, High-RAM and 3 runtimes at once.
- **Stop rules.** Monitor: the nominal and 0.40 m/s walks, a 90° turn, the stop and two nudges, 3 episodes each at
  10 ms every 2.5M steps, with the look measures. A **gain**: at least 2 more of its 18 episodes succeeding with the
  look than at the best earlier check, or mean survival 1 s longer.
  - G8a fails: no run.
  - G8b at 5M: the nominal walk stays up in 2 of 3; else the one fix, set now and applied in the running job: the
    failing monitor tasks' training weights ×2 (decision 32's kind), imitation weights unchanged; a second failure 5M
    later stops the run.
  - From 20M, three checks without a gain stop the run.
  - No best-model selection; a body and seed's two heads are compared at the step both reached, each run's upper end
    at its own last checkpoint.
  - Cap: 220 runtime-hours or 8 days; then every run is judged at the last checkpoint its comparison reached
    (`STAGE1_PLAN.md` §6), and unstarted jobs are cut in this order: fine-tunes, R6, `-X` seed 43.
  - Freed hours go to fine-tunes, then third seeds for the head comparison. If R5 fails G8b on both seeds of a body,
    that body is reported with R2L (the bar at 0, 5 and 10 ms) and the owner told before more Colab hours are spent.
  - A rule changed after a result is labelled so (S1 §9).

## 7. Deliverables and order

1. **First, the look review:** kinematic clips and sheets, new viewers' scores, the tail-phase check, the quick turns'
   range and grip needs, the `-X` model and its costs.
2. After L1: references, models, gates; R2L; validation and S2-4; the notebook, steps sheet and smoke; Colab's three
   phases, each with a short status note.
3. The report: the bar per body and head (three readings, the look first, with their floors), the ladders' upper ends,
   the head's cost, Track D, ablations, budget, draft decision-log rows; clips; L1b; L2 once raters are found.

**In brief, for the owner:** an hour's smoke session, then about 13 launches over about 6 days with 3 runtimes at once,
each checked by four printed lines; about $3 of compute on High-RAM CPU ($66 on L4); results in your Drive. Nothing is
adopted without you.

**Owner decisions** (3–6 taken on 11 Nov, P29; 1, 2 and 7 at the look review; 8 later):
1. The look features (L1), and whether a feature may cost a speed rung (A.3).
2. Duty: a fixed swing time puts the nominal walk at about 0.69 ([look] F3), over the spec's 0.65: widen the band to
   0.72 below 0.30 m/s, or start F3 at brisk?
3. **Taken:** the 60 cm Hybrid is dropped; Colab trains `60-N-T-10` and `75-M-H-10` (S2-1).
4. **Taken:** the expressive head as modelled (S2-2), in simulation only, jaw included (P2's deferral stands for the
   build).
5. **Taken:** the bar's definitions as in section 5: the 90° turn within 2.0 s, both grips, judged at 10 ms (S2-7).
6. **Taken:** Colab on High-RAM CPU runtimes, up to 3 at once, 24 h sessions, capped at 220 runtime-hours (section 6).
7. Raters for L2, 3–5 new viewers for L1, and a private rating page.
8. When the bench tests below run.

## 8. Risks and what this round will not show

- **Simulation only:** ideal PD servos behind an envelope and a delay; no backlash, heat beyond A1 or fall damage; joint
  friction, damping and the carpet unsourced (A16); grip estimated (F22); the pad-edge flag untestable (S1 N1).
  Randomisation narrows the gap to hardware; it does not close it.
- **Learning may fall short:** Disney's character policies train about 22 GPU-days each ([look] §2). "Not learned" says
  nothing about the body; R2L's verdicts then stand.
- **Colab:** vCPUs unknown until the smoke; a changed image refuses a resume unless flagged (`train.py`); Drive writes
  can be cut short; rendering on a CPU runtime untested until the smoke.
- **The quick turn** may exceed the thigh twist's range, unassessed for packaging beyond ±0.30 rad (S1 §8, G5).
- **Not shown:** hardware; the real delay and stiffness; a carpet verdict; autonomy (S1 §8, G12); sound; session heat;
  sizes above 75 cm; any adoption (P24).
- **Bench tests outside the round:** each drive type's delay and stiffness (draft A17); the STS3215 tail at the speed
  walks' loads, with its joint friction (A1; S1b §8); tread grip on the owner's floors (P21).

*Research notes:* [look] `stage2/research/research_look.md`; [pipeline] `stage2/research/research_pipeline.md` (10 Nov;
their scratch scripts were not kept).
