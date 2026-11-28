# Pre-registration: the two strongest 50-80 cm candidates, tested further (experiments E1-E5)

28 Nov 2026, branch `claude/pensive-pasteur-o3bqtv` (head `fb1e80c8` when written). **Written before any experiment
ran.** Desk and simulation only: nothing is built, bought, weighed or measured, and nothing here is adopted. Every
design change named here (a stop, a reduction, a transmission, a fan, trot settings, a size) is the owner's decision.
0.40 m/s is not a walking target and appears nowhere below.

Evidence labels used throughout: **DESK** (geometry, arithmetic, the build's thermal model on stored or simulated
torques), **SIMULATION** (MuJoCo 3.10.0 runs of the build's env, `rl/trike_env.py`), **MEASURED** (none). Every number
carries its source and units. Any choice made after a result is seen will be labelled **"changed after a result"**,
with this file's wording kept beside it.

## 0. What is frozen, and what was checked before writing

| File | sha256 (first 16) | Role |
|---|---|---|
| `sim/candidate_checks.py` | `008aa308debdf1b6` | the shared code every experiment calls (logging, clash geometry, stops, desk heat and speed math, the series-stiffness model variants and the timestep-only baselines, the speed cap, the policy episode with its 10 ms delay at any step, the per-policy heat table, the trot's body-pitch setpoint, the E5 settings and rules, provenance) |
| `sim/tests/test_candidate_checks.py` | `9662e0a6cba7a679` | 45 unit tests on small cases (no experiment) |
| `sim/results/candidates_2026-11-28/plan.json` | `1403a4b326aaf2aa` | the frozen settings, written by `python sim/candidate_checks.py plan`: seeds, conditions, the 17 candidate stops, stiffnesses (the belt's one-span case too), the belt pretension needed, efficiencies (judged and supporting), the E2(b) variants with their builders, steps and delays, sources, the 66 E5 stage-1 settings, the checkpoints' hashes, and the code's provenance (commit and sha256 of every file below) |
| `sim/generated/trike60_C_closed_capsule_fore128.xml` | `372ac07b14338c26` | the adopted 60 cm model (read only; every variant is written to the scratch folder) |
| `rl/trike_env.py`, `walk/walk60.py`, `rl/robustness/sweep.py`, `sim/check_packaging60.py`, `design/form_factor_study.py`, `rl/telemetry/thermal.py` | `febfe2ba23a835d3`, `5225c64a2006d8b4`, `7b1c8ca0b1b2634f`, `d5494c537e107489`, `4679079486ecd6bc`, `c368079ddf88d894` | the build's tools, imported and never edited (the 10 other build files the code imports or reads, from `rl/analysis/gait_pipe1.py` to `sim/data/drives.json`, are hashed in plan.json `provenance`) |

Scratch (raw logs, variant models, per-episode files): `/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/candidates/`.
Committed results (summaries only, JSON / CSV / small Markdown, each under 500 KB): `sim/results/candidates_2026-11-28/`.
The report will be `design/CANDIDATES_2026_11_28.md`.

**Checks of the harness, run before this was written (they test the code, not any hypothesis):**
- **Checkpoints (DESK, file hashes).** All 11 policies' final pairs (fore128-s0..s3, fore160-s0..s3, run3-s0, run3-s1,
  run2b) match, zip and VecNormalize pickle, the sha256 their stored sweeps record
  (`rl/robustness/results/<run>/sweep_<run>.json.gz`): "checkpoints verified 11 of 11" (`plan` command, 28 Nov 2026).
  They are loaded only through the sweep's loaders (`gait_pipe1.load_obs_stats`, a restricted unpickler;
  `gait_pipe1.build_policy`, `torch.load(weights_only=True)`), via `candidate_checks.load_policy`, after that check.
- **Unit tests:** 45 passed (the test file, run with `-p no:cacheprovider`, 28 Nov 2026, after the fixes of 0.1).
  They include: the clash geometry reproduces the packaging record's #10 (-8.49 mm), #9 soles (-48.0 mm) and #9
  struts (-19.2 mm, soles aside), and the -17 deg roll stop's 4.79 mm; the bound prefilter equals a brute-force
  `mj_geomDistance` over all part pairs; the logger reproduces `rl/demo_course.py`'s drive-side rule to 1e-12; the desk
  heat path reproduces the study's policy s3 (98.3 C, 0.914) and s1 (74.4 C, 0.746), s3's mirror (0.509) and even
  split (0.740, 73.4 C); the supporting rod at 0.90 and r = 1 at the knees gives today's figures exactly; the series
  spring's force is -k x deflection; the speed cap's envelope; the trot with a 0 deg pitch setpoint gives the frozen
  controller's actions bit for bit; E5's rule on synthetic episodes. Added with the fixes: B025 differs from the
  adopted model only in its timestep (the XML text in that one attribute; every other field of the compiled model,
  over 200 arrays and scalars, equal, force limits and armatures included); `policy_env_config` refuses a model whose
  own timestep is not the step asked for; `policy_episode` gives a 10 ms delay at 1 ms (10 steps) and 0.25 ms (40
  steps); an "unstable_simulation" episode is not a fall and leaves the comparison not judged; the verdict's three
  values; item 4's choice of step and baselines; the speed check on the reduced drives only; the belt's one-span
  stiffness is half and its pretension; `forced_air_c1` refuses a heat path other than winding-to-ambient or total;
  every result's provenance.
- **Integration step of the series variants (SIMULATION, zero action, 0.5 s standing, fore128-s0's env config).** At
  the env's own 1 ms step the rod variant ended "unstable_simulation" in its first control step and the cable variant
  in its second (MuJoCo "Nan, Inf or huge value in QACC"); the belt variant stood. At 0.25 ms (frame_skip 80, so the
  control stays 50 Hz) all three and the rigid variant stood 0.5 s with no termination (largest spring deflection
  0.15 / 3.4 / 0.57 mrad for rod / belt / cable). This is why E2(b) runs at 0.25 ms (section 3, item 4).

### 0.1 Fixes after an independent check, before any experiment ran (28 Nov 2026)

An independent checker reviewed this file and the code before any experiment ran and reported six findings. All six
were checked and accepted; none is "changed after a result", since no result existed. The earlier hashes were
`f71e347cb9d31e50` (code), `6a6ae94b5b93dfa4` (32 tests) and `62cf89eec9f4b5a1` (plan.json).

| Finding | What was wrong | What changed |
|---|---|---|
| F1 (high) | The code could not build E2(b)'s judged baseline B025 unchanged. `write_variant(transmission=None, r=1.0)` still rescaled every hip-pitch and knee force limit by 0.95 / today's eta (reproduced here: left_fore_pitch 21.0 -> 19.95 N m, left_fore_flexion 37.8 -> 39.9, left_hind_flexion 18.9 -> 19.95). `policy_env_config(..., MODEL, physics_dt=0.00025)` set frame_skip 80 on the 1 ms model, i.e. an 80 ms control step (reproduced: physics_dt 0.001, frame_skip 80, dt 0.08) | B025 (and B0125, item 4) are built by `write_timestep_copy`: a text edit of the one `timestep` attribute, every other byte kept; tested against the adopted model text and field by field after compiling. `write_variant` refuses no transmission with r = 1. `policy_env_config` refuses a model whose own timestep differs from the step asked for. Every variant's model comes from `e2b_model(name)` |
| F2 (medium) | The handling of an "unstable_simulation" episode was open (item 4 said rerun at 0.125 ms on the same seeds; section 6 said new seeds), and `e2b_compare` would count such an episode as a new fall (`sweep.run_episode` marks every termination as a fall) | Settled in item 4: such an episode is not a fall, is flagged, and leaves that comparison not judged (`e2b_compare`: `pass` None); which run is judged is fixed (`e2b_judged_runs`: the variant rerun at 0.125 ms on the same seeds against B1 and B0125, the adopted model at 0.125 ms); section 6's new-seed rule does not apply to it. The verdict has three values (`e2b_verdict`) |
| F3 (medium) | The rod was judged at the build's own 0.90, which has no source (`sim/data/drives.json` `transmission_efficiency`, no source field, added in dc2e405d, 18 Nov 2026), although the lead's rule says 0.95 when there is no dated source, and 0.90 is not uniformly stricter under the build's braking convention | The rod is judged at 0.95 in E2(a) and E2(b). 0.90 is reported beside it, supporting: in E2(a) as 15 more rows (its r = 1 knee row is today's linkage, the control row); in E2(b) by re-mapping the judged rod run's torques to the drive at 0.90 (`relog_drive_sq`) |
| F4 (low) | The runner code was not in the frozen file: `sweep.run_episode` passes the point's `delay_ms` straight to the env as physics steps (a 2.5 ms delay at 0.25 ms by default), and `speed_check` tests every drive it is given | `policy_point` / `policy_episode` (E1 and E2(b) policy episodes through `sweep.run_episode`, delay `delay_steps_for(step)`: 10 / 40 / 80 steps at 1 / 0.25 / 0.125 ms; the trot episodes too); `e2a_speed_check` slices to `reduced_joints(placement)`; `provenance` records the commit and the sha256 of the code, the runner script and the build files in every result file (section 6) |
| F5 (low) | The belt's stiffness assumes both spans stay taut, which needs a pretension of at least half the effective tension; none is stated or sourced | No sourced installation tension was found. The pretension needed is computed (`belt_pretension`, DESK: 385 N per span on the X4 drives, 770 N on the X6 fore knees, at the belt variant's joint-side limits), and a supporting one-span belt (half the stiffness, `belt_1span`) is run on the same seeds; the "upper bound" wording is kept |
| F6 (low) | E4's acceptance did not say which heat path a sourced R_th must cover; the build's model is one node, winding to ambient, and `forced_air_c1` scales that whole R_th | Only a winding-to-ambient (or total) R_th pair counts (item 12); `forced_air_c1` requires `heat_path` and refuses any other |

The checker also recomputed the stiffness figures of section 4 (they agree with this file's) and gave coupled
rotor-and-leg mode frequencies from the variants' mass matrices (rod 481-546 Hz, belt 99-112 Hz, cable 246-279 Hz;
omega x dt at most 0.86 at 0.25 ms): the checker's DESK figures, not recomputed here, context only. The "spring
frequency on the rotor" in the stiffness table below holds the leg fixed and is labelled so.

## 1. The candidates (the owner's request of 28 Nov 2026, as the study's lead framed it)

- **(A) 60 cm, today's drives** (10 MyActuator X4-36-L and 2 X6-60, 12S), with a policy chosen or trained for heat, the
  packaging fixes of trial layout C (`design/PACKAGING_PUSHROD_60CM.md`, section 1.5), joint stops for clashes #9
  (front and hind legs of one side meeting under the body) and #10 (the hind soles at the inward roll stop), and
  optionally a small hip-pitch / knee reduction through a rod or belt with the drives moved to the hip or body.
- **(B) 50 cm, today's drives, layout L1** (`design/FORM_FACTOR_STUDY_50_80CM.md`, section 1.2 row 1).
- **The cable question.** Would cable-driven legs help? The lead's view, to be tested by E2: drives moved off the legs
  help packaging and cooling, and a 1.1-1.3:1 reduction helps heat within the 83 rpm drive limit, while cables add
  stretch and upkeep.

E1-E4 are on the 60 cm adopted model and bear on A. E5 runs at both sizes and is the only experiment here that tests B
directly. Nothing below adopts a candidate.

## 2. The rules, fixed by the lead on 28 Nov 2026, before any experiment ran (verbatim)

> RULES FIXED BY THE LEAD ON 28 NOV 2026, BEFORE ANY EXPERIMENT RAN (the pre-registration may add operational detail,
> seeds and episode counts, but must not loosen any of these; if one cannot be run as written, say so in the
> pre-registration before running):
> E1 Clash reach. Log joint angles every 10 ms for the four fore128 policies (0.25 and 0.15 m/s walking and standing,
> as the demo loop; the sweep's sideways pushes of 0.5 and 1.0 m/s; 2 seeds x 4 episodes each per condition) and for
> the scripted trot (PREREG_WALK60's J1-J4 conditions, seeds 101-104, plus 0.5 m/s pushes back and to the right) on
> the adopted 60 cm model. For every logged pose compute the clearance of clash pairs #9 and #10 with
> sim/check_packaging60.py's geometry. #9 (or #10) is REACHED if any logged pose overlaps (clearance < 0) for at least
> two of the five controllers (four policies and the trot). For each candidate stop in the packaging record, a stop is
> ACCEPTABLE if over every logged sample it would keep clearance >= 5 mm and would clip (change) at most 1 % of
> samples, for each of the five controllers.
> E2 Transmission and speed cap. (a) Desk: for reductions r = 1.0, 1.1, 1.2, 1.25 and 1.3 at hip pitch and at the
> knees (separately and together), the drive torque is joint torque / r (with a stated transmission efficiency from a
> dated source, 0.95 if none), the drive speed joint speed x r; recompute C1 (hottest winding <= 90 C at 35 C, the
> build's thermal model) and C2 (RMS <= 0.8 of continuous) on the stored demo torques of the four fore128 policies,
> and the speed check: the drive's 99th-percentile speed <= 83 rpm and its peak <= 1.1 x 83 rpm (from stored or new
> logs). (b) Simulation: add a series stiffness between each reduced drive and its joint for three transmissions (a
> rigid rod, a toothed belt, a cable; torsional stiffness from dated sources for a stated pulley or crank radius),
> with the drive's speed limited to 83 rpm, and run the four fore128 policies at 0.25 m/s and with 0.5 m/s sideways
> pushes, 2 seeds x 4 episodes per condition. A transmission is COMPATIBLE WITH TODAY'S POLICIES if, for at least 3 of
> the 4 policies, against the same policy and seeds with no added stiffness and no speed limit: no new falls, mean
> speed within 0.02 m/s, and worst-drive RMS over continuous (at the drive, after the reduction) not worse by more
> than 0.03.
> E3 Per-policy heat (desk). For every available policy (fore128 x4, fore160 x4, run3 x2, run2b if present) give each
> drive's demo-loop RMS over continuous and hottest winding at 35 C, the deciding drive and its mirror, and the
> even-split figure. A policy MEETS THE HEAT RULES AT 60 CM if its own C1 and C2 pass at 35 C with today's drives
> (worst drive, the build's ratings). Report which do. This is a per-policy reading, not the study's worst-of-four
> criterion.
> E4 Forced air. Search for dated, sourced figures for the thermal resistance of similar enclosed or open BLDC robot
> drives with and without forced air (a fan or blower at a stated flow). Compute C1 at 35 C with a forced-air R_th
> ONLY if a source gives one for a comparable drive; otherwise report that no usable figure was found.
> E5 Retuned trot. Tune the scripted trot's parameters (stride, period, duty factor, body pitch, knee lift; no new
> controller structure) on the adopted 60 cm model and on the 50 cm scaled model (build_scaled_trike, adopted gains)
> with tuning seeds 101-110 only, to avoid the slow gait (walk/fore128_slow_gait.py's rule) in J1-J4 and after
> 0.3-0.5 m/s back and right pushes. Then judge the chosen setting ONCE on held-out seeds 301-310 and 401-410. The
> retune SUCCEEDS at a size if on both held-out seed sets: no falls in J1-J4, J3 mean speed >= 0.225 m/s (90 % of
> 0.25), slow-gait episodes <= 1 of 10 per condition, and worst hind knee RMS <= 1.0 of continuous. Tuning budget: at
> most 60 candidate settings per size, logged.

Also binding (the task's general rules): a verdict needs the pre-registered rule, **at least two seeds** and, where
the rule says so, two controllers; simulation, desk and measured are kept apart; no training run is started (training
is the owner's, on Colab); at most 2 worker processes.

## 3. Where a rule cannot be run exactly as written, and the reading used (all decided before any run)

1. **"2 seeds x 4 episodes" (E1, E2(b)).** Read as two evaluation seed blocks of four episodes: block A episode seeds
   700000-700003, block B 710000-710003, in every condition. The "seeds" are evaluation seeds; the four policies are
   the four training seeds. A verdict needs both blocks.
2. **"Sideways pushes of 0.5 and 1.0 m/s" (E1) and "0.5 m/s sideways pushes" (E2(b)).** Both directions are run as
   separate conditions (+ left, - right), since #9 and #10 are side-dependent. The push is the sweep's: at 8 s, for
   0.1 s, impulse = robot mass x dv along the chassis's left axis at onset (`rl/robustness/sweep.py` `Pusher`).
3. **"The stored demo torques" (E2(a)).** The stored demo courses (`rl/analysis/results/d22/demo/fore128-s*/`) keep per
   segment each drive's mean squared drive-side torque only (`tau_drive_sq_mean`: tau / 0.90 when motoring, tau x
   0.90 when braking at the knees' 1:1 linkages, tau at the hip). Mapping it through a new transmission needs each
   joint's motoring / braking split. It is taken from E1's re-run of the same episodes (block A contains the stored
   demo's three segments exactly: walk 0.25 m/s seed 700000, walk 0.15 seed 700001, stand seed 700002, same env
   config). It is used ("exact") only if the re-run reproduces the stored `tau_drive_sq_mean` within 1e-6 relative on
   every joint; otherwise the conservative bound is used (every sample at the worse of the motoring and braking
   factors), with the optimistic bound beside it (`reduced_profile`). The "at least two seeds" rule needs a second demo
   course: block B's seed0 710000 (new SIMULATION, `rl/demo_course.py` `run` through the sweep's loaders), whose split
   comes from E1's block B episodes, the same episodes.
4. **The rod's stiffness in simulation (E2(b)).** At the env's 1 ms step the rod and cable variants are numerically
   unstable (section 0). All E2(b) variants therefore run at a 0.25 ms physics step with frame_skip 80 (50 Hz control,
   as trained) and a 40-step (10 ms) servo delay (`policy_episode`). "Against the same policy and seeds with no added
   stiffness and no speed limit" is then judged against **both** baselines: B1, the adopted model as trained (1 ms),
   and B025, the same model with only its timestep changed to 0.25 ms (`write_timestep_copy`: force limits,
   armatures and every other field unchanged; section 0.1, F1). A transmission must pass against both (stricter than
   either alone).
   **"unstable_simulation" episodes (fixed before any run; section 0.1, F2).**
   (a) An E2(b) episode that ends "unstable_simulation" (`rl/trike_env.py`: a non-finite state or MuJoCo's
   huge-acceleration warning) is a harness failure, not a fall: in `e2b_compare` its seed is left out of the falls and
   of the pairing (whichever run it is in), it is listed and flagged, and that comparison is not judged (`pass` None).
   (b) The judged run (`e2b_judged_runs`): every run of the start set (B1, B025, R, rod, belt, cable, belt_1span) is run
   once at its step. A variant with at least one such episode at 0.25 ms (or every variant, if B025 has one) is rerun
   at 0.125 ms (frame_skip 160, delay 80 steps), all four policies, all three conditions, **the same seeds**
   700000-700003 and 710000-710003, and B0125 (the adopted model with only its timestep changed to 0.125 ms) is run
   once on the same seeds; that variant is then judged against B1 and B0125, so the comparison stays at one step size.
   Its 0.25 ms run is reported beside it, supporting. B1 (the plant as trained) is never rerun. No smaller step is
   tried: a comparison still holding such an episode at 0.125 ms stays not judged.
   (c) Section 6's new-seed rerun rule does not apply to these episodes: they are an outcome this item foresees and
   handles on the same seeds, not a fault found after a run.
   (d) The verdict (`e2b_verdict`) is three-valued: a policy fails if any of its comparisons fails, is not judged if
   none fails and one is not judged, and passes otherwise; the transmission is COMPATIBLE if at least 3 of the 4
   policies pass, NOT COMPATIBLE once 2 or more fail, and NOT JUDGED otherwise (never reported as compatible).
   E1, E3 and E5 run unchanged models at the build's own 1 ms step, as every earlier report did: there an
   "unstable_simulation" ending is counted as the build's tools count it (a fall) and flagged in the output, and any
   logged pose with a non-finite angle is dropped from E1's clearances and counted (`StepLogger`).
5. **E2(b)'s reduction.** The rule simulates "each reduced drive" without naming r. Fixed here: **r = 1.25 at hip pitch
   and at the knees together** (all 8 drives; the middle of the lead's 1.1-1.3), each transmission with its own
   efficiency (section 5). A supporting variant R (reduced, rigid, capped; eta 0.95) separates the cap and the
   reduction from the compliance; it is not judged. A second supporting variant, belt_1span, is the belt with one
   span carrying the load (half its stiffness: the slack span unloads when the pretension is under half the
   effective tension, and no sourced installation tension was found); it is not judged (section 0.1, F5).
6. **The 83 rpm cap on the X6 (E2).** The rule's 83 rpm is the X4's cap above 24 V (`DECISION_LOG.md` E1). It is
   applied to every reduced drive, the fore knees' X6-60 too, which is stricter than the X6's own 108 rpm allowance
   (`rl/analysis/gait_pipe1.py` `RPM_ALLOW`: 0.70 x 176 rpm x 42 / 48); the 108 rpm figure is shown beside it.
7. **E3's verdict on two seeds.** Each policy's verdict needs C1 and C2 to pass on two demo courses: the stored one
   (seed0 700000) where it exists (fore128, fore160), and a new one at seed0 710000; run3-s0, run3-s1 and run2b have no
   stored demo, so both of theirs are new (700000 and 710000). run3 is on `trike60_C_closed_capsule.xml` (188 mm
   foreleg) and run2b on `trike60_C_closed_ankleroll.xml` (box foot, passive roll ankle): each is read on its own
   model, not the adopted one, and labelled so.
8. **E5's "stride".** In `walk/walk60.py` the stance stroke is the commanded speed x duty x period, so the stride is not
   a separate parameter. The stride parameter tuned is `max_reach_m` (the touchdown's reach limit); period and duty
   set the stroke.
9. **E5's "body pitch".** `WalkParams` has no body pitch. It is set by `PitchedTrot`: the trot's gravity reading is
   turned by -theta about the body's y axis (so its existing pitch feedback, `k_pitch`, holds theta instead of level),
   and each foot target is raised by x tan(theta) (x its fore-aft position in the chassis frame), so the feet stay on
   level ground at that pitch. It adds no loop and no state: a setpoint and a feed-forward offset on two existing
   paths. At theta = 0 the actions equal the frozen controller's bit for bit (tested). **Flagged for the lead:** if
   this is read as new controller structure, body pitch is held at 0 and the same stage-1 points are run with that
   coordinate set to 0; that decision must be made before the first E5 run.
10. **E5's 50 cm model.** `sim/build_scaled_trike.py build_xml(0.50, "D0", "adopted", layout="L1")`: today's drives,
    the adopted A2 servo gains unchanged, **layout L1** (candidate B's layout), written to the scratch folder. The
    command is 0.25 m/s at both sizes (the rule's J3 line is 90 % of 0.25), not the Froude-scaled 0.228 m/s.
11. **E5's slow-gait rule.** `walk/fore128_slow_gait.py` counts "slow" (10-20 s speed under 0.225 m/s) and "slow and
    hot" (also the larger hind knee at or over 1.0 of its 6.75 N m rating over 10-20 s), and names the second "the
    slow gait". That is the rule used (`is_slow_gait`); "slow" is reported beside it. A fall counts as a slow-gait
    episode in every condition (stricter). "Worst hind knee RMS" is the larger hind knee's RMS / 6.75 N m, the worse of
    the 2-20 s and 10-20 s windows, in every episode of every condition (J1-J4 and the six pushes).
12. **E4's "a source gives one".** A thermal resistance (K/W or C/W) given for the same drive, or a comparable one,
    both without and with forced air at a stated flow, **for the winding-to-ambient path (or the drive's total
    resistance to the air)**. The build's thermal model is one node, the winding against the ambient air
    (`rl/telemetry/thermal.py`), and `forced_air_c1` scales that whole resistance by rho = R_forced / R_passive, so a
    housing-to-air, frame-to-air or heat-sink pair does not count: the winding-to-housing part does not change with the
    airflow, and scaling the whole R_th by it would overstate a fan (the code refuses any `heat_path` but
    "winding_to_ambient" or "total"; section 0.1, F6). Such a pair, or a pair of continuous ratings with and without a
    fan, is reported if found, but not converted into an R_th for C1 (a derived figure is not "given"); with no
    qualifying pair the result is "no usable figure was found".
13. **E2(a)'s speed check, "the drive's ... speed".** Tested on the reduced drives of each placement only
    (`e2a_speed_check`: the four hip pitch drives, the four knees, or all eight); the unreduced drives, the hip roll
    among them, are left out, since a reduction does not change their speed (section 0.1, F4).
14. **The rod's efficiency.** No dated source was found for a rod-end linkage, so the rule's 0.95 is used and judged
    (E2(a), E2(b)). The build's own 0.90 (`sim/data/drives.json`, no source field) is reported beside it as supporting
    (section 0.1, F3). Today's knees keep the build's 0.90 in every baseline figure (the stored demo courses, B1, B025,
    E3), as the build's tools compute them.

## 4. The experiments

### E1 Clash reach (SIMULATION logs; DESK geometry)

**Controllers and conditions** (`candidate_checks`: `POLICY_CONDITIONS`, `E1_BLOCKS`, `TROT_PUSHES`).
- The four fore128 policies, each on its run record's config through the sweep's measuring env (`policy_env_config`:
  flat floor, randomisation off, observation noise on, 10 ms delay, 20 s episodes, the adopted model at 1 ms):
  walk 0.25 m/s, walk 0.15 m/s, stand (0 m/s), push left 0.5, push right 0.5, push left 1.0, push right 1.0 m/s (the
  pushes at the 0.25 m/s command). Seeds 700000-700003 and 710000-710003 in every condition: 7 x 8 = 56 episodes per
  policy, 224 in all. Each episode through `policy_episode` (`sweep.run_episode` at `policy_point`, the 10 ms delay
  as `delay_steps_for` the step; the sweep's own record kept).
- The frozen trot (`walk60.TrotController(WalkParams())`, `PitchedTrot` at 0 deg): PREREG_WALK60's J1-J4
  (`judge_walk60.JUDGED`: randomisation and noise on, friction 1.0 / 0.4, impratio 1 / 10, 0.25 m/s, 10 ms), seeds
  101-104, as `fore128_walk60.judged_episode` runs them (`trot_judged_episode`); and pushes back 0.5 m/s (along the
  chassis's forward axis at onset, - = back) and right 0.5 m/s on the sweep's nominal plant (`trot_push_episode`),
  seeds 101-104: 6 x 4 = 24 episodes.

**Logged** (`StepLogger`, every physics step): the 16 leg joints' angles (roll, pitch, flexion, ankle of each leg)
every 10 ms from the first 10 ms to the episode's end (fallen episodes up to their termination); from 2 s on, each
drive's motoring and braking mean squared torque, drive-side mean square and a 0.1 rpm speed histogram with its exact
peak (for E2(a) and E3).

**Computed** (`ClashGeometry`, DESK, with `check_packaging60.py`'s parts and calls): for every logged pose, the
clearance of #9 on each side (all solid parts of the front leg against all of the hind leg, limb covers left out, the
full capsule soles in, as the record's "solid parts (no covers)") and of #10 (the two hind legs, the same parts), and
each with the soles aside (supporting: the real foot is a half-capsule rocker, F17).

**Rules** (`e1_verdict`, `evaluate_stop`, `coupled_limit`).
- #9 / #10 **REACHED** if at least two of the five controllers have a logged pose with clearance < 0 (#9: either
  side).
- Each candidate stop is applied to every logged sample (the limited joints clipped to the stop on both sides; a
  sample is "clipped" if any of its limited joints changes) and the addressed clash's clearance recomputed for the
  clipped samples. **ACCEPTABLE** if, for each of the five controllers separately, the least clearance over all its
  samples is >= 5 mm and at most 1 % of its samples are clipped. This tests the logged poses as they are; a gait
  with a stop in place would adapt, which is not simulated here.
- The candidate stops (all 17 of `check_packaging60.LIMITS`, which the record's items 9 and 10 list), with the record's
  least clearance over the full narrowed ranges (DESK, `sim/results/packaging60_2026-11-27/packaging60.json`; context,
  not part of the rule):

| Clash | Stop (applied to both sides) | Record's least over the narrowed range, mm |
|---|---|---|
| #10 | hind inward roll >= -17 / -16 / -14 deg (today -18) | 4.79 / 18.11 / 44.82 |
| #9 | fore pitch <= +20 / +10 / +5 / 0 deg (aft stop; today +38) | -48.00 / -25.50 / 0.53 / 26.38 |
| #9 | hind pitch >= -30 / -20 / -15 / -10 deg (forward stop; today -38) | -48.00 / -42.08 / -11.39 / 19.32 |
| #9 | the same stop on both: fore <= p, hind >= -p, p = 20 / 15 / 10 deg | -34.02 / 1.46 / 44.16 |
| #9 | elbow and knee flexion >= -30 / -20 / -10 deg (today -47) | -48.00 / -24.02 / 36.87 |
| #9, #10 | the record's coupled software limit, taken at a 5 mm margin: it changes exactly the samples under 5 mm (idealised, exact on the model's shapes); ACCEPTABLE if those are <= 1 % per controller | - |

**Reported, not judged:** per controller the least #9 and #10 clearances with their poses, the share of samples under
0 and under 5 mm, the 1st / 50th / 99th percentile of every leg joint (no gait report records these today), and the
same with fallen episodes left out. Outputs: `e1_summary.json`, `e1_joint_percentiles.csv`.

**What the outcomes would mean.**
| Outcome | For A | For B |
|---|---|---|
| A clash not REACHED and at least one stop for it ACCEPTABLE | That stop leaves today's controllers their logged ranges: a candidate fix for the CAD (the owner's decision) | Suggestive only: L1 at 50 cm keeps the same joint ranges, but no 50 cm logs are taken |
| REACHED, but a stop ACCEPTABLE | The gaits enter the region rarely; the stop changes them in at most 1 % of samples. A policy trained with the stop in the model should confirm it before it is adopted | As above |
| No stop ACCEPTABLE, coupled limit ACCEPTABLE | Only a software limit on the combined pose keeps the gaits' ranges: firmware and retraining work | As above |
| Nothing ACCEPTABLE | The study's condition "no stop or coupled limit for #9 leaves the gaits their ranges" holds on these logs: the legs need a layout change no configuration in the study has; A's packaging stays open | The same joint ranges, so the same risk at 50 cm (untested) |

### E2 Transmission and speed cap

**(a) DESK.** On the four fore128 policies' demo loops (90 s walking at 0.25 m/s + 30 s standing, repeated for the
stored run's one-pack minutes), seed blocks A (stored) and B (new, item 3):
- Transmissions and efficiencies (judged): **rod 0.95** (the rule's default: no dated source for a rod-end linkage,
  item 14), **belt 0.98**, **cable 0.96** (section 5). Reductions r = 1.0, 1.1, 1.2, 1.25, 1.3; placements **pitch**
  (the four hip pitch drives), **knees** (the four elbow / knee drives) and **both**. 3 x 5 x 3 = 45 judged rows per
  policy and block (`e2a_grid`, `judged` true). Supporting, not judged: the rod at the build's own 0.90
  (`rod_eta0.9`), 15 more rows. The r = 1.0 rows show the transmission's own loss against today's model of the knees
  (0.90): the supporting rod at 0.90 at the knees at r = 1.0 is today's linkage exactly (the control row); the judged
  rod at 0.95 there is today's linkage with the rule's efficiency; at the hip any r = 1.0 row adds a loss today's
  direct drive does not have.
- Drive torque: tau / (r eta) motoring, tau x eta / r braking (`drive_sq_from_split`; the rule's "joint torque / r"
  with its efficiency, and the build's braking convention); unreduced drives unchanged.
- **C1**: hottest winding over the demo at 35 C by the build's thermal model (`rl/telemetry/thermal.py` through
  `rl/analysis/thermal_ambient.py`, method A, via `form_factor_study.heat`), 8 min time constant (4 and 15 min
  beside); pass <= 90 C. **C2**: the worst drive's loop RMS over its continuous rating (X4 7.5 N m, X6 12 N m,
  drive side; `sim/data/drives.json`); pass <= 0.80 (`heat_rules`).
- **Speed check**: from E1's 1 ms histograms of the same policies (walk 0.25 m/s and standing, weighted 0.75 / 0.25 as
  the loop, the four episodes of each block pooled), each reduced drive's 99th-percentile speed and its peak, joint
  speed x r: pass if p99 <= 83 rpm and peak <= 91.3 rpm (1.1 x 83) on every reduced drive, and only the reduced
  drives (`e2a_speed_check`, item 13). Walk 0.15 m/s reported beside.
- **Verdict per judged (transmission, r, placement): PASSES** if C1, C2 and the speed check all pass for the worst of
  the four policies on both seed blocks (the study's worst-of-four reading). Per-policy passes and the supporting rod
  at 0.90 are reported beside (supporting). Output: `e2a_grid.csv`, `e2a_summary.json`.

**(b) SIMULATION.** Variants of the adopted model (`e2b_model`, scratch: the series and rigid variants by
`write_variant`, the baselines B025 and B0125 by `write_timestep_copy`), r = 1.25 at hip pitch and the knees:
- **Series model.** Each reduced drive becomes a rotor (a hinge in joint-side units, carrying the drive's armature x
  r^2, `sim/data/drives.json` X4 0.00324, X6 0.004304 kg m^2 at the drive output), coupled to its leg joint by a
  fixed-tendon spring of the transmission's joint-side stiffness and a damping of 2 x 0.05 x sqrt(k J) (an assumed
  structural damping ratio of 0.05, mostly numerical). The servo acts on the rotor with the adopted kp and kv in
  joint-side units (so the joint-side stiffness in the rigid limit is the adopted A2 value); the policy observes the
  rotor (the drive's encoder / r). Force limit at the joint side r x eta x 0.70 x the drive's peak. The leg joint
  keeps its range and damping; its armature moves to the rotor. Masses are the adopted model's: moving a drive to the
  hip or body is not modelled (it would change the leg inertia; a separate question).
- **Speed cap** (`SpeedCap`): above 0.95 x 83 rpm the drive's torque in the direction it turns falls linearly to zero
  at 83 rpm (drive speed = r x joint-side speed, i.e. 66.4 rpm at the joint); it can always brake.
- **Variants:** B1 (the adopted model itself, 1 ms; judged baseline), B025 (the adopted model with only its timestep
  changed to 0.25 ms; judged baseline), R (rigid, r 1.25, capped, eta 0.95; supporting), **rod** (eta 0.95),
  **belt** (0.98), **cable** (0.96) (series, capped; judged), belt_1span (the belt at one span, half its stiffness;
  supporting). Physics step 0.25 ms except B1. Only under item 4(b): B0125 (the adopted model with only its timestep
  changed to 0.125 ms) and the 0.125 ms rerun of the variant(s) concerned.
- **Conditions:** the four fore128 policies at 0.25 m/s, push left 0.5 and push right 0.5 m/s; seeds 700000-700003 and
  710000-710003: 7 runs x 4 policies x 3 conditions x 8 = 672 episodes. (B1's episodes are E1's same episodes,
  rerun.) Under item 4(b), 96 more for B0125 and 96 for each variant rerun at 0.125 ms.
- **Measured per episode** (`policy_episode`, with a `StepLogger` and, on the capped variants, a `SpeedCap`): the
  termination (`fell`, and `unstable` for "unstable_simulation", item 4), the delay actually used (10 ms: 10 / 40 / 80
  steps at 1 / 0.25 / 0.125 ms; checked every episode), forward speed 2-20 s, each drive's drive-side RMS over 2-20 s
  over its continuous rating (`StepLogger` with the variant's r and eta; baseline drives with today's 0.90 at the
  knees), the share of samples at or over the cap, the largest spring deflection. Supporting, for the rod: the same
  episodes' drive-side RMS re-mapped at eta 0.90 (`relog_drive_sq`; the run's force limits are the 0.95 ones).
- **Rule** (`e2b_compare`, `e2b_verdict`, `e2b_judged_runs`): per policy and condition, against each judged baseline
  on the same seeds: no seed falls that did not fall in the baseline; |mean speed - baseline's| <= 0.02 m/s and
  worst-drive RMS over continuous (pooled over the seeds that stayed up in both) - baseline's <= 0.03; an
  "unstable_simulation" episode is no fall and leaves the comparison not judged (item 4). **COMPATIBLE WITH TODAY'S
  POLICIES** if at least 3 of the 4 policies pass every condition against both judged baselines (B1 and B025, or B1
  and B0125 for a variant judged at 0.125 ms under item 4(b)); NOT COMPATIBLE once 2 or more fail; otherwise NOT
  JUDGED. Only rod, belt and cable get a verdict; R and belt_1span are read with the same rule but are supporting.

**Transmission stiffness at r = 1.25, joint side** (DESK, `transmission_stiffness`; plan.json `E2`; sources in section
5). Beside it, the series combination with the adopted servo stiffness (598.5 N m/rad, A2) and the spring's frequency
on the rotor with the leg held fixed (sqrt(k / J), J the rotor's armature x r^2), as desk context computed before any
run, not a result:

| Transmission | Hip pitch (100 mm span, assumed) | Fore knee (128 mm, the model's upper link) | Hind knee (188 mm) | With the servo, hind knee | Spring frequency on the rotor, leg held fixed, hind knee |
|---|---|---|---|---|---|
| Rod: M8 class 8.8, A 36.6 mm^2, E 200 GPa; crank 35 mm (r x 28 mm); at the 43 deg minimum transmission angle | 41,707 N m/rad | 32,584 | 22,185 | 583 (-2.6 %) | 333 Hz |
| Belt: 3M GT2, 15 mm wide, EA 78,807 N per unit strain; joint pulley 70 teeth, 33.42 mm pitch radius; both spans | 1,761 | 1,376 | 937 | 365 (-39 %) | 68 Hz |
| Belt, one span (supporting, belt_1span) | 880 | 688 | 468 | 263 (-56 %) | 48 Hz |
| Cable: 3.0 mm 7x19 316, EA 446,660 N (97.1 GPa x 4.6 mm^2); pulley 35 mm; both strands | 10,943 | 8,549 | 5,821 | 543 (-9.3 %) | 171 Hz |

Each is an upper bound on its transmission's stiffness: rod ends, pins and bearings (rod), tooth deflection (belt),
and constructional stretch and pretension loss (cable) are not included; no dated source sizes them. The two-span
belt figure also needs both spans taut: a pretension of at least half the effective tension, i.e. at the belt
variant's joint-side force limit (today's limit / today's efficiency, 1.0 at the hip and 0.90 at the knees, x 0.98 x
1.25) and the 33.42 mm joint pulley, **385 N per span** on
the X4 hip pitch and hind knees (25.7 N m) and **770 N per span** on the X6 fore knees (51.5 N m) (DESK,
`belt_pretension`, plan.json `E2.belt_pretension`). No sourced installation tension was found, hence belt_1span.
Desk sizing caveats (not part of the rule): at the drive's model limit the 15 mm belt's tight side carries 785 N (X4)
and 1,571 N (X6) (drive side, 0.70 x peak at the 26.74 mm drive pulley), against a total ultimate strength of about
2,890 N per span for 15 mm (scaled from the Gates sheet's 2,200 lb for two spans at 25.4 mm), so the X6 knee would
want a wider, stiffer belt; the 3 mm cable carries 750 / 1,500 N against 5.3 kN, at a 28 mm drive pulley (18.7 x the
cable radius, against the about 18:1 found).

**What the outcomes would mean.**
| Outcome | For A | For B |
|---|---|---|
| (a) some (transmission, r, placement) PASSES | A small reduction is a desk route to A's heat with today's torques; it would need a policy trained with it and CAD (the owner's decisions) | Not tested: at 50 cm the hip pitch already peaks at 1.10 of the cap on s3 (study, desk), so a reduction there makes speed worse |
| (a) C1 and C2 pass only where the speed check fails | The 83 rpm cap blocks the reduction route at 60 cm with today's gaits | As above |
| (a) nothing passes C1 and C2 | Reduction alone does not bring A under the heat lines with today's policies | - |
| (b) rod COMPATIBLE, belt or cable not | In simulation, compliance is what fails today's policies; a rod reduction is the route that needs no retraining; supports the lead's view on cables and belts | - |
| (b) all three COMPATIBLE | Compliance of this size does not break today's policies; the choice rests on packaging, upkeep and cost (desk) | - |
| (b) none COMPATIBLE, R not either | The cap (66.4 rpm at the joint) or the reduction itself breaks today's policies: any reduction needs a policy trained with the cap | - |
| (b) R passes, series variants fail | Compliance, not the cap, is the problem | - |

### E3 Per-policy heat (DESK on SIMULATED torques)

- **Policies:** all 11 (fore128 x4 and fore160 x4 on their own 128 / 160 mm models; run3-s0, run3-s1 on the 188 mm
  capsule model; run2b on the box-foot roll-ankle model).
- **Demo courses:** stored (seed0 700000) for fore128 and fore160; new for every policy at seed0 710000, and for run3 x2
  and run2b also at 700000 (`rl/demo_course.py` `run` with the default loop, through `demo_make`, which builds the
  walker exactly as `demo_course.policy_factory` does but loads through the sweep's loaders). 14 new demo courses,
  3 segments of 20 s each (42 episodes).
- **Computed** (`policy_heat`): each drive's loop RMS over continuous and hottest winding (35 C, method A, 8 min; 4 and
  15 min beside), the deciding drive and its mirror (with both standing RMS), the even left-right split's C2 and C1
  (desk arithmetic, not a pass).
- **Rule:** a policy **MEETS THE HEAT RULES AT 60 CM** if C1 <= 90 C and C2 <= 0.80 (worst drive, the build's
  ratings) on **both** of its demo courses. Per-policy reading, not the study's worst-of-four criterion.
- Output: `e3_policy_heat.csv`, `e3_summary.json`.

| Outcome | For A | For B |
|---|---|---|
| One or more fore128 policies MEET | A has an existing policy on the adopted model that meets the heat rules on its own (choosing it is the owner's decision; it does not make the worst-of-four criterion pass) | No reading (60 cm) |
| Only fore160, run3 or run2b policies MEET | Some 60 cm policy meets heat, but not on the adopted model: training on the adopted model with a heat or symmetry term is the route | - |
| None MEETS | A's heat needs a policy trained for heat (the study's step 1) or a hardware lever | - |

### E4 Forced air (DESK search)

- **Search** (web search only, generic engineering terms; nobody is asked and nothing is requested): "BLDC actuator thermal
  resistance forced air", "robot joint actuator fan cooling thermal resistance", "frameless BLDC motor thermal
  resistance airflow", "quasi-direct-drive actuator cooling winding temperature fan", "outrunner motor thermal
  resistance air velocity", "servo motor continuous torque with fan cooling", and the same with "planetary gear
  actuator". Each figure kept with its URL, read date, the drive, the flow (m/s or m^3/h) and whether the page was
  opened or only summarised.
- **Comparable drive:** a BLDC motor or geared actuator of 40-100 mm diameter and 0.2-1.5 kg, enclosed or open.
- **Computation only if a source gives R_th without and with forced air, winding to ambient or total** (item 12):
  ratio rho = R_forced / R_passive; every drive's fitted R_th x rho with its heat capacity kept (`forced_air_c1`, with
  the source's `heat_path` stated); C1 at 35 C for the four fore128 policies, both seed blocks. A housing-to-air or
  heat-sink pair is reported but not used. Otherwise the report says "no usable figure was found".
- Output: `e4_sources.json` (and `e4_c1.json` if computed).

| Outcome | For A | For B |
|---|---|---|
| A usable figure, and C1 <= 90 C for the worst policy with it | Forced air is a heat lever for A (fan, flow and power from the source; no shell model; the owner's decision) | B's desk C1 is 69.4 C already (study, supporting reading); forced air would add margin, not tested at 50 cm here |
| A usable figure, C1 still over 90 C | Forced air alone is not enough on today's worst policy | - |
| No usable figure | No conclusion; the bench heat test (the study's 3.2) remains the way to know | - |

### E5 Retuned trot (SIMULATION), at 60 cm and 50 cm

- **Models:** 60 cm the adopted model; 50 cm `build_scaled_trike.build_xml(0.50, "D0", "adopted", layout="L1")`
  (item 10). Plants: J1-J4 as `judge_walk60.JUDGED` on `walk60.plant_config(model_path=...)` (randomisation on,
  friction fixed, impratio); pushes on the sweep's nominal plant (`trot_sweep_config`). Command 0.25 m/s, 10 ms
  delay, 20 s episodes.
- **Conditions (10):** J1, J2, J3, J4; push back 0.3, 0.4, 0.5 m/s (forward axis, -); push right 0.3, 0.4, 0.5 m/s
  (sideways, -); pushes at 8 s for 0.1 s.
- **Parameters (5)** and boxes at 60 cm (50 cm: lengths x 0.833, times x 0.913, the Froude rule of
  `build_scaled_trike`): period_s 0.55-0.85 (frozen 0.70); duty 0.50-0.65 (0.55); body_pitch_deg -2 to +3 (0; + nose
  down); lift_m ("knee lift") 0.020-0.050 (0.035); max_reach_m ("stride", item 8) 0.060-0.100 (0.08). Every other
  WalkParams field at its frozen value (Froude-scaled at 50 cm, `scaled_walk_params`).
- **Search, at most 49 settings per size (budget 60), every setting logged:**
  - Stage 1 (screen): 33 settings fixed now (plan.json `E5.sizes.*.stage1`): the frozen controller (setting 00) and
    32 scrambled-Sobol points over the box (`scipy.stats.qmc.Sobol`, seed 20261128), rounded. Each on J1, J3, push
    back 0.5 and push right 0.5, seeds 101-105 (20 episodes). Ranked by `e5_screen_score`: falls in J1 and J3, then
    slow-gait episodes, then the worst hind knee, then the higher J3 speed.
  - Stage 2: the 6 best of stage 1 on all 10 conditions, seeds 101-110 (100 episodes each); each scored by E5's rule
    on these tuning seeds (`e5_success`).
  - Stage 3, only if no stage-2 setting meets the rule on the tuning seeds: up to 10 settings one step from the best
    stage-2 setting, one parameter at a time (`e5_neighbours`; steps 0.05 s, 0.03, 0.5 deg, 5 mm, 10 mm at 60 cm),
    all conditions, seeds 101-110.
  - **Choice** (`e5_choose`): of the settings meeting the rule on 101-110, the lowest worst hind knee, then fewer
    slow-gait episodes, then the higher J3 speed; if none meets it, the fewest failed checks first. Only seeds
    101-110 are used until the choice is written to `e5_choice_<size>.json`.
- **Judged once:** the chosen setting on seeds 301-310 and 401-410, all 10 conditions (200 episodes per size). The
  frozen controller on the same held-out seeds is reported beside it (supporting, not judged).
- **Rule** (`e5_success`, per held-out seed set): no falls in J1-J4; J3 mean speed (2-20 s) >= 0.225 m/s; slow-gait
  episodes <= 1 of 10 in every condition; worst hind knee <= 1.0 in every episode of every condition. The retune
  **SUCCEEDS** at a size if both sets pass. Reported, not judged: the drive speeds (peak and 99th percentile, against
  83 rpm), slip, the 10-20 s pitch, "slow" episodes.
- Outputs: `e5_tuning_<size>.csv` (every setting, stage, score), `e5_heldout_<size>.json`.

| Outcome | For A | For B |
|---|---|---|
| SUCCEEDS at 60 cm | A's scripted fallback avoids the slow gait in the judged conditions and after the pushes, with these settings (a candidate, not adopted) | - |
| Fails at 60 cm | The trot's slow, hot gait stays a risk for A's fallback | - |
| SUCCEEDS at 50 cm | - | Removes one of the study's cautions for B ("a retune does not remove the trot's slow gait at 50 cm"), with the adopted gains and L1 |
| Fails at 50 cm | - | The study's "plan is wrong" item holds for B: the scripted fallback runs hot at the size heat favours |

## 5. Sources (dated)

Web-search summaries: the pages were not opened (the hosts are outside this session's network); each figure is the
search tool's summary of the page, read on the date given. They are typical or catalogue values, not design
allowables.

| Item | Value | Source (read date) |
|---|---|---|
| Steel modulus | 200 GPa | matweb AISI 1018 sheet, web-search summary, as cited by `design/PACKAGING_PUSHROD_60CM.md` (27 Nov 2026) |
| M8 tensile stress area | 36.6 mm^2 | engineeringtoolbox.com, roymech.org, web-search summaries, as cited there (27 Nov 2026) |
| Rod geometry | 28.00 mm cranks; 43.0 deg minimum transmission angle | `design/pushrod60.py`, record section 2.1 (desk, 27 Nov 2026) |
| 3M GT2 belt tensile modulus | 30,000 lbf per inch of width per unit strain = 5,254 N per mm width (neoprene / fiberglass); 2M GT2 18,000 lb/in/in; 3M total ultimate 2,200 lb / 9,786 N (two spans, 25.4 mm) | Gates "Synchronous Belt Data Summary: Belt Tensile Properties", dated 3/07, hosted at fab.cba.mit.edu (BeltTensileProperties.pdf); web-search summaries (28 Nov 2026). The column reading is inferred from the summaries (the headers did not survive extraction): a source of moderate confidence |
| 7x19 316 wire rope modulus | 97.1 kN/mm^2 (14.1 x 10^6 psi) | Ronstan catalogue figures via theboatwarehouse.com.au and ronstantensilearch.com, web-search summaries (28 Nov 2026) |
| 3.0 mm 7x19 316 rope | metallic area 4.6 mm^2, breaking load 5.3 kN (another supplier 4.69 kN; swaging costs about 10 %) | theboatwarehouse.com.au listing of Ronstan RON-WR6719-03M, web-search summary (28 Nov 2026) |
| Steel cable bend ratio | about 18:1 bend radius to cable radius near 3 mm | web-search summary of cable-driven arm papers on arxiv.org (2507.15693, 2502.12963; the giving paper not identified with certainty) (28 Nov 2026); used only to choose the pulley radius |
| Toothed belt efficiency | 0.98 | Gates energy-savings material (98-99 % for PowerGrip GT2), a US DOE tip sheet citing Gates (98 %); vendor-sponsored; web-search summary (28 Nov 2026) |
| Cable efficiency | 0.96 (= 0.98^2, two pulleys) | 98-99 % per passive pulley at D/d 15, earlier work cited in arxiv.org 2509.11506 ("Design and Development of a Remotely Wire-Driven Walking Robot"); web-search summary (28 Nov 2026) |
| Rod efficiency (judged) | 0.95 | no dated source found for a rod-end linkage (searched 28 Nov 2026), so the lead's E2 rule's default ("0.95 if none") |
| Rod efficiency (supporting) | 0.90 | the build's own 1:1 linkage efficiency, `sim/data/drives.json` `transmission_efficiency` (no source field; added in dc2e405d, 18 Nov 2026); also today's knees in every baseline figure |
| Default efficiency | 0.95 | the lead's E2 rule (28 Nov 2026); used for the rod and the supporting variant R |
| Belt installation tension | none found | no sourced figure for a 15 mm 3M GT2 belt (28 Nov 2026); the pretension needed for both spans taut is computed (385 / 770 N per span, desk) and a one-span supporting variant is run |
| X4 and X6 ratings, armature, peaks | X4 7.5 N m continuous, 30 N m peak, 0.00324 kg m^2; X6 12 N m (the X6-50-L's 0 rpm table), 60 N m, 0.004304 kg m^2 | `sim/data/drives.json` (maker figures via the repo catalogue, read 7 Nov 2026) |
| X4 speed cap | 83 rpm above 24 V | `DECISION_LOG.md` E1; confirmed on the maker's V4.1 sheet, read 28 Nov 2026 (as the study records) |
| Servo stiffness | kp 598.5 N m/rad, kv 11.51 (pitch, flexion) | `DECISION_LOG.md` A2 (an assumption, not a maker figure) |
| Room and heat rule | 35 C room; 90 C winding | D48; bench B3 |

## 6. Order, budget and stop rules

- Order: E1 (its logs feed E2(a) and E3), E3 and E2(a) (desk), E2(b), E4 (search, any time), E5 (60 cm, then 50 cm).
- Workers: at most 2. Estimated wall time on 2 workers: E1 about 20-30 min; E3 about 5 min plus the desk tables;
  E2(b) about 55-70 min (672 episodes; each 0.125 ms rerun under item 4(b) about twice its 0.25 ms run); E5 up to
  about 2.5-3 h per size.
- **One judged pass.** No rerun to improve a result. A harness fault found after a run (not a controller's
  behaviour) is fixed, the fix reviewed, and the affected experiment rerun on new seeds (E1, E2(b): 720000-720003 and
  730000-730003; E5 held-out: 501-510 and 601-610), labelled as a rerun; both results are kept. An E2(b) episode
  ending "unstable_simulation" is not such a fault: it is handled by section 3 item 4 (the same seeds, at 0.125 ms,
  decided before any run), and this new-seed rule does not apply to it.
- E5's tuning budget is counted over all stages (at most 60 settings per size); every setting's id, values, stage and
  score is written as it finishes.
- Every result file records `candidate_checks.provenance(runner)`: the commit (`git rev-parse HEAD`, read only), the
  sha256 of `sim/candidate_checks.py`, of the runner script that wrote it (wherever it lives, the scratch folder
  included), of the 16 build files the code imports or reads (`BUILD_TOOLS`) and of any variant model it ran; and the
  seeds and the wall time.

## 7. What this does not test

- No hardware: nothing is measured. The policies read 11 simulator-only inputs and run an ideal, frictionless servo.
- No turning, starts or stops (the demo loop is straight); no shell or housing heat; no 50 cm policy (no training).
- Moving a drive to the hip or body is modelled only as a reduction, a compliance and a cap, not as the changed leg
  inertia, the drive's new mounting or its cooling.
- The stops are tested on the logged poses of today's controllers, not on gaits trained with the stops in place.
- E5 tests one controller (the trot); E2(b) and E1 have the two-controller reading only where the rules say so.
