# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — Into the Wild (v0.4.0)

### Added
- **One record of everything tried on the T. rex tail saturation habit, with
  the entropy continuation screen judged, final: line 3, no difference
  detected** (#654; decision D-D52's amendment and D-D53 to D-D55;
  `docs/investigations/TREX_TAIL_SATURATION_HABIT_2026_10.md`, with its
  ledger, and the appended §28 of
  `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md`, with a dated
  status note at its top; `docs/NEXT_STEPS.md` §3, `docs/KNOWN_ISSUES.md`
  (Update 23 of the T-Rex stance entry and a dated update to the
  action-saturation entry), `docs/README.md`; evidence in `ecs_results/`,
  `com_revisit/`, `tail_probes/` and `tail_record/` of
  `docs/investigations/stance_2026_10/`: the run log's lines 721 to 759, the
  screen's approved criteria, approval, tools, run records, reading and two
  independent checks, the centre-of-mass revisit's report with its checks on
  trained policies and the creep scan, the tail clamp test, the normaliser
  probe and the three posture tests, each with its verifier's notes, and the
  record's ledger and checks). Records only. The note, written at the
  maintainer's request and fact-checked before it was put here, gathers in one
  place what the records hold on the habit from 2026-07-09 to 2026-10-29: what
  it is (a policy command held at its bound, almost always the tail-tip pitch
  actuator at +1, not a saturated joint or motor) and how it is detected; where
  it stands; every observation, change, probe, screen and round in date order,
  each a row of a 167-row ledger; the evidence for and against each candidate
  cause; the interventions; lessons learned; and the open questions and the
  next steps already proposed, none chosen. The screen continued two of the
  first r10 round's checkpoints with the entropy coefficient as recorded
  against 0, 24 CPU continuations: seed 42's railed in all 12, one of seed 80's
  six per arm flagged on the neck, and removing the remaining bonus from these
  mid-rise states changed nothing it could detect. The report-only work that
  followed it, none of which trained, found the rail no balance aid, costing
  reward and persisting through a learned positive feedback from the observed
  tail angle to the tail command, whose growth sits in the network's weights
  rather than in the observation normaliser's drift; and no nose-down or
  forward centre-of-mass shift before the tail rose, though from mid-rise on
  the railed lineage's networks read their nose-down posture as a tail offset.
  No T. rex stance is certified, and nothing trains on `main`.
- **Decision D-D52's design and outcome and decisions D-D53 to D-D55 recorded:
  the entropy continuation screen read its line 3, no difference detected;
  next, the centre-of-mass follow-ups with a tail clamp test, a normaliser
  probe, three posture tests and one record of everything tried on the T. rex
  tail saturation habit** (#654; the maintainer's decisions and requests of
  2026-10-28 21:05 UTC to 2026-10-29 23:21 UTC in the stance run log;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2, `docs/CONSOLIDATION_PLAN_2026_09.md`
  §6). Records only. D-D52 is amended with the screen's design, as the
  maintainer took every recommendation of its draft (starts s42 at 3,145,728
  steps and s80 at 4,194,304, 6 CPU continuations per arm and start, the
  entropy coefficient as recorded against 0, an exact stratified test
  two-sided at 0.025 per side), its approval and launch, and its outcome,
  final: all 24 continuations ran to their end; line 3, "NO DIFFERENCE
  DETECTED" (s42 6 of 6 in both arms, every flag on the tail-tip actuator; s80
  1 of 6 in both, on the neck), agreed by two independent checks; its text
  outputs on a never-merged orphan branch at the maintainer's word. D-D53: the
  centre-of-mass regression tests revisited, with centre-of-mass checks on the
  screen's end policies, and a tail clamp test both ways (simulation only,
  report only). D-D54: a centre-of-mass test change in its own pull request
  (#653), a creep scan of two more models before any plant change, a
  report-only centre-of-mass check on trained policies in the next round's
  pre-registration, and a no-training normaliser probe of the tail loop.
  D-D55: one record of everything tried on the tail habit, three no-training
  posture tests, and that record as its own investigation note with its
  evidence: this change. Nothing trains on `main`, and no gate, reward, plant
  or criterion changes.
- **Absolute stance targets for the centre-of-mass stance tests, each a
  strict expected failure where a model misses it today** (#653; tests and test
  helpers, and one input check in the reduction; the maintainer's decision of
  2026-10-29; `com_stance_helpers.TARGETS`, `Target`, `TargetMissed` and
  `ComStanceQualityBase.check_target`, a `test_target_*` per missed target in
  each species' `test_com_stance_quality.py`, and a target report in
  `com_stance_calibration`). The CoM stance tests' bars are calibrated on
  today's plants, so they catch a change, not a poor stance the plant already
  has; a target is what a good stance does. Six, their numbers proposals for
  the maintainer: the body ends the episode turned under 1° from its spawn
  (read at the end, not the largest turn along the way); no rocking after 3 s
  (every whole second from 3 s to the horizon's end at a CoM speed RMS of
  1 mm/s or under, the settle rule read from 3 s, so a longer horizon does
  not thin it out; and under 1 mm of fore-aft spread, one spread over the
  window from 3 s); settled by 2 s (the calibration's settle rule); the CoM
  within 5 mm of the feet's centre sideways; under 0.3° of kept roll; and,
  from 1 s (after the landing), no foot slipping farther than a 1° turn about
  the feet's centre slides the farthest foot, at each body's own distance and
  never under 1 mm (the T. rex 2.44 mm, the Velociraptor 1.40, the
  Compsognathus 1.00, the robot 1.13, the Brachiosaurus 13.58, the
  Dibothrosuchus 2.63). Each reads the noise-free statue, the two noisy test
  spawns and, where those meet a target that other spawns miss, the worst
  spawn that misses it (`target_seeds`: the Compsognathus 101 and 124, the
  robot's creeping 111, the Brachiosaurus 126). Measured on the statue and
  the 80 committed spawns per model: 22 of the 36 model-target pairs are
  missed today (the T. rex 4, the Velociraptor 3, the Compsognathus 3, the
  robot 4, the Brachiosaurus all 6, the Dibothrosuchus 2); each is a
  `pytest.mark.xfail(strict=True, raises=TargetMissed)` whose reason states
  the measured values (the Velociraptor's also that its foot site under-reads
  its toes' slip, a median 1.4x), so a model change that meets one makes it
  pass and so fail, and the mark comes off. A fall, a reading the reduction
  cannot give (a renamed key, a NaN) or a non-finite sample in any series a
  target reads raises something other than `TargetMissed` and fails the test
  outright, and `stance_track_series` now refuses a non-finite `com_vel`, as
  it refuses a non-finite CoM (no track sets one today). The settle rule
  (`com_stance_calibration.settle_time`) and the per-second reading share
  their seconds, which now reach a 20 s episode's last one: MuJoCo's summed
  time ends it at 19.999999999999794 s, and its last second, from 19 s to
  20 s, was left out; no pinned window and no measured value moves. The other
  14 pairs are met by the statue and all 80 spawns and are not tested. The
  noisy-spawn test now takes its episode from the per-class cache the targets
  read (the same `run_episode` call; its outcomes are unchanged). Cost: about
  10 s on one core. No environment, physics, reward, gate, observation,
  policy interface or digest changes.
- **The centre-of-mass stance calibration hands its re-measured pins over as
  a reviewable diff, and writes a looser bar only when told to** (#653;
  tests and test helpers only; the maintainer's decision of 2026-10-29;
  `environments/shared/tests/com_stance_calibration.py`,
  `com_stance_helpers.py` and `test_stance_track_reduction.py`).
  `python -m environments.shared.tests.com_stance_calibration
  <module>:<class>` printed fresh pins for a re-pin to be copied by hand. It
  gains `--diff`, which prints on stdout a unified diff of the species file
  with the class's pins (`physics_revision`, `window_start_s`, `reference`,
  `spread_width` and `noisy_seeds`) replaced by the re-measured ones, which
  `git apply` takes (the report goes to stderr); `--write`, which makes the
  same change in place; and `--allow-loosen`. Each change is classed as its
  bar reads it: a wider width, a later window, a one-sided reference moved
  toward its bad side or a dropped key is looser, the opposite tighter, and a
  two-sided reference, the physics revision and the noisy seeds moved.
  `--write` refuses a looser change without `--allow-loosen` (exit status 2,
  the file untouched), so after a model fix the committed bars only tighten
  unless a reviewer accepts the loosening the diff shows. Both refuse when the
  running MuJoCo is not the canonical one or the window is under the
  shortest. A pin whose value is unchanged keeps its lines, so on an
  unchanged plant the diff is empty (the robot's, measured); a changed pin
  also re-reads the class's provenance line and the two comments that state
  the held-out and noisy-seed numbers, never a hand-written one, and the
  report says whether each known poor spawn still breaks a width or its
  strict expected failure is to go. The provenance test's message names the
  commands. Nine synthetic tests (40 in the file). No library code,
  environment, physics, reward, gate, observation, policy interface or digest
  changes.
- **The first T. rex stance round on physics r10 recorded as judged, final:
  line 1, a tail rail formed on r10 in training** (#652; decision D-D51's
  amendment and D-D52; the appended §27 of
  `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md`, with a
  dated status note at its top; `docs/NEXT_STEPS.md` §3,
  `docs/KNOWN_ISSUES.md` (Update 22 of the T-Rex stance entry and two dated
  updates), `docs/README.md`; evidence in `stance6_results/` of
  `docs/investigations/stance_2026_10/`: the run log's lines 681 to 720, the
  scorer's output, the completeness check per run, the independent check of
  the reading, the report-only readouts, the review, the next-step proposals
  and the fixer's record, K1–K4 per run, the D-9 and D-10 summaries, the
  fetch record, and the measurement panels' summaries and the judge's
  files). Records only. §27 records the runs' ends, noted from Drive;
  the final fetch, its inline transfers made by an exact-copy script at the
  maintainer's decision; the reading, final, with all three runs complete:
  seed 42 pinned `tail_3_pitch_act` at +1 in training (22 ledger alarms from
  3,932,160 steps) and failed its gate, seed 60 parked it (MEAN_a +0.642
  against the 0.60 line), and seed 80 passed its gate as the round's one
  positive run with 6 alarms on `neck_pitch_act`; line 1 an existence result,
  not a rate; statistic A 2 of 3 against LTSB's r9 2 of 5 (one-sided Fisher p
  0.93, nothing significant); the reading agreed by an independent check; the
  report-only readouts D-1 to D-16 (no head-low reading, no fall in the
  deterministic panels, the in-run K3 panel equal to the judge's K3), the
  entropy analysis the maintainer asked for (its inferred answer: the entropy
  bonus no direct driver, its role in when an onset happens untested) and the
  comparison with r9 and the two looks at the running files; the process
  departures; and the review, the next-step proposals and the maintainer's
  decision D-D52. No T. rex stance is certified, and nothing trains on `main`.
- **Decision D-D51's outcome and decision D-D52 recorded: the first T. rex
  stance round on physics r10 read its line 1, a tail rail formed on r10 in
  training; next, a CPU continuation screen of the entropy coefficient, its
  criteria to be approved before any training** (#652; the maintainer's
  requests and decisions of 2026-10-27 23:59 UTC to 2026-10-28 19:30 UTC in
  the stance run log; `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2,
  `docs/CONSOLIDATION_PLAN_2026_09.md` §6). Records only. D-D51 is amended
  with how the round was read: a second report-only snapshot of the running
  files, each run's end noted from Drive while the maintainer slept, the final
  fetch's inline transfers copied byte for byte by an exact-copy script after
  a hand transcription was cut off (the fetch rules unchanged), and the
  durable copy's first commit pushed to a never-merged orphan branch, each at
  the maintainer's word; and with its outcome, final: all three runs complete,
  line 1 ("TAIL RAIL FORMED ON r10 IN TRAINING (ledger)"), statistic A 2 of 3
  against LTSB's r9 2 of 5 (one-sided Fisher p 0.93, nothing significant);
  seed 60 parked its tail-tip actuator, seed 42 railed it at +1 and failed its
  gate, and seed 80 passed its gate as the round's one positive run, its
  alarms on the neck. An existence result, not a rate: r10 as shipped does not
  prevent the tail rail (the review's inference from the wording of line 1).
  D-D52: the next step is a CPU continuation screen from the round's own
  checkpoints, the entropy coefficient as recorded against 0, its
  pre-registration drafted and approved before any training (a training-loss
  change for scratch continuations only), and this records change. Nothing
  trains on `main`.
- **Centre-of-mass stance regression tests: a stance that stands but leans,
  drifts toward a support edge, sags, turns or twists on its feet, splays or
  rocks now fails** (#650; tests and test helpers only; the maintainer's
  request of 2026-10-28; `environments/shared/tests/stance_track_helpers.py`,
  `com_stance_helpers.py`, `com_stance_calibration.py`,
  `test_stance_track_reduction.py` and a `test_com_stance_quality.py` in each
  species' tests directory). The static-balance tests read the CoM at the home
  keyframe and check survival; these track the whole-body CoM through the
  standing episode. A pure-numpy reduction (`stance_track_quality`) turns a
  CoM track, the root's attitude and the feet (and, where known, the loaded
  support points, the per-foot loads and the foot headings) into 35 named
  numbers in six families: the CoM's and the extrapolated CoM's margins in
  the support polygon (to the nearest edge, sideways and fore-aft), lean and
  sideways drift, fore-aft posture, height, pivot (the turn over the window,
  at its end and from the spawn, the twist over the feet, and the feet's
  turn, splay and slide) and sway (the CoM's speed and the spread of the CoM
  and of the root's roll, pitch and yaw about their trends). It reads the
  stance in its own heading frame, so a policy's rollout, an evaluation
  episode the gait recorder captured (`track_from_episode_trace`) or a
  robot's log reads the same way, and `lean_sign_summary` flags a lean of one
  sign across seeds. Each model gets 13 tests: the pins' provenance (physics
  revision and MuJoCo version), six on the noise-free statue (a key mirror
  symmetry fixes holds that value, every other key the statue's own pinned
  value, each to twice a resolution floor of 0.5 mm, 0.5 mm/s, 0.05° or
  0.005 of the load), two held-out spawns at reset noise 0.02 (widths twice
  the worst of 30 calibration spawns, checked on 50 held-out ones), three
  scripted poor stances that stand the full horizon and must reach twice
  their bars (a 1° hip-roll lean, 4.7x to 37.6x; a yaw torque, 8.6x to
  22.4x; a fore-aft rock, 4.5x to 13.0x), and a check that the sampler reads
  what the evaluation recorder records. The window opens at each species'
  measured settle (2 to 12 s); the T. rex and the Brachiosaurus, which rock
  or bounce for 9 and 12 s after the reset, run 2,000 steps. Readings for the
  maintainer: after a reset-noise-0.02 spawn the T. rex, the Velociraptor and
  the Compsognathus turn up to 4.9°, 5.7° and 5.1° on their feet (80 spawns
  each, nearly all of it in the first seconds); the Brachiosaurus keeps a
  lean of up to 23 mm; 10 of the Compsognathus robot's 280 measured spawns
  end buzzing on their feet and creeping 58 to 79 mm backwards over 18 s
  (one kept as a strict expected failure); and the r9 T. rex checkpoint on
  the r10 plant, which passes every existing test, rolls left side up on 10
  of 10 seeds (sign test p = 0.002) where the statue at the same spawns does
  not.
  Cost: about 71 s added to the species suite on one core, 104 s under CI's
  coverage flags (Python 3.11), all of it on every push (there is no slow
  marker).
  `python -m environments.shared.tests.com_stance_calibration` re-measures a
  species' pins. No library code, environment, physics, reward, gate,
  observation, policy interface or digest changes.
- **The first T. rex stance round on physics r10 recorded as set up, approved
  and launched, results pending, with the report-only telemetry built for it
  and walking pilot 2's retirement** (#647; decisions D-D45 to D-D51; the
  appended §26 of
  `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md`, with a
  dated status note at its top; `docs/NEXT_STEPS.md` §3,
  `docs/KNOWN_ISSUES.md` (Update 21 of the T-Rex stance entry and two dated
  updates), a dated note in §17 of `docs/GAIT_QUALITY_PLAN_2026_09.md` and its
  header, `docs/README.md`; evidence in `stance6_setup/` of
  `docs/investigations/stance_2026_10/`, and the round's pre-registration,
  approved before any training step, in `stance6/` there: the criteria and the
  texts they cite as byte copies, the approval record, the launch steps and
  fetch brief and the pinned tools, all from before the first training
  step). Records only. §26 records physics r10 and the heading-test fix
  reaching `main`; the telemetry of #644 and #645, with no new MuJoCo sensor
  and the in-run K3 panel, and the compact K3 extract; walking pilot 2 retired,
  never launched; the round's design (arm LTSB ported to r10, seeds 60, 42 and
  80 from scratch, 12M steps each on Colab L4s, on the never-merged branch
  `claude/screen-trex-r10-armltsb`, one pre-registered statistic against
  LTSB's r9 2 of 5); its run rules, drafted as full rules over twelve checked
  versions and replaced, after the maintainer's question and a rule-model
  comparison, by the maintainer's choice of all-or-nothing rules (LEAN-R with
  PRESENCE; the rule model gave no wrong reading under either [model]); its
  validations, all passed; its approval on 2026-10-27 before any training step
  (criteria `99a44469…`, branch head `8737cd3d`); and its launch that day at
  about 14:57 UTC, three runs (on L4 runtimes as their launch lines record
  them), each passing its configuration-only launch check. The runs are
  training: no result is recorded, and nothing trains on `main`.
- **A compact K3 extract of the five band runs' judge files, read by the
  training-history report's judge-K3 readout** (#647; the maintainer's
  decision of 2026-10-25 17:42 UTC in the stance run log, D-D46;
  `docs/investigations/stance_2026_10/stance5/k3extract/`,
  `environments/shared/tests/test_training_history.py`). The readout
  (`plot_training_history.py --judged`) reads a run's judge files, which
  stayed in the session's scratch. The extract holds, for `stance5_LTSB_s52`,
  `_s60`, `_s42`, `_s70` and `_s80`, each `judge_<steps>.json` and
  `judge_final.json` (39 files, about 0.17 MB against their 15.8 MB)
  cut to the keys the readout reads (the schema, mode, seed and recorded
  final step, and of the joint-space and own-sigma noise tests `n`, `falls`,
  the spawns, the sigma and each episode's seed, length and horizon flag),
  values unchanged, so `--judged` reads a run's directory there as it reads
  the judge's; an index, `k3_extract.csv`, with each source's sha256; and the
  script that wrote them. Measured: the readout of the extract equals that of
  the judge files for every run, and at K3's checkpoints it gives the
  replication score of record's K3 falls and pass/fail verdicts (held for s52,
  s70 and s80; not for s60 and s42); a new test checks the extract against its
  index and that score. No library code changes.
- **Decisions D-D45 to D-D51 recorded: walking pilot 2 retired; report-only
  telemetry, no MuJoCo sensor and an in-run K3 panel for the first T. rex
  stance round on physics r10; that round's design, its all-or-nothing run
  handling, its approval and its launch** (#647; the maintainer's decisions
  of 2026-10-25 to 2026-10-27 in the stance run log;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2, `docs/CONSOLIDATION_PLAN_2026_09.md`
  §6). Records only. D-D45: walking pilot 2 (D-D36, physics r9, never
  launched) is retired, a new pilot to be designed once r10 has a stance
  parent; D-D36 notes it. D-D46: head posture, hip roll and the judge-K3
  readout, then commands, posture, energy, drift and slip and four `MjData`
  extras, in appended columns and pages (#644 and #645).
  D-D47: no MuJoCo sensor before that round (a sensor is a plant change).
  D-D48: the in-run K3 noise panel, off by default, on in that round. D-D49:
  arm LTSB of stance5 ported to r10, seeds 60, 42 and 80 from scratch, 12M
  steps each on Colab L4s, once both telemetry changes were on `main`, one
  pre-registered statistic (alarm runs against LTSB's r9 2 of 5), on a
  never-merged run branch. D-D50: the runs read all or nothing (LEAN-R with
  PRESENCE), after the maintainer asked why the criteria carried so much
  edge-case handling. D-D51: the criteria (sha256 `99a44469…`) approved on
  2026-10-27 before any training step, and the three runs launched that day at
  about 14:57 UTC (on L4 runtimes as their launch lines record them, which the
  launch check does not see), each passing its configuration-only launch check,
  with D-D51 amended that day for a report-only snapshot of the running files
  at the maintainer's request (the rules unchanged, the round read from its
  final fetch only); their results are pending. Nothing trains on `main`.
- **Commands, tail and body posture, energy, drift and slip, and four
  `MjData` extras per evaluation episode in `eval_episodes.csv`** (#645; the
  maintainer's decisions of 2026-10-25 15:44 UTC in the stance run log,
  items (1) to (3), and 16:15 UTC: no MuJoCo sensor, all four MjData-derived
  extras, the specification's other defaults accepted;
  `environments/shared/gait/body_telemetry.py`,
  `environments/shared/gait/recorder.py`,
  `environments/shared/eval_diagnostics.py`; documented in
  `docs/TRAINING_ARTIFACT_PRESENTATION.md`, "The body columns"). Report
  only. Where the recorder runs (the T. rex, both Compsognathus models and
  the Dibothrosuchus; the Velociraptor and the Brachiosaurus keep the
  documented gap), 118 columns on the T. rex, appended after the posture
  columns, over the gate's window: (1a) each actuator's window-mean command
  and the at-rail and parked actuators named with their signs; (1b) each
  tail hinge's angle (+ tip up, + tip left), the tail-tip chord angle,
  the pelvis height, pitch range and roll; (2) per group (legs, tail,
  neck and head) the absolute and net mechanical work, peak power, torque
  RMS (and as a fraction of the force limit) and command jitter from the
  actuator force and velocity at every substep, the episode's
  `action_delta` term and each leg's work; (3) the screen's drift copied,
  its fore-aft and lateral parts, the path length, the signed yaw change,
  each foot's slip (the force-weighted sliding speed of its floor contacts
  from `efc_vel`) and net site travel; and the extras: (g) each limited tail
  hinge's and leg joint's share of substeps on its stop and mean stop
  torque, (h) the CoM's and capture point's margins to the toe and heel
  lines and the centroidal angular momentum, (c, d) the tail-base and
  neck-base wrench from `cfrc_int`, (a) the head's angular-rate RMS. The
  recorder's new opt-in captures (`record_actuators`, `record_slip`,
  `record_joint_limits`, `record_momentum`, `record_wrenches`,
  `record_body_velocity`; the tail tip joins the head tip in the one
  `record_sites` call) copy `MjData` and never write it, behind
  `report_extensions`; no new MJCF sensor, no `gate_progress.npz` key, gate
  input, `StanceEpisodeMetrics` field or `progress.json` entry, no digest
  moves, and `train_episodes.csv` is unchanged. Measured: the extended
  in-suite inertness proof trains and evaluates bit for bit alike with them
  on and off; on the r10 statue and the r9 checkpoint `stance5_LTSB_s42`
  4,194,304 (deterministic, K3 joint-space and own-sigma noise; seeds 3042
  and 3043) all 116 numeric columns equal an independent per-substep
  recomputation with MuJoCo's own functions to 2.3e-14, the copied drift is
  the screen's bit for bit, and a Newton-Euler free-body check agrees with
  the base wrenches to 4e-4 N standing (0.42 N of 154 under joint noise);
  per window-step weighted evaluation the command means and jitter are its
  `action_dc_per_actuator` and `action_delta` to 1e-12. Cost [measured,
  loaded host]: about 67 ms per 1000-step statue episode attributed in
  place (4.4%), median A/B +2.3% (statue) and +6.0% (r9 policy), within the
  specification's +10%; a T. rex row grows by about 1.6 to 2.0 KB, to
  about 3.2 KB.
- **The in-run K3 noise panel** (#645; the maintainer's decisions of
  2026-10-25 15:44 UTC in the stance run log, item (6), and 16:15 UTC: the
  judge's full K3 every 1,048,576 steps and at the end, synchronous, off by
  default and enabled at launch, the offline judge's K3 staying the K3 of
  record, the specification's other defaults accepted;
  `environments/shared/k3_panel.py`, wired in `train_base.train(...,
  k3_panel=...)`, the command line's `train --k3-panel` and the SB3
  notebook's `K3_PANEL` cell; documented in
  `docs/TRAINING_ARTIFACT_PRESENTATION.md`, "The in-run K3 noise panel").
  Report only, never a stage key (no digest moves). On the stance stage, at
  every 1,048,576 steps and at training end (after the final checkpoint
  pair is saved, so an interruption during it cannot cost the pair), it
  rolls the stance5
  replication judge's two K3 noise tests with the judge's arithmetic (20
  spawns each, own sigma and the joint-space sigma read at launch from a
  JSON file and recorded by sha256) on a CPU copy of the live policy and a
  copy of the normalizer's statistics, in an env of its own, inside a guard
  of the global generators, into `k3_panel.csv` and `k3_panel.json`
  (`mesozoic.k3-panel/v1`) beside the ledgers; never `gate_progress.npz`,
  `progress.json`, TensorBoard or any gate input. A resume keeps the prior
  panels, copies the prior files aside and rolls a missing grid panel on the
  restored pair's statistics. Measured: the extended in-suite inertness
  proof trains bit for bit alike with the panel on and off, with stock PPO
  and with mirror-symmetric PPO (`[ppo] symmetry = "mirror_kl"`), its
  panels every 128 steps (three inside a rollout), and its panels equal an
  offline run on the checkpoint pairs saved at their steps; on the
  judges' own tree (cf80ecb1) the panel reproduces the judge files of
  `stance5_LTSB_s80` at 4,194,304 (joint-space falls 2 of 20, at seeds 3051
  and 3061) and `stance5_LTSB_s42` at 4,194,304 (0 and 0) episode for
  episode, lengths, termination reasons and returns bit for bit; a
  40-episode panel of that checkpoint (38 of 40 episodes to the horizon)
  takes about 85 s on the local 4-vCPU host. [inferred] One that holds all
  40 takes about 88 s, so a 12M-step run's 12 panels about 17.6 min; on a
  Colab L4 about 18 to 26 min, about 2 to 3% of a 14.4 to 15.4 h run.
- **Training-history pages 13 to 19** (#645; the maintainer's decisions of
  2026-10-25 15:44 UTC, item (4), and 16:15 UTC: the observation
  normalization read through a whitelisting stub unpickler;
  `environments/shared/reporting/training_history.py`; documented in
  `docs/TRAINING_ARTIFACT_PRESENTATION.md`, "Training history pages"). Report
  only, after the hip roll page so every earlier page keeps its name:
  `13_commands` (each actuator's window-mean command with the parked and
  rail lines, and the episodes at rail and parked by actuator and sign),
  `14_tail_body` (the tail's chord and hinges, the pelvis's height, pitch
  and roll), `15_energy` (work, peak power, torque and command jitter per
  actuator group, and per leg), `16_drift_slip` (the displacement, path,
  signed yaw, and each foot's slip beside its site travel),
  `17_exploration_k3` (the leg actuators' exploration std from
  `policy_probe.npz` beside K3's falls from the in-run panel and the judge
  files), `18_obs_normalization` (the normalization's drift per checkpoint
  from the VecNormalize pickles in `models/`, read through a stub
  unpickler that runs no numpy, SB3 or gymnasium code and decodes the
  statistics from their raw bytes as plain numbers, protocol 4 or 5, with
  the probe's per-update statistics) and `19_body_dynamics` (the four
  MjData extras: joint stops, base wrenches, the CoM's and capture point's
  margins, the centroidal angular momentum and the head's rate, on one page
  of their own so pages 13 to 18 are the specification's); the JSON gains
  `k3_panel`, `exploration` (each leg actuator's std at every K3 point, the
  point's own sigma where recorded) and `obs_normalization`. A ledger without the body columns, a run without the
  probe, the pickles or the panel renders labelled absent panels. Measured:
  the probe's rows equal the pickles cast to float32 at their shared steps
  (the stance5 replication run s80 at 4,194,304, and in the suite).
- **The stance5 replication judge's K3 criterion, read out over training in
  the training-history report** (#644; the maintainer's decisions of
  2026-10-25 14:13 UTC in the stance run log, item (4): task #47's remaining
  judge-K3 overlay;
  `environments/shared/reporting/training_history.py`,
  `environments/shared/scripts/plot_training_history.py`; documented in
  `docs/TRAINING_ARTIFACT_PRESENTATION.md`, "Training history pages").
  Report only, and never a verdict. The ledgers hold none of K3's inputs:
  they are the judge's own perturbation panels on spawns 3042-3061 at each
  judged checkpoint. `plot_training_history.py --judged JUDGED_DIR ...` (one
  per stage directory, `-` for none) reads a run's `judge_<steps>.json` and
  `judge_final.json` (`noise_matched_control_joint` and `noise_matched_own`:
  `falls` of `n`) and the outcomes page gains a last panel: both counts per
  judged checkpoint against K3's bars ("K3, robustness at 6,291,456 and at
  final: joint-space noise falls at most 2 of 20, and own-sigma falls at
  most 1 of 20", `docs/investigations/stance_2026_10/stance5/replication/criteria.txt`),
  K3's checkpoints filled and annotated; a comparison draws a strip per run.
  The JSON (`judge_k3`), the HTML and the checkpoint table carry the points,
  with within-bar flags at K3's checkpoints only, withheld where a panel is
  not K3's 20 episodes on spawns 3042-3061; only the stance3, stance4,
  stance5 and stance5 replication judges' files are read (the same K3),
  each named by its own judge, and a file of another schema, one that is
  not a full noise run, records another seed or lacks a panel is skipped
  with its reason. Measured: on the judge files of `stance5_LTSB_s70` and
  `stance5_LTSB_s80` the readout gives 0/0 and 1/0 (s70) and 1/0 and 2/0
  (s80) at 6,291,456 and the final, the replication score of record's K3
  values, and on the stance4 and stance5 rounds' judge files it equals
  those scores' K3 falls and pass too (8 of 8 runs: stance4 s52, s60, s42;
  stance5 s52, s60, s42; the replication's s70, s80); a test rebuilds the
  judge files of all five band runs from the replication's score
  (`docs/investigations/stance_2026_10/stance5/replication/results/scores/score_final.json.gz`)
  and checks the readout against its K3 values and pass, and against
  `k3readouts/out`'s judge falls. Without `--judged` (the stage-end report)
  nothing is drawn.
- **Head posture and hip roll drawn over training** (#644; the maintainer's
  decisions of 2026-10-25 14:13 UTC in the stance run log, items (2) and
  (4);
  `environments/shared/reporting/training_history.py`; documented in
  `docs/TRAINING_ARTIFACT_PRESENTATION.md`, "Training history pages").
  Report only. Two pages after the policy std page, so every earlier page
  keeps its name: `11_head_posture` (the head tip's window mean and minimum,
  with the `head_clearance` reward's target height where the stage records
  it, and the neck and head pitch) and `12_hip_roll` (per foot, the window
  mean, + abduction, and the maximum |angle from home|); the asymmetry page
  gains the left-minus-right hip roll (per episode, the left leg's mean minus
  the right's), and the checkpoint table the head's values and the
  left-minus-right hip roll. A ledger written before the columns renders
  them as labelled absent series (the captions and the JSON's `absent` say
  so), and a column the model has no site or hinge for (nan on every
  episode, as the Compsognathus robot's neck and head pitch) is drawn as
  absent with that reason. The posture panels keep a least y span (0.05 m,
  1 degree) with plain tick labels, so a statue's near-constant head tip is
  not drawn at micrometre scale. The stage-end report now holds twelve
  pages.
- **Each evaluation episode's head posture and hip roll, in the eval-episode
  ledger** (#644; the maintainer's decisions of 2026-10-25 14:13 UTC in
  the stance run log, items (2) and (4):
  head posture as report-only telemetry, no gate change; task #47's
  remaining hip-roll telemetry; `environments/shared/gait/posture_telemetry.py`,
  `environments/shared/gait/recorder.py`,
  `environments/shared/eval_diagnostics.py`; documented in
  `docs/TRAINING_ARTIFACT_PRESENTATION.md`, "Evaluation episodes"). Report
  only. The stance gate does not read where the head is: on T. rex physics
  r10 the walking parent of stance3 passed 37 of 40 clean zero-shot with its
  head tip down from 1.24 to 0.61 m and its neck pitched about +33 degrees
  (the CPU checks of 2026-10-25), and the ledger's left-right offsets stop at
  hip pitch, knee and ankle. `eval_episodes.csv` now ends with, per episode
  and over the gate's window `[settle_steps, T)`:
  `head_tip_height_mean_m` and `head_tip_height_min_m` (the `head_tip` site,
  else `snout_tip`; bit for bit the env's own `info["head_tip_z"]` on flat
  ground), `neck_pitch_mean_deg` (the sum over the neck's pitch hinges) and
  `head_pitch_mean_deg` (each from its home angle, + nose down), and per foot
  `<label>_hip_roll_mean_deg` (from home, + abduction on either side: the
  mirror convention) and `<label>_hip_roll_max_abs_deg`. The recorder gains
  an opt-in `record_sites` (one `site_xpos` read per control step, behind
  `report_extensions`); `PostureLayout.from_env` fixes each hinge's sign at
  the home keyframe on a scratch `MjData`; what a model lacks reads `nan`
  for a stated reason (no head-tip site, no neck hinge, no `head_pitch`; the
  Velociraptor Mongoliensis (Anatomical) and the Brachiosaurus Altithorax
  (Anatomical) run no evaluation recorder, so their ledgers carry none of
  these columns and the callback names none: its `posture_columns` is empty
  there, the gap the maintainer's decision of 2026-10-25 17:42 UTC keeps). Validated
  (measured) on the r9 tree
  (`3ffc18e9` with this change) with stance3 Run 2's final, the walking
  parent: over the stance panel's 40 episodes the neck and head pitch
  reproduce the zero-shot check's -2.158 and +0.142 degrees to 9e-16 per
  episode, and the head tip reads 1.24084 m against the check's 1.24086
  (its forward kinematics at the control boundary; this column is the
  boundary substep's, one physics step earlier: at most 3.8e-5 m per
  episode in the window mean and 2.9e-4 m in the window minimum); on the
  r10 statue the window means read 0.959 m, 12.27 and 12.64 degrees; and
  on physics r10 the same parent reads the dropped head the gate missed
  (measured in review over the same 40 episodes): head tip 0.6126 m, neck
  +32.86 and head +33.00 degrees, the zero-shot check's values (its
  forward-kinematics head tip 0.61255 m; the pitches equal to every
  printed digit). What moves: eight ledger columns on the T. rex (a row from
  68 to 76 columns, about 155 bytes more), appended after every other
  column. What does not: no gate verdict, reward, observation, action,
  termination, task, checkpoint selection, plant, hyperparameter or
  stage-config digest, no `StanceEpisodeMetrics` field, panel column,
  balance key, `gate_progress.npz` key, `progress.json` block, env info key
  or MJCF sensor, and no random draw; the in-suite inertness proof's full run
  carries the columns and still trains and evaluates exactly like the plain
  run, and the digest snapshot is unchanged.
- **The T. rex tail_3 bound screen of D-D43 run, scored and verified: neither
  bound is adopted (outcome C); the options study and the no-training probes
  recorded, report only** (#641; D-D43's amendment of 2026-10-25;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2, `docs/CONSOLIDATION_PLAN_2026_09.md`
  §6, a dated note in §17 of `docs/GAIT_QUALITY_PLAN_2026_09.md`,
  `docs/NEXT_STEPS.md` §3; the appended §24 of
  `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md`, and
  `stance5/tailscreen/` of `docs/investigations/stance_2026_10/`). Records
  only: no library code, configuration, test, notebook, digest or generated
  file changes with it. A scratch CPU continuation screen of a training-loss
  bound on `tail_3_pitch_act`'s Gaussian mean, upper side only, coef 2.4, in
  three arms (no bound, b = 0.90, b = 0.60) and 18 continuations of 1,048,576
  steps from the 4,194,304 checkpoints of `stance5_LTSB_s42` and
  `stance5_LTSB_s80` on seeds 101, 202 and 303, from two scratch commits on
  `cf80ecb1` (never pushed; kept here as a patch file), judged by criteria
  approved before any training step (sha256 `9930ee72…`, re-pinned once for one
  gate test). Run on 2026-10-24 on the session's CPUs, scored that day, final,
  and verified on 2026-10-25: BOUND90 NOT ADOPTED (RAIL: every replicate had
  tail-up saturation episodes at R1 and R2, s42's from a railed start and s80's
  from a clean one), BOUND60 NOT ADOPTED (PARKED: the rail cleared, with R1/R2
  mean commands of 0.600 to 0.632, at or above the 0.60 line; s42's three clear
  it by as little as 0.00017, and the verdict also holds on s80 alone), CONTROL
  RAIL, CURED and RAIL from s42 and RAIL three times from s80; OTHER 0
  everywhere. Report only: the bound acted on the training rows (the share of
  rows above b lower under the bound in every seed-paired comparison), and the
  dose ordering of the arms' mean commands is false at both states. Before it,
  also report only: the options study's draft and corrected briefs (two
  skeptics, 17 findings) and no-training probes on the two checkpoints (holding
  the tail at 0 gains reward under the policy's own noise in 40 of 40 pairs;
  the unclipped tail settles at a fixed point of a positive loop; the policy
  probe's bank is not an early detector). The maintainer's decision after the
  screen: this records PR first, then the posture options with evidence, before
  any decision. `stance5/tailscreen/` holds the criteria (byte-identical), the
  approval and its request, the stale pin list, the calibration, the patch, the
  gate records, the tools, the score, `panels.csv`, the verification and the
  run log's lines 505 to 577 (ids removed), and `options/` and `probes/` the
  study and the probes. `SOURCES.txt` and `SHA256SUMS` (now 528 files) are
  extended. `docs/README.md` adds a note to the record's and the gait plan's
  rows, and `docs/KNOWN_ISSUES.md` gains Update 19 of the T-Rex stance entry.
- **Decision D-D43 recorded: a scratch CPU continuation screen of a bound on
  the T. rex tail actuator's mean, the first step of D-D42's round; D-D42
  amended with three more report-only tools** (#641; the maintainer's
  decisions of 2026-10-24; `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2,
  `docs/CONSOLIDATION_PLAN_2026_09.md` §6, a dated note in §17 of
  `docs/GAIT_QUALITY_PLAN_2026_09.md`, `docs/NEXT_STEPS.md` §3; §24.1–§24.4 of
  `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md`). Records
  only. At 16:00 and 17:46 UTC, on the options study's corrected brief, the
  maintainer decided no-training probes first and then a scratch CPU
  continuation screen of a training-loss bound on `tail_3_pitch_act`'s
  Gaussian mean, upper side only: three arms (no bound, b = 0.90 as a rail
  guard, b = 0.60 as a posture bound), three replicates on each of two band
  checkpoints, one continuous 1M-step run each, a durable copy on an orphan
  branch never merged, and a tail parked below the rail (a command of 0.60 or
  more with the rail clear) counting as not adopted and going to the posture
  decision. The criteria were approved at 19:06 UTC with coef 2.4 for both
  bounds, before any training step, and re-pinned at 19:10 UTC after the
  maintainer had one gate test deselected (`main`'s rule against stage configs
  that set `[ppo] symmetry*`, which fails on the band commit itself); D-D43 is
  amended for that re-pin. D-D42 is amended with the maintainer's report-only
  items of 15:56 UTC: the band runs' `policy_probe.npz` read from Drive, the
  per-step trace tool, the training rollouts' clip shares and the stage-end
  training-history report, in their own pull requests. Nothing trains on
  `main`. `docs/KNOWN_ISSUES.md` gains Update 18 of the T-Rex stance entry.
- **The saturating actuator, named per evaluation episode and drawn over
  training** (#640; the maintainer's decision of 2026-10-24, "both, from the
  ledger"; `environments/shared/gait/stance_metrics.py`,
  `environments/shared/eval_diagnostics.py`,
  `environments/shared/reporting/training_history.py`; documented in
  `docs/TRAINING_ARTIFACT_PRESENTATION.md`, "Evaluation episodes" and
  "Training history pages"). Report only. `max_actuator_saturation_fraction`
  flags an episode without saying which actuator; naming the tail held at +1
  (`tail_3_pitch_act`) that failed four runs took checkpoint downloads and
  offline rollouts. `eval_episodes.csv` now carries, per episode,
  `saturated_actuator` (the actuator with the largest fraction of post-settle
  window steps at `|action| >= 0.99`, the lowest index on a tie; empty when
  none saturates), `saturated_actuator_fraction` (the screen's metric bit for
  bit: one shared reduction, the same window, threshold and NaN rule),
  `saturated_actuator_positive_share` (its sign), `saturated_actuator_ties`
  and `actuators_saturated`, read from the trace the recorder already closed.
  The training-history report counts each evaluation's flagged episodes by the
  actuator and sign they name (never smoothed: stacked per evaluation on one
  run's page, a cell coloured by the most-named actuator per run on a
  comparison), draws `saturated_actuator_fraction` against the stage's bar,
  labels each alarm with its actuator (the one its flagged episodes name most,
  counted over its signs), and carries the counts per evaluation, per alarm
  and in the checkpoint table, so `final`, `best` and `robust_best` are read
  from their own evaluation's rows. What moves: five ledger columns
  (a T. rex row from 63 to 68 columns, about 13 to 50 bytes) and the report's
  saturation page, JSON and HTML. What does not: no gate verdict, reward,
  observation, action, termination, task, checkpoint selection, plant,
  hyperparameter or stage-config digest, no `StanceEpisodeMetrics` field,
  panel column, `gate_progress.npz` key, env info key or MJCF sensor, and no
  random draw; the digest snapshot is unchanged. An older ledger without the
  columns loads everywhere and reads as an absent series.
- **The training rollouts' clip shares, per actuator, in the policy probe and
  on the saturation page** (#640; the maintainer's decision of 2026-10-24;
  `environments/shared/policy_probe.py`,
  `environments/shared/reporting/training_history.py`; documented in
  `docs/TRAINING_ARTIFACT_PRESENTATION.md`, "The policy probe" and "Training
  history pages"). Report only. SB3 clips a sampled action only at
  `env.step` and trains on the unclipped sample and its log-probability, while
  deterministic play acts on the mean, clipped; in a scratch T. rex experiment
  `tail_3_pitch_act`'s mean was beyond +1 on 2.5% and 27.5% of two arms'
  training rows against 21 to 46% of deterministic window steps, and nothing
  in the repo recorded the training side. `PolicyProbeCallback` now reads, at
  every rollout's end and before `train()`, the rollout buffer that update
  trains on: per actuator, the share of sampled actions above the high bound
  and below the low one, the share at `|clip(a)| >= 0.99` (the gate's
  threshold), and, from one `torch.no_grad()` pass of the buffer's
  observations through the policy's distribution (the bank's pass; float32,
  in chunks of 16,384 rows), the share of means beyond each bound and the
  means' average, median and 95th percentile. Nine `rollout_*` arrays join
  `policy_probe.npz` (about 0.5 KB per update on the T. rex, 0.7 MB over a
  12M run), three `policy_probe/rollout_*_share_max` scalars join
  TensorBoard and `progress.json`'s `policy`, and a same-stage resume from a
  file written before them keeps its rows as NaN; a buffer pass that fails
  costs a row its `rollout_*` arrays only. It added 41 to 46 ms per rollout
  on the T. rex stance config (4 envs, `n_steps` 2048, one CPU thread),
  about 0.25% of the rollout's collection and update.
  The training-history report draws both shares per rollout on the saturation
  page (the actuators that ever reach 0.01 at a bound named, at most six, in
  their flagged episodes' colour or one no flagged actuator holds, the rest
  one grey line; a comparison draws each run's largest, labelled with the
  actuator it traces; rollouts a resume did not measure are said on the
  panel; never smoothed) and adds `rollout_shares` (each actuator's peak,
  last value and first rollout at 0.01) to the JSON; a run without the
  arrays is an absent series.
  No sampling, gradient, parameter, optimizer, buffer, generator, training
  flag, gate, reward, observation, action, task, checkpoint selection, plant,
  hyperparameter or digest change: the in-suite inertness proof, which trains
  a short T. rex stance with the probe and without it, now also pins the
  arrays in its full run and still sees the same rollout data, updates,
  generators, evaluations and checkpoints, and the digest snapshot is
  unchanged.
- **The training-history report drawn at the end of every stage, report-only**
  (#639; the maintainer's decision of 2026-10-24, which widens D-D42's (2) to
  scope (b); `environments/shared/reporting/stage_artifacts.py`,
  `environments/shared/reporting/stage_layout.py`; documented in
  `docs/TRAINING_ARTIFACT_PRESENTATION.md`, "At stage end:
  `figures/training_history/`", and one sentence in tip 4 of
  `website/docs/training/hyperparameters.md`; a dated amendment of D-D42's rows
  in `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2 and
  `docs/CONSOLIDATION_PLAN_2026_09.md` §6). D-D42 (its entry below) sent the
  report to its own pull request at scope (a), the standalone report as built,
  which landed in #636; a further decision of the maintainer's that day widens
  it to scope (b), as the stage report's design of 2026-10-18 specifies, and
  D-D42's rows now say so. `generate_stage_artifacts` now draws the stage's
  training history from its ledgers into `figures/training_history/`
  (the ten pages, `training_history.html` and `training_history.json`, as
  `plot_training_history.py STAGE_DIR` draws them by default), so every node
  the SB3 training notebook trains or judges has it without a manual step, and
  the bundle saved after the node declares and hashes it. It is drawn after the
  verdict is recorded and the stage summary written, after
  `figures/gate_progress.png` and in its own guard, before the replays and the
  stance probes: a failure logs one warning and costs this report only. It is
  rendered into a private directory and moved into the staging tree whole, so a
  failure part-way publishes none of it; a stage judged again whose report is
  not drawn keeps the previous copy, as the other figures do. It follows
  `generate_graphs`, and a stage directory with neither `eval_episodes.csv` nor
  `train_episodes.csv` (a run from before the ledgers) gets none. The library writes it, not the
  script, whose refusal of an `--out` inside a run directory still keeps manual
  writes out of runs. Report only: no training, evaluation, gate, reward,
  observation, action, task, checkpoint selection, plant, configuration, digest
  or notebook change, and `training_history.py` is unchanged; the results
  `generate_stage_artifacts` returns, the verdict and the stage summary are the
  same with the report, without it and when it fails, and the std reads leave
  the random number streams the replays and probes see as they were. The
  ordering pins in `test_reporting_stage_artifacts.py` and
  `test_stance_gate_report.py` gain the new step. On the stance5 band run s42's
  ledger (91 evaluations) it takes about 5.5 s and writes 1.7 MB.
- **A per-step trace of single episodes of a checkpoint, and the policy's
  command and raw mean in the replay's stance CSV, report-only** (#638; the
  maintainer's decision of 2026-10-24, after the T. rex tail habit,
  `tail_3_pitch_act` commanded at +1, took scratch tools to measure in
  `docs/investigations/stance_2026_10/stance4/satdiag/`). The stance gate
  flags actuator saturation, but nothing in the repository showed per step
  what the policy asks for. Nothing reads either output but a person: no
  gate, reward, observation, action, task, checkpoint selection or plant
  changes, and the 649-line digest snapshot (plant identities, policy
  interfaces, task and gate digests, reward captures) is byte-identical.
  - `environments/shared/scripts/episode_trace_report.py` (library:
    `environments/shared/reporting/episode_trace.py`) rolls a checkpoint and
    its named VecNormalize statistics, loaded frozen as the stance report
    loads them and checked against the current plant, on the stage's own
    env, one episode per seed. Its default rollout is the stance judge's:
    the trace of seed `3042 + i` sends the command episode `i` of the
    stance panel sends, at every step, on the v1 and v2 panels and the
    heading certificate's 0 degree cell, with the same rewards and lengths
    (checked against `run_panel`). Per step and per actuator it writes the
    raw Gaussian mean before the clip (`get_distribution(obs).mode()` under
    `torch.no_grad()`, the policy probe's pass), the command, `data.ctrl`,
    the driven joint's position and velocity, and the actuator force and
    its fraction of the force limit, one CSV per episode. `episode_trace.json`
    (`mesozoic.episode-trace/v1`) summarizes each episode and traced
    actuator over the gate's post-settle window at its `SATURATION_ABS`:
    window means, the shares of steps at `|command| >= 0.99` and `>= 0.9`
    and with `|raw mean| > 1`, the raw mean's extremes (and, apart, the
    whole episode's, the reset transient included), the first step at the
    rail (anywhere and in the window), and the episode's return and
    length. Its `step` is the action's 0-based index; the replay's stance
    CSV counts from 1, so its row k is the trace's row k - 1. `--actuators`
    narrows it; `--stochastic SEED` samples from the policy's own Gaussian
    (`numpy.random.default_rng(SEED + i)`, the stance screens'
    `noise_matched_own` draw when `SEED` is the first reset seed);
    `--hold NAME=VALUE` replaces one actuator's command, for counterfactuals
    such as holding `tail_3_pitch_act` at 0. Both are recorded as probes.
    A missing `--vecnorm` is refused unless `--allow-unnormalized`, and an
    output directory inside the run directory of the checkpoint or its
    statistics is refused, as `plot_gate_progress.py` refuses one.
  - The replay's per-frame `*_stance.csv` (`record_stage_video`) gains, per
    actuator, `<actuator>_command` (the action sent), `<actuator>_raw_mean`
    (the Gaussian mean before the clip; absent for SAC) and
    `<actuator>_joint_pos` (`stance_diagnostics.policy_command_columns`).
    The raw mean is a second forward pass of the observation `predict` was
    given, which draws no random number and leaves the policy's
    distribution as it found it: the replay's observations and actions,
    and so its video and every other column, are those of a replay without
    it (tested on a real T. rex checkpoint). It costs about 0.15 ms per
    frame on a [512, 256] policy, against about 1.2 ms per T. rex physics
    step. No column the snapshot writes on any of the six models ends in
    these names, and nothing in the repository parses the CSV's columns.
  - Tests: `test_episode_trace.py` (the judge's rollout step for step, the
    columns and the summary against the plant and the gate's own saturation
    function, both probes, the writer, the refusals and the script), the
    replay's columns in `test_evaluation.py`, and the columns' names and
    reads on every model in `test_stance_diagnostics.py`.
    `test_episode_trace.py` needs SB3, so the SB3 job's integration step runs
    it, and `test_repository_root.py` lists the script's `sys.path`
    bootstrap. Documented in `docs/TRAINING_ARTIFACT_PRESENTATION.md`.
- **Decision D-D42 recorded: the next T. rex stance round targets the tail
  saturation habit, with a training-history report and an actuator saturation
  report, both report only; D-D41 amended with its outcome** (#637; the
  maintainer's decisions of 2026-10-24; `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2,
  `docs/CONSOLIDATION_PLAN_2026_09.md` §6, a dated note in §17 of
  `docs/GAIT_QUALITY_PLAN_2026_09.md`, `docs/NEXT_STEPS.md` §3; §23.7 of
  `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md`). Records only:
  no library code, configuration, test, notebook, digest or generated file
  changes with it. After D-D41's replication read its line 2 (the entry below),
  D-D42 takes the first of the three options that line offers: the next round
  targets the tail saturation habit (`tail_3_pitch_act` pinned at +1 on the
  saturation rail), and is designed with fixed criteria and brought back to the
  maintainer for approval before any training. The training-history report
  goes to its own pull request at scope (a), the standalone report as built
  (report-only pages from the run ledgers). An actuator saturation report,
  during training and for finished models, comes from report-only
  telemetry naming the saturating actuator and its sign on each
  evaluation-episode row. No training follows from it, and when the copied run
  log ends (15:27 UTC) the round is not designed: an options study for it has
  started, with no training and no Drive fetch, its brief to come to the
  maintainer before any design is fixed; the two reports are to come in their
  own pull requests, not in this change. Walking pilot 2 is not launched.
  D-D41's row gains a dated amendment with its package, runtime and outcome
  (line 2, not adopted). Screens certify nothing. `docs/README.md` adds a note
  to the gait plan's row, and `docs/KNOWN_ISSUES.md` gains Update 17 of the
  T-Rex stance entry.
- **The stance5 replication of D-D41 judged: one of the two new per-foot
  centre-of-pressure band runs holds (line 2), and the band is not adopted; the
  K3 diagnosis, the scratch tail experiment and the K3 readouts recorded, report
  only** (#637; D-D41's amendment of 2026-10-24;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2, `docs/CONSOLIDATION_PLAN_2026_09.md`
  §6, a dated note in §17 of `docs/GAIT_QUALITY_PLAN_2026_09.md`,
  `docs/NEXT_STEPS.md` §3; the appended §23
  of `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md`, and
  `stance5/replication/`, `stance5/k3diag/` and `stance5/tailexp/` of
  `docs/investigations/stance_2026_10/`). Records only: no library code,
  configuration, test, notebook, digest or generated file changes with it. The
  two 12M runs of arm LTSB, unchanged (`stance5_LTSB_s70`, `stance5_LTSB_s80`),
  trained from scratch to 12,001,280 steps on Colab L4 GPU runtimes from the
  screen branch `claude/screen-trex-r9-armlts-copband` (`cf80ecb1`, never
  merged) and were judged on 2026-10-24 by the criteria fixed before training
  (sha256 `a38536de…`), final, with the three D-D40 runs counted as recorded.
  s70 holds: 40, 40, 40, 39, 40 and 40 of 40 clean, a K2c range of 13.7 mm, 0
  and 1 of 20 joint-space noise falls at 6,291,456 and the final. s80 fails K1
  only: 40, 4, 14, 26, 39 and 38 of 40 clean, the three checkpoints below 33 of
  40 on the actuator-saturation rail (36, 25 and 14 episodes; the alarm at
  4,456,448, `tail_3_pitch_act` pinned at +1, as in D-D40's seed-42 run), and 1
  and 2 of 20 joint-space noise falls. H = 2 of 5, and the decision table's line
  is 2, final: not adopted; the maintainer chooses the next round from the
  stance5 criteria's menu. Report only: the K3 pass fraction (3 of 5 against
  the control's 2 of 3), not a veto; s52's final still ties D-D37 Run 2's final
  under the parent rule; the notebook's own six-heading gate on each run's
  `robust_best` (both PASS) and the stage's six-heading certificate rolled
  offline on the finals (40 of 40 and 38 of 40 seeds, both PASS; not gate
  evidence). Also recorded, report only: a diagnosis of the D-D40 band runs'
  joint-space noise falls (seed 42's are not the tail habit; hip-pitch noise,
  mostly its left-right differential part, is the main cause; the mechanism of
  the break between 6,291,456 and 7,340,032 is open); a scratch experiment on
  the tail habit (a quadratic bound on the policy mean, one 1M-step CPU
  continuation against a paired control: lower tail saturation, relapsing, its
  effect against training alone not established; the scratch clone never
  pushed); and report-only K3 readouts on the new runs, fixed by an addendum
  before any of their J checkpoints existed (full-noise falls 0, 1, 1 and 2 of
  20; every band final falls 20 of 20 to a ±0.3 differential hip kick).
  `stance5/replication/` holds the criteria, the Colab steps, the sigma sidecar,
  frozen copies of the judge and scorer, `results/` (13 files and its README,
  about 0.16 MB: the run log, tool ids removed, the verdict of record, the
  notebook's gates and the certificates, trimmed, and the saturation watch's
  files) and the K3 readouts' addendum, tool, validation and outputs;
  `stance5/k3diag/` and `stance5/tailexp/` the two reports. `SOURCES.txt` and
  `SHA256SUMS` (now 418 files) are extended. `docs/README.md` adds a note to the
  record's and the gait plan's rows, and `docs/KNOWN_ISSUES.md` gains Update 16
  of the T-Rex stance entry.
- **A training-history report: one run's feet, balance, asymmetry and
  saturation over its training, or several runs overlaid**
  (#636; `environments/shared/reporting/training_history.py`,
  `environments/shared/scripts/plot_training_history.py`; documented in
  `docs/TRAINING_ARTIFACT_PRESENTATION.md`, "Training history pages"). Report
  only: new files, one test-table line each in `test_policy_loading.py` (the
  new call-time read of `_PERIODIC_CHECKPOINT_RE`) and
  `test_repository_root.py` (the script's `sys.path` bootstrap), and a
  docstring line in `reporting/__init__.py`; no training, evaluation, gate,
  configuration, digest or notebook changes. From a stage directory's
  `eval_episodes.csv`, `train_episodes.csv`, `stage_config.json`,
  `progress.json`, `plant_identity.json` and checkpoint zips it reduces each
  evaluation (clean, falls, saturation-flagged episodes and the mean return;
  the rails' own per-episode numbers and the full-horizon episodes over each
  declared bar; per foot the whole-foot and pad centres of pressure, the
  episodes outside the CoP band's edges, digit share, toe margin, shear and
  cone use; the centre of mass, root pitch and XcoM margins; the
  left-minus-right hip pitch, knee and ankle offsets;
  `actuator_force_frac_max`), the training falls per 262,144-step window, and
  the per-actuator policy std `exp(log_std)` at the checkpoints (only
  `policy.pth`'s `log_std` is read, with `weights_only=True`; the actuator
  names only when the run's physics digest is the checkout's). It finds the
  saturation alarms from the ledger itself (an evaluation with at least 5 of
  its episodes listing `max_actuator_saturation_fraction`; two in a row are an
  alarm) and marks them, with the checkpoints given by `--mark-steps`, on ten
  pages of graphs written as PNGs and a self-contained HTML page, beside a
  JSON summary (`mesozoic.training-history/v1`) of every series, the declared
  bars and band settings and what was absent. Counts are never smoothed: a
  comparison draws them as one strip per run. A missing file or column is an
  absent series, never an error; an output inside a run directory is refused,
  as `plot_gate_progress.py` refuses it. It is the first piece of the stage
  report card's posture-over-training page and per-actuator std. On the
  stance5 band run s42 it finds the first flagged evaluation at 3,538,944
  steps and the alarm at 3,670,016, and on the stance4 control run s42 the
  first saturated episodes at 1,835,008 and the alarm at 2,228,224, the steps
  the scorers recorded. On the stance4 control run s52 it counts 204
  full-horizon episodes over the 0.80 pad rail in 69 of 91 evaluations (7 of
  10 at 6,553,600 steps, where the signed per-foot means read +0.14 and
  +0.11); the five band runs have none.
- **Decision D-D41 recorded: a replication of the per-foot centre-of-pressure
  band screen on two more seeds, read three of five; D-D40 and D-D36 amended**
  (#634; the maintainer's decisions of 2026-10-22;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2, `docs/CONSOLIDATION_PLAN_2026_09.md`
  §6, a dated note in §17 of
  `docs/GAIT_QUALITY_PLAN_2026_09.md`, `docs/NEXT_STEPS.md` §3; the appended §22
  of `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md`). Records
  only: no library code, configuration, test, notebook, digest or generated file
  changes with it. After D-D40's screen read its line 3 (the entry below), D-D41
  replicates the unchanged band recipe (the screen branch at `cf80ecb1`) on two
  band runs only, seeds 70 and 80, judged on K1–K4 against the existing D-D39
  control, not retrained, by new criteria fixed before training (12M steps,
  physics r9, `n_envs` 4 and a Colab L4 runtime follow from "unchanged";
  inferred). The band is adopted only if both new seeds hold (3 of the 5 band
  runs); the K3 pass fraction against the control is reported, not a veto (the
  log's reason: under stance5's rule (b), the credit rule of D-D40's screen
  criteria, adoption would be impossible, the band arm already having two K3
  failures). The new seeds' joint-space noise sigma is the element-wise mean
  of the three control seeds' `control_sigma`: the session's default, stated to
  the maintainer with the reading rule, then confirmed by the maintainer.
  Walking pilot 2 keeps D-D37 Run 2's final as its parent for now, and D-D40's
  s52 final, which ties it, is reconsidered after the replication. Run-handling
  rules for the replication's restarts, resumes and refusals were adopted the
  same day. Its package is built and is being re-fixed before training; nothing
  has trained. Two report-only workflows the maintainer agreed to the same day
  (a diagnosis of the band runs' joint-space noise falls, and a scratch
  experiment on the tail habit) have no result yet. Two rows gained dated
  amendments. D-D40's records its build (band off byte-identical to the D-D39
  commit; a band-on overhead of +0.245 to +0.378 ms per env step, so the build's
  own 0.15 ms goal was not met), its package and the screen branch's push at
  `cf80ecb1`, which supersedes "not committed anywhere public", its Colab L4
  runtime and its outcome (line 3). D-D36's, dated 2026-10-21, records walking
  pilot 2's rebuild for D-D37 Run 2's final: its criteria amended before any
  pilot-2 training step, the parent committed on its branch, its start-up checks
  re-run, and the branch pushed at `8c5478a6` with the maintainer's OK and held;
  not yet trained, and its parent kept by D-D41. Screens and pilots certify
  nothing. `docs/README.md` adds a note to the record's and the gait plan's
  rows, and `docs/KNOWN_ISSUES.md` gains Update 15 of the T-Rex stance entry.
  The replication's outcome is not yet known.
- **The stance5 screen of D-D40 judged: one of the three per-foot
  centre-of-pressure band runs holds (line 3), and the band is not adopted; the
  band's build, the stance4 seed-42 saturation diagnosis and walking pilot 2's
  rebuild recorded** (#634; D-D40's amendment of 2026-10-22;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2, `docs/CONSOLIDATION_PLAN_2026_09.md` §6,
  a dated note in §17 of `docs/GAIT_QUALITY_PLAN_2026_09.md`,
  `docs/NEXT_STEPS.md` §3; the appended §21 of
  `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md`, and
  `stance5/`, `stance4/satdiag/` and three files in `pilot2/` of
  `docs/investigations/stance_2026_10/`). Records only: no library code,
  configuration, test, notebook, digest or generated file changes with it. The
  three 12M runs of arm LTSB (D-D39's arm LTS plus a per-foot centre-of-pressure
  band at weight 0.30; `stance5_LTSB_s52`, `stance5_LTSB_s60`,
  `stance5_LTSB_s42`) trained from scratch to 12,001,280 steps on Colab L4 GPU
  runtimes from the screen branch `claude/screen-trex-r9-armlts-copband`
  (`cf80ecb1`, pushed, never merged) and were judged on 2026-10-22 by the
  criteria fixed before training (sha256 `3b28963f…`), final. s52 holds: 40 of
  40 clean at every judged checkpoint, K2a and K2b 0 throughout, no noise falls
  at 6,291,456 or the final (1 of 20 joint-space at 8,388,608 and at 10,485,760,
  which K3 does not read). s60 fails K3 only: 3 of 20 joint-space noise falls at
  the final (2 allowed).
  s42 fails K1 and K3: 15 to 38 of 40 clean, the actuator-saturation rail
  failing in most of its failing episodes (the alarm at 3,670,016;
  `tail_3_pitch_act` pinned at +1, as in D-D39's seed-42 run and D-D37 Run 2),
  and 13 of 20 joint-space noise falls at the final. H = 1 and K2p = 3; the band
  arm passes K1, K2 and K4 at least as often as the control (the three D-D39
  runs, not retrained) but K3 less often (1 of 3 against 2 of 3), so the credit
  rule fails, and the decision table's line is 3, final: not adopted, "replicate
  on 2 more seeds before any commit". s52's final ties D-D37 Run 2's final under
  the parent rule (40 of 40, no noise falls), report only. Screens certify
  nothing. Report only: the fresh, tail and common-noise panels on the finals
  (s52 FAIL 38 of 40 on the fresh panel, with two hop-or-fall episodes), the
  notebook's own six-heading gate on each run's `robust_best` (all three PASS),
  the stage's six-heading certificate rolled offline on the s52 and s60 finals
  (40 of 40 and 39 of 40 seeds, both PASS; not gate evidence), the diagnostic
  panels at every judged checkpoint, and the band readouts: one full-horizon
  episode of a band run had a foot's window-mean centre of pressure out of the
  band (s60 at 8,388,608), against 10 to 28 of 40 per judged checkpoint in the
  control's seeds 52 and 60. Also recorded: the band's build, D-D40's three
  fixes and the review's fixes on a scratch branch (band off byte-identical to
  D-D39's commit; the band's readings equal to the recorder's on 16,000
  foot-steps; a band-on overhead of +0.245 to +0.378 ms per env step, so the
  build's own 0.15 ms goal, which no decision set, was not met), and the screen
  branch's push, which supersedes the D-D40 entry's "not committed anywhere
  public" and the prototype's "never pushed"; the offline, report-only diagnosis
  of D-D39's seed-42 saturation (the tail-tip actuator `tail_3_pitch_act`
  commanded at +1 from between 1,048,576 and 2,097,152 steps, the command pinned
  but not the joint or the motor; holding it at 0 gains +85.9 and +73.1 per
  800-step window; candidate fixes listed, none acted on); and walking pilot 2
  rebuilt on 2026-10-21 for D-D37 Run 2's final as its parent (its criteria
  amended before any pilot-2 training step, the parent committed on
  `claude/pilot-walk-r9-b`, its start-up checks re-run, and the branch pushed at
  `8c5478a6` with the maintainer's OK and held until its seeds are judged; not
  yet trained). `stance5/` holds the criteria, their notes, the Colab steps and
  frozen copies of the judge and scorer; `stance5/results/` (22 files and its
  README, about 0.56 MB) the run log (tool ids removed), the verdict of record,
  a per-checkpoint summary (`checkpoints.csv`, 36 rows, with its script), the
  extras, the notebook's gates and the certificates (trimmed), the panels'
  summary with its script, and the saturation watch's files; `stance4/satdiag/`
  the diagnosis' report and five tables; and `pilot2/` gains the amended
  criteria and Colab steps and the rebuild's summary beside the copies of
  2026-10-18. `SOURCES.txt` and `SHA256SUMS` (now 381 files) are extended.
  `docs/README.md` adds a note to the record's and the gait plan's rows, and
  `docs/KNOWN_ISSUES.md` gains Update 14 of the T-Rex stance entry.
- **Decision D-D40 recorded: a from-scratch T. rex stance screen of a per-foot
  centre-of-pressure band; D-D39 and D-D36 amended** (#633; the maintainer's
  decisions of 2026-10-20; `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2,
  `docs/CONSOLIDATION_PLAN_2026_09.md` §6, a dated note in §17 of
  `docs/GAIT_QUALITY_PLAN_2026_09.md`, `docs/NEXT_STEPS.md` §3; the appended §20
  of `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` and
  `docs/investigations/stance_2026_10/stance4/copband/design.md`). Records only:
  no library code, configuration, test, notebook, digest or generated file
  changes with it. After D-D39's screen read its line 5 (the entry below), D-D40
  trains a from-scratch screen of a per-foot centre-of-pressure band rather than
  the v6b ankle spring: a new reward term, an opt-in `[env]` option of the T.
  rex env, off by default, that after the settle charges a foot whose whole-foot
  centre of pressure lies outside [−0.6, +0.4] span half-lengths, or whose pad
  centre of pressure lies outside [−0.65, +0.65], up to half the term's weight
  per step, with guards that give an unloaded or digit-borne foot the full price
  and a lightly loaded one a ramped price (the first fix below). It is the
  design note's prototype (a scratch commit, never pushed) with three fixes and
  its weight: the 35% load-share guard becomes a ramp (the full price at a load
  share of 0.25 or less, no guard price at 0.35 or more, linear between); a
  speed pass, one fused per-step reduction pinned byte-identical to the
  recorder's readings (the prototype measured +0.62 ms per env step over
  band-off; no numeric target was decided); the contact reduction moves out of
  `environments/shared/gait/` into a neutral shared module, so that D-D23's gait
  layering rule holds with no exception, and the test that pins its readings to
  the recorder's moves with it; and weight 0.30. Three 12M runs train from
  scratch on seeds 52, 60 and 42 on Colab L4 GPU runtimes (the seeds and the 12M
  from the option text the maintainer locked in the chat; physics r9 and
  `n_envs` 4 follow from the arm), judged by criteria fixed before training,
  with K1–K4 unchanged. The arm is D-D39's (L+T plus the stagger penalty, 0.30
  and 0.045 m) plus the band, so the three D-D39 runs are its control, one
  difference apart; the design note had recommended L+T plus the band alone, and
  the recommendation changed because the overnight band test ran on stagger
  parents and the stagger term prices the route the note calls its main risk
  (each foot's centre of pressure held in band by asymmetric ankle torque while
  the feet stay staggered). The v6b ankle spring (ankle stiffness 40 → 250
  N·m/rad, damping 45 → 90, the spring reference at the statue's settled 78.6°;
  a plant revision) is held in reserve behind a trigger line in the band
  screen's decision table: the band fixes the split but runs fail K2 on drift, a
  common fore-aft shift. Actuator saturation gets no training change: the band
  screen's evaluation ledgers are read at every check-in and an alarm is
  reported at once (run, step, actuator), and an offline, report-only diagnosis
  of the saturation of `stance4_LTS_s42` runs in parallel, started after the
  band build's timing measurement; fails stay fails, and no run is stopped or
  restarted on an alarm. Mirrored-leg spawns and the prey straight ahead stay
  diagnostic panels, run at every scored checkpoint, report only. The band's
  code, criteria, judge and Colab steps are being built separately and are not
  committed anywhere public. The same day two rows were amended. D-D39's records
  its outcome and the maintainer's confirmation of the Colab L4 GPU runtime for
  all three of its runs, which closes the confirmation its entry left pending.
  D-D36's records that walking pilot 2 is to be rebuilt for D-D37 Run 2's final
  as its parent: its criteria, which still name r9 arm L seed 60 at 6M, are
  amended before any of its training steps, the new parent is committed on its
  branch and its start-up checks are re-run, right after the band screen's
  package is ready, and the maintainer starts it when a fourth Colab slot is
  free. Screens and pilots certify nothing. `docs/README.md` adds a note to the
  record's and the gait plan's rows, and `docs/KNOWN_ISSUES.md` gains Update 13
  of the T-Rex stance entry. The band screen's outcome is not yet known.
- **The stance4 screen of D-D39 judged: all three runs fail their fixed criteria
  (line 5); the stagger penalty is not adopted; two overnight scratch
  experiments recorded, report only** (#633; D-D39's amendment of 2026-10-20;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2, `docs/CONSOLIDATION_PLAN_2026_09.md` §6,
  a dated note in §17 of `docs/GAIT_QUALITY_PLAN_2026_09.md`,
  `docs/NEXT_STEPS.md` §3; the appended §19 of
  `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` and the folders
  `results/`, `night/`, `night2/` and `copband/` of
  `docs/investigations/stance_2026_10/stance4/`). Records only: no library code,
  configuration, test, notebook, digest or generated file changes with it. The
  three 12M runs of arm LTS (D-D37's L+T plus the stance stagger penalty, weight
  0.30, tolerance 0.045 m; `stance4_LTS_s52`, `stance4_LTS_s60`,
  `stance4_LTS_s42`) trained from scratch to 12,001,280 steps on Colab L4 GPU
  runtimes (the `gpu` block of each run's `stage_config.json`) and were judged
  on 2026-10-20 by the criteria fixed before training (sha256 `978a93da…`),
  final. s52 and s60 fail K1 and K2: five of their six judged checkpoints are
  below 33 of 40 clean (21 to 30), with K2a 10 to 19 of 40 at each; both reach
  39 of 40 at the final. s42 fails K1–K4: 0 to 17 of 40 clean, the
  actuator-saturation rail failing in most panel episodes up to the final (which
  actuator is not recorded), and 8 of 20 joint-space noise falls and 4 of 40
  roll-rail failures at the final. H = 0 and K2p = 0, the credit rule fails
  against the control (D-D37's Runs 2 and 3, not retrained: K1 0 of 3 against 1
  of 2, K3 and K4 2 of 3 against 2 of 2), and the decision table's line is 5,
  final: the stagger penalty is not adopted, and the next round tests a per-foot
  centre-of-pressure band term or the v6b ankle spring. The best stagger final,
  s52's (39 of 40, no noise falls), is one episode short of D-D37 Run 2's final
  (40 of 40), which stays the walking pilot 2 parent the criteria offer. Screens
  certify nothing. Report only: the 1,048,576-step and selection checkpoints,
  the fresh, tail and common-noise panels on the finals, the notebook's own
  six-heading gate on each run's early `robust_best` (s52 and s42 PASS, s60
  FAIL), diagnostic panels (mirrored leg noise, the prey straight ahead, both)
  at every judged checkpoint of both arms, and the readouts: by 4,194,304 the
  penalty had lined the feet up (window stagger sd 1.2–4.8 mm, against 11.9 for
  D-D37 Run 2), but in s52 and s60 the left-right split stayed as a per-foot
  loading split, still driven by the spawn's leg asymmetry and the prey's
  bearing, and s52's stagger grew again later (7.3–12.7 mm). Report only too,
  entering no verdict (the new runs' judge files are relaxed-binding): a
  pre-registered replicate study of the stagger penalty in continuation from
  D-D37 Run 3's 8,388,608 checkpoint on two new seeds (stagger n = 3 against no
  change n = 4), whose reading is line (2), partial, met through a flagged run:
  the window stagger sd separates completely (one-sided p 0.029, optimistic
  because three of the seven runs suggested the idea; 0.086 given their known
  order) and K2a does not (p 0.200); and a pre-registered paired continuation
  test of a per-foot centre-of-pressure band prototype (an opt-in term, off by
  default, on a scratch commit never pushed) from s52's and s60's 4,194,304
  checkpoints, whose reading is line (2), partial: the band runs are lower on
  K2a (0 against 18 and 3 against 22) and on the split's sd for both parents,
  but the s60 band run raised new safety flags (3 panel falls, a reward ratio of
  0.928), and the band's guards cannot see an ankle-torque route that keeps the
  stagger (in the s52 pair the band run's stagger sd was the higher, 9.4 against
  6.7 mm). `stance4/results/` (13 files and its README, about 0.32 MB) holds the
  run log (tool ids removed), the verdict of record, a per-checkpoint summary
  (`checkpoints.csv`, 51 rows, with the script that recomputes it from the judge
  files), the extras, the notebook's gates (trimmed), the panels' summary with
  its script and the split-form analysis, `stance4/night/` and `stance4/night2/`
  hold each experiment's pre-registration, review handling and readout, and
  `stance4/copband/` the band prototype's design note; `SOURCES.txt` and
  `SHA256SUMS` (now 344 files) are extended. `docs/README.md` adds a note to the
  record's and the gait plan's rows, and `docs/KNOWN_ISSUES.md` gains Update 12
  of the T-Rex stance entry.
- **Decision D-D39 recorded: another T. rex stance screen, three runs of L+T
  with a foot-stagger penalty against two D-D37 runs as the control** (#632;
  the maintainer's decisions of 2026-10-19; `docs/BEHAVIOR_RECIPES_PLAN.md`
  §6.2, `docs/CONSOLIDATION_PLAN_2026_09.md` §6, a dated note in §17 of
  `docs/GAIT_QUALITY_PLAN_2026_09.md`, `docs/NEXT_STEPS.md` §3; the appended
  §18 of `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` and
  `docs/investigations/stance_2026_10/stance4/criteria.txt`). Records only: no
  library code, configuration, test, notebook, digest or generated file changes
  with it. After the three D-D37 runs failed their fixed criteria (line 3b) and
  the overnight scratch experiment left a foot-stagger penalty unattributed at
  one run per arm, D-D39 trains three new 12M runs of arm LTS: D-D37's L+T (arm
  L with `leg_home_pose_weight` 1.0 and `leg_home_pose_tolerance` 0.05 rad)
  plus a stance stagger penalty (`stance_stagger_weight` 0.30,
  `stance_stagger_tolerance_m` 0.045; after the settle, each step pays −0.30 ×
  (1 − exp(−(s / 0.045)²)) for the right foot's fore-aft lead s over the left).
  They train from scratch on physics r9, seeds 52, 60 and 42
  (`stance4_LTS_s52`, `stance4_LTS_s60`, `stance4_LTS_s42`), from the screen
  branch `claude/screen-trex-r9-armlt-stagger` (`1af2d4d7` plus `ff7114c0`, the
  opt-in term off by default, and `c1e146a8`, the two keys on), which is never
  merged and has no pull request. The maintainer's decision named Colab CPU
  runtimes, as D-D37's Colab steps had. Measured, D-D37's runs trained on Colab
  L4 GPU runtimes: the `gpu` block of each run's `stage_config.json`, as
  fetched at 4,194,304 steps, reads L4. It records only the session that wrote
  it, so that the control stayed on L4 throughout is inferred. Before any of
  the three runs started, the criteria were re-fixed for L4, the control's
  runtime type, for the maintainer to confirm when starting the runs; a later
  PR records the confirmation. The control is not retrained: D-D37's Runs 2 and
  3, as already judged. The criteria were fixed before training (sha256
  `978a93da…`; a first file, `8bda933e…`, named a Colab CPU runtime): K1–K4
  exactly as D-D37's, with K2c's 30 mm bar unchanged. The term is adopted, that
  is proposed for committing, only if at least 2 of the 3 runs hold and the arm
  passes each of K1–K4 at least as often as the control (here: no stagger run
  fails K3 or K4); a five-line decision table names the next step for every
  other outcome; and the better of the best stagger final and D-D37 Run 2's
  final is offered as walking pilot 2's parent, while pilot 2 stays parked. The
  ankle anchor and arm L are dropped from further screens. Report only: every
  1,048,576-step checkpoint, diagnostic panels (mirrored leg noise, the prey
  straight ahead, both), a saturation alarm on the eval ledger and stagger
  watches. Screens certify nothing. `stance4/criteria.txt` is byte-identical to
  the criteria file; `SOURCES.txt` and `SHA256SUMS` (now 322 files) are
  extended. `docs/README.md` adds a note to the record's and the gait plan's
  rows, and `docs/KNOWN_ISSUES.md` gains Update 11 of the T-Rex stance entry.
  The outcome is not yet known.
- **The three T. rex stance screen runs of D-D37 judged: all three fail their
  fixed criteria; the readouts and an overnight scratch experiment recorded,
  report only** (#632; D-D37's amendment of 2026-10-19;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2, `docs/CONSOLIDATION_PLAN_2026_09.md`
  §6, a dated note in §17 of `docs/GAIT_QUALITY_PLAN_2026_09.md`,
  `docs/NEXT_STEPS.md` §3; the appended §17 of
  `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` and the folder
  `docs/investigations/stance_2026_10/stance3/results/`). Records only: no
  library code, configuration, test, notebook, digest or generated file changes
  with it. The three 12M runs trained to their final 12,001,280 steps and were
  judged on 2026-10-18 and 2026-10-19 by the criteria fixed before training
  (sha256 `1ce910dc…`) with the frozen judge and scorer, final. Run 1
  (`stance3_armL_s60`, arm L) fails K2 on the no-drift range (33.9 mm against
  30) and K3 (7 and 11 of 20 joint-space noise falls at 6,291,456 and the
  final, against 2). Run 2 (`stance3_armLT_s60`, arm L with
  `leg_home_pose_weight` 1.0 and `leg_home_pose_tolerance` 0.05 rad) fails K1
  and K2 early: clean 0, 0 and 36 of 40 at 4,194,304, 5,242,880 and 6,291,456,
  with one tail actuator pinned at full command, and K2a 10 and 6, K2b 14, 10
  and 6. Run 3 (`stance3_armLT_s52`) fails K2: K2a 6 at 8,388,608, K2b 4 at the
  final and a no-drift range of 30.6 mm. The decision table's line is 3b: a
  joint-space price does not hold the stance, and the line names a fore-aft
  term that needs code or the v6b ankle spring as the next round's candidates.
  The best final checkpoint by the fixed rule is Run 2's (40 of 40 clean, no
  noise falls), which the criteria offer as walking pilot 2's parent. Screens
  certify nothing, and the stage's six-heading certificate rolled offline on
  the three finals (all three PASS) is report only. Report only too: the
  readouts (arm L's ankle-carried whole-body drift, L+T's left-right foot split
  left from the spawn stagger, Run 2's tail-actuator saturation) and an
  overnight scratch experiment of three continuations from a scratch clone that
  is never merged (the ankle anchor refuted at its pre-registered dose by its
  own line, a foot-stagger penalty inconclusive against the matched no-change
  control, the same penalty on Run 1 refuted). The spec's harness-matched local
  no-change continuation of Run 3, compared afterwards, did not recover (clean
  25 of 40, K2a 15 and a stagger sd of 25.9 mm at 9,961,472, against the
  stagger arm's 40, 0 and 7.5 mm), but the Colab no-change run from the same
  checkpoint did (40, 0 and 9.7 mm), so with one run per arm the penalty's
  effect stays unattributed, report only (`stance3/night/`). The next round,
  walking pilot 2's parent and any criteria or telemetry change were not
  decided when the runs were judged; the record lists them as proposals for the
  maintainer, and D-D39 (above) decides the next round. `stance3/results/` (28
  files and its README, about 0.38 MB) holds the run log, the scorer's
  printouts, a per-checkpoint summary (`checkpoints.csv`, 37 rows, with the
  script that recomputes it from the judge files), the certificates, the
  readouts and the night experiment's pre-registration, readouts,
  interpretation and verification, and `stance3/night/` the no-change
  comparison and the interpretation it cites; `SOURCES.txt` and `SHA256SUMS`
  (now 321 files) are extended. `docs/README.md` adds a note to the record's
  and the gait plan's rows, and `docs/KNOWN_ISSUES.md` gains Update 10 of the
  T-Rex stance entry.
- **Decision D-D37 recorded: three full T. rex stance screen runs in place of
  walking pilot 2; the offline diagnosis behind them** (#631; the maintainer's
  decision of 2026-10-18; `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2,
  `docs/CONSOLIDATION_PLAN_2026_09.md` §6, a dated note in §17 of
  `docs/GAIT_QUALITY_PLAN_2026_09.md`, `docs/NEXT_STEPS.md` §3; the appended
  §16 of `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` and
  the folder `docs/investigations/stance_2026_10/stance3/`). Records only: no
  library code, configuration, test, notebook, digest or generated file
  changes with it. After walking pilot 1 read STILL, the maintainer asked for
  three stance runs with "different reward settings, slightly different
  mechanics, or a refinement", and the offline diagnosis of 2026-10-18 (rolls of
  existing checkpoints, the statue and held poses; no training) found that
  the committed stance reward pays for a forward lean of about 23 mm through
  its keyframe-centred leg pose term, that arm L's settled leg target removes
  the pay but prices a 14 mm lean at only about 0.3% of the reward, and that
  the lean lives in the ankles (about 10.7 mm per degree). D-D37 trains three
  12M stance screen runs on `main`'s physics r9, started on 2026-10-18 on
  Colab L4 runtimes (the maintainer kept L4 for all three): arm L as #624
  ships it on seed 60 (`stance3_armL_s60`, branch
  `claude/screen-trex-r9-arml-12m` at `7e6b557d`), and arm L with
  `leg_home_pose_weight` 1.0 and `leg_home_pose_tolerance` 0.05 rad on seeds
  60 and 52 (`stance3_armLT_s60`, `stance3_armLT_s52`, branch
  `claude/screen-trex-r9-armlt` at `1af2d4d7`); neither branch is ever
  merged or has a pull request. They are judged by criteria fixed before
  training (sha256 `1ce910dc…`: K1 the gate held, K2 fore-aft held with the
  screen-only K2b, a signed per-foot whole-foot centre of pressure, and K2c,
  no drift, K3 robustness, K4 roll) and a decision table; screens certify
  nothing, and whether a screen run could ever count toward
  `certification_seeds` is left to the maintainer. It supersedes, for these
  runs, the 2026-10-16 rule against further stance-only reward experiments
  unless walking points back to stance, and parks walking pilot 2 (its branch
  and criteria kept). Run 1's `stage_config.json` on Drive confirms its
  commit, arm L's keys, its task and PPO hyperparameter digests and an L4
  GPU, and its run folder's `zero_action_baseline.json` a statue standing at
  3980.37 (39 of 40); Runs 2 and 3's `stage_config.json` files confirm
  commit `1af2d4d7`, seeds 60 and 52, the two leg keys, collapse reference
  4460.5 and an L4. `stance3/` holds the criteria (byte-identical), the
  Colab steps, the recommendation and frozen copies of the judge, the scorer
  and its tests (80 cases); `SOURCES.txt` and `SHA256SUMS` (now 291 files)
  are extended. `docs/README.md` adds a note to the record's and the gait
  plan's rows, and `docs/KNOWN_ISSUES.md` gains Update 9 of the T-Rex stance
  entry. The outcome is not yet known.
- **Balance, force and policy telemetry at every evaluation and PPO update,
  and the T. rex stance's onset checkpoints kept, report-only** (#630; the
  maintainer's approval of 2026-10-17 of the T. rex stance telemetry
  recommended after the physics-r10 screen). The T. rex stance runs that
  nosedived reached the toe line with every foot rail still inside its
  bar; 15 to 17% of one r9 arm L seed's training episodes fell at 4.7 to
  5.2M steps while its evaluations stood; the mean exploration noise hid a
  9- to 13-fold spread between actuators; and only the newest five
  checkpoints survived a run, so when a lean set in could not be replayed.
  Nothing recorded any of it. No gate verdict, reward, observation,
  action, termination, task, checkpoint selection, plant or
  hyperparameter digest moves, no env info key or MJCF sensor is added,
  and nothing draws a random number on a training or evaluation path: the
  exact reward captures (`digest_snapshot --exact`, 37,034 lines) are
  byte-identical; the digest snapshot moves only the T. rex stance's two
  `stage_config_view_sha256` lines, for the retention key below; and a
  short T. rex stance run (57,344 steps on 4 envs) on main and on this
  change adds the same data to every rollout, draws the same random
  numbers, evaluates the same and saves the same parameters and
  VecNormalize statistics.
  - Whole-body balance and floor forces (`gait/balance_telemetry.py`), from
    the floor-truth recorder's opt-in balance capture
    (`SubstepContactRecorder(..., balance=True)`, evaluations only, its
    other fields unchanged): per evaluation, as full-horizon means in
    `gate_progress.npz` and TensorBoard (`diagnostics/eval_<key>`), the
    centre of mass ahead of the feet and the root pitch, the extrapolated
    centre of mass (Hof's XcoM), Hof's split of the centre of mass's
    forward acceleration into its centre-of-pressure and counter-rotation
    terms, the achieved leg joint offsets (left minus right and
    mirror-even), the actuator force-limit fraction (also per actuator, as
    a matrix) and the fraction of steps at a limit, the floor's shear over
    its normal force and the friction-cone use per foot; on the T. rex's
    declared span, ankles and hinged digits, the XcoM's margins to the toe,
    heel and digit-tip lines, the fraction of steps past the toe line, the
    time to the toe line, the XcoM's lead before a fall and the ankle
    moment arm. #620's per-foot rule: the four box-soled models record
    them, and no model attaches a recorder it did not before. The capture
    and its reduction each have their own guard, so a failure costs only
    these keys. On the T. rex an evaluation takes about 3% longer.
  - Every model gains the mirror-even commanded leg DC
    (`leg_dc_<l>_<r>_mean_<joint>_deg`) and each reward term split into
    the evaluation's full-horizon and fallen episodes (`full_horizon_term_*`,
    `fall_term_*`).
  - The in-training `stance_quality/v2` screen: on the T. rex and the
    anatomical Compsognathus stances, every evaluation is judged on the
    stage's own v2 bars at yaw 0 without the statue-relative criteria
    (`screen_v2_*` in `gate_progress.npz`, TensorBoard's `screen_v2/`). It
    is never a verdict, and its names keep it apart from the gate's
    recorded-history reading. A criterion it skips keeps its fail-count
    key, which reads NaN, not 0.
  - `eval_episodes.csv`: one row per evaluation episode (return, length,
    cause, the screen's per-episode result, the spawn, the per-foot pad,
    whole-foot and roll-rail values, the gate's per-episode maxima, and the
    balance and force values), rewritten atomically after every
    evaluation.
  - `train_episodes.csv` (`rollout_ledger.py`): one row per training
    episode (its cause and length, the post-settle and last pre-terminal
    centre of mass and root pitch, the limited-force fraction and, on the
    T. rex, each touch sensor's share of its foot's load), copying raw
    `mjData` reads per step and reducing once per episode, rewritten on
    the checkpoint cadence. `diagnostics.npz` gains `rollout_episodes`,
    `rollout_falls` and `rollout_term_count_<reason>` (named so that no
    reader takes them for termination fractions) and
    `reward_action_jerk`, which the env emitted and nothing recorded; the
    jerk term also reaches W&B and the post-training results CSV
    (`eval_reward_action_jerk`).
  - `policy_probe.npz` (`policy_probe.py`): at every PPO update, the
    per-actuator exploration std, VecNormalize's per-dimension
    statistics, the entropy coefficient, the learning rate the coming
    update applies, and the critic's value and the deterministic action on
    a fixed bank of ten states (the statue and held ankle leans on two
    seeds), under the current and the session-start normalization. The bank
    is built once per session on a separate env, with any scheduled pushes
    off (a recovery stage's would topple its states), under a guard that
    restores the global generators (it draws none), and each update is a
    gradient-free forward pass. TensorBoard gets `policy_probe/*`.
  - A `PostureTrend` advisory (`curriculum/posture_watch.py`): one warning
    line when 2 evaluations in a row have a fore-aft over-bar fraction of
    0.1 or more and the signed centre of pressure moves toward that edge,
    at most once per 10 evaluations, with four `[curriculum]` diagnostic
    knobs that no stage sets. Replayed on the r9 and r10 arm L screens'
    series, it first warns on r9 seed 60 one evaluation into the drift
    that lasted, four evaluations before `PostureDrift`; at those screens'
    10-episode evaluations it warned on all six series, including the r10
    seed that passed `stance_quality/v2`.
  - `figures/gate_progress.png` at stage end, in its own guard, and
    `scripts/plot_gate_progress.py` for a copy of a stage directory mid-run
    (it refuses to write into a run directory, which the result bundle's
    manifest would then reject).
  - `handoff_in_training.json`: the stance handoff checkpoint's
    in-training step, evaluation and selection score, written beside the
    certificate after it returns; the certificate, its verdict and
    `gate_verdict.json` are untouched.
  - `progress.json` gains `balance`, `forces`, `policy`,
    `training_rollouts`, `evaluation.screen_v2`, `posture.leg_q_deg` and
    `checkpoints.periodic_timesteps`, under the same schema string: it now
    changes only when a key is removed or renamed or changes meaning,
    units or type.
  - The archive rule (`curriculum/checkpoints.py`): with the new retention
    key `checkpoint_archive_every_timesteps = A`, checkpoint retention also
    keeps the newest periodic checkpoint of every interval of `A` steps.
    The T. rex stance sets 500,000, its own commit and the only
    stage-configuration change (the one other configuration edit drops
    `compare_run_diagnostics.py` from the coverage omit list, now that a
    test imports it): at the notebook's 100,000-step cadence a 12M run
    keeps 28 checkpoint pairs (113 MB) instead of 5 (20 MB), against 120
    (485 MB) for `max_checkpoints = 0`. The key enters no gate, task or
    hyperparameter digest; `configs/digest_snapshot.generated.txt` was
    regenerated with the tool, and exactly its two T. rex stance
    `stage_config_view_sha256` lines moved.
  - Tests: new `test_gait_balance_telemetry.py` (bit-identity with and
    without the capture on every box-soled model, the force decode against
    body weight, the statue and a scripted nosedive, the frames, and the
    failure isolation), `test_episode_ledger.py`,
    `test_curriculum_rollout_ledger.py`, `test_curriculum_policy_probe.py`
    (no random number drawn, nothing written to the policy, the bank whole
    at a short horizon), `test_curriculum_checkpoint_archive.py` and
    `test_curriculum_balance_telemetry.py`, a 768-step T. rex stance run
    with every addition on and off: the same hash of every rollout-buffer
    addition, the same policy, optimizer and generator state at every
    update, the same evaluations, checkpoint tensors and VecNormalize
    statistics, the base's key sets when off and exactly this change's
    keys added when on. `test_eval_diagnostics.py`, `test_diagnostics.py`,
    `test_curriculum_posture_watch.py` (the replays),
    `test_curriculum_progress_report.py`, `test_visualization.py`,
    `test_stance_gate_report.py` and others are extended. The
    `test_curriculum_*.py` files run in the SB3 job.
  - Documented in `docs/TRAINING_ARTIFACT_PRESENTATION.md` (every new key
    and file, their NaN rules, the screen, the advisory and its replays,
    the figure, the sidecar, the archive rule and its cost, the additive
    schema rule and which model records what) and the website's training
    tips.
- **Decision D-D36 amended: walking pilot 1 read STILL, and a second pilot with
  wider exploration; and the evidence of the T. rex stance follow-ups of
  2026-10-17** (#629; the maintainer's decision of 2026-10-17;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2, `docs/CONSOLIDATION_PLAN_2026_09.md`
  §6, a dated note in §17 of `docs/GAIT_QUALITY_PLAN_2026_09.md`,
  `docs/NEXT_STEPS.md` §3; the appended §15 of
  `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md`, seven folders
  in `docs/investigations/stance_2026_10/` and 28 rows of
  `results/offline_judgements.csv`). Records only: no library code,
  configuration, test, notebook, digest or generated file changes.
  D-D36's walking pilot, now pilot 1, was judged on 2026-10-17 by its fixed
  criteria and read STILL on all three seeds, final: each seed STILL on 40 of
  its 40 panel episodes at 4,194,304 steps, with no WALK episode, which by the
  criteria is too short to tell; the outcome is the criteria's, not a gate's.
  Report only: its forward-velocity term earned 0.5 to 3.2 per episode at the
  2M, 3M and 4.2M evaluations (0.1 to 7.5 over every evaluation from 2M to
  4.2M steps; the September T. rex walkers: 148 to 234 at 4.2M), and its
  action std was 0.108 to 0.111 at 4,194,304 steps against the parent's 0.095.
  The same day the maintainer chose a second pilot, the same with one change:
  the action std reset to 0.2231 (`log_std` −1.5, the T. rex stance stage's own
  starting noise) on every actuator at the warm start, from the branch
  `claude/pilot-walk-r9-b` (`ba464855`), which is never merged; it is
  pre-registered, not yet trained and has no result, it is still an exception
  to the gait plan's GQ-3 (a) and GQ-17, and it certifies nothing.
  The stance record's appended §15 records pilot 1's outcome, the second
  pilot's criteria and the offline work of 2026-10-17, all report only: the
  stage's six-heading certificate rolled on the r9 arm L symmetry-screen
  checkpoints (seed 60 40/40 and seed 52 38/40 PASS, seed 42 29/40 FAIL at 6M,
  on checkpoints from a scratch harness that the judge would refuse on task);
  an offline battery (the passes replicate on fresh seeds but thinly, and the
  reward-selected `robust_best` checkpoints are clean on 0 of 40 under their
  own action noise); the fore-aft bar's two blind spots (it reads the pad's own
  CoP, and it is unsigned); a foot and ankle audit (nothing in the anatomy
  pushes the stance forward; seven candidate physics changes, none decided);
  and a literature report. The evidence folder gains 215 files (about 3.0 MB),
  with `SOURCES.txt` and `SHA256SUMS` (now 285 files) extended; per-episode
  files, traces and checkpoints stay in the session's scratch or on Drive. The
  ledger's 28 rows (19 certificates, 17 PASS and 2 FAIL, and pilot 1's nine
  panels, STILL) are marked offline and report only, and each points at its
  evidence file. `docs/README.md` adds a note to the record's and the gait
  plan's rows, and `docs/KNOWN_ISSUES.md` gains Update 8 of the T-Rex stance
  entry.
- **The cross-run results table saved with the offline judgements:
  `logs/runs_summary.csv` and `results/offline_judgements.csv`** (#628; the
  maintainer's approval of 2026-10-17: "Persistent results ledger: the Drive
  summary notebook saves its cross-run table as logs/runs_summary.csv, with
  offline judgements addable as rows marked 'offline'"). Every run writes its
  own records to its Drive folder, and the Drive summary notebook read them
  into one table and wrote it to `logs/runs_summary.csv` in place. A judgement
  made outside training (a gate certificate scored on a saved checkpoint after
  its run, or a pilot's fixed criteria applied by a script) had no place in it,
  since the session agent that makes these never writes to Drive. No training,
  reward, task, gate, plant or digest change: the 649-line digest snapshot is
  byte-identical.
  - `results/offline_judgements.csv`, a ledger tracked in the repository with
    one row per judged checkpoint, added by pull request: the date, the model,
    the algorithm, the run (`run_id` or `run_path`), the stage, the
    checkpoint's step and sha256, the gate kind (or the criteria's name), the
    verdict, the key numbers (the passing count and panel size, the lower bound
    and its bar, the binding bar), the panel's seeds, the judging tree's
    commit, the tool, the evidence, a note, and `source` `offline`. It is
    committed with its header alone; the judgements of 2026-10-17 are added by
    a separate records change. `results/README.md` documents its 22 columns.
  - `environments/shared/reporting/runs_summary.py`, re-exported by
    `environments.shared.reporting`: `load_offline_judgements` reads the
    ledger strictly (the header exactly the 22 columns in order; the date, the
    model against the species manifest, the stage as `collected_results.csv`
    writes it, the hash, the commit, the counts and the bounds checked; every
    problem listed with its line; an empty file or a header alone has no rows),
    `merge_runs_summary` appends its rows to the notebook's (a PASS verdict
    reads `stage_passed` true, FAIL false and any other verdict empty), and
    `write_runs_summary` writes the union of the rows' columns atomically (a
    temporary file and `os.replace`, as `progress.json` is written).
  - `notebooks/google_drive_summary.ipynb`, section 9 ("Save the Cross-Run
    Table", cell 24), loads the ledger from its checkout, shows its rows, and
    saves `logs/runs_summary.csv`, each model's `runs_summary.csv` (its offline
    rows included) and each model's `sweeps/sweeps_summary.csv` through these,
    printing where it wrote and how many rows of each source. Two things
    change in those files: each is replaced atomically instead of rewritten in
    place, and a missing cell is written empty instead of `nan` (pandas reads
    both as missing). The notebook still runs without Drive (it writes nothing
    when the logs directory does not exist, and never creates it), without
    runs (the offline rows are saved alone) and without the ledger (no offline
    rows). Its run reader stays in the notebook.
  - `.github/workflows/python-ci.yml`'s path filter gains
    `results/offline_judgements.csv`, so a records PR that only adds rows runs
    their check.
  - Tests: `test_reporting_runs_summary.py` (the committed ledger, its
    documentation and its CI trigger; the loader's types and each refusal;
    the merge; the writer's column union, empty cells and atomic replacement)
    and, in `test_drive_summary_notebook.py`, the section 9 cell run with runs
    and a ledger, with each alone, with neither, and with no logs directory.
  - Documented in `docs/RESULT_BUNDLES.md` (the cross-run table and the
    ledger), `results/README.md` (the ledger's columns) and the notebook's own
    markdown.
- **Decision D-D36 recorded: one short T. rex walking pilot, a deliberate
  exception to the gait plan's GQ-3 (a) and GQ-17; and a record of the
  T. rex stance experiments of 2026-10-12 to 2026-10-17** (#627; the
  maintainer's decision of 2026-10-17; `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2,
  `docs/CONSOLIDATION_PLAN_2026_09.md` §6, the dated §17 of
  `docs/GAIT_QUALITY_PLAN_2026_09.md`, `docs/NEXT_STEPS.md` §3;
  `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` and its
  evidence directory `docs/investigations/stance_2026_10/`). Records only:
  no library code, configuration, test, notebook, digest or generated file
  changes.
  After the T. rex physics-r10 stance screen of arm L (the T. rex stance with
  #624's three switches on: mirror-symmetric PPO, the lateral lean term and
  the settled leg target, all off by default) failed all four of its fixed
  criteria (K1 seed 42 alone passing `stance_quality/v2` at 6M; K2 seeds 52
  and 60 on their toes, 37 to 40 of 40 panel episodes over the whole-foot
  fore-aft centre-of-pressure rail at 4M, 5M and 6M; K3 16 of 60 joint-space
  noise falls; K4 12 of 120 roll-rail failures), while arm L on physics r9 had
  failed only K2 (on the pad's centre-of-pressure rail) and K3, the maintainer
  parked draft #623 (`main` keeps physics r9) and decided to run one short
  T. rex walking pilot now. It is an exception to GQ-3's recommended option
  (a) ("no walker session on today's rewards") and to GQ-17's retrain order
  (robot first), both still open, and it starts from an uncertified stance
  checkpoint: r9 arm L seed 60 at 6,291,456 steps (sha256 `70f3a444…353e`), a
  recorded departure from the r10 screen's parent rule, which named r10 arm L
  seed 42 at 6M on the unmerged r10 foot. The production locomotion stage,
  stock PPO, trains from it under `initialize_next_stage` through the
  notebook's manual single-node cell on seeds 42, 52 and 60, 4,194,304 steps
  each (production 8,000,000; the learning rate follows production's first
  4,194,304 steps, and an evaluation runs every 131,072 steps), and the frozen
  2026-09 gait-audit probe judges it on hop-detecting criteria fixed before
  any pilot training step, with the maintainer's video review required before
  a seed counts as walking. It runs from the branch `claude/pilot-walk-r9`,
  which is never merged; nothing it trains is certified or published, and no
  digest on `main` moves (the branch's shortened locomotion stage moves only
  that stage's PPO hyperparameter and stage-config-view digests; its task and
  gate digests are production's). Its outcome is not yet known; a later PR
  records it.
  The record (`docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md`)
  is the dated technical account of the T. rex stance work since the blog post
  of 2026-10-11, and the D-D36 rows cite it for the screens' evidence: run
  `20261011_165957` (seed 56, 12M steps) refused by `stance_quality/v2` with
  its right pad on the toe edge (1/40 clean), from a small
  left/right-antisymmetric leg command that the closed chain of two planted
  legs turns into a one-pad edge under a reward flat in the centre of
  pressure; the lower-noise, CoP-term and capped-leg-authority screens, each
  failing its predeclared criteria; the symmetry screen (arm L and the
  fore-aft arm C, verdict O9: symmetric policies, a forward creep and
  joint-noise falls); the r10 screen above; the toe-lean diagnosis (no reward
  term pays for the lean, the r10 digits let a held lean that most spawns
  survive reach a whole-foot CoP of 0.89 to 0.94, against 0.76 to 0.79 on r9,
  past the 0.80 rail, and what starts it is unexplained); the pilot's
  pre-registration and launch, its outcome to be appended as a dated note; and
  corrections, withdrawn claims and method lessons. The evidence directory
  holds 72 files (about 1.2 MB): the criteria texts, the frozen judges,
  scorers and verdict scripts with their tests (research copies, imported by
  nothing and run by no CI job), the score and verdict files, the diagnosis,
  the pilot's criteria and scoring tools, `SOURCES.txt`, which maps each copy
  to its source, a README that says how to re-check each verdict, and
  `SHA256SUMS`, which pins every file but the README and itself; checkpoints,
  rollouts and traces stay on Drive or in the session's scratch.
  `docs/README.md` gains its row, and `docs/KNOWN_ISSUES.md` Update 7 of the
  T-Rex stance entry.
- **A training progress report and posture telemetry at every evaluation,
  report-only** (#626; the maintainer's request of 2026-10-16 for better training
  artifacts and reporting). The T. rex physics-r10 screen's two drifting
  seeds were found by reading `gate_progress.npz` by hand: at 2.3 to 2.5M
  steps seeds 52 and 60 stood on their toes (whole-foot centre of pressure
  1.1 to 1.25 span half-lengths, digit share 0.7 to 0.9) while seed 42 stood
  clean, and nothing in the training output or in a file a person could open
  from Drive said so. This change adds a readable snapshot of each stage, the
  gate's two roll rails and the evaluation episodes' termination causes to
  the per-evaluation telemetry, and one advisory warning line. No gate
  verdict, reward, observation, action, checkpoint selection, existing
  schema, task, plant or hyperparameter digest moves: the 649-line digest
  snapshot and the exact reward captures (`digest_snapshot --exact`) are
  byte-identical, and a short T. rex stance run with and without every
  addition evaluates the same and saves the same parameters and
  VecNormalize statistics.
  - `progress.json` in every stage directory
    (`curriculum/progress_report.py`, schema
    `mesozoic.training-progress/v1`, strict JSON of 4 to 6 KB plus 0.6 to
    0.9 KB per kept warning, under about 15 KB): the model, stage, run id,
    run directory and seed; steps against the session's
    budget, wall-clock time, observed steps per second and the projected
    finish (UTC); the latest evaluation (mean and standard deviation of
    reward, mean episode length, full-horizon fraction, falls and each
    termination cause's count); the zero-action statue's reward when the run
    captured one; per rail the model measures (the pad and whole-foot
    centre of pressure, the sole tilt and corner lift) the stage's bar, the
    mean of each episode's own rail statistic, the fraction over the bar and
    the per-foot values; the left-minus-right leg DC; the last five
    evaluations; the newest periodic checkpoint and the best and robust best
    models' steps; and the advisories raised. It is rewritten atomically (a
    temporary file and `os.replace`, safe on a Drive mount) when training
    starts, after every evaluation and when it ends, from what the evaluation
    callbacks already recorded (`StageGatePlateauCallback.gate_progress_row`);
    a failure to write it is logged once and costs the file, never the run.
  - The roll rails: `pad_telemetry.episode_rail_telemetry` keeps each box
    sole's post-settle mean tilt and corner lift
    (`<label>_sole_tilt_deg`, `<label>_sole_corner_lift_m`), whose maxima
    over the feet are `stance_quality/v2`'s `max_sole_tilt_deg` and
    `max_sole_corner_lift_m` bit for bit on every model.
    `gate_progress.npz` and TensorBoard (`diagnostics/eval_<key>`) gain their
    full-horizon means, `max_sole_tilt_deg_mean` and
    `max_sole_corner_lift_m_mean` (each episode's own maximum), and, where
    the stage declares the bar (the T. rex and the anatomical Compsognathus),
    `max_sole_tilt_deg_over_bar_fraction` and
    `max_sole_corner_lift_m_over_bar_fraction`. They follow #620's per-foot
    rule, the box-soled feet, so no model attaches a recorder it did not
    before; the Brachiosaurus's ellipsoid soles are left out (their tilt is
    mostly the pitch they are authored at, and recording it would add 45% to
    every Brachiosaurus evaluation for a bar none of its stages declares). On
    the T. rex the reduction costs no measurable evaluation time. An episode
    whose roll rails cannot be reduced reads NaN on them alone, logged once;
    its pad and whole-foot fields stand.
  - Termination causes: `StageAwareEvalCallback` keeps each evaluation
    episode's cause (the env's `termination_reason`, `terminated` when it
    gave none, or `truncated` at the time limit), and `gate_progress.npz`
    gains one `termination_<cause>` count per cause seen, on every evaluation
    (0 where none ended that way; NaN on rows a resume carried over from a
    file recorded before the counts), with TensorBoard's
    `terminations/eval_<cause>`.
  - The posture-drift advisory (`curriculum/posture_watch.py`): on a stage
    that declares a fore-aft centre-of-pressure bar (`max_sole_cop_fore_aft`
    or `max_foot_cop_fore_aft`), one `PostureDrift` warning line in the
    training output, which the notebook shows, when the rail's over-bar
    fraction is 0.5 or more on 2 evaluations in a row. It names the rail,
    the fractions, the steps and each foot's signed centre of pressure,
    repeats at most once per 10 evaluations while the drift lasts, and is
    listed in `progress.json`; it never stops or changes training. Two in a
    row, not three, warn earlier: replayed on the r10 screen's own
    whole-foot series to 6.16M steps, two first warn on seeds 52 and 60 at
    917,504 steps and never on seed 42 (through 6.03M), while three first
    warn on seed 52 at 2,359,296 (1.44M steps later) and on seed 60 one
    evaluation later; over 69 pad-rail series of 3M steps or more from the
    earlier T. rex stance screens, three warn on every series two warn on,
    1 to 10 evaluations later, and on no other. An evaluation with no
    full-horizon episode has no over-bar fraction and ends the streak, so a
    seed whose episodes all fall goes quiet here; `progress.json`'s falls
    still show it. It is modelled on the below-baseline warning, with its
    knobs (`posture_drift_warn_over_bar_fraction`,
    `posture_drift_warn_consecutive_evals`,
    `posture_drift_warn_repeat_evals`) registered as diagnostic keys like
    `baseline_warn_after_budget_fraction`: no committed stage sets them and
    they enter no gate, task or hyperparameter digest. A same-stage resume
    starts it past the seeded rows, as the hop watch.
  - The evaluation-plateau, below-baseline and hop-watch warnings are
    recorded for `progress.json` too, with their log lines unchanged.
  - Tests: `test_curriculum_progress_report.py` (the record, the atomic
    rewrite and its failure, and a 768-step T. rex stance run on CPU whose
    `progress.json` matches its `evaluations.npz` and `gate_progress.npz`,
    run again with every addition switched off: the same evaluations,
    `gate_progress.npz` values and checkpoint parameters),
    `test_curriculum_posture_watch.py` (trigger, no trigger, rate limit,
    knobs, wiring, resume, and the r10 screen's three seeds replayed), and
    in `test_gait_pad_telemetry.py` and `test_eval_diagnostics.py` the roll
    rails against the gate on synthetic traces and every model, a failed
    roll-rail reduction, and the termination counts and their NaN rules.
    Both new files run in the SB3 job (`test_curriculum_*.py`).
  - Documented in `docs/TRAINING_ARTIFACT_PRESENTATION.md` (the
    `progress.json` schema, the new `gate_progress.npz` keys and their NaN
    rules, the advisory) and the website's training tips (open
    `progress.json` from Drive while a Colab run trains).
- **Mirror-symmetric PPO and two T. rex stance options, all off by default**
  (#624; the maintainer's approval of 2026-10-16: land the symmetry module and the
  two reward options as config switches, off by default so nothing changes
  until a config turns them on). The T. rex stance policies of 2026-09 and
  2026-10 settled into lopsided postures, a lean onto one pad's edge or a
  left-right ankle offset, that stock PPO has no reason to avoid. A
  pre-registered screen trained this method with the two options on the T.
  rex stance (three seeds to 6M steps): the policies were mirror-symmetric
  (every joint group's median mirror asymmetry under 0.1 degree), fell in
  none of the 120 final (6M) panel episodes and passed `stance_quality/v2` on
  two seeds of three at the panel's single heading (the stage's six-heading
  certificate, `stance_quality_heading/v1`, was not run), but failed the
  screen's pad centre-of-pressure and action-noise robustness criteria, so no
  stage config enables any of it.
  Every plant, policy-interface, task, gate, hyperparameter and
  `stage_config.json` digest, the committed digest snapshot and the exact
  reward captures (`digest_snapshot --exact`) are unchanged.
  - `environments/shared/symmetry.py` derives a model's mirror maps from its
    compiled MJCF: bodies, geoms, sites, joints, actuators and sensors are
    paired by their side token (`r_`/`l_` and the robot's and quadrupeds'
    forms) and checked against the geometry, joint axes and ranges, inertias,
    gravity compensation, contact and fluid parameters, colliding mesh
    vertices and actuator gains (with a per-body frame correction for CAD
    frames that differ between the sides, which no committed plant needs; a
    synthetic biped in the tests exercises it and the suffix side tokens), and
    a model that fails a check is refused. The checks cover the model fields
    the plants use, so a new species' map comes with a paired-rollout
    equivariance test like the T. rex's. The five anatomical plants derive;
    the Compsognathus robot is refused (its MJCF is not mirror-symmetric as
    authored). `build_env_mirror` adds the observation map from the plant's
    observation layout (`bipedal-target-heading-free/v1`, the T. rex's and the
    anatomical Compsognathus's) and the action map, after checking that the
    env's action scaling commutes with it; `mirror_env_state` writes the
    mirror image of an env's simulator and episode state. Every map is a
    signed permutation and an involution, validated on construction.
  - `configs/tyrannosaurus_rex/anatomical/mirror_map.json` is generated
    (`python -m environments.shared.symmetry --write --species
    tyrannosaurus_rex --embodiment anatomical`) and checked (`--check`, run by
    `test_symmetry.py`). It records the plant's `physics_sha256` and
    `policy_interface_sha256`, and `MirrorMap.load` refuses another plant and
    a file that differs from a fresh derivation. A T. rex plant revision
    regenerates the map.
  - `environments/shared/mirror_kl_ppo.py`: `MirrorKLPPO` is SB3 2.9.0's PPO
    with a KL-form mirror loss (`kappa` times the KL divergence from the
    policy at the mirrored observation to the mirror image of the policy,
    the noise scale held fixed and floored), a mirrored critic, observation
    statistics kept bitwise mirror-symmetric (`SymmetricRunningMeanStd`,
    which pickles as the stock class) and a `log_std` tied across mirror
    pairs. Its `train` is the stock `PPO.train` with four added lines, each
    marked `# [mirror]`, and refuses another SB3 by the stock source's
    digest. Mirror augmentation was rejected: at the stance's small
    exploration noise its clipped corrective signal vanishes for an
    asymmetry of a few hundredths of an action unit, while the KL form's
    does not. Checkpoints stay a stock policy zip and a stock VecNormalize
    pickle, plus the JSON attribute `mesozoic_symmetry` with the settings;
    the logger gains `train/mirror_kl_loss`, `train/mirror_asymmetry` and
    `train/mirror_value_loss`.
  - `[ppo]` keys: `symmetry = "mirror_kl"` selects it on fresh and resumed
    models (`symmetry_kappa` 0.3, `symmetry_sigma_floor` 0.01 and
    `symmetry_mirror_critic = true` by default, the screened values, which a
    stage that enables it writes out so its digest pins them); absent, or
    `"none"`, training is stock PPO, bit for bit (the settings a table
    writes are validated either way). The keys enter the stage's
    `hyperparameters_sha256`, not its task. A same-stage resume
    refuses a checkpoint whose recorded symmetry differs from the stage's,
    either way; a warm start takes the child stage's, and a child that
    enables it needs a parent trained with it (a stock parent's noise scales
    and observation statistics are not mirror-symmetric, and are refused). A
    behavior, which trains stock PPO, drops a parent's `mesozoic_symmetry`.
  - T. rex `[env]` options: `lateral_lean_weight` (0.0, off) and
    `lateral_lean_tolerance_m` (0.015) pay a Gaussian on the whole-body
    centre of mass's lateral offset over the foot-site midpoint, in the
    heading frame, after `stance_width_settle_steps`, under a zero command
    and with both feet loaded (calibrated at 0.25 and 0.015 m, the values
    the screen used; a bonus raises the statue's return by about 200, so
    `collapse_peak_floor_reference` is re-measured, and `reward_lateral_lean`
    joins the diagnostics reward keys, before a stage enables it);
    `leg_home_pose_reference = "settled"` (default `"keyframe"`) centres
    `leg_home_pose` on the statue's settled leg pose, mirror-symmetrised
    (`_LEG_SETTLED_QPOS`, physics r9, at the stance's reset noise; re-measured
    with any plant revision). Off, they add no reward and no info key.
  - Task fingerprints: an env class may declare `OPT_IN_TASK_KWARGS`, and
    `task_fingerprint.inert_opt_in_env_kwargs` carves an option's kwargs out
    of the effective `[env]` section and of the defaults `save_stage_config`
    records while its switch holds its declared off value, as the command
    kwargs are under `command_mode = "none"`; on, every kwarg it lists is
    hashed, defaults included. A declared switch or kwarg that is not a
    constructor parameter is refused (`TaskFingerprintError`), so a typo
    cannot read as always off and keep a live option out of the fingerprint.
    Without the carve-out the three new defaults would have moved every
    T. rex task digest. Both options are reward
    settings a behavior transition may change
    (`behavior_checkpoint._TRANSITION_REWARD_SETTINGS`).
  - Tests: `test_symmetry.py` (derivation on every plant and on a synthetic
    biped with rotated link frames, kinematics of the mirrored configuration,
    the committed map's regeneration and its plant binding, paired stance
    rollouts equivariant to float rounding in observation, reward, info and
    termination, the `[ppo]` keys), `test_mirror_kl_ppo.py` (symmetric
    statistics and their pickling, the tie through the gradient clip and Adam,
    the loss, the confined `train` copy, bitwise equality with stock PPO at
    `kappa` 0 without the mirrored critic, stock checkpoints, the train_base
    wiring; in the SB3 job) and `test_anatomical_lean_and_settled_legs.py`
    (the lean term's formula, gates and mirror invariance, the settled target
    and its re-measurement, and that the defaults change nothing);
    `test_task_fingerprint_effective.py` refuses a misdeclared
    `OPT_IN_TASK_KWARGS`; `test_behavior_checkpoint.py` turns the options on
    and off across a behavior transition and drops a parent's symmetry record.
  - Documented in `website/docs/training/mirror-symmetry.md`.
- **The whole-foot centre of pressure at every evaluation, report-only**
  (#622; the maintainer's approval of 2026-10-16: "record whole-foot centre of
  pressure, report-only, with no physics change"). `stance_quality/v2`'s pad
  rail, `max_sole_cop_fore_aft`, reads one contact geom, the plantar pad. Once
  a foot's digits carry load ahead of its pad, the pad's CoP is no longer the
  quantity that decides toppling about the ankle: the whole foot's is, the
  floor load's moment about the ankle over the limb's floor force. This
  change measures it, per step and per foot, beside the pad's. Nothing reads
  it but telemetry and reports: no reward, observation, action, checkpoint
  selection, policy interface, plant or task digest moves, and no gate
  verdict once a report measured before this change is re-rolled (until
  then it is refused; see Migration).
  - `FOOT_SUPPORT` may declare a foot's `ankle` (a hinge of the sole's own
    body, a name template) and its support span, `heel_line` and `toe_line`,
    in m ahead of the ankle axis along the sole's forward axis; all three or
    none (`gait/morphology.py` refuses a half-declared or inverted span, an
    ankle that is not a hinge of the sole's body, and one whose axis is not
    the sole's lateral axis, the one axis about which a contact's distance
    along the sole is its lever arm). Only the T. rex declares them:
    `{s}_ankle`, 0.070 and 0.210, its pad's own heel and toe edges, so its
    reading is in pad half-lengths. The other box feet (the anatomical
    Compsognathus's, the Dibothrosuchus's and the Compsognathus robot's) sit
    on a toe or ankle-roll hinge below the ankle, so which axis counts as
    their ankle is left to a later decision. A hinged digit (a registered
    support capsule on a child body of the sole's, joined to it by one hinge,
    its MTP joint) is found by rule: the T. rex's three per foot, none
    elsewhere.
  - The floor-truth recorder's `EpisodeTrace` gains `foot_cop`: every floor
    contact of the limb (pad, digits, any other leg geom), force-weighted,
    along the sole's forward axis of its own substep, in m ahead of the ankle
    axis, pooled over the step's substeps as `sole_cop` is; NaN without a
    declared ankle. It also gains `foot_support_span` and
    `support_load_pos`, each hinged digit's force-weighted load position, 0
    at the MTP axis and 1 at the tip. Times the limb's floor force,
    `foot_cop` is the vertical load's moment about the ankle axis while the
    sole is level. It is a lever arm along the sole, so the two differ by at
    most `|d.n| sin(tilt) + |d.f| (1 - cos(tilt))` per contact: within 1.0 mm
    on the statue after its settle, up to 21 mm under N(0, 0.1) command noise
    that tilts the pad 5°. It is not the ankle actuator's torque, which
    also carries the floor's shear times the ankle's height.
  - `StanceEpisodeMetrics.max_foot_cop_fore_aft` reduces it as the pad rail
    reduces the pad: per foot, the window mean of the per-step |CoP| in span
    half-lengths, a step whose limb carries no floor force above 0.1 N on any
    substep counting 1; the maximum over the feet; NaN on a model whose feet
    declare no span. It is one of `STANCE_METRIC_LATER_FIELDS`. The matching
    `stance_quality/v2` key `max_foot_cop_fore_aft` is optional, with its bar
    strictly inside (0, 1) (`CRITERION_BAR_RANGES`). It is plumbed through the
    manager's `StageThreshold` and copied keys, the catalog's exported keys,
    the website adapter, and the stance report's report-only rows. No stage
    declares it, so it is in no verdict. The v2 panel CSV
    (`stance_panel_selected.csv`), the six-heading grid
    (`stance_heading_panel_selected.csv`) and the report's episode evidence
    rows gain a `max_foot_cop_fore_aft` column; no report schema is bumped, as
    with D-D28's `max_sole_cop_fore_aft`.
  - `pad_telemetry` adds, per foot that declares a span:
    `<label>_foot_cop_fore_aft` (signed), `<label>_foot_cop_fore_aft_gate`
    (whose maximum over the feet is `max_foot_cop_fore_aft` bit for bit),
    `<label>_foot_cop_std_m` and `<label>_foot_cop_dominant_hz` (the spread
    and the strongest frequency of the per-step series, which the window mean
    cannot see), and `<label>_toe_margin_nm` / `<label>_heel_margin_nm`, the
    floor-load moment left to each line. It also adds per digit
    `<label>_<digit>_load_share`, and on a hinged digit
    `<label>_<digit>_load_pos`. A field exists only on a foot that has what it
    measures, as in #620: whole-foot fields on the T. rex only, digit shares
    on the T. rex and the anatomical Compsognathus, load positions on the
    T. rex only, and none of these on the Velociraptor, the Brachiosaurus,
    the Compsognathus robot or the Dibothrosuchus.
  - `gate_progress.npz` and TensorBoard (`diagnostics/eval_<key>`) carry them
    as means over full-horizon episodes, with `max_foot_cop_fore_aft_mean`.
    No `max_foot_cop_fore_aft_over_bar_fraction` is written, because no stage
    declares the bar; it would follow a declared bar as the pad rail's
    over-bar fraction does. The stance replay CSV gains the per-step
    whole-foot CoP, both margins and the per-digit columns.
  - On the T. rex physics-r9 statue, the certification seeds 3042-3081, an
    independent per-substep probe of the same panel (every limb floor contact
    decoded with `mj_contactForce`, in the pad's own frame; not in the
    repository) reads the same whole-foot numbers to 8e-17. Over the 39
    full-horizon episodes `max_foot_cop_fore_aft` has a median of 0.130 (0.058
    to 0.250; seed 3081 nosedives). The whole-foot CoP sits 142.5 mm (right)
    and 145.2 mm (left) ahead of the ankle, median of window means. The toe
    margins are 28.3 and 27.3 N m and the heel margins 30.5 and 31.7 N m. The
    per-step spread has a median of 7.3 and 7.8 mm (up to 21.6 and 23.8 mm).
    The series' strongest frequency (the largest bin of its spectrum, which
    need not hold most of the power) is 0.5 Hz, a slow 2 s fore-aft sway, on
    32 of 39 episodes (right) and 31 of 39 (left), and the window's lowest
    bin, 0.125 Hz, a drift, on the rest, where the sway is still present (on
    seed 3042 the 0.5 Hz bin holds 36 and 45 percent of the detrended power).
    The widest spread is seed 3050's, 21.6 mm (right) and 23.8 mm (left), with
    61 and 57 percent of the detrended power at 0.5 Hz; the right foot's
    series ranges over 106 mm, more than a sinusoid of that spread would
    (61 mm), so its range includes excursions beyond the sway. The stance task
    has no pushes, so this sway is the statue's own. The same panel's
    `stance_quality/v2` verdict (PASS, 39/40), every existing per-episode
    metric and pad field, and its `task_sha256` are identical before and after
    this change. The 649-line digest snapshot is byte-identical and the plant
    manifest is current.
  - `MEASUREMENT_VERSION` is `floor-truth/v2`, and the measurement manifest
    schema `mesozoic.floor-truth-measurement/v3`: its registry block records
    `ankle`, `heel_line` and `toe_line`, and its feet the resolved ankle, span
    and MTP joints. No constant changed and no adopted key reads differently,
    but a registry entry an adopted stage reads did change, and
    `gait/constants.py` versions every such change (see Migration).
  - Tests: the recorder against an independent probe (the ankle's world
    anchor, `mj_contactForce`) and against the vertical-load moment with the
    tilt bound, each hinged digit's load position, the seed-3042 statue
    reading, the registry's span pinned to the pad's edges on the plant,
    refusals, the metric and the telemetry on synthetic and real traces of
    every model, the npz keys, and the replay CSV.
  - The recorder's extra reads cost no measurable time on top of #620's
    recorder. On best-of-five 1,000-step T. rex statue episodes, alternating
    two runs per tree, the recorded episode took 1.175 and 1.186 s before
    this change and 1.201 and 1.175 s after it (0.89 to 0.91 s unrecorded);
    the difference is within the run-to-run spread.
  - Documented in `docs/TRAINING_ARTIFACT_PRESENTATION.md`.
- **Per-foot pad telemetry and the sagittal leg DC at every periodic
  evaluation** (#620; the maintainer's approval of 2026-10-12, after the T. rex
  stance run `20261011_165957` was refused by `stance_quality/v2` on its pad
  centre-of-pressure rail, `max_sole_cop_fore_aft`, failing it on 39 of 40
  episodes at a median of 0.9999; 1/40 clean).
  The left-right ankle offset that put its right pad on the toe edge had
  formed by about 1M steps, 11M steps before the verdict, and was readable
  only from the unnamed per-actuator action DC. Its artifacts kept only the
  per-episode maximum over the feet, so they could not say which foot was on
  which edge, or when the posture formed. Report-only: no gate verdict, reward,
  observation, checkpoint selection, training rollout or digest changes.
  - A new pure module, `environments/shared/gait/pad_telemetry.py`, reduces
    the floor-truth recorder's `EpisodeTrace` per foot. The pad feet are those
    whose `FOOT_SUPPORT` sole is a box (`PadLayout.from_morphology`). It
    records the signed fore-aft pad centre of pressure, in pad half-lengths
    with + toward the toe, averaged over the window steps the gate counts as
    loaded. It records the gate's own per-foot statistic, whose maximum over
    the feet is `max_sole_cop_fore_aft` bit for bit. It records the digits'
    share of the limb's floor force. Each value has a per-episode form
    (`episode_pad_telemetry`) and a per-step form (`pad_step_columns`).
  - A field exists only on a foot that has what it measures. There is no pad
    field on the Velociraptor's capsule feet or the Brachiosaurus's
    ellipsoids. There is no digit share on the Compsognathus robot's soles
    or the Dibothrosuchus's pads. NaN in an existing field means that it was
    not measured.
  - `StageAwareEvalCallback` takes the `model_id`, which `_build_core_callbacks`
    passes through `build_stage_evaluation_callbacks`. It attaches the
    recorder to the in-process evaluation env for each evaluation only, never
    to the training envs, and detaches it as each sub-env finishes its share
    of the panel. It reduces each episode, keeping the values aligned with the
    episode lengths. A failure costs that evaluation's pad values (NaN), never
    the run.
  - `StageGatePlateauCallback` writes the mean over full-horizon episodes of
    each pad field into `gate_progress.npz` and TensorBoard
    (`diagnostics/eval_<key>`). The keys are `<label>_sole_cop_fore_aft`,
    `<label>_sole_cop_fore_aft_gate` and `<label>_digit_share`.
  - Beside them it writes the rail's own per-episode number, averaged over
    the same episodes: `max_sole_cop_fore_aft_mean`, the mean of each
    episode's maximum over the feet. The maximum over the feet of the
    per-foot means is only a lower bound of it: when the worse pad changes
    between episodes, both means can sit under the bar while most episodes
    fail. Where the stage declares the bar (T. rex 0.80, anatomical
    Compsognathus 0.70) it also writes
    `max_sole_cop_fore_aft_over_bar_fraction`, the fraction of those
    episodes above the bar.
  - It also writes `leg_dc_<left>_minus_<right>_<joint>_deg` for hip pitch,
    knee and ankle. Each is the left joint's commanded offset from home minus
    the right one's, in degrees, signed as a rotation about the animal's
    leftward axis. The values come from `action_dc_per_actuator`, mapped
    through the env's own action map (`leg_dc_degrees`). Each joint's sign is
    its axis along the right-to-left foot direction at the home keyframe
    (`sagittal_leg_dc_pairs`). So a mirror-symmetric posture reads 0 on every
    model, including the Compsognathus robot. That robot's hip pitch axes and
    control ranges are mirrored, so a symmetric 0.03 rad hip pitch offset
    read -0.036 or +0.61 as a raw action difference. The mean action of run
    `20261011_165957` reads -3.2° at the ankle and -2.1° at the knee. Each
    side's joints are spelled with the prefix of its foot site. All six
    models have these keys, with two pairs on the quadrupeds.
  - The stance replay CSV (`*_selected_stance.csv`, `*_final_stance.csv`)
    gains `<label>_sole_cop_fore_aft` and `<label>_digit_share` per step, from
    the recorder attached to the replay env.
  - Measured on the T. rex, the recorder adds 34% to an evaluation's
    simulation time (best of three 4,000-step evaluations, 5.66 s to 7.61 s,
    single-threaded). On the 40-episode panel that is about 20 s per
    evaluation on that machine, and about 80 minutes over a 12M stance at the
    default `eval_freq` of 50,000. A model without a box pad attaches no
    recorder.
  - `gate_progress.npz` and the CSV carry no schema or version, and the new
    keys are additive. `StanceEpisodeMetrics`, the v2 panel CSV and every
    report schema are unchanged.
  - Documented in `docs/TRAINING_ARTIFACT_PRESENTATION.md`, which now also
    describes `gate_progress.npz`.
- **A six-heading stance certificate, `stance_quality_heading/v1`** (#617; decision
  D-D32, the maintainer's approval of 2026-10-11;
  `docs/HEADING_INVARIANCE_DESIGN_2026_10.md`). `stance_quality/v2` judges a
  stance at the heading it trained at, and certified the seed-52 T-Rex stance
  `20261009_155723` (39/40), which stands 0/8 with the whole scene turned
  −90° or −45°. The new kind (`curriculum/stance_heading_gate.py`, pure: the
  standard library, v2's statistic, `binomial_lcb` and the gait row type)
  applies the full v2 decision at each heading of a fixed grid of
  whole-scene turns, −90, −45, 0, 45, 90 and 180° (protocol
  `whole_scene_yaw/v1`, fixed in code: a list-valued threshold would defeat
  `same_threshold`, and another grid is another protocol), each heading
  against the zero-action statue rolled at that heading, and passes only
  when every heading passes AND the exact one-sided 95% bound on seeds clean
  at every heading clears `min_clean_stance_lcb` at `min_eval_episodes`
  seeds. It consumes and requires exactly v2's keys, so registering it moves
  no digest and a stance moves to it by its `gate_kind` alone; those two
  keys then state both each heading's panel and the grouped one (37/40 seed
  groups pass at 0.80, LCB 0.8174; 36/40 fail, 0.7856). The rotations of a
  seed are never pooled: one different failing seed at each heading pools
  to 234/240 (an apparent 0.9513) while only 34/40 seeds stand at all six
  (0.7253), and each heading alone still passes v2 at 39/40; the pooled
  count is reported, labelled as not the statistic. Headings are
  canonicalised into (−180°, 180°], and a duplicate or synonym (−180 beside
  180, 0 beside 360), a missing or extra heading, a reorder, or an
  incomplete or duplicated (seed, heading) grid fails closed.
  `reporting/stance_report.py` gains `build_stance_heading_report`: before
  anything is rolled it refuses another kind, a non-strict block or a settle
  window inside the spawn grace, and a stage whose task the turn does not
  rotate (`heading_protocol_refusals`: a live command mode, scheduled pushes
  or a heightfield); it then rolls, on a fresh environment per heading, the
  policy panel through the floor-truth recorder with
  `SpawnYaw(yaw, whole_scene=True)` (at 0° the turn is the identity, and the
  cell is the v2 panel bit for bit) and the statue panel at the same
  heading. A rollout that raises at any heading raises
  `StanceGateReportError` naming it, and nothing is written. The certificate
  (`mesozoic.stance-heading-report/v1`) records what its readers bind: the
  protocol, grid and exact seed list, `policy_deterministic`, the frozen
  VecNormalize record (`_load_policy` now attaches it to its predict), the
  horizon, settle window and control step, the gate view and its digest,
  the measurement manifest (identical at every heading), the handoff
  digests, task and plant identity, every probe marker as `None`, and per
  heading the v2 result and every policy and statue row with its seed and
  heading. A heading-kind stage writes `stance_heading_report.{txt,json}`
  and the grid CSV `stance_heading_panel_selected.csv` (480 rows of
  `STANCE_HEADING_PANEL_FIELDNAMES`: role and heading, v2's columns, the
  certificate's stamps), and no `stance_gate_report.*` or
  `stance_panel_selected.csv` (`write_stance_gate_report` refuses a
  certificate); the v2 panel reader refuses a file with a `yaw_deg` column.
  The judge (`reporting.gates.stance_heading_report_refusals`, its arm in
  `evaluate_stage_gate`) runs v2's binding checks, split out unchanged into
  `_stance_report_identity_refusals`, `_statue_block_refusals` and
  `_stance_report_stage_refusals`, refuses by name a v2 report (it certifies
  yaw zero only and cannot be promoted), the report-only probe, no
  certificate (never a fallback to the yaw-zero panel), another protocol,
  grid or seed list (the fresh-audit block at 7042 among them), a stochastic
  policy, any normalization but the frozen statistics, a `gate_sha256` that
  is not the judged gate's and a reused or unreducible statue block, and
  re-derives the verdict from every (seed, heading) row, refusing a recorded
  `passed`, clean-seed count, per-heading count or clean flag the rows
  contradict; `evaluate_recorded_gate` reads a heading history as `None`.
  `generate_stage_artifacts` rolls the certificate through the same handoff
  selection and non-fatal path as v2 (no report fails the gate), copies the
  admitted certificate's grouped numbers into `gate_verdict.json`'s
  `stage_result` (`stance_heading_n_seeds`, `stance_heading_clean_seeds`,
  `stance_heading_clean_seed_fraction`, `stance_heading_clean_seed_lcb`,
  `stance_heading_statue_mean_reward_min`; schema unchanged), and points the
  hold, release and impulse probes at the certificate's 0° cell
  (`heading_zero_panel`). Publication re-derives a recorded PASS from the
  grid CSV alone, bound to the certification-panel role, the certified pair,
  the task, the measurement definition and the gate digest, with the policy
  and statue grids complete and each heading's stamped statue reference what
  its rows reduce to. The backfill tool backfills a heading directory from
  `stance_heading_report.json` only. Ancestor reuse gains rule 8: a node
  whose current gate is the certificate is reused only on a certificate the
  judge admits and whose re-derived verdict passes, evaluated at the source
  (ancestor records copy no certificate), so a forged or stale PASS beside a
  failing or missing certificate is refused. The command-line curriculum's
  post-training judge (`_post_training_stance_verdict`) covers both
  floor-truth kinds, with `judged_by`
  `train_base.train_curriculum/stance_heading_report`; the in-training
  manager refuses the kind as it refuses v2. `stance_gate_report.py`
  builds the certificate for such a stage (`--out-dir` writes its files;
  `--seed 7042` rolls the fresh-audit grid, which the judge refuses as the
  gate's evidence); the notebook's zero-action preflight scores its 0° cell
  only. The catalog and the website render the kind (`formatGate`'s new arm,
  no new raw keys) and headline the grouped bound. Measured on one CPU at 40
  seeds (MuJoCo 3.10.0): the T-Rex r9 statue's certificate in 6m13s, clean
  on 39/40 seeds at every heading (seed 3081's nosedive), grouped bound
  0.8868 (its rewards agree across headings to 1.3e-10); a checkpoint's
  certificate, with its six statue panels, about 13 minutes on the T-Rex and
  21 on the anatomical Compsognathus, against about 2.1 and 3.6 for v2; the
  report JSON is about 1.7 MB and the CSV 0.7 MB. Tests: the pure statistic
  and grid (`test_stance_heading_gate.py`), 21 corruptions and every refusal
  of the judge (`test_stance_heading_judge.py`), real-physics grids at 2
  seeds (`test_stance_heading_report.py`: a statue per heading, the turned
  statue repeating the unturned one to 1e-8 on both species, the 0° cell
  equal to the v2 panel, exact JSON and CSV round trips, a mid-grid failure
  leaving nothing), the case where yaw zero passes and −90° and −45° fail
  and the stage gate, verdict, reuse (a forged PASS included), publication,
  backfill and the CLI curriculum all refuse it
  (`test_stance_heading_end_to_end.py`), and heading twins of the dispatch,
  publication, backfill, reuse, artifact, verdict, CSV, preflight and probe
  tests.
- **Handoff tooling warns, without blocking, when a whole-scene row of the
  heading probe is below 0.95, and every biped stance declares the probe**
  (#610; the maintainer's decisions of 2026-10-11;
  `docs/HEADING_INVARIANCE_DESIGN_2026_10.md`). `stance_quality/v2`
  certified the seed-52 T-Rex stance `20261009_155723` (39/40), whose own
  report-only probe stands 0/8 with the whole scene turned −90° and −45°.
  The probe stays report only (D-D27) until the heading certificate replaces
  it, and now says so where a stance is handed on.
  `reporting/stance_report.py` gains `heading_probe_warnings` (pure; it
  never raises), `stage_heading_probe_warnings` and
  `declares_heading_probe`. A whole-scene row (`whole_scene: true`) whose
  policy full-horizon fraction is below `HEADING_PROBE_WARN_BELOW`, 0.95,
  the v2 per-panel `min_full_horizon_fraction` rail, warns, naming each such
  row with its statue fraction: one fall in the probe's 8 episodes warns,
  exactly 0.95 does not. The unturned control and the animal-only rows never
  warn (the gate judges the first; the second add the prey's bearing). A
  statue below 0.95 on a whole-scene row, or one not recorded as a share, is
  a separate warning that the control did not stand or is unmeasured. A
  record that cannot be read (another schema, no episodes or rows, a
  missing, non-finite or out-of-range policy fraction, no whole-scene row)
  warns that heading robustness is unmeasured, and so does an absent probe
  on a stage whose recorded `[curriculum]` declares one
  (`stance_probe_spawn_yaw_deg` under a stance gate kind; the command-line
  curriculum rolls no probe). The warnings are a `HANDOFF WARNING` block of
  `stance_heading_probe.txt`, which
  `stance_gate_report.py --spawn-yaw-offsets` prints, and warning-level log
  lines where `generate_stage_artifacts` writes the probe, where `train` (a
  cross-stage load of the handoff that `select_handoff_checkpoint` picks in
  `<stage_dir>/models/`, the checkpoint the probe rolled) and
  `train_curriculum` start a node on its parent's handoff, and where
  `record_ancestor` reuses a node, before it records. Report only: no
  verdict, gate, exit code or schema changes. `stance_heading_probe.json`
  stays `mesozoic.stance-heading-probe/v1` with the same keys, and
  `gate_verdict.json`, `stance_gate_report.{txt,json}` and the files an
  ancestor record copies are unchanged. The T-Rex stance declares the probe
  (D-D27), and by the maintainer's approval of the same day so do the
  Velociraptor, anatomical Compsognathus and Compsognathus robot stances:
  `stance_probe_spawn_yaw_deg = [-90.0, -45.0, 45.0, 90.0]` in
  `configs/velociraptor/stage1_balance.toml`,
  `configs/compsognathus/stance.toml` and
  `configs/compsognathus_robot/stance.toml` (the robot's
  `stance_quality/v1` report rolls the probes too), each costing a stance
  stage 18 panels of 8 episodes. The two quadrupeds' `reward_and_length/v1`
  stances roll no stance report and declare none; a probe file of any
  species is read. Rolled through `generate_stage_artifacts`' probe writer
  on the zero-action statue, each new species' statue stands 8/8 on every
  row, with the unturned control's reward on each whole-scene row, and
  raises no warning. A command-line curriculum run rolls no probe, so its
  handoff logs the "missing … heading robustness is unmeasured" warning for
  these species too, as for the T-Rex: the maintainer's choice to warn on a
  declared probe that is missing. No plant, policy-interface, task, gate or
  hyperparameter digest moves; 6 `stage_config_view_sha256` lines of the
  digest snapshot move (PPO and SAC of the three stance nodes that now
  declare the probe), and its other 643 lines are equal. Tests: seed 52's
  rows, the 0.95 boundary and one fall in eight, the control and animal-only
  rows, the statue control, unmeasured and malformed records (an integer
  past float range included), the probe text with an unchanged JSON key set,
  the stage reader on the writer's file and on a probe declared by a
  recorded T-Rex or Velociraptor stance config and absent, and on a
  Velociraptor stance recorded before its config declared one, every biped
  stance config declaring the probe and the two quadrupeds' not (by the
  species manifest's gait), the probe-write log, `record_ancestor`'s log
  with its copied files unchanged, and `train` and `train_curriculum`
  logging the warning and still training, `train` only on the handoff.
  Records: KNOWN_ISSUES's heading entry, a dated note on the heading
  design's Velociraptor sentence, and the T-Rex, Velociraptor and
  Compsognathus READMEs; the consolidation plan's table, which also records
  the landings of #604 to #608.
- **A floor-truth stance gate, `stance_quality/v2`, and the measurement
  library it reads** (#599, decision D-D23, 2026-10-06;
  `docs/investigations/STANCE_HACK_AUDIT_2026_10.md`, and the dated §11 of
  `docs/GAIT_QUALITY_PLAN_2026_09.md`). The 2026-10 stance-hack audit
  replayed every current stance node and found none standing as its
  zero-action statue does: the three passing trex stances, seed 42 included,
  hop at reset inside the unscored settle window and prop a leg on a rolled
  pad; velociraptor crouches on saturated servos and chatters; compsognathus
  marches on one toe; the robot stacks its feet; dibothrosuchus certified
  the untrained statue, three-legged in some episodes. v1 reads the touch
  sensors, averages duty over the panel and skips the settle window, so it
  admits all of them, as the reward rails do. The new package
  `environments/shared/gait/` measures stance on floor truth:
  `SubstepContactRecorder` attaches to `_substep_probe_hook` (chaining any
  hook already there, detached in a `finally`), decodes each contact's
  normal force exactly as `mj_contactForce` reports it and reduces an
  episode at once into an `EpisodeTrace`; `episode_stance_metrics` turns
  that into a `StanceEpisodeMetrics` row (NaN when unmeasured), which
  round-trips through CSV. A leg is down when its floor force exceeds 0.1 N
  on at least half a step's substeps (`MEASUREMENT_VERSION =
  "floor-truth/v1"`, with a 20 ms debounce, a 0.1 s spawn grace, a 1 s load
  window, saturation at |a| ≥ 0.99 and a 2° rolled-sole bar;
  `measurement_manifest` and its sha256 record them). `morphology.py` holds
  an explicit support-geom registry for all six species, four of which
  differ from the generic rule (trex without the metatarsus capsule,
  velociraptor with digit IV and the metatarsus and no sole, the robot
  without its roll cheeks, brachiosaurus without the metapodial capsule).
  The recorder leaves the trajectory bit-identical, its forces equal a
  per-contact `mj_contactForce` loop exactly, and it costs +0.19 to +0.41 ms
  per control step on a quiet machine (measured on its prototype, which
  recorded less), so it runs on evaluation envs only. No species env module
  imports the package.

  The kind (`curriculum/stance_gate_v2.py`, pure) classifies each episode of
  the 40-episode certification panel (seeds 3042 + i) clean when it reaches
  the horizon and every declared criterion holds on a finite metric, and
  certifies the exact Clopper-Pearson bound on the clean count
  (`recovery_gate.binomial_lcb`; a bar of 0.80 admits 37/40). Required:
  `min_eval_episodes`, `min_clean_stance_lcb`, `settle_steps`,
  `min_all_feet_support`, `max_touchdown_rate`, `max_window_displacement_m`,
  `min_foot_load_share`, `max_actuator_saturation_fraction`,
  `max_settle_airborne_substeps` and `max_settle_peak_floor_force_bw` (the
  settle keys after the spawn grace); optional, applied only when declared:
  foot-on-foot, phantom support, non-foot load, the windowed load share, the
  settle width change, support-geom duty and coverage, sole tilt and tilt
  excess, sole corner lift, loaded sole contacts, the reward rail, a
  full-horizon floor and two statue-relative ratios judged against a
  zero-action panel rolled in the same report on the same seeds.
  `GATE_SCHEMA_VERSION` stays 1.

  Its plumbing fails closed at every entry point: the stance report writes a
  `mesozoic.stance-gate-report/v3` report for the kind (the code that scored
  it, the thresholds with `settle_steps`, the measurement manifest, the
  handoff digests, the `task_sha256` of the `[env]` block and plant it
  rolled, the statue block, every episode's metrics and reasons) and a v2
  `stance_panel_selected.csv` with its own columns; probes on a v2 stage
  keep v1's report; the judge refuses a report scored by other code, under
  other thresholds or another measurement, for another checkpoint, task,
  horizon, control step or seed block, with the plant contract waived, with
  a statue panel that is not a separate zero-action one, or with a statue
  reference its rows contradict, refuses a stage that records no task or
  horizon, and re-derives the verdict, refusing a report it cannot
  re-derive; publication re-derives it from the CSV, bound to the certified
  pair, the stage's recorded task and this checkout's measurement
  definition (constants and registry entry), and v1's reader refuses a v2
  file; backfill refuses rather than write a FAIL for any report the judge
  cannot re-derive; `evaluate_recorded_gate` never falls into the reward
  arm; `gate_verdict.json` persists the clean count, panel size, clean
  fraction, bound and statue reward; the species catalog gains the kind's
  headline; `zero_action_baseline.py` prints the stance gate's verdict on
  the statue for both stance kinds, which closes its KNOWN_ISSUES entry. A
  v1 report gains a `scored_gate_kind` key (`stance_quality/v1`), which the
  v2 judge reads, and its text NOTE on a stage of another kind says "not a
  stance kind" where it named v1; v1's panel CSV and its stance-stage text
  are unchanged. In training
  `CurriculumManager` refuses the kind with a log, and `train_curriculum`
  judges a v2 node after training from the stance report on its handoff pair
  (`judged_by` `train_base.train_curriculum/stance_gate_report`). CI's SB3
  job runs `test_gait_vec_path.py`.

  Validated on 2026-10-06 on 40-episode panels through the report path, with
  per-species candidate bars: every statue of the six species is clean on
  40/40, and every one of the twelve audited trex, velociraptor,
  compsognathus, compsognathus_robot and dibothrosuchus checkpoints fails
  (ten at 0/40; the two untrained dibothrosuchus nodes at 27/40 and 21/40).
  Out of scope: no stage TOML declares the kind, so every stance stage still
  certifies as before and no digest moves (the golden is current at 656
  lines); the in-training screen (the recorder on the evaluation env) is not
  built; the website's gate adapter has no v2 arm yet, which a stage's
  adoption needs; each species' adoption is a gate revision of its own, with
  bars re-measured on its own statue. Records: D-D23 in
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2 and the consolidation plan's table;
  KNOWN_ISSUES's stance-gate entry carries the audit's findings and the new
  kind, and its `zero_action_baseline.py` entry is deleted.

### Migration
- **Every T. rex checkpoint must be retrained; none can be widened** (#643;
  T. rex physics revision 9 → 10, decision D-D44). Each load path compares
  the checkpoint's recorded `physics_revision` and `physics_sha256` with the
  plant's and refuses a mismatch (`PlantCompatibilityError`; measured with a
  band checkpoint on an r10 tree: "physics_revision: checkpoint=9,
  current=10"), `widen_checkpoint.py` needs the same physics digest, and
  ancestor reuse refuses every recorded T. rex node (its `task_sha256` moved),
  so `TRUNK_FROM = "auto"` and `--trunk-from` find no T. rex trunk. No
  certified trunk is lost: no T. rex stance has certified on physics r9. What
  stops loading on `main`: every physics-r9 T. rex checkpoint, among them the
  symmetry screen's r9 runs and walking pilot 1 with its parent (r9 arm L
  seed 60 at 6,291,456 steps; D-D36), every T. rex stance screen run of
  D-D37, D-D39, D-D40, D-D41 and D-D43, all uncertified, and walking pilot
  2's parent, `stance3_armLT_s60`'s final; walking pilot 2 can still run on
  its own r9 branch, but its result would not carry to r10 (inferred). The
  policy interface (r15) and visual layer (r4) do not move, and no other
  species changes. Nothing is to train on r10 until the maintainer approves
  the first r10 round's own criteria.
- **`BEHAVIOR = "stand"` is refused; use `"stance"`** (D-D38). `"stance"`
  trains exactly what `"stand"` did since D-D35: the stance stage alone, on
  every model. A run started with `BEHAVIOR = "stand"` re-enters with
  `BEHAVIOR = "stance"` under the same `RUN_ID`: the same chain, the same
  target key (`"1"`) and the same trunk selection, since no artifact records
  the label. The same holds for `curriculum --target stand` and
  `train_curriculum(target="stand")` (use `stance`), for a custom
  `stages.toml` declaring `recipe = "stand"` (declare `recipe = "stance"` on
  the stance entry; the loader refuses `"stand"` naming it) and for a species
  manifest `deliverable = "stand"` (use `"stance"`). Each refusal reads
  "'stand' was renamed to 'stance' (decision D-D38); use 'stance'".
- **`BEHAVIOR = "stand"` trains the stance stage alone on every model**
  (D-D35). On the T. rex and both Compsognathus models it used to mean
  stance then push recovery; that chain is now `BEHAVIOR = "recover"`. A
  run started with `"stand"` before this change and left before its
  recovery node finished continues with `"recover"`: re-entered with
  `"stand"`, it targets stance alone and does not resume recovery.
- **A `stance_quality/v2` or six-heading report or panel measured under
  `floor-truth/v1` is refused, and is judged again only once re-rolled**
  (`MEASUREMENT_VERSION` `floor-truth/v2`, the whole-foot centre of
  pressure). The judge refuses a report whose measurement constants or
  registry entry are not this checkout's, and publication refuses a panel CSV
  stamped with another version. Ancestor reuse (`--trunk-from`, reuse rule 8,
  which re-derives a six-heading certificate: the T. rex's and the anatomical
  Compsognathus's stance nodes) and `backfill_gate_verdict.py` apply the same
  check, so such a node certified under v1 is reused, and a v1 report
  backfilled, only after it is re-judged on its handoff pair. Re-roll the
  panel on the handoff pair (the notebook JUDGE branch, or
  `generate_stage_artifacts`). A re-rolled panel scores every earlier key
  bit for bit: on the T. rex statue's 40 seeds the verdict, every metric and
  every pad field are identical. No checkpoint, task, gate or plant digest
  moves, so nothing needs retraining.
- **Nothing recorded under the old identifiers resolves** (D-D33). These are
  refused or not found when they are named `velociraptor`, `trex`,
  `brachiosaurus`, `dibothrosuchus`, `compsognathus` or
  `compsognathus_robot`: checkpoints, VecNormalize statistics, run
  directories, bundles, provenance, gate verdicts, ancestor and trunk
  records, stage configs, result summaries of schema 3 or 4, and W&B runs.
  The plant contract refuses every existing checkpoint (its plant identity
  is v1, and every policy-interface revision moved); start fresh weights and
  statistics.

  `TRUNK_FROM = "auto"` and `--trunk-from` find no trunk, so the first run
  of each model trains every node.

  This change migrates, identity only, the two recovery calibrations (their
  `calibration_id` renamed, the old one kept in `restamp_history`), the
  Phase C reset golden and the four committed result summaries. Drive trees
  under `mesozoic-labs/logs/<old id>/` stay where they are; the Drive
  summary notebook lists them as skipped, and new runs write
  `logs/<species>/<embodiment>/`. To read or rerun a pre-rename run, check
  out `f40c11d3`.

  Old → new, one model at a time (the species-level material, README,
  `data/`, `scripts/` and `tests/`, moves from `environments/<old id>/` to
  `environments/<species>/`):
  - `velociraptor` (alias `raptor`) → `velociraptor_mongoliensis`,
    `anatomical`. `RaptorEnv` → `VelociraptorMongoliensisEnv` (species base)
    and `VelociraptorMongoliensisAnatomicalEnv`; `MesozoicLabs/Raptor-v0` →
    `MesozoicLabs/VelociraptorMongoliensis-Anatomical-v0`;
    `environments.velociraptor.envs.raptor_env` →
    `environments.velociraptor_mongoliensis.base_env` and
    `environments.velociraptor_mongoliensis.anatomical.env`;
    `environments/velociraptor/assets/raptor.xml` →
    `environments/velociraptor_mongoliensis/anatomical/assets/velociraptor_mongoliensis_anatomical.xml`;
    `configs/velociraptor/stage1_balance.toml`, `stage2_locomotion.toml`
    and `stage3_strike.toml` →
    `configs/velociraptor_mongoliensis/anatomical/stance.toml`,
    `locomotion.toml` and `strike.toml`.
  - `trex` (alias `t-rex`) → `tyrannosaurus_rex`, `anatomical`. `TRexEnv` →
    `TyrannosaurusRexEnv` (species base) and `TyrannosaurusRexAnatomicalEnv`;
    `MesozoicLabs/TRex-v0` → `MesozoicLabs/TyrannosaurusRex-Anatomical-v0`;
    `environments.trex.envs.trex_env` →
    `environments.tyrannosaurus_rex.base_env` and
    `environments.tyrannosaurus_rex.anatomical.env`;
    `environments/trex/assets/trex.xml` →
    `environments/tyrannosaurus_rex/anatomical/assets/tyrannosaurus_rex_anatomical.xml`;
    `configs/trex/` → `configs/tyrannosaurus_rex/anatomical/` (`stance.toml`,
    `recovery.toml`, `locomotion.toml` and `behavior.toml` keep their
    names).
  - `brachiosaurus` (alias `brachio`) → `brachiosaurus_altithorax`,
    `anatomical`. `BrachioEnv` → `BrachiosaurusAltithoraxEnv` (species base)
    and `BrachiosaurusAltithoraxAnatomicalEnv`; `MesozoicLabs/Brachio-v0` →
    `MesozoicLabs/BrachiosaurusAltithorax-Anatomical-v0`;
    `environments.brachiosaurus.envs.brachio_env` →
    `environments.brachiosaurus_altithorax.base_env` and
    `environments.brachiosaurus_altithorax.anatomical.env`, and
    `environments.brachiosaurus.mjx_config` →
    `environments.brachiosaurus_altithorax.anatomical.mjx_config`;
    `environments/brachiosaurus/assets/brachiosaurus.xml` →
    `environments/brachiosaurus_altithorax/anatomical/assets/brachiosaurus_altithorax_anatomical.xml`;
    `configs/brachiosaurus/stage1_balance.toml`, `stage2_locomotion.toml`
    and `stage3_food_reach.toml` →
    `configs/brachiosaurus_altithorax/anatomical/stance.toml`,
    `locomotion.toml` and `food_reach.toml`.
  - `dibothrosuchus` (alias `dibo`) → `dibothrosuchus_elaphros`,
    `anatomical`. `DibothrosuchusEnv` → `DibothrosuchusElaphrosEnv` (species
    base) and `DibothrosuchusElaphrosAnatomicalEnv`;
    `MesozoicLabs/Dibothrosuchus-v0` →
    `MesozoicLabs/DibothrosuchusElaphros-Anatomical-v0`;
    `environments.dibothrosuchus.envs.dibothrosuchus_env` →
    `environments.dibothrosuchus_elaphros.base_env` and
    `environments.dibothrosuchus_elaphros.anatomical.env`, and
    `environments.dibothrosuchus.mjx_config` →
    `environments.dibothrosuchus_elaphros.anatomical.mjx_config`;
    `environments/dibothrosuchus/assets/dibothrosuchus.xml` →
    `environments/dibothrosuchus_elaphros/anatomical/assets/dibothrosuchus_elaphros_anatomical.xml`;
    `configs/dibothrosuchus/stage1_balance.toml`, `stage2_locomotion.toml`
    and `stage3_snap.toml` →
    `configs/dibothrosuchus_elaphros/anatomical/stance.toml`,
    `locomotion.toml` and `snap.toml`.
  - `compsognathus` (alias `compso`; the "biological" model) →
    `compsognathus_longipes`, `anatomical`. `CompsognathusEnv` →
    `CompsognathusLongipesEnv` (species base, shared with the robot) and
    `CompsognathusBiologicalEnv` → `CompsognathusLongipesAnatomicalEnv`;
    `MesozoicLabs/Compsognathus-v0` →
    `MesozoicLabs/CompsognathusLongipes-Anatomical-v0`;
    `environments.compsognathus.envs.compsognathus_env` →
    `environments.compsognathus_longipes.base_env` and
    `environments.compsognathus_longipes.anatomical.env`;
    `environments/compsognathus/assets/compsognathus.xml` →
    `environments/compsognathus_longipes/anatomical/assets/compsognathus_longipes_anatomical.xml`;
    `configs/compsognathus/` → `configs/compsognathus_longipes/anatomical/`
    (file names kept); calibration ID
    `compsognathus-fixed-command-recovery-2026-10-v2` →
    `compsognathus_longipes-anatomical-fixed-command-recovery-2026-10-v2`.
  - `compsognathus_robot` (alias `compso-robot`,
    `variant_of = "compsognathus"`) → `compsognathus_longipes`, `robot`.
    `CompsognathusRobotEnv` → `CompsognathusLongipesRobotEnv`;
    `MesozoicLabs/CompsognathusRobot-v0` →
    `MesozoicLabs/CompsognathusLongipes-Robot-v0`;
    `environments.compsognathus.envs.compsognathus_env` →
    `environments.compsognathus_longipes.robot.env`;
    `environments/compsognathus/assets/compsognathus_robot.xml`, its
    `meshes/` and `references/` →
    `environments/compsognathus_longipes/robot/assets/compsognathus_longipes_robot.xml`,
    `robot/assets/meshes/` and `robot/references/`;
    `configs/compsognathus_robot/` → `configs/compsognathus_longipes/robot/`
    (file names kept); calibration ID
    `compsognathus_robot-fixed-command-recovery-2026-09-v1` →
    `compsognathus_longipes-robot-fixed-command-recovery-2026-09-v1`.
- **A T-Rex or anatomical Compsognathus stance is judged, reused and
  published on its six-heading certificate only** (D-D32). Both stances'
  `gate_sha256` moves, so reuse rule 7 refuses every verdict judged under
  their v2 block (none certified on the current plants: T-Rex r9 and
  Compsognathus r2 have no reusable stance), and rule 8 re-derives the
  certificate of every node reused under it; a stage directory with only a
  v2 report is re-judged (the notebook JUDGE branch /
  `generate_stage_artifacts`, or `stance_gate_report.py`), which rolls 12
  panels, about 13 minutes on the T-Rex and 21 on the Compsognathus. Bundles
  already published stay as they are. Digest golden (643 lines, on top of
  D-D31's heading-free observations below): 6 lines move, the T-Rex and
  Compsognathus stances' `gate_sha256` and `stage_config_view_sha256.PPO`
  and `.SAC` (T-Rex `gate_sha256` ab4191e7 → 06375c21, Compsognathus
  8938b294 → 4df04357); no task, recipe, plant, measurement or other
  species' line moves.
- **Every T-Rex and anatomical Compsognathus checkpoint must be retrained;
  none can be widened** (T-Rex policy interface revision 13 → 14, anatomical
  Compsognathus 3 → 4, observation schema `bipedal-target-heading-free/v1`;
  decision D-D31). The plant contract refuses each recorded checkpoint and
  VecNormalize sidecar on its `policy_interface_sha256` and `observation_dim`
  on every load path (the trainer's warm start, resume and statistics load,
  `load_sb3_checkpoint`, `evaluation.evaluate`, the stage video and the
  behavior preparation), `widen_checkpoint.py` refuses the width (64 + 3 ≠ 63
  and 56 + 3 ≠ 55: the observation was reframed, not widened), and ancestor
  reuse refuses every recorded node of the two species (its `task_sha256`
  moved), so `TRUNK_FROM = "auto"` and `--trunk-from` find no trunk for them.
  Stranded: the pre-migration T-Rex stances `20261008_163256`,
  `20261008_163410` and `20261009_155723` and the recovery trained from it
  (already refused on physics r9, and research-only by the maintainer's
  ruling), the T-Rex runs of the 15-run reward pilot batch (#608), and any
  anatomical Compsognathus run on interface r3. Start fresh PPO weights,
  optimizer state and VecNormalize statistics. A Compsognathus recovery
  session re-freezes its `gate_resolution.json` on the restamped
  `configs/compsognathus/recovery_calibration.json` (its `profile_sha256`
  moved). `compsognathus_robot`, the Velociraptor, the Brachiosaurus and the
  Dibothrosuchus keep every digest.
- **Every velociraptor, brachiosaurus and dibothrosuchus checkpoint must be
  retrained; none can be widened** (velociraptor physics revision 3 → 4,
  brachiosaurus 4 → 5, dibothrosuchus 1 → 2; decision D-D30). Each load path
  compares the checkpoint's recorded `physics_sha256` with the plant's and
  refuses a mismatch (`PlantCompatibilityError`), `widen_checkpoint.py`
  needs the same physics digest, and ancestor reuse refuses every recorded
  node of the three species (their `task_sha256` moved), so
  `TRUNK_FROM = "auto"` and `--trunk-from` find no trunk for them. Stranded:
  every velociraptor physics-r3 checkpoint, none of which certified under
  `stance_quality/v2` (the 6M stances `20261007_132051` and
  `20261009_155731`, the nine velociraptor runs of the 15-run reward pilot
  batch, #608, and any velociraptor run the maintainer has in progress on
  r3); every dibothrosuchus physics-r1 checkpoint, the statue-level stance
  PASS of `20260923_020654` (`1d3527cc…`) and the session-4 re-run
  `20260928_012318` included; nothing on brachiosaurus, which has not
  trained since its July run on physics r1. The policy interfaces (r11, r8,
  r7) and visual layers (r4, r2, r1) do not move.
- **Every T-Rex checkpoint must be retrained; none can be widened**
  (T-Rex physics revision 8 → 9, decision D-D29). As at r8, each load path
  compares the checkpoint's recorded `physics_sha256` with the plant's and
  refuses a mismatch (`PlantCompatibilityError`), `widen_checkpoint.py`
  needs the same physics digest, and ancestor reuse refuses every recorded
  T-Rex node (its `task_sha256` moved), so `TRUNK_FROM = "auto"` and
  `--trunk-from` find no T-Rex trunk. No trunk is lost: the physics-r8
  stances `20261006_185343`, `20261006_185704`, `20261008_163256` and
  `20261008_163410` failed `stance_quality/v2`, the one r8 PASS,
  `20261009_155723`, and the recovery trained from it are research-only by
  the maintainer's ruling of 2026-10-11, and they, the D-D27 study
  checkpoints and the T-Rex runs of the 15-run reward pilot batch are
  pre-migration runs the heading-free policy revision would strand anyway.
  The policy interface (r13) and visual layer (r4) do not move. Train no
  T-Rex between this revision and that one: an r9 checkpoint on policy
  interface r13 would be stranded by it.
- **Every anatomical Compsognathus checkpoint must be retrained; none can be
  widened** (Compsognathus physics revision 1 → 2, policy interface revision
  2 → 3, visual revision 1 → 2; decision D-D26). The plant contract refuses
  each recorded anatomical Compsognathus checkpoint on its `physics_sha256`
  and `policy_interface_sha256`, `widen_checkpoint.py` needs the same
  physics digest, and ancestor reuse refuses every recorded node (its
  `task_sha256` moved, and stance's `gate_sha256` too), so
  `TRUNK_FROM = "auto"` finds no Compsognathus trunk: the certified stance
  and walker of `20260921_203149` and the stance of `20261001_225856` start
  again from a fresh stance, which now certifies under `stance_quality/v2`.
  A recovery session also re-freezes its `gate_resolution.json` on the
  re-measured `configs/compsognathus/recovery_calibration.json`.
  `compsognathus_robot` keeps every digest. The bundles already published
  stay as they are; the catalog labels their verdicts with the gate they
  were earned under.
- **A T-Rex stance run, verdict or `stance_quality/v2` report recorded under
  D-D27's gate block is not reused, and its report is judged again only once
  its panel is re-rolled** (decision D-D28). The stance `gate_sha256` moves
  (the block gains `max_sole_cop_fore_aft`), so `resume_same_stage` and
  ancestor reuse refuse a T-Rex stance node or verdict recorded under
  D-D27's block, and a T-Rex v2 report or `stance_panel_selected.csv`
  rolled before D-D28 is refused under the current block (it records no
  reading for the new key); re-roll it on the handoff pair (the notebook
  JUDGE branch, or `generate_stage_artifacts`). Nothing certified is
  stranded: no T-Rex stance has run on the D-D27 task.
- **A T-Rex stage run started on the task of D-D24 cannot resume across the
  stance follow-up, and its nodes are not reused** (decision D-D27). Every
  T-Rex `task_sha256` moves (the stance task's seven new `TRexEnv` kwargs
  enter every T-Rex stage's effective configuration, at their defaults where
  a stage does not set them), and so do the stance `gate_sha256` and the
  stance and recovery `hyperparameters_sha256.PPO`, so `resume_same_stage`
  and ancestor reuse refuse a node recorded under D-D24's task. Nothing
  certified is stranded: the plant is unchanged, so every physics-r8
  checkpoint still loads, and no T-Rex stance has certified on r8 (the two
  runs on D-D24's task, `20261006_185343` and `20261006_185704`, are both
  `stance_quality/v2` FAILs). Start the next T-Rex stance fresh.
- **Every Velociraptor checkpoint must be retrained; none can be widened**
  (Velociraptor physics revision 2 → 3, policy interface revision 10 → 11,
  visual revision 3 → 4; decision D-D25). The plant contract refuses each
  recorded Velociraptor checkpoint on its `physics_sha256` and
  `policy_interface_sha256`, `widen_checkpoint.py` needs the same physics
  digest, and ancestor reuse refuses every recorded node (its `task_sha256`
  moved, and stance's `gate_sha256` too), so `TRUNK_FROM = "auto"` finds no
  Velociraptor trunk: the certified stance and walker of `20260922_125248`
  and the run `20260929_112244` that reuses that stance start again from a
  fresh stance, which now certifies under `stance_quality/v2`. The bundles
  already published stay as they are; the catalog labels their verdicts with
  the gate they were earned under.
- **Every T-Rex checkpoint must be retrained; none can be widened**
  (T-Rex physics revision 7 → 8, decision D-D24). Each load path compares
  the checkpoint's recorded `physics_sha256` with the plant's and refuses a
  mismatch (`PlantCompatibilityError`); `allow_legacy_plant` covers only
  unstamped artifacts, and `widen_checkpoint.py` needs the same physics
  digest, so it refuses too. Ancestor reuse refuses every recorded T-Rex
  node (its `task_sha256` moved, and stance's `gate_sha256` too), so
  `TRUNK_FROM = "auto"` and `--trunk-from` find no T-Rex trunk: the
  certified seed-42 and seed-44 stances (`20260914_123816`,
  `20260920_010912`), the seed-44 recovery, both walkers
  (`20260914_123816`, `20260925_033501`) and the hunt start again from a
  fresh stance, which now certifies under `stance_quality/v2`. A recovery
  session also re-freezes its `gate_resolution.json`
  (`harnesses/freeze_recovery_gate.py`) on the new certified stance. The
  bundles already published stay as they are; the catalog labels their
  verdicts with the gate they were earned under.

### Changed
- **The centre-of-mass stance reduction moves into the gait package, beside
  the evaluation telemetry** (#653; a pure move: no key, number, bar or test
  outcome changes; the maintainer's decision of 2026-10-29;
  `environments/shared/tests/stance_track_helpers.py` is now
  `environments/shared/gait/stance_track.py`). A gate, an evaluation callback
  or a pre-registered policy check may import `environments/shared/gait/`
  but not the tests, so this is the step before any such use of the
  reduction; none is made here. The module is unchanged but for one line of
  its docstring; its three importers, all in `environments/shared/tests/`
  (the CoM stance helpers, the calibration tool and the reduction's
  synthetic tests), import `environments.shared.gait.stance_track`, and the
  docstrings that named the old module name the new one. It imports only the
  standard library and numpy, which `test_gait_stance_metrics.py`'s list of
  pure gait modules now enforces, and the gait package's module list names
  it. As package code it ships in the wheel and counts in the coverage union;
  a script outside the repository that imports the old path needs the new
  one. No environment, physics, reward, gate, observation, policy interface
  or digest changes: the exact reward captures (`digest_snapshot --exact`,
  37,016 lines, on the base and on this change on one machine) are
  byte-identical.
- **The development version is `0.4.0.dev2`** (#647; `pyproject.toml`; the
  maintainer's decision of 2026-10-25 20:13 UTC in the stance run log: bump
  it once the current work is done, as #619 did, carried by the next records
  change). Every run records the package version in its provenance
  (`dependency_versions`) and its stage config (`library_version`), so the
  runs started from `main` from here on are told apart from the `0.4.0.dev1`
  runs: every run since 2026-10-12, the three runs of the first T. rex
  physics-r10 stance round among them (launched on 2026-10-27 from the
  never-merged branch `claude/screen-trex-r10-armltsb` at `8737cd3d`, on
  `main` `06b1a8e1`, whose `pyproject.toml` reads `0.4.0.dev1`). No code,
  config, plant or digest changes; `[Unreleased]` stays open.
- **The T. rex's tail pitch springs are preloaded, so its passive tail holds
  level, and the stance's tail target follows it** (#643; breaking — plant
  change, physics revision 9 → 10; every physics-r9 T. rex checkpoint is
  invalidated; decision D-D44, the maintainer's decisions of 2026-10-25 after
  a posture study, a CPU probe and CPU checks, none of which trained;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2, `docs/CONSOLIDATION_PLAN_2026_09.md`
  §6, `configs/plant_versions.toml` note 20, and the appended §25 of
  `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` with its
  evidence in `stance5/tail_sr3/` of `docs/investigations/stance_2026_10/`).
  This "physics r10" is the plant counter's next integer, not the foot that the
  parked draft #623 screened under that name (D-D34).
  - What changed: `tyrannosaurus_rex_anatomical.xml` gives `tail_1_pitch`,
    `tail_2_pitch` and `tail_3_pitch` a spring rest angle (`springref`) of
    52.5, 25.9 and 10.4° (the probe's candidate "SR3") and nothing else that
    moves a digest; stiffness, damping, ranges, actuators, masses and geometry
    are r9's, and the env-built compiled model differs from r9's only in
    `qpos_spring` (0.91630, 0.45204 and 0.18151 rad; 595 fields compared,
    measured). The physics digest moves (`3d68aec7…` → `6a47a0eb…`); the policy
    interface stays revision 15 (`ffbf6631…`) and the visual layer revision 4
    (`7042dcd5…`).
  - Why: through r9 the springs rested at zero, so the passive tail could not
    hold the keyframe's level pose against gravity. The zero-action statue
    settled `tail_1` on its −12° ventral stop (30.3 N·m on the limit; `tail_2`
    just above its own, at −11.6°), its tail 23.4° below horizontal from base
    to tip, and a level tail needed commands of
    +3.50, +2.16 and +1.15, beyond the ±1 range (measured, the probe), so every
    stance trained around a drooped tail stuck on its stop, the setting of the
    `tail_3` habit of D-D42 and D-D43. The posture study found that no posture
    reachable on r9 holds a level tail, and that passive support, not more servo
    reach, is the route off the rail (its skeptics' corrected reading). On r10
    the statue's tail settles at the keyframe zeros on its springs (chord
    +0.1°), the servos hold about 0 N·m and no joint touches a limit (measured).
  - The evidence before the decision (CPU only; measured unless marked): the
    r10 statue clean on 40/40 on 3042–3081 at each of the six-heading
    certificate's headings (bound 0.928; r9 39/40) and on 278 of the 280 panel
    seeds (r9 275), its one fall, seed 9047, backward at step 273, and seed
    3121's pad tilted 2.087° (r9 2.061°) against the 2.0° bar; its centre of mass
    10.7 mm further back (the probe). Eight r9 checkpoints rolled zero-shot on r10 all
    stay up (640 of 640 episodes per plant), but their gate-clean episodes fall
    from 293 to 229 of 320 deterministic, two of them pinning a tail actuator at
    +1 and one losing its rail. The checks' skeptic (no blocking finding)
    corrected their reading: the pooled McNemar p-values overreach (at the
    checkpoint level 6 go down and 2 up, sign test p 0.29); zero-shot gives no
    sign that r10 cures the rail (saturation-failure episodes 27 → 80
    deterministic); the backward-fall risk under perturbation is untested at
    useful power; the zero-shot rewards used r9's tail target and are not r10
    values; two zero-shot verdicts are each one episode from flipping (the
    walking-pilot parent passes at 37/40, the band's seed-42 checkpoint at
    4,194,304 steps fails at 36/40); and the walking-pilot parent drops its
    head tip from 1.24 to 0.61 m on r10 and still passes the stance gate, a
    blind spot of that gate. Nothing tested whether a policy trained on r10
    avoids the rail.
  - The reward change D-D44 approved: `tail_home_pose`'s target,
    `TyrannosaurusRexEnv._TAIL_SETTLED_QPOS`, is re-centred on r10's measured
    settled tail, (−0.2107, 0, −0.2029, −0.0926) → (0.00002, 0, 0.00006,
    0.00018) rad (the settled-pose test's helper: noise 0, seed 0, mean of steps
    200–999); its 0.05 rad tolerance stays.
  - Every constant pinned to r9 re-measured on the r10 tree with the procedure
    that set it, each procedure first run on a `git archive` of `3ffc18e9`,
    where it gave r9's own figures (measured; the tail and neck targets, the
    704.4 m·g·h literal and the 1329.1 recovery reference are pins from before
    r9 that r9 carried, and r9 re-measures them within 1.6e-4 rad, 0.02% and
    0.5 standard errors): `_NECK_SETTLED_QPOS` (0.21432, 0,
    0.21963) → (0.21401, 0, 0.22054); `_LEG_SETTLED_QPOS` re-measured over steps
    200–999 (the ankle moves most, 4.9 mrad; its largest deviation from the
    keyframe 2.74 → 2.60°); `natural_pitch` 0.027 → 0.017 (the noise-free
    statue's `forward_z` −0.01666), so the stance's
    `nosedive_termination_threshold` 0.493 → 0.503 keeps the absolute −0.520
    cut; `target_z` 0.9260 carries; the settled statue's lateral m·g·h 704.6 →
    729.1 N·m/rad (the test's literal 704.4 → 729.1; 704.4 at 0.8377 m was
    r8's), so the hip-roll servos' 1280 is 1.76× (1.82×). The statue
    references, each over 40 episodes on 3042–3081: stance 3756.8 → 3766.6
    (3766.56 ± 16.21, 3.8 standard errors), so `min_avg_reward` 2250 → 2260;
    recovery's pushed statue 1329.1 → 1429.1 (1.35 standard errors; its 40 ends
    nosedive 20, fallen 19, `tail_contact` 1, against r9's 26, 0 and 14);
    locomotion 1091.5 → 1092.6 (1.3 standard errors); behavior 602.0 and its
    361 rail carry (600.93 ± 174.59); `statue_constants_physics_revision` 9 → 10
    in all four stage files. The MJCF's settle, pitch and m·g·h comments follow
    (its "CoM 3.7 mm behind the foot-contact centroid at home" is the keyframe's
    geometry and stays), as do the provenance comments in `base_env.py`,
    `stages.toml`'s "no r10 seed yet", and the recovery safe set's re-measure,
    which waits on the first r10 stance (`recovery.toml`,
    `recovery_evaluation.py`). The MJCF's sensor-block comment, which still
    named the `mjx_config.py` deleted at note 18 (deferred there to the next
    T. rex physics revision), names `TyrannosaurusRexEnv._cache_ids`, where
    the sensor indices have lived since (the maintainer's decision of
    2026-10-25; it moves the T. rex source closure only).
  - Every stance bar is kept, with the r10 statue's extremes in its comment in
    `configs/tyrannosaurus_rex/anatomical/stance.toml`: on 3042–3081 sole tilt
    1.60°, corner lift 2.6 mm, pad CoP 0.322, yaw change 14.1°; out of sample
    2.087° (seed 3121), 3.4 mm, 0.404 and 14.8° (seed 5061). The yaw bar's
    comment calls 25° twice the statue's largest, which r10 makes 1.69×; the
    maintainer kept the bars on 2026-10-25, with `healthy_z_range`'s 0.70
    floor and `floor_impact_threshold_bw` 1.4, their comments updated only.
  - Tests and goldens, each re-measured by its own body or fixture: the Phase C
    reset golden's two T. rex captures re-taken (`RECAPTURED` 9 → 10); the
    statue's whole-foot CoP golden (renamed for r10; the left foot now sets the
    metric); the pad-edge fixture's lean −0.0825 → −0.09 (−0.0875 leaves a row
    at 0.806 under its 0.82 assertion, −0.095 trips the corner-lift bar); the
    m·g·h, settled-leg and statue-constant pins. The T. rex suite (161), the
    other species (359) and the shared suite pass, but for the tests the
    no-training guard stops and one: the full-horizon heading trace test's
    T. rex case (rewards 5.08e-9 against its 1e-9 bound), whose own check
    already fails 7 of 20 r9 seeds through float32 rounding; it is fixed by a
    test-only pull request of its own (D-D44 (3): #642, the branch
    `claude/heading-test-float32-bounds`, head `153b297a`), and with that
    branch brought into a scratch copy of r10, not this branch, the heading
    tests pass (24, at `d5dfc167` and at `153b297a`). That test runs only at
    CI's full depth, so the fix should land first or with this revision
    (inferred); on 2026-10-25 the maintainer decided to open this revision's
    pull request with that branch merged in and the `full-ci` label.
  - Digest golden (649 lines): 39 lines move, all T. rex — the plant's
    `physics_revision`, `physics_sha256` and `source_closure_sha256`; the four
    stages' `task_sha256`, `stage_config_view_sha256.PPO` and `.SAC`; the stance
    `gate_sha256`, which moves with the rail (r9's revision moved none); the
    four reward stages' `summary`, `shape_sha256` and `rounded_values_sha256`;
    and the 11 behavior recipes' `task_sha256`. No policy line moves. The plant
    manifests, the T. rex mirror map and the species catalog are regenerated
    with their tools.
  - What it costs: every physics-r9 T. rex checkpoint is refused (Migration);
    no T. rex stance has certified on r9, so no certified trunk is lost, but the
    stance screens' runs and walking pilot 2's parent stop loading on `main`,
    and retraining is estimated at 85–170 GPU-h (inferred, the probe). Nothing
    trains with this change: the first r10 T. rex training round needs its own
    criteria, approved by the maintainer, and is not yet designed.
  - Records: D-D44 in `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2 and the
    consolidation plan's tables; note 20; `docs/KNOWN_ISSUES.md` (Update 20 of
    the T-Rex stance entry, the fore-aft entry's r10 update and the stance
    gate's head-posture blind spot); `docs/NEXT_STEPS.md`; a dated note in §17
    of `docs/GAIT_QUALITY_PLAN_2026_09.md`; `docs/README.md`; the T. rex README
    and website pages; the stance record's §25 and `stance5/tail_sr3/` (106
    files: the posture study, the probe, the CPU checks, the re-measurement,
    the six-heading certificate re-rolled under the final gate block and the
    run log's lines 578 to 603, ids removed; `SOURCES.txt` and `SHA256SUMS`,
    now 634 files, extended). This change also records #641's landing in the
    consolidation plan's status table and puts its number in #641's two
    entries.
- **The standing behavior has one name: the stance stage's recipe label is
  `stance`, and `stand` is refused with a pointer** (#631; decision D-D38,
  the maintainer's decision of 2026-10-18, "make the standing behavior's name
  consistent"). The notebook's `BEHAVIOR` dropdown said `stand` while the
  stage id, its config (`stance.toml`), its run folder (`01_stance`) and its
  gates (`stance_quality…`) said `stance`. The stance entry of all six
  `configs/<species>/<embodiment>/stages.toml` files now says
  `recipe = "stance"`, so the label is the stage's own id on every model;
  stage ids, configs, folders, gates, legacy numbers and edges are unchanged.
  - Before → after, on every model (`resolve_behavior`, and the chain the
    notebook and `train_curriculum` walk): `stand` was the stance stage
    (stance) and is refused; `stance` named the stance stage as a stage id
    and is now its label too, with the same chain (stance); `recover`, `walk`
    and `hunt` are unchanged; recipe labels `stand, [recover,] walk, hunt` →
    `stance, [recover,] walk, hunt`.
  - The loader rule: a recipe label equal to a stage id was refused (since
    manifest v2, 2026-09-06; restated by D-D35); it is now accepted only
    when it resolves to that stage, the deepest deliverable carrying it
    being the node with that id, so `resolve_behavior`'s label-before-id
    precedence still never names two nodes. A label equal to another stage's
    id, equal to its own while a deeper deliverable shares it, or carried
    only by non-deliverables is refused as before.
  - The refusal: `stand` is retired, not aliased, by D-D33's rule that
    retired identifiers do not resolve (`RETIRED_RECIPE_LABELS` and
    `retired_recipe_label_message` in `environments/shared/stage_manifest.py`).
    `resolve_behavior("stand")` raises `<model> has no behavior 'stand':
    'stand' was renamed to 'stance' (decision D-D38); use 'stance'`, so the
    notebook's resolve cell, `curriculum --target stand` (a usage error,
    exit 2), `train_curriculum(target="stand")` (before any directory is
    written) and a species-manifest `deliverable = "stand"` all refuse it, and
    a manifest declaring `recipe = "stand"` is refused at load. Nothing that
    takes a stage id (`RETRAIN_FROM`, `RESUME_STAGE`, `--stage`) ever
    resolved it, and `train_behaviors` takes behavior files whose parent is a
    stage id.
  - The notebook's `BEHAVIOR` dropdown is `stance`, `recover`, `walk`,
    `hunt` (free input kept); its intro, `BEHAVIOR` bullet and configuration
    comment follow. On the T. rex the resolve cell prints
    `Behavior 'stance' -> deliverable 'stance'; chain: stance`. The
    `--target` help names the four labels.
  - The species manifest names the six stance deliverables `stance` (and
    words the two reward-gated definitions "the stance deliverable is
    labelled by its gate kind"). The species catalog is regenerated with its
    tool: `docs/SPECIES_CATALOG.md` (12 lines: six recipe cells
    `stance (deliverable)` and six deliverable lines `stance (1 — Stance)`)
    and `website/src/data/species.generated.json` (18 lines: 6
    `deliverable`, 10 `recipe` and 2 `definition`); the README's generated
    block is unchanged.
  - Tests: `test_stage_manifest.py`'s `TestRecipeLabels` (replacing
    `TestRecipeLabelsAreFourVerbs`) pins that `stance` is the stance stage
    with chain [stance] on every model, that `stand` is refused with the
    exact pointer on every model and on a v1 and a synthetic manifest, each
    model's labels, and that a label equal to a stage id names that stage;
    new loader cases accept a label on its own deliverable id and refuse a
    deeper deliverable sharing it, a non-deliverable carrying it, another
    stage's id and `recipe = "stand"`; `TestRetiredRecipeLabelsAreNamedNowhere`
    checks that no committed configuration value and no notebook code cell
    (string constant or `@param` option) names a retired label.
    `test_cli.py` (`--target stance` the stance stage and `--target stand`
    refused, on all six models), `test_train_base.py` (`target="stance"`
    walks what `1` walks; `target="stand"` raises the pointer and leaves no
    directory), `test_sb3_notebook_pins.py` (the dropdown, and the resolve
    cell executed for the T. rex with `"stance"`, `"recover"` and `"stand"`),
    `test_species_catalog.py` (a `deliverable = "stand"` refused naming
    `stance`) and `test_result_schema_v4.py` follow.
  - Recipe labels enter no training, task, gate, hyperparameter or plant
    digest, and no artifact records one (a bundle names its target by the
    stage key, `"1"` for stance; progress files, ledgers and the offline
    judgements key on stage ids): the digest snapshot (649 lines), the plant
    manifest and the T. rex symmetry map are byte-identical and their checks
    current, no checkpoint is invalidated and nothing retrains.
  - Records: D-D38 in `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2 (with D-A11,
    D-A24 and D-D35 amended, and a dated note on its design text) and in the
    consolidation plan's tables; the recipes guide, the PPO, SAC,
    hyperparameter and custom-model pages, the six model pages, the intro,
    installation and quick-start pages, `species.ts`'s comment, the README's
    two hand-written behavior lines, the Brachiosaurus, Dibothrosuchus and
    Compsognathus READMEs and calibration guide, NEXT_STEPS,
    TRAIN_DIRECTION_AND_TERRAIN and TRAINING_ARTIFACT_PRESENTATION. Dated
    records of sessions that ran with `BEHAVIOR = "stand"` stay as written,
    with a NEXT_STEPS §3 note.
- **The recovery stage has its own recipe label, `recover`, and `stand` is
  the stance stage on every model** (#625; decision D-D35, the maintainer's
  decision of 2026-10-16). A recipe label resolves to the deepest deliverable
  carrying it, and the recovery stage was labelled `stand` too, so `stand`
  meant stance then push recovery on the three models that have a recovery
  stage and stance alone on the other three: someone choosing `stand` in the
  notebook could not tell which, and `stand` and the stage id `stance` read
  as the same thing. The recovery entries of
  `configs/tyrannosaurus_rex/anatomical/stages.toml` and
  `configs/compsognathus_longipes/{anatomical,robot}/stages.toml` now say
  `recipe = "recover"` (a label never equals a stage id, and `recover` is not
  one), so the labels read as four verbs: stand, recover, walk, hunt.
  - Before → after, per model (`resolve_behavior`, and the chain the notebook
    and `train_curriculum` walk). `tyrannosaurus_rex`/`anatomical`,
    `compsognathus_longipes`/`anatomical` and `compsognathus_longipes`/`robot`:
    `stand` was recovery (stance → recovery) and is stance (stance);
    `recover` was unknown and is recovery (stance → recovery); `stance`,
    `recovery`, `walk` and `hunt` are unchanged; recipe labels
    `stand, walk, hunt` → `stand, recover, walk, hunt`.
    `velociraptor_mongoliensis`, `brachiosaurus_altithorax` and
    `dibothrosuchus_elaphros` (`anatomical`): unchanged (`stand` is stance,
    `recover` is unknown, as `recovery` is).
  - Trade-off: on those three models `stand` no longer publishes the
    push-robust standing policy; that deliverable is now `recover`.
    `BEHAVIOR = "stand"` and `"stance"` give the same chain on every model.
  - The notebook's `BEHAVIOR` dropdown is `stand`, `recover`, `walk`, `hunt`
    (free input kept, so a stage id is accepted too); its configuration cell
    and markdown say what each label trains. On the T. rex the resolve cell
    now prints `Behavior 'stand' -> deliverable 'stance'; chain: stance`
    (it printed `-> deliverable 'recovery'; chain: stance -> recovery`).
  - `curriculum --target stand` on those three models is now a stance-only
    run (it was refused, its chain running through the non-advancing
    recovery stage); `--target recover` takes that refusal, naming
    `BEHAVIOR = 'recover'`. The `--target` help names the four labels.
  - The species manifest names every model's deliverables by label: the
    three models' stance and recovery deliverable metrics, addressed by id
    while `stand` meant recovery, are `stand` and `recover`. The species
    catalog is regenerated: the three recovery rows read
    `recover (deliverable)` and their deliverable lines `stand` and
    `recover` (`docs/SPECIES_CATALOG.md`,
    `website/src/data/species.generated.json`; the README is unchanged).
  - Tests: `test_stage_manifest.py`'s `TestRecipeLabelsAreFourVerbs` pins,
    on every committed manifest, that `stand` resolves to stance with the
    same chain as `stance`, that `recover` is stance → recovery on the three
    recovery models and unknown elsewhere, each model's labels, and that no
    label equals a stage id; `test_cli.py`, `test_train_base.py` (a T. rex
    `target="stand"` walks what `"stance"` and `1` walk),
    `test_sb3_notebook_pins.py` (the dropdown, and the resolve cell executed
    for the T. rex), `test_species_catalog.py` and
    `test_compsognathus_longipes_training.py` follow the new labels.
  - Recipe labels enter no training, task, gate, hyperparameter or plant
    digest: the digest snapshot, the plant manifest and the T. rex symmetry
    map are byte-identical, no checkpoint is invalidated and nothing retrains.
  - Records: D-D35 in `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2 (with D-A11 and
    D-A24 amended, and a dated note on its §3/§4 design) and in the
    consolidation plan's tables; the recipes guide, the T. rex, Compsognathus
    and custom-model pages, the PPO, SAC, hyperparameter, installation,
    quick-start and intro pages, the T. rex and Compsognathus READMEs and
    calibration guide, NEXT_STEPS, TRAIN_DIRECTION_AND_TERRAIN and
    TRAINING_ARTIFACT_PRESENTATION.
- **The development version is `0.4.0.dev1`** (#619; `pyproject.toml`; the
  maintainer's request of 2026-10-11 to bump the version when training
  restarts). Training restarts on the D-D33 identifiers. Every run records
  the package version in its provenance (`dependency_versions`), so the runs
  from here on are told apart from the `0.4.0.dev0` runs, which include every
  run recorded under the retired identifiers. No code, config, plant or
  digest changes.
- **Species are named by their binomial, and every model is a species and an
  embodiment** (#618; breaking — identifier change: every species ID,
  package, module, class, Gymnasium ID, MJCF, config path, record field,
  plant key, results path and model page; no alias remains; decision D-D33,
  the maintainer's approval of 2026-10-11).

  The species IDs are `velociraptor_mongoliensis`, `tyrannosaurus_rex`,
  `brachiosaurus_altithorax`, `dibothrosuchus_elaphros` and
  `compsognathus_longipes`. A model is the pair (species, embodiment), with
  embodiment `anatomical` or `robot`: the five anatomical models, and
  `compsognathus_longipes`/`robot`, which was the species
  `compsognathus_robot` (`variant_of = "compsognathus"`). `anatomical`
  replaces "biological" (`CompsognathusBiologicalEnv`, `--model biological`,
  `MODEL_PATHS["biological"]`). In code a model is a `ModelId`
  (`environments/shared/model_id.py`), held in `model_id` variables.

  One rule names everything from the pair:
  - the species base class `<Species>Env` in
    `environments/<species>/base_env.py`, which holds the observation,
    action and sensor code the species' embodiments share;
  - the package `environments.<species>.<embodiment>`, module `env`, class
    `<Species><Embodiment>Env` (`TyrannosaurusRexAnatomicalEnv`,
    `CompsognathusLongipesRobotEnv`), which carries the embodiment's model
    path, foot-support geoms and reward probe;
  - the Gymnasium ID `MesozoicLabs/<Species>-<Embodiment>-v0`, registered
    from the manifest by `environments/__init__.py`;
  - the MJCF
    `environments/<species>/<embodiment>/assets/<species>_<embodiment>.xml`;
  - the configs `configs/<species>/<embodiment>/`, each stage file named by
    its stage with no number (`stance.toml`, `locomotion.toml`,
    `strike.toml`, `snap.toml`, `food_reach.toml`; `recovery.toml` and
    `behavior.toml` keep their names; the `stage1_balance.toml`,
    `stage2_locomotion.toml` and `stage3_*.toml` names are gone);
  - the runs `<LOG_BASE>/<species>/<embodiment>/<algo>/<run_id>/`;
  - the results `results/<species>/<embodiment>/<algo>/`;
  - the model pages `/docs/models/<species>/<embodiment>`.

  Records carry `species` and `embodiment` as two fields, and every
  schema-versioned record bumps its schema:
  - plant identity and plant manifest v2 (`[plants.<species>.<embodiment>]`,
    `plant_versions.toml` schema 2);
  - task fingerprint v3;
  - result summary 5 (readers accept 2 and 5; the four curated schema-2
    summaries carry their identity amended in place) and provenance 2;
  - gate verdict, gate resolution, recovery calibration, recovery evaluation
    and floor-truth measurement v2;
  - the recovery-calibration measurement and PPO-probe outputs
    `schema_version` 2;
  - the stance gate reports v4 and v5 and the heading report v2;
  - the action-filter sweep, constant-hold probe and impulse-recovery probe
    v2, and the constant-hold ablation v4;
  - behavior run and preparation v3;
  - widen lineage and report v2;
  - zero-action baseline v2;
  - the Phase C reset golden v2 (`captures[species][embodiment][stage]`);
  - the Compsognathus tooling labels, renamed from `compso-*`:
    `mesozoic.compsognathus-longipes-preflight/v3`, `-shell-screen/v2`,
    `-model-prototype/v3` (key `anatomical`, was `biological`) and
    `-sensors/v1`;
  - species manifest 3 (`[species.<s>.embodiments.<e>]`, `training_backends`
    required; no `variant_of`, no `aliases`);
  - catalog 5.

  Every identity check compares both fields, so the two Compsognathus plants
  never pass for each other; mypy runs with `strict_equality`. A plant
  identity whose species or embodiment names no model is refused when it is
  read. `backfill_gate_verdict.py`'s `--species` and `--embodiment` name the
  model only for a `stage_config.json` that does not name it whole, and an
  override that disagrees with one that does is refused. The plant
  contract's revision check pairs a baseline's plants by key, then by
  physics digest, and refuses a plant that leaves the manifest.

  The training CLI and every single-model tool take `--species` and
  `--embodiment`, both required, canonical IDs only; a species' own scripts
  (`view_model`, `test_actuators`, `test_env`) take a required
  `--embodiment`. Report tools that cover several models take
  `--model-id S E`, repeated, and keep their default panels.
  `calibrate_recovery.py` moved to `environments/shared/scripts/` and serves
  any model with a recovery stage. The SB3 notebook selects `SPECIES` and
  `EMBODIMENT` from two dropdowns. Adding a robot to a species takes
  `environments/<species>/robot/`, `configs/<species>/robot/`, its manifest
  embodiment table and its plant table, and no edit to shared code.

  Plant contract: policy-interface revision +1 on all six models
  (11 → 12, 14 → 15, 8 → 9, 7 → 8, 4 → 5, 2 → 3;
  `configs/plant_versions.toml` note 19). Only names moved: the hashed
  qualnames (now on the species base classes), the brachiosaurus and
  dibothrosuchus MJX registration module and the compsognathus
  `pelvis_subtree_body_ids` helper (was `robot_body_ids`). Physics and
  visual digests are unchanged, which the PR-base check proves by pairing
  the v1 baseline's plants by physics digest.

  Digest golden (643 → 649 lines): every model line gains the embodiment
  column and six `embodiment` lines are new. After mapping the 643 old rows to
  their models, 232 values move: plant 36 (`species`, `schema`, `model_path`,
  `source_closure_sha256`, `policy_interface_sha256` and
  `policy_interface_revision` of each model), policy 32, stage 30 (21
  `task_sha256`, 9 `config_file`), recovery 2 (the two calibrations'
  `file_sha256`), behavior 132 (every recipe's `recipe_sha256` and
  `task_sha256`). None of the 255 physics, visual, dimension, gate,
  hyperparameter, stage-config-view or reward values moves, and the
  harness's `--exact` mode finds all 37,034 bit-exact digests (every stage
  capture and every behavior recipe's env) equal after mapping.

  Records: D-D33 in `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2, with D-A10's third
  amendment and A5 superseded, and in the consolidation plan's tables;
  `docs/SPECIES_NAMING.md` rewritten; CONTRIBUTING (layout, adding a
  species, adding an embodiment); the README and `results/README.md`; the
  species READMEs and the Compsognathus calibration guide; PLANT_CONTRACT,
  RESULT_BUNDLES, TRAINING_ARTIFACT_PRESENTATION, KNOWN_ISSUES, NEXT_STEPS,
  TRAIN_DIRECTION_AND_TERRAIN, ROADMAP and the hardware README; the docs
  map's note on pre-rename names; both notebooks; the website's model pages
  (moved), sidebar, footer, guides and landing page, and the six model links
  of the 2026-10-11 blog post; the species catalog.
- **The T-Rex and anatomical Compsognathus stances certify under the
  six-heading certificate, and no longer declare the heading probe**
  (#617; D-D32). `configs/trex/stance.toml` and `configs/compsognathus/stance.toml`
  declare `gate_kind = "stance_quality_heading/v1"` with every v2 bar and
  rail unchanged (the comments say what `min_eval_episodes` and
  `min_clean_stance_lcb` mean under it), and drop
  `stance_probe_spawn_yaw_deg`: the certificate rolls every seed at every
  heading, the probe's whole-scene rows were replays of its first 8 seeds at
  four of them, and config load now refuses the key beside the kind. The
  velociraptor and Compsognathus robot stances keep v2 and v1 and the
  warn-only probe; `declares_heading_probe` is false for a heading-kind
  stage, and `stage_heading_probe_warnings` reads its certificate instead of
  a probe: silent when the certificate re-derives to a pass, otherwise one
  warning that it is missing, unreadable, refused or failed, since a manual
  `train(..., task_load_mode="initialize_next_stage")` load reads no
  verdict. Statues (zero action, 40 seeds, through
  `stance_gate_report.py --zero-action`): the T-Rex r9 statue passes, 39/40
  seeds clean at all six headings (bound 0.887, as at 0°); the
  Compsognathus r2 statue passes, 40/40 seeds clean at all six headings
  (bound 0.928, as at 0°); each statue's reward is the same at every heading
  to 1e-10, and the runs took 6 and 10 minutes on one CPU. The species
  manifest's stance deliverables are relabelled "Stance quality at six
  headings (stance_quality_heading/v1)". The `SpawnYaw` docstring now gives
  the right reason a whole-scene turn changes nothing physical (every plant
  declares the elliptic friction cone, D-D29 and D-D30; under the pyramidal
  cone a 45° turn perturbed box and ellipsoid feet), says that it changes
  nothing in the T-Rex r14 or anatomical Compsognathus r4 observation
  (D-D31) and only the world-frame channels of the velociraptor and the
  Compsognathus robot, and names the certificate as the gate; the probe's
  note and text say so too.
  Records: D-D32 in `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2 and the
  consolidation plan's tables, the heading design's status and its
  carried-out note, KNOWN_ISSUES's heading entry, NEXT_STEPS (the
  publication grid and the manual 7042 audit), RESULT_BUNDLES,
  TRAINING_ARTIFACT_PRESENTATION, the T-Rex and Compsognathus READMEs and
  website pages, and the species catalog.
- **The T-Rex and the anatomical Compsognathus no longer observe their world
  heading** (#616; breaking — policy interface change: T-Rex revision 13 → 14, 64 →
  63 wide; anatomical Compsognathus 3 → 4, 56 → 55; every existing checkpoint
  of both is invalidated; decision D-D31, which the maintainer approved on
  2026-10-11 for these two species only, carrying out
  `docs/HEADING_INVARIANCE_DESIGN_2026_10.md`'s observation migration and
  D-D27's 2026-10-07 amendment; the design's heading certificate is a separate
  change, D-D32, above). Motivation: both fed their policies the root's
  world quaternion, world linear velocity and world prey direction, so a
  stance could depend on its world heading where no stance criterion sees it;
  `stance_quality/v2` certified the seed-52 T-Rex stance `20261009_155723`,
  which stands 0/8 with the whole scene turned −90° or −45°. The new
  `environments/shared/heading_observation.py` (pure NumPy) builds the schema
  `bipedal-target-heading-free/v1` from the free root's own quaternion
  `qpos[3:7]`, validated and normalized in float64 (the constructor calls
  `_get_obs` before any forward pass, where `xmat` and the orientation sensor
  still read zero): projected gravity `Rᵀ(0, 0, −1)` replaces the 4-wide world
  quaternion, the root's world linear velocity `qvel[0:3]` is expressed in the
  gravity-aligned heading frame (forward, left, up) and the prey direction in
  pelvis coordinates. The heading frame's forward axis is the horizontal
  projection of pelvis X; below 1e-8, a terminal pose with pelvis X vertical,
  the horizontal projection of pelvis Y turned clockwise 90° stands in, which
  turns with the scene; a quaternion that is not four finite numbers or has
  norm below 1e-12 raises `ObservationStateError`. Joint state, the gyro and
  accelerometer (already in their site frame, not rotated again), the per-foot
  contacts, the prey distance (still measured from `xpos[pelvis]`) and the
  trailing command are unchanged, so the new layout is joint positions and
  velocities, gravity 3, gyro 3, heading velocity 3, accelerometer 3, feet 2,
  prey direction 3, distance 1, command 3. `TRexEnv._get_obs` is rewritten,
  and the anatomical Compsognathus gets its own
  `CompsognathusBiologicalEnv._get_obs`, so `CompsognathusRobotEnv` keeps
  `CompsognathusEnv`'s world-frame observation and every digest. The plant
  contract branches on the schema (`plant_contract/policy_layer.py`; the
  legacy segment table is unchanged): the new segments name each block's
  frame, source and threshold and describe the feet per foot by sensor name,
  their widths must sum to the observation width with the command last, the
  helpers' function tokens are hashed beside `_get_obs`
  (`interface_implementations.heading_free_observation`), and a `jax-mjx`
  declaration on the schema is refused: the frozen MJX builders (D-D17) emit
  only the legacy layout. The T-Rex is therefore SB3-only from r14
  (`supported_training_backends` and the species manifest's
  `training_backends`; its two JAX success metrics leave the manifest and the
  catalog), `environments/trex/mjx_config.py` is deleted with its
  `pyproject.toml` ruff exclusion, and `test_plant_contract_frozen_mjx.py`
  pins the core for the two dual species left, keeping the low-pass filter
  pins the T-Rex interface still hashes. As for every SB3-only species, the
  T-Rex's `_cache_ids` tokens now enter its policy digest
  (`interface_implementations.sb3_sensor_cache`), so a token edit there,
  reward-target setup included, is a policy-interface revision (the
  implementation's reading of the approved SB3-only declaration, for the
  maintainer to confirm). Measured (MuJoCo 3.10.0): a global yaw leaves the
  float32 observation unchanged on 100 lifted, tilted, moving states × 7 turns
  across the wrap per species (0 against the design's 2e-6), `q` and `−q` read
  the same, `R` matches `mju_quat2Mat` to 4.4e-16, and the vertical fallback
  holds to 3.3e-16; one seed turned with its whole scene to −90°, −45°, 0°,
  45°, 90° and 180° repeats its episode — the same float32 actions, joint state
  to 2.9e-13 and rewards to 3.0e-12 per step over 250 steps (zero action and a
  fixed linear policy), and over the full 1000 steps of seeds 3042–3043 behind
  an untrained PPO policy and a frozen VecNormalize the actions again agree
  exactly, joint state to 4.5e-13 and every floor-truth stance metric to
  2e-10; single-threaded torch (as earlier SB3 tests leave it) turns one
  rotated state's rounding into a one-ulp action difference on the
  Compsognathus, which the stable loop carries to 7.3e-8 in joint state and
  5.4e-9 in a metric, so that test bounds them at 1e-6. Statistics gathered
  under the zero action alone made that untrained policy's closed loop
  unstable on the Compsognathus, amplifying a one-ulp action difference at
  step 145 to 0.03 by the end: the full-horizon test gathers them under small
  random actions. Rewards, terminations and dynamics are bit-identical to the
  previous interface at corresponding states: the reset golden's reset,
  second-reset and trajectory halves re-took byte-identical, and the digest
  harness's `--exact` captures of the base and the head differ only in the
  trex and compsognathus `reset` and `obs` streams (586 of 37,034 lines; every
  reward, flag, `info:*`, `qpos` and `qvel` stream is identical). `_get_obs`
  costs about 34 µs on the T-Rex and 30 µs on the Compsognathus, against 13
  and 10 before (1 to 1.5% of a step). `observation_ablation_report.py`'s
  T-Rex slices follow the new layout, and `reset_golden.py`'s recapture
  records name the policy interface and observation schema an observation was
  taken under. `configs/compsognathus/recovery_calibration.json` is restamped
  with `restamp_recovery_calibration.py` and a heading-interface reason, not
  re-measured (physics r2 and the recovery `[env]` block unchanged); the reset
  golden's trex stance and recovery and compsognathus recovery captures are
  re-taken (only the seven observation records move). Tests:
  `test_heading_observation.py` (the frame algebra),
  `test_heading_free_observation.py` (the layout against the plant contract's
  segments, construction, reset, step and terminal observations, no second
  rotation of the local sensors, whole-scene invariance, animal-only turns,
  `q`/`−q`, and a 250-step six-heading paired rollout per species and policy
  in the shared legs, about 5 s per species),
  `test_plant_contract_heading_free.py`, `test_heading_free_checkpoints.py`
  (SB3 job: save, load, evaluate and resume on the new environments, and the
  pre-migration pair refused on every load path) and
  `test_heading_free_rollouts.py` (the full-horizon traces and floor-truth
  metrics, about two minutes, in a new step of the SB3 job that runs at full
  depth only); the stance heading probe's turn tests now assert that a
  whole-scene turn leaves the observation unchanged and an animal-only turn
  moves only the prey bearing. Records: D-D31 in
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2 and the consolidation plan's tables;
  note 18, both plant manifests and the species catalog; the heading design,
  PLANT_CONTRACT, the cleanup plan, CONTRIBUTING, KNOWN_ISSUES, NEXT_STEPS,
  the two species READMEs and website pages. Digest golden 649 → 643 lines,
  every moved line a trex or compsognathus one: plant `observation_dim`,
  `policy_interface_revision` and `policy_interface_sha256` (6); the two
  policy sections' `WHOLE`, `interface_implementations`, `observation`,
  `observation_probe`, `observation_schema`, `observation_segments` and
  `.sb3_observation` (14), the T-Rex's `backend_observation_equal`,
  `jax_interface` and `.home_reset` (3), twelve T-Rex MJX lines removed and
  six lines added (`.heading_free_observation` on both, and the T-Rex's
  `.sb3_sensor_cache`, `.supported_training_backends`, `jax_interface.backend`
  and `.supported`); the stage (8) and behavior (22) `task_sha256`; and
  `recovery compsognathus file_sha256`. Every reward, gate, hyperparameter,
  stage-config-view and recipe line holds, and so does the T-Rex's
  `.action_low_pass_filter`.
- **Every species stands on an elliptic friction cone** (#612; breaking — plant
  change: velociraptor physics revision 3 → 4, brachiosaurus 4 → 5,
  dibothrosuchus 1 → 2; all existing checkpoints of the three species are
  invalidated; decision D-D30, which the maintainer chose on 2026-10-11: one
  friction model for every species, even where the benefit is small).
  `raptor.xml`, `brachiosaurus.xml` and `dibothrosuchus.xml` each gain
  `cone="elliptic"` on `<option>`, with a comment saying why, as the T-Rex
  did at r9 (D-D29) and both compsognathus plants were built; the
  velociraptor and dibothrosuchus settle comments are the only other MJCF
  edits, so no policy interface or visual layer moves. Motivation: a box or
  ellipsoid on the floor keeps its contact tangent frame on the world axes
  whichever way the animal faces, and MuJoCo's default pyramidal cone is
  weaker between those axes. Through the heading probe's path (`run_panel`
  with a whole-scene `SpawnYaw`, zero action, seeds 3042–3049) the statue's
  ±45° rewards differed from 0° by up to 0.84 on brachiosaurus r4 (ellipsoid
  feet, the largest of any species), 3.7e-2 on dibothrosuchus r1 (box pads)
  and 1.8e-12 on velociraptor r3, whose capsule feet turn their frames with
  the foot; on the new plants they agree to 1.4e-12. From the home keyframe
  under fixed random commands a 45° turn separated the joints by 7.9e-3,
  8.7e-2 and 5.4e-3 rad (the velociraptor once it fell onto its ellipsoid
  head), and on the new plants by at most 4e-14. Only the cone moves: on
  MuJoCo 3.10.0 the other solver options already equal the compsognathus's,
  every geom is condim 3, and Newton converges in at most 6 iterations per
  substep on each statue and 10 under N(0, 0.22) commands (seeds
  3042–3044), never at the cap, with no warning. Statues re-measured (zero
  action, reset noise 0.05): the velociraptor's on `stance_quality/v2` is
  clean on 40/40 of the certification seeds 3042–3081 and on 238/240 out of
  sample (3082–3201, 5042–5081, 7042–7081, 9042–9081; r3 239/240): seed
  9065 falls on both plants, and seed 5060's lighter foot carries 0.342 of
  one 1 s block against the 0.35 `min_foot_load_share_windowed` bar (r3
  0.432), the one bar a full-horizon r4 statue episode fails. Its feet
  share their load a little less evenly and rest digit IV a little less
  steadily (full-horizon episodes, certification / out of sample: windowed
  share 0.412 / 0.342 against r3's 0.457 / 0.425, support-geom duty 0.964 /
  0.869 against 1.000 / 0.970). Every bar is kept, and each comment in
  `configs/velociraptor/stage1_balance.toml` records the r4 extremes; its
  statue reference carries (2843.00 ± 2.70 against 2842.76 ± 2.62, inside
  one standard error) with `statue_constants_physics_revision = 4`, and
  `natural_pitch` carries (the statue settles at 0.3494 rad, r3 0.3508).
  Brachiosaurus's stance statue reads 1738.38 ± 1.46 (1739.08 ± 1.17; −0.7
  is about three standard errors, so the figure moves rather than carries),
  dibothrosuchus's 2598.53 ± 0.87 (2598.29 ± 0.86), both 40/40 and 80/80 on
  3082–3161, and their 0.60× and 0.75× rails still round to 1040 / 1300 and
  1560 / 1950; the configs carry the new figures. The brachiosaurus
  locomotion statue reads 2201.3 ± 200.6 (2242.7 ± 7.8; median 2244.9): two
  of 40 episodes lose part of `gait_symmetry` to a briefly unloaded foot.
  The dibothrosuchus pads stand nearer their heel edge (fore-aft CoP at most
  0.795 on 3042–3081, r1 0.706), and its trunk settles 0.3 mm higher, above
  `target_z`, whose clipped term pays the statue in full either way. Tests:
  the new `test_friction_cone.py` guards that every registered plant
  declares the elliptic cone and that a 45° whole-scene turn under random
  commands reproduces the joints to 1e-10 on the three species;
  `test_gait_recorder.py` keeps the recorder's pyramidal decode covered on
  dibothrosuchus with the cone switched in memory, since no plant declares
  it now, and its dibothrosuchus statue CoP ceiling (0.72 → 0.78) and
  spawn-pop count (9 → 8 substeps) follow r2; the catalog and velociraptor
  rail tests take the new revisions, and the two
  `test_plant_contract_layers.py` cases that edit the raptor's and the
  brachiosaurus's `<option>` match its new text and assert the edit applied
  (the brachiosaurus one had become a silent no-op). MJX: nothing in the
  repo compiles a model under MJX (D-D17 retired the runtime; the plant
  contract's MJX probe runs the frozen observation builder in NumPy, and
  passes), and MuJoCo's MJX feature-parity table lists both cones, so the
  brachiosaurus and dibothrosuchus `jax-mjx` declarations stand. The reset
  golden holds no capture of the three species, so nothing is re-taken.
  Records: D-D30 in `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2 and the
  consolidation plan's tables; note 17, both plant manifests and the species
  catalog; KNOWN_ISSUES, NEXT_STEPS, the heading design's cone paragraph,
  the three species READMEs and website pages. Digest golden (649 lines): 79
  lines move, all of the three species — plant `physics_revision`,
  `physics_sha256` and `source_closure_sha256` (9); stage `task_sha256` of
  stance, locomotion and behavior (9) and the velociraptor stance's
  `stage_config_view_sha256.PPO` and `.SAC` (2, with the pin); the 33
  behavior recipes' `task_sha256`; and `summary`, `shape_sha256` and
  `rounded_values_sha256` of the nine reward stages (26: the brachiosaurus
  stance's shape holds). No `gate_sha256` and every other line hold.
- **The T-Rex stands on an elliptic friction cone, so every world heading is
  physically identical** (#611; breaking — plant change, physics revision 8 → 9;
  all existing T-Rex checkpoints are invalidated; decision D-D29, which the
  maintainer chose on 2026-10-11 with the heading decisions of
  `docs/HEADING_INVARIANCE_DESIGN_2026_10.md`). `trex.xml`'s `<option>`
  gains `cone="elliptic"`, with a comment saying why; its settle comment's
  pelvis height is the only other edit, so the policy interface stays
  revision 13 (`0b43de…`) and the visual layer revision 4 (`7042dc…`).
  Motivation: the plantar pads are boxes, whose contact tangent frames stay
  on the world axes whichever way the animal faces, and MuJoCo's default
  pyramidal cone is weaker between those axes, so a ±45° whole-scene turn
  was a small physical perturbation as well as an observation change, and a
  heading certificate could not treat T-Rex's ±45° cells as replicates of
  0°. Through the heading probe's path (`run_panel` with a whole-scene
  `SpawnYaw`, zero action, seeds 3042–3049) the statue's ±45° rewards
  differed from 0° by up to 2.6e-2 on r8 and agree to 3e-11 on r9, as at
  ±90° and 180°; from the home keyframe under fixed random commands, a 45°
  turn separated the joints by 7.7e-2 rad on r8 and by 4.8e-14 on r9. Only
  the cone moves: on MuJoCo 3.10.0 the other solver options already equal
  the anatomical Compsognathus's, and Newton converges in at most 10
  iterations per substep, never at the cap, with no warning; `impratio = 10`
  was measured and left out (the statue clean on 25/40, 4 nosedives). The
  noise-free settle carries (pelvis 0.92616 m against 0.92601, `forward_z`
  −0.02714 against −0.02778, the settled neck and tail within 1.6e-4 rad),
  so `target_z`, `natural_pitch`, `nosedive_termination_threshold` and the
  settled-pose targets keep their pins. The stance statue is not neutral to
  the change. On the certification seeds 3042–3081 it is clean on 39/40
  (seed 3081 nosedives at step 292; r8 40/40), which uses the whole
  `max_hop_or_fall_episodes = 1` allowance, and on 236/240 more seeds (r8
  238/240): 4 falls in 280 against r8's 2 (Fisher exact p = 0.69), a worst
  sole tilt of 2.061° on 1 of the 240 against the 2.0° bar, and median
  loaded sole contacts down from about 3.8 to about 2.6 (least 1.956
  against the 1.5 bar). Every stance bar is kept, as the heading design
  requires, and each comment in `configs/trex/stance.toml` records the r9
  extremes; the appended §12 of
  `docs/investigations/STANCE_HACK_AUDIT_2026_10.md` has the panels.
  Statue constants re-measured on r9 with
  `statue_constants_physics_revision = 9` in all four files, each move
  inside one standard error, so every reference and rail carries: stance
  3755.72 ± 19.93 over the 39 standing episodes (3756.8; `min_avg_reward`
  stays 2250), recovery's pushed statue 1369.4 ± 519.6 (1329.1), locomotion
  1092.14 ± 5.74 standing (1091.5) and behavior 593.41 ± 187.19 (602.0,
  rail 361); the same commands reproduce every r8 figure exactly. Tests:
  `test_trex_env.py` pins the cone and that a 45° whole-scene turn under
  fixed random commands reproduces the joints to 1e-10 (raw MuJoCo, under
  a second); `test_gait_recorder.py` covers the pyramidal decode with
  dibothrosuchus (box feet) instead of the trex, asserts each cone species'
  cone, and lifts its trex statue CoP ceiling 0.40 → 0.47 (the r9 statue
  reads 0.441 on its episode); the pad-edge case of
  `test_stance_gate_v2_report.py` leans the left ankle to −0.0825 (−0.075
  left one of its three r9 episodes clean) and reads its rows above 0.82
  (was 0.85), each still refused by the CoP bar alone. The recorder's
  docstring lists the trex with the elliptic species. Records: D-D29 in
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2 and the consolidation plan's tables;
  note 16, both plant manifests and the species catalog; the phase-C reset
  golden re-takes its two trex captures, whose reset halves are unchanged;
  KNOWN_ISSUES's heading and fore-aft entries; NEXT_STEPS; the heading
  design's cone paragraph, now carried out, and the rows of its change
  table that named the r8 anisotropy.
  Digest golden (649 lines): 38 lines move, all trex — plant
  `physics_revision`, `physics_sha256` and `source_closure_sha256`; stage
  `task_sha256`, `stage_config_view_sha256.PPO` and
  `stage_config_view_sha256.SAC` of stance, recovery, locomotion and
  behavior (the views move with the pin); the 11 behavior recipes'
  `task_sha256`; and `summary`, `shape_sha256` and `rounded_values_sha256`
  of the four trex reward stages. No `gate_sha256` and every other line
  hold.
- **The release ablation classifies with one-sided Fisher exact tests, not
  fixed 8/8 and 0/8 shares** (#609; the maintainer-approved follow-up to #606;
  diagnostic only: no training, gate, plant, policy-interface or digest
  change). #606 read a variant as standing at a full-horizon share ≥ 0.95 and
  as falling at ≤ 0.05, so at the pipeline's 8 episodes only 8/8 and 0/8
  counted, and a `hold_all` that stood 7/8 still let "contributes" rest on
  8/8 against 7/8, a one-sided p of 0.5. `reporting/stance_report.py` now
  tests each group's `release_G` against `hold_all` (stands more often) and
  `hold_zero` (less often), and its `only_G` against `hold_zero` (less) and
  `hold_all` (more), each one-sided at p ≤ 0.05 inclusive, in exact rationals
  from `math.comb` (no runtime scipy; the tests check it against
  `scipy.stats.fisher_exact` on every table up to 10 episodes per arm). A
  variant stands, falls, sits between the endpoints, is separated from
  neither, has too few episodes, or is unmeasured; the verdict labels are
  unchanged, and each now rests on one significant test per variant.
  `hold_zero` keeps its fixed 0.95 statue check, and `hold_all` must reach
  the horizon significantly less often than it, so 5/8 to 7/8 against an 8/8
  statue is now not applicable (p = 0.10, 0.23, 0.50), where #606 classified
  it. Counts are k/n, recovered from each report's share and `episodes` and
  checked against its `episode_evidence`: a share that is not a whole number
  of episodes is invalid, and a comparison whose best reachable p,
  1/C(n₁+n₂, n₁), exceeds 0.05 reads "insufficient episodes" (fewer than 3
  per arm at equal sizes). The p-values are not adjusted across groups, and
  the variants share reset seeds, a pairing the test ignores, which errs
  conservative when a seed's outcomes move together across variants; the
  report says both. With clean endpoints the test also labels partial
  effects the thresholds left inconclusive: a release that stands 5/8
  against a 0/8 full pose is now necessary (p = 0.0128). The per-group
  table prints each variant's k/n beside its four p-values (`*` p ≤ 0.05,
  `n/a` too few episodes, `-` unmeasured or unassessed). The JSON is
  `mesozoic.constant-hold-ablation/v3`: a `statistics` block (`fisher-exact`,
  one-sided, alpha 0.05, multiplicity `none`, statue share 0.95) replaces
  `stands_threshold` and `falls_threshold`, `classification.endpoints`
  records the endpoint counts and their separation, each variant row adds
  `episodes` and `full_horizon_episodes`, and each verdict its `necessity`
  and `sufficiency` comparisons and position. On `20261006_185704`
  knees_ankles keeps its CAUSE (p = 1/78 and 1/12870), tail, head_neck and
  hip_rolls become bystanders, and left_leg and right_leg stay inconclusive
  on a 5/8 variant; on `20260805_011234`, the bounce investigation's passing
  checkpoint, `only_hip_rolls` at 7/8 makes hip_rolls a bystander and the
  report reads OVER-DETERMINED (saturated, toes) again. The help for
  `stance_gate_report.py --hold-release-ablation` and `--hold-episodes` says
  so. (This supersedes the `hold_all` rule and the v2 schema of #606's
  "Suppress unsupported held-pose ablation classifications" entry under
  Fixed, shipped earlier in this release.)
- **Training output names species and the stance stage in full, and every
  species' stance and recovery stages write the same replays** (#607;
  presentation only: no plant, policy-interface, task, gate or
  hyperparameter digest moves; decision D-A10's 2026-10-10 amendment,
  approved by the maintainer on 2026-10-11;
  `docs/TRAINING_ARTIFACT_PRESENTATION.md`). The six stance configs record
  `name = "stance"` (was `balance`), and `species_display_name()` /
  `stage_display_name()` label CLI logs, text summaries, gate and probe
  reports, plot legends, notebook tables, the species catalog and W&B run
  titles (`Tyrannosaurus Rex / Stance`, was `trex-stage1`); run directories,
  replay file IDs, semantic stage IDs and W&B resume IDs and tags are
  unchanged. Twelve `stage_config_view_sha256` lines of
  `configs/digest_snapshot.generated.txt` move (PPO and SAC for the six
  stance nodes); its other 637 lines are equal. Historical results recorded
  as `balance` show as `1 — Stance` in the RESULTS block and on the website,
  with their numbers and verdicts unchanged. Stance and recovery stages of
  every species (was T. rex only) add side and front views and a per-frame
  `*_stance.csv` to their selected and final replays, and every stage replay
  plays at `1 / env.dt`: 100 fps for the 0.01 s species and 50 fps for both
  Compsognathus (was 50 fps everywhere, so 0.01 s replays played at half
  speed); the W&B callback's evaluation video also plays in real time (was
  30 fps), keeping every n-th frame above 50 fps because a GIF cannot play
  faster. A final checkpoint without its `_vecnorm.pkl` is skipped rather than
  replayed on raw observations. The CSV measures each biped at the support
  registry's `foot_site` and the leg joints spelled as that site spells its
  side, so every biped's CSV has foot positions, the support midpoint, the
  pelvis-over-support offset, leg home errors and `stance_width` (the foot
  sites' `|Δy|`, as on T. rex). The T. rex CSV keeps every historical column
  with unchanged values and adds `time_s`, `reward`, `terminated`,
  `truncated`, the `root_*` pose and the rest of the env's scalar info.
- **The anatomical Compsognathus stands on its whole foot, PPO can explore
  around its statue, and its stance is judged on floor truth** (#604; breaking —
  plant change, physics revision 1 → 2, policy interface revision 2 → 3,
  visual revision 1 → 2; all existing anatomical Compsognathus checkpoints are
  invalidated; the robot is unchanged; decision D-D26, which the maintainer
  chose on 2026-10-07). Both certified r1 stances, `20260921_203149` and
  `20261001_225856`, are noise-robust tiptoe marches on the digit-III tip (the
  stance-hack audit's §3.3). The compsognathus standing review traced them to
  an action map that put the statue's whole holding torque inside 1% of the
  action and to two plant defects that made the flat stance fragile, and found
  the right foot built as a left foot; the implementation found the MTP servo
  ringing numerically. The five fixes land in one revision so the species'
  digests move once (`configs/plant_versions.toml` note 15). In
  `build_models.py`'s `build_biological` and `data/model_parameters.json`: the
  tail's 178 g is spread at one density over its tapered meshes,
  tail_1..tail_4 0.080 / 0.055 / 0.028 / 0.015 → 0.1173 / 0.0461 / 0.0125 /
  0.0021 kg (583 / 1019 / 1918 / 6223 → 854–871 kg/m³; the r1 tip was denser
  than bone), body mass still 1.000 kg, which moves the home centre of mass
  9.8 mm forward (3.6 → 13.4 mm ahead of the hip), the settled statue's from
  19.6 to 29.3 mm ahead of the plantar pad's rear edge (0.25 → 0.38 of the 78
  mm foot), and halves the tail's pitch inertia about tail_1 (0.00825 →
  0.00434 kg·m²); the metatarsus collision capsule, also the drawn shank, ends
  4 mm short of the MTP joint along its own axis
  (`metatarsus_collision_inset_m`), because the metatarsal head rests on the
  pad and is never support, yet the settled r1 statue held it 1.07 mm off the
  floor, so a 1 mm soft-contact sink of the sole touched it with 0 N on it (48
  of 48 first touches) and the terminating contact ended upright, flat-footed
  episodes as falls (settled clearance now 4.81 mm; contact still terminates);
  digit II is medial on both feet (r1's right digit-II tip sat laterally at
  world y −64 mm), which is mechanically neutral but makes the home preload
  left-right symmetric; and the MTP joint's armature rises from 5e-5, the
  default of the generator's joint class, to 2e-4 kg·m² (`toe_armature_kg_m2`,
  taken by the maintainer on 2026-10-07). At 5e-5 the toe servo (kp 12 on a
  DOF of 6.6e-5 kg·m², three quarters of it armature) rang at 68 Hz, ω·dt 0.85
  at the 2 ms step, and on the other r2 changes sustained a numerical 19–21 Hz
  heel chatter after the settle on 15 of 40 zero-action statues (r1: 6 of 40,
  at 18.5 Hz), which no touchdown or support bar sees (the pad unloads on
  about 4% of substeps while the foot stays loaded) but which failed 2 of
  `validate_models`' 10 seeded holds (`supports_weight`, a 5.4% ground-force
  error against the 3% bar) and with them `test_balance.py`'s seeds 49 and 51.
  At 2e-4 it rings at 37 Hz: no statue of 40 chatters, every hold passes (the
  worst settled ground-force error of the ten is 0.31%), the regenerated
  `data/preflight_training_v1.json` records `passed: true`, and
  `test_balance.py` runs every seed unmarked; statics, keyframe and preload
  are unchanged, and the armature moves the physics digest alone.
  `compsognathus.xml` is regenerated (`fc6ef368…` → `e47a3c50…`) with the same
  key qpos and root height, so `nominal_height` stays, and a regenerated
  gravity preload in its home ctrl; `compsognathus_robot.xml` comes out
  byte-identical. The policy interface: the species' env is the new
  `CompsognathusBiologicalEnv(CompsognathusEnv)` (the species registry,
  `configs/species_manifest.toml`'s `env_entrypoint` and
  `MesozoicLabs/Compsognathus-v0`), whose action mapping, the new
  plant-contract mode `home-keyframe-residual-softcubic/v1`, shapes the ten
  leg residuals (hip pitch, hip roll, knee, ankle and toe, found by actuator
  name) to b = 0.1a + 0.9a³ before the unchanged home-keyframe span: action
  zero is still the home ctrl, ±1 still the ctrlrange ends and the map stays
  monotone, with a tenth of the linear slope at home. Every leg servo
  saturates at forcerange / kp = 0.05 rad of position error, and the linear
  residual spent 0.007–0.015 of action on the whole knee or ankle holding
  torque, so under PPO's initial exploration noise (σ 0.135, `log_std_init`
  −2.0) the r1 statue fell on 20 of 20 episodes in a median of 12 steps and
  PPO never sampled the quiet stance. Neck, jaw and tail keep the linear
  residual. The plant contract records the shaping (`residual_shaping`: form
  `soft-cubic/v1`, the per-actuator slope and mask) and refuses a slope
  outside (0, 1], a mask that is not one bool per actuator or selects nothing,
  and a JAX/MJX declaration; the `home-keyframe-residual/v1` payload is built
  as before, so no other species' digest moves. The subclass builds its leg
  mask in `__init__`, so the policy digest still hashes
  `CompsognathusEnv._cache_ids`. `CompsognathusEnv` and
  `CompsognathusRobotEnv` are unchanged.

  The MTP servo keeps its 0.6 N·m cap (the maintainer first chose 0.4 N·m
  and revised it the same day), now the generator parameter
  `toe_forcerange_nm`, so a later cut is one number and leaves this XML
  unchanged. The cap holds one body weight on the digit-III tip (0.594 N·m),
  so the single-support tiptoe stays physically possible on r2: the
  `20260921_203149` march, replayed on the r2 plant through its own linear
  interface, still survives 20 of 20 episodes on that tip (seeds 3042–3061;
  `20261001_225856` survives 9 of 20), and the stance reward and gate are
  what refuse it. A cut waits on a measured walker toe-torque budget after
  training. The proportions fidelity pass is deferred too: an anatomically
  long tail at the same density and mass would give back about 43% of the
  centre-of-mass gain.

  Zero-action statue on the stance task (`frame_skip` 10), r1 → r2: settled
  metatarsus clearance 1.07 → 4.81 mm; pad / distal-digit share of each
  foot's load 0.44 / 0.14 → 0.41 / 0.25; the largest 0.1 s pelvis push held
  on 5 of 5 seeds forward 0.225 → 0.275, backward 0.125 → 0.20 and lateral
  0.45 → 0.45 body weights; the largest initial tilt held on 3 of 3 seeds
  nose-down 2 → 5°, nose-up 4 → 7° and roll 15 → 15° (every fore-aft push or
  pitch failure ends on tail_4); zero-mean white action noise through each
  revision's own interface (paired streams, seeds 3042 + i): σ 0.02 5/20 →
  20/20 full episodes, σ 0.135 0/20 (median 12 steps, 16 of 20 on the
  metatarsus) → 39/40 (151/160 over four seed blocks, one with an
  independent noise stream; all nine falls end on tail_4, at steps
  174–938), and on r2 σ 0.10 40/40 and σ 0.20 0/40 (median 130–180 steps
  over two noise streams). Without the toe armature σ 0.135 kept 30/40. The
  cliff between the recipe's initial σ and 0.20 is a constraint on
  `log_std_init` and on any entropy schedule (the stance TOML's
  `log_std_init` comment).

  The stance task (`configs/compsognathus/stance.toml` `[env]`) takes ten
  reward terms through twenty `CompsognathusBiologicalEnv` kwargs, every one
  inert at its default (there the reward, its legacy components and the
  observations are bitwise equal to `CompsognathusEnv`'s); the weights,
  tolerances and saturation force are named as on `TRexEnv`, and so, since
  D-D27, are the stance-width reference, the per-foot load gate
  `foot_terms_min_support_force` and the floor-impact and airborne terms,
  with this species' own forms (the load gate reads the aggregation below,
  not the trex's substep-minimum touch; the impact term is capped,
  min(peak − 2.5, 1), at a 2.5 BW threshold); `support_force_aggregation`
  and the support-geom coverage are this species' own:
  `support_force_aggregation = "mean"`, which feeds only the new terms the
  substep-mean foot load (the legacy alive, posture and height terms keep
  their support gate on the substep minimum, which prices a two-foot hop; a
  one-foot unload leaves the other foot's minimum above that gate, so a
  one-foot chatter is priced only by the bilateral and coverage terms, about
  3% of the statue's reward at 130–190 lift-offs an episode);
  `bilateral_support_weight = 0.5` at `foot_contact_saturation_force = 4.4` N
  (0.45 body weight) and `support_conditioned_alive_fraction = 0.2`;
  `foot_flatness_weight = 0.5` at 2.0° and `stance_width_weight = 0.1` at 0.03
  m, each paid only on a foot carrying `foot_terms_min_support_force = 2.45` N
  (the trex r8 terms pay an airborne foot about 80% of the flatness term and
  98% of the width term), the width around the animal's own width at the end
  of the settle (`stance_width_reference = "settled"`,
  `stance_width_settle_steps = 200`), not the keyframe's, which paid both trex
  r8 seeds to re-seat their feet in the unscored settle; `leg_home_pose_weight
  = 0.5` at 0.15 rad; `support_geom_coverage_weight = 0.2`, each foot's share
  of its four support geoms loaded on at least half of the step's substeps, a
  geom loaded on a substep when its floor contacts' normal forces sum above
  0.1 N (the gate's rule), and a digit summing only its contacts beyond the
  capsule's centre, away from the MTP joint (this term's own rule: the capsule
  starts at that joint, under the pad, so its joint end carries load whether
  or not the tip reaches the floor); `floor_impact_weight = 0.25` above 2.5 BW
  and `airborne_substep_weight = 0.2`, on every substep from the first step;
  and `action_saturation_weight = 0.3` at 0.9 and `action_jerk_weight = 0.5`
  on the raw command (this species has no action filter). The new reward key
  `reward_support_geom_coverage` joins the diagnostics and their graphs
  (`reward_floor_impact` and `reward_airborne_substeps` joined them with
  D-D27). Each
  weight was measured by re-scoring recorded rollouts (exact: equal to the
  env's own total to 5e-13 per episode): the statue stays the reward optimum,
  4570.4 ± 7.2 on seeds 3042–3081 (4565.8 on 7042–7081), 1571.3 of the 1580
  the terms can pay, short mainly on coverage, because the settle seats the
  digit tips last (coverage is 8.1 of the 8.7 per episode it leaves on
  average; 25.9 at most in one episode), and no near-statue wins it back
  (constant toe offsets of ±0.01, ±0.02 and −0.05 rad end 2.5–82 below it on
  average); the two certified r1 marches, rolled on the r1 plant through their
  own interface, score 0.612 / 0.610 of that plant's statue deterministically
  (0.936 / 0.934 under the old reward) and 0.538 / 0.559 stochastically, their
  best episode 1694 below it. On a 20-episode re-score (seeds 3042–3061)
  rolled feet keep 0.852–0.892, legs swept back 17° 0.80–0.89 (a level, bilateral
  crouch that only the leg pose term prices), a whole-episode two-foot hop
  0.521–0.837, the same hop started after the settle 0.649–0.911 and a settle
  stomp 0.957–0.989 (50–198 below the statue). The noisy statue keeps 0.934 of
  the statue at σ 0.05, 0.824 at σ 0.10 and 0.710 at σ 0.135 (the old reward
  alone keeps 0.841 and 0.741 at the last two, so the terms cost it 1.9% and
  4.2% more), and the same noise around every non-statue posture measured
  falls on 19–20 of 20 episodes (median 23–440 steps over two noise streams)
  and keeps at most about 0.25, so the noise cost cannot pull PPO's mean off
  the statue. At initialisation (σ 0.135) the penalties cost 0.169 per step, a
  sixth of the 1.0 alive bonus.

  The stance gate moves from `stance_quality/v1`, which both marches cleared,
  to the floor-truth `stance_quality/v2` (D-D23), the third stage to adopt it,
  with every bar set on the r2 statue's 40-episode panel, toe armature
  included, through the report path: the clean-episode bound ≥ 0.80 at
  `settle_steps` 200, every required bar (`min_all_feet_support` 0.98,
  `max_touchdown_rate` 0.25, `max_window_displacement_m` 0.05,
  `min_foot_load_share` 0.40, saturation 0.10, settle airborne substeps 15 and
  settle peak 3.0 BW), the pad bars (`max_sole_tilt_deg` 2.0,
  `max_sole_corner_lift_m` 0.003, `min_sole_contacts` 1.5), the support-geom
  bars (`min_support_geom_duty` 0.50, `min_support_geom_coverage` 0.80: the
  pad and digits II–IV), the settle width change ≤ 0.011 m (about three times
  the statue's worst; the trex r8 runs re-seated their feet by 3.6–4.4 times
  theirs), the window hop pair below (40 airborne substeps, 2.0 BW), the load
  share ≥ 0.80 of the statue's, four guards and the rails (full horizon ≥ 95%,
  `min_avg_reward` 2740, 0.60 × the statue's 4570.4 rounded to the nearest 10,
  and ≥ 0.60 of the statue panel each report rolls). The settle bars are
  looser than the trex's and the velociraptor's because noise bounces this
  light plant (the statue with σ 0.05 command jitter reaches 10 airborne
  substeps and 2.12 BW in the settle), and the reward prices both from the
  first step. `stance_quality/v2` gains two optional per-episode criteria for
  it, each applied only where declared and needing no statue panel
  (`stance_gate_v2.py`, `gait/stance_metrics.py`, `manager.StageThreshold`;
  `GATE_SCHEMA_VERSION` stays 1): `max_window_airborne_substeps` and
  `max_window_peak_floor_force_bw`, the settle window's own hop and impact
  detectors read over the window (`window_airborne_substeps`,
  `window_peak_floor_force_bw`). A leg is down on a step when it is loaded on
  half of the step's ten substeps, and this plant's hop flights last 1–3 of
  them, so a two-foot hop started after the settle reads a statue on every
  step-level bar: both legs down on every step, no touchdown, a quiet settle.
  `from_curriculum`, and `gate_schema` at config load, refuse a negative
  airborne bar and a peak bar under one body weight (`WINDOW_HOP_KEY_FLOORS`,
  beside D-D28's `CRITERION_BAR_RANGES`), and the pair joins D-D28's pad
  centre of pressure in `STANCE_METRIC_LATER_FIELDS`: a report row or panel CSV
  recorded before it reads both metrics as unmeasured, so every earlier
  report keeps its verdict under a block that does not declare the pair, and
  under one that does such a panel is refused by name, never failed (re-roll
  the panel); no trex or velociraptor digest moves, and the website's species
  data and the catalog carry the two keys. The pair counts toward D-D27's
  hop-or-fall rail where a block declares both (none does yet); the rail's
  failure text and the text report name only the hop keys a block declares,
  and a block that does not gate the pair (the T-Rex's, the velociraptor's)
  lists its readings among the report-only ones.
  Validated, verdicts re-derived from the recorded
  rows under the committed block: the r2 statue is clean on 40/40 (bound
  0.928) on each of seeds 3042–3081, 7042–7081 and 9042–9081; the statue with
  N(0, σ) command jitter on 20/20 at σ 0.005 and 0.01, on 40/40 at σ 0.02 and
  on 120/120 over three seed blocks at σ 0.03, while at σ 0.05 the window peak
  bar refuses 64 of 120 (that much noise lands this plant as hard as the
  softest square-wave hop, and the bar refuses the hop rather than admit the
  noise); spawned yawed ±45° on 10/10 and ±90° on 20/20; two 200k-step PPO
  runs on this `[env]` (seeds 11 and 12, `log_std_init` −2.0) on 40/40 each
  with their deterministic means (statues: 4569.5 and 4569.2, no airborne
  substep), which shows only that the mean does not drift to a march or a hop
  in the first 200k steps, while their stochastic rollouts reach the horizon
  on 20/20 and are clean on 0/20 (the exploration noise bounces this plant;
  the gate reads the mean); the two certified r1 marches, rolled on the r1
  plant they load on with their own interface, on 0/20 each, every episode
  failing 9–11 bars (support, touchdowns, support-geom duty and coverage, sole
  tilt, corner lift, contacts and pad centre of pressure, the window peak),
  and the r1 statue there on 20/20. Scripted hacks on r2 (seeds 3042 + i,
  10-episode gate panels): rolled feet 0/40, settle stomps 0/30, the
  whole-episode hop 0/40, an open-loop one-leg stance 0/10 and a 39/61 weight
  shift 0/10, and the hop started after the settle (knees, ankles and hip
  pitch in phase, square waves at 10 and 12.5 Hz, amplitude 0.07–0.15) 0/410
  over seeds 3042 + i, 7042 + i and 9042 + i, refused by the window peak bar
  on all 410, the window airborne bar on 368 and the displacement guard on
  349, and by the pair alone on 61 of the 120 episodes at amplitude 0.07 and
  10 Hz (and on 10 of 10 of a hop started at step 600); a swept-back crouch, a
  level, bilateral stance on every support geom that the leg pose term prices,
  stands on its pads' front edges and is refused by the pad centre-of-pressure
  bar alone (below; 8/10 clean without it, its two unclean episodes walking
  off). The pair narrows the gap the step-level bars leave on this plant but
  does not close it (40-episode panels through the report path, seeds
  3042–3081): the same hop driven by a sine at 5 Hz and 2.5 Hz lands at
  1.33–1.80 BW, flies 0–32 window substeps and certifies on 40/40 at
  0.989–0.990 of the statue's reward, one 10 Hz cycle every 0.4 s on 39/40,
  and, because a foot lifted for less than half a step reads down, a one-leg
  pump that lifts the foot for one substep 130–190 times an episode and a 6.25
  Hz weight shuttle on 40/40 each (0.905–0.972; a toe tap certified too until
  the pad centre-of-pressure bar, below). No count bar separates these from
  the admitted σ 0.03 jitter, so KNOWN_ISSUES carries them, a stance's
  lift-offs are checked by hand until a periodicity criterion is measured, and
  the window peak bar's jitter margin is 0.12 BW, not the middle of the gap
  (the σ 0.03 jitter reaches 1.88 BW on an independent noise stream). The
  committed stance was put through the post-stage pipeline with the zero
  command as a scripted checkpoint: the report and the judge pass it on 40/40
  (bound 0.928) and write `gate_verdict.json`, publication admits the panel
  and backfill re-derives the verdict (`gate_sha256 8938b294…`, `task_sha256
  6c667a99…`). v1's `min_avg_reward` 1800 and its two duty keys stay in the
  file as comments, a superseded record (v2 refuses the duty keys). Panels and
  per-bar counts are in the dated §11 appended to
  `docs/investigations/STANCE_HACK_AUDIT_2026_10.md`. Statue constants
  re-measured on r2: stance 4570.4 ± 7.2 (`zero_action_baseline.py
  compsognathus --episodes 40 --seed 3042`), so `min_avg_reward` 1800 → 2740,
  with `collapse_peak_floor_reference = 4570.4` and
  `statue_constants_physics_revision = 2` beside it; the collapse detector
  stays unarmed (no floor), as before.

  Recovery inherits the new terms through `extends` and is recalibrated on
  r2: the r2 feet widen the home contact AABB margin from 0.0196 to 0.0293
  m, so every capture-velocity multiple pushes 1.5 times harder, and the
  toe armature lets the zero-command null ride out more of each push, so
  the same selection rule picks 1.3 (r1 1.32; 1.2 without the armature), a
  1.250 N pulse for 0.2 s (r1 0.848 N); `configs/compsognathus/recovery.toml`,
  `configs/compsognathus/recovery_calibration.json` and
  `RECOVERY_CALIBRATION.md` carry the re-measured profile (its
  `calibration_id` ends in `-v2`, with no restamp history; the robot's run
  reproduced its own profile, but for last-bit reward sums from a NumPy
  version change). Held out, both nulls succeed on 15 of 40 episodes; on
  the soft-cubic legs the calibration's fixed brace (±0.005 of action)
  moves the hip-pitch and ankle targets a tenth as far as on r1, so it is
  nearly the zero-command null. Zero action under the push schedule keeps
  0.971 of the quiet statue's per-step reward, the new terms taking 0.104
  of the 0.134 per step it loses, for the pads and digits a push rocks off
  the floor: they pay the return to the flat stance. The phase-C reset
  golden's `compsognathus/recovery` capture is re-taken: both seeds' reset
  draws are unchanged, only the observation and trajectory halves move.

  Shared code: `BaseDinoEnv.step`, which keeps every substep's per-foot
  force beside the running minimum since D-D27 (`_substep_foot_force_block`,
  a copy whose column minimum is the minimum), now also calls a read-only
  per-substep hook, `_accumulate_substep`, after each `mj_step`, a no-op except on the
  compsognathus coverage term, so the floor-truth recorder keeps
  `_substep_probe_hook`;
  `test_the_substep_force_block_and_hook_leave_every_species_bit_identical`
  rolls all six plants with a reading hook and finds observations, rewards,
  info and state bitwise equal. `support_force_aggregation` joins the
  behavior preparation's transition comparison (the four other non-weight
  settings this stance shares with the trex joined it with D-D27). New tests: the reward terms'
  inertness, MEAN/MIN split, load gates, width reference, coverage (a digit
  carrying the foot on its tip counts, its MTP end alone does not, each
  geom's contacts are summed before the threshold, and the hook reads each
  substep after it is stepped), impact and airborne arithmetic
  (`environments/compsognathus/tests/test_biological_rewards.py`); the
  stance declaration, with a recorded post-settle hop that the window pair
  and each of its keys alone refuse, and a recorded sine bounce and one-leg
  pump it certifies, the measured blind spot
  (`test_compsognathus_stance_gate_config.py`); the window pair's criteria,
  floors and old panels (`test_stance_gate_v2.py`) and a real-physics
  scripted post-settle hop that the report fails and the judge refuses from
  the written file (`test_stance_gate_v2_report.py`); the plant and
  interface (`test_models.py`, `test_training_env.py`,
  `test_plant_contract_layers.py`); and the noise cliff
  (`test_noise_tolerance.py`: σ 0.10 reaches the horizon on the first 5
  episodes, σ 0.135 on at least 10 of the first 12, the review's 0.8).

  D-D28 (#603) merged first and left the compsognathus bar to this revision,
  so the stance also declares `max_sole_cop_fore_aft` = 0.70, measured on the
  r2 plant through the report path (the audit's §11): the statue reads
  0.143–0.155, heel side, over 200 episodes on five seed blocks (r1's
  0.454–0.473 do not carry over), the statue under N(0, σ) command jitter at
  most 0.289 at σ 0.03 and 0.404 at σ 0.05, and the two short PPO runs'
  means at most 0.246. This plant's pad and digits are one plate whose digits
  reach past the pad's front edge, so a posture that leans the animal along
  its feet keeps the pads level and every digit loaded and no other bar sees
  it: both MTP servo targets ramped to −0.10 rad stand both pads on their
  front edges (0.860–0.927) and to +0.08 rad on their heel edges
  (0.844–0.862), and the right MTP alone at +0.04 rad the left pad on its
  heel edge (0.903–0.916), each clean on 40/40 under every other bar; 33 of
  the swept-back crouch's 40 episodes stand on their front edges (1.000);
  and the right toe tapped at ±0.25 and 10 Hz, which the block certified on
  40/40, rocks its pad from edge to edge (0.826–0.848). The bar refuses all
  of them, and both ankles ramped to −0.15 rad (0.936–0.986) on 40/40 by
  itself; the bounce, burst, one-leg pump and shuttle the block still
  certifies read 0.14–0.48. It is 0.70 rather than the T-Rex's 0.80 because
  this statue stands at 0.15 and 0.80 would clear the heel-edge panel by
  0.04; a partial edge stance still passes on up to (0.70 − s)/(1 − s) of the
  window, 65% at the statue's 0.15 (KNOWN_ISSUES). The committed stage passes
  the post-stage pipeline again with the bar, and the statue pad centre of
  pressure that `test_gait_recorder.py` bounds is re-measured on r2: 0.173 at
  seed 3042, heel side (0.464 on r1), under a ceiling of 0.20 (0.50 on r1).
  New tests: the recorded edge stances, a swept-crouch episode and the toe
  tap fail this bar alone while the statue, jittered statue and PPO means
  clear it (`test_compsognathus_stance_gate_config.py`), a real-physics
  stance leaned onto both pads' front edges fails the report and the judge
  on it alone (`test_stance_gate_v2_report.py`), and the compsognathus joins
  the T-Rex as the stances that declare the key
  (`test_stance_gate_config.py`).

  Records: note 15, both plant manifests (compsognathus entries only), the
  species catalog (the stance published as `stance_quality/v2`, policy r3,
  physics r2, visual r2) and `data/recovery_calibration_v1.json`;
  KNOWN_ISSUES carries what the revision leaves open (the deferred toe cap,
  the deferred proportions pass, the noise cliff, the soft bounce, one-foot
  flutter and weight shuttle the gate certifies, and an observation that
  reads the world heading while every spawn faces +x); the gait plan (its
  §16) and the stance-hack audit (its §11) take dated amendments,
  NEXT_STEPS's compsognathus rows say what the revision strands, and the
  species README, recipe review, training validation and recovery
  calibration notes follow the plant. Digest golden (649 lines, unchanged
  in count): every one of the 48 moved lines is a compsognathus line —
  plant `physics_revision`, `physics_sha256`, `policy_interface_revision`,
  `policy_interface_sha256`, `source_closure_sha256`, `visual_revision` and
  `visual_sha256`; policy `WHOLE`, `action_mapping`,
  `interface_implementations` and
  `interface_implementations.sb3_action_mapping`; stage `task_sha256`,
  `stage_config_view_sha256.PPO` and `stage_config_view_sha256.SAC` of
  stance, recovery, locomotion and behavior, and stance `gate_sha256`; the
  recovery calibration's `file_sha256`; the 11 behavior recipes'
  `task_sha256`; and `summary`, `shape_sha256` and `rounded_values_sha256`
  of the four compsognathus reward stages. Out of scope: no r2 policy has
  been trained; the toe cap and the proportions pass are deferred, as above;
  a heading-free observation is decided for a later PR, the maintainer's
  heading decision of 2026-10-07 (the D-D27 row's amendment): one
  cross-species PR for the T. rex and the compsognathus after this revision
  and before either trains, which moves the compsognathus policy interface
  again and makes the spawn-yaw probe a gate check, so the maintainer's
  two or more compsognathus stance seeds train on that revision, not on this
  one (this observation carries the world-frame pelvis quaternion, linear
  velocity and target direction; the statue is clean at every spawn yaw
  tried); the robot is untouched.
- **`stance_quality/v2` reads where along a pad the floor pushes, and the
  T-Rex stance refuses a foot standing on its pad's front edge** (#603, decision
  D-D28, which the maintainer took on 2026-10-07 after the review of the
  physics-r8 seed-44 stance `20261006_185704`; a gate-only change: the
  plant, the recorder, every task and `MEASUREMENT_VERSION` are unchanged).
  That stance stands its left pad 0.7° toe-down on two loaded corners with
  its digits carrying about 1% of the foot's load each (the statue's 6–8%),
  and no other v2 bar sees it: on the episodes every other bar passes (13/40
  on 3042–3081, 8/40 on 7042–7081) that foot is still on its pad's edge.
  `min_sole_contacts` (1.5) cannot rise to catch it, because the statue
  reaches 2.086 out of sample. A new `StanceEpisodeMetrics` field,
  `max_sole_cop_fore_aft`, is the worse box pad's window mean of its
  per-step |fore-aft centre of pressure| divided by the half-length (0
  centred, 1 on the heel or toe edge), read off the recorder's existing
  force-weighted sole CoP: per step, because the seed-42 stance rocks its pad
  heel to toe and its signed mean looks centred; a step with no loaded sole
  contact (none above `CONTACT_THRESHOLD_N`, as `min_sole_contacts` counts
  them) counts 1, so lifting the pad, or grazing the floor with it, cannot
  hide an edge stance; NaN on a foot with no box sole (the velociraptor's
  soleless foot, the brachiosaurus's ellipsoids), so a bar is set per
  species. The same name is a new optional v2 key (direction max, after
  `max_settle_touchdowns`; not a hop-or-fall key) whose bar must lie strictly
  inside (0, 1), through a new `CRITERION_BAR_RANGES` table that both
  `StanceV2Thresholds.from_curriculum` and `gate_schema.validate_gate_config`
  apply, so a typo such as 80 stops the run at config load rather than after
  it trained.
  It is the first of `STANCE_METRIC_LATER_FIELDS`, the metric fields added
  after the panel CSV contract shipped: `StanceEpisodeMetrics.from_row` and
  `read_stance_v2_panel` read one absent as unmeasured (the panel evidence
  names it in `absent_metrics`), so every report and `stance_panel_selected.csv`
  recorded before this change keeps its verdict, bit for bit, under a block
  that does not declare the key. Under one that does, such a panel is
  refused, never failed as unmeasured: publication, which re-derives from the
  CSV, refuses it by name (`stance_gate_v2.unrecorded_criteria`: re-roll the
  panel); the judge and backfill refuse a report of that age because the
  thresholds it records lack the key, and by name through
  `unrecorded_criteria` when they match. Re-rolling the panel on the handoff
  pair measures it. The decision asked for a "reader/version migration";
  this change reads it as a reader-only one (`STANCE_METRIC_LATER_FIELDS`,
  no `MEASUREMENT_VERSION`, measurement-digest or report-schema bump),
  because a new metric changes no adopted key's meaning, so a panel CSV
  written before it and one written after carry the same measurement stamps
  and differ only by the column; that reading is the maintainer's to confirm
  at review. The T-Rex stance declares the key at 0.80: the statue reads
  0.206–0.362 on 3042–3081, at most 0.368 on four more seed blocks and 0.480
  on 5042–5081 (seed 5048, a fore-aft sway clean on every other bar), over
  238 full-horizon episodes, and up to 0.745 under N(0, 0.05) command jitter
  on every step (13042–13081); the seed-44 left foot reads 0.850–1.000, over
  the bar on 140/140 episodes on four seed blocks, and seed 42's rock
  0.96–0.99. Not 0.75: the D-D27 369k-step study checkpoint stands on a flat
  pad loaded forward to 0.88, and 0.75 would fail its 3042–3081 panel (38/40
  clean → 34/40) where 0.80 leaves 37/40. The stance report lists the
  reading as report-only where it is not gated; the manager's v2 mirror, the
  catalog and the website adapter carry the key. Digest golden (649 lines):
  3 lines move, all T-Rex stance (`gate_sha256`,
  `stage_config_view_sha256.PPO` and `.SAC`); no task, recipe, plant or
  other species' line moves, and no measurement digest moves. The
  end-to-end report test of the edge refusal runs on the T-Rex, the species
  that declares the key; the compsognathus readings the other tests pin were
  its r1 plant's when this change landed, before the compsognathus revision
  (D-D26, #604), which re-measures them on r2. Records: D-D28 in
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2 and the consolidation plan's table,
  which also records the landings of #601 and #602; the stance-hack audit's
  §10 and the gait plan's §15; KNOWN_ISSUES's front-edge entry records what
  the bar leaves open (partial-edge stances, among them a drift onto the
  edge after about the first third of the window and a rolled pad on three
  corners; only the T-Rex declares it), and its heading entry records the
  maintainer's decision of the same day (heading-free observations for the
  T-Rex and the compsognathus, in a later PR).
- **The T-Rex stance prices what its first two physics-r8 runs exploited,
  starts its exploration quieter, and its gate counts settle re-plants,
  turns and hop-or-fall episodes** (#602, decision D-D27, which the maintainer took
  on 2026-10-07 after a review of the first two T-Rex stance runs on the
  physics-r8 task of D-D24; the plant does not change: physics r8, policy
  interface r13, visual r4). The runs, `20261006_185343` (seed 42) and
  `20261006_185704` (seed 44), trained D-D24's `configs/trex/stance.toml`
  byte for byte for 11M steps, and both fail `stance_quality/v2` as recorded
  and as re-derived on `main`: seed 42 is clean on 0/40, a 12.5 Hz
  synchronous two-foot micro-hop on raw ±1 commands at 0.80 of the statue's
  reward; seed 44 on 13/40 (bound 0.204; 8/40, bound 0.104, on the fresh
  seeds 7042–7081): it stands like the statue after the settle, but stomps
  or hops its feet back to the keyframe width in the settle (settle peaks
  over 1.5 BW on 27/40 and 32/40), hops for the whole episode on 9 of 207
  resets, stands on the front edge of its left pad and falls when spawned
  90° off its training heading. The r7 hacks (the hip-roll splay, the
  rolled-pad prop, the widened stance, the raised head) are gone. The same
  config split into two basins: seed 44 escaped the early hop regime between
  5.0M and 6.0M, and seed 42 locked in at 6.5M as the entropy bonus ran out
  at 7M. The record is the dated §9 appended to
  `docs/investigations/STANCE_HACK_AUDIT_2026_10.md`.
  The stance task (`configs/trex/stance.toml` `[env]`, which recovery
  inherits) takes seven new `TRexEnv` kwargs, each inert at its default (the
  legacy arithmetic and every existing info stream bit for bit;
  `test_trex_rewards.py`): `stance_width_reference = "settled"` with
  `stance_width_settle_steps = 200` centres the width term on the animal's
  own width at the end of the settle and pays it only after (the keyframe
  target paid seed 44 +12.5 per episode over the statue for re-seating its
  spawn, and a quiet constant re-seat now gains −2.0 in width instead of
  +6.7, though still about +3 in total, mostly through foot flatness);
  `foot_terms_min_support_force = 168.0` pays a foot's flatness share, and
  the pair's width term, only while it carries 0.20 BW (seed 42's lifted
  feet earned about 80% of the flatness reward in the air);
  `floor_impact_weight = 2.0` per body weight above
  `floor_impact_threshold_bw = 1.4` and `airborne_substep_weight = 1.0`
  price a step's peak summed foot force and its fully airborne substeps over
  the whole episode, settle included (the statue's per-step peak after the
  0.1 s spawn grace is at most 1.331 BW over 200 episodes, and it is never
  airborne after its spawn steps); and `action_penalty_source = "raw"`
  prices smoothness, jerk and saturation on the policy's own clipped command
  instead of the 10 Hz-filtered one, energy staying on the applied command,
  because the first-order filter passes 24–54% of 12.5–50 Hz content and
  so charged seed 42's bang-bang 8.8× (smoothness), 11.4× (jerk) and 222×
  (saturation) too little. Under "raw" the `action_delta`, `action_jerk` and
  `action_saturation` info keys, and the training `diagnostics.npz` series
  that record them, read the raw command (the evaluation plateau's action
  statistics and frequency estimate always read the policy's own action
  and do not move), and `BaseDinoEnv.step`'s
  comment on pricing the filtered command says why T-Rex does not.
  `recovery.toml` overrides `stance_width_settle_steps = 140`, ahead of its
  first push (200 ± 50 steps); what a stepping recovery's landings cost is
  unmeasured. Re-scored exactly on the recorded panels (no reward term moves
  a trajectory): the statue 3766.1 → 3756.8; seed 42 0.80 → 0.28 of the
  statue; seed 44 against the statue +5.7 → −17.5 per episode on 3042–3081,
  +17.3 → −5.6 on 5042–5081 and −97 → −153 on 7042–7081, a shift of about
  −23 on the first two blocks and −56 on the third, where its three hop
  episodes lose about twice what they lost before (−18 on its 36 standing
  episodes); so the v2 settle bars, not the reward, are what refuse its
  stomp.
  New: `BaseDinoEnv.step` keeps every substep's per-foot force beside the
  MIN the support terms read (`_substep_foot_force_block()`; the MIN and
  every existing key unchanged); `reward_sole_flatness(...,
  foot_weights=None)`; the reward keys `reward_floor_impact` and
  `reward_airborne_substeps` (`diagnostics.py`, `visualization.py`); the info
  keys `stance_width_target`, `r_foot_terms_supported`,
  `l_foot_terms_supported`, `peak_foot_force_bw` and `airborne_substeps`;
  and `behavior_checkpoint` carries the five non-weight kwargs into
  transition rewards.
  The stance recipe starts at `[ppo.policy_kwargs] log_std_init = -1.5`
  (std 0.22; recovery inherits it): with the new terms a fresh policy pays
  2.88 / 1.00 / 0.41 per step at std 1.0 / 0.37 / 0.22, against the recipe's
  rule of about half the 1.0 alive bonus, and in short PPO runs on the task
  the stochastic policy's unsupported duty is 0.52–0.54 at std 1.0 (the r8
  runs: 0.55–0.57 at 0.1M) against 0.044–0.046 at 0.22, falling to 0.003 by
  451k steps; whether the start changes the 5–7M basin selection is
  untested. `ent_coef_decay_timesteps`'s comment records that its own
  falsifiable prediction failed in both r8 runs (algo_std 0.750 / 0.724 at
  4M, not "well under 0.2"; unsupported duty 0.409 / 0.358, not below 0.28)
  and that the learning rate it names next is untested; the anneal and the
  learning rate are unchanged. `collapse_peak_warmup_timesteps` stays 1.0M,
  and its comment and `EvalCollapseEarlyStopCallback`'s docstring say its
  premise was the std-1.0 start.
  An opt-in hop watch (`environments/shared/curriculum/hop_watch.py`, wired
  by `train_base._build_core_callbacks`): when a stage's `[curriculum]` declares
  `hop_watch_max_unsupported_duty`, the first evaluation at or after
  `hop_watch_after_timesteps` whose unsupported duty (`gate_progress.npz`,
  read through the new `StageGatePlateauCallback.gate_progress_series`) is
  above the bar logs a warning, writes `hop_watch.json` and, with
  `hop_watch_stop`, ends training; the post-stage report then judges the
  handoff pair as usual, and a stop certifies nothing. The keys join the
  early-stop family (`gate_schema._COLLAPSE_KEYS`) and enter no digest. The
  T-Rex stance declares 0.05 after 7M (= `ent_coef_decay_timesteps`) with
  stop: on the r8 series it would have stopped seed 42 (exactly 0.375 at
  every evaluation from 6.5M to 11M) with 4M steps unspent and left seed 44
  alone (under the bar from 5.35M; 0.013 at 7M and at most 0.018 after, 2.8×
  under it).
  `stance_quality/v2` gains three optional keys, each applied only where
  declared (`stance_gate_v2.py`, `gate_schema.py`, `manager.StageThreshold`;
  `GATE_SCHEMA_VERSION` stays 1): the per-episode criteria
  `max_episode_yaw_change_deg` (on `episode_yaw_change_deg`, report-only
  until now) and `max_settle_touchdowns` (on `settle_touchdowns`), and the
  panel rail `max_hop_or_fall_episodes`, which caps the episodes that end
  early or fail `min_all_feet_support`, `max_touchdown_rate`,
  `max_window_displacement_m` or `max_actuator_saturation_fraction`
  (`HOP_OR_FALL_KEYS`) and counts from the classification reasons, so a
  recorded panel CSV re-derives it; the result carries `hop_or_fall_episodes`
  only when the rail is declared, so every other stage's report serialises as
  before. The T-Rex stance block declares `max_settle_stance_width_change_m =
  0.08`, `max_settle_touchdowns = 2`, `max_episode_yaw_change_deg = 25.0` and
  `max_hop_or_fall_episodes = 1`; `min_sole_contacts` stays 1.5 (the statue
  reaches 2.086 out of sample). At n = 40 the rail passes the statue's
  1-in-120 window-failure rate 0.956 of the time and seed 44's 4.8%
  hop-or-fall rate (9 whole-episode hops and a roll-over in 207 resets) about
  0.42 (0.87 under the bound alone): it halves that admission, it does not
  remove it. Validated: the statue is clean on 40/40 with no hop-or-fall
  episode on 3042–3081 and 7042–7081; seed 42 fails the rail, the yaw bar and
  the settle-touchdown bar on 40/40; seed 44 passes the rail on 3042–3081 (1
  episode) and fails it on 7042–7081 (4). The website's gate adapter and the
  species catalog render the rail and the two criteria.
  A report-only heading probe: `[curriculum] stance_probe_spawn_yaw_deg =
  [-90.0, -45.0, 45.0, 90.0]` (a diagnostic key) re-rolls the policy and the
  zero-action statue on 8 seeds with the spawn turned about vertical by each
  offset, the animal alone and with the prey and the heading term's
  reference direction, beside an unturned control, and writes
  `stance_heading_probe.{txt,json}`; `stance_gate_report.py
  --spawn-yaw-offsets` (with `--spawn-yaw-episodes`) runs it standalone. The
  statue stands 8/8 at every offset both ways; seed 44 stands 0/8 at ±90°,
  and 0/8 at ±45° with the prey turned too. Whether stance certification
  should require heading robustness (spawn-yaw randomisation or a
  heading-invariant observation, both policy-interface revisions) is not
  decided. The v2 judge refuses a heading-probe report as a probe: its
  probe check now reads the report writer's marker registry
  (`stance_report._PROBE_MARKERS`) instead of a list of its own, and a
  `stance_quality/v2` report records `spawn_yaw: null` beside the other
  probe markers.
  Statue constants re-derived under the new task, with
  `statue_constants_physics_revision = 8` kept: stance 3756.8 ± 17.3
  (`zero_action_baseline.py trex --stage 1`, seed 3042, 40 episodes; −9.1
  width, −0.2 flatness), so `min_avg_reward` 2260 → 2250 and
  `collapse_peak_floor_reference` 3766.1 → 3756.8; recovery's pushed statue
  1337.5 → 1329.1 (`roll_recovery_panel`, with the capture at 140).
  Locomotion (1091.5) and behavior (602.0, rail 361) set none of the new
  kwargs and keep theirs; their comments say so.
  Digest golden (649 lines): 36 lines move, all trex — stage `task_sha256`,
  `stage_config_view_sha256.PPO` and `stage_config_view_sha256.SAC` of
  stance, recovery, locomotion and behavior; stance `gate_sha256`; stance and
  recovery `hyperparameters_sha256.PPO`; the 11 behavior recipes'
  `task_sha256`; the reward `summary` of stance and recovery; and
  `shape_sha256` and `rounded_values_sha256` of the four trex reward stages
  (their new info keys). No plant, policy-interface or other species' line
  moves. Records: D-D27 in `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2 and the
  consolidation plan's table; KNOWN_ISSUES's T-Rex stage-1, stance-gate and
  fixed-seed-block entries and its `action_saturation` note, and two new
  entries (heading dependence; a foot on the front edge of its pad); the gait
  plan's dated §14; NEXT_STEPS's trex rows.
- **The Velociraptor stands on its servos on flat toes, feels its whole foot,
  leaves the frozen MJX core, and its stance is judged on floor truth**
  (#601; breaking — plant change, physics revision 2 → 3, policy interface revision
  10 → 11, visual revision 3 → 4; all existing Velociraptor checkpoints are
  invalidated; decision D-D25, which the maintainer chose on 2026-10-06). Four
  measured defects of the r2 plant
  (`docs/reviews/VELOCIRAPTOR_PLANT_REVIEW.md`, the stance-hack audit's §3.2),
  fixed in one revision so the species' digests move once
  (`configs/plant_versions.toml` note 14):
  every leg spring pulled toward qpos 0, because no leg joint set `springref`
  (MuJoCo's default is 0, not `ref`), so the stiffness-20 springs pulled the
  settled hip, knee and ankle with 11.9 / 15.8 / 28.0 N·m and the servos spent
  10.4 / 15.1 / 34.5 N·m cancelling them; the keyframe kept every leg joint at
  `ref`, which tips the toes 20° toe-down, digit III 44.6 mm inside the floor,
  and every reset settles to the keyframe's own clearance (reset peak floor
  force 2.08 BW mean, 2.42 max); the foot touch sensor saw digit III only, 55%
  of the foot's floor load; and the r2 statue settled at 24.0°, 4° past
  `natural_pitch`, paying −95.3 per episode of nosedive charge. In
  `raptor.xml` every leg joint's `springref` is now the standing pose; the
  ankle (100 → 90.5536°) and toes (10 → −0.5536 / 0.6464°) flatten the foot
  while hip 38° and knee −50° keep the centre of mass over it, with the root
  at 0.505543 m (0.5 mm of contact, the T-Rex convention). Re-posing the hip
  instead, to the 18° crouch the keyframe comment named (38° = 18° + 20° of
  lean), flattens the toes but moves the support 7.6 cm ahead of the centre of
  mass: the statue falls backward in 0.66 s on 10 of 10 seeds, still with the
  legs held near-rigid (kp × 20 and × 100), so it is geometry, not servo
  strength. Springs anchored at the stance are not enough on their own either:
  with the home ctrl at the pose the statue lands on its body within 1.1–1.34
  s on 10 of 10 seeds, because a servo with no error holds no torque, so the
  joints sag 3–4° and the centre of mass rolls past the toe tips. The home
  ctrl therefore carries the gravity preload (ctrl − qpos = hip −0.606°, knee
  +0.166°, ankle +3.206°, digit III +2.651°, digit IV +1.127°: the gravity
  torque of a near-rigid hold over kp), and the nominal servos, kp, forcerange
  and ctrlrange unchanged, hold the pose with 1.60 / 0.48 / 5.56 / 2.33 N·m
  (0.2–5.8% of forcerange), on 40 of 40 episodes even with every leg spring
  deleted (r2: 0 of 20). The review's "re-size the leg actuators at the same
  time" is not needed. Metatarsus and digit-IV touch sites and sensors are
  appended (sensordata 27–30, right foot first), so existing indices hold and
  the appended sensors alone move no dynamics (max |dq| = 0 over 2000
  perturbed steps); the env sums digit III, metatarsus and digit IV per foot
  (`_foot_sensor_groups`, and `_get_obs` through `_foot_contact_forces()`),
  and the summed reading equals the floor force (`foot_sensor_report.py`:
  0.553 → 1.000). Observation 70 and action 22 are unchanged.
  The summed foot cannot be mirrored by the frozen MJX registration, which
  D-D17 never edits, so the Velociraptor leaves the core inside this
  policy-interface revision, as KNOWN_ISSUES and
  `docs/CLEANUP_PLAN_2026_09.md` §4.1 prescribed:
  `RaptorEnv.supported_training_backends = ("stable-baselines3",)`,
  `training_backends = ["stable-baselines3"]` in
  `configs/species_manifest.toml` (its two JAX/MJX success metrics go), and
  `environments/velociraptor/mjx_config.py` is deleted with its
  `pyproject.toml` ruff exclusion. Three species (trex, brachiosaurus and
  dibothrosuchus) stay on the core; §4.1's batched checklist (one base
  `_scale_action` and `_get_obs`, the midpoint mapping and the reset height
  channel) is not part of this revision.
  Zero action, 40 episodes, reset noise 0.05, seeds 3042–3081, r2 → r3: full
  horizon 40/40 → 40/40; settled pitch 24.02° → 20.10° (0.3508 rad, so the
  0.35 `natural_pitch` default is right again, now pinned by a freshness
  test); pelvis 0.4932 → 0.5052 m; reset peak floor force 2.08 / 2.42 → 1.34 /
  1.76 BW (mean / max; noise-free 2.26 → 1.01); nosedive charge −95.3 → −3.0
  per episode; reward at the old weights 1745.8 → 1845.9.
  The stance task (`configs/velociraptor/stage1_balance.toml` `[env]`) takes
  five reward terms through eight new `RaptorEnv` kwargs with the `TRexEnv`
  names, inert at their
  defaults (there the new code's reward, its 19 components and qpos are
  bitwise equal to the old code's over 800 steps, measured on r2):
  `support_conditioned_alive_fraction = 0.2`, `bilateral_support_weight = 0.5`
  at `foot_contact_saturation_force = 50.0` N (floor-true since the sensor
  sum), `leg_home_pose_weight = 0.5` at `leg_home_pose_tolerance = 0.20` rad
  (the keyframe is now the standing equilibrium), `action_saturation_weight =
  0.5` at `action_saturation_threshold = 0.9`, and `action_jerk_weight = 1.0`
  (computed before smoothness), with reward keys `reward_bilateral_support`,
  `reward_leg_home_pose`, `reward_action_jerk` and `reward_action_saturation`
  and their info keys. The statue stays the reward optimum (2842.8 ± 2.6 of a
  2850 ceiling, every new term at its maximum). Re-scored from replays of the
  audited stance `20260922_125248` (robust_best and final, on the r2 plant
  they load on), which gained 53.5 and 58.2 per episode over the r2 statue by
  escaping its nosedive charge, the terms price the least-priced episode 876
  and 724 below the r3 statue, and each family alone prices it above that
  gain; jerk is a guard against command chatter only (9.3 and 4.7). At
  initialisation the penalties cost 0.70 per step, 40% of the 1.75 alive
  bonus. The gait plan's metatarsus penalty is not used: the r3 statue carries
  28% of each foot's load on the metatarsal head; nor is `foot_load_balance`,
  since the bilateral term already pays the weaker foot.
  `leg_home_pose_error` is one of the keys that switch on stance diagnostics,
  so training logs and evaluation metrics now report them for the
  Velociraptor (reporting only).
  The stance gate moves from `reward_and_length/v1`, which the statue cleared
  by design and which certified the audited crouch, to the floor-truth
  `stance_quality/v2` (D-D23), the second stage to adopt it: the clean-episode
  bound ≥ 0.80 over the 40-episode panel, `settle_steps` 100 (measured: both
  feet down by step 5, floor force settled by step 15), every required bar
  (`min_foot_load_share` 0.40, settle airborne substeps 0, settle peak 2.0 BW,
  saturation 0.10), the flatness bars of a foot with no box sole
  (`min_support_geom_duty` 0.50 and `min_support_geom_coverage` 0.80: digit
  III, digit IV and the metatarsal head each loaded), the settle width change
  ≤ 0.05 m, the load share ≥ 0.80 of the statue's, four guards and the rails
  (full horizon ≥ 95%, reward ≥ 1710 and ≥ 0.60 of the statue panel each
  report rolls). `min_avg_episode_length` stays in the file as a superseded
  record (v2 refuses the key). The foot bars and the settle peak sit mid-gap
  between the statue and the audited hacks: first set just under the statue's
  worst (duty 0.90, coverage 0.95, peak 1.5 BW), they refused the statue with
  N(0, 0.03) command jitter on 24–27 of 40 episodes, at 98.5% of its reward,
  because its lightly loaded digit IV unloads on part of the window, a loss
  no reward term sees; at the committed bars the jittered statue is clean on
  40/40 at every sigma from 0.02 to 0.05. Validated through the report path,
  verdicts re-derived from the CSVs: the r3 statue is clean on 40/40 (bound
  0.928), on 80/80 more seeds, on 40/40 of a third block rolled after
  calibration and on 200/200 fresh seeds; the
  audited stance's two checkpoints, rolled on the r2 plant with the block
  swapped in, are clean on 0/40 each, every unclean episode failing at least 9
  (robust_best) and 6 (final) bars, and saturation, the settle bars, the foot
  bars and the width bar each refuse every one of their episodes alone; the r2
  statue is clean on 40/40. The committed stage also passed the post-stage
  pipeline end to end with the r3 statue as a scripted checkpoint (report,
  judge, `gate_verdict.json`, publication's re-derivation, backfill; the stage
  declares no probes). Panels, per-bar counts and margins are in the dated §8
  appended to `docs/investigations/STANCE_HACK_AUDIT_2026_10.md`.
  Statue constants re-measured on r3: stance 2842.76 ± 2.62
  (`zero_action_baseline.py velociraptor --episodes 40 --seed 3042`; the v2
  report's own statue panel agrees), so `min_avg_reward` 1050 → 1710 and
  `collapse_peak_floor` 1300 → 2130 (0.75 ×, still absolute), with
  `collapse_peak_floor_reference = 2842.8` and
  `statue_constants_physics_revision = 3` recorded beside them, so the
  freshness test fails a later physics bump that forgets to re-measure the
  statue (the explicit floor still wins over the reference); locomotion and
  strike carry placeholder rails and no statue-derived constant, and neither
  do the 11 recipes. Unchanged: kp, forcerange and ctrlrange, the joint ranges
  and `ref`, `natural_pitch`, the claw motors, the locomotion, strike and
  recipe `[env]` blocks, the reset draw stream and every other species.
  Records: note 14, both plant manifests and the species catalog (the
  Velociraptor stand is published as `stance_quality/v2`, and its JAX/MJX
  success lines go); the phase-C reset golden holds no Velociraptor capture
  and does not move; KNOWN_ISSUES closes the foot-sensor entry and deletes
  the stage-1 gate, spring, touch-sensor, natural-pitch and hardware-note
  items; `docs/PLANT_CONTRACT.md` counts
  three dual-backend species; the cleanup plan's §4.1 and §4.2 and the gait
  plan (its §13) take dated amendments, and NEXT_STEPS's Velociraptor rows say
  what the revision strands.
  Digest golden (656 → 649 lines): every moved line is a Velociraptor line —
  plant `physics_revision`, `physics_sha256`, `policy_interface_revision`,
  `policy_interface_sha256`, `visual_revision`, `visual_sha256` and
  `source_closure_sha256`; policy `WHOLE`, `action_mapping`,
  `backend_observation_equal`, `dimensions`, `interface_implementations`,
  `jax_interface`, `observation_probe`, `observation_segments`, `sensors`,
  `sites`, `interface_implementations.home_reset` and
  `interface_implementations.sb3_observation`, with the eight
  `jax_interface.*` lines and `interface_implementations`'
  `backend_neutral_observation`, `jax_action_mapping` and
  `jax_observation_callers` gone and
  `interface_implementations.sb3_sensor_cache`,
  `interface_implementations.supported_training_backends`,
  `jax_interface.backend` and `jax_interface.supported` added (the SB3-only
  payload); stage `task_sha256`, `stage_config_view_sha256.PPO` and
  `stage_config_view_sha256.SAC` of stance, locomotion and behavior, and
  stance `gate_sha256`; the 11 behavior recipes' `task_sha256`; and `summary`,
  `poses`, `shape_sha256` and `rounded_values_sha256` of the three
  Velociraptor reward stages (`poses` because the neck-low state probe now
  ends on `body_contact`, which the golden structure test admits).
- **T-Rex hip-roll servos sized for the plant's mass, with the stance task
  and gate revised on the new plant** (#600; breaking — plant change, physics
  revision 7 → 8; all existing T-Rex checkpoints are invalidated; decision
  D-D24, which the maintainer chose on 2026-10-06). `trex.xml`'s
  `r/l_hip_roll_act` go kp 150 → 600 and forcerange ±120 → ±480 N·m,
  keeping the 0.8× spike cap `test_actuator_bounds.py` pins; ctrlrange,
  joints, bodies, keyframe, sites and sensors do not move, so the policy
  interface stays revision 13 (`0b43de…`) and the visual layer revision 4
  (`7042dc…`), with observation 64 and action 15. The stale keyframe
  checklist in its comment now names `stance.toml` and the settled-pose
  constants. Motivation, measured on the zero-action statue's 40-episode
  panel (seeds 3042–3081, reset noise 0.05): the July home-equilibrium
  repair mass-scaled hip pitch, knee and ankle only, and with both feet
  planted the hip-roll servos and their 40 N·m/rad springs hold lateral
  sway only while 2 × (kp + 40) exceeds the settled m·g·h of 704.4 N·m/rad
  — kp 150 gave 380, kp 600 gives 1280 — so the reset's hip-roll noise
  stayed in the stance: the worse pad rolled up to 3.42° and the outer
  digit lifted on 4/40 episodes, the propped-foot signature of the audited
  stances on the statue itself, which the new gate scores clean on only
  36/40 on r7. On r8 the worst pad is 1.15° (1.53° over 80 more seeds),
  quasi-static lateral push capacity rises from 3.3% to 5.1% of body weight
  (note 13 names the protocol; fore-aft stays 3.2%, set by leg-chain
  compliance, a fragility KNOWN_ISSUES now records), and the pushed
  recovery statue loses all 20 of its lateral falls. The symmetric settle
  does not move (noise-free reward 3517.26, pelvis 0.9260 m, `forward_z`
  −0.0279), so `target_z`, `natural_pitch` and
  `nosedive_termination_threshold` keep their pins. Stiffer passive digits
  and `tail_1` authority were measured and left out (note 13 of
  `configs/plant_versions.toml` has why).
  The stance task (`configs/trex/stance.toml` `[env]`, which recovery
  inherits) takes three new `TRexEnv` kwargs, each inert at its default
  (at the defaults the new code's reward and every existing info stream are
  bitwise equal to the old code's, measured on r7): `neck_posture_reference
  = "settled"` centres the neck
  term on the statue's own settle (`_NECK_SETTLED_QPOS = (0.21432, 0.0,
  0.21963)` rad, with a freshness test) instead of the keyframe, so lifting
  the head to the keyframe, which paid +45.6 per episode, costs 36.4;
  `foot_flatness_weight = 0.15` at `foot_flatness_tolerance_deg = 3.0`
  rewards level plantar pads; `stance_width_weight = 0.1` at
  `stance_width_tolerance_m = 0.05` the home stance width (0.280 m between
  the foot sites). Re-scored on replays of the four audited r7 stances, the
  three terms cost them 184.6–234.6 per episode against the 29.0–51.3 each
  gained over the statue, and on r8 no constant deviation in a hack
  direction beats the statue (40 seeds: raising the head to the keyframe
  costs 36.4, a ±0.05 hip-roll splay 3.7–5.2). The statue is 23.2 short of
  the terms' maxima, inside its 25.9 episode sd, and it is not the reward
  optimum: a policy that corrects its own spawn can gain about 15 per
  episode from flatness and width together (the best constant hip-roll
  offset per seed, better on 36/40 seeds) and about 2 by holding its head
  up through the settle (a 0.05–0.10 neck lift, better on 40/40), both in
  the intended direction. At initialisation the terms only add reward
  (+0.052 per step), so they cost nothing against the alive bonus. New
  pure terms `reward_sole_flatness` and `reward_stance_width`
  (`reward_functions.py`; the width is the planar foot-site distance, so a
  fore-aft stagger counts as width error), reward keys
  `reward_foot_flatness` and `reward_stance_width` and their info keys.
  The stance gate moves from `stance_quality/v1` to the floor-truth
  `stance_quality/v2` (D-D23), the first stage to adopt it: the clean-episode
  bound ≥ 0.80 over the 40-episode panel, `settle_steps` 200 (now measured:
  the statue's floor force has settled by step 47), every required bar, the
  pad bars trex needs (window-mean tilt ≤ 2°, corner lift ≤ 4 mm, loaded
  contact points ≥ 1.5), four guards and the rails (full horizon ≥ 95%,
  reward ≥ 2260 and ≥ 0.60 of the statue panel each report rolls). v1's
  duty pair and settle prefix stay in the file as a superseded record; the
  four probes keep running, on v1-shaped reports. Validated through the
  report path, verdicts re-derived from the CSVs: the r8 statue is clean on
  40/40 (bound 0.928) and on 80/80 more seeds (and on 39/40 of a third
  block, rolled after calibration: seed 3174 nosedives on reset noise
  alone, on r7 too, so the statue figures are sample extremes); the four
  audited checkpoints (`20260914_123816`, `20260920_010912`,
  `20260930_024929`, `20261001_225601`), rolled on the r7 plant they load
  on with the block swapped in, are clean on 0/40 each, every unclean
  episode failing at least 3 bars; the r7 statue is clean on 36/40 (bound
  0.786). Without the pad and settle bars, seed 42's stance would pass
  40/40. The committed stage also passed the post-stage pipeline end to
  end with the r8 statue as a scripted checkpoint (report, judge,
  `gate_verdict.json`, publication's re-derivation, backfill, probes).
  Panels, per-bar counts and margins are in the dated §7 appended to
  `docs/investigations/STANCE_HACK_AUDIT_2026_10.md`; the website's gate
  adapter gains the v2 arm (`species.ts`, `formatGate`), pinned both ways
  by the catalog tests.
  Statue constants re-measured on r8, with `statue_constants_physics_revision
  = 8` in all four files: stance 3766.1 ± 25.6 (`zero_action_baseline.py
  trex --episodes 40 --seed 3042`; `stance_quality_baseline.py 0.05 40`
  agrees; +270.9 over 3495.2 = +1.8 plant, +41.2 neck target, +141.4
  flatness, +86.5 width), so `min_avg_reward` 2100 → 2260 and
  `collapse_peak_floor_reference` 3495.2 → 3766.1; recovery's pushed
  statue 974.7 → 1337.5 (`roll_recovery_panel`, 40 episodes: +275.4 from
  the plant, +87.4 from the inherited terms); locomotion 1091.5 and
  behavior 602.0 (rail 361) re-measured and kept. Unchanged: the recovery
  gate's frozen `recovery_quality/v1` thresholds, the locomotion, behavior
  and recipe `[env]` blocks, the reset draw stream, and every other
  species. `CALIBRATED_POSTURE_ONLY` and `CALIBRATED_HEIGHT_REFERENCE_M`
  (`recovery_evaluation.py`) were measured on the retired r7 stance and
  are re-measured on the next certified one. Records: note 13, both plant
  manifests and the species catalog; the phase-C reset golden re-takes only
  its trex captures, whose reset halves were unchanged (the revision draws
  nothing extra), and records each recapture with the physics it was taken
  on; KNOWN_ISSUES's trex and stance-gate entries, and NEXT_STEPS's trex
  certification rows, say what the revision strands.
  Digest golden (656 lines): 39 lines move, all trex — plant
  `physics_revision`, `physics_sha256` and `source_closure_sha256`; stage
  `task_sha256`, `stage_config_view_sha256.PPO` and
  `stage_config_view_sha256.SAC` of stance, recovery, locomotion and
  behavior, and stance `gate_sha256`; the 11 behavior recipes'
  `task_sha256`; and `summary`, `shape_sha256` and `rounded_values_sha256`
  of the four trex reward stages. The `poses` lines and every other line
  hold.
- **A live child warm-started from a parent that never saw a command
  starts exactly command-blind** (#597, consolidation PR-10 of
  `docs/CONSOLIDATION_PLAN_2026_09.md`, 2026-10-05). Under
  `initialize_next_stage`, when the child's task fingerprint has a
  `command` section and the parent's recorded one has none (or the parent
  records none), `train_base._create_or_load_model` zeroes the command
  columns of the policy's first layers and their optimizer moments
  (`policy_loading.neutralize_command_columns`: PPO's actor and critic;
  SAC's actor and every Q network of its critic and target critic, whose
  command columns sit before the action) and proves on the child's
  normalizer, before the first update, that the actions and values are
  the parent's on zero commands and do not move with live ones
  (`policy_loading.assert_command_blind`: 64 seeded observations, the
  reference taken before the zeroing, then uniform commands and the
  probe vector, within 1e-6). The task lineage records
  `zeroed_command_parameters`. A parent trained under `"none"` never gave
  those columns a gradient, so a live child would otherwise start from a
  policy that reacts to commands it was never taught; a parent whose
  fingerprint has a `command` section keeps its columns. The behavior
  preparation zeroes and probes through the same two functions, with the
  same zeroed parameters, report, refusals and messages, plus one refusal
  on the probe vector that an exact zeroing never triggers; the widen
  tool binds `VERIFICATION_ROLLOUT_SEED`, `ACTION_DELTA_ATOL` and its
  optimizer table from `policy_loading`. Every committed stage is
  `"none"`, so no stage trains differently and no digest moves.
- **Commands come through the reserved hook, and a behavior recipe's
  identity is its task fingerprint** (#595, consolidation PR-9 of
  `docs/CONSOLIDATION_PLAN_2026_09.md`, decisions D-D2 and D-D9,
  2026-10-05). `BaseDinoEnv` and the five species constructors take
  `command_mode` and one `command_config: DirectionCommandConfig | None` in
  place of `command_speed_range`, `command_lateral_range`,
  `command_yaw_rate_max`, `command_switch_interval` and
  `command_switch_jitter`, which nothing read. Under `"heading"` or
  `"heading_and_speed"` the env builds a `DirectionCommandController`,
  seeds it in `_draw_episode_command()` with one draw appended to the reset
  stream (`_command_rng()`) and advances it at the end of `step()` through
  `_update_command()`, after the reward, so the returned observation
  carries the next command; `command_manifest()` is the controller's
  manifest and `_heading()` is the base env's. A live mode needs a
  `command_config`, `"none"` refuses one, and `"heading"` holds
  `cruise_speed` (it refuses `speed_range` and stops). SB3 accepts both
  live modes (`validate_command_mode` keeps its MJX refusal); under
  `"none"` nothing is built or drawn. The task fingerprint's carve-out pops
  `command_mode` and `command_config` while the effective mode is
  `"none"`, so no committed `task_sha256` moves, and records the config
  field by field otherwise. Its `command` section is the controller's
  manifest, passed exactly when the mode is live (a live `[env]` without
  one, or one under `"none"`, is refused), and `stage_task_fingerprint`
  takes an optional `command_manifest`, forwarded only when given.
  Following D-D2 strictly is the maintainer's choice of 2026-10-05: runs
  recorded before PR-9 are not kept rebuildable from their recorded
  constructor kwargs (a reader must drop
  `command_frame.RETIRED_COMMAND_ENV_KEYS`, the five names, first, and no
  path on main does so today: the gait audit's hand-run probe,
  `docs/investigations/gait_2026_09/gait_probe.py`, run with
  `--use-recorded-env-kwargs`, is one such reader and now refuses a pre-PR-9
  record), and committed stage fingerprints, and so the reuse of certified
  walkers as trunks, are unchanged. The constructors refuse the five
  (`TypeError`); by this PR's own default, the task fingerprint refuses each
  by name in every mode rather than hash a task that matches nothing. The
  behavior env runs `"heading_and_speed"` with its `commands`
  (`create_behavior_env` refuses `command_mode` and `command_config` by
  name) and keeps its own command stream, so every recipe's resets,
  observations, rewards and infos are bit-identical (`--exact`, 32,389 lines
  on base and head). A behavior checkpoint carries the canonical stamps: the
  plant identity, the recipe env's task fingerprint (its stage,
  `command-terrain/v2`, names the implementation and is never a stage id, so
  no manifest node resumes or warm-starts from one) and the preparation
  report as its task lineage, whose `child_task_sha256` an adaptation moves
  to the adapted task; preparation requires that fingerprint and refuses one
  whose stage is a stage id, and preparation and adaptation refuse one
  without its `task_sha256` by name, since nothing could resume or adapt the
  artifact. `run.json` and `bundle.json` (`mesozoic.behavior-bundle/v2`)
  record `task_fingerprint`, and `--adapt` compares fingerprint sections
  under the same transition settings. A bundle or checkpoint saved earlier
  records no task fingerprint and is refused by name (D-D9:
  evaluation-only). `stage_config.json` records `command_config` in place of
  the five keys, so the digest golden's 42 `stage_config_view_sha256` lines
  move, a measured consequence of D-D2 and D-D22 (the golden pins each
  stage's recorded view); its behavior section goes from 450 to 132 lines
  (one `task_sha256` per recipe replaces the identity digest and its 318
  source lines), and no reward, plant, policy, recovery, task, gate or
  hyperparameter line moves.
- **One terrain selector in the behavior env, and every terrain recipe
  states its blocks** (#594, consolidation PR-8 (a) of
  `docs/CONSOLIDATION_PLAN_2026_09.md`, 2026-10-04). `SpeciesBehaviorMixin`
  takes a `terrain_sampler` and its `reset()` calls `select_terrain_family`
  itself; the `options={"terrain_family": ...}` override stays, and every
  behavior env has a `terrain_families` property (the sampler's families,
  else its one surface). The sampler gains a sixth family, `terrain_contact`
  (the zero-height heightfield), appended with weight 0, so
  `TerrainSamplerConfig()` and every existing block are unchanged; each
  family sets its own surface on the `[terrain]` map, so a sampler may sit
  on a `mode = "flat"` terrain. The 42 single-template recipes
  (`flat_probability = 0.25`) carry a `[terrain_sampler]` with `flat = 1`
  and their own family at 3, the 12 general ones state
  `terrain_contact = 0`, and a `[terrain_sampler]` table states all six
  families. This is the distribution change the plan's §8 names: those 42
  recipes move from an independent per-episode draw to shuffled blocks of
  four with exactly one plane episode (from the digest harness's reset seed
  1042, episodes 1, 6 and 9 of 0-9 change surface; every other reset of the
  54 terrain recipes starts on the same surface with the same command seed,
  observation and joint positions, and its reset info differs only in its
  `terrain_sampling` record, new in the 42 and with a `terrain_contact: 0`
  weight in the general recipes). Evaluation visits every recipe's families
  in turn (`balanced_round_robin`; the plane alone for `follow_direction`
  and `follow_direction_speed`, which run without terrain), as the
  certification panel already did, so a five-episode evaluation of a
  single-template recipe scores three plane and two terrain episodes.
  `flat_probability` leaves the settings an `--adapt` transition may change.
  Every behavior identity moves (its hashed sources changed), so no earlier
  behavior bundle resumes or adapts; decision D-D9 already made them
  evaluation-only. A behavior env's reset refuses a seed that is not a
  nonnegative integer for every recipe, as the sampled env did, and the
  certification panel refuses a family outside the task with the env's own
  message, "terrain_family must be an enabled family: ..." (for a recipe
  without a sampler it was "Certification requested a terrain outside this
  task"). The digest golden moves only in its behavior section, 932 to 974
  lines (every identity, the behavior env, command and sampler source lines,
  the 54 rewritten recipes' digests, and a sampler source line for each of
  the 42 single-template recipes); no reward, stage, plant, policy or
  recovery line moves.
- **Behavior sidecars are plain `VecNormalize` files whose command
  statistics follow the reseed rule, and the command constants have one
  source** (#594, consolidation PR-8 (b) and (c), decision D-D3,
  2026-10-04).
  Preparation reseeds the three command inputs' statistics to mean 0 /
  variance 1 (count kept), so commands enter the policy at O(1); from then
  on they update, scale and clip like every other input (the recipes plan's
  invariant 8). A sidecar saved before this change pickles the deleted
  passthrough class and is refused with a `BehaviorCheckpointError` naming
  D-D3 (its policy saw other inputs; evaluate it at the commit that trained
  it); the command line's `--resume` and `--adapt`, and the certification
  panel, refuse such a bundle earlier, by its identity or recipe. The
  preparation report's `command_normalization` and `run.json`'s
  `observation_normalization` say so. `direction_commands.py` imports
  `COMMAND_WIDTH`, `COMMAND_COMPONENTS` and `COMMAND_RANGE` from
  `command_frame` (same values); `lateral_speed_scale` stays until PR-11 and
  PR-12 replace these recipes with stage TOMLs (D-D12).
- **The direction and terrain guide records where terrain and the notebook
  stand** (#594, consolidation PR-8; the maintainer's request of
  2026-10-04).
  `docs/TRAIN_DIRECTION_AND_TERRAIN.md` gains a dated status section on the
  deferral (terrain work waits for the walking gait), the terrain blocker,
  the node order, an evaluation cell for Colab and a gentle terrain recipe.
- **The root README is a short front page, and the full species catalog is a
  generated docs page** (#593, cleanup CU-17 of `docs/CLEANUP_PLAN_2026_09.md`,
  2026-10-04; the maintainer's request of 2026-10-03 and choices of
  2026-10-04). `README.md` goes from 536 lines to 134: the title, the
  historical balance GIF with its provenance caption, what the project is,
  install, a headless random-action example, one training command, a
  documentation map, the citation and the license, now stated as "MIT, as
  declared in `pyproject.toml`". Its generated SPECIES block is one row per
  species (description, gait, task, observation and action dimensions, and
  links to the species README, its catalog entry and its model page), and
  every species anchor (`README.md#velociraptor`, `#t-rex`, ...) still
  resolves. Each species' full entry (specifications, stages with their
  recipes, budgets and gates, success definitions) and the published run
  summaries move to `docs/SPECIES_CATALOG.md`, which
  `python -m environments.shared.species_catalog` writes whole and `--check`
  verifies in CI (new: `render_catalog_page`, `render_readme_species_summary`,
  `--catalog-page`); its RESULTS block stays pinned to the D-A10 golden, with
  its four source links one directory up (decision D-A10 amended). The
  velociraptor's external model-card link is dropped. The Quick Start's
  flag reference and notebook settings live in the behavior recipes guide,
  the roadmap and the planned species in `docs/ROADMAP.md`, and the
  development commands in `CONTRIBUTING.md`, which gains a repository layout
  and the list of changes that need the catalog regenerated. The species
  READMEs, ROADMAP and RL_TRAINING_PLAN link the catalog page; the catalog
  JSON is unchanged, and no digest moves.
- **A PR's landing is recorded once, in the consolidation plan's status table
  and in the PR's CHANGELOG entry** (#593, cleanup CU-17 of
  `docs/CLEANUP_PLAN_2026_09.md`, 2026-10-04; the maintainer's ruling of
  2026-10-02). A `docs/README.md` convention states the rule, which governs
  landings from CU-17 on, and `test_landing_records.py` fails a landing record
  of a later PR anywhere else in the living docs. The consolidation plan's
  landed sections (PR-1..PR-7, the notebook-only PR-12 slice, PR-14a..PR-14c)
  point to their full text at commit `20ab100` (`0.3.9`); NEXT_STEPS.md's
  section 1 narrative, "Landed on `main`" table, notebook history and section
  5 restatement of the decisions, the consolidation plan's §4 notebook-size
  narrative and the cleanup plan's header chronicle and §1 items 1 and 2 point
  to `ace8112`; the cleanup plan keeps its §3.1 records as the cleanup's
  history. Moved first, as their only copies: PR-4's reproducibility note and
  PR-7's parity reason (with a dated correction) into their status rows, and
  NEXT_STEPS.md section 5's operational choices of 2026-09-20, the
  coverage-floor rule among them, into its §8. PR-15's goal is restated
  without its done and stale items, and the conventions NEXT_STEPS.md copied
  live once, in `docs/README.md`. Six KNOWN_ISSUES entries over 40 lines drop
  history and dead recipes kept elsewhere (PLANT_CONTRACT.md's widening
  section, the released CHANGELOG, the status table, or the file at
  `20ab100`); every measurement stays in its entry, and no entry is renamed or
  closed. `CONTRIBUTING.md` and `results/README.md`, which the check scans,
  join both path filters of `python-ci.yml`, so a pull request that edits only
  one of them runs it. Docs, one test and those two filter entries; no digest
  moves.
- **The SB3 notebook's resume walk and archive-load preflight are library
  functions, and the RESUME cell no longer evaluates the node it trains**
  (#592, cleanup CU-6 of `docs/CLEANUP_PLAN_2026_09.md`, 2026-10-03; §3.2 CU-6
  row). A behavior change for a resume session: the RESUME cell prints no
  evaluation report, and the stage directory holds no `evaluation_final.csv`
  or `evaluation_selected.csv` until the chain loop's JUDGE branch, next in
  the same Run all, evaluates the node from disk and judges it (since
  ROW-4/6 it does so for every node the RESUME cell trains). The
  infrastructure cell's `train_stage` takes `evaluate=True`; the RESUME cell
  passes `False`, which skips the evaluation of the final and selected
  checkpoints (up to 60 episodes) and keeps the 6-tuple, and the chain
  loop's TRAIN branch and the manual cell evaluate as before. New in
  `environments.shared.curriculum`: `newest_intact_periodic_pair(model_dir,
  name_prefix)`, the RESUME cell's newest-first walk over a node's periodic
  checkpoint pairs, which returns the newest pair `checkpoint_pair_problem`
  passes and the reason for each newer one it skipped. It reads SB3's
  `<prefix>_<steps>_steps.zip` names through
  `policy_loading._PERIODIC_CHECKPOINT_RE`, the pattern the trainer and the
  command line read, instead of the cell's own copy, and picks the pair the
  cell picked (the same names, step order, sidecar names and tie order); the
  cell keeps its texts byte for byte: a WARNING per skipped pair, its two
  refusals and its report. The WARNING lines now print once the walk
  returns instead of during it, so a walk that raises (an I/O error on a
  mount, an interrupt) prints none of them. Only the notebook's copy of the
  pattern is folded: `result_bundle/evidence.py`'s name matcher and
  `prune_periodic_checkpoints`' glob stay as they are. New in
  `environments.shared.policy_loading`: `sb3_archive_load_preflight(species,
  root_reference, *, trunk_dir)`, the body of the section-4 preflight cell,
  which is now one call: it loads the trunk run's root handoff, else a
  throwaway PPO saved into a temporary directory, through `load_sb3_model`,
  prints the cell's two lines, the first flushed immediately before the one
  load, and returns the archive's inspection; `policy_loading` still imports
  only the standard library at import time, and the preflight loads no
  `train_base`. The setup cell's `import numpy as np`, which only the
  throwaway's env read, goes. The notebook is 1,424 source lines (1,480
  before). Tests: one per reason the walk skips a pair, and its name, order,
  sidecar and tie rules; both RESUME refusals executed (no test ran either
  before) and the report order; the preflight executed on a trunk handoff
  under each directory name, on the 3.12 and 3.13 fixture archives and on
  the throwaway, with its flushed line observed before the load starts and a
  failed load propagating; `train_stage` with and without the evaluation;
  and a pin that no notebook cell encodes the periodic checkpoint name
  itself. No digest moves.
- **The SB3 notebook records the trunk a session resolved in the run's
  `trunk_run.json`, and the RESUME cell checks a resume against it**
  (#591, decision 4 (a); cleanup ROW-4/6 of `docs/CLEANUP_PLAN_2026_09.md`,
  2026-10-03; §2 row 4). A behavior change for notebook sessions that
  re-enter a run. The resolve cell now writes `RUN_DIR/trunk_run.json`
  (`mesozoic.trunk-run/v1`: the `TRUNK_FROM` value that reproduces the trunk
  it resolved, `""`, a run id or an absolute run directory, with no
  timestamp) and prints a `Trunk record:` line. It writes nothing into a
  `complete` run, nothing when the session's trunk is the recorded one, and
  nothing once the run holds an `ancestors/` record: those records came
  through the recorded trunk when sections 2-3 ran in order before the chain
  loop (`docs/KNOWN_ISSUES.md` documents the out-of-order case), so a
  session under another trunk gets a WARNING naming the `TRUNK_FROM` to set,
  and a run opened before this change that holds records is never recorded
  with a guess. Before, nothing on disk recorded the trunk `"auto"`
  resolved, and a resume re-supplied it by hand. The RESUME cell, right
  after its complete-run guard and before it trains, refuses three things in
  a run that is not `complete`: a resume of a node the run holds as an
  `ancestors/` record, which stays that node in the run (the chain loop
  takes it while a trunk certifies the node, and a bundle refuses a node
  both reused and trained here), so the resume could never be certified;
  and, once the run holds a record, a resume under another trunk than the
  file names (before, `record_ancestor` refused a second parent only after
  the resume had trained, and `TRUNK_FROM = ""` trained the recorded
  ancestors again) or with a file it cannot read. A run without records
  resumes under any trunk, and one opened before this change keeps the
  resume recipe's manual route. The resume recipe (section 5, step 1) and
  the recipes page set `TRUNK_FROM` from `trunk_run.json`. The file is a
  plain run file: the next bundle write declares and hashes it, and the
  export does not regenerate it, so a complete bundle certifies it. New in
  `environments.shared.result_bundle`: `TRUNK_RECORD_NAME`,
  `TRUNK_RECORD_SCHEMA`, `trunk_from_value`, `read_trunk_record`,
  `record_trunk_run` and `refuse_trunk_other_than_recorded`. Tests: the pins
  move with the cells (the resolve cell's guard ends with the printed
  record; the RESUME cell's resuming branch calls the check right after the
  complete-run guard; the resume prose names `trunk_run.json`); 8 executed
  resolve-cell cases (a run id, `""`, an absolute directory, the `"auto"`
  selection, a rewrite without records, the record kept and warned about
  once records exist, no guess, an unreadable file) and 11 executed
  RESUME-cell cases; the executed helpers bind `LOG_BASE` and `ALGORITHM` as
  the storage and configuration cells do. `docs/KNOWN_ISSUES.md` narrows the
  trunk entry to what stays open (a session that does not resume is only
  warned, and the file names the wrong trunk when the cells ran out of
  order) and adds the file to the reuse-only re-entry entry;
  `docs/RESULT_BUNDLES.md` lists it. No digest moves.
- **The SB3 notebook's chain loop judges a node the run trained but never
  judged before it consults the trunk** (#591, decision 6 (b), amending
  D-A17 and D-C13; cleanup ROW-4/6 of `docs/CLEANUP_PLAN_2026_09.md`,
  2026-10-03; §2 row 6). A behavior change for notebook sessions with a
  trunk run (`TRUNK_FROM`). For an ancestor whose stage directory in
  `RUN_DIR` holds any checkpoint (`models/*.zip`) and no
  `gate_verdict.json`, and that the run does not hold as an `ancestors/`
  record, the loop no longer asks the trunk: an intact final pair goes to
  the JUDGE branch, anything else to the interrupted-node refusal, which
  names `RESUME_STAGE`. Before, a trunk that certified the node stood in for
  it: a node the RESUME cell had finished was reused from the trunk and
  never judged, and an interrupted one was abandoned for the trunk's copy.
  The loop judges such a node only on the parent this session resolved
  (reuse rule 4): one trained on another parent is refused before anything
  is judged or trained, naming a fresh `RUN_ID`. It prints why it skips the
  trunk. A judged node that fails now stops the chain where the trunk's copy
  used to be reused. A run that already took the trunk's copy over its own
  keeps that `ancestors/` record, and the loop still consults the trunk for
  that node. The target, a node `RETRAIN_FROM` covers, a node holding a
  verdict and a session without a trunk behave as before. A root widened on
  the command line is judged first the same way (one widened into a run that
  already holds that root as an `ancestors/` record is refused, with or
  without a trunk, naming a new run id), so the resolve cell no longer calls
  D-C13's `refuse_trunk_over_unjudged_widened_root` (still exported and
  tested). The RESUME cell's refusal under `RETRAIN_FROM`, the resume recipe
  (section 5) and the recipes page no longer route a node through
  `BEHAVIOR`. New in `environments.shared.result_bundle`:
  `unjudged_stage_dir` and `refuse_judging_off_the_resolved_parent`. Tests:
  the notebook pins move with the cells (the resolve cell's guard holds the
  complete-run refusal only, and no cell names D-C13's function; the loop's
  candidates, rule-4 call and order; the `RETRAIN_FROM` message and the
  resume prose), and 16 new executed loop cases run the configuration,
  resolve and chain-loop cells, 14 of them against a trunk that certifies
  every node and two without a trunk; the widen tool's round trip checks the
  judge-first helpers on its real output. `docs/KNOWN_ISSUES.md` lists, as a
  LOW, that the JUDGE branch applies no chain check to the target or without
  a trunk. No digest moves.
- **The command-line curriculum seeds model construction and records each
  node's duration, as `train()` does** (#590, decision D-D11; cleanup CU-10b
  of `docs/CLEANUP_PLAN_2026_09.md`, 2026-10-03; §3.2 CU-10 row). A behavior
  change for CLI curriculum runs. Each node's model is built (the root) or
  its warm start re-seeded (each child) under the run's `--seed`, unless
  the stage's algorithm block names a seed (`--override ppo.seed=N`, which
  is kept), so a run starts from seeded initial weights instead of the
  state the environments' construction left in the global generators.
  Without a block seed, every SB3 checkpoint a node saves records the
  `run.seed` its `stage_config.json` records; with one, the checkpoint
  records the block's seed, as under `train()`. Each trained node's
  `stage_config.json` now records `run.duration_seconds` (D-A15), its time
  from the stage directory to the final save, an interrupted node's
  included, which the curriculum never wrote; `curriculum_results.csv`
  still records the time `learn()` took. `docs/RESULT_BUNDLES.md` and the
  recipes page's run-block table say which runs record the duration.
  `_train_stage_body` now seeds every stage it trains. No certified result
  comes from the CLI curriculum (its verdicts are training-time signals),
  so none changes. Under the next entry's frozen clock, 19 runs before and
  after this change: the 10 `train()` runs (CLI and the notebook's call)
  wrote the same 162 files and log lines; with the curriculum's
  construction seed popped, its 9 runs differ only by the added
  `duration_seconds` in each trained node's `stage_config.json` and in the
  ancestor records that copy one (the SAC run without a block seed too,
  once OS entropy is fixed in both); with it, everything downstream of the
  initial weights moves (the checkpoints, their sidecars, the evaluations,
  the hashes the verdicts and the children record, the CSV's rewards; no
  verdict changes), and a run whose algorithm block names a seed only
  gains the duration. A SAC curriculum run, whose warm-up actions were
  drawn unseeded, now repeats byte for byte. Tests: the curriculum's model
  is built or warm-started under the run's seed and an algorithm-block
  seed is kept; each node records its duration after its final save and
  before its verdict, an interrupted node's too; the curriculum mock
  harness records the durations and refuses one for a node whose config
  it did not save; and the CLI end-to-end ladder checks that every
  checkpoint a node saves records the seed and environment count its
  `stage_config.json` run block records. No digest moves.
- **`train()` and the command-line curriculum train a stage through one
  body, and the evaluation environment's seed has a name** (#590, cleanup
  CU-10b of `docs/CLEANUP_PLAN_2026_09.md`, 2026-10-03; §3.2 CU-10 row).
  `train_curriculum` kept its own copy of `train()`'s stage body; both now
  call one private helper in `train_base.py`, `_train_stage_body`. It
  builds the training and evaluation environments, loads the VecNormalize
  statistics, builds the algorithm kwargs, starts W&B, creates or loads
  the model, sets up a same-stage resume, builds the callbacks, trains,
  saves the final pair and returns both environments open. The curriculum
  hands it `CurriculumCallback`, which still goes after the entropy decay
  and before the stage-entry shaping, and gets back whether a Ctrl-C ended
  the node; `curriculum_results.csv` still records the time `learn()`
  took. What only one side did stays with it: `train()` refuses an
  undeclared parent, writes `metrics.json` and logs its own lines, and the
  curriculum loads its parent's sidecar as before and judges no interrupted
  node; `train()`'s seeded model construction (D-D11) and duration record
  (D-A15) reach the curriculum in the entry above. The new public
  `train_base.eval_env_seed(seed)` returns `seed + 1000`, the evaluation
  environment's seed on both paths; the SB3 notebook's
  `CHECKPOINT_SELECTION_SEED` stays `SEED + 1000`, and a pin checks the two
  agree. Under a frozen clock, 16 runs (CLI `train` from scratch, two
  periodic resumes, `initialize_next_stage`, an interrupt and SAC;
  `train(report_metrics=False, save_on_interrupt=False)` as the notebook
  calls it, from scratch, into the next stage, interrupted and resumed; the
  CLI curriculum's velociraptor ladder, `--trunk-from`, `--retrain-from`, a
  closed gate, an interrupt and compsognathus's stance node) wrote the same
  files with the same bytes and the same log lines before and after, SB3
  archives apart from object addresses, the TensorBoard event files aside
  (the comparison leaves them out because their names carry the process id).
  Tests: ten new ones (both entry points reach the body with their own
  switches, the callback order, the curriculum's callback built where it
  was, the open environments, the interrupt, the learn-only CSV duration,
  `eval_env_seed` on both paths, and a node's model released before the next
  node trains), and the eight source pins that read `train()` or
  `train_curriculum` now read the body and what each caller passes it. No
  digest moves.
- **The Drive summary notebook checks out `REPO_REF`, and its run reader
  is tested** (#589, cleanup CU-15 of `docs/CLEANUP_PLAN_2026_09.md`,
  2026-10-02; §3.2 CU-15 row). `google_drive_summary.ipynb`'s setup cell
  takes the SB3 notebook's `REPO_REF = "main"` knob and its Git block,
  statement for statement, in place of a `--depth 1` clone of the default
  branch that a rerun reused without updating: a `--no-checkout` clone, a
  fetch of `REPO_REF`, `FETCH_HEAD^{commit}`, the refusals after an import
  of the package or over local edits, and a detached checkout. In both
  notebooks the block also checks out a clone that has no index yet, as
  a failed first fetch (a mistyped `REPO_REF`) leaves it; a rerun used to
  print its commit over an empty tree. Drive is
  mounted only when `google.colab` imports and `/content/drive` is not
  mounted yet, and the install cell runs `pip install -e` only when
  `environments` cannot be found, asking before it puts the checkout on
  `sys.path`. The layout text, two docstrings, the `show_full_config`
  example and the strict-validation note describe today's
  `<algorithm>/<run_id>/NN_<stage id>/` runs and the `canonical-valid`
  bundle. Tests: the two Git blocks are pinned AST-identical, the
  ref-change test runs on both notebooks, and the new
  `test_drive_summary_notebook.py`, which the SB3 job lists for pandas,
  runs the mount with `google.colab` absent, unmounted and mounted, the
  install cell with the package found and not, and the reader cells whole
  and in order on a complete, a partial, a failed, a conflicting and an
  interrupted run in today's layout, an older flat run, a March 2026
  sweep and a `sweeps/<algorithm>_<timestamp>/` folder that gives no row
  (`test_sb3_notebook_pins.py` goes from 121 tests to 127; the new module
  has 6). Each fails under a mutant: a changed Git-block statement (the
  pin, and the ref-change test unless it is the clone); a reader that
  skips `<algorithm>/<run_id>/` runs or runs found only by an
  `NN_<stage id>` directory, misroutes a partial or failed bundle,
  renames a summary key, drops the sweep or legacy rows or walks into
  `sweeps/`; the unguarded mount; an unconditional install, or the path
  insert moved back into the setup cell. The `run_index.py` move is not
  taken, and the `library_version` column's two meanings become a
  KNOWN_ISSUES LOW. No library code or digest changes.
- **The recovery stage TOMLs inherit from stance through `extends`**
  (#588, cleanup CU-13 of `docs/CLEANUP_PLAN_2026_09.md`, 2026-10-02;
  §3.2 CU-13 row). A stage TOML may declare a top-level
  `extends = { stage = "stance", tables = [...] }`: each listed table is the
  parent's, with the file's own keys overriding in place or appended, one
  level deep (a nested table such as `[ppo.policy_kwargs]` is replaced
  whole). The parent id resolves through the manifest of the file's own
  species directory, so a copy outside it is refused with a clear error, as
  are `curriculum` or an unknown table in the list, a listed table the
  parent lacks, a parent that itself extends, a missing parent id, a
  malformed value and an `extends` written inside any table, sub-tables and
  arrays of tables included, whether in the file or in its parent. The
  T-Rex, Compsognathus and Compsognathus robot `recovery.toml` files now
  declare only their deltas (the stage name and description, the
  perturbation block and `ent_coef_decay_timesteps`; the T-Rex keeps its
  three `foot_contact_*` keys written out) besides their own
  `[curriculum]`. The T-Rex lists `stage`, `env` and `ppo`, never `sac`, so
  it stays PPO-only; the Compsognathus pair also lists `sac`. All 21
  resolved stage configs are byte-identical, key order included, so no
  digest moves, and the hand-mirror test becomes ordered assertions on the
  resolved tables of all three species. A stance edit to a key its recovery
  stage inherits now moves that stage with it (for the Compsognathus pair
  an `[env]` edit also makes its recovery calibration refuse until
  recalibrated, and for the T-Rex a reward-weight edit means re-measuring
  recovery's pushed-statue collapse reference), as each `stance.toml` now
  says.

### Removed
- **The species aliases, selection by display name, and the per-species
  training wrappers** (D-D33):
  - the aliases `raptor`, `t-rex`, `brachio`, `dibo`, `compso` and
    `compso-robot`;
  - `species_names.resolve_species_id`, `species_display_names` and their
    case and separator folding;
  - `config.SPECIES_NAMES` and `species_registry.SPECIES_FACTORIES` with its
    six per-species factories (`get_species_config` builds the config from
    the model's environment class and stage manifest);
  - the manifest's `aliases` and `variant_of`;
  - the README and catalog alias anchors;
  - `environments/<species>/scripts/train_sb3.py` (five; use
    `python -m environments.shared.train --species … --embodiment …`);
  - the `environments/<species>/envs/` packages and their re-exports;
  - the shared per-species tables `SUPPORT_REGISTRY` and `REWARD_SPECIES`
    and the `gym.register` calls in the env modules (the embodiment classes
    and the manifest carry them);
  - `configs/__init__.py` and `configs/compsognathus{,_robot}/__init__.py`;
  - `view_model --model biological` (the Compsognathus `view_model.py`
    takes `--embodiment anatomical|robot`);
  - the result-summary readers of schema 3 and 4;
  - the task-fingerprint v1 transition valve and the compsognathus
    quiet-task carve-out, with `test_compsognathus_task_compatibility.py`
    and its fixture;
  - the stage-manifest synthesizer for `stage{N}_*.toml` names.
- **`environments/velociraptor/mjx_config.py`, the Velociraptor's frozen
  MJX registration** (#601, decision D-D25, 2026-10-06), with its
  `pyproject.toml` ruff exclusion: the Velociraptor is SB3-only since its
  policy interface revision 11 (the Velociraptor entry under Changed).
- **The five numeric `command_*` kwargs, `SB3_COMMAND_REFUSAL`,
  `behavior_identity`, `canonical_env_parameters`,
  `sampler_source_identity`, `BEHAVIOR_IDENTITY_SCHEMA` and
  `PREPARATION_ATTRIBUTE`** (#595, consolidation PR-9, 2026-10-05). Pass
  `command_config=DirectionCommandConfig(...)` with a live `command_mode`
  (the five names are `command_frame.RETIRED_COMMAND_ENV_KEYS`); a behavior
  env's identity is `env.task_fingerprint`, and the behavior loaders take
  `task_fingerprint=` (was `behavior_identity=`), which
  `prepare_behavior_checkpoint` now requires. The preparation report is the
  checkpoint's `mesozoic_task_lineage` (was
  `mesozoic_behavior_preparation`); `judge_behavior_panel` reads the
  fingerprint's `command.config`, `env.terrain` and `env.course_distance`
  (was the identity's `commands`, `terrain` and `course_distance`); and the
  mixin's `_heading()` and `command_manifest()` are `BaseDinoEnv`'s.
- **`TerrainSamplingMixin`, `get_sampled_behavior_env_class`,
  `_PanelTerrainMixin`, `BehaviorVecNormalize` and `flat_probability`**
  (#594, consolidation PR-8, 2026-10-04). A sampled env is
  `get_behavior_env_class(species)(terrain=..., terrain_sampler=...)` (or
  `train_behaviors.create_behavior_env`), and the certification panel is the
  recipe's own env. `SpeciesBehaviorMixin(flat_probability=...)` is a
  `TypeError`, the attribute and the identity key are gone, and a recipe's
  `[env] flat_probability` is refused with the migration in its message:
  `flat = 1` beside the terrain's own family at 3 is the former 0.25. The
  reset label `"certification_override"` is no longer written (a forced
  reset of a sampled env is `"evaluation_override"`; an env without a
  sampler writes no `terrain_sampling` record).

### Fixed
- **A comparison's run names sit at their lines' ends in the
  training-history report** (#644; report only;
  `environments/shared/reporting/training_history.py`). Each run's name at
  the right of a line panel was placed by a transform read while the
  panel's autoscale was still pending, so on a panel with no J mark or
  alarm line (whose vertical lines happen to apply it first) the name sat
  where its line would be under earlier view limits. Measured with the
  r10 walking parent, the r10 statue and `stance5_LTSB_s70` compared: 36
  of 77 names more than 0.15 of the axes height from their line on
  `08a4d189`, none after the fix. The view limits are now settled first.
- **The full-horizon heading trace test bounds the closed loop it runs**
  (#642; tests only; approved by the maintainer on 2026-10-25;
  `environments/shared/tests/heading_free_helpers.py`,
  `environments/shared/tests/test_heading_free_rollouts.py`; documented in
  `docs/HEADING_INVARIANCE_DESIGN_2026_10.md`, "Migration and acceptance").
  `test_full_horizon_traces_repeat_at_every_heading` passed by chance: on T. rex
  physics r9 its own check, run with one torch thread over seeds 3042–3061,
  failed 7 of 20 seeds (3047, 3048, 3051, 3056, 3057, 3060 and 3061; with four
  threads 5 fail instead, 3044, 3053, 3054, 3057 and 3060), and its seeds
  3042–3043 are among neither. A turn leaves float64 rounding noise of about
  1e-13, which now and then puts an observation element on the other side of a
  float32 rounding boundary than at 0°; the float32 actions then differ in their
  last bits (usually several elements at once, one to twelve, by up to 1.9e-9),
  and the stable closed loop carries a bounded difference to the end (with one
  torch thread joint state up to 6.9e-8 and reward up to 2.1e-8, with four
  9.4e-8 and 2.45e-8; an observation element by one float32 ulp: 3.05e-5 on a
  256–512 N foot force, 6.1e-5 above 512 N). That tripped the 1e-9 reward bound
  the test shared with the 250-step comparison (the zero action and a weak fixed
  linear policy), and its flat 1e-5 observation bound. No dependence on world
  yaw was found (a state audit and a reading of the observation and reward
  code). The test now gives `assert_paired` a closed-loop reward bound,
  `CLOSED_LOOP_REWARD_ATOL = 1e-6` (the precedent of the floor-truth metrics'
  `METRIC_ATOL`), and a float32-aware observation bound: each element may differ
  by `CLOSED_LOOP_OBSERVATION_ULPS = 4` float32 ulps of its 0° value or by 1e-5,
  whichever is larger (`float32_ulp_bound`; a NaN or infinite observation fails
  it). That widens the bound only on elements of magnitude 32 or more, the
  T. rex foot forces (up to 1.2e-4 N on forces of 256–512 N and 2.4e-4 N on
  512–1024 N); the float64 forces behind them differed by at most 0.95 ulp with
  one torch thread (1.03 with four), and a float32 difference beyond four ulps
  needs a float64 one of at least 3.5. A leak moves the observation by O(1) from
  the first step and still fails: in a scratch copy ε·sin(world yaw) added to
  the T. rex projected gravity along z failed the test at each ε tried from 0.1
  to 1e-7 (0.1, then 1e-3 to 1e-7 by decades; at 1e-8 the float32 observation
  does not change), and added to a foot force at 0.1, 1e-3 and 1e-4 N (1e-5 only
  barely), where from 1e-3 down only the observation bound sees it behind this
  policy. Measured on this change over seeds 3042–3061 with one thread: T. rex
  worst actions 1.63e-8, joint state 6.93e-8, reward 2.09e-8 and observation
  3.05e-5 (13 of 20 seeds pass the old bounds, 20 the new; with four threads 15
  and 20); Compsognathus 2.70e-8, 3.20e-7, 5.4e-11 and 9.5e-7 (20 of 20 under
  both). `assert_paired`'s defaults are unchanged, so the 250-step comparison of
  `test_heading_free_observation.py` keeps actions to 1e-6, joint state and
  reward to 1e-9 and the observation to 2e-6. No environment, physics, reward,
  gate, policy interface, seed, thread setting or digest changes.
- **A same-stage resume continues `gate_progress.npz` instead of truncating
  it** (#621; found by the 6M screen). On a resume `seed_resume_eval_state` seeds
  SB3's evaluation history from `evaluations.npz`, and
  `StageAwareEvalCallback.seed_prior_history` pads the per-evaluation capture
  series with an empty row per seeded evaluation. `StageGatePlateauCallback`
  started its own rows from nothing, though, and took the per-actuator rows as
  the first *n* of those series for its session-local row count *n*: every
  resumed write read the padding and left out `action_dc_per_actuator` and
  `action_ac_rms_per_actuator`, and the file held the resumed session's
  evaluations only, without the prior session's gate criteria, pad telemetry
  or leg DC. The plateau callback now reads each evaluation's per-actuator
  rows by its absolute index, and `seed_resume_eval_state` seeds its rows from
  the `gate_progress.npz` beside the `evaluations.npz` it seeds from
  (`seed_prior_gate_progress`): the prior rows stay as they were, the resumed
  evaluations append, and rows after the resumed checkpoint are dropped as the
  evaluation history drops them. A key the prior file lacks, such as #620's
  pad and leg DC keys on a run interrupted before it, reads NaN on the seeded
  rows. The hop watch, which reads those rows, starts past the seeded ones
  like every evaluation cursor. The TensorBoard scalars and `evaluations.npz`
  were already aligned; `diagnostics.npz` is not continued either, an older
  gap now in KNOWN_ISSUES. `test_gate_progress_resume.py` runs in the SB3 job,
  and `test_ci_tool_pins.py` now requires every test file that skips without
  SB3 to be named in a step of that job. Report-only: no gate verdict, reward,
  observation, checkpoint selection, schema or digest changes.
- **`pre-commit run --all-files` modifies nothing on a fresh checkout** (#618;
  with D-D33). `website/static/img/logo.svg` and `logo-dark.svg` ended without a
  final newline, so `end-of-file-fixer` rewrote both on every run (noted in
  `docs/investigations/TREX_REVIEW_2026_07.md`); each now ends with one. The
  hook keeps its scope, and no digest data file changes.
- **Statistics or a checkpoint of another observation width are refused by
  plant, not by SB3's shape errors** (#616; with the heading-free observation,
  decision D-D31). SB3's `VecNormalize.load` re-binds the unpickled wrapper to
  the env (`set_venv`), asserting equal observation shapes before any caller
  can read the plant identity pickled with it, and `PPO.load` rebuilds a model
  around the training env before `train_base` validated it, so a pre-migration
  pair was refused with "spaces must have the same shape"
  (`load_vecnorm_stats`, `evaluation`), "Observation spaces do not match" (the
  trainer) or, through `load_sb3_checkpoint`, a misleading "Re-copy the file
  in binary mode", none naming the plant. `policy_loading.load_vecnormalize`
  is `VecNormalize.load` that, when the bind fails and a plant identity is
  given, reads the pickled identity and raises the plant contract's
  `PlantCompatibilityError` (naming `policy_interface_revision` and
  `observation_dim`, or a pre-contract file's missing identity); a bind
  failure the identity does not explain is re-raised unchanged and a
  successful load is returned for the caller to validate as before. The
  trainer's statistics loader, both evaluation loads, `load_sb3_checkpoint`
  and the behavior preparation and resume (`behavior_checkpoint`) use it, and
  `load_sb3_checkpoint` no longer re-labels a plant refusal as a read error;
  `train_base._create_or_load_model` checks the identity the archive's `data`
  member recorded before SB3 loads the model. It keeps `VecNormalize.load` as
  the loading call, so the tests that stub it still exercise the real call
  sites.
- **Suppress unsupported held-pose ablation classifications** (#606). The report now
  requires a `hold_all` control that does not stand — a partial or complete
  full-pose fall to explain — and a standing `hold_zero` control before
  attributing a fall to actuator groups. A full held pose that stands reports
  classification as not applicable, with the reason and raw variant
  measurements retained; one that falls in only some episodes is still
  classified. Missing, duplicate or invalid controls, and a statue control that
  does not stand, also suppress causal labels. An unmeasured variant cell
  renders as `-` instead of raising. The diagnostic JSON is
  `mesozoic.constant-hold-ablation/v2`: it records classification applicability
  and uses `null` for unassessed necessity/sufficiency claims; incomplete or
  inconclusive group pairs cannot support an overall causal conclusion. No
  training, stance-gate or plant identity changes.

- **The bundled plant manifest resolves an SB3-only species** (#601, with
  the Velociraptor revision, decision D-D25, 2026-10-06). When the
  repository's species manifest is absent, `current_plant_identity` builds
  the species entries from the bundled `plant_manifest.generated.json`,
  which records no training backends, and checked each environment against
  the dual-backend default it never declared, so an SB3-only species raised
  "backends differ": both compsognathus plants did, untested. A bundled
  entry now declares none (`training_backends = None`) and the environment's
  own `supported_training_backends` stands (`plant_contract/versions.py`,
  `manifest.py`); neither file feeds a digest, and
  `test_runtime_identity_falls_back_to_bundled_manifest` covers
  velociraptor, trex and both compsognathus plants.
- **The command-line curriculum judges a `stance_quality/v1` stage at an
  overridden `env.max_episode_steps`** (#587, cleanup CU-10a of
  `docs/CLEANUP_PLAN_2026_09.md`, 2026-10-02; §3.2 CU-10 row).
  `CurriculumCallback._eval_horizon` took the horizon from the stage TOML
  that `CurriculumManager` re-reads, while the evaluation env ran the
  overridden stage config. With `curriculum --override
  env.max_episode_steps=32` no 32-step episode reached the TOML's 1,000-step
  horizon, none supplied an unsupported duty, and the gate could not pass;
  an override above the TOML's horizon counted an episode that ended between
  the two as full (the KNOWN_ISSUES entry, now deleted).
  `CurriculumCallback` gains an optional `eval_horizon`, after its existing
  parameters (default `None`: the TOML's horizon, as before), and
  `train_curriculum` passes the overridden stage config's
  `max_episode_steps`, the expression the in-training gate-progress
  diagnostics already use. The notebook and `train --stage` build no
  `CurriculumCallback` and are unaffected. Tests: the callback given 32 or
  2,000 under a 1,000-step TOML uses it, and unset reads the TOML; 40
  episodes of 32 steps all count as full, and under 2,000 an episode ending
  at 1,500 does not; the constructor takes it after the existing parameters;
  and an end-to-end CLI run of compsognathus's stance node at 32 steps
  (`--target stance`), with the settling window, the reward rail and
  `required_consecutive` (1, for its single evaluation) overridden, writes a
  passed `gate_verdict.json`. `train_curriculum` without the argument fails
  the end-to-end test, and `_eval_horizon` reading the TOML again fails it
  and five of the six unit tests. No digest moves.

## [0.3.9] - 2026-10-02 — Backend Retirement & Cleanup

### Added
- **A golden of reward, info and termination values** (#573, cleanup CU-11 of
  `docs/CLEANUP_PLAN_2026_09.md`, 2026-09-30; §3.2 CU-11 row, §4.4). The
  digest snapshot gains a `reward` section: for each of the 21 stages, a
  fixed capture (a noisy roll that ends in a held kick, an unseeded second
  reset, zero action through the pushes of the recovery stages, one-step
  probes, and twelve posed states scored against the stage's own
  thresholds) becomes four lines of `configs/digest_snapshot.generated.txt`:
  a summary (steps, end and reward sum per part), the ends of the posed
  states, a digest of the discrete records and one of every value rounded
  to 6 decimals. CI's existing digest step checks them on every event, so a
  change that moves a reward, info or termination value by about 1e-6 or
  more, an end or a reason fails its own pull request unless it
  regenerates the golden; the rounded lines cannot see an ulp. `--exact` prints the same captures, and
  every behavior recipe's, bit-exact, for a same-machine base/head `diff`:
  the acceptance of refactors that claim to move no number. The first 848
  lines of the golden are unchanged.
- **The digest snapshot is a committed golden that CI checks** (#571,
  cleanup ROW-16, D-D22, 2026-09-29; `docs/CLEANUP_PLAN_2026_09.md` §2 row 16,
  §3.1 item 4 and §4.4). `configs/digest_snapshot.generated.txt` holds the
  full output of `environments/shared/harnesses/digest_snapshot.py`, 848
  lines: every plant, policy, stage, recovery and behavior digest. The
  plant-contract job's new step, "Verify the digest snapshot golden", runs
  the full harness (never `--skip-behaviors`) with `--check` on every
  event, so a change that moves only behavior identities fails its own pull
  request. The harness gains `--check [PATH]`, which names every moved line
  and prints the diff and the regeneration command, and `--write [PATH]`,
  which refuses a run with an ERROR line. A PR that moves a digest on
  purpose runs `python -m environments.shared.harnesses.digest_snapshot
  --block-optional-backends --write` and commits the golden in its own
  diff (`docs/PLANT_CONTRACT.md`, `CONTRIBUTING.md`);
  `test_plant_contract_digest_snapshot.py` pins the whole CI step, the
  golden's shape and the `--check`/`--write` wiring; `.gitattributes`,
  which sets the checkout's line endings, joins both path filters, since
  the digests hash file bytes; and the readers of `python-ci.yml` move from
  `test_ci_tool_pins.py` into `ci_workflow_helpers.py`. No digest moves.
- **Decision D-D22 recorded; D-D21 amended (CU-7, CU-8, CU-14 and CU-16
  split)** (#569, cleanup CU-4, 2026-09-29;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2,
  `docs/CONSOLIDATION_PLAN_2026_09.md` §6, `docs/CLEANUP_PLAN_2026_09.md`
  §2–§3, `docs/NEXT_STEPS.md` §5). The maintainer took the cleanup plan's
  decision 16 as D-D22: the output of the digest-snapshot harness is
  committed as a golden and checked in CI's plant-contract job, with the
  full run, behavior identities included, on pull requests rather than
  `--skip-behaviors`. A PR of its own builds the check, before CU-7b,
  CU-8a, CU-12 and CU-13. Four gate PRs are split: CU-7 into CU-7a (dead
  code and import cost) and CU-7b (retired-backend wording, with CU-16's
  stage-TOML comments folded in), CU-7's regex and validators going to
  CU-8c; CU-8 into CU-8a (one derivation), CU-8b (`policy_loading` owns
  the SB3 import and the sidecar resolver) and CU-8c (one reader, one root,
  one regex, one set of validators); CU-14 into CU-14b (the per-process
  plant-identity cache, the plan's decision 10 (c)), which comes first,
  and CU-14a (the workflow YAML); and CU-16 into CU-16a (docs text) and
  CU-16b (orphan assets). D-D21's gate names cover every part, so the gate
  is unchanged.
- **A gait audit of the certified nodes and a gait-quality plan** (#567,
  gait PR-G0 of `docs/GAIT_QUALITY_PLAN_2026_09.md`, docs only;
  `docs/investigations/GAIT_AUDIT_2026_09.md` and its evidence directory
  `docs/investigations/gait_2026_09/`). The audit note records CPU replays,
  on 2026-09-28, of 15 nodes, every certified node on Drive (12) among them,
  with contact read both from the touch sensors and from the floor's normal
  force: three of the five certified walkers hop (trex seed 42
  `20260914_123816` and seed 44 `20260925_033501`, and compsognathus_robot
  `20260924_031815`), while only compsognathus `20260921_203149` walks and
  velociraptor `20260922_125248` runs; the zero-action statue passes all six
  stance gates; the robot's touch sensors count sole-on-sole force as
  support; and the in-training dibothrosuchus re-run `20260928_012318` is a
  three-legged skid at 4.3M steps that clears its gate as written. The
  evidence directory holds the audit's hand-run probe `gait_probe.py`
  (imported by nothing), `gait_audit_2026_09.csv` (one row per audited node
  and checkpoint) and `SHA256SUMS`, which pins the 153 JSON, trace, plot and
  contact-sheet files the audit produced; those files stay outside the
  repository. The plan proposes measuring gait on floor contact, per-episode
  gates (`locomotion_gait/v1` first), per-species reward revisions and a PR
  order; its 18 decisions (GQ-1..GQ-18) are all open, and its code PRs build
  on 0.3.9. New KNOWN_ISSUES entries: no locomotion gate reads a foot
  contact, compsognathus_robot's stacked feet and phantom touch support, and
  stance gates that admit non-stances (HIGH); a lunge-then-fall passing on
  per-episode means, a recovery safe set without a support clause, hunt
  success counted on a step that ends in a fall, the brachiosaurus statue
  reaching the food in 11 of 40 hunt episodes, and knee, shin and
  proximal-tail floor contact that never ends an episode on four species
  (MEDIUM); the stance diagnostic reading only the forefeet on quadrupeds,
  and `zero_action_baseline.py` judging a `stance_quality` stage by its
  reward rail alone (LOW); and dated updates to existing entries. No
  library, test or configuration file changes, and no digest moves.
- **A pin test for the frozen MJX interface core** (#566, cleanup PR-B,
  decision D-D17;
  `environments/shared/tests/test_plant_contract_frozen_mjx.py`).
  Its 15 tests pin the token digests of the seven anchor functions whose
  tokens the policy-interface digests of trex, velociraptor, brachiosaurus
  and dibothrosuchus hash, and of the two `action_filter` low-pass
  functions that trex's digest also hashes (9 tests); each `mjx_config.py`
  registration's exact key set (trex 9, brachiosaurus 8, velociraptor and
  dibothrosuchus 7; 4 tests); that the species declaring `jax-mjx` in
  `configs/species_manifest.toml` are exactly those four (the dual-species
  tripwire); and the `[tool.ruff]` exclusion that keeps ruff off the core.
  An edit fails naming the function and D-D17 rather than as a stale plant
  manifest. The plant-contract job ran it through its
  `test_plant_contract_*.py` glob until cleanup CU-14a dropped that job's
  pytest step (Changed, below); the three `test (<python>, shared)` legs
  run it, and it is deleted with the core. `test_config.py` also gains
  `test_a_retired_jax_table_is_rejected` (Migration, below).
- **The `[curriculum]` schema's fail-closed checks keep their tests**
  (#566, cleanup PR-B;
  `environments/shared/tests/test_gate_schema_validation.py`).
  The refusal of an unknown `gate_kind` and the validation of a
  `[curriculum.jax]` override table run on every SB3 curriculum run but were
  tested only through the JAX curriculum; 11 tests now drive them through
  `validate_gate_config`, `thresholds_from_configs` and `load_stage_config`.
- **D-D17's remaining amendments appended** (#566, cleanup PR-B, 2026-09-28;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6–§8, `docs/CONSOLIDATION_PLAN_2026_09.md`
  §3 and §5–§8). A7 now carries "Narrowed by D-D17", and D-A5, D-B13, D-C3,
  D-C4, D-C7, D-C16 and G1 (in both plans) "Amended by D-D17"; with PR-A's
  five amendments and A6, every row D-D17 names now carries its note. New
  paragraphs follow the recipes plan's MJX risk and its invariant 9, and the
  consolidation plan's PR-3b, PR-9 and three §8 risks record what the
  retirement changed.
- **An end-to-end test of the command-line curriculum path** (#565, cleanup
  PR-A2, decision D-D17;
  `environments/shared/tests/test_curriculum_cli_end_to_end.py`). It runs
  velociraptor's `train_sb3.py curriculum` in a subprocess with 64-step
  stages and every gate opened through `--override`, and checks the stage
  directories and records, the hash-bound gate verdicts, the handoff
  lineage, `curriculum_results.csv`, the `ancestors/` records of a
  `--trunk-from` run and a ladder stopped by a closed gate. It lands before
  the GCS upload removal and passes before and after it, and it runs in
  CI's SB3 job. Sizing it found a new KNOWN_ISSUES entry: the command-line
  curriculum judges `stance_quality/v1`'s full horizon against the stage
  TOML, not an overridden `env.max_episode_steps`; cleanup CU-10 fixes it.
- **Decision D-D21 recorded; the archive points settled; D-D17's
  amendments appended** (#564, cleanup PR-A, 2026-09-27;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.1–§6.2,
  `docs/CONSOLIDATION_PLAN_2026_09.md` §6, `docs/CLEANUP_PLAN_2026_09.md` §2,
  `docs/NEXT_STEPS.md` §5). The release cut of D-D19 landed as #563, and the
  maintainer tagged its first commit, `afad625`, as `0.3.8` (a lightweight
  tag, without the `v` that the `[0.3.8]` entries below anticipate) and
  published it as a GitHub pre-release. The maintainer then settled the
  cleanup plan's decision 2 as (c): no archive tags. The code this section
  removes is recoverable from the `0.3.8` tag and from git history. D-D21
  (the plan's decision 20): 0.3.9 is the clean, refactored base release that
  the consolidation work builds on, and `0.3.8` stays the pre-cleanup
  release. 0.3.9 is cut once thirteen PRs have landed: PR-A, PR-A2, PR-B,
  CU-2, CU-4, CU-5, CU-7, CU-8, CU-9, CU-11, CU-12, CU-14 and CU-16. CU-6,
  CU-10, CU-13, CU-15, CU-17 and consolidation PR-8..PR-15 are deferred, not
  dropped. D-A12, D-A15, D-B1, D-B12 and D-D11 now carry "Amended by D-D17",
  and A6 "Superseded by D-D17".

### Migration
- **`from environments.shared import <name>` no longer works for the 19
  names the package re-exported** (#572, cleanup CU-7a of
  `docs/CLEANUP_PLAN_2026_09.md`): `CurriculumManager`, `BaseDinoEnv`,
  `load_stage_config`, `save_result_bundle` and the rest now raise
  `ImportError`. Import each from its submodule, e.g. `from
  environments.shared.config import load_stage_config`; nothing in the
  repository or the notebooks imported them through the package.
  `environments.register_all()`, a no-op, is gone too: the environments
  register on `import environments`, as before.
- **The `jax` and `jax-cpu` extras are gone** (#566, cleanup PR-B, decision
  D-D17), and `[all]` is `[train,viz,dev]`. pip only warns ("does not
  provide the extra 'jax'") and installs the package anyway, so an install
  line that still names `jax` or `jax-cpu` succeeds without JAX, Flax,
  Optax or `mujoco-mjx`; drop the name. The plant contract's MJX probe needs
  none of them (CI's plant-contract job installs no JAX).
- **A `[jax]` table in a stage TOML is now refused.** `load_stage_config`
  accepts only `[stage]`, `[env]`, `[ppo]`, `[sac]` and `[curriculum]` and
  raises `ValueError` ("unknown top-level table(s) ['jax']; …") on anything
  else, as it already did for a misspelled table. The 12 committed stage
  TOMLs lost their `[jax]` and `[jax.policy_kwargs]` tables in this PR;
  delete them from any copy of your own. Their `foot_contact_*` `[env]`
  keys stay, because `task_sha256` hashes them, and a `[curriculum.jax]`
  override table is still validated.
- **No JAX notebook, trainer or evaluator.** `notebooks/jax_training.ipynb`,
  `python -m environments.shared.jax_training` and the other
  `environments/shared/jax_*` modules (all but the frozen `jax_setup.py`)
  are removed, and so are `reporting.save_jax_stage_artifacts` and
  `plant_contract.validate_mjx_environment_plant`. Train and evaluate with
  the SB3 notebook or `train_sb3.py`. To look at the old code, check it out
  from the tag, e.g. `git checkout 0.3.8 -- notebooks/jax_training.ipynb`:
  every file this PR deletes except `website/docs/training/jax.md` is
  byte-identical there, and `jax.md` as cleanup PR-A2 left it is in this
  PR's first parent. The site's `/docs/training/jax` page now returns 404;
  the site has no redirect plugin.
- **Drive:** nothing to move. The JAX run folders stay (the cleanup plan's
  §4.8 counts 27, 2026-03-30..04-03, none a certified parent), and their
  records still validate: `result_schema` still allows the `jax-mjx`
  training backend, legacy `JAX_PPO` labels still normalise, and the Drive
  summary still reads them.
- **`curriculum --gcs-bucket` / `--gcs-project` are gone** (#565, cleanup
  PR-A2, decision D-D17): argparse exits 2 with "unrecognized arguments".
  `train_curriculum()` loses its `gcs_bucket` / `gcs_project` parameters,
  and `config.upload_curriculum_artifacts` is removed. To keep a copy in
  Cloud Storage, write the run to a fresh `--output-dir` on a mounted bucket
  or copy it afterwards (e.g. `gcloud storage cp -r`); a run under `/gcs/…`
  still buffers TensorBoard locally and stages its checkpoint pairs.
- **`reporting.write_results_csv` refuses `gs://` paths** with a
  `ValueError`, in batch and append mode. Before, batch mode uploaded the
  file and append mode raised a different error.
- **The `[gcp]` extra is gone**, and `[all]` is `[train,jax,viz,dev]`
  (cleanup PR-B then drops `jax`, above). pip only warns about an extra the
  package no longer provides; install `google-cloud-storage` or
  `google-cloud-aiplatform` yourself if you need them.
- **No Docker image and no Vertex AI route.** The `Dockerfile`,
  `.dockerignore`, `scripts/setup_vertex_ai.sh` and
  `website/docs/training/vertex-ai.md` are removed, and the site's
  `/docs/training/vertex-ai` page now returns 404. Train locally or with the
  SB3 notebook. The image files are byte-identical at the `0.3.8` tag (e.g.
  `git checkout 0.3.8 -- Dockerfile .dockerignore`); the script and the page
  are in git history. Cloud resources the script created are untouched:
  its Artifact Registry repository (`mesozoic-labs` by default, image
  `trainer`) and any bucket it used; delete them if unused.
- **The `ray` and `mjlab` extras are gone, and `train` no longer installs
  `cloudml-hypertune`** (#564, cleanup PR-A, decision D-D17). `[all]` became
  `[train,jax,viz,gcp,dev]` (cleanup PR-A2 then drops `gcp`, and cleanup
  PR-B `jax`, above). pip only warns about an extra the package no longer
  provides, so an install line that still names `ray` or `mjlab` succeeds
  without them; drop the name (CI's SB3 job installs `.[train,test,viz]`).
- **No sweep command or sweep notebook.** `python -m
  environments.shared.scripts.sweep`, `notebooks/ray_tune_sweep.ipynb` and
  the `configs/<species>/sweep_{ppo,sac}.json` search spaces are removed.
  Tune by running one job per candidate with `--override` (the tuning
  workflow in `website/docs/training/hyperparameters.md`). To look at the old
  code, check it out from the tag, e.g. `git checkout 0.3.8 --
  notebooks/ray_tune_sweep.ipynb`. The site's `/docs/training/sweeps` page
  now returns 404; the site has no redirect plugin.
- **Drive:** nothing to move. The March 2026 sweep folders stay, and the Drive
  summary still reads them; no new sweep folders are written.

### Changed
- **One `stage_config.json` reader, one repository root, one sha256
  pattern and one set of record field validators** (#583, cleanup CU-8c
  of `docs/CLEANUP_PLAN_2026_09.md`, 2026-10-01; §3.2 CU-8 row).
  `config.read_recorded_stage_config`, over the new
  `file_io.read_json_object`, gives the nine lenient readers' answers from
  one place. `save_result_bundle`, `audit_result_bundle` and
  `validate_evaluation_evidence` now refuse a `stage_config.json` that is
  not UTF-8 with their own error instead of a raw `UnicodeDecodeError`.
  `environments/shared/paths.py` defines the repository root once
  (`plant_contract.constants.REPOSITORY_ROOT` stays the patch point), and
  `environments/shared/record_fields.py` holds the sha256 pattern and the
  field validators that `result_schema` and `species_catalog` bind; the
  catalog's number message ("must be null or a finite number") becomes
  the schema's two ("must be a number or null", "must be finite or
  null"). No digest moves. `docs/KNOWN_ISSUES.md` records the same
  non-UTF-8 leak on several of the run directory's other JSON reads (the
  summary, manifest, plant identity, gate resolution, provenance and
  zero-action baseline), which stays open.
- **`policy_loading` owns the SB3 import helper and the VecNormalize sidecar
  resolver** (#582, cleanup CU-8b of `docs/CLEANUP_PLAN_2026_09.md`,
  2026-10-01; §3.2 CU-8 row). `_ensure_sb3`, `_resolve_vecnorm_sidecar` and the
  periodic-checkpoint pattern move from `train_base` with their bodies
  unchanged and keep their private names; `train_base` still exposes them
  as aliases for notebooks saved before PR-14c, and every library caller
  looks them up in `policy_loading` when called, so one patch reaches all
  of them.
  `resolve_vecnorm_path` no longer imports `train_base`, so resolving a
  sidecar no longer loads SB3 and torch. No digest moves.
  `docs/KNOWN_ISSUES.md` records a gap found on the way: the
  final-checkpoint replay video runs on raw observations, with no warning,
  when the final pair's sidecar is missing.
- **CI runs six test jobs instead of eighteen, and pull requests skip the
  robot's six real-training runs** (#581, cleanup CU-14a of
  `docs/CLEANUP_PLAN_2026_09.md`, 2026-10-01; §3.2 CU-14 row). The `test`
  matrix is three Pythons × {`shared`, `species`}: the `species` leg runs
  the five species test directories in one pytest run, so the checks named
  `test (<python>, <species>)` become `test (<python>, species)`, and CI
  runs 10 jobs instead of 22. Both triggers share one anchored path list,
  the matrix installs the `test` extra instead of `dev`, the plant-contract
  job no longer re-runs its tests (the shared legs run them), and
  `test_phase_c_interface.py` leaves the SB3 job. Pull requests and pushes
  skip the six compsognathus_robot parametrisations of
  `test_compsognathus_training.py` (the cleanup plan's decision 10 (a),
  settled by the maintainer on 2026-09-30); the nightly schedule, a manual
  dispatch and the `full-ci` label still run all twelve. A new species
  adds its test directory to the `species` suite and to `testpaths`, which
  a test ties together.
- **The four dual species share their reward, contact, termination and
  keyframe code** (#580, cleanup CU-12 of `docs/CLEANUP_PLAN_2026_09.md`,
  2026-10-01; §3.2 CU-12 row). `BaseDinoEnv._progress_terms`,
  `_nosedive_term`, `_heading_terms` and `_speed_terms` replace the eight
  thin `_compute_*` wrappers, `_contact_geom` is the one contact query (the
  T. rex bite, the velociraptor strike, the dibothrosuchus snap and the
  live floor scan), `_root_termination` the one height/tilt/nosedive
  prefix, and `_cache_home_keyframe` the one home-keyframe setup. The
  T. rex and Brachiosaurus `_foot_contact_forces` overrides are deleted:
  the base sums each foot group in the order they used, so every value is
  bit-identical. `TRexEnv._foot_load_imbalance`, unused since PR-B, is
  deleted, and comments that described the retired MJX runtime as present
  are reworded. Compsognathus is unchanged. No number moves (`--exact`
  prints the same 32,389 lines on the base and on this PR), but the four
  species' behavior identities do, since they hash their env files'
  bytes: the digest golden's 88 `behavior` lines are regenerated, and
  their pilot behavior bundles (evaluation-only, D-D9) stop certifying.
- **One stage-level task-fingerprint derivation** (#579, cleanup CU-8a of
  `docs/CLEANUP_PLAN_2026_09.md`, 2026-09-30; §3.2 CU-8 row).
  `task_fingerprint.stage_task_fingerprint(species, stage, *,
  stage_config=None, env_kwargs=None, plant_identity=None)` is how training,
  the curriculum walk, `select_trunk`, the recovery freeze, the recovery
  calibration and its restamp tool, `widen_checkpoint` and the SB3
  notebook's chain loop derive a stage's task fingerprint. It names the
  stage by its manifest reference, whatever form it is given in, and
  refuses a stage the manifest does not name. `FINGERPRINT_BACKEND` is
  defined once, there; the recovery freeze and `widen_checkpoint` no longer
  define their own, and `select_trunk`'s `backend` argument now governs only
  the `provenance.json` comparison. The ignored-edits check is public as
  `config.ignored_hyperparameter_edits`. `derive_stage_task_fingerprint`
  and `hyperparameter_diff` stay public and unchanged. No digest moves:
  every recorded `task_sha256` is reproduced, CI's digest command reports
  the golden current (932 lines, 0 errors), and a new test ties the helper
  to each stage's golden digest.
- **The landing page's roadmap agrees with ROADMAP** (#578, cleanup CU-16b
  of `docs/CLEANUP_PLAN_2026_09.md`, 2026-09-30; §3.2 CU-16 row). Phases 2
  and 3 read IN PROGRESS, as `docs/ROADMAP.md` has them; the terrain and
  turning items say piloted (#540/#541), and Phase 3's new-species item says
  two of four are delivered (Dibothrosuchus and Compsognathus).
  `docs/WEBSITE_PLAN.md`'s status and its `docs/README.md` row say the two
  apex GIFs stay.
- **The docs record what became of the 2026-08 gap review's findings, and
  stale living-doc text is corrected** (#577, cleanup CU-16a of
  `docs/CLEANUP_PLAN_2026_09.md`, 2026-09-30; §3.2 CU-16 row).
  `docs/reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md` gains an appended status
  appendix for all 120 of its ids (78 fixed, 16 fixed and then retired with
  their code by D-D17, 9 retired without a fix, 3 fixed with a residue, 6
  duplicates, 8 open). The eight open findings (TC8, SS3 with SS4, EP1, EP2,
  EP4, NB9, OP9), the residues SM7, DU4 and CI8 (a `trex.xml` pointer, a
  `stance_quality/v1` stage's 10-episode fallback, a site deploy that does
  not wait on python-ci) and the website logo's size are KNOWN_ISSUES entries,
  each checked against the code (EP1 and EP4 executed), and KNOWN_ISSUES'
  "Sweeps / infrastructure" section is now "Infrastructure". Corrected: the
  API overview lists all six environment ids; the quick start names the
  backfill tool by module; the recipes plan's node vocabulary no longer
  lists a `[jax]` table, and a dated note marks its retired-backend design
  text; `RESULT_BUNDLES.md` describes the `jax_ppo/` path as the home of
  runs recorded before D-D17; the pilot and certificate sentences no longer
  say such output is on Drive; ROADMAP, the README's roadmap and planned
  species, and the status of the stage-1 split plan (implemented), the
  recommendations (a dated snapshot) and the website plan (complete) are
  brought up to date. Docs only: no code, configuration, test or asset
  changes, so no digest moves.
- **Code comments, docstrings and log messages no longer describe the retired
  backends as present** (#576, cleanup CU-7b of
  `docs/CLEANUP_PLAN_2026_09.md`, 2026-09-30; §3.2 CU-7 and CU-16 rows). The
  wording PR-A, PR-A2 and PR-B left about the tuning sweeps, Ray Tune, the
  Vertex AI route and its HPT report, GCS as the only mount and the JAX/MJX
  trainer now describes the code as it is, in the library, the tests and the
  comments of the stage TOMLs (no key, value or line count changes there; the
  `foot_contact_*` keys, which only the retired MJX reward read, stay, as the
  cleanup plan keeps them, §4.9). Log messages change with it: the
  post-training metrics log `Recorded metric: ...` (was
  `HPT metric reported: ...`), the checkpoint evaluation
  `Post-training eval: ...` (was `HPT eval: ...`), and a failed TensorBoard
  sync logs `TensorBoard sync to the remote mount failed.` or
  `Periodic TensorBoard sync to the remote mount failed.` (was
  `... to GCS failed.`); a stage TOML without `[ppo]` or `[sac]` warns that
  SB3, not "every backend", will train on library defaults, and the
  task-success evidence warning names `metrics.json` instead of a sweep row.
  The species catalog drops its check of the deleted `jax_training` notebook,
  which the unknown-notebook check already covered, and a new test fails any
  line of the library, the tests, the stage configuration, the SB3 notebook or
  the build and CI files that names Vertex, Ray or HPT without marking it
  retired (the docs and the Drive summary notebook, which still reads legacy
  sweep folders, are not scanned). The species env files keep their MJX
  comments, since behavior identities hash their bytes. No digest moves.
- **The SB3 notebook's text is cut and corrected, and its dead parameters
  are gone** (#575, cleanup CU-5 of `docs/CLEANUP_PLAN_2026_09.md`,
  2026-09-30; §3.2 CU-5 row). The markdown goes from 223 to 111 source
  lines, keeping every pinned phrase and D-D16 rule (a new pin reads the
  resume recipe's trunk-recovery and D-D16 text). Fixed: the quote of a
  `configs/trex/stance.toml` comment that no longer exists, "all four"
  species (there are six), a stale "review F3" note, and the chain loop's
  branch labels (now 1 REUSE, 2 JUDGE, 3 TRAIN, the loop's and section 6's
  order). Gone: the plotting wrappers' `save_path`, `save_dir`, `_show` and
  unused `fig1, fig2`, `write_training_summary`'s `species=None` and
  `save_run_bundle`'s `run_dir=None`; `train_stage`'s `run_dir` is required
  (its fallback would have trained a node outside the run). `node_budget()`
  is the one budget derivation of the chain loop, the RESUME cell and the
  manual cell. 33 cells, 1,599 → 1,468 lines; no library, configuration or
  digest change.
- **Coverage counts the certification code** (#574, cleanup CU-9 of
  `docs/CLEANUP_PLAN_2026_09.md`, 2026-09-30; §3.2 CU-9 row).
  `[tool.coverage.run] omit` names the hand-run and one-off files one by
  one, each with its reason, instead of omitting every `scripts/` and
  `harnesses/` file; the recovery gate's freeze producer,
  `widen_checkpoint.py`, `backfill_gate_verdict.py`,
  `restamp_recovery_calibration.py` and `calibrate_recovery.py`, and the
  statue baselines `zero_action_baseline.py` and
  `stance_quality_baseline.py`, now count; `digest_snapshot.py` and the
  viewers stay omitted; and `test_coverage_config.py` pins the list and
  that nothing else (a report-time omit, an `include`, another coverage
  config file) takes a file out of the union. `brace_controller` and
  `BRACE_SETTLE_SEEDS`/`BRACE_SETTLE_STEPS` moved from
  `harnesses/recovery_offdist_panel.py` to `recovery_evaluation.py`, which
  breaks the import cycle between the freeze producer and that panel.
  Measured locally, the union reads 19,626 statements, 2,405 missed,
  87.75 percent (91.24 before). No digest moves.
- **`import environments` no longer loads Stable-Baselines3, torch and
  wandb** (#572, cleanup CU-7a of
  `docs/CLEANUP_PLAN_2026_09.md`).
  `environments/shared/__init__.py` re-exported 19 names that nothing
  imported through the package, and importing any `environments.shared`
  submodule runs it first, as every species environment does. It now holds
  only its docstring. Measured with SB3 installed, `import environments`
  takes about 0.36 s and loads 569 modules, against 3.45-3.55 s and 2,480;
  without SB3 the time is unchanged within noise. The usage docstrings of
  seven species scripts, whose old `python <name>.py` usage fails without
  an installed package, now give `python -m
  environments.<species>.scripts.<name>`.
- **The plant identity is built once per process and species** (#570, CU-14b of
  `docs/CLEANUP_PLAN_2026_09.md`, the plan's decision 10 (c)).
  `current_plant_identity` built an environment, compiled the declared model
  and fingerprinted both on every call, seconds each for the mesh-heavy
  `compsognathus_robot`, several times per training test and once per
  behavior env built. It now keeps one build per species for the life of
  the process. Every call, hit or miss, still re-reads `configs/`,
  re-hashes the model's source closure, re-runs the policy-interface layer
  on a fresh env and, unless `verify_generated=False`, compares the result
  with the committed plant manifest; a hit reuses the declared model's
  compilation and the physics and visual fingerprints. A build is reused only while its
  key is unchanged (the species entry, the plant version, the repository
  root, the resolved model path, the source-closure digest, the environment
  class and the MuJoCo version) and the policy layer still matches, and only
  a build equal to the committed manifest entry, hashed from the bytes the
  key read, is stored, so a stale plant rebuilds on every call. The manifest
  build, `--write` and `--check` never read the cache, so CI's
  plant-contract check always recomputes. One limit is kept on purpose:
  Python code is not in the key, so a hit does not see environment code
  patched in a live process in a way that changes the compiled model (not
  the MJCF). The live checks compare the physics of every env they are
  given instead: `validate_environment_plant` for each env training's
  `make_env` builds with a plant identity and for the envs the evaluation
  paths build, and `validate_compiled_plant` for each behavior env and,
  new here, for each env `config.build_env` builds. The frozen recovery
  gate (its nulls, brace and policy panel), the recovery calibration, the
  zero-action baseline's preflight and `widen_checkpoint` build through
  `build_env` and had relied on `current_plant_identity` alone (the report
  scripts that build through it had no plant check at all): with a warm
  cache, a
  mid-process patch of `CompsognathusEnv.__init__` (the pelvis 1.5 times
  heavier) was frozen and rolled, where `c69a169`'s rebuild refused it.
  `build_env` now refuses it. It reads the identity with
  `verify_generated=False`, since the callers that certify compare it with
  the committed manifest in their own calls. Once the species' identity is
  built in the process, the check adds about 0.03 s per env for the small
  plants and 1.2–1.4 s for the robot; a process that had not built it (the
  report scripts) also pays that build on its first env per species, about
  7.4 s for the robot and under 1 s for the others. A process that patches
  environment code and builds envs another way (a `make_env` without a
  plant identity, as `probe_ppo_updates.py` does) calls the new
  `clear_plant_identity_cache()` (exported from `plant_contract`). A physics
  recheck on every hit would cost 1.2 s of the robot's 7.4 s build, so there
  is none. The new `test_plant_contract_identity_cache.py` (18 tests)
  covers the key's inputs (the resolved model path through the species
  entry and the root), the guards (including the committed-manifest read,
  which never raises), the uncached check path and that patch on the
  recovery gate's path, and
  `docs/PLANT_CONTRACT.md` describes the cache. Measured locally: the
  robot's first call takes 7.4 s and each repeat call about 0.1 s
  (compsognathus 0.43 s, then 0.03 s); CI's three SB3 pytest steps, run
  locally with CI's flags at pull-request depth, took 2,609 s before and
  855 s after (the integration step 36:08, then 11:05). CI runners differ,
  so this PR's own CI durations are the record. The maintainer settled
  decision 10 as (c) first on 2026-09-29; its option (a), gating
  `test_compsognathus_training.py` behind the depth switch, waits for
  CU-14a, which decides it from those durations.
  `manifest.py` is one of the files cleanup PR-A and PR-B kept
  byte-identical as the hasher, but no digest hashes its bytes: the
  digest-snapshot harness, run with the optional backends blocked, printed
  848 lines with 0 errors, byte-identical on the base and on this PR, and
  `plant_contract --check` and `--check --baseline` report the manifest
  current. mypy finds no issues in 314 source files (from 313), and the
  suite collects 3,969 tests (from 3,951).
- **One notebook-cell reader for the tests and CI, shared test helpers, and
  no library-only duplicates in the notebook pins** (#569, CU-4 of
  `docs/CLEANUP_PLAN_2026_09.md`).
  The new `environments/shared/tests/notebook_cells.py` imports only the
  standard library, and the 16 places in 9 test files that read notebook
  cells now use it (`test_sb3_notebook_pins.py` keeps thin wrappers). CI's
  lint job, which does not install the package, runs it by path as its
  "Validate notebooks (parse every code cell)" step in place of an inline
  script, with the same messages; with no arguments it checks the
  repository's `notebooks/*.ipynb`, found from the file's location, and
  fails when there are none. The new `test_notebook_cells.py` (5 tests)
  resolves every `environments` import in both notebooks from source,
  without importing the package: 78 name imports (66 in
  `sb3_training.ipynb`, 52 of them distinct, and 12 in
  `google_drive_summary.ipynb`) from 21 modules, so a library rename that
  a notebook still imports fails in CI rather than in a Colab session. Two
  of its tests run `notebook_cells.py` by path under `python -I -S`, on the
  real notebooks and on a broken one, and one checks that a failure names
  a repository notebook as `notebooks/<name>.ipynb`, as the inline script
  did.
  The canonical-JSON pin covers `google_drive_summary.ipynb` too.

  Four pins that tested only the library, each with a twin in the shared
  (non-SB3) matrix, leave the notebook pins file:
  `test_the_manifest_mechanism_the_notebook_relies_on`
  (`test_stage_manifest.py`), `test_the_library_rule_the_notebook_relies_on`
  (`test_ancestors.py`), `test_the_selection_it_delegates_to`
  (`TestSelectTrunk`) and `test_the_gate_kind_set_the_notebook_relies_on`
  (`test_recovery_gate_config.py`); the file goes from 118 tests to 115.
  The manifest pin asserted what its twin did not, and those assertions
  move into `test_stage_manifest.py`: trex's `chain_for("recovery")` and
  `chain_for("locomotion")`, the recipe-label checks on all six committed
  species, the compsognathus pair included (the twin covered the other
  four), and trex's `resolve_behavior("stance")`. Dropping
  `recipe = "stand"` from compsognathus's recovery stage fails the moved
  check and passed the old twin. The consolidation plan's PR-4, which
  kept the library-rule pin, gains a dated note, as the cleanup plan's
  decision 9 asks, and so does its PR-15, whose `notebook_cells.py` slice
  CU-4 took. The notebook pins that repeat each other across files
  (`RUN_RECOVERY_STAGE` three times, the no-inference pin and the
  `chain_results` routing twice each) stay for PR-15, and the widen-help
  and `TestCommandSliceReseed` pins stay.

  No test module imports another any more: the trunk builders and
  constants that `test_replication.py` and `test_train_base.py` imported
  from `test_ancestors.py` move, byte for byte, into the new
  `ancestors_helpers.py`, and `_canonical_summary` and its helpers move
  from `test_result_summaries.py` into `result_bundle_helpers.py`. The new
  `tiny_env_helpers.tiny_env_class` replaces eight copies of a `TinyEnv` in
  five files, the one inside `test_curriculum_staged_pairs.py`'s subprocess
  script included; the copies differed only in observation shape and step
  `info`. The change is +726 / −642 lines in 25 files (net +84), about 300
  of them moved, and not the net reduction of about 200 lines the plan
  estimated: the import-resolution check, its tests and `notebook_cells.py`
  outweigh the deletions, and the estimate counted
  `_clean_repository_state`, which cleanup PR-B had already deleted. Only
  tests, test helpers and the lint step change, and none of them enters a
  digest: the digest-snapshot harness, run with the optional backends
  blocked, printed 848 lines with 0 errors, byte-identical on the base and
  on this PR. mypy finds no issues in 313 source files (from 309, the four
  new modules), and the suite collects 3,951 tests (from 3,949).

### Removed
- **Eleven unreferenced asset files** (#578, cleanup CU-16b of
  `docs/CLEANUP_PLAN_2026_09.md`, 2026-09-30; §3.2 CU-16 row), 2,089,577
  bytes, each the only copy of its blob in the tree and named by no page,
  configuration or code: the three stage videos
  `website/static/videos/raptor_stage{1_balance,2_locomotion,3_strike}.mp4`
  (1,974,435 bytes), which the site stopped showing on 2026-03-18 when it
  moved to the `velociraptor_ppo_stage*_best.mp4` names, and their three
  posters under `website/static/img/posters/` (19,787 bytes); the four icons
  `apple-touch-icon.png`, `favicon-96x96.png` and
  `web-app-manifest-{192x192,512x512}.png` under `website/static/img/`
  (62,739 bytes), which no page, web manifest or `docusaurus.config.ts` ever
  linked (the site's icon, `img/favicon.ico`, stays); and
  `environments/compsognathus/data/robot_camera_view.png` (32,616 bytes), a
  640×480 head-camera snapshot that nothing reads and the package data
  (`data/*.json`) does not ship. The site's static files shrink by 2,056,961
  bytes. The two apex GIFs stay, unused, by the maintainer's decision of
  2026-09-30, so the `Images/` correction under Fixed stands. No digest
  moves: no deleted file is an MJCF asset or another hashed input, and CI's
  digest command reports the golden current (932 lines, 0 errors).
- **Dead code and the unread gymnasium entry points** (#572, cleanup CU-7a of
  `docs/CLEANUP_PLAN_2026_09.md`). Deleted, each with the tests that
  covered only it: `gate_schema.apply_backend_overrides` and
  `has_backend_overrides` (no caller since D-D17; a `[curriculum.jax]`
  table is still validated), `stance_gate.summarize_stance_panel` and
  `episode_unsupported_duty` (the live reduction is
  `stance_panel_from_episode_duties`),
  `recovery_evaluation.paired_success_differences`,
  `reward_functions.reward_height_maintenance` and
  `check_nosedive_termination`, `stage_layout.FIGURE_NAMES`,
  `iter_figures` and `iter_generated_artifacts`,
  `StageManifest.by_position`, `stage_manifest.STAGE_MANIFEST_SCHEMA`,
  `constants.DEFAULT_FRAME_SKIP` and `environments.register_all`. The
  `gymnasium.envs.__root__` and `gymnasium.envs.MesozoicLabs` entry-point
  groups leave `pyproject.toml`: gymnasium 0.29's plugin loader reads only
  the `gymnasium.envs` group, and 1.0 onward has no loader, so no
  supported gymnasium version read them. Their KNOWN_ISSUES bullet and the
  CONTRIBUTING step that told a new species to add one go too, and the
  ROADMAP item now names the real id, `MesozoicLabs/Raptor-v0` (it said
  `Velociraptor-v0`). No digest moves.
- **The JAX/MJX runtime** (#566, cleanup PR-B of
  `docs/CLEANUP_PLAN_2026_09.md`, decision D-D17). 30 files and 15,665 lines
  are deleted: the thirteen
  `environments/shared/jax_{checkpoint,curriculum,eval,hooks,normalization,ppo,reward_termination,train_fn,trainer,trainer_types,training,training_utils,viz}.py`
  modules (6,777 lines), fourteen `test_jax_*.py` / `test_mjx_*.py` files
  and `trex/tests/test_trex_mjx_reward_parity.py` (7,256),
  `notebooks/jax_training.ipynb` (1,372) and `website/docs/training/jax.md`
  (260) with its sidebar entry. Also removed: everything in `mjx_env.py`,
  `jax_setup.py` and `mjx_utils.py` outside the frozen core below
  (`MJXDinoEnv`, `MJXEnvConfig`, `canonicalize_env_kwargs`, `setup_species`
  and the rest), and every `mjx_config.py` key the probe does not read; the
  `jax` and `jax-cpu` extras; the `[jax]` and `[jax.policy_kwargs]` tables
  of the 12 stage TOMLs (304 lines; no `foot_contact_*` line touched);
  `reporting.save_jax_stage_artifacts`;
  `plant_contract.validate_mjx_environment_plant`; the notebook's
  `species_manifest.toml` entry (the README catalog and
  `species.generated.json` are regenerated); the JAX parts of kept tests;
  `velociraptor/requirements.txt`'s JAX lines; and the `test-jax-cpu` CI
  job with its coverage `needs` entry (23 jobs → 22). On the site and in
  the README: the JAX install instructions and dependency table, the JAX
  notebook in the notebook lists, the intro's JAX/MJX feature and link, and
  the landing page's "Hyperdrive" (JAX/MJX backend) milestone.

  Kept, as the frozen MJX interface core: 549 lines in 8 files, seven of
  them opening with a FROZEN (D-D17) notice (464 without module docstrings;
  the cleanup plan estimated 473), namely `register_species_mjx`,
  `_SPECIES_CONFIGS` and `build_mjx_observation` in `mjx_env.py`,
  `make_obs_fn` in `jax_setup.py`,
  three functions of `mjx_utils.py`, all of `obs_functions.py` and the
  `mjx_config.py` registrations of trex, velociraptor, brachiosaurus and
  dibothrosuchus. Those four species hash its tokens into their
  policy-interface digests, and the plant-contract probe runs
  `build_mjx_observation` on every SB3 training and evaluation run; it is
  never edited, and a species leaves it only by declaring itself SB3-only in
  its next deliberate `policy_interface_revision` bump. Two
  type-checking-only aliases (`MJXEnvConfig = Any`, `SpeciesContext = Any`)
  keep the hashed signatures type-checkable, and `[tool.ruff]` excludes the 8
  files (`extend-exclude`, with `force-exclude = true`). Its anchor token
  digests, unchanged and pinned by the new test:
  - `mjx_env.build_mjx_observation`: `sha256:d1a8ac56e6f533670690897f47756ebf07cc2e499d6f56da89a5bea7d2446ac4`
  - `jax_setup.make_obs_fn`: `sha256:66d4e404736f5c23159eec010e125a3a9a5c0981fae300897040c16b6fa923c5`
  - `mjx_utils.scale_action_around_nominal_jax`: `sha256:f01be749e8a78ac9c9266d462f31511a06e9f0c0fef2b7d7292c4b8c431b2eae`
  - `mjx_utils.reset_mujoco_data_to_home`: `sha256:f780f506759e615e1a0e9292deeae31fa8113a6feef5442993bd26bfbec28eb0`
  - `obs_functions._array_mod`: `sha256:cbcfa3fe9292bf98e8d8c70d88d4f1ef03e49e11cfcbd50d0d6f87e6967d48f4`
  - `obs_functions.build_bipedal_obs`: `sha256:8b46a7e02e14059c38d723ff46f60eae31b633bb3b61bb2455caf862dc6f75e4`
  - `obs_functions.build_quadruped_obs`: `sha256:7b4e74c4c97a78b143f80902d6e4eb542ecc8a3830579182c1921ad5065dbded`
  - and trex's `action_filter.low_pass_alpha`: `sha256:f32bb3f1cda4b52493b8bbd4c61fff9c86231f705ddcf9d462900f97af7f0908`
    and `action_filter.apply_low_pass`: `sha256:8c619d7021723c3f74bfc3d5ccb45c27738c931d176c487a3a76a18b1affaa19`.

  The `foot_contact_*` parameters and `[env]` keys (`task_sha256` inputs),
  the readers of recorded JAX runs, `command_frame.py`'s MJX refusal (until
  consolidation PR-9), the `[curriculum.jax]` override validation and the
  generated catalog's `jax-mjx` metric rows stay (the last until the
  cleanup plan's decision 17). No plant, task, gate, hyperparameter,
  stage-config, recipe, behavior-identity or recovery digest moves: the
  digest-snapshot harness prints the same 848 lines on the base and on this
  PR, with the optional backends blocked and unblocked. The code this entry
  removes is recoverable from the `0.3.8` tag and from this PR's first
  parent (Migration, above).

  **The drift list** (D-D17's entry condition for adding a backend back; the
  cleanup plan's §5.3). Mirrored copies drifted into live defects with no CI
  failure:
  - defect 1: every Ray Tune PPO trial raised `TypeError` on `ent_coef_end`
    (removed with #564; listed under its entry below);
  - defect 3: the JAX command-line curriculum advanced a stage without
    checking `min_avg_forward_vel` (the `reward_and_length/v1` arm of
    `jax_curriculum.check_stage_gate` checked reward and episode length
    only, while the SB3 `CurriculumManager` also checks velocity and
    success), and it would have ignored `min_success_rate`, had its one
    caller ever judged the final stage;
  - defect 4: MJX training never paid dibothrosuchus
    `snap_snout_proximity_weight` (the step kernel looked the weight up
    under the other species' keys only), while the JAX CPU evaluation that
    gated the stage did, and neither matched SB3's fixed 1.5 m range;
  - defect 9: 7 of the 12 sweep configs sampled 18 env keys that no
    constructor accepts (removed with #564; listed below);
  - defect 5's never-executed copies: the sweep notebook's
    `LocomotionMetrics()` and its `_sim_dt` assumed a 0.01 s control step,
    so its compsognathus metrics were 2× off (removed with #564; the
    library's `sim_dt` default survives, latent, for cleanup CU-2; Fixed,
    below).

  **KNOWN_ISSUES entries that went with the code** (the JAX half of the
  re-add checklist; #564's entry below lists the sweep half):
  - the "Known SB3 ↔ JAX divergences (documented, deliberate for now)"
    section: the forward-velocity reference frame, eval target placement,
    stage-3 success semantics, curriculum gates and PPO advantage
    normalization, with its standing recommendation of a per-component
    SB3↔JAX reward parity test;
  - JAX evaluation cannot produce per-episode foot duty for quadrupeds;
  - the JAX backend cannot finalise a stance-gated result bundle;
  - the JAX command-line curriculum advances a stage without checking
    `min_avg_forward_vel` (defect 3);
  - MJX training never pays dibothrosuchus `snap_snout_proximity_weight`,
    while the JAX CPU evaluation that gates the stage does (defect 4);
  - a LOW (JAX) entry: two same-named `check_stage_gate` functions with
    different signatures; a logged `learning_rate` decaying faster than the
    real schedule; the KL early stop inside `lax.scan` computing, then
    discarding, the remaining minibatch gradients; `StabilityMonitor`'s
    `kl_warn=100` firing only after total collapse; per-step reward
    diagnostics decomposing forward velocity in world-X;
  - the JAX `TrainingCSVLogger` flushing per update straight to the output
    path, one network write per update on a `/gcs` FUSE mount;
  - velociraptor's SB3/MJX termination asymmetry, and its
    `nosedive_termination_threshold`, hardcoded in the SB3 env but read from
    the stage config on MJX;
  - the SB3↔JAX reward parity test (Testing / CI), and open question 1:
    whether SB3↔JAX reward parity is a hard goal (the "Open questions"
    section's only item, so the section went too);
  - clauses of kept entries: the JAX-checkpoint neighbour of the
    pre-Phase-C checkpoint entry, the MJX settle raising on a heightfield
    floor (collidable necks), the JAX trainer's clip in the
    action-saturation entry and its mention in that entry's latent-trap
    paragraph, the JAX `foot_contact_gate` in the velociraptor foot-sensor
    entry and the JAX notebook's pins in the notebook-pins bullet.

  Reworded, not deleted: the `foot_contact_*` LOW, which now covers the
  T-Rex and Dibothrosuchus SB3 envs and says why the knobs stay
  (`task_sha256` inputs). Kept, reworded as latent: the quadruped MEDIUM
  (two code paths decide whether a species is a quadruped). The probe still
  runs `build_mjx_observation`, but no new registration reaches it; a fix
  inside the frozen core would move four species' digests, while a one-root
  assertion in the plant contract's probe would not (left to CU-7). A known
  leftover: the comment at `environments/trex/assets/trex.xml:554` still
  names `mjx_config.py`'s `target_standing_z`, `_NATURAL_PITCH` and
  `healthy_z_range`, which the frozen registration no longer holds; MJCF
  bytes enter the plant identity, so the next trex plant revision corrects
  it (the cleanup plan's §4.1 batch).
- **The single-job Vertex AI route and GCS artifact upload** (#565, cleanup
  PR-A2 of `docs/CLEANUP_PLAN_2026_09.md`, decision D-D17). Deleted: 4 files
  and 957 lines, `scripts/setup_vertex_ai.sh` (183), the `Dockerfile` (44),
  `.dockerignore` (32) and `website/docs/training/vertex-ai.md` (698) with
  its sidebar entry. Also removed: `config._upload_to_gcs` and
  `config.upload_curriculum_artifacts` (213 lines), the `curriculum
  --gcs-bucket` / `--gcs-project` flags and `train_curriculum`'s matching
  parameters, the `gs://` branch of `write_results_csv`, the `[gcp]` extra,
  and the 11 tests of the removed code (7 in `test_config.py`, 4 in
  `test_stage_layout.py`); the append-mode `gs://` test now pins the
  refusal. On the site and in the README: the Docker sections of the README
  and the installation and quick-start pages, the JAX guide's "Vertex AI
  with JAX" section and the landing page's "Docker support". `tb_sync.py`
  and the `/gcs/` mount detection stay. No digest moves: the digest-snapshot
  harness prints the same 848 lines on the base and on this PR. No
  KNOWN_ISSUES entry had its cause in the removed code.
- **Ray Tune, the Vertex AI hyperparameter-tuning sweeps and the mjlab
  scaffold** (#564, cleanup PR-A of `docs/CLEANUP_PLAN_2026_09.md`, decision
  D-D17). 42 files and 13,222 lines are deleted:
  `environments/shared/scripts/sweep/` (13 files), `shared/mjlab_env.py`,
  `velociraptor/mjlab_config.py` and `velociraptor/scripts/train_mjlab.py`,
  the twelve `configs/*/sweep_{ppo,sac}.json`, `configs/quality_scoring.toml`,
  `notebooks/ray_tune_sweep.ipynb`, eleven `test_sweep_*.py` files and
  `website/docs/training/sweeps.md`. Also removed:
  `visualization.plot_trial_comparison` (only the sweep notebook called it),
  the `[ray]` and `[mjlab]` extras with the two mjlab coverage omits,
  `cloudml-hypertune` and the Vertex tuning report in `train_base`, the
  notebook's `species_manifest.toml` entry (the README catalog and
  `species.generated.json` are regenerated), and the SB3 job's Ray install,
  Ray import check and `test_sweep_ray_plant_contract.py` run.
  `test_sweep_reporting.py`, which tests `generate_stage_artifacts`, is kept
  as `test_stage_artifacts_generation.py`. `_report_hpt_metrics` keeps its
  name and still writes `metrics.json` for the CLI `train` subcommand. No
  plant, task, gate, hyperparameter, stage-config, recipe, behavior-identity
  or recovery digest moves: the digest-snapshot harness prints the same 848
  lines on the base and on this PR. The KNOWN_ISSUES entries that went with
  the code, listed as the re-add checklist the cleanup plan asks for:
  - every Ray Tune PPO trial raises `TypeError` on `ent_coef_end` before it
    trains (defect 1);
  - 7 of the 12 sweep configs crash every stage-3 trial, sampling 18 env
    keys no constructor accepts (defect 9);
  - `ray_orchestration.py` (1,006 lines) was wired to the sweep notebook only
    through `export_best_trial`;
  - quality scoring weighted `cost_of_transport` / `vel_consistency`, which
    only the notebook path exported, so scores were not comparable across
    paths;
  - from a mixed LOW entry (the rest is kept): `_handle_stage_failure` used
    `os._exit(1)`; `scoring.compute_quality_scores` sorted the caller's list
    in place; `plot_sweep_results` used the deprecated `tempfile.mktemp`;
    `collect_ray_results` could emit empty `trial_id`s;
    `load_resume_settings` read a `gpu_model` key that was never written and
    could not store `seed=0`; `_is_retryable_gcp_error` treated the generic
    `GoogleAPICallError` name as retryable;
  - the post-training artifact wishes for the Vertex path: a sweep manifest
    at submit time, a `best_trial_config.json` per stage with a
    `model_manifest.json` beside every exported `best_model.zip`, and a run
    timestamp or `resume_run` id on sweep CSV rows;
  - the sweep notebook duplicated `ray_orchestration.py`, its
    `EVAL_EPISODES` knob did not reach the in-trial evaluation, and its
    analysis wrote the sweep's `training_summary.txt` from the last-ranked
    of the top trials (gap review NB3);
  - the Drive summary skipped sweep folders written under the current naming
    (gap review NB2; none exists on Drive);
  - open question 2: published sweeps ran with identical seeds per trial,
    and the top three configs were to be re-run with three seeds before
    being locked into the TOMLs.

  The `sim_dt` default entry stays, reworded: the callers where it showed
  left with this PR, so it is latent (CU-2 fixes it; Fixed, below).

### Fixed
- **The SB3 notebook's chain loop no longer crashes on a reused ancestor
  whose `stage_config.json` is not a JSON object** (#579, cleanup CU-8a of
  `docs/CLEANUP_PLAN_2026_09.md`, 2026-09-30; §3.2 CU-8 row). Its copy of
  the ignored-edits check lacked the library's guard, so a readable file
  holding a list, `null`, a string or a number raised `AttributeError`
  after the reuse; the loop now calls `config.ignored_hyperparameter_edits`,
  which names such a file `<unreadable stage_config.json>` in its warning.
- **The 0.3.8 entry on the removed `Images/` directory left out the second
  surviving copies and misstated which copies the site and README show**
  (#577, cleanup CU-16a of `docs/CLEANUP_PLAN_2026_09.md`, 2026-09-30; §3.2
  CU-16 row). Each of its three GIFs survives twice, under
  `website/static/img/` and under `results/velociraptor/`
  (`ppo/stage1_balance.gif`, `ppo/stage3_strike.gif` and
  `sac/stage3_strike.gif`, byte-identical); the README shows the `results/`
  copy of the balance GIF, the site shows only `raptor_balance_ppo.gif`, and
  no page shows the two apex GIFs (`results/README.md` lists their
  `results/` copies).
- **The plant contract's MJX probe requires exactly one root body**
  (#572, cleanup CU-7a of `docs/CLEANUP_PLAN_2026_09.md`). The frozen
  `build_mjx_observation` roots its observation on `torso` whenever the
  registration maps one, while the probe takes the root from the
  observation schema (`torso` for `quadrupedal-target/v1`, otherwise
  `pelvis`) and checked only that it was mapped, so a registration mapping
  both could be rooted differently by the two. The probe now also requires
  that the registration maps exactly one of them, the one the schema names
  (`docs/PLANT_CONTRACT.md`), which closes the quadruped-detection MEDIUM
  in KNOWN_ISSUES; no committed registration maps both. The probe is not
  hashed, so no digest moves.
- **`imageio` is declared** (#572, cleanup CU-7a).
  `compsognathus/scripts/view_model.py --video` imports `imageio.v2`,
  which no extra required; the `viz` extra now requires `imageio>=2.16.1`
  (`imageio.v2` needs 2.16, and 2.16.0 was yanked).
- **`render_mode="human"` no longer crashes on the first step** (#568,
  CU-2 of `docs/CLEANUP_PLAN_2026_09.md`). `BaseDinoEnv.render` called
  `mujoco.viewer.launch_passive`, but `import mujoco` does not load the
  `mujoco.viewer` submodule, so a human-mode env raised `AttributeError`
  from its first `step`, and `train_sb3.py eval` without `--no-render`
  crashed (the KNOWN_ISSUES entry, now deleted; `test_env.py --render`
  escaped, because the harness package it imports loads the viewer). The
  human branch now imports the viewer itself, under its own name (`from
  mujoco import viewer as mujoco_viewer`), since a bare `import
  mujoco.viewer` would make `mujoco` local to `render` and break the
  rgb_array branch. One `_configure_camera` helper aims both the viewer's
  camera and `_make_camera`'s; the viewer's camera is aimed under the
  viewer's lock, since its render thread reads the camera on every frame.
  Tests: a human render, with `mujoco.viewer` removed from the package and
  a fake in `sys.modules`, launches the viewer once, aims its camera like
  `_make_camera` and under the lock, syncs every step and closes; an
  rgb_array render works with a fake `mujoco.Renderer`. Four mutations
  each fail at least one of them: the old attribute access, a bare import,
  no camera setup and no lock. `render` and `_make_camera` enter no digest,
  and `base_env.py` is not byte-hashed.
- **A stage summary built from `evaluations.npz` prints the node's own sim
  time** (#568, CU-2 of `docs/CLEANUP_PLAN_2026_09.md`). Called with
  `stage_results=None`, `generate_stage_artifacts` built its results with
  `build_stage_results_from_eval_data`, which took `sim_dt` from
  `env_kwargs["sim_dt"]` or 0.01 s. No stage config sets it, and the
  compsognathus pair steps at 0.02 s, so their summaries would print 1,000
  steps as "10.00s sim time" instead of 20 s; no caller has passed `None`
  since #564 removed the sweep trials, so the defect was latent (the
  KNOWN_ISSUES entry, now deleted). `generate_stage_artifacts` now takes
  `sim_dt` from a bare env of the node's task (closed afterwards), as
  `evaluate_stage_checkpoints` already did, and
  `build_stage_results_from_eval_data` gains a `sim_dt` keyword that both
  pass; its fallback stays, documented, for other direct callers.
  `backfill_gate_verdict.py` still calls it without `sim_dt`, and nothing
  it persists reads the value. Tests: the probe env's `dt` reaches the
  results and the env is closed; an explicit `sim_dt` wins over the
  fallback; both compsognathus envs step at 0.02 s. Two mutations each
  fail one of them: `generate_stage_artifacts` ignoring the probe's `dt`
  and `build_stage_results_from_eval_data` ignoring `sim_dt`. The
  `reporting` package enters no digest. The digest-snapshot harness, run
  with the optional backends blocked, printed 848 lines with 0 errors,
  byte-identical on the base and on this PR. A node the notebook re-enters
  from its own `gate_verdict.json` still prints 0.01 s in
  `training_summary.txt`, since the verdict does not persist `sim_dt` (a
  new KNOWN_ISSUES entry).

## [0.3.8] - 2026-09-27 — Reproducible Runs & Velociraptor Stage-1 Diagnosis

### Added
- **Decision D-D20 recorded; D-D17 amended** (#562, 2026-09-26;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2, `docs/CONSOLIDATION_PLAN_2026_09.md`
  §6, `docs/CLEANUP_PLAN_2026_09.md` §2, `docs/NEXT_STEPS.md` §5). The
  maintainer took the cleanup plan's decision 7 as D-D20: a Colab reclaim
  may not leave the final pair or a handoff pair truncated, nor mixed with
  the previous pair in a way the pair's checks accept (carried out by CU-3;
  Fixed, below). D-D17 is amended:
  PR-A is split in two, each with its own review, so D-D17 lands as PR-A,
  PR-A2 and PR-B. PR-A deletes the sweeps and mjlab and stays pure deletion; PR-A2 removes the single-job Vertex AI route
  and GCS upload, with an end-to-end test of the command-line curriculum path
  (the cleanup plan lands it after PR-A).
- **Decisions D-D17, D-D18 and D-D19 recorded** (#561, 2026-09-26;
  `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2, `docs/CONSOLIDATION_PLAN_2026_09.md`
  §6, `docs/CLEANUP_PLAN_2026_09.md` §2, `docs/NEXT_STEPS.md` §5). The
  maintainer took three of the cleanup plan's decisions. D-D17:
  Stable-Baselines3 is the only training, evaluation and evidence backend
  until every species' stage chain and behavior is certified. PR-A retires
  Ray Tune, the Vertex AI tuning sweeps and mjlab and, wider than the plan
  proposed, the single-job Vertex AI route (`scripts/setup_vertex_ai.sh`, the
  `Dockerfile`) and GCS artifact upload (`curriculum --gcs-bucket`, the `[gcp]`
  extra); PR-B retires JAX/MJX behind a frozen MJX interface core, so no
  digest moves. D-D18: CI type-checks with SB3 and torch installed (Changed,
  below). D-D19: this CHANGELOG is cut before PR-A; the maintainer tags
  `v0.3.8`, and `0.3.9.dev0` follows. The stale pull requests #527 and #498
  were closed. Nothing is removed yet: PR-A and PR-B carry out D-D17.
- **Cleanup and backend retirement plan, and the digest-snapshot harness**
  (#560, 2026-09-26; `docs/CLEANUP_PLAN_2026_09.md`, `docs/KNOWN_ISSUES.md`,
  `environments/shared/harnesses/digest_snapshot.py`). A living plan for the
  cleanup left after #558 and the #558 follow-up (#559). It orders a
  CI-signal PR (mypy with SB3 installed, where 21 type-only errors show
  today), the retirement of Ray Tune, the Vertex AI tuning sweeps and mjlab
  (PR-A) and of JAX/MJX (PR-B) behind a 473-line frozen MJX interface core,
  so that no plant, task, gate, hyperparameter, stage-config, recipe,
  behavior-identity or recovery digest moves, and sixteen smaller cleanup
  PRs, with the prerequisites each consolidation PR from PR-8 on needs. It
  lists the decisions they need (the retirement as proposed D-D17, not yet
  taken; the archive tags; the release cut), the 2026-09-25 eval-only check
  of the three certified walkers on the direction/terrain recipes (each
  survived the plane in every episode; 1 of 39 flat-heightfield episodes
  reached full horizon), and the lessons of the #558 reviews. The new
  hand-run tool `environments/shared/harnesses/digest_snapshot.py`
  (coverage-omitted, like the other harnesses) prints every plant identity,
  policy-interface sub-digest, per-stage task, gate, hyperparameter and
  stage-config digest, recovery calibration and behavior identity, one
  tab-separated line each, so a change that claims to move no digest is
  checked by diffing its output on the base and the head (run it by file
  path with `PYTHONPATH` set to the checkout; `--block-optional-backends`
  proves the digests compute without JAX, Ray or the cloud packages,
  `--skip-behaviors` cuts a run from about 3.6 min to about 50 s). On
  `be63a58` it prints 848 lines with no errors, byte-identical to the
  `f850815` snapshot, so #559 moved no digest. KNOWN_ISSUES gains twelve
  entries (among them the heightfield falls, the unchecked best and
  robust-best handoff pairs, `train --load` resuming into a judged stage
  directory, the `render_mode="human"` crash, the Ray PPO `ent_coef_end`
  crash and CI's mypy without SB3 types) and corrects seven, including the
  stage-3 sweep-key count (18 keys in 7 files). The docs index, NEXT_STEPS
  and the consolidation plan's status table point to the plan, and #559 is
  recorded as landed. The harness is the only code change, one new
  coverage-omitted tool; no library, training or digest changes.
- **Training session 4 recorded; living docs corrected** (#551, 2026-09-23;
  `docs/NEXT_STEPS.md` §2–§4, `docs/KNOWN_ISSUES.md`). Dibothrosuchus seed
  42, run `20260923_020654`: the collapse backstop stopped both nodes at
  1,450,000 steps (see Fixed); the stance PASSED `reward_and_length/v1` on
  the 50k statue-level checkpoint (2597.49 ± 1.95 against a 2598.29
  statue), the locomotion FAILED on forward velocity (0.0012 m/s against
  0.9, a stand-still checkpoint at 2249.86); bundle `partial`. The session
  is re-run with `RETRAIN_FROM = "stance"`, because the fix moves no digest
  and `TRUNK_FROM = "auto"` would otherwise reuse the statue-level stance.
  KNOWN_ISSUES gains three entries (a complete run cannot take a new node in
  place, so NEXT_STEPS session 7 becomes a fresh run and the seed-42
  bundle-rebuild step is dropped; an early stop is invisible in the run
  records; the quadruped gait-symmetry reward pays a motionless statue).
  A docs truth pass corrects what the living docs said about
  `TRUNK_FROM`'s default and what `"auto"` selects for trex, the
  pre-session-1 status rows of the recipes plan, the date of the Colab
  image move (the 2026-09-14/15 runs already ran Python 3.13.15; 3.12.13
  was the August parent's image), the r11 labels and backfill claims of
  two KNOWN_ISSUES entries, and adds the missing `docs/README.md` rows.
  The consolidation sequence, paused after PR-6, resumes with the
  notebook-only PR-12 slice.
- **Training sessions 1–3 recorded** (2026-09-20 .. 2026-09-23;
  `docs/NEXT_STEPS.md` §2–§3 and the widened-interface note's §3, §7 and §8).
  Trex seed 44: the r11 stance `20260815_205206` widened to r13 as
  `20260920_010912` (`WIDEN_MAX_REVISION_GAP = 2`), re-paneled identically to
  its 2026-08 certificate (3408.3 ± 88.5, duty 0.0069 / UCB 0.0117), then a
  recovery node trained 3M in place and PASSED `recovery_quality/v1` (28/40,
  LCB 0.56; paired delta against the statue null 0.70, LCB 0.58); the bundle
  records trex stance at replication 2 with `20260914_123816`, so the
  `certification_seeds = 2` bar is met. Compsognathus seed 42: the r1 stance
  `20260909_162812` widened to r2 as `20260921_203149` (gap 1), re-paneled
  identically (2801.6 ± 51.2, duty 0.0131 / UCB 0.0141), then locomotion 3M
  PASSED (3308.6 ± 13.0, 0.34 m/s, length 1000). Velociraptor seed 42: a fresh
  r10 chain `20260922_125248`, stance 6M PASSED (certified checkpoint
  1740.03 ± 301.5, length 970.3 against rails 1050 / 950) and locomotion 8M
  PASSED (2683.64 ± 8.56, 3.30 m/s, length 1000). All three bundles are
  `complete`; the widened roots carry a null `best_eval_reward` under the rule
  of #546. The open question "does a widened stance reproduce its panel?" is
  answered yes for both parents, and decision D-D14's condition for retiring
  the notebook widen path is met.
- **Automatic trunk selection** (`docs/BEHAVIOR_RECIPES_PLAN.md` decision
  D-A25). `environments/shared/ancestors.select_trunk` scans the runs beside a
  new run (the notebook's `logs/<species>/<algo>/`; on the command line the
  siblings of the run directory), applies the §4.2 reuse rule node by node
  root-first (following ancestor records), and picks the run whose certified
  ancestors cover the most of the chain, newest on a tie. A run whose
  `provenance.json` or root `stage_config.json` names another species,
  algorithm or backend is refused first, and no unreadable neighbour aborts
  the scan. It reports the runs it refused with the rule that refused them
  (the first 20 printed; every run scanned is in the selection), the
  replication each reused node rests on (a provisional ancestor is reused and
  labelled, never refused) and any run whose root passed under an older
  policy interface that `widen_checkpoint` can bridge as a `WIDEN_FROM`
  candidate. The SB3 notebook's `TRUNK_FROM` defaults to `"auto"`
  (a run id still pins one; `""` trains every node here) and the chain loop no
  longer consults the certified library for canonical ancestors;
  `curriculum --trunk-from auto` does the same on the command line.
- **Direction-following and randomized-terrain pilots** (#540, #541;
  2026-09-15). A world-direction command controller
  (`environments/shared/direction_commands.py`: requests mapped onto the
  three reserved command inputs as heading-frame forward/lateral speed and a
  turn rate, episode schedules drawn from their own seeded generator), a
  deterministic opt-in heightfield terrain generator and a balanced
  terrain-family sampler (`terrain.py`, `terrain_sampling.py`: one finite
  heightfield replaces the floor with a flat spawn apron, every shuffled
  block holds the configured number of episodes per family), a behavior env
  mixin for every registered species (`environments/shared/behavior_env.py`)
  and the original T-Rex subclass (`environments/trex/envs/behavior_env.py`,
  retained beside it), a behavior trainer and notebook
  router (`train_behaviors.py`, `behavior_notebook.py`: one CPU env, its own
  recipe dialect), a behavior checkpoint preparer
  (`behavior_checkpoint.py`: clears a canonical walker's reserved command
  connections and carries its observation statistics), a behavior
  certification certificate (`behavior_certification.py` with
  `configs/behavior_certification.toml`: 20 episodes per terrain family,
  20 s minimum horizon, survival LCB 0.80, success LCB 0.60, tracking and
  settle fractions 0.60; written as `certification/certificate.json`, never
  a `gate_verdict.json`), a replay recorder with terrain maps
  (`behavior_replay.py`), 66 recipe TOMLs under
  `configs/<species>/behaviors/` (11 templates × 6 species) plus 8 T-Rex
  pilot recipes under `configs/trex/behavior_pilots/`, the certified library
  trio (`certified_library.py`, `certified_canonical.py`,
  `certified_comparison.py`) and the guides
  `docs/TRAIN_DIRECTION_AND_TERRAIN.md` and `docs/CERTIFIED_MODELS.md`. This
  is a separate pipeline beside the recipes machinery: it bypasses the
  reserved `BaseDinoEnv._draw_episode_command` hook (writing `self._command`
  directly), judges outside `GATE_KINDS`, reuses outside
  `find_certified_ancestor` and keys checkpoint identity on source-file
  hashes. `docs/CONSOLIDATION_PLAN_2026_09.md` folds it back into manifest
  nodes (decisions D-D1–D-D10).
- **`PUBLISH_CERTIFIED` defaults to `False`** (#542, 2026-09-16; the
  consolidation plan's PR-1). Library publication required a training-origin
  stamp that a widened root handoff lacks, so a `WIDEN_FROM` session under
  the old default would have disconnected the Colab runtime right after the
  stance passed its gate.
- **Documentation handoff (2026-09-19)**: `docs/NEXT_STEPS.md` (the living
  entry point for the program state, the Drive state, the run plan and the
  open items), `docs/CONSOLIDATION_PLAN_2026_09.md` (the fifteen-PR sequence
  folding the pilots back; PR-1 = #542 and the auto-trunk PR = #543 landed,
  PR-2 is this documentation pass, PR-3–PR-15 on hold pending the
  maintainer's review), `docs/investigations/DRIVE_RUN_SURVEY_2026_09.md`
  (which Drive runs are certified under the Phase C interface, surveyed
  2026-09-17) and `docs/BEHAVIOR_RECIPES_PLAN.md` §6.2 (decisions D-D1–D-D12
  and G1–G4 from the 2026-09-17 review).
- **Stage manifest v2** (`docs/BEHAVIOR_RECIPES_PLAN.md` Phase A, part 1).
  `environments/shared/stage_manifest.py` reads `mesozoic.stage-manifest/v2`
  beside v1: per-stage `warm_start_from` (an EARLIER entry; self and forward
  references are fatal), `deliverable` and `recipe`, with ids an open
  vocabulary matching `^[a-z][a-z0-9_]*$` (the four reserved ids and every
  legacy-number, no-rewrite and no-reorder rule are unchanged). A v1 or
  synthesized manifest derives its edges as "previous advancing entry" and one
  deliverable (the last advancing entry), bit-identical to before and pinned.
  `StageManifest` gains `deliverables`, `recipe_labels`, `parent_of`,
  `ancestors`, `chain_for` and `resolve_behavior`; `stage_ref_from_dirname`
  resolves `NN_<id>` directories through the species' manifest so
  `detect_stage_from_path`, artifact upload and the sweep collector accept
  any declared id. Every species now commits a v2 `stages.toml` (T-Rex,
  Velociraptor, Brachiosaurus, Dibothrosuchus, Compsognathus and the
  Compsognathus robot) declaring exactly the ids, numbers, config files and
  edges it had, so stand and walk become deliverables without moving any
  stage directory, file label or task fingerprint.
- **`gate_verdict.json`** (`environments/shared/result_bundle/gate_verdict.py`,
  schema `mesozoic.gate-verdict/v1`): a per-stage verdict record written
  beside the handoff checkpoint and hash-bound to it and its normalization
  sidecar, so a later run can prove an ancestor passed without the run-level
  CSV; `ANCESTORS_DIRNAME` and the per-ancestor record filenames are declared,
  and load lineage gains an optional `parent_run_id` (audited as a non-empty
  string).
- **Warm-starts keyed on the manifest edge** (Phase A, WS2). Parentage was
  inferred from position in three places; each now reads the node's declared
  `warm_start_from`, which on every v1 and synthesized manifest is set
  exactly when the position is > 1, so a legacy curriculum is unchanged:
  `_stage_entry_shaping_callbacks` takes `parent_id` and fires only when the
  node has an edge and the load crosses it (a root never warms up) — every
  caller, the Ray sweep worker's stage-entry shaping included, passes the
  trial stage's edge; `train()`
  and the notebook's `train_stage` refuse an `initialize_next_stage` load
  whose recorded stage is not the declared parent
  (`task_fingerprint.validate_declared_parent`); `train_curriculum` resolves
  each node's parent from its edge out of the run's certified nodes, stops
  with a warning naming a missing ancestor instead of training from scratch,
  and writes `gate_verdict.json` for every trained node. New
  `environments/shared/ancestors.py` (`find_certified_ancestor`,
  `record_ancestor`) and `curriculum --trunk-from RUN_DIR` reuse an earlier
  run's certified nodes under the plan's rule (passed verdict hash-bound to
  the handoff pair, plant identity validating, task hash equal), recording
  them under `ancestors/<stage_id>/` and their children's `parent_run_id`;
  `generate_stage_artifacts` writes the verdict beside the handoff it judged
  and `environments/shared/scripts/backfill_gate_verdict.py`
  (`python -m environments.shared.scripts.backfill_gate_verdict <stage_dir>`)
  re-derives it for pre-Phase-A stage directories, refusing missing evidence
  (`recovery_quality/v1` cannot be backfilled). Every run judged before
  Phase A has no `gate_verdict.json` and is refused as a `--trunk-from` /
  `TRUNK_FROM` trunk until its stage directories are re-judged —
  `generate_stage_artifacts`, or the backfill tool for `stance_quality/v1`
  and `reward_and_length/v1` verdicts; `recovery_quality/v1` must be
  re-judged through the notebook chain.
- **Catalog and website per deliverable** (Phase A, WS4): species catalog
  `schema_version` 4 and `configs/species_manifest.toml` `schema_version` 2
  (decision D-A8). Stage rows carry `deliverable`, `warm_start_from` and
  `recipe` from the manifest; result rows carry a `deliverables` list with
  `primary_deliverable` / `target_deliverable` for a schema-4 summary, each
  deliverable headlined by its gate kind through a registry that must cover
  every `GATE_KINDS` entry (stance and recovery values are null until the
  summary records them, D-A9), and `[]` for a v2/v3 ladder summary, which is
  never relabelled certified. `[[species.deliverable_metrics]]` states the
  success semantics of every deliverable per training backend, resolved
  through the manifest and failing closed on an unknown deliverable, backend
  or duplicate; stage videos are keyed by stage id with integer aliases. The
  README SPECIES table gains Recipe and Warm-start-from columns and the
  per-deliverable semantics; the RESULTS block is byte-identical for the four
  committed summaries and pinned by a golden fixture (D-A10). The website
  adapter declares every exported key (the seven stance/recovery gate keys
  were undeclared), `formatGate` mirrors the Python renderer branch for
  branch, the home page keys video cards by stage id and labels them by the
  manifest label (D-A13), published run summaries render byte-identically,
  and `deploy.yml` type-checks the site with `tsc` before building
  (`@docusaurus/tsconfig` was never installed, so the check failed on a
  pristine checkout). `validate_result_summary` now raises a
  `ResultSchemaError` instead of a bare `StopIteration` when
  `primary_deliverable` is spelled differently from the deliverables map.
- **Retrain-from, no silent overwrite, and the frozen-null gate set** (Phase A,
  WS5 part 1). `curriculum --retrain-from STAGE_ID` (an id or legacy number,
  `train_curriculum(..., retrain_from=...)`; decision D-A19) trains the named
  advancing node and every advancing node after it in this run even when
  `--trunk-from` holds a certified copy, reusing only the certified ancestors
  strictly above it; it is a usage error without `--trunk-from` or naming a
  non-advancing or unknown stage, refused before the run directory exists.
  `config.refuse_occupied_stage_dir` (`STAGE_DIR_OCCUPANCY_FILES`,
  `StageDirectoryOccupiedError`; D-A20) makes `train()` and
  `train_curriculum` refuse to write into a stage directory that already
  holds `stage_config.json` or `gate_verdict.json` unless the load is an
  explicit `--load <checkpoint> --load-mode resume_same_stage`, so a variant
  is a new run directory rather than an overwritten record.
  `curriculum.FROZEN_NULL_GATE_KINDS` (`{"recovery_quality/v1"}`) names the
  gate kinds whose null resolution is frozen before training and rolled
  after, for the notebook chain loop to key on instead of a stage id.
- **Trunks compose through ancestor records** (Phase A follow-up; decision
  D-A23). `find_certified_ancestor` rule 1 still prefers the candidate run's
  stage directory and now, when the run holds only
  `ancestors/<stage_id>/ancestor.json` for the node, follows the record to
  the `source_run_dir` it names — as recorded, else the run of the same name
  beside the candidate (Colab, Drive and bucket layouts keep runs side by
  side), else a refusal naming both paths — and applies every rule at the
  source, whose handoff pair must still hash to the record's
  `handoff.model_sha256` / `handoff.normalization_sha256`. At most
  `ANCESTOR_RECORD_HOP_LIMIT` (8) records are followed, a record pointing
  back at a run already on the path is a cycle, and every refusal met on the
  followed path is re-raised prefixed with the hop taken. `CertifiedAncestor`
  describes the SOURCE and gains `via` (the followed runs, outermost first;
  logged, never written to `ancestor.json`, whose schema is unchanged), so a
  run trunked from a run that reused stance records the run that certified
  stance as stance's `parent_run_id`. Following is opt-in
  (`find_certified_ancestor(..., follow_records=True)`, default off):
  `train_curriculum` passes it for `--trunk-from`, which is another run by
  construction, and `curriculum --trunk-from` says so in its help. The
  notebook's chain loop passes `follow_records=candidate is not RUN_DIR`: its
  `TRUNK_DIR` candidate composes, while its own `RUN_DIR` — tried first, with
  `same_run` read off the candidate — never follows, because a record there
  (the reuse an earlier pass made from the trunk) would present the trunk's
  node as this run's own on a re-run: its results re-entered as trained here,
  the bundle refusing the node as both trained and reused, later nodes losing
  `parent_run_id`, and the target reused across runs (pinned by
  `test_sb3_notebook_pins`). `ancestor.json` now
  records `source_run_dir` / `source_stage_dir` as absolute paths (the handoff
  paths already were), so a record made from a relative `--trunk-from
  logs/<run>` follows from any working directory; the sibling fallback is for
  a moved layout only.
- **`curriculum --target BEHAVIOR`** (Phase A follow-up; decision D-A24).
  The command-line curriculum names the behavior it certifies — a recipe
  label (`walk`) or a deliverable's stage id (`locomotion`), resolved as the
  notebook's `BEHAVIOR` knob resolves them, or a legacy number (`2`),
  resolved as `--stage` resolves it (the notebook accepts no number) — and
  walks that target's chain (`manifest.chain_for`) root-first, stopping at
  the target, so a certified walk-only run no longer needs the notebook.
  The default is the last advancing stage and walks the whole advancing
  ladder exactly as before (pinned: every curriculum test passes unchanged
  with the default). Every node of an explicit target's chain must be
  advancing AND the chain must be a prefix of the advancing ladder: `--target
  stand` on the T-Rex or Compsognathus, whose chain runs through the
  non-advancing recovery node, is refused before any directory is written,
  naming the advancing ids and the notebook's `BEHAVIOR` knob, and so is a
  chain that skips a ladder stage (a manifest whose `behavior` edge points at
  `stance`, which the loader accepts — legacy numbers pin ids and order, not
  the edges between advancing nodes — would otherwise have the integer-keyed
  manager judge behavior against locomotion's thresholds); an unknown label
  lists the labels; `--retrain-from` must name a node on the chain. The
  target-is-never-reused rule, the budget-exhausted stop and the
  start banner follow the target. The `CurriculumManager` stays
  integer-keyed over the full advancing ladder and is never advanced past
  the target (a walk-only run leaves it mid-ladder, where one more
  `advance()` is legal and a second raises — pinned). `train_curriculum`
  gains `target=None`; `_resolve_curriculum_target` / `_resolve_retrain_from`
  give the CLI's `parser.error` and the trainer's `ValueError` the same words.
- **Hyperparameter digest, run label, and the ignored-edit warning** (Phase
  A, WS5 part 2; decision D-A21). Every saved `stage_config.json` run block
  now records `hyperparameters_sha256` — `config.hyperparameters_sha256`
  over the stage's algorithm block and its `warmup_`/`ramp_` shaping keys,
  key-order independent, untouched by env kwargs or gate thresholds — and,
  with `train --label TEXT` / `curriculum --label TEXT`
  (`train(..., label=)`, `train_curriculum(..., label=)`), a `label`.
  `config.hyperparameter_diff` names the dotted keys (`ppo.learning_rate`,
  `shaping.warmup_timesteps`) on which a config differs from a recorded
  one; `train_curriculum` uses it after a successful `--trunk-from` reuse
  to warn that this run's edit is ignored, naming the node, the ancestor's
  run, the keys, and `--retrain-from` — never a refusal. Result schema v4
  deliverable records may carry the two OPTIONAL fields
  (`result_schema.OPTIONAL_DELIVERABLE_RECORD_FIELDS`), which
  `save_result_bundle` copies from each deliverable stage's run block and
  the audit cross-checks against it; earlier v4 records validate unchanged
  and the catalog needs nothing. W&B runs are tagged `hp:<12 hex of the
  digest>` and `label:<label>`.
- **The SB3 notebook trains a behavior chain** (Phase A, WS5 notebook; plan
  §4.7, decisions D-A11, D-A15, D-A17–D-A21). `notebooks/sb3_training.ipynb`
  replaces its per-stage cells and the opt-in `RUN_RECOVERY_STAGE` pilot with
  `BEHAVIOR` (default `"hunt"`; a recipe label or a deliverable's id),
  `TRUNK_FROM`, `RETRAIN_FROM` and `RUN_LABEL` knobs and one
  `# ===== BEHAVIOR CHAIN LOOP =====` cell that walks `MANIFEST.chain_for(
  TARGET_NODE.id)` root-first: each node is REUSED through
  `find_certified_ancestor` (this run first, then the trunk for ancestors —
  never the target across runs; chained by the parent's `model_sha256`;
  cross-run ancestors recorded under `ancestors/`, never copied), JUDGED
  through the new `evaluate_stage_checkpoints` when it was trained but never
  gated, or TRAINED from its parent's handoff with a declared
  `task_load_mode`; a frozen-null gate kind freezes before training and
  rolls its panel after, and every verdict — recovery's included — is
  enforced after the bundle is written and the runtime released. A reuse
  that ignores a hyperparameter edit prints the differing keys and points at
  `RETRAIN_FROM`; `train_stage` refuses an occupied stage directory unless
  resuming, records the real (accumulated) stage duration, and passes the
  label. `evaluate_stage_checkpoints` is the notebook-side judge; the manual
  single-node cell (`MANUAL_NODE`) records a verdict without enforcing it
  and never feeds the chain; the evaluation, curves, replay and completion
  cells read `NODE_HANDOFF` / `completed_stages` for the chosen behavior.
  A same-stage resume (`train --load <periodic> --load-mode
  resume_same_stage`, the notebook's RESUME cell) of a stage that entered
  from its parent now keeps that edge's lineage keys in the re-saved
  `stage_config.json` and records the continued-from checkpoint under
  `config.RESUME_LINEAGE_KEYS` (`resume_load_path`,
  `resume_checkpoint_sha256`), so a resumed-then-judged node still chains
  by digest and can be reused by a later pass or as a `TRUNK_FROM`
  ancestor; before, the re-save replaced the edge with the periodic
  checkpoint and the reuse rule refused the node forever.
- **Publication per deliverable** (Phase A, WS3): result schema v4 (v2 and v3
  still read verbatim). `provenance.deliverables` records every deliverable
  the run trained with its checkpoint hashes and a `certified` flag (own gate
  plus every `warm_start_from` ancestor's, in the run or as an `ancestors/`
  record); `primary_deliverable` / `target_deliverable` name the published
  model and the node the run aimed at. Bundle status is `complete` when the
  target and every present deliverable are certified, `partial` when at
  least one is, `failed` otherwise; `summary.json` is written whenever
  something is certified and `selected_model_path` is the primary's — so a
  failed hunt publishes the certified walk, a walk-only run is a complete
  bundle when walk was its target, and a certified stance-only run publishes
  as partial. `validate_result_summary` / `validate_result_bundle` gain
  `require_publishable`; the audit reports `canonical-partial`, binds a
  `parent_run_id` lineage to the ancestor record carrying that checkpoint,
  and cross-checks the deliverables and ancestors maps against disk.
  Evidence validation re-derives a recorded pass and binds, without
  re-gating, a recorded failure. Under a v1 or synthesized manifest the
  rules collapse to the pre-Phase-A outcomes (pinned). The catalog, website
  and the notebook chain loop follow in the rest of Phase A.
- **Behavior recipes plan** (`docs/BEHAVIOR_RECIPES_PLAN.md`): the adopted
  design for turning the linear stage curriculum into a DAG of behavior
  recipes — stand, walk, hunt and follow direction — each a separately
  certified, separately published policy on one shared certified trunk.
  Records what the tree already does (three of the four behaviors exist as
  separate checkpoints; parentage is inferred from position; only the last
  numbered stage publishes; the stage vocabulary is closed), the manifest-v2
  design (`warm_start_from`, `deliverable`, derived recipes), per-deliverable
  publication, per-deliverable capability gates (absorbing the gap review's
  held decisions: CF2 by deleting the hunting gate's velocity term, SS2, CF3,
  and SS1 as provenance), the
  one batched policy-interface revision that reserves body-relative command
  dims for follow-direction with a widening tool for the certified stance
  checkpoint, the notebook's `BEHAVIOR` chain loop with certified-ancestor
  reuse, the phase/PR sequence, the decisions taken and their veto points.
- **The frozen recovery gate is reachable end to end** (gap review Phase G,
  `docs/reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md` findings EE1/EE2). Three
  seams, each previously fail-closed with no path to a verdict:
  - `generate_stage_artifacts` / `_apply_stage_gate` now forward the stage
    directory and a new `recovery_successes_by_seed` argument into
    `evaluate_stage_gate`, so a `recovery_quality/v1` stage with a frozen
    `gate_resolution.json` and a rolled panel produces a real verdict from
    the one shared entry point instead of the unconditional no-stage-dir
    refusal (every missing input still refuses).
  - New `freeze_recovery_gate.roll_policy_panel(stage_dir, policy_zip,
    vecnorm)` rolls the trained policy over EXACTLY the frozen decision
    procedure — the recorded panel seed, episode count, recovery window, and
    the calibrated judge the null manifest was measured under — after
    re-verifying the resolution against the stage's current task
    fingerprint. Chain-verified: an untrained policy fails with the LCB
    criterion named; a 20/40 panel (the measured grade of the existing 3M
    checkpoint) passes the frozen thresholds.
  - `StageGatePlateauCallback` records its per-evaluation capture
    (`gate_progress.npz` appends, `diagnostics/eval_*` scalars) BEFORE the
    plateau-metrics early-out, so a stage whose gate declares none of the
    legacy scalar thresholds — the recovery stage — no longer trains with
    zero gate diagnostics.
  - The training notebook's recovery flow now freezes the resolution
    (pre-registered nulls and thresholds), trains, rolls the frozen panel,
    and records the verdict; its setup gains a `PPO.load` preflight (which
    releases the Colab runtime before raising, like the gate-failure cells)
    and pinned installs — the jax and ray notebooks' installs are pinned to
    the same tested set.
  - `roll_policy_panel` enforces, rather than asserts, the frozen
    procedure: it refuses when a frozen null was measured under a different
    safe set than the current calibrated judge (a recalibration is outside
    task identity, so the resolver's staleness check alone cannot see it),
    when the frozen nulls' panel size is below the frozen minimum (a
    rehearsal freeze can never certify), and when the stage directory's
    recorded task fingerprint disagrees with the one the committed config
    derives.
- **Stage-scoped `--override` accepts semantic stage ids**
  (`recovery.env.push_interval_steps=250`, or any manifest id via the
  species' manifest), and both an unknown stage and an unknown config
  section now raise with the available choices instead of silently
  no-opping (a typo'd override used to train the full stage budget at
  unmodified hyperparameters) or crashing with a bare `KeyError` (gap
  review TC10/CI6).
- **Recovery stage first-runs and P3 calibration record**
  (`docs/investigations/TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md`). Covers
  the two recovery training runs (3M and 5M), the three earlier attempts
  that died at the stance→recovery boundary and why, all four certification
  panels (statue / brace / stance warm-start / trained policy), the §8.1
  transfer pilot, and the P3 safe-set derivation. Load-bearing findings:
  both nulls score 0/40 while the trained policy reaches 21/40 (LCB95
  38.5%) under the calibrated safe set, paired +0.30 (LCB +0.15) over its
  own warm-start parent; the brace null dies *faster* than the statue, so
  the policy's stance is held by feedback rather than by a stable pose; and
  the provisional per-step support criterion **fails the certification
  target itself** (quiet certified stance transiently reads 0.0 N on a
  foot), so it must be dropped or windowed before it can gate. Also records
  that both training runs peaked at roughly a third of their budget, making
  the warm-started entropy/LR schedule — not the step budget — the binding
  constraint, and names the off-distribution schedule test as the
  experiment that blocks P5.

#### Phase B (BEHAVIOR_RECIPES_PLAN §4.4/§4.5)
Phase B landed 2026-09-13 as four pull requests — the gate-configuration
digest and reuse rule 7 (#533, WS-B3), the `task_success/v1` hunting gate
with the measured collapse floor (#534, WS-B1 + WS-B2), seed replication as
provenance (#535, WS-B4), and the docs pass recording decisions D-B1–D-B17 in
plan §6.1 (WS-B5); the bullets below are per workstream.
- **`task_success/v1`, the hunting gate kind** (Phase B, WS-B1; plan §4.4,
  decisions D-B1, D-B2, D-B3, D-B12, D-B13, D-B14).
  `environments/shared/curriculum/task_success_gate.py` certifies the
  selected checkpoint's per-episode task success through the exact one-sided
  95% Clopper-Pearson lower bound (`recovery_gate.binomial_lcb`; 20/30 clears
  0.5 at 0.5006, 19/30 does not, the committed 29/30 bounds at 0.851) at a
  declared panel size: `GATE_KINDS` gains the kind consuming
  `min_success_lcb`, `min_eval_episodes`, `min_avg_reward` (a collapse rail,
  never the gate), `min_avg_episode_length` and `required_consecutive`, with
  the bar AND the panel size required. One implementation, three consumers:
  `reporting.gates.evaluate_stage_gate` judges it post-stage from
  `evaluation_selected.csv` hash-bound to the handoff pair — the checkpoint
  digest AND the VecNormalize sidecar digest when the rows record one (an
  absent or unbound file, or one rolled under other observation statistics,
  is a refusal; `task_success_statistics` reads the count, and the dispatch
  is now closed — a registered kind with no arm is refused by name instead
  of routed to the reward conjunction), railing `min_avg_reward` and the
  optional length floor on the SAME panel's mean reward / length rather than
  on the argmax EvalCallback panel a sweep trial's `stage_results` carry;
  `result_bundle.evidence` re-derives a recorded pass from the same rows
  bound to the certified checkpoint and requires the published
  `selected_model_success_count` / `_n_episodes` to equal the rows' k/n;
  the sweep collector judges a row from the trial's recorded
  `success_count` / `n_success_episodes` and rails on the same panel's
  `selected_mean_reward` / `selected_mean_episode_length` (new
  `metrics.json` keys and CSV columns beside `success_lcb_threshold`; a
  non-integral count is not evaluable), and one shared trainer panel
  (`train_base.run_success_panel`, behind `train()` and the Ray Tune
  worker) sizes the hunt panel at `max(post-eval n, min_eval_episodes)`,
  writes it as the trial's evidence CSV and records the count ONLY beside
  that file — a panel that did not evaluate the handoff pair records no
  count, so the row is "not evaluable" beside the refused verdict —
  while `generate_stage_artifacts` rolls a panel from the handoff pair on
  the publication seed when none is bound or the bound one is smaller
  than `min_eval_episodes` (`task_success_panel_episodes` overrides the
  size; 0 skips), so the row and the on-disk verdict agree by
  construction. `evaluate_recorded_gate` (what
  `google_drive_summary.ipynb` reads over `evaluations.npz`) judges the
  kind on per-evaluation `success_count` / `n_success_samples` and reads a
  history without them as incomplete, never as a reward-rail pass; the
  notebook exports `success_lcb_threshold`. In training,
  `CurriculumManager` forms the same bound from the EvalCallback panel's
  successes (`success_count` / `n_success_samples` in the eval history;
  `StageThreshold.min_success_lcb` defaults to `+inf`), and the CLI
  curriculum's verdict records the count it was judged on (`success_count`
  / `n_success_samples` persisted in the verdict's `stage_result`). The verdict,
  `summary.json` stage rows (`selected_model_success_count` /
  `_n_episodes` / `_success_lcb`, optional and validated) and the catalog
  headline (`task success LCB95`) carry the judged numbers;
  `backfill_gate_verdict.py` re-derives the kind from a hash-bound CSV and
  refuses without one; the JAX path refuses the kind by name and, per
  D-B13, refuses a final stage declaring it before any training (on the
  declared kind alone — the final stage's block is still not
  schema-validated, so single-stage pilots train as before). Every other
  species' hunt stays on `reward_and_length/v1` (D-B14); the trex
  `behavior.toml` adoption, the catalog gate formatter and the website
  formatter land with WS-B2.
- **The trex hunt is gated on `task_success/v1` with a measured collapse
  floor** (Phase B, WS-B2; plan §4.4 and plan D1; 2026-08 review CF2, CF3,
  SS2; decisions D-B1, D-B2, D-B3, D-B4, D-B5, D-B14). `configs/trex/
  behavior.toml` `[curriculum]` declares `gate_kind = "task_success/v1"`
  with `min_success_lcb = 0.5` — PROVISIONAL, pre-pilot (D-B2): the exact
  one-sided 95% Clopper-Pearson bound needs 20/30 at n = 30 (LCB 0.5006;
  19/30 bounds at 0.4669), 26/40 at n = 40 (0.5081; 25/40 at 0.4828), and
  the committed 2026-03 result recomputes to 29/30 → 0.8514 from its rounded
  mean (a recomputation, not a re-judgement: no per-episode evidence, judged
  at `min_success_rate` 0.25 with no velocity term) — and
  `min_eval_episodes = 30` (D-B1, coupled to the notebook's selected panel
  and provenance `evaluation_episodes`). `min_avg_forward_vel = 2.0` is
  retired (CF2 / D1: a bite ends the episode after ~0.5 m, so a bite episode
  cannot average 2.0 m/s) and the raw-mean `min_success_rate = 0.5` (SS2:
  43% false-block at threshold) is replaced by the bound; `min_avg_reward`
  becomes the collapse RAIL at 361 = round(0.6 × 602.13) (D-B4), the statue
  ratio the stance rail uses. The absolute `collapse_peak_floor = 100.0`
  (CF3; 6× below the do-nothing reward) is REMOVED, not shadowed, for the
  relative pair `collapse_peak_floor_reference = 602.0` /
  `collapse_peak_floor_fraction = 0.45` (floor 270.9) from the hunting statue
  measured 2026-09-13: 602.13 ± 175.35 over 40 episodes, 40/40 full horizon,
  `zero_action_baseline.py trex:3 --episodes 40 --seed 3042`, mujoco 3.10.0,
  physics revision 7, pinned by `statue_constants_physics_revision = 7` so
  `test_statue_constant_freshness.py` now guards the stage;
  `collapse_peak_warmup_timesteps = 1_000_000` (D-B5 — a judgment bounded by
  the 600k stage-entry window and the half-budget test bound, not a replay;
  re-measure follow-up in KNOWN_ISSUES) and explicit `collapse_min_evals /
  collapse_patience / collapse_drop_fraction = 20 / 10 / 0.5` (locomotion's
  tuned values, not the 12/8/0.4 defaults). `required_consecutive = 3` stays
  as in-training hysteresis only (D-B3); `[env]`, `timesteps` and the
  warm-up/ramp keys are untouched, so the task fingerprint is unchanged and
  only the behavior stage's gate digest moved. The catalog gate formatter
  and its website mirror gain the kind's branch (`task success LCB95 ≥ 0.5;
  reward rail ≥ 361; ≥ 30 episodes/evaluation; verdict from the selected
  checkpoint's evaluation_selected.csv (post-stage; fail-closed when
  absent)`, no consecutive-passes tail; the mirror test pins none < recovery
  < task_success < generic), `configs/species_manifest.toml` names the kind
  and the bar on the trex hunt deliverable and its D-A9 pointers say "a
  later phase (D-B15)", and `species.generated.json` / the README SPECIES
  block are regenerated (RESULTS block unchanged). The committed-config test
  loops cover every species with a stage manifest (D-B14; a stage may omit
  the collapse floor only when its TOML says the omission is deliberate),
  and the trex stage-3 catalog pin now asserts no velocity target, the 0.5
  bar and no raw success rate. Every other species' hunt stays on
  `reward_and_length/v1`.
- **The gate verdict records the gate it was judged under, and reuse checks
  it** (Phase B, WS-B3; decisions D-A22, D-B6, D-B7, D-B8). `gate_verdict.json`
  gains `gate` — the `curriculum.gate_schema.gate_config_view` projection of
  the `[curriculum]` block the verdict was judged under: `gate_kind`,
  `gate_schema_version` and ONLY the thresholds the kind consumes (every
  advancing kind's `min_eval_episodes` / `required_consecutive` included;
  schedule, collapse, diagnostic, retention and provenance keys and any
  `[curriculum.jax]` table excluded) — and `gate_sha256`, its digest with
  every numeric threshold `float()`-normalised
  (`result_bundle.hashing.gate_config_sha256`; `100` and `100.0` are the same
  gate; `gate_config_differences` names the thresholds two blocks differ on).
  The schema stays `mesozoic.gate-verdict/v1`: a file judged before D-A22
  carries neither field and still reads, but when both are present
  `gate_sha256` must be the digest of `gate` (a verdict edited after judging
  is refused everywhere the reader is used — reuse, the ancestor records,
  the bundle audit). `write_gate_verdict(gate_config=...)` is a REQUIRED
  keyword, and every writer passes the block it judged under: the post-stage
  judge (`generate_stage_artifacts`) and the in-training verdict pass the
  current block; `scripts/backfill_gate_verdict.py` passes the block the
  directory's `stage_config.json` recorded, or, with the new
  `--gate current`, the checkout's block for the stage (the re-judge path
  after a threshold edit for a `reward_and_length/v1` directory, whose
  evidence rows are re-aggregated; `--gate recorded` is the default). A
  `stance_quality/v1` verdict is read off `stance_gate_report.json` and
  re-derives nothing, so the tool refuses a report whose recorded
  `thresholds` are not the block being judged under, under either `--gate`
  — a stance directory whose rail moved is re-judged through the notebook
  JUDGE branch, which measures a fresh panel. The post-stage
  judge logs a WARNING naming the differing thresholds when the block it
  judges under is not the one the directory's `stage_config.json` recorded
  — the notebook's JUDGE branch judges under the session's config — and
  never refuses (D-B8: re-judging under an edited gate is the intended path;
  edit a threshold, re-judge, never retrain). `find_certified_ancestor`
  gains rule 7, between the task and the chain: `current_gate_config` (the
  current `[curriculum]` block) is a required keyword-only argument, None
  refuses, a verdict without `gate_sha256` is refused naming both re-judge
  paths (the notebook JUDGE branch / `generate_stage_artifacts`, or
  `scripts/backfill_gate_verdict.py --force [--gate current]`), and a
  differing digest is refused naming every threshold that differs; only the
  gate the verdict was judged under counts, never the block the directory
  trained under, so a re-judged directory is reusable under the gate it was
  re-judged under. `CertifiedAncestor` gains `gate_sha256`; `ancestor.json`
  and the provenance projection are unchanged (the copied
  `gate_verdict.json` carries the digest). `train_curriculum --trunk-from`
  passes the node's block and the notebook's chain loop passes
  `config.get("curriculum_kwargs", {})` (pinned). **Every `gate_verdict.json`
  written before this change is refused by rule 7 until re-judged**: the
  certified trex stance directories (`20260810_145546`, and the seed-44
  replicate `20260815_205206`) must be re-backfilled with `--force`, and
  re-judged through the notebook JUDGE branch when the refusal then names
  a threshold that moved since (the stance rail `min_avg_reward` did on
  2026-08-10) — a refused trunk is otherwise retrained in the new run; see
  `docs/KNOWN_ISSUES.md`.
- **Seed replication is provenance: `certification_seeds`, the
  `certification_panel` role, discovered replicates and the provisional
  label** (Phase B, WS-B4; plan §4.5 / SS1, decisions D-B9, D-B10, D-B11,
  D-B16, D-B17). A stage's `[curriculum]` may declare `certification_seeds`
  (a new publication-key class in `gate_schema`, positive int, default 1;
  never a threshold, so it enters neither `gate_sha256` nor
  `hyperparameters_sha256` nor `task_sha256` — the certified stance trunk
  stays reusable); `configs/trex/stance.toml` declares 2, citing the seed
  42 PASS / 43 FAIL / 44 PASS record. New
  `environments/shared/replication.py` discovers a deliverable's replicates
  among the run's `LOG_BASE/<species>/<algo>/` siblings — the SAME recipe:
  a passed, reusable verdict for the node with equal `task_sha256` and
  `gate_sha256`, a `stage_config.json` with the same plant identity and
  `hyperparameters_sha256` (recorded, else derived from the recorded blocks
  through the new `config.recorded_hyperparameters_sha256`, so a pre-D-A21
  sibling is never skipped for the missing field alone; a pre-D-A22 verdict
  without `gate_sha256` IS skipped until re-backfilled), and a different
  training seed; every skip is logged with its reason and nothing raises
  on a malformed neighbour. `save_result_bundle(replicates=...)` records
  them: each deliverable's `replication.runs` lists this run first, then
  the replicates (distinct run ids and seeds, refused otherwise), plus the
  optional record fields `certification_seeds` (the stage's declaration)
  and `provisional` (`count < certification_seeds`); the schema
  (`OPTIONAL_DELIVERABLE_RECORD_FIELDS` grows to four) and the audit fail
  closed on a malformed or inconsistent record. The SB3 notebook passes the
  discovered replicates and declares `certification_panel =
  PUBLICATION_SEED_START` (3042) in both `seed_roles` dicts;
  `initialize_result_bundle`'s default roles carry it too. The audit binds
  the role per evidence — `stance_panel_selected.csv` row `i` must have run
  on `role + i`, a recovery `gate_resolution.json` must register
  `panel_seed_start == role` — pins a declared role to the registered block
  itself, and refuses a recorded stance PASS whose provenance lacks the
  role (pre-Phase-B stance bundles need republishing;
  `docs/KNOWN_ISSUES.md`). A complete bundle regenerates its derived
  artifacts when a re-run of the publication changes nothing but the
  replication record (a replicate that certified later); every other
  difference is still refused as immutable. The catalog validates the record and re-derives
  `provisional` from the CURRENT `certification_seeds` (never aggregating
  across bundles), exports `certification_seeds` on stage rows and
  `certification_seeds` / `provisional` on deliverable rows, and renders
  `N run(s) of M seed(s)` plus `provisional` identically in
  `species_catalog._format_replication` and the site's `formatReplication`
  (pinned); `headlineFor` appends ` (provisional, N of M seeds)` to a
  provisional primary. `RESULT_SCHEMA_VERSION`, `GATE_VERDICT_SCHEMA` and
  `SPECIES_MANIFEST_SCHEMA_VERSION` are unchanged (every addition is an
  optional field) and the README RESULTS block is byte-identical.

#### Phase C (BEHAVIOR_RECIPES_PLAN §4.6)
- **Command-frame helpers and the command-slice reseed plumbing** (Phase C,
  WS-C0; plan §4.6 "Normalization of the command slice", invariant 8).
  `environments/shared/command_frame.py` is the one home of the
  body-relative command frame: `COMMAND_WIDTH = 3`, the components
  `v_x_cmd` / `v_y_cmd` / `yaw_rate_cmd` pre-scaled to `[-1, 1]`, the modes
  `none` / `heading` / `heading_and_speed`, the six `command_*` env keys,
  the probe vector `(0.25, -0.5, 0.75)`, `zero_command`,
  `validate_command_mode` (an unknown mode is refused on every backend; a
  live mode is refused on `jax-mjx` until the MJX command path lands and on
  `stable-baselines3` until Phase D), `command_slice`,
  `reseed_command_slice` (the trailing slice to mean 0 / var 1 in place,
  count carried) and `pad_running_stats` (a new `RunningMeanStd` with a
  reseeded slice appended — the widen tool's primitive). The module imports
  numpy only. `load_vecnorm_stats(reseed_command_slice=True)` reseeds the
  carried `obs_rms` on BOTH the train and eval destinations (a slice that
  was constant zero during the parent run has var ≈ 1e-11 and would clip a
  live command at `clip_obs`; the eval copy is taken before the train
  reseed so each destination is reseeded on its own line);
  `train_base._load_vecnorm_into_envs` derives the flag from the stage's
  `command_mode` at both call sites — never on a `resume_same_stage` load,
  whose parent is the same task and already trained under the live slice
  — and `policy_loading.load_sb3_checkpoint(reseed_command_slice=True)`
  offers the same to evaluation for a sidecar whose slice was never live.
  In Phase C every stage is `none`, so the plumbing is the pinned Phase D
  hook (`test_command_frame.py`).
- **The one Phase C interface revision: a 3-dim body-relative command
  segment on all six species** (Phase C, WS-C1; plan §4.6 "The one
  interface revision", decisions D-C1–D-C7, D-C16, amendments A1–A9, A13).
  Every species' `_get_obs`, `obs_functions.build_bipedal_obs` /
  `build_quadruped_obs` (`command=None` means zeros) and
  `mjx_env.build_mjx_observation(command=)` append `v_x_cmd` / `v_y_cmd` /
  `yaw_rate_cmd` LAST, so every existing slice offset is unchanged and
  `obs[-3:]` is the command; `policy_layer.observation_segments` gains the
  `command` segment. Widths: trex 61 → 64, velociraptor 67 → 70,
  brachiosaurus 83 → 86, dibothrosuchus 77 → 80, compsognathus 53 → 56,
  compsognathus_robot 43 → 46; `policy_interface_revision` velociraptor
  9 → 10, trex 12 → 13, brachiosaurus 7 → 8, dibothrosuchus 6 → 7,
  compsognathus 1 → 2, compsognathus_robot 1 → 2 (`plant_versions.toml`
  note 12; physics/visual revisions, `observation_schema` strings, Box
  bounds and the statue pins unchanged; both manifests, the species
  catalog and the README species rows regenerated). `BaseDinoEnv` and
  `MJXEnvConfig` accept the six task-level kwargs `command_mode`,
  `command_speed_range`, `command_lateral_range`, `command_yaw_rate_max`,
  `command_switch_interval`, `command_switch_jitter` with inert defaults;
  only `command_mode = "none"` is implemented — SB3 refuses every live
  mode ("reserved for … Phase D") and MJX refuses it in
  `canonicalize_env_kwargs` and `MJXDinoEnv.__init__` ("not implemented on
  the jax-mjx backend", invariant 9). `BaseDinoEnv.reset` calls the new
  `_draw_episode_command()` hook once, after the push block (zeros, no RNG
  draw in Phase C; Phase D replaces the hook body, never `reset()`), and
  `EnvState.command` rides the MJX pytree; both backends expose
  `command_manifest()` (`None`). While the effective `command_mode` is
  `"none"` the six keys are carved out of the task-fingerprint env section
  for every species, so a stage's `task_sha256` moves only through the
  plant's `policy_interface_sha256`; `compute_task_fingerprint` /
  `derive_stage_task_fingerprint` take `command_manifest=` and record a
  `command` payload section only when one is passed. The SB3 and MJX
  plant-contract probes inject the non-zero `COMMAND_PROBE_VECTOR`, so
  backend observation parity is asserted on the new slot, and
  `validate_mjx_environment_plant` now checks the MJX observation width
  against the identity. The seeded reset draw stream is pinned by
  `environments/shared/tests/fixtures/phase_c_reset_golden.json`
  (captured at 7db7f8e by `tests/reset_golden.py`: trex stance seeds
  42/3042/3043/3081, trex recovery 3042, compsognathus recovery
  1042/7042, compsognathus_robot recovery 1042; reset qpos/qvel/targets/
  push starts and the generator state exact, an unseeded second reset
  after the 20-step zero-action trajectory exact too, the trajectory
  itself advisory; the hook is also pinned directly — no `np_random`
  access under `"none"`, and it runs after the push schedule and before
  `_get_obs`), which is
  also the evidence behind restamping — not re-measuring — the two
  compsognathus recovery calibrations with
  `environments/shared/scripts/restamp_recovery_calibration.py` (plant
  identity + task hash + `restamp_history` with a `--reason` that defaults
  to the Phase C justification; idempotent; refuses a physics change, a
  changed task and an unknown species): their `profile_sha256` moves, so
  every compsognathus recovery
  freeze (`gate_resolution.json`) made before this revision is refused
  and must be re-frozen. `environments/velociraptor/mjlab_config.py`
  `obs_dim` 67 → 70 (pinned against the identity);
  `observation_ablation_report.SLICES["trex"]` gains `command: (61, 64)`.
  Pins: `test_phase_c_interface.py` (golden replay, six-species layout,
  SB3 inertness and refusals, the task-fingerprint carve-out, the restamp
  tool, the mjlab width), `test_mjx_phase_c_interface.py` (MJX refusals on
  `MJXDinoEnv`, `canonicalize_env_kwargs` and `jax_setup.setup_species`,
  the widened-but-bit-identical `"none"` path, the MJX width check) and
  `test_plant_contract_phase_c.py` (non-zero probe parity on both
  backends, non-vacuous); every existing width / negative-index pin moved
  with the segment (foot contacts at `obs[-9:-7]`, direction at
  `obs[-7:-4]`, distance at `obs[-4]`); `test_command_frame.py` and
  `test_phase_c_interface.py` join the SB3 CI job's explicit list. The
  stale `obs is N` comments in the trex, dibothrosuchus and brachiosaurus
  stage TOMLs were refreshed (comments never enter
  `hyperparameters_sha256`; every stage's digest is unchanged), and the
  compsognathus README / RECOVERY_CALIBRATION.md, plan §3.2,
  SIM_TO_REAL_PLAN and MJX_CONVERSION_PLAN prose now state the Phase C
  layout.
- **`widen_checkpoint`: an SB3 PPO/SAC checkpoint pair widened r → r+k
  (1 ≤ k ≤ `--max-revision-gap`, default 1) into a judge-ready root** (Phase C, WS-C2; plan §4.6 "Widening instead of
  retraining" / "Exact transfer", invariant 7; decisions D-C8–D-C10, D-C12,
  amendment A13). `environments/shared/scripts/widen_checkpoint.py`
  (`widen_checkpoint(...) -> WidenResult`, `python -m ... --species trex
  --stage stance --from-stage-dir <run>/01_stance --to-stage-dir
  <newrun>/01_stance`, or the explicit `--model/--vecnorm/--algorithm/--seed/
  --n-envs/--timesteps` pair) gates the parent's recorded plant identity
  against the current one (same species / physics digest / nq / nv / nu /
  action_dim, `1 <= current - parent policy_interface_revision <=
  max_revision_gap`, `observation_dim + 3`; every differing field named; a
  missing identity needs `--allow-legacy-plant`),
  rewrites the archive through SB3's own serializer — three zero columns
  APPENDED to every first layer that reads the observation (PPO
  `policy_net.0` / `value_net.0`, SAC `actor.latent_pi.0`) and INSERTED
  after the observation block of the SAC critics (refused on a PPO archive),
  the Adam `exp_avg` / `exp_avg_sq` of those tensors padded identically by
  parameter order with shape assertions (a missing optimizer member or a
  stale moment is refused), the observation `Box` widened, `_last_obs`
  cleared, `num_timesteps` kept — and re-pickles the VecNormalize sidecar
  with `pad_running_stats` (the appended slice born reseeded), the widened
  `Box`, the current identity and one `set_venv` over the real env. Output:
  the parent's own handoff name plus byte-identical `<stage_label>_final.*`
  copies, `stage_config.json` whose run block carries the parent's run facts
  and the new `config.WIDEN_LINEAGE_KEYS` (never `LOAD_LINEAGE_KEYS` — a
  widened node is a root), `plant_identity.json`, `task_fingerprint.json`
  (the stage's CURRENT fingerprint, also re-stamped into the archive with a
  `mesozoic_widen_lineage` record of the parent hashes) and
  `widen_report.json` (`mesozoic.widen-report/v1`); never a verdict, an
  evaluation, provenance or a periodic checkpoint. Before returning, the tool
  verifies that every padded column is exactly zero and that the widened
  policy's actions match the parent's within 1e-6 over a seeded 200-step
  rollout with the command slice zero and with the probe vector in it (the
  measured delta is recorded), deleting the target directory on any failure.
  Refusals: a failed parent verdict, an algorithm that disagrees with the
  archive, an occupied or non-empty target, a target not named as the
  stage's directory or nested inside the parent's run, an explicit `--model`
  that is not a handoff checkpoint, a non-Adam optimizer moment the tool
  cannot pad, a missing or unstamped sidecar; the self-verification asserts
  every widened width before it inspects a zero block.
  `docs/RESULT_BUNDLES.md` documents the keys and the report;
  `test_widen_checkpoint.py` (SB3 job) builds a narrow r-1 PPO and SAC
  parent on the real trex stance env, widens each once, and pins invariant
  7: exact-zero padded columns, the SAC critics' inserted block, action
  equality on zero and probe commands over the seeded rollout, padded Adam
  moments (a missing member / a stale moment refused), one PPO update under
  `initialize_next_stage` through `train_base` leaving the columns zero,
  the lineage in the archive and the run block, the refusal matrix, every
  repo loader accepting the pair, the widened directory reusable as a root
  after a passed verdict (rules 3 and 4 still biting), hash stability
  after the self-verification, and the CLI round trip in both forms.
  Decision D-C17 (from the PR-C3 review) bounds the revision gap instead of
  pinning it to one: both certified trex stance parents (20260810_145546
  seed 42, 20260815_205206 seed 44) carry policy interface r11, and trex
  went r11 → r12 in 8795e28 as a fingerprint-only bump (observation_dim
  61 both sides, physics r7, action_dim 15 unchanged) before PR-C1's r12 →
  r13, so the one-bump gate refused the very parents the "widen, never
  retrain" premise (D-C14 / D-C15) rests on. `widen_checkpoint(...,
  max_revision_gap=1)` / `--max-revision-gap N` (an integer >= 1; below 1
  is refused at the API and exits 1 at the CLI) admits a parent up to N
  revisions behind while every other gate condition still applies — same
  species, physics digest, nq / nv / nu, action_dim and `observation_dim +
  3` — so only a chain of fingerprint-only bumps can be crossed and a
  width-changing intermediate hop stays refused; a parent further behind
  than N is refused naming both revisions, the measured gap, the bound and
  the flag, with the note that opting in asserts from `plant_versions.toml`'s
  numbered notes that no intermediate bump changed an observation layout,
  physics or action field. `widen_report.json` records `revision_gap` /
  `max_revision_gap`, the archive's `mesozoic_widen_lineage` and
  `WidenResult` carry `revision_gap` (`null` for an `--allow-legacy-plant`
  parent without an identity, pinned). `test_widen_checkpoint.py` widens a
  PPO parent two revisions back under `max_revision_gap=2` through every
  invariant-7 pin (exact-zero columns, action equality on zero and probe
  commands, lineage gap 2, report fields, every loader, root reuse), keeps
  the two-revisions-behind refusal as the default (its message naming the
  gap and the flag), refuses r-3 under gap 2 and a width-changing hop under
  gap 2, refuses `max_revision_gap=0` at the API and `--max-revision-gap 0`
  at the CLI, and round-trips the r-2 parent through the CLI with
  `--max-revision-gap 2`.
- **Notebook `WIDEN_FROM` / `WIDEN_MAX_REVISION_GAP` knobs and widen cell,
  the AST pins, and the Phase C documentation** (Phase C, WS-C3; plan §4.6,
  decisions D-C13–D-C15 and D-C17, critique amendments A12, A14, A15).
  `notebooks/sb3_training.ipynb`: the configuration cell gains
  `WIDEN_FROM = ""` (an earlier run id under `LOG_BASE/<species>/<algo>/` or
  an absolute run directory whose certified ROOT handoff is widened to this
  checkout's policy interface) and `WIDEN_MAX_REVISION_GAP = 1` (how many
  policy-interface revisions behind that parent may be — the tool's
  fail-closed default; the two certified trex r11 stance parents need 2,
  D-C17) beside `TRUNK_FROM` / `RETRAIN_FROM` / `RUN_LABEL`, and one new
  code cell after
  the RESOLVE cell (not a `# ===== ` section marker; it calls no
  `train_stage`) that resolves the parent, refuses this run's own directory
  and a parent without `provenance.json` (`load_provenance` inside `except
  ResultBundleError`) or of another species / algorithm / backend, refuses
  a parent stage without a `stage_config.json` or a recorded run seed, and
  `SEED != ` the parent stage's recorded `run.seed` (a widened run's
  provenance publishes `training_seed = SEED` and replication counts distinct
  seeds; D-C14 — the refusal names the remedy: correct `SEED` and restart
  the runtime, since the storage cell has already minted `RUN_DIR` under the
  wrong seed and will not re-mint it), locates the parent's root stage
  directory through `stage_dir_candidates`, refuses an occupied target
  (`refuse_occupied_stage_dir(task_load_mode=None)`) and calls
  `widen_checkpoint(...)` into `RUN_DIR / stage_dirname(SPECIES, root)`
  with `label=RUN_LABEL or None`, the parent's run id and
  `max_revision_gap=WIDEN_MAX_REVISION_GAP`, printing the widen report's
  numbers (the revisions crossed included); with `WIDEN_FROM` empty it
  prints one line and
  does nothing. It runs before the infra cell, so it imports every name it
  uses. The chain loop is untouched: it refuses to reuse the verdict-less
  directory (loudly), finds the `<stage_label>_final.*` pair and JUDGES it
  under the current gate — a widened checkpoint enters through `WIDEN_FROM`
  in a NEW run id, never by pointing `RUN_ID` at the old run (D-C13). The
  infra cell's direct `load_vecnorm_stats` call carries
  `reseed_command_slice=` derived from the stage's `command_mode` and never
  set on a `resume_same_stage` load — the rule
  `train_base._load_vecnorm_into_envs` applies, since the RESUME cell resumes
  through the same `train_stage` (always false in Phase C; A12). The RESUME
  markdown says a widened checkpoint comes in through `WIDEN_FROM` in a new
  `RUN_ID`, bounded by `WIDEN_MAX_REVISION_GAP`, with `SEED` set to the
  parent's before the storage cell runs.
  `test_sb3_notebook_pins.py`: the config-cell knob pin covers `WIDEN_FROM`
  (declared, a string constant, default `""`), `WIDEN_MAX_REVISION_GAP` (an
  integer constant equal to the tool's default of 1, its comment naming
  D-C17) and the cell-6 SEED comment;
  a new `TestWidenCell` pins exactly one widen cell between RESOLVE and
  CHAIN, no section marker, no `train_stage`, the occupied-target refusal
  with `task_load_mode=None`, `load_provenance` under `ResultBundleError`,
  the `RUN_DIR.resolve()` comparison, the SEED-equals-parent check (read
  from the parent's `stage_config.json` run block, ahead of the
  occupied-target guard), no `gate_verdict` write and no write of any kind,
  the `label=RUN_LABEL or None` and `max_revision_gap=WIDEN_MAX_REVISION_GAP`
  keywords (the exact keyword set) against `widen_checkpoint`'s signature,
  the printed `policy-interface revisions crossed` line read from the
  returned result's `revision_gap`, the RESUME prose naming `WIDEN_FROM`,
  `WIDEN_MAX_REVISION_GAP` and D-C14/D-C17 for a pre-bump checkpoint,
  every name it loads imported in the cell or bound in the
  config / storage / resolve cells, and the chain loop's JUDGE precondition
  on `<stage_label>_final`; `TestCommandSliceReseed` pins the
  `reseed_command_slice` expression on the infra cell's `load_vecnorm_stats`
  call (both envs passed; the library keyword stays opt-in) and that every
  committed stage is `command_mode = "none"`; `TestWidenCellExecution`
  (SB3-gated) executes the cell's source over a narrow PPO parent and
  checks the widened root, the refused re-run, the SEED refusal, the empty
  knob and the provenance-less parent, and over one r-2 parent the refusal
  under `WIDEN_MAX_REVISION_GAP = 1` (the tool's message naming the gap and
  the flag, nothing written) and the widening under 2 with `revision_gap` 2
  in the report, the result and the printed line; the four `# ===== ` cells
  and three `train_stage` callers are unchanged. Docs:
  `docs/PLANT_CONTRACT.md` states the parity scope as the four dual-backend
  species (the compsognathus pair reports `None`), the non-zero
  `COMMAND_PROBE_VECTOR` in both probes, and a new "Widening a checkpoint
  across a policy-interface bump" subsection (the identity gate, what is
  and is not written, `WIDEN_LINEAGE_KEYS`, the reseed rule, the
  self-verification, the bounded revision gap of D-C17, JAX not widened);
  `docs/KNOWN_ISSUES.md` gains the Phase C entry beside the D-A22 inventory
  — every pre-Phase-C checkpoint refused as a trunk by reuse rule 3 (the
  task hash carries `policy_interface_sha256`) and rule 6, every existing
  `gate_resolution.json` stale, the D-A22 re-judge and D-B16 republish
  recipes DEAD for pre-Phase-C runs (the storage cell's
  `initialize_result_bundle` refuses the old directory on its recorded
  `plant_identity`; past that the JUDGE branch's `validate_model_plant`
  refuses the pre-Phase-C archive, and a republish mints an r13 provenance
  the audit rejects) and superseded in place by `widen_checkpoint` +
  `WIDEN_FROM` in a new run, the two certified stance parents
  `20260810_145546` (seed 42) and `20260815_205206` (seed 44) with the
  `1 run of 2 seeds; provisional` label until both are widened and
  re-paneled with `SEED` set per parent — and the finding that both are
  policy-interface r11 archives (trex r11 → r12 landed 2026-08-16, after they
  trained, as a fingerprint-only bump), two revisions behind r13, which the
  tool's default bound refuses, so those two sessions set
  `WIDEN_MAX_REVISION_GAP = 2` (D-C17, the remedy the entry now records in
  place of a pending decision) — and one sentence each for
  in-flight Ray sweeps (restart under a new experiment directory), JAX
  checkpoints (not widened, fail closed) and pre-Phase-C compsognathus
  recovery freezes (re-freeze; the profile hash moved);
  `docs/BEHAVIOR_RECIPES_PLAN.md` gains the §4.6 "As implemented (Phase C)"
  paragraph, the §5 Phase C row as implemented with the maintainer's
  priority (stance → recovery → walking before hunting), the §6.1 rows
  D-C1–D-C17 with the amendments folded in (D-C14 records the r11 finding
  and its D-C17 remedy; D-C17 the bounded gap), and the §10 Phase B owed
  items deferred on 2026-09-13; `docs/PLANT_CONTRACT.md` states the
  bounded-gap rule and, with `docs/BEHAVIOR_RECIPES_PLAN.md` §4.6, names the
  r11 parents and the bound that admits them;
  `docs/investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md`
  is the WS-C4 template (header, the sessions' exact knobs and commands,
  §1–§5 skeleton with placeholders, the two widen sessions under
  `WIDEN_MAX_REVISION_GAP = 2` with `SEED` per parent),
  linked from `docs/README.md` together with the eight investigation notes
  the table lacked; the root `README.md` and `website/docs/training/recipes.md`
  knob lists gain `WIDEN_FROM` and `WIDEN_MAX_REVISION_GAP`.

### Migration
- **Two new Drive directories from #540/#541.** Notebook pilot output went to
  `logs/<species>/ppo/behaviors/<behavior>/<run-id>/` (`bundle.json`,
  `run.json`, `certification/certificate.json`; never a `gate_verdict.json`)
  and the certified library to `mesozoic-labs/certified` beside `logs`.
  Nothing canonical reads either (canonical chains stopped consulting the
  library in #543; since the notebook-only PR-12 slice a pilot is a
  command-line run writing to its `--output` directory); leave them in place.
- **Pilot bundles are evaluation-only** (decision D-D9): no #540/#541
  behavior bundle is carried forward as a training parent. A pilot run is a
  command-line run (`python -m environments.shared.train_behaviors`; the
  notebook's direction/terrain mode left with the notebook-only PR-12 slice)
  and takes an explicit `--checkpoint` / `--vecnormalize` pair in every mode
  (`PUBLISH_CERTIFIED` left with PR-4, `SOURCE_SELECTION` and the certified
  library with PR-5).
- **`imageio-ffmpeg` joined the `viz` extra** (the replay recorder's video
  encoder); reinstall the extra to pick it up.

### Changed
- **CI type-checks with Stable-Baselines3 and torch installed; one ruff and
  one mypy version everywhere** (#561, CU-1 of `docs/CLEANUP_PLAN_2026_09.md`,
  decision D-D18). The lint job's mypy runs without SB3 or torch, so both
  resolved to `Any` and CI reported no issues, while the tree had 21 type
  errors locally and 23 in the SB3 job's own environment (Python 3.12, numpy
  2.5.3, torch 2.13.0, ray installed). All were type-only. They are fixed
  with casts and annotations and no runtime change: typed PPO/SAC casts in
  `StageWarmupCallback` and `EntCoefDecayCallback` and a `VecEnv` cast in
  `RewardRampCallback` (the SB3 imports are typing-only, so the curriculum
  package still imports without SB3), the widen tool's state dicts and
  VecNormalize statistics, `pad_running_stats`' return,
  `constant_action_controller` accepting the array the recovery-gate harness
  passes it, and the Ray scheduler variable; the four in the SB3-absent
  fallback of `DiagnosticsCallback.init_callback` get
  `# type: ignore[misc,assignment]`. The SB3 job pins `stable-baselines3[extra]==2.9.0` (the SB3
  notebook's pin) and runs `mypy environments/ --ignore-missing-imports`
  (mypy 2.3.1), which must report no errors; a deliberate SB3 attribute typo
  fails it and passes the lint job's mypy. The lint job,
  `.pre-commit-config.yaml` (which pinned ruff 0.4.4, whose formatter would
  have rewritten 22 files CI's ruff accepts, and mypy 1.15.0) and
  `pyproject.toml`'s `dev` extra now pin ruff 0.16.9 and mypy 2.3.1. The
  pre-commit ruff hooks cover `environments/` only, as CI does, because this
  ruff also formats notebooks and the Python blocks in Markdown files. The
  new `test_ci_tool_pins.py` keeps the pins, and the SB3 pin, in agreement.
  Pre-commit now excludes the digest data files (the MJCF plant sources and
  meshes, the recipe TOMLs, the plant manifests, `plant_versions.toml`, the
  recovery calibrations; the byte-hashed Python modules stay under the hooks,
  which CI's pinned ruff keeps from changing them): `end-of-file-fixer`
  under `pre-commit run --all-files` would have appended a newline to three
  compsognathus MJCF files and moved both compsognathus plant identities.
  `.pre-commit-config.yaml` joins the workflow's path filters, so a
  pre-commit-only pull request runs these checks.
  Every CI pytest step passes `-vv -rfEs --durations=30`, because
  `pyproject.toml`'s `addopts -q` cancelled the old `-v`, so the logs showed
  no test ids, skip reasons or durations. No digest moves: the
  digest-snapshot harness prints byte-identical output on `8e03483` and on
  this change. KNOWN_ISSUES drops the two Testing / CI entries it closes.
- **Resuming an interrupted node is "set `RUN_ID` and `RESUME_STAGE`, then Run
  all"** (#558, the notebook-safety PR, decision D-D16). The RESUME cell moved ahead
  of the chain loop, after the helpers that define `train_stage`, so it
  finishes the node's budget before the loop judges it. It used to sit after
  the loop, whose interrupted-node refusal stopped Run all first, so a resume
  took three hand-run steps. The sections regroup around the run: 3 "Open the
  run and resolve the chain" (the storage and resolve cells; the old "Explore
  the Environment" header covered chain resolution and trunk selection), 4
  "Preflights" (the archive load and the zero-action baseline), the helpers
  without a number, 5 Resume, 6 Train the chain, 7 Manual node, 8 Report
  (evaluation, curves, videos and the bundle check under one header) and 9
  Auto-disconnect. The section references in the cells, their messages, the
  knob comments and the website recipes page follow. Measured: 33 cells (17
  code) before and after; 1,515 lines before, 1,557 after (the guards below
  and their prose).
- **The auto-trunk tie-break is described as it runs** (#557, the notebook
  follow-up to PR-7). `select_trunk` breaks a coverage tie by the greatest
  run directory name, which is the newest run only among names of one
  timestamp form; a custom `RUN_ID` such as `my_run` sorts after every
  timestamp id. The notebook's configuration prose and knob comment, the
  resolve-cell comment, the `--trunk-from` help, the `select_trunk` /
  `train_curriculum` docstrings and the website recipes page said "newest on
  a tie"; they now name the rule (decision D-A25 gains a clarification). The
  storage cell's trunk comment and the recipes page no longer say a trunk must
  be a "finished bundle": a pinned trunk needs a `provenance.json` naming the
  same species, algorithm and backend, neither a pinned nor an automatic trunk
  is gated on its bundle status, and each node is reused only when the reuse
  rule certifies it. No selection changes.
- **Species heights are measured above the ground under them** (#556, consolidation
  PR-7). `BaseDinoEnv` gains `_ground_height_at(xy)`, the authored plane's 0.0,
  and `_clearance(xyz)`, `z` minus it. Every species height reward,
  head/snout clearance and height termination reads through it, as do the step
  loop's substep minima of the trex head tip and skull and the dibothrosuchus
  snout; velociraptor's pelvis termination is included (the plan's list missed
  it). On the plane nothing changes: 16 seeded episodes per species, compared
  observation, reward, termination and every info value, are bit-identical to
  the previous commit, and `plant_contract --check` reports the manifest
  current. The behavior env overrides only `_ground_height_at` (off the map, 0.0
  as before) and loses its four height overrides and its re-derived height
  reward. That block repeated the formula of each species it served
  (brachiosaurus, dibothrosuchus, compsognathus); it would have used 0.3129 m
  for trex, which had its own class and never reached it. No other species'
  behavior reward or termination moves: six seeded episodes per species, moved
  8-12 m onto 6° slopes or 0.08 m bumps, score and end as before
  (compsognathus's reward within 4.4e-16). On heightfield episodes every
  species' `pelvis_height` and `torso_height` info (and the dibothrosuchus
  `snout_tip_z`), and with it the behavior env's `mean_pelvis_height` metric,
  is now a clearance rather than world z. The behavior env's root height
  check is now the step's final sample, as in the canonical envs and MJX,
  instead of the substep minimum, then maximum. T. rex joins the generic
  behavior class: its floor-contact reasons become categorized
  (`tail_contact`, `head_contact`, `torso_contact` rather than
  `body_contact`), the terrain boundary is the
  geom-bounds rule every species uses (it was `|x|` or `|y|` ≥ extent − 3 m),
  `head_clearance_m` leaves the step info (`head_tip_z` and `pelvis_height` are
  the clearances), and the neck probe ends an episode as `neck_ground_contact`
  after the canonical checks. The mixin also zeroes `bite_bonus`,
  `bite_approach_weight` and `bite_head_proximity_weight`, as the trex class
  did. `get_behavior_env_class` caches by resolved species id, so a display
  name and its id return one class, as `get_sampled_behavior_env_class`
  already did. Behavior identities change for every species (the trex source
  digest leaves `sources`; trex gains `species`, the tracking tolerances and
  repository-relative source keys), so exact resume, `--adapt` and saved-panel
  evaluation refuse bundles trained before this change. Those bundles were
  already evaluation-only (decision D-D9). Over 60 seeded
  random-action episodes per species, a canonical env built from the behavior
  env's own parameters and the behavior env with `terrain=None` agree on every
  state, reward term, termination and reason; reward minus tracking agrees
  within 1.4e-14.
- **The SB3 notebook trains through `train_base.train`** (#555, consolidation PR-14c,
  decisions D-D7 and D-D11). `train_stage` keeps its signature and 6-tuple and
  becomes its argument refusals, the node banner, one
  `train_base.train(..., report_metrics=False, save_on_interrupt=False)` call
  and the evaluation, instead of a 384-line copy of `train()`'s body.
  `train()` now seeds model construction with `seed` unless the stage's
  algorithm block names one (`--override ppo.seed=N` still wins), so CLI
  `train` runs and Vertex sweep trials (`sweep/trial.py`) start from other
  initial weights than before and their archives record the seed; it also
  records `run.duration_seconds` at the final save, added to the value the
  stage directory already records on a same-stage resume (D-A15). `curriculum`
  runs and Ray Tune trials neither seed model construction nor record a duration. Four
  loads pass `seed=None` so SB3 does not re-seed their env with a seeded
  archive's training seed: the CLI's post-training panels (`metrics.json`) and
  the `task_success/v1` evidence re-roll keep their evaluation seeds (the
  re-roll's rows record the publication seed, which a notebook archive's load
  used to replace), and a Ray trial warm-started from a seeded archive keeps
  its own seed. `train()` takes `parent_run_id`, an explicit `vecnorm_path`,
  `report_metrics` and `save_on_interrupt`, whose defaults keep the CLI's
  behaviour; the notebook's interrupt still propagates before the final save,
  so a stopped node is finished by the RESUME cell, and a missing named
  VecNormalize sidecar is still refused, now before anything is written. `evaluate_stage_checkpoints`
  moves into `environments.shared.reporting` with the notebook's globals as
  parameters, which `train_stage` and the chain loop's JUDGE branch pass. What
  the notebook writes changes in three places: a node entered from its parent
  no longer copies six warmup/ramp values into its run block (no code reads
  them; a value its TOML declares stays in the `curriculum` block, and
  `forward_vel_weight` in `reward_weights`, while an undeclared key's resolved
  default is no longer written out); on Drive, TensorBoard events sync on
  the checkpoint cadence, so an interrupted node's events up to its last
  checkpoint reach the stage directory and a RESUME on a fresh runtime
  continues that session's run directory (`PPO_1`) instead of opening `PPO_0`;
  a refused declared parent no longer leaves an empty `models/` directory.
  Every other file a node writes is unchanged apart from the recorded
  duration's timing (old and new paths compared under a frozen clock, on and
  off a simulated Drive mount). The notebook goes from 2,079 to 1,492 source
  lines.
- **The SB3 notebook has one disconnect path, in
  `environments/shared/notebook_runtime.py`** (#554, consolidation PR-14b, decision
  D-D15). `disconnect_runtime(reason, *, in_colab, auto, flush_drive)` moves out
  of the infrastructure cell, where it read the globals `IN_COLAB`,
  `AUTO_DISCONNECT` and `USE_GOOGLE_DRIVE`; the auto-disconnect cell passes the
  three knobs when it runs. The chain loop's gate refusal, the one
  disconnect-then-raise site left, calls the new `halt` with the same knobs,
  which releases the runtime and then raises the same `RuntimeError`.
  `display_stage_videos(stage_dir)` moves beside them and plays each replay
  mp4 through IPython's `Video(..., embed=True)` instead of decoding and
  re-encoding it with mediapy (still muted, autoplaying and looping, as
  mediapy's player was); it loses the `stage` parameter that only the
  mediapy-missing message read. `google.colab` and `IPython` are imported only
  inside the calls, so the module is tested in the shared matrix
  (`test_notebook_runtime.py`, with fake modules). mediapy stays installed for
  `record_stage_video`.
- **The zero-action cell is its knobs and one call** (#554, PR-14b). Its measure and
  verdict loop, table and saves move into
  `environments/shared/scripts/zero_action_baseline.py` as `preflight(...)`
  (`report()` stays the command-line printer). The
  `mesozoic.zero-action-baseline/v1` payload, the table, the file paths, the
  run copy that `curriculum.baseline_watch` reads and PR-14a's skip of that
  copy on a complete bundle are byte-identical (compared under a frozen clock
  for all five verdicts, one and five species, and each bundle state).
  `test_zero_action_baseline.py` still executes the cell by its marker, now
  patches `preflight`'s module and also calls `preflight` on two species under
  a frozen clock (the table, each file and the run copy). With the Removed
  entry below, the notebook goes
  from 34 cells (18 code), 2,271 source lines to 33 cells (17 code), 2,079.
  Measured with `git diff --numstat` against `3571b24` (new files counted
  whole): code +183 / −1, tests +245 / −50, notebook JSON +27 / −229, docs
  +131 / −37.
- **`RUN_ID` is a configuration-cell knob of the SB3 notebook** (#553,
  consolidation PR-14a, decision D-D15). `RUN_ID = ""` keeps the run this runtime's storage
  cell resolved last (a fresh timestamped run on the first pass), a new id
  starts a fresh run, and an existing run's id re-enters it in place (a run
  id, never a path: a path, `.`, `..` or surrounding whitespace refuses). The
  storage cell resolves it into the `_ACTIVE_RUN_ID` memo (the knob, else the
  run this runtime resolved last, else a timestamp; an explicit id beats the
  memo and is memoised in turn once `initialize_result_bundle` accepts it, so
  an id whose provenance refuses the session never becomes the memo), no
  longer rebinds `RUN_ID`, and prints whether it minted a new run or
  re-entered one (its directories and bundle status);
  `initialize_result_bundle` takes the resolved id and `save_run_bundle` reads
  `_ACTIVE_RUN_ID`, so re-running the configuration cell, which now resets the
  knob to `""`, never hands `run_id=""` to a bundle write. A `TRUNK_FROM`
  naming the run `RUN_ID` resolved to names `RUN_ID` as the knob to change.
  `TestStorageCellRerun` now runs the cell under a clock that ticks per call,
  so only the memo can keep a run (at 2ebed89 two executions in the same
  second minted the same id and the test could not tell), and pins that an
  explicit id beats the memo and that a refused id never reaches it.
- **A root widened on the command line keeps the widen cell's guards on
  disk** (#553, PR-14a; decisions D-C13, D-C14, D-D14). The storage cell refuses a
  `SEED` other than the recorded `run.seed` of a stage directory whose run
  block names `widened_from_run_id` (and a `provenance.json` minted under
  another seed before the widen, which no `SEED` can fix: the refusal says to
  widen again into a new run id) before it binds `RUN_DIR` or the memo and
  before `initialize_result_bundle` writes anything
  (`result_bundle.refuse_widened_seed_mismatch`). The resolve cell refuses a
  resolved trunk until that root holds a verdict
  (`refuse_trunk_over_unjudged_widened_root`), so the chain loop judges the
  widened copy instead of reusing the root from another run.
  `TrunkSelection.describe()` names the command-line tool and the
  `--max-revision-gap` an older-interface root needs instead of the deleted
  knobs (the CLI's `--trunk-from auto` prints it too), and
  `widen_checkpoint`'s docstring and `--to-stage-dir` help give the recipe the
  notebook then judges.
- **The `test-sb3` CI job is bounded** (consolidation PR-3, #546, 2026-09-20). It
  ran 52 minutes on the #544 merge (notebook smoke 6, behaviors 14,
  integration 31; 69 at #541) because #540/#541 added their suites to its
  lists, ten of which import no SB3 and already run three times in the
  `test` matrix (`test_behavior_env`, `test_behavior_recipes`,
  `test_terrain_sampling`, `test_behavior_evaluation`,
  `test_behavior_certification`, `test_behavior_publication`,
  `test_certified_library`, `test_certified_comparison`,
  `test_sb3_notebook_certified`, trex `test_behavior_env`; verified locally
  with SB3, torch and ray blocked), and because every real-PPO smoke ran for
  all six species and all four notebook parameters on every pull request.
  Those suites leave the SB3 lists; pull-request and push runs keep one
  real-PPO smoke per body of work (the `compsognathus_robot` parameter of the
  notebook training smoke and of the species training smoke); the full sets
  run on a new nightly schedule (`05:17` UTC), on `workflow_dispatch`, and on
  any pull request carrying the `full-ci` label, all inside the same job so
  the required-check name is unchanged (the `pull_request` trigger gains the
  `labeled` activity type, so adding the label to an open pull request starts
  the run that reads it; the label itself is created once on the repository).
  The `walker` fixture of
  `test_behavior_species_training.py` is module-scoped (6 PPO builds instead
  of 18). Measured on #546's two CI runs: the lean job 48:02 (smoke 2:56,
  behaviors 4:59, integration 38:00 for 1,384 tests), the labelled full job
  45:21 (5:51, 9:01, 28:39); the union coverage gate (`fail_under = 70`) read
  90 percent. The integration step is now the whole cost and the JAX job (39
  to 51 minutes) the longest pull-request job; both are follow-up candidates.
  The same change records the maintainer's 2026-09-20 decisions in the plan
  documents: the consolidation hold lifted in a notebook-first order (D-D13),
  D-D11 and D-D12 confirmed, and the widen path kept for the two pending
  parents then demoted to CLI-only (D-D14).
- **`stage_manifest.KNOWN_STAGE_IDS` is renamed `RESERVED_STAGE_IDS`, with
  no alias** (Phase A, WS1). Ids are an open vocabulary
  (`^[a-z][a-z0-9_]*$`); the four historical ids stay reserved, `stage{N}`
  is refused as an id, and species-free readers (`detect_stage_from_path`,
  the sweep collector) recognise only the reserved ids while species-aware
  readers accept any declared id (decision D-A12). Importers of the old name
  fail at import time.
- **Ancestor reuse is chain-aware and never reuses the target** (Phase A,
  WS2 follow-up). `find_certified_ancestor` gains rule 4: a non-root
  candidate must record, in its `stage_config.json` run block, an
  `initialize_next_stage` load whose `parent_checkpoint_sha256` equals the
  digest of the checkpoint resolved for its declared parent (the new
  `parent_model_sha256` argument), and a root candidate must not have
  entered from a parent at all; ids are never a substitute for the
  digests, and an unresolved parent refuses the child. `curriculum
  --trunk-from` resolves each node's parent before consulting the trunk,
  never looks up a child of a node trained in the same run (nothing earlier
  descends from a checkpoint the run just produced), and never reuses the
  run's target — the last advancing stage is always trained, and an earlier
  run's certified target is that run's deliverable. Every non-reuse is
  logged with its reason.
- **The SB3 notebook's opt-in recovery pilot now shapes the bundle status**
  (Phase A, WS3 consequence). Under result schema v4 recovery is a
  deliverable, so a recovery that fails its frozen gate leaves the run
  bundle `partial` — the certified numbered stages still publish — instead
  of riding along without affecting the status. The completion cell reports
  a publishable-but-partial bundle, naming the uncertified deliverable,
  rather than failing the run; the chain loop that replaces the per-stage
  cells enforces the verdict outright. With #530 the pilot is gone: recovery
  runs as the second node of the `stand` recipe (`BEHAVIOR = "stand"`) and
  its frozen verdict is enforced like every other node's (decision D-A11).
- **SAC gets the stage-entry warm-up in the notebook** (Phase A, WS5
  notebook; plan assumption A9). The chain loop's `train_stage` builds its
  shaping through `_stage_entry_shaping_callbacks` unfiltered, so a SAC node
  entering from its parent under `initialize_next_stage` receives the same
  warm-up clamps and forward-velocity ramp the CLI gives it (closes
  gap-review DU1).
- **`train_curriculum` records no gate verdict for an interrupted node.** A
  Ctrl-C partway through a stage's budget used to write a FAILED
  "budget exhausted" `gate_verdict.json`, which every later `--trunk-from`
  would have read as a genuine gate failure; the node is now left unjudged,
  with a warning to resume or re-judge it before reuse.
- **`upload_curriculum_artifacts` mirrors `ancestors/<stage_id>/` records**
  (the reused node's `ancestor.json` and copied verdict, config, fingerprint
  and plant identity — never a checkpoint): the bundle audit requires them,
  so a GCS mirror of a `--trunk-from` run could not audit without them.
- **Cleanup batches from the gap review** (Phase K,
  `docs/reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md` §6 — every entry
  re-verified against the tree after the five fix phases; no behavior
  change beyond the three alignments called out below).
  - One home each: species configs (`species_registry.get_species_config`
    replaces the inline copies in the four `train_sb3.py` scripts and the
    sweep trial), env construction for the report scripts
    (`config.build_env` / `config.SPECIES_NAMES`), the publication seed
    (`constants.PUBLICATION_SEED_START`), the stance gate's 40-episode
    minimum (`DEFAULT_MIN_EVAL_EPISODES_STANCE`) and the SB3 curriculum
    path's 10-episode fallback (`curriculum.manager.DEFAULT_MIN_EVAL_EPISODES`),
    the stance gate's curriculum mapping
    (`StanceGateThresholds.from_curriculum`, replacing four copies; the SB3
    curriculum manager keeps its own field copy,
    `StageThreshold.stance_thresholds()`, because its 10-episode
    `min_eval_episodes` default differs), the stage-entry shaping block
    (`train_base._stage_entry_shaping_callbacks`, now used by the Ray Tune
    worker and the SB3 notebook — so the notebook now honours
    `[curriculum] warmup_lr_scale` like the CLI and, like the CLI, applies
    no shaping to an explicit `initialize_next_stage` call that has no
    load path), the fake test plant
    (`reporting_helpers.make_plant_identity`, replacing eight copies), and
    the SB3 checkpoint + VecNormalize loader shared by the report scripts
    (`policy_loading.load_sb3_checkpoint`; the three diagnostic scripts now
    report a corrupt sidecar with the certification tool's diagnosable
    message instead of a raw unpickling traceback, still exit 1).
  - Moves: `select_handoff_checkpoint` to `curriculum.checkpoints`; the
    notebook-path PPO loop (`train`, `TrainConfig`, `TrainResult`,
    `_build_jit_fns`) to `jax_train_fn.py`, re-exported from `jax_trainer`;
    the stance report builders out of `scripts/stance_gate_report.py` into
    `reporting/stance_report.py` so library code no longer imports from a
    script. `roll_recovery_panel` passes the height reference to
    `_safe_step` as a parameter instead of stamping the env.
  - Retired markers: the "provisional until P5" text in `recovery_gate`,
    `gate_schema` and the curriculum manager; KNOWN_ISSUES entries the stance
    gate overtook (pruned per the file's own policy); the website's
    hyperparameters tree and trex page; the renamed `stage1_balance.toml`
    pointer in `trex_env.py` (the twin in `trex.xml` stays: the plant
    manifest hashes that file); `quality_scoring.toml`'s falsified TODOs;
    the two "plan §W5" pointers on the task-fingerprint valves, now tracked
    in the stage-1b plan.

- **The T-Rex recovery gate is frozen: `recovery.toml` declares
  `recovery_quality/v1`** (plan P5, `docs/STAGE1B_IMPLEMENTATION_PLAN.md` §4;
  measurements in `TREX_RECOVERY_STAGE_FIRST_RUNS_2026_08.md` §4 and §9). The
  `none/v1` placeholder is gone and the stage's `[curriculum]` block carries
  measured thresholds: the P3 calibrated **posture-only** safe set (height
  error ≤ 0.0168 m against the *fixed* measured settled reference 0.9267 m
  rather than the per-episode reset stamp, tilt ≤ 0.0825 rad, planar speed
  ≤ 0.3203 m/s, and **no per-step support term** — quiet certified stance
  itself transiently reads 0.0 N on a foot, so that criterion failed the very
  thing it certifies and had masked 13 of the trained policy's 21 real
  recoveries), `recovery_t_recover_steps` 100 and `recovery_dwell_steps` 50
  (validated rather than retained: measured re-entry p90 = 84 steps), the
  registered 40-episode panel from seed 3042, `min_recovery_success_lcb`
  **0.30**, and `min_paired_success_delta_lcb` **0.20**. Both thresholds are
  **attainable, not aspirational** — measured episode-success LCB95 0.361 (3M)
  / 0.338 (5M) against a statue null whose UCB95 is 0.072, and paired
  policy-minus-statue LCB95 +0.365 / +0.340 — so the plan's aspirational 0.725
  (the LCB of 34/40) is deliberately not frozen, and what the stage cannot yet
  do is recorded instead: the capability ceiling sits between 165.5 N and
  210 N, where the paired margin collapses to LCB95 −0.009 / +0.019. Flipping
  the kind moves no verdict into the config — advancement still requires a
  `gate_resolution.json` in the stage directory (produced by the
  `freeze_recovery_gate` harness, enforced by `curriculum/gate_resolver.py`,
  cross-checked against the TOML declaration), and its absence, staleness, or
  tampering still refuses, fail-closed. The config's header stops claiming the
  stage is not runnable and records the envelope instead. New
  `environments/shared/tests/test_recovery_gate_config.py` pins the frozen
  declaration, recomputes every threshold's bound from the §9 panel counts so
  a silently edited number has to disagree with the measurement, and holds the
  no-resolution-no-pass guard.
- **Stage-1 review cleanup, final batch (§2.3)** — the cosmetic and
  small-code items closing out the 2026-08-15 review's cleanup list:
  - **Evaluation CSVs: `success` column renamed `task_success`** (gate-pass
    postmortem, follow-up 1). The column records the stage's TASK event
    (bite/strike/food-reached), which a stance-gated stage can never emit —
    it reads False in every stage-1 row by construction and was repeatedly
    misread as the gate verdict, which lives only in
    `stance_gate_report.{txt,json}` and `stage_summary.txt`. The bundle
    reader accepts the legacy `success` header so pre-rename bundles still
    audit (pinned by a test); the writer emits only the new name.
  - **The biped-empty "diagonal pair" contact panel is fixed**
    (narrow-tolerance postmortem, item 6). `plot_foot_contacts` keyed its
    quadruped layout off key *presence* in `diagnostics.npz` — but
    `DiagnosticsCallback` saves every info key for every species, so a
    biped's file contains `rr/rl_foot_contact` as all-NaN series: the figure
    grew an empty diagonal panel and routed the two real feet through the
    four-foot branch under FR/FL labels. Detection now requires finite data;
    a regression test pins the all-NaN-keys biped case the existing
    omitted-keys fixture could never catch.
  - **`stage1_balance.toml` comment corrections**: the release-ablation cost
    justification updated from the r6-era "13 panels" to the current 17
    (3 controls + 2 × 7 matched groups; toes skips); the self-contradictory
    `posture_weight` ("Increased from 1.5") and `nosedive_weight`
    ("Increased from 3.0", a decrease) provenance claims replaced with
    honest notes — both date from the file's first commit and compare
    against unrecoverable pre-repo values; the `[sac]` block gains a
    staleness warning and corrected cross-references (its "match PPO"
    lr/gamma/budget comparisons were written against a [ppo] block that has
    since moved to 3e-5 / 0.98 / 10M — the block itself has never been run);
    `ent_coef_end`'s entropy-floor analysis is era-marked as the r6
    21-actuator measurement, with a note that the figures still hold on r7.
  - **`_actuator_pose_mapping`'s docstring** now describes the deleted toe
    actuators in the past tense with the r6/r7 framing
    (plant_versions.toml note 9) instead of presenting an impossible
    example as current.
- **KNOWN_ISSUES pruned against the r7/r11 campaign** — the §2.2 batch of the
  2026-08-15 review (`docs/STAGE1B_IMPLEMENTATION_PLAN.md`), applying the
  file's own policy ("when an item here gets fixed, delete it"). Three
  T-Rex stage-1 entries deleted, each fixed or falsified by commits already
  on `main`; their evidence lives on where cited:
  - **"A passing policy cannot stand if its actions are filtered"** — the r6
    measurement behind "an action filter cannot be retrofitted." The r11
    10 Hz command low-pass (#503) did what the entry prescribed — filter
    present during training, unfiltered checkpoints invalidated — and the
    certified policy survives every probed cutoff down to 5 Hz at 96% of
    reward (`TREX_STAGE1_GATE_PASS_RUN_2026_08.md` §4; original analysis
    archived in `TREX_STAGE1_BOUNCE_2026_08.md` §5).
  - **"The passing policy stands in a pose it cannot hold without continuous
    feedback"** (HIGH) — its stated closure condition (re-run the
    hold-constant probe on an r7-trained policy) was met: 3/10 episodes to
    the full horizon, mean 611 steps, gentle nosedives — near-statically
    stable, measured on one seed (gate-pass postmortem §4). The entry's
    bolded advice that raising `smoothness`/`action_jerk` is contraindicated
    described the deleted r6 pose and was empirically falsified by the
    0.1→2.0 smoothness escalation the certified run trained under.
  - **"Half the actuators sit saturated and nothing opposes it"** — the #504
    `action_saturation` penalty (0.5 / 0.9) now opposes it, and the r7
    counts were re-measured: 5 of 15 parked in the narrow-tolerance run,
    **0 of 15** in the gate-pass run (max |DC| 0.890, parked under the
    ramp). One-sided-headroom residue stays tracked in #491.
  - The surviving **"passes or bounces"** entry gained a dated update: its
    numbers are r6-era, and the seed-replicate ask now targets the r11
    gate-pass configuration (n = 1).
  - The **latent raw-vs-clipped reward trap** paragraph was rewritten with
    function-level anchors (its `base_env.py` line numbers had rotted by
    ~250 lines) and scoped to filter-free species — r11's `_filter_action`
    clips before the reward terms read the action, closing it for the T-Rex.
  - Same-family staleness in `docs/hardware/SIM_TO_REAL_PLAN.md` §3.3 fixed:
    two rotted `base_env.py` line anchors re-pointed at
    `BaseDinoEnv._scale_action` / `__init__`, and "no action filtering
    anywhere in the step loop" corrected — the T-Rex has filtered in-loop
    since r11; the other species still do not.
- **Stage-1 closeout cleanup** — the four pre-training-run items from the
  2026-08-15 review (`docs/STAGE1B_IMPLEMENTATION_PLAN.md` §2.1):
  - **`DiagnosticsCallback`'s raw saturation metric renamed
    `diagnostics/raw_action_saturation`** (was `diagnostics/action_saturation`).
    The #504 pack gave the env an `action_saturation` info key — the
    post-filter ramp fraction behind `reward_action_saturation` — which the
    callback recorded at rollout end and then **overwrote in the same pass**
    with its pre-clip |a| ≥ 0.99 fraction under the identical TB key. The two
    disagree by construction (pre-clip raw sample vs filtered command; 0.99
    hard threshold vs 0.9 ramp). `diagnostics/action_saturation` is now
    unambiguously the env's; dashboards tracking the raw quantity should
    follow the rename (TB traces from runs between #504 and this fix carry
    the env value only until the first rollout end, then the raw overwrite).
    `action_bound_report.py`'s cross-references updated to match.
  - **T-Rex filter-probe sweep re-derived for the r11 plant**:
    `stance_probe_filter_hz` `[5, 10, 20, 30, 35]` → `[1.0, 2.5, 5.0, 8.0,
    10.0]`. The old sweep predates the plant's own 10 Hz command low-pass —
    probe cutoffs above 10 Hz stack a wider filter onto a narrower one and
    measured approximately the unprobed checkpoint (30/35 Hz were
    near-duplicate rows of the gate panel at 10 episodes each). The gate-pass
    checkpoint also survives every old cutoff, so the informative region
    moved **below** the plant cutoff; the new sweep locates the degradation
    knee there, keeping 10.0 as the cross-run continuity point and
    stack-sanity row. Same five-panel cost.
  - **Release-ablation group test extended to the r7 groups**:
    `test_every_group_gets_both_directions` now covers `knees_ankles`,
    `left_leg`, and `right_leg` — the #501 groups the passive-toes run's
    knee-localization conclusion rests on, which had no coverage — and the
    `_TREX_ACTUATORS` fixture gains `l_knee`/`l_ankle` so the per-side
    groups resolve more than a hip roll.
  - **Shaping-pack constants pinned exactly in the stage-1 factory test**:
    `tail_home_pose_weight` 0.25 / `tolerance` 0.05, `action_saturation_weight`
    0.5 / `threshold` 0.9, `leg_home_pose_broad_fraction` 0.25 /
    `broad_scale` 6.0 (was `> 0.0` for three of them, while every neighboring
    assertion pinned exact values). The statue rails (2100 / 3495.2) were
    measured at exactly these constants, so a silent TOML drift on any of
    them invalidated the rails without failing a test.
- **T-Rex stage-1 reward shaping pack** — three measured responses to the
  knee-lock run (`docs/investigations/TREX_STAGE1_NARROW_TOLERANCE_RUN_2026_08.md`),
  each opt-in with zero defaults in both backends:
  - **`action_saturation` penalty** (weight 0.5, threshold 0.9): prices
    commands parked past 90% of their range. A saturated command cannot
    oscillate, so it was invisible to smoothness/jerk — the run parked six
    of fifteen actuators at hard stops. The parked stance now pays
    ~−200/episode; the statue commands zero and pays nothing.
  - **`tail_home_pose` term** (weight 0.25, tolerance 0.05 rad): prices tail
    *joint positions*; the existing tail term reads only tail-tip angular
    velocity, which a tail frozen against a stop satisfies perfectly.
    Targets are the measured **settled droop**, not the keyframe — the
    passive tail rests on its ventral stops under gravity
    (−0.2107/−0.2029/−0.0926 rad with sub-milliradian spread, 40 seeds),
    so the authored zeros are unreachable and would cap the ideal statue at
    ~0.25 quality (`TRexEnv._TAIL_SETTLED_QPOS` = MJX registry
    `tail_home_pose_targets`, pinned equal by the stage-1 factory test).
  - **Long-range gradient for `leg_home_pose`** (`broad_fraction` 0.25,
    `broad_scale` 6): mixes a 6×-wider Gaussian into the term so distant
    poses still feel a pull. The narrow Gaussian's gradient is zero a few
    widths out, which the run demonstrated by drifting monotonically from
    0.20 to 0.545 rad of home error without ever being pulled back; at the
    measured crouch distance the mixture still pays 0.44 quality with live
    gradient. At home both components are 1, so the maximum is unchanged.
  Statue rails re-derived with the pack (zero_action_baseline.py seed 3042,
  40 episodes, noise 0.05; stance_quality_baseline.py agrees): statue
  3241.3 → **3495.2 ± 13.9** (+250 tail term at its settled targets, ~+4
  broad-leg at the statue's own sag, +0 saturation), `min_avg_reward`
  1940 → **2100** (0.60×, nearest 10), `collapse_peak_floor_reference`
  3241.3 → **3495.2**. Physics stays r7; `statue_constants_physics_revision`
  unchanged.
- **T-Rex commands are low-passed at 10 Hz during training and evaluation**
  (breaking — plant change, policy interface revision 10 → 11; all existing
  T-Rex checkpoints are invalidated; physics revision stays 7 and the
  statue-derived constants carry, since the statue's zero action filters to
  zero). Both backends now run the commanded action through a first-order
  low-pass (`environments/shared/action_filter.py`, the same discretization
  as the stance-gate probe filter), seeded with the first post-reset action;
  the dynamics **and** the action-derived reward terms (energy, smoothness,
  jerk) consume the filtered command. The cutoff is plant-level by
  construction — an SB3 class attribute and a `register_species_mjx` field
  guarded by `_PLANT_INTERFACE_CONFIG_FIELDS` — so stage TOMLs cannot tune
  it, and the plant contract records the cutoff plus the filter fingerprint
  and asserts backend agreement. Motivation
  (`docs/investigations/TREX_STAGE1_NARROW_TOLERANCE_RUN_2026_08.md`): both
  2026-08 stage-1 runs converged on statically unstable poses stabilised by
  16.8–18.7 Hz command chatter — the narrow-tolerance run locked both knees
  against their stops and pumped the ankles ±28–33°, and its constant-hold
  probe fell backward onto the tail in 9/10 episodes within ~3 s — while
  the filter probe puts the balance task's real bandwidth at ~1.1–1.4 Hz.
  Amplitude penalties cannot price the strategy out (a saturated command is
  perfectly smooth); removing the bandwidth makes such poses unreachable
  attractors instead of cheap ones. Other species keep the exact legacy
  step arithmetic (cutoff 0.0, byte-identical policy fingerprints).
- **T-Rex stage 1 `leg_home_pose_tolerance` narrowed 0.20 → 0.10 rad**, and the
  two statue-derived constants re-derived with it (`min_avg_reward`
  1950 → 1940, `collapse_peak_floor_reference` 3270.3 → 3241.3, both from a
  fresh 40-episode zero-action baseline at seed 3042). Motivated by the
  passive-toes run postmortem: the 10M checkpoint stands on a pose the
  release ablation convicts (left knee 0.21 rad, left ankle 0.17 rad off
  home) that the 0.20-wide Gaussian priced at only ~29 of the term's 500
  points, while the chatter stabilising that pose forfeits ~1030 in the
  support-linked terms. The width was chosen from measurement, not taste:
  0.10 doubles the statue-vs-chatterer pricing differential (109 → 180
  points) while staying 2× above the statue's own p99 settle error
  (~0.05 rad hip-pitch gravity sag); 0.075 and 0.05 sit past the knee of
  the curve, paying −26/−83 more statue reward for +23/+33 differential.
  The statue remains the reward optimum with a wider margin over the
  measured chatterer (~1128 vs ~1057 points).
- **T-Rex toes made passive** (breaking — plant change, physics revision 6 → 7
  **and** policy interface revision 9 → 10; all existing T-Rex checkpoints are
  invalidated). The six per-digit `<position>` actuators (kp 60, forcerange
  ±50 N·m, ctrlrange −0.4363..0.8727) are deleted; the digits keep their
  hinges, and the `leg_joint` class already supplies the passive stance —
  stiffness 40 with `springref` 12.5° holds the old commanded home pose,
  damping 15 dissipates strike energy. Action dimension drops **21 → 15**
  (3 neck/head + 8 legs + 4 tail); the home-keyframe ctrl vector shrinks with
  it. The observation is **unchanged at 61** — toe joints stay in qpos/qvel,
  and all 16 sensors (including the six per-digit touch sensors) keep their
  indices. `visual_revision` deliberately stays 4: no geom, site, material,
  camera or light moved, and the visual layer carries no actuator or keyframe
  fields.
  Motivation is the stage-1 bounce postmortem
  (`docs/investigations/TREX_STAGE1_BOUNCE_2026_08.md`): the release ablation
  showed holding **only the six toes** at their commanded DC reproduces the
  whole constant-hold failure (125.5 steps vs the full pose's 128.6) while
  tail, neck and head held alone all stand the full horizon — the policy's
  learned 0.8727 rad claw curl splayed adjacent digits to opposite ends of a
  75° range, changing the support polygon. A splayed digit is a pose the
  passive spring cannot hold, so the failure mode is now structurally
  impossible rather than merely unrewarded.
  Statue re-measured on the new plant (`zero_action_baseline.py` seed 3042
  and `stance_quality_baseline.py 0.05 40` agree): standing reward
  **3270.3 ± 12.3**, 40/40 full horizon, unsupported duty 0.000 — within one
  standard error of the r6 figure 3271.8, as expected with the springs
  holding the same pose the servos did. `collapse_peak_floor_reference`
  updates to the measured 3270.3; `min_avg_reward` stays 1950 (0.60 × the
  statue, not re-rounded over a within-noise move). The settled pelvis height
  is unchanged at 0.9260, so `target_z`, `natural_pitch` and the nosedive
  threshold keep their pins.
  Also folded in, completing the substep-aggregation lockstep: the SB3
  site/body **height terminations** (T-Rex `head_tip_z` < 0.12 / `skull_z` <
  0.45, dibothrosuchus `snout_tip_z` < 0.04) now consume the per-substep
  MINIMUM height recorded by the step loop, so a head dip that recovers
  between control-boundary samples terminates exactly as MJX's any-substep
  height emulation does; the `head_tip_z`/`snout_tip_z` info keys keep
  reporting the boundary sample.
  The statue constants get the freshness guard their comments have begged
  for since #491: the stage TOML now records
  `statue_constants_physics_revision` = the plant revision the constants
  were measured on, and `test_statue_constant_freshness.py` cross-checks it
  against the manifest — so the NEXT plant bump cannot land without either
  re-measuring the statue or consciously updating the pin where a reviewer
  sees the constants did not move. And the action-filter probe sweep gains
  the 30/35 Hz cutoffs (`stance_probe_filter_hz`), closing the survival
  curve at the control Nyquist's edge instead of stopping at 20 Hz with the
  measured 22.7 Hz tremor unsampled above it.
- **Reverted T-Rex stage 1 `leg_home_pose_weight` to 0.5**, and the two statue-
  derived constants with it (`min_avg_reward` 2550 → 1950,
  `collapse_peak_floor_reference` 4250.4 → 3271.8). The 1.5 experiment **could
  not have worked**, and the instrumentation added alongside it says why
  (issue #491).
  `leg_home_pose` governs eight joints — r/l `hip_pitch`, `hip_roll`, `knee`,
  `ankle` — and measured per actuator, those eight carry **1.2% of the
  policy's commanded pose offset**. The other 98.8% is in the tail, neck, head
  and toes, which no term in this stage touches; 10–12 of 21 actuators sit
  pinned at `|action| = 1.000`. The governed joints were **already** near home
  (|DC| 0.035–0.263), so tripling their weight had nothing to pull on.
  Measured, it didn't: commanded DC moved **0.766 → 0.738**, 3.7%, while the
  run took roughly 2M extra steps to escape its early collapse. The joint
  *list*, not the weight, is the problem.
  What survives the revert is the discipline: both constants are documented as
  derived from the statue and neither updates itself, so any reward-weight
  change has to re-measure with `zero_action_baseline.py` and re-derive both.
  Because the statue commands `action = 0` at any weight its trajectory never
  changes and only the affected term rescales — which makes the new value
  exact and cheap to obtain, and makes forgetting it silent.
  `stance_probe_filter_hz` is **kept**, and widened from a single cutoff to a
  **sweep** — `[5.0, 10.0, 20.0]` on trex 1a. A single cutoff answers a yes/no
  that is already answered: the checkpoint that PASSED the gate falls at every
  cutoff from 5 to 35 Hz against a 100 Hz control rate, so its PASS/FAIL
  carries no information. What can move is how long the filtered policy
  survives — measured 101 steps at 5 Hz, 199 at 10, 288 at 20, against a
  1000-step horizon — and reading that against cutoff measures *how much*
  high-frequency content the policy depends on rather than merely that it
  depends on some (#491).
  Balance correction on this plant is a ~1.1–1.4 Hz phenomenon and every cutoff
  passes it essentially untouched (gain 0.963 at 1.4 Hz even at 5 Hz), while
  the measured 22.7 Hz tremor is cut 13.3 dB at 5 Hz and 3.6 dB at 20 — so a
  short episode means dependence on content well above what the task requires.
  `stance_probe_filter_hz` now accepts a number or a list; unusable entries are
  dropped with a warning rather than costing the cutoffs that are fine.
  The sweep lands in `stance_gate_probe_filtered.{txt,json}` as one curve
  sorted by cutoff, and is written even if a cutoff raises — a partial curve is
  still a curve. It rolls **10 episodes per cutoff** rather than the gate's 40:
  it certifies nothing, and the effect it measures is enormous (101 steps
  against 1000), so it does not need the sample size the bound's power is
  specified at. The whole sweep costs about what the old single 40-episode
  probe did.

### Removed
- **`TRexBehaviorEnv` and the `mesozoic.trex-command-terrain/v1` identity
  schema** (#556, consolidation PR-7). `environments/trex/envs/behavior_env.py` (497
  lines) is deleted. T. rex behaviors use the generic `SpeciesBehaviorMixin`
  subclass like every other species (Changed). The mixin gains the model-pool
  checks that it used to borrow, and it builds the collision-only probe pool
  for any species class that declares `_terrain_contact_probe_geoms`, which
  `TRexEnv` sets to `("neck_geom",)`. `--adapt` now accepts only
  `mesozoic.command-terrain/v1`. `environments/trex/tests/test_behavior_env.py`
  and `test_behavior_training.py` are deleted. Their environment cases run
  across all six species in `environments/shared/tests/test_behavior_env.py`,
  with the neck-probe cases kept for trex. The recipe refusals moved to
  `test_behavior_recipes.py`, and the command-line cases moved to
  `test_behavior_checkpoint.py` beside the `CommandEnv` they import. This
  removes the test-to-test import that the consolidation plan still listed as
  open. CI drops the trex file from the SB3 list.
- **The SB3 notebook's random-baseline cell and its `_HAS_MEDIAPY` probe**
  (#554, consolidation PR-14b). The cell rolled five random-action episodes that any
  trained policy beats; the zero-action cell after it measures the floor that
  decides whether stage 1 learned anything, and its markdown loses the sentence
  that pointed at the random cell. No later cell read its names (a symtable
  scan). The mediapy probe left with the video display it guarded (Changed).
- **The SB3 notebook's widen cell and its `WIDEN_FROM` / `WIDEN_MAX_REVISION_GAP`
  knobs** (#553, consolidation PR-14a, decisions D-D14 and D-D15). Both pending
  parents were widened and re-paneled (`20260920_010912`, `20260921_203149`),
  so widening is a command-line step: `python -m
  environments.shared.scripts.widen_checkpoint ... --to-stage-dir
  <LOG_BASE>/<species>/<algo>/<new run id>/<stage_dirname(species, root)>
  [--max-revision-gap N] [--label L]` with a new timestamp id
  (`YYYYMMDD_HHMMSS`), run before the notebook opens that run id (on Colab,
  the three steps in the tool's module docstring: section 1, a scratch cell
  that mounts Drive, never the storage cell, then the tool), then
  the notebook with `RUN_ID` set to the new run id, `SEED` to the parent's
  recorded `run.seed` and `TRUNK_FROM = ""`. Gone with the cell: the
  widen-seed comment of the configuration cell, the load preflight's
  `WIDEN_FROM` branch (it proves the trunk run's root handoff, else a
  throwaway), the chain loop's widen branch, `select_trunk`'s caller-less
  `widen_from` parameter, and the living copies of the "restart the runtime
  (or `del _ACTIVE_RUN_ID`) and delete the stray directory" remedy. What the
  cell guarded is kept on disk (Changed). `TestWidenCell`,
  `TestWidenCellExecution` and the notebook's D-C17 knob pin are deleted;
  `DEFAULT_MAX_REVISION_GAP == 1` stays pinned in `test_widen_checkpoint.py`
  and, for the shared matrix, in the configuration-knob pin, and
  `TestJudgeBranch` keeps the JUDGE-branch contract the tool relies on. The
  notebook goes from 35 cells (19 code), 2,330 source lines to 34 cells (18
  code), 2,271. Measured with `git diff --numstat` against `2ebed89`: code
  +380 / −20 (`result_bundle/reentry.py` is 293 new lines), tests +894 / −680,
  notebook JSON +163 / −230, docs +543 / −217.
- **The SB3 notebook's direction/terrain mode and `behavior_notebook.py`**
  (#552, consolidation PR-12, notebook-only slice, 2026-09-23; decision D-D13).
  `notebooks/sb3_training.ipynb` loses `COMMAND_TERRAIN_BEHAVIOR`, the ten
  `BEHAVIOR_*` knobs (`BEHAVIOR_LOAD_MODE`, `BEHAVIOR_CHECKPOINT`,
  `BEHAVIOR_VECNORMALIZE`, `BEHAVIOR_SEED`, `BEHAVIOR_STEPS`,
  `BEHAVIOR_EVAL_ONLY`, `BEHAVIOR_EVAL_EPISODES`, `BEHAVIOR_RECORD_VIDEO`,
  `BEHAVIOR_VIDEO_FPS`, `BEHAVIOR_RUN_ID`), the eleven direction/terrain values
  of the `BEHAVIOR` dropdown (now `stand`, `walk`, `hunt`, stage ids by free
  input), its six behavior cells (storage and run plan, run, saved-evidence
  display, the behavior disconnect) and the 15 guard sites; the guarded
  canonical cells are dedented unchanged but for two lint fixes the dedent
  exposed to `ruff check .` (41 cells, 23 code, 2,463 lines → 35
  cells, 19 code, 2,329 lines). `environments/shared/behavior_notebook.py` (294
  lines) and `test_behavior_notebook.py` (815) are deleted; the configuration
  defaults, free-form stage ids, the dropdown annotations and the setup cell's
  `REPO_REF` safety keep their tests in `test_sb3_notebook_pins.py`, whose
  guard-stripping helper goes. The CI wheel step names the eleven recipe files
  itself and still checks that all 66 ship and parse. The pilots have no
  notebook path until PR-11 adds the follow and terrain manifest nodes; until
  then they run from the command line (`python -m
  environments.shared.train_behaviors`, an explicit `--checkpoint` /
  `--vecnormalize` pair in every mode), a runner the rest of PR-12 deletes;
  `docs/TRAIN_DIRECTION_AND_TERRAIN.md` loses its notebook walkthrough. Every canonical knob keeps its name and
  default; notebook pilot runs already on Drive stay evaluation-only (D-D9).
  Measured: −996 code, test and CI lines, −134 notebook source lines.
- **The T. rex pilot recipes, the `[pilot]` recipe dialect and the trex behavior
  shim** (consolidation PR-6, #549, 2026-09-20). `configs/trex/behavior_pilots/` (8
  TOMLs, 239 lines), its `pyproject.toml` package-data line and
  `environments/trex/scripts/train_behaviors.py` are deleted. Six of the eight
  were the `configs/trex/behaviors/` recipes with a `[pilot]` header and the
  defaults left unwritten (`trex_gentle_terrain` was `sloped_terrain`);
  `trex_combined_terrain` and `trex_follow_direction_speed` commanded a minimum
  speed of 0.5 where the supported recipes command 0.525. `read_recipe` reads
  `[behavior]` only: a recipe without `[behavior] species`, `name` and `parent`
  is refused instead of defaulting to trex, and a `[pilot]` table is an unknown
  section; every committed `[behavior]` recipe reads unchanged, so exact resume
  of behaviors-trained bundles is unaffected. `behavior_notebook` loses the
  `PILOT_RECIPES`, `NotebookPilotPlan`, `validate_pilot_selection`,
  `build_notebook_pilot_plan`, `run_notebook_pilot` and `display_notebook_pilot`
  aliases (nothing imported them) and stops accepting the
  `mesozoic.behavior-pilot-run/v1` run-manifest schema (nothing has written it
  since #541; saved runs still display). `mesozoic.trex-command-terrain/v1`
  stays accepted for `--adapt` because `TRexBehaviorEnv` still writes it; PR-7
  deletes both. Saved #540 pilot runs stay readable as evidence (their `run.json`
  and `bundle.json` hold the recipe inline) and are evaluation-only under D-D9.
  Measured: 256 net code and configuration lines removed.
- **The certified library and the behavior trainer's library path**
  (consolidation PR-5, #548, 2026-09-20). `environments/shared/certified_library.py`,
  `certified_comparison.py`, their tests, `test_behavior_publication.py` and
  `docs/CERTIFIED_MODELS.md` are deleted with the `.gitignore` `/certified/`
  line. `train_behaviors` loses `--auto-source`, `--publish-certified`,
  `--certified-library` and `--comparison-episodes`,
  `_copy_explicit_certified_pair`, `certified_source.json` and the run.json keys
  `certified_source`, `certification_requested`, `comparison_episodes` and
  `certification` (the schema string is unchanged; the
  `certification_training_*` lineage keys stay); every mode takes an explicit
  `--checkpoint` / `--vecnormalize` pair. `behavior_certification.py` loses
  `certify_and_publish_behavior` (and with it the `certification/certificate.json`
  writer: no run produces a behavior certificate until PR-13 registers the gate
  kind; `judge_behavior_panel` stays, reached from tests only),
  `comparison_from_panel` and `behavior_library_key`;
  `configs/behavior_certification.toml` loses `certification_seeds`, the
  comparison seeds and the `[comparison]` table. The SB3 notebook loses
  `SOURCE_SELECTION` and its prose (2,477 to 2,463 lines): blank
  `BEHAVIOR_CHECKPOINT` / `BEHAVIOR_VECNORMALIZE` are refused in the
  configuration cell in every mode, before Drive mounts; the plan drops its
  `certified_library` / `auto_source` / `publish_certified` /
  `comparison_episodes` / `certification_skip_reason` fields; and the display
  cell shows saved diagnostics only, so a pre-PR-5 run made with
  `--publish-certified` no longer prints its certificate or library lines. The
  CI wheel step drops the `RULES_PATH` and `["comparison"]["episodes"]` asserts
  (its species-manifest read still exercises the top-level `configs/*.toml`
  package data). Measured: 1,602 whole-file lines and about 1,880 net code and
  configuration lines removed. A `mesozoic-labs/certified` directory on Drive,
  if one exists, is left in place; nothing reads it.
- **The canonical certified-library wrapper and the notebook's library hooks**
  (consolidation PR-4, #547, 2026-09-20). `environments/shared/certified_canonical.py`
  (the training-origin stamp, the `certified_inputs/` copy of a reused trunk
  ancestor, canonical publication and benchmarking), its test and
  `test_sb3_notebook_certified.py` are deleted. The SB3 notebook loses the
  `CERTIFIED_LIBRARY_ROOT`, `PUBLISH_CERTIFIED` and
  `CERTIFIED_COMPARISON_EPISODES` knobs, the stamp block of `train_stage`, the
  reuse branch's copy call and the publish block of the chain loop (about 90
  cell lines: 99 removed, 13 added); a reused trunk ancestor is loaded from the
  run that certified it, as
  the CLI always did (plan A10), and the gate refusal is the loop's only runtime
  release. The behavior storage cell passes `publish_certified=False`
  explicitly, so the notebook never publishes (#542 kept);
  `train_behaviors --auto-source` without `--resume`, which selected a
  canonical locomotion parent through the deleted module, now refuses with a
  message naming the explicit `--checkpoint` / `--vecnormalize` pair.
  `certified_comparison.py` stays until PR-5 (its only importer,
  `behavior_certification.py`, goes there). Runs made between #543 and this
  change that hold `certified_inputs/` copies stay valid: their `ancestors/`
  records point at the copy, reuse rule 1 follows it and their
  `artifact_manifest.json` hashes the copies, so leave those directories in
  place. `test_sb3_notebook_pins.py` regains the A10
  `..._never_copies_checkpoints` pin, gains the seed-at-construction and
  storage-cell-rerun pins the deleted file held, and asserts the knobs and
  blocks are gone.
- **`min_avg_forward_vel`, `min_success_rate` and the absolute
  `collapse_peak_floor` leave `configs/trex/behavior.toml`** (Phase B,
  WS-B2; 2026-08 review CF2, SS2, CF3; plan D1, decisions D-B2, D-B4). The
  hunt is gated on `task_success/v1`'s `min_success_lcb` at the declared
  `min_eval_episodes`, and `gate_schema` rejects the two retired threshold
  keys for that kind, so neither can return silently; the absolute floor is
  replaced by the measured `collapse_peak_floor_reference` /
  `collapse_peak_floor_fraction` pair — removed, not shadowed, because an
  explicit absolute key wins over the pair in
  `collapse_settings_from_config`. All three keys stay valid for the
  `reward_and_length/v1` stages of every other species.
- **Dead code the gap review found** (Phase K): the in-code sweep
  search-space defaults (search spaces load from
  `configs/<species>/sweep_*.json`), `RolloutProfiler`,
  `jax_reward_termination._check_foot_contact`, `mjx_utils.unscale_action_jax`,
  `sweep/ray_tune._sync_to_drive`, `compare_run_diagnostics._fmt`,
  `reward_functions.check_distance_contact`, `stage_layout.find_figure` /
  `GENERATED_DIRNAMES`, five uncalled reward-test helpers, `train_base`'s
  unused `DiagnosticsCallback` / `evaluate` / `record_stage_video`
  re-exports, and `DiagnosticsCallback`'s long-deprecated
  `plateau_window` / `plateau_threshold` parameters (now a `TypeError`).

### Fixed
- **A Colab reclaim no longer truncates the stage config, metrics, stance
  reports or evidence CSVs, or a final or handoff checkpoint pair, and a
  crash no longer strands a sidecar temporary the bundle hashes** (#562, CU-3 of `docs/CLEANUP_PLAN_2026_09.md`,
  decision D-D20). `train()` saved the final pair and the best and
  robust-best handoff pairs straight to the Drive mount (only the periodic
  pairs were staged), so a reclaim during a save could cut one short; a cut
  handoff pair beside an intact final pair made the RESUME cell treat the
  node as finished and JUDGE fail to load it (the KNOWN_ISSUES entry, now
  deleted). On a mount these pairs are now saved to local scratch and
  published by the new `curriculum.publish_staged_pair`, which first removes
  the destination file it publishes last: a handoff pair publishes its zip
  last, so a reclaim leaves at worst a sidecar without its zip, which
  `select_handoff_checkpoint` skips; the final pair publishes its sidecar
  last, and an empty placeholder takes the final zip's place (and the final
  sidecar is removed) before the final save begins, so a reclaim at any point
  of that save leaves a final zip that `checkpoint_pair_problem` rejects (the
  placeholder or the new zip without its sidecar), as it rejected a final
  pair a save straight to the mount cut short: D-D16's RESUME and chain-loop rules, the
  early-stop refusal included, see what they saw before. Off a mount the
  pairs are written in place, as before.
  `file_io` gains `atomic_write_json` and `atomic_write_csv`, through which
  `stage_config.json`, `metrics.json`, the stance gate text and JSON
  reports, the three evidence CSVs (`save_evaluation_episodes`,
  `write_stance_panel_evidence`, `write_recovery_evidence`) and the
  `gate_resolution.json`, `task_fingerprint.json` and `plant_identity.json`
  sidecars are now written, each with its bytes unchanged; the three
  sidecar writers had stranded a fixed `<name>.json.tmp` after a crash,
  which the result-bundle manifest hashed instead of discarding. Other
  run-tree records (the stage and training summaries, the results CSVs, the
  run's `zero_action_baseline.json` among them) are still written in place.
  File names and bytes are unchanged, so no digest moves.
- **Follow-up to #558: a final pair cut short is never judged; the run memo
  stays in its tree** (#559, decision D-D16, amended). Two reviews of #558 after it
  merged found these, each reproduced:
  - **A final pair cut short counted as a finished node.** `train()` writes
    the final pair straight to the mount (as it does the best pairs; only the
    periodic pairs are staged and published atomically), so a reclaim during
    the final save can truncate it. #558's RESUME cell skipped such a node
    ("Nothing to resume"), and the chain loop's JUDGE branch then failed
    loading it on every Run all with a bare load error. The check the RESUME
    cell ran on periodic pairs moves into the library as
    `curriculum.checkpoint_pair_problem` (the zip's `testzip`, SB3's outer
    `data` and `policy.pth` members, a sidecar that unpickles). The RESUME
    cell runs it over the final pair and every periodic candidate. A broken
    final pair means training ended and the final save began: within one
    checkpoint cadence (100k steps) of the budget the cell resumes over it with
    a warning, and the resume's final save replaces it; further short the node
    had stopped early, and the cell refuses it as a new attempt (D-D16); a
    spent budget names the broken pair. The chain loop runs it before JUDGE: a broken final pair is an
    interrupted node, and the refusal says what is wrong and names
    `RESUME_STAGE` (it used to say "`<label>_final.zip` is missing" also when
    only the sidecar was).
  - **A `QUICK_TEST` toggle reused the memo's id in the other tree.** With
    `RUN_ID = ""`, a quick test followed by the real session in one runtime
    opened the real run as `ppo/<the quick test's timestamp>`. The memo now
    counts only in the tree `SPECIES`, `ALGORITHM` and `QUICK_TEST` select;
    after one of them changes, `""` mints a fresh run.
  - **A node `RETRAIN_FROM` covers is refused before it is resumed.** The chain
    loop trains such a node from its parent instead of judging it, and D-A20
    refuses its occupied directory, so the resume could never be judged. Such
    a node was trained here although the trunk run certifies it, so the loop
    would reuse the trunk's copy once `RETRAIN_FROM` is cleared; the recipe
    and the refusal now say to set `BEHAVIOR` to the node as well (the loop
    looks for its target only in this run) and to continue the deeper chain
    in a fresh `RUN_ID` trunked from this run. The recipe also says to pin
    `TRUNK_FROM` to the trunk the interrupted session's resolve cell printed
    (if that output is lost, the run named in the ancestor record of the
    reused ancestor nearest the interrupted node; a shallower record names the
    run that certified that node, which is not the trunk when the trunk itself
    reused it), and `""` only when that session trained every node itself: the
    loop never follows this run's own ancestor records, so `""` would train
    those again here.
  - The chain loop's interrupted-node message names the refused spent-budget
    case; the storage-cell comment and the recipes page say quick tests may
    count one another as replicates (never a real run's); decision D-D16 is
    carried into NEXT_STEPS, the consolidation plan's decision lists and the
    docs index.

  Tests: `checkpoint_pair_problem` on an intact pair, a missing sidecar, a
  truncated zip, a CRC mismatch, missing SB3 members and a truncated sidecar;
  the chain loop sending three broken final pairs to the interrupted-node
  refusal and an intact one to JUDGE; the RESUME cell resuming over a broken
  final pair within one cadence of the budget, refusing one further short and
  naming one when the budget is spent, skipping a broken newest periodic
  pair, judging nothing when a verdict or final pair has no periodic pair (a
  widened root), and refusing a node `RETRAIN_FROM` covers before anything
  trains while training one it does not; the memo across `QUICK_TEST`,
  `ALGORITHM` and `SPECIES` changes. Fifteen mutants of the new guards each
  fail a test. The notebook goes from 1,557 to 1,599 lines (33 cells, 17 code).
- **The RESUME cell never retrains a finished or judged node** (#558, the
  notebook-safety PR, decision D-D16). The cell measured what was left against
  the node's newest periodic checkpoint only. A node that an early stop ended
  short of its budget (session 4 stopped both of its nodes at 1.45M of 6M and
  12M, and both were judged), or one trained with an `N_ENVS` that does not
  divide the 100k checkpoint cadence (3 envs leave 2,999,970 of 3M), trained
  the rest of that budget when `RESUME_STAGE` named it: `train()` accepts
  `resume_same_stage` into a directory that holds `gate_verdict.json`, so the
  judged directory's final pair, `stage_config.json` and evaluation evidence
  were rewritten after judging. Whenever a post-resume evaluation beat the
  seeded best, the handoff pair the verdict hashes (`robust_best_model` /
  `best_model`) was rewritten too, and the chain loop's reuse rule then
  refused the node, leaving the partial run stuck. The cell now reads the node's
  verdict and its final pair (`<stage_label>_final.zip` and its
  `_vecnorm.pkl`) before the checkpoint scan; either one prints "Nothing to
  resume" and trains nothing, and the chain loop reuses, refuses or judges the
  node, so a `RESUME_STAGE` left set no longer stops a later Run all. A spent
  budget without the final pair (a runtime stopped between the last periodic
  save and the final one) is refused with the fresh-`RUN_ID` remedy; the cell
  used to point at the chain loop's JUDGE branch, which needs that pair.
  Executed tests cover a passed and a failed verdict, a final pair beside
  periodic steps of 1.45M and 2,999,970, a half-written final pair (resumed)
  and the spent budget.
- **A `QUICK_TEST` run can no longer become a trunk or a seed replicate of a
  real run** (#558, the notebook-safety PR, decision D-D16). A quick test trained 50,000 steps a node
  into the same `LOG_BASE/<species>/<algo>/` tree as real runs, under the same
  task, gate and hyperparameter digests (the budget enters none of them). The
  zero-action statue clears the stance gate of velociraptor, brachiosaurus and
  dibothrosuchus, so a quick-test stance could certify, win a
  `TRUNK_FROM = "auto"` coverage tie as the newest run, and count as a seed
  replicate under another `SEED`. Quick tests now live under
  `<algo>_quick_test/`, beside the real `<algo>/` tree: the trunk selection
  scans `<algo>/` and replicate discovery scans each run's own tree, so quick
  tests may count one another but never a real run; a quick test can still
  reuse real certified ancestors. No digest or provenance field changes, and a real session's paths
  are unchanged.
- **The RESUME cell takes a stage's id as well as its number** (#557, the
  notebook follow-up to PR-7). `RESUME_STAGE = "locomotion"` raised
  `KeyError: 'locomotion'` at `STAGE_CONFIGS[RESUME_STAGE]`: the stage
  configs, the `stage2_*` checkpoint names and the chain loop key a numbered
  stage by its number, while the section 6 prose invited a "semantic id".
  The cell now resolves `RESUME_STAGE` through the manifest to the stage's
  reference, as the manual cell already did, so `"locomotion"` and `2`
  resume the same node and `"recovery"` keeps its id; it prints what it
  resolved to, and its no-checkpoint error names that node beside `RUN_ID`.
  A node off `BEHAVIOR`'s chain is refused before anything is trained: the
  chain loop visits only the chain, so it would never have judged it. The
  section 6 prose says which number is meant (the N of the `stageN_*`
  checkpoints, not the resolve table's position or the `NN_` directory
  prefix), asks for the interrupted session's `SPECIES`, `BEHAVIOR` and
  `SEED`, and numbers the chain-loop run that judges the node (the RESUME cell
  writes no gate verdict). Executed tests resume a `compsognathus_robot` run
  by id and by number and refuse an off-chain node.
- **A complete run can no longer take a new node in place** (#553,
  consolidation PR-14a, decision D-D15; moved from KNOWN_ISSUES, verified 2026-09-23). A
  node trained into a run whose `artifact_manifest.json` records `complete`
  (the in-place recipe, `RUN_ID` set to the run) wrote its stage directory and
  verdict, then the chain loop's `save_run_bundle` raised `completed result
  bundle is immutable, but certified artifact(s) changed after publication`
  and "Run all" stopped before the auto-disconnect. The resolve cell now ends
  with `result_bundle.refuse_complete_run_session`, which refuses such a
  session before anything is trained or written: on a complete run,
  `RETRAIN_FROM` or any chain node whose loop directory holds no
  `gate_verdict.json` (a non-target node held only as an `ancestors/` record
  passes while a trunk is resolved) raises `ResultBundleError`, naming a fresh
  `RUN_ID` and `TRUNK_FROM` set to the complete run. What that prediction
  cannot see without applying the reuse rule, the chain loop refuses before
  the write (`refuse_write_into_complete_run`): a node held only as an
  `ancestors/` record that the trunk no longer certifies (it would be judged
  or trained into the run), and a trunk's copy of a node whose own verdict the
  reuse rule refused (its `ancestors/` record would be written into the run).
  The manual cell (before its freeze) and the resume cell (before
  `train_stage`) refuse the same write. The zero-action cell rewrote
  `RUN_DIR/zero_action_baseline.json` with a new `captured_at` on every run,
  so a re-entry that ran it left a complete bundle that no longer verified; it
  now keeps a complete bundle's copy. A reuse-only session in a complete run
  (the same or a shallower `BEHAVIOR`) passes and writes nothing into the run
  (a reuse-only re-entry of a `partial` or `failed` run still stops at the
  cleanup cell, now a KNOWN_ISSUES LOW entry).
  `test_result_bundle_reentry.py`, `test_zero_action_baseline.py` and
  `test_sb3_notebook_pins.py` cover it in the shared matrix.
- **A completed "Run all" of the SB3 notebook reaches the auto-disconnect
  again** (#552, 2026-09-23). The training-curves cell re-plotted every node of the
  run with `save_path` / `save_dir`, which wrote `training_curves.png`,
  `locomotion_health.png` and `behavioral_metrics.png` straight into each
  stage directory after the chain loop had sealed the bundle; the manifest
  declares only the node's `figures/` set (written by
  `generate_stage_artifacts`), so the cleanup cell's `validate_result_bundle`
  raised `undeclared bundle artifacts` from its fallback branch and "Run all"
  stopped before the auto-disconnect cell. Every completed session run
  (`20260920_010912`, `20260921_203149`, `20260922_125248`) ended there, left
  its runtime connected, and carried three undeclared PNGs per trained stage
  directory, byte-size duplicates of the declared `figures/` copies; an audit
  of the three run trees found nothing else undeclared, missing or resized,
  and the 18 PNGs went to Drive's trash on 2026-09-23, so each tree matches
  its manifest by path and size again. The cell
  now only displays; `test_sb3_notebook_pins.py` executes it over a stage
  directory and pins that it writes nothing. Reuse was never affected: trunk
  selection reads the per-node files, not the manifest.
- **The collapse backstop no longer arms on the untrained or the standing
  policy on dibothrosuchus and brachiosaurus** (#551, 2026-09-23). Under
  `home-keyframe-residual/v1` action 0 commands the nominal stance, so an
  untrained stance policy scores the statue, above the absolute floors of
  0.75 x the statue (1950 / 1300); and `EvalCallback` scores locomotion at
  the full forward weight from step 0, so a warm-started policy that is
  still standing scores about 22x the locomotion floor of 100. Run
  `20260923_020654` (dibothrosuchus, NEXT_STEPS session 4) armed on both at
  its first eligible evaluation (1.0M) and stopped each node at 1.45M of
  6M / 12M, the failure `EvalCollapseEarlyStopCallback`'s docstring records
  for trex run `20260803_012355`. `collapse_peak_warmup_timesteps` now keeps
  early evaluations from setting the peak: 1,000,000 on both species' stage
  1 (the trex stance value; on that run's series any warm-up of 400k or
  more never arms), and `warmup_timesteps + ramp_timesteps` on stage 2
  (3,300,000 dibothrosuchus, 4,000,000 brachiosaurus): the D-B5 bound of trex
  behavior, conservative because the clip/entropy warm-up and the forward
  ramp both run from stage step 0. On that run's locomotion series a
  0.8-1.05M warm-up arms on the fallen plateau under the floor of 100 and
  anything past 1.05M stays disarmed, so the replay bounds the warm-up
  below at 1.1M; the committed values are the D-B5 bound, not a replay
  result. Replayed through the real
  callback on the six series that trained to budget (velociraptor stance
  and locomotion, compsognathus locomotion, trex stance, locomotion and
  recovery), no candidate setting stops any of them. Collapse keys enter no
  digest, so no `task_sha256`, `gate_sha256` or `hyperparameters_sha256`
  moves and certified nodes stay reusable. New replay tests in
  `test_curriculum_early_stopping.py` pin the stop without the warm-up and
  its absence with it. The floors stay absolute until the planned plant
  updates of both species (KNOWN_ISSUES).
- **A widened root's run bundle writes** (#546, 2026-09-20). The first widen session
  on the fixed loader (`20260920_010912`, the seed-44 trex stance widened
  r11 → r13 and re-paneled: PASS, reward 3408.3 ± 88.5, duty 0.0069, UCB
  0.0117) died right after its gate verdict, in the chain loop's
  `save_run_bundle`: `ResultBundleError: best_eval_reward must be a finite
  number for canonical stage 1`. A widened root never trains in its run, so
  the JUDGE branch finds no `evaluations.npz` and
  `build_stage_results_from_eval_data` leaves `best_eval_*` unmeasured (`""`),
  which the canonical summary rule rejected. `validate_result_summary` now
  accepts a null `best_eval_reward` for exactly the stage whose
  `provenance.deliverables` record names `widened_from_run_id` (copied from
  the widen tool's run block by the bundle writer, new in
  `RUN_BLOCK_DELIVERABLE_RECORD_FIELDS`); the judged panel and final
  evaluation stay required, and a root that trained here still needs its
  curve. The run's `01_stance/gate_verdict.json` is valid as written; the
  session continues in the same run directory (NEXT_STEPS.md §3).
- **SB3 archives load on whatever Python the Colab image ships; the widen
  session no longer kills the kernel** (2026-09-19). SB3 stores a model's
  `learning_rate` / `lr_schedule` / `clip_range` members through cloudpickle,
  and the closure `train_base.linear_schedule` returned (every stage TOML
  with `learning_rate_end`) was pickled by value with its code object;
  `PPO.load` / `SAC.load` execute it while rebuilding the optimizer, so an
  archive saved under Python 3.12 loaded under 3.13 (or the reverse)
  segfaulted the process with no traceback — reproduced both ways with
  torch held constant, and the mechanism behind the two dead widen
  sessions of NEXT_STEPS.md session 1 (runs `20260919_170528` and
  `20260919_190251`: the widen tool wrote the four widened artifacts, then
  its self-verification's bare `alg_cls.load` of the r11 parent died) and
  behind the 2026-08-28 "PPO.load segfaults on Colab's current image" note.
  - `environments/shared/policy_loading.load_sb3_model` is now the one way
    an SB3 archive is opened: it reads the archive's `data` JSON and
    `system_info.txt` without unpickling, finds the members whose payload
    embeds function bytecode, supplies the algorithm's schedule members
    through SB3's `custom_objects` (the stage config's schedules on a
    training load, inference placeholders otherwise) so they are never
    unpickled, and refuses — `PolicyLoadError`, naming the members — an
    archive whose OTHER members carry bytecode from another Python minor
    version. `load_sb3_checkpoint` and every former bare load call it:
    `train_base` (the warm-start load and both post-training eval-panel
    loads, handoff and legacy `best_model`),
    `evaluation.evaluate`, `reporting/stage_artifacts` (task-success
    evidence and both replays), `harnesses/freeze_recovery_gate`,
    `scripts/widen_checkpoint._verify`, `behavior_checkpoint._load_ppo`,
    `certified_canonical._load_pair`, the Ray sweep worker and
    orchestration, the SB3 notebook's `evaluate_stage_checkpoints` and the
    Ray sweep notebook's standalone-reload ranking and evaluation cells. `test_policy_loading.py` pins that
    no bare `PPO.load` / `SAC.load` / `alg_cls.load` exists outside the
    loader, in `environments/**` or any notebook.
  - `linear_schedule` / `cosine_schedule` return instances of the new
    picklable-by-reference `curriculum.schedules.LinearSchedule` /
    `CosineSchedule` (same values), so archives saved from now on carry no
    bytecode at all; `widen_checkpoint` re-states a parent's schedule
    members from its recorded `hyperparameters` block, or from the current
    stage config's algorithm block when the parent's `stage_config.json`
    predates that block (the r11 trex parent's does not; the widen report of
    `20260920_010912` names `parent_stage_config`), through
    `schedule_members_from_hyperparameters` when the parent archive stores
    them as bytecode, so a widened archive is bytecode-free instead of a
    3.13-labelled zip full of 3.12 code; `widen_report.json` records the
    members re-stated and their source.
  - The SB3 notebook's load preflight moved from the infrastructure cell
    (which ran AFTER the widen cell and loaded a throwaway model saved by
    the same interpreter, so it could never see a cross-interpreter fault)
    to a cell right before the widen cell that loads the `WIDEN_FROM`
    parent's real root handoff through `load_sb3_model` (else the trunk
    run's, else a throwaway), printing what it loads and which Python
    saved it, flushed, before the load.
  - Fixture archives saved under Python 3.12 and 3.13 with the legacy
    closure (`environments/shared/tests/fixtures/sb3_archives/`, regenerated
    by `make_fixtures.py`) pin the inspector, the round trip through the
    loader on whichever interpreter runs the suite, the training-keyword
    override, the refusal, the widen re-statement, and — in a subprocess —
    that the bare load of the foreign archive still dies where the loader
    survives.
- **The JAX/MJX backend evaluates, trains and resumes on the same plant as
  SB3** (gap review Phase J, `docs/reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md`
  findings JX1–JX9, EP3). Mechanical parity fixes only; the recovery gate
  itself stays refused on the JAX side by design.
  - The CPU-eval reward composer omitted the action-jerk term the MJX
    training step pays, so eval rewards, the `min_avg_reward` rail
    comparison and the component panel overscored chattering policies
    relative to training and to SB3. `compute_total_reward` /
    `compute_reward_components` take `prev_prev_action` and charge jerk
    exactly as the kernel does, pinned by a step-vs-composer parity test on
    a real MJX env (JX1). The evaluation video is rendered under the plant's
    mandatory 10 Hz command filter like training and the gate eval (JX7).
    `evaluate_policy_cpu` applies recovery pushes from the env's
    `perturbation_*` config on the shared schedule and force derivation, so a
    recovery evaluation no longer certifies push-free episodes (JX3), and the
    MJX push calibration uses the resolved home keyframe rather than
    keyframe 0 (EP3).
  - Both JAX trainers bootstrapped a mid-rollout time-limit truncation from
    the post-reset observation and let the GAE carry leak the next episode's
    advantages into the ended one — the synchronized full-horizon regime the
    stance gate selects for. Truncated-but-not-terminated steps bootstrap
    from `V(final_obs_t)` (JX5). `mean_episode_return` uses `_finite_mean`,
    so one update without a completed episode no longer fails the gate with
    NaN (JX8).
  - `[jax.policy_kwargs] net_arch` was inert on the JAX path; it now reaches
    every `make_actor_critic` construction site, training and load alike
    (JX9). The JAX CLI and `jax_setup` accept semantic stage ids, so
    `"recovery"` no longer crashes on an int comparison (JX3).
  - Same-stage resume on the JAX path re-initialised the optimizer and
    decayed the observation statistics instead of restoring the checkpoint's
    `opt_state`: `train_jax` gains `init_opt_state` / `resume_from` (via
    `restore_train_state`, no decay), and the notebook's `RESUME_FROM`
    restores the full train state, refuses a silently mismatched optimizer
    structure, and trains the stage's remaining budget rather than
    `NUM_UPDATES` more; the cross-stage handoff keeps its deliberate fresh
    optimizer and decay (JX2). The notebook's stage-2/3 auto-resume resolves
    the previous stage through the manifest and `stage_dir_candidates`
    instead of the retired `stage{N}` layout, keeping the publication-gate
    check on that path (JX4).
  - `jax_normalization` no longer claims VecNormalize equivalence (it
    normalizes observations only); an additive `[curriculum.jax]` table lets
    the scalar gate thresholds be calibrated per backend, and the JAX path
    warns once when a reward-judged stage has no override, naming the
    alive-gate and fall-penalty differences behind the shared number (JX6).
- **Evaluation numbers now describe the policy they claim to** (gap review
  Phase V, `docs/reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md` — the
  evaluation-validity findings). Every fix is mechanical; the two design
  decisions the review raised (the behavior-stage gate and seed
  multiplicity) are deliberately untouched pending sign-off.
  - `evaluate()` and the three diagnostic report scripts guessed the
    VecNormalize sidecar as `<stem>_vecnorm.pkl`, which never matches the
    `<label>_vecnormalize_<steps>_steps.pkl` name `CheckpointCallback`
    writes, so every periodic checkpoint was scored on raw observations
    (silently, in two of the scripts). All four resolve the sidecar through
    `_resolve_vecnorm_sidecar` and refuse to run without one unless
    `--allow-unnormalized` is passed, which prints an `UNNORMALIZED EVAL`
    banner instead (ER1/OP7). `evaluate()` also seeds its environment
    (`--seed`, default `DEFAULT_EVAL_SEED`) so CLI numbers reproduce and
    checkpoint comparisons are paired (ER3).
  - The sweep judged stage pass/fail from the four retired scalar
    thresholds regardless of the stage's declared `gate_kind`, so a trex
    stance sweep passed the statue (3271.8) and the chatterer (2133.4) on
    the 2100 collapse rail and chained the most statue-like trial forward.
    Gate rows now carry `gate_kind` / `gate_evaluable` / `stage_passed`
    (`None` when the kind cannot be judged offline) / `gate_reason`;
    panel-based kinds never fall back to reward, `_best_trial_model_path`
    refuses to chain a not-evaluable stage without `--force-continue`
    (labeled reward-ranked, not gate-passed), and a NaN reward fails
    instead of passing (OP2/OP8). Cross-stage chaining passes
    `initialize_next_stage`, so warm-started trials no longer die on the
    task fingerprint at startup (OP1, `--load-mode` on the trial CLI).
  - `build_stage_results_from_eval_data` fabricated `mean_forward_vel=0.0`
    (and friends) instead of reading the `metrics.json` it already opened,
    so every stage-2/3 sweep trial recorded a velocity gate failure at
    "0.00 m/s". Panel metrics come from `metrics.json`; absent ones are
    omitted and the gate names the unmeasured metric (ER4).
  - Result-bundle audit gaps a resumed or raced run could slip through:
    seed roles may no longer collide (publication ≠ training ≠ selection,
    enforced in both validators — `save_jax_stage_artifacts`'s default
    evaluation seed moved off the training default it shared) (ER6);
    dot-files are audited and stray `.*.tmp` leftovers are discarded before
    hashing (ER7); summary timesteps are cross-checked against the stage
    budget and periodic checkpoints past the recorded timesteps are errors
    (RP1); evaluation CSVs record `checkpoint_sha256` at evidence time and
    the audit binds it to the certified checkpoint (RP3, wired in the JAX
    artifact writer and the SB3 notebook); `stage_config.json`'s run block
    records `load_path` / `load_mode` / `parent_checkpoint_sha256` /
    `parent_task_sha256` when a checkpoint was loaded, and the audit
    validates and surfaces them (RP4); `save_result_bundle` validates a
    prospective manifest before writing the `complete` marker and heals a
    bundle wedged by a failed final validation instead of raising forever
    (RP6).
  - `train_curriculum` walked `range(1, 4)` and silently skipped the
    recovery stage; it walks the manifest and logs the skip of any
    non-advancing stage (TC7/OP6). `load_stage_config` rejects unknown
    top-level tables and warns on an empty `[env]` or algorithm table, so a
    misspelled `[environment]` no longer trains constructor defaults
    unannounced (CF4). The JAX artifact writer and `save_results_json`
    write atomically (ER5). Published summaries carry `gate_kind` beside
    `stage_passed` and the catalog/website render passes earned under a
    since-replaced gate as "passed retired gate" (SS5).
  - Smaller: the recovery-stage plateau diagnostic no longer dies on a `%d`
    format of a semantic stage id (ER2); `train()` gains
    `post_eval_episodes` / `--post-eval-episodes` so smokes can skip the
    unskippable 50/30-episode post-training panels (EE4); Drive-mounted
    TensorBoard logs get the same local-buffer + periodic sync GCS paths
    had (CO5); the baseline-watch and collapse warm-up advisories anchor on
    the session's actual target timesteps rather than the TOML budget
    (TC9).
- **Six readers the NN_id layout would have broken, found by the pre-PR
  adversarial review and each reproduced by execution before fixing.**
  The worst: the complete-bundle audit hardcoded its nine required
  `stage{N}/...` paths, so a full curriculum run in the new layout passed
  every save-side check, wrote a `status="complete"` manifest, then
  failed its own final validation as canonical-conflict — permanently
  wedged, since retries die in preflight against the complete status.
  Required paths now resolve to the run's actual directory names. Also:
  `sweep collect-results` discovered stages only by the `stage` name
  prefix (new runs collected zero rows — the legacy number now comes from
  the id suffix, never the position digits); `evaluate()`'s stage
  auto-detect silently fell back to stage 1 for NN_id paths (extracted as
  `detect_stage_from_path`, id-aware, recovery passes through as itself);
  `save_jax_stage_artifacts` accepted NN_id directory names its own
  prior-stage `stage*/` globs could not see; the Drive-summary notebook's
  collector walked only `stage{N}` dirs; and the GCS sync iterated stages
  1–3 by number, which would have silently never uploaded recovery's
  replays — it now mirrors every stage directory the run actually wrote,
  in any generation. Regression tests pin each fix.

### Changed
- **Run directories now name their stages `NN_id`** — `01_stance`,
  `02_recovery`, `03_locomotion`, `04_behavior` — so a run's folders sort
  in curriculum order and say what they trained (project decision,
  2026-08-20). The id suffix is the key and the numeric prefix is
  provenance: it records the stage's manifest position when the run
  happened, and it is deliberately NOT `stage{position}` — `stage2`
  already means locomotion to every pre-manifest artifact, and a
  `stage2_recovery` folder would smuggle the renumbering hazard back in
  through the filesystem. `stage_dirname()` is the writer-side authority;
  readers (bundle validation/evidence/audit, GCS sync) go through
  `find_stage_dir()`, which accepts every generation — `stage{N}`, bare
  ids, and the new form — so existing runs keep collecting. File-level
  prefixes (checkpoints, `*_final`, videos) stay on `stage_label()`.
- **Trex stage configs are named by stage id**: `stance.toml`,
  `locomotion.toml`, `behavior.toml` (recovery already was), reversing the
  earlier keep-historical-names choice — the manifest's `config` fields
  are the single source of truth, and every resolution path (integer refs
  included) now goes through the manifest instead of the `stage{N}_*`
  glob, which survives only inside the synthesizer for manifest-less
  species. Diagnostic scripts that carried their own filename maps
  (zero-action/stance-quality baselines, foot-sensor, action-bound,
  joint-excursion, observation-ablation reports) resolve through the
  manifest too, so they keep working for every species regardless of
  naming era. Catalog and historical investigation documents keep their
  recorded names — history is not rewritten.

### Fixed
- **The recovery stage records its replay videos.** `record_stage_video`
  seeded the replay environment with `seed + 2000 + stage` — integer
  arithmetic that raises `TypeError` for a semantic stage id, inside the
  replay recorder's best-effort try/except, so the first field recovery run
  produced figures but silently no `replays/`. Replay seeding now goes
  through `replay_seed()`: integer stages keep the historical arithmetic
  bit-for-bit, semantic ids map to a stable crc32 offset (not `hash()`,
  which varies per process), and tests pin both. Recovery replays also
  carry the stance side/front camera views and per-frame stance CSV — the
  same task plus pushes, and the side view is where a shove is visible.
- **The recovery stage can now start from Colab** (found by three field
  runs, each dying silently at the stance/recovery boundary). Plant
  identities record `model_path` repository-relative, and
  `derive_stage_task_fingerprint` passed it straight to
  `mujoco.MjModel.from_xml_path`, which resolves against the process cwd —
  correct in the test suite (cwd = repo root), fatal in Colab, whose
  notebook adds the clone to `sys.path` without chdir-ing into it. The
  first pushed-stage fingerprint therefore raised `ParseXML` before the
  stage directory existed, halting run-all with nothing in Drive to show
  why; push-free stages never load the model, which is exactly why three
  full stance runs sailed through the same code. The path now resolves
  against the repository root (absolute paths pass through), a missing
  model raises a `TaskFingerprintError` that names both paths, and a
  regression test pins that the fingerprint is cwd-independent.
- **The plant contract now records the perturbation engine** (CI caught
  what the pre-merge review missed). Every policy-interface fingerprint
  moved when `BaseDinoEnv.reset`/`step` learned to derive and apply push
  schedules — the interface hash covers `reset`'s source through
  `home_reset`, which all four species carry — but the manifest was never
  regenerated, so `plant-contract`, `test-sb3`, `test-shared`, and
  `test-jax-cpu` all failed on the same staleness error. Recorded as
  `plant_versions.toml` note 11 with a policy-interface revision bump for
  all four species (velociraptor 8→9, trex 11→12, brachiosaurus 6→7,
  dibothrosuchus 5→6; physics and visual untouched — no MJCF edit), then
  regenerated the manifest and species catalog. The note pins the
  compatibility facts: with perturbation off the new code is inert end to
  end (no extra RNG draw, no force written), so a perturbation-free
  episode is bit-identical to the previous plant and existing checkpoints
  remain valid on their own stages. This also retires the "7 pre-existing
  container failures" claim from the review round: those local
  `test_stage_layout`/`test_sweep_reporting` failures were this branch's
  own staleness (a stash bisect could not catch it — the perturbation
  changes were already committed), and all pass after regeneration.
- **The recovery stage arms its collapse backstop.** `recovery.toml`
  configured no collapse floor, so the backstop could resolve only to
  `inf` and never arm — exactly the silent failure
  `test_curriculum_early_stopping` exists to prevent, and it correctly
  refused the new stage. Recovery now mirrors stance's relative pair
  (`collapse_peak_floor_fraction = 0.45` ×
  `collapse_peak_floor_reference = 3495.2`) plus the 1M-step arming
  delay, with a comment recording why the **un-pushed** statue is the
  right reference for now (it errs in the arming direction; the pushed
  task's own zero-action baseline is what the P3 null panels measure, and
  P3 re-derives the pair).
- **Ten findings from the pre-merge review of the 1b branch**, the four
  severe ones first: (1) `recovery_quality/v1` was **fail-open** through
  the shared `evaluate_stage_gate` dispatch — it fell through to the
  reward-and-length gate, certifying a pushed stage on return alone; it now
  fails closed with a pointer at the resolver, the only supported verdict
  path. (2) Horizon-adjacent pushes whose recovery window could not fit
  were counted as failed shoves, structurally capping panel success below
  any threshold for every controller; the harness now judges only pushes
  that were **delivered** (window opened before episode end — undelivered
  phantom shoves are no longer recorded either) **and judgeable** (push +
  dwell fits the horizon). (3) The documented recovery warm-start
  (`--load` a stance checkpoint) would have hard-failed under
  `resume_same_stage` once checkpoints carry fingerprints; `train()` and
  the CLI gain `--load-mode initialize_next_stage`, and the notebook
  chooses the mode from manifest position. (4) The stage-entry warm-up
  injected a `forward_vel_weight` ramp (0.1 → 0.0) into the recovery task,
  whose config zeroes that weight; the ramp is now skipped when the
  stage's target weight is 0. Also: the notebook now actually computes,
  persists, attaches, and mode-validates task fingerprints (its recovery
  cell's lineage claim was previously unbacked); `require_gate_resolution`
  now verifies the resolution's own integrity hash, so a hand-edited
  frozen record is detected instead of trusted; twenty `Stage %d` log
  formats widened to `%s` (string stages raised inside logging on the
  recovery path); the safe-set docstring now describes the reference the
  code implements (panel-start height, P3 recalibrates); `--stage
  locomotion`-style semantic ids for legacy stages resolve through the
  manifest instead of erroring; and the no-subcommand CLI default path
  tolerates the absent `--load-mode`.

### Added
- **Recovery stage in the notebook, docs, and website**: the training
  notebook gains an opt-in "5b. Recovery Stage" cell pair —
  `RUN_RECOVERY_STAGE = False` by default — that warm-starts
  `train_stage("recovery", ...)` from the Stage 1 checkpoint, generates
  stage artifacts, and **records** the `none/v1` pilot verdict instead of
  enforcing a gate (contrast the numbered stage cells, which raise); the
  run is deliberately excluded from the result bundle until the
  integer-stage bundle schema migrates. `train_stage` itself is
  manifest-aware: artifact directories via `stage_label` (semantic runs
  land in `recovery/`, integers keep `stage{N}/`) and warm-up keyed off
  manifest position. Two library sites gained label-safe naming so
  recovery artifacts read `trex_ppo_recovery_*` / `recovery_final` rather
  than a fabricated number (`evaluation.record_stage_video`,
  `_record_stage_replays`; integer naming byte-identical). Docs: the
  README explains the stage manifest and the T-Rex four-stage curriculum
  with the `--stage recovery` invocation; the trex environment README
  gains the recovery example; the website's T-Rex page gains a Curriculum
  section (stance → recovery → locomotion → behavior, what each certifies,
  and why integers keep their historical meaning) and notes the toes are
  passive since r7; quick-start wording is species-neutral. The catalog
  tables / generated species data stay numbered-curriculum until the
  manifest migration's final part (recorded in the plan's status block).
- **The pushed-panel harness and the gate resolver (stage 1b, W4 part 2 +
  W5)** (`environments/shared/recovery_evaluation.py`,
  `environments/shared/curriculum/gate_resolver.py`): the evaluation
  machinery that puts real evidence behind `recovery_quality/v1`.
  `roll_recovery_panel` rolls any controller over the seeded pushed panel,
  measures the per-step safe set from **physical state** (pelvis height vs
  the panel-start reference, tilt, planar speed, bilateral foot load —
  provisional thresholds recorded in every evidence file), reads the push
  schedule from the environment itself, and writes one row per episode AND
  one per shove (controller, seed, push vector/timing, recovery step) —
  the split plan's pair-identity-is-part-of-the-estimand requirement.
  Pairing is structural and test-pinned: identical seeds produce identical
  push schedules across controllers with no coordination. Null controllers
  (zero-action statue, constant brace) ship as plain callables; a
  push-free environment is refused. The resolver freezes capability spec +
  null manifest (per-seed outcomes, exact-UCB headline) + decision
  procedure into an atomic, hashed `gate_resolution.json`; a **missing
  resolution blocks advancement**, a **stale one** (task-fingerprint
  mismatch) demands recalibration by name, and
  `evaluate_recovery_gate_from_resolution` is the only supported path to
  an advancing recovery verdict — thresholds from the frozen spec, paired
  differences from the frozen null panel, never a fresh roll. What
  remains before the first gated recovery run is measurement, not
  machinery: the P3 calibration panels (real null baselines, safe-set and
  threshold calibration) and the §8.1 transfer pilot.
- **The recovery gate statistic: `recovery_quality/v1` (stage 1b, W4
  part 1)** (`environments/shared/curriculum/recovery_gate.py`, registered
  fail-closed in the gate schema): certifies **per-shove recovery** —
  after each scheduled push the body must re-enter the safe set within
  `recovery_t_recover_steps` and dwell there for `recovery_dwell_steps`
  (touching the safe set mid-fall is not recovery); an episode succeeds by
  reaching the horizon AND recovering every push. The gate bounds episode
  success with an **exact Clopper-Pearson LCB** (scipy-free bisection on
  the binomial tail — the right shape for a binary event, where the
  stance gate's Student-t is not), reproducing the split plan's own
  pinned arithmetic to the digit as regression tests: LCB95(34/40) =
  0.72526, one-sided 95% upper bound of 0/40 = 7.216%. The paired
  null-superiority statistic (same seeds, same schedules, t-bounded mean
  difference — the pairing the schedule PRF exists for) ships as an
  optional criterion that **fails closed when declared without its null
  panel** rather than skipping; it becomes authoritative once the
  resolver (W5) freezes the baselines. All thresholds provisional until
  P3/P5 calibration; `recovery.toml` keeps `none/v1` until then, its
  comment now pointing at the waiting kind. Not yet built (W4 part 2,
  with W5): the evaluation harness that rolls pushed panels, computes the
  per-step safe mask, runs the null suite, and writes per-shove evidence
  rows.
- **Stage identity through the consumers (stage 1b, W3 part 2)**: the
  single-stage SB3 train path now accepts semantic stage references
  end-to-end — `--stage recovery` parses at the CLI (digits stay legacy
  numbers), `load_all_stages` walks the manifest (trex gains a
  `"recovery"` key in curriculum order; the integer keys and every other
  species are untouched, test-pinned), artifacts and directories label
  semantic runs by ID via `stage_label` (`recovery_final.zip` under a
  `recovery_*` dir — never a fabricated number; legacy integers keep
  their historical `stage{N}` form everywhere), the task fingerprint
  records the semantic ID as its stage, and stage-entry warm-up now keys
  off manifest *position* (identical behavior for integers; correct for
  `recovery` at position 2). The legacy numeric curriculum deliberately
  ignores semantic-only stages (`thresholds_from_configs` filters to
  integer keys) — including recovery there would validate its `none/v1`
  placeholder under advancement and correctly-but-prematurely refuse the
  whole run; the manifest-walking curriculum lands with the W4 gate.
- **The seed-44 run recorded — replication now 2/3** (addendum in
  `docs/investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md`, KNOWN_ISSUES
  update 3): run `20260815_205206`, gate **PASS**, the strongest yet —
  full-horizon 40/40, duty 0.0069 / UCB 0.0117, panel reward 3408.3 ± 88.5
  (97.5% of the statue), zero non-truncated terminations, same unsaturated
  stance. The decisive datum: **AC 0.132** — the same quiet post-anneal
  endpoint as seed 42 (0.135), nowhere near seed 43's stalled 0.329. The
  2/3 split tracks the noise floor exactly; "what decides the anneal's
  endpoint" is now stage 1's sharpest open question.
- **Semantic stage manifest and the recovery stage config (stage 1b, W3
  part 1)** (`environments/shared/stage_manifest.py`,
  `configs/trex/stages.toml`, `configs/trex/recovery.toml`): the T-Rex
  curriculum is now FOUR stages — stance → recovery → locomotion →
  behavior — identified by stable semantic IDs, not numbers
  (STAGE1_SPLIT_PLAN §4). The no-silent-renumbering guarantee is
  load-bearing and test-pinned: **an integer stage reference means the
  legacy number forever** (`resolve(2)` is locomotion even though
  locomotion's position is now 3), recovery — having no numeric history —
  is reachable only by ID, and a manifest that tries to reassign or
  reorder legacy numbers is rejected at load. Manifest-less species
  synthesize their legacy three-stage manifest, which is how
  "recovery for the T-Rex only" is expressed. `load_stage_config` accepts
  semantic IDs (`load_stage_config("trex", "recovery")`); integer loading
  is byte-for-byte untouched. The recovery config is stage 1's `[env]`
  verbatim plus exactly the five `perturbation_*` keys at the adopted
  §3.3 values (165.5 N derived on r7), with a freshness test pinning the
  mirror so stage-1 shaping changes cannot silently strand it; its gate
  is `none/v1` — the schema's own non-advancing-pilot rule refuses to
  advance through it, fail-closed, until `recovery_quality/v1` (W4) and
  the frozen null baselines (W5) exist. Not yet migrated (W3 part 2):
  the curriculum loop, CLIs, and sweep still iterate legacy integers;
  artifacts do not yet carry stage IDs.
- **Task/evaluation fingerprint and checkpoint load modes (stage 1b, W2)**
  (`environments/shared/task_fingerprint.py`): closes the provenance gap
  STAGE1_SPLIT_PLAN §3.2 names — a `step()`-level task change like the
  scheduled pushes moves no `policy_interface_revision`, so checkpoints
  stay mechanically loadable while being unvalidated for the new task. The
  fingerprint hashes the task-defining configuration (species, stage,
  backend, plant physics/interface hashes, the full `[env]` kwargs, and
  the perturbation block with its per-species **derived** newtons plus the
  schedule-PRF implementation name). Every SB3 stage now writes
  `task_fingerprint.json` beside `plant_identity.json`, embeds it in
  `stage_config.json`, uploads it to GCS, and attaches it to the model so
  SB3 persists it in the checkpoint ZIP. On load there are exactly two
  modes: `resume_same_stage` (exact match or a fatal error naming the
  differing sections — a changed task can never be resumed silently) and
  `initialize_next_stage` (the curriculum handoff: a boundary crossing is
  expected once and recorded as a parent/child lineage that travels on
  the new checkpoint). `train()`'s user `--load` is same-stage; the
  curriculum loop's only loads are handoffs. Fail-closed core with one
  dated transition valve: checkpoints minted before 2026-08-15 carry no
  fingerprint, so the train paths pass `allow_unfingerprinted=True`
  (warn, not fail) until fingerprinted checkpoints are the norm —
  tightening lands with the gate resolver (W5). The JAX training path
  does not attach fingerprints yet; `derive_stage_task_fingerprint` is
  backend-tagged and env-free, so wiring it there is mechanical.
- **Species-generic scheduled pushes, SB3 path (stage 1b, W1)**: every
  species' environment now accepts five `perturbation_*` parameters,
  default **off** — with the multiple at 0.0 no schedule exists, no RNG is
  drawn, and trajectories are byte-identical to the pre-perturbation code
  (pinned by test). When enabled, `BaseDinoEnv` derives the push force from
  the plant itself (`environments/shared/perturbation.py`: root-subtree
  mass, home-keyframe CoM height, floor-contact support hull → capture-point
  velocity; on the r7 trex, multiple 1.5 over 0.20 s derives to 165.5 N,
  reproducing STAGE1_SPLIT_PLAN §3.3's ~150 N from first principles),
  generates a deterministic per-episode schedule from a `lowbias32` hash
  (bit-identical on NumPy and JAX, so paired policy-vs-null evaluation gets
  identical pushes from identical seeds), and writes the self-clearing
  force to `xfrc_applied` at the root each control step. Reward and
  observation are untouched — the push changes the task, not the interface.
  `perturbation_manifest()` exposes the derived per-species constants for
  run provenance.
- **Scheduled pushes, MJX/JAX path (stage 1b, W1 completion)**: the five
  `perturbation_*` keys are `MJXEnvConfig` fields (so shared TOMLs flow to
  both backends), deliberately outside the versioned plant interface — a
  push changes the task, not what a checkpoint's actions mean; the coming
  task fingerprint (W2), not the policy interface, distinguishes pushed
  runs. The step kernel gates at trace time exactly like the r11 action
  filter: multiple 0.0 bakes the push-free trace with an empty `(0,)`
  schedule carry, so existing training is untouched. When enabled, the
  per-episode schedule is carried through `EnvState` (regenerated by the
  fused auto-reset from a `fold_in` of the reset key, leaving the existing
  draw sequence intact), and the self-clearing force is written into
  `xfrc_applied` at the root before the substep loop. Both backends derive
  from the same host model, so the per-species constants are identical to
  machine precision (pinned by a cross-backend test); schedule *content*
  differs across backends only through the seed draw, never through the
  PRF, which is bit-identical by construction.
- **The seed-43 replicate postmortem**
  (`docs/investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md`): the first
  replicate of the certified gate-pass configuration, and a gate **FAIL** —
  duty 0.0597 / UCB 0.0747, full-horizon 0.925, 0 of 200 panels passing,
  behind a deceptively healthy 3093 panel reward. The run reproduces the
  gate-pass trajectory's three acts and converges on the **same stance**
  (zero saturated actuators, head lowered ~−20°, tail in its settled droop,
  hip-roll balance), but the act-3 re-descent stalls at a broadband
  command-noise floor (AC 0.329 at the estimator's white-noise saturation,
  concentrated on the hip rolls) instead of quieting to seed 42's 0.135.
  Statue controls read 3493.8/3497.1 against the configured 3495.2, so the
  rails are fresh and the deficit is real. **n = 2 on the identical config:
  1 pass / 1 fail — seed-sensitive**, which resolves the "solved or lucky"
  question ("neither") and strengthens the stage-1b rationale: robustness by
  lucky seed is now measured. The config's ~4M prediction failed on the duty
  axis (0.345 vs < 0.28 predicted); the `algo_std` half remains checkable in
  the run's `diagnostics.npz`.
- **Stage 1b implementation plan**
  (`docs/STAGE1B_IMPLEMENTATION_PLAN.md`): maps the unbuilt recovery half of
  `docs/STAGE1_SPLIT_PLAN.md` onto the tree as it stands after the first
  certified stance-gate pass. Records the stage-1 closeout review's verdict —
  no blocker to starting the 1b build, with a verified cleanup list (a
  `diagnostics/action_saturation` logger-key collision, a filter-probe sweep
  that predates the plant's own 10 Hz filter, untested release-ablation
  groups, unpinned pack weights in the parity factory test, and four stale
  KNOWN_ISSUES entries falsified by the r11/gate-pass commits) — then lays
  out the build as six workstreams (perturbation engine, task fingerprint and
  load modes, stage identity, `recovery_quality/v1` gate, gate resolver,
  diagnostic tool modes), a run plan starting with the seed replicate the
  postmortems keep asking for, and seven decisions that need review before
  their workstreams start.
- **The passive-toes run postmortem**
  (`docs/investigations/TREX_STAGE1_PASSIVE_TOES_RUN_2026_08.md`): the first
  10M stage-1 run on the r7 plant under substep-honest metrics. Best survival
  ever recorded (100% full-horizon locked from 4.5M, reward 2213 vs the 1950
  rail, no collapse) and still a gate FAIL at duty 0.41 vs 0.02 — an 18.7 Hz
  foot chatter the boundary-sampled metric could have passed. The run
  falsifies the entropy hypothesis (ent_coef reached exactly zero at 7M with
  no break in tremor or duty), and the regrouped release ablation localizes
  the unholdable pose to the **left knee and ankle**, priced at ~29 of the
  home-pose term's 500 points while the chatter stabilising it forfeits
  ~1030 in support terms. The release ablation gains
  `knees_ankles` and per-side groups, and the impulse probe's prose no longer
  attributes asymmetry to toe splay the r7 plant cannot produce.
- **A constant-hold probe, which separates "needs bandwidth" from "needs
  feedback".** `stance_gate_report.py --hold-constant` replaces the policy's
  commanded action with the constant it commands *on average* — its own
  post-settle per-actuator DC — partway through each episode, and scores that.
  Runs automatically after the gate report when a stage sets
  `stance_probe_hold_constant`, writing
  `stance_gate_probe_constant.{txt,json}`.
  The existing low-pass probe could not answer this. A filtered policy still
  responds to what it sees, just slowly, so a fall under the filter shows the
  policy needs *bandwidth* without showing whether it needs *feedback* at all —
  and the two have opposite fixes. Cutting the loop outright decides it.
  Five variants bracketed by two controls: the unmodified policy (expressed as
  a handoff at the horizon, so it still carries the probe marker and can never
  land on the certification filenames) and the zero hold, which is the statue.
  Between them the measured pose is held from settle, hard and ramped, and from
  reset. The ramped variant exists to answer the obvious objection to a hard
  switch — that a step transient rather than the loss of feedback knocked the
  animal over — with a measurement instead of an argument.
  Measured on the passing `20260805_011234` checkpoint: **both controls reach
  the 1000-step horizon** (policy 3006.9, statue 3271.0) and **every held
  variant collapses** — 348.9 steps from settle, 330.2 ramped, 133.3 from
  reset, all at full-horizon 0.0000. The pose that policy holds requires
  continuous feedback, so the tremor is stabilisation rather than waste and
  raising `smoothness`/`action_jerk` is contraindicated. Full analysis in the
  2026-08-06 addendum to `TREX_STAGE1_BOUNCE_2026_08.md`.
- **All four stance probes now run automatically on every SB3 run.** The
  notebook calls `generate_stage_artifacts` and nothing else, so a probe
  reachable only from the CLI produces nothing on a real run — and a diagnostic
  nobody runs is a diagnostic that does not exist. `_run_stance_probes`
  dispatches to all four, each gated by its own `[curriculum]` key:
  `stance_probe_filter_hz`, `stance_probe_hold_constant`,
  `stance_probe_release_ablation`, `stance_probe_impulse_speeds`. All four are
  on for trex stage 1; every other stage and species is unaffected.
  The probes run **dead last** in `generate_stage_artifacts` — after the gate
  verdict is recorded, the summary written, and the graphs and replays saved —
  because they are the most expensive artifact step and nothing downstream
  reads them: a runtime lost mid-probe costs the probes alone, and a
  probe-wiring failure has nothing left to sink. (They briefly ran inside the
  gate report's own `try`, where a caller-side exception would have been
  caught by the report's handler and recorded a FAIL for a stage whose
  on-disk report says PASS — the founding defect of `_apply_stage_gate`, one
  layer up.) Pinned by behaviour tests through the real call sites — kwarg
  drift fails them, not just a deleted call — plus a source-order check that
  the verdict precedes the probes, so a fifth probe added beside them cannot
  quietly go unwired the way these two nearly did. Each is individually
  non-fatal at both levels: the helpers guard their own rollouts, and the
  runner guards each call site.
  Cost, per stage, at the trex settings: 3 filter panels at 10 episodes, 5 hold
  panels at 10, 13 ablation panels at 8, and 14 impulse panels at 8 — the
  impulse sweep doubled because the statue control is rolled over the same
  magnitudes and is not optional.
- **An impulse recovery probe**, `--impulse-probe`, which answers the one
  question no training artifact can: does a stage-1 policy actually *correct*,
  or has it only learned to stand still? Stage 1 declares no in-episode
  disturbance — the sole perturbation is joint-angle noise at reset — so the
  gate, the reward and all three earlier probes are silent on it by
  construction. This applies a step change to the root's linear velocity
  mid-episode (a shove, fully specified by `delta_v`) and sweeps magnitude in
  **both lateral directions**, with the **zero-action statue as the control**:
  it commands a constant and cannot respond, so its survival is the plant's
  passive robustness and only the policy's margin over it is active control.
  Measured on the passing checkpoint: **recovery envelope 0.50 m/s one way,
  0.00 the other.** It takes a 0.5 m/s shove to the horizon 8/8 pushed one
  direction and falls 0/8 pushed the other (424 steps against the statue's
  337); at 1.0 m/s and above it is within noise of the statue on both sides.
  Narrow *and* asymmetric — which is what a pose with differently-splayed toes
  predicts. So stage 1's gap is a **metric** gap: the capability partly exists
  and the task cannot see it, which makes the envelope a ready-made acceptance
  metric for `STAGE1_SPLIT_PLAN`'s 1b.
  The read-out is keyed off that envelope rather than a mean step margin. The
  first version averaged the margin across rows and reported +135, of which 82%
  came from one row — a number equally produced by a policy that recovers
  everywhere. That is the **third** pooled statistic in this investigation to
  hide the structure it summarised, after `action_std` over actuators-and-time
  and the saturation count over differing `ctrlrange`s. The pooled margin is
  still printed, second, with its caveat attached.
- **Per-actuator pose reported in degrees, and the table ordered by them.**
  `|action| = 1` is not one physical event: `_scale_action` maps `[-1, 1]` onto
  each actuator's own control span, and on the T-Rex those differ by 6× — a
  saturated tail joint is **8–12°** of deflection while a saturated toe is
  **37.5°**. Sorted by normalised `|dc|`, the per-actuator table put twelve
  joints at exactly `±1.000` at the top with no way to tell them apart, and the
  most conspicuous of them (the tail) was the one later measured to be
  mechanically inert.
  This is the same error the DC/AC split was introduced to fix, one level down:
  a pooled *normalised* offset cannot separate moving 8° from moving 37.5° any
  more than a pooled standard deviation can separate "sitting in the wrong
  place" from "shaking". Report schema v2 now records applied `ctrl_mean` and
  `ctrl_ac_rms` in native units, preserves the separate `action_zero_ctrl` and
  `home_ctrl` anchors, and marks which actuators are direct angular position
  controls. Only those receive `dc_deg`, `ac_rms_deg`, `range_deg`,
  `zero_offset_deg`, `home_qpos`, and `home_preload_deg`; geared motors are not
  mislabeled as radians. It adds `dc_rms_deg` across the compatible controls
  and **orders the table by degrees** — which puts the five
  saturated toes on top at ±37.5° and drops `tail_1_yaw` (±8°) out of the
  printed twelve. The normalised `dc` is kept alongside, because that is what
  inverts through the action penalties; the degree fields are reduced from the
  controls actually applied during rollout.
  This distinction matters for `home-keyframe-residual/v1`: T-Rex maps the two
  halves piecewise around `key_ctrl[home]`, so neither the mean nor RMS physical
  target can be reconstructed from normalised moments with one full-range
  scale. Action zero maps exactly to the home control for all 21 actuators. The
  ankles' +5.5° `key_ctrl - key_qpos` is an intentional gravity preload, now
  reported separately instead of being mislabeled as policy displacement or a
  reason to re-centre the control range.
  The same correction is now swept through the prose that taught the old
  identity: docstrings in `eval_diagnostics.py` and the hold/ablation helpers,
  the rendered ablation header, the CLI help, and the KNOWN_ISSUES saturation
  entry all say "home control" where they said "home keyframe" for what
  `action = 0` commands — the keyframe's *pose* keeps its name. The
  constant-hold docstring also now states its one deliberate approximation:
  the held command is `f(mean(action))`, not `mean(f(action))`, which differs
  only on zero-crossing actuators with asymmetric spans (neck/head pitch on
  this plant, bias ~0.01°).
- **A release ablation on top of it**, `--hold-release-ablation`, which asks
  *which* joints make a held pose unholdable. Each actuator group gets two
  variants: `release_G` holds everything except G (is G **necessary** — does
  removing it rescue the pose?) and `only_G` holds G alone (is G
  **sufficient** — does it break the statue by itself?). One side alone
  misleads, and a group can look implicated purely because it is conspicuous.
  Measured on the passing checkpoint: **holding only the six toe joints
  reproduces the entire failure** (125.5 steps against the full pose's 128.6,
  `tail_contact` 8 of 8 in both), while **holding only the four tail joints at
  ±1.000 stands the full horizon** at reward 3254.0 — within 0.6% of the statue
  — and head/neck likewise at 3286.1. No group is necessary (releasing all
  twelve saturated actuators still falls at 224.6), so the pose is
  over-determined, which the artifact now says in as many words.
  **This refutes the saturation hypothesis** recorded the same day: saturation
  is conspicuous, not causal, and the largest saturated group is provably inert.
  Every ablated variant is commanded **from reset**, which is load-bearing and
  was wrong in the first run: handing off at `settle_steps` snapped joints from
  ±1.000 to 0 mid-episode, the transient dominated, all thirteen variants landed
  within 311–349 steps of each other, and the `hold_zero` row — the statue,
  known to stand 1000 — fell at 323. The control caught it; without it the flat
  table would have read as "no group matters", which is a conclusion and a wrong
  one. `TestReleaseAblationVariants` now pins the from-reset requirement.
  Probe branding is now a registry (`_PROBE_MARKERS`) rather than a chain of
  `is not None` tests. The filter probe's guard was a single such test, and a
  second probe added beside it would have silently inherited the certification
  filenames and the `stance_panel_selected.csv` evidence a bundle is certified
  from.
- **The deterministic policy's action, measured instead of inferred.** Every
  action number on disk came from *training* rollouts, so all of it carried
  exploration noise; diagnosing issue #489 meant recovering the commanded pose
  and the tremor about it by inverting per-episode reward totals under a
  narrowband assumption. `StageAwareEvalCallback` now measures them directly
  during evaluation and `gate_progress.npz` carries them:
  `action_dc_rms` (the static distance from the home keyframe, which under
  `home-keyframe-residual/v1` is exactly `action = 0`), `action_ac_rms` (the
  tremor about it), `action_delta`, `action_jerk`, and `action_freq_hz`.
  The DC/AC split is the point: a pooled standard deviation over actuators and
  time — which is what `diagnostics.action_std` computes — cannot separate
  "sitting in the wrong place" from "shaking", and those have different causes
  and different fixes. Measured over the post-settle window, the same one the
  duty uses, so the reset transient does not inflate the AC term.
- **`action_dc_per_actuator` / `action_ac_rms_per_actuator`**, the same two
  quantities as `(n_evals, n_actuators)` matrices. Which actuators carry the
  offset decides whether a leg-pose weight can reach it at all: `leg_home_pose`
  covers only the leg joints, so a displacement living in the tail or neck is
  invisible to it and unfixable by it. Logged per actuator rather than
  per group because grouping is an analysis decision, and resolving joint names
  through the VecEnv wrappers is exactly the kind of best-effort lookup that
  silently returns nothing.
- **`term_*` in `gate_progress.npz`** — every reward term, per evaluation.
  Only `mean_reward` was kept, so comparing two runs meant comparing their
  final checkpoints and nothing in between, with no way to ask *when* a policy
  adopted the pose it ended up with.
- **`action_jerk` in `diagnostics.npz`.** The environment has always emitted it
  (it is what `reward_action_jerk` charges) and `INFO_KEYS` has always dropped
  it, so the signal was computed every step and discarded. With both
  differences recorded the effective frequency is free:
  `jerk/delta = (2 sin πfΔt)²`, which unlike either alone is blind to a
  constant offset.
- **`stance_gate_report.py --filter-actions HZ`**, a probe for whether a
  policy's high-frequency action content is load-bearing or waste. It
  low-passes the action between the policy and the plant and scores *that*:
  if the policy still stands, the tremor was waste and the fix belongs on the
  action path; if it falls, the tremor is closed-loop stabilisation and the fix
  belongs elsewhere (#489). Open-loop experiments on the statue bound what a
  tremor *can* do; only filtering the real policy answers it for that policy.
  A filtered rollout scores a **modified** policy, so the report records
  `filter_actions_hz` and the text form prints the warning *above* the verdict
  — a PASS obtained this way must never be mistakable for a gate result.
  **Wired into the training path**, not left as a manual script: with
  `stance_probe_filter_hz` set in `[curriculum]` (5.0 on trex 1a) the artifact
  writer re-scores the selected checkpoint after the gate report and emits
  `stance_gate_probe_filtered.{txt,json}`, so every run answers the question
  without anyone remembering to. Unset elsewhere, because it costs a second
  40-episode panel per stage and per sweep trial.
  The probe can corrupt the record two ways, and both are made structurally
  impossible rather than left to the caller: it gets its own filenames derived
  from the report itself, and it never writes `stance_panel_selected.csv` —
  that file is the per-episode evidence `result_bundle.evidence` certifies a
  stance-gated stage from, and a filtered panel there would certify a policy
  that was never run. `stance_probe_filter_hz` is registered in
  `gate_schema._DIAGNOSTIC_KEYS`, without which the fail-closed unknown-key
  check would make the setting unreachable.
- Every new evaluation series is recorded to the **SB3 logger as well as the
  npz**, so TensorBoard and W&B carry them live: `diagnostics/eval_action_*`
  and `reward_terms/eval_*`. A diagnostic that exists only in a file nobody
  opens mid-run is how the stance gate criteria stayed invisible for the whole
  of run `20260804_143747`.
- **`stance_gate_report.py` now reports the commanded action per actuator** —
  DC, AC rms, and the effective frequency, over the post-settle window, with
  real joint names (`r_hip_pitch`, `r_knee`, …). This script holds the
  environment directly rather than through VecEnv wrappers, so the
  actuator→joint mapping is exact rather than a best-effort lookup.
  It means **one five-minute run against an existing checkpoint** tests both
  premises of the `leg_home_pose_weight` change: whether the tremor is
  load-bearing (`--filter-actions`) and whether the static offset is even in
  the leg joints, which is the other way that change can fail by construction.
  Neither previously needed a training run to answer — they needed instrumentation
  that did not exist.

### Changed
- **T-Rex stage 1 `leg_home_pose_weight` 0.5 → 1.5, and the two constants
  derived from the statue re-measured with it.** Run `20260805_011234` **passed**
  `stance_quality/v1` — unsupported duty 0.0000 against a 0.0200 ceiling, 40/40
  episodes at the horizon, bilateral support 1.0000, confirming the entropy
  diagnosis in #487 — but scored **3004.3 against the zero-action statue's
  3274.4** and never crossed it in 200 evaluations (issue #489).
  The whole 270-point gap has one cause. Inverting the action penalties through
  their closed forms gives a per-actuator DC offset of **0.800** with an AC
  tremor of **rms 0.277 at an effective 22.6 Hz** (from `J/D = (2 sin πfΔt)²`,
  which is DC-blind); reproducing that tremor on the statue matches the trained
  policy's per-step penalties to within 2%. Both halves are open-loop fatal:
  a static offset of just **0.10 rms falls in every direction tested**
  (107–268 steps), and a 0.277 rms tremor collapses the statue at **every**
  frequency from 2 to 40 Hz (38–80 steps), while `action = 0` stands
  indefinitely. So the policy is holding a displaced pose upright with
  closed-loop feedback that the home keyframe does not need — the tremor is the
  cost of the pose, not a separate defect.
  That is why the fix is not on `action_jerk_weight`, whose penalty sits at
  1.4% of its normaliser and looks like the obvious lever. Penalising the
  tremor attacks load-bearing feedback, and the cheapest response to a bigger
  jerk penalty is to fall over. Raising the pose weight instead makes the
  displaced operating point uncompetitive — the pose gap goes 85 → 255 points,
  dominating the tremor's 112-point cost by 2.3× — and if the policy returns to
  the home keyframe the feedback becomes unnecessary and its cost goes too:
  **245 points, not the 112 a smoothness penalty could reach.**
  `min_avg_reward` 1950 → **2550** and `collapse_peak_floor_reference` 3271.8 →
  **4250.4**. Both are documented as derived from the statue (0.60× and 1.00×),
  and neither updates itself — leaving them would arm the collapse backstop
  against the wrong scale, the exact staleness the config's own comments warn
  about. The new baseline is measured, not scaled: 4250.40 ± 13.86 over 40
  episodes at noise 0.05, 100% full-horizon, from `zero_action_baseline.py`.
  The ratios are unchanged (rail 0.60×, floor 0.45× → 1913), which is what
  carries across a rescale; the absolute numbers do not, and neither do the
  historical run figures quoted in the config, now flagged as old-scale.

### Added
- **A watch on the do-nothing baseline.** Run `20260804_143747` trained T-Rex
  stage 1 for its full 10,002,432 steps — 8h 13m — and finished at **2686.9
  against the zero-action statue's 3271.8**: below the do-nothing policy on
  every major reward term, for the entire run. Nothing reported it. Every
  logged signal looked healthy (reward climbing, full-horizon 100%, evaluation
  sd down to 5, the collapse backstop correctly silent), and the run was only
  found to be in a worse-than-trivial local optimum by scoring `action = 0` by
  hand afterwards (issue #486).
  `curriculum.baseline_watch.BaselineProgressCallback` now reports every
  evaluation against the run's own captured `zero_action_baseline.json` — one
  float already written to the run directory before training starts, so the
  comparison is free — and warns **once** if the policy has still never beaten
  it past `baseline_warn_after_budget_fraction` of the budget (0.35 on trex
  1a, i.e. 3.5M steps with ~5.5 hours still to save).
  Deliberately **advisory**: it logs and warns, never stops. On stage 1 the
  statue *is* the reward optimum, so a learning policy legitimately sits below
  it for millions of steps; aborting on that would kill healthy runs for the
  same reason tightening the collapse detector does. A run that beats the
  baseline at any point is never warned about again — falling back under it is
  what the collapse backstop is for.
  `_build_core_callbacks` takes `species` to construct it, and a test pins that
  `sb3_training.ipynb` passes it, because an optional argument the trainer
  forgets is exactly how the stance gate became dead code on the Colab path.
- **`gate_progress.npz`, the gate criteria per evaluation.** The deterministic
  panel-estimator numbers the stance gate actually tests — `duty_episodes`,
  `unsupported_duty`, `unsupported_duty_ucb`, `bilateral_support_duty`,
  `mean_reward` and `full_horizon_fraction`, against `timesteps` — were computed
  every evaluation and recorded only to the SB3 logger. Diagnosing run
  `20260804_143747` therefore had to substitute the *training-rollout* duty
  from `diagnostics.npz`, which is contaminated by exploration noise; it
  happened to agree to three decimals, but that was luck.
  `StageGatePlateauCallback` now writes the file to the stage directory (beside
  `diagnostics.npz`, not the scratch eval dir) after each evaluation, so the
  gate's own view of a run is readable from Drive mid-run without TensorBoard.
  A separate file rather than columns in `diagnostics.npz` because the two are
  on different clocks — per evaluation versus per training rollout — and
  merging them would force one series to be NaN-padded against the other's
  timeline.
- **`action_abs_mean` and `action_std` in `diagnostics.npz`.** Mean |action| is
  the direct distance from the zero-action statue, which under
  `home-keyframe-residual/v1` is exactly `action = 0`. Neither existing series
  gives it: signs cancel in `action_mean`, so it sits near zero even for a
  large-magnitude policy, and `action_abs_max` is dominated by whichever single
  joint is most saturated. It separates the two ways an entropy collapse can
  end — std falls and the mean settles *on* the statue, versus std falls and
  the mean settles on some other committed pose — which are indistinguishable
  in `algo_std` alone and mean opposite things for what to do next.
### Changed
- **T-Rex stage 1 entropy now decays to zero, over 70% of the budget rather
  than 30%** (`ent_coef_end` 0.001 → 0.0, `ent_coef_decay_timesteps` 3M → 7M).
  Run `20260804_143747` trained the full 10M and finished at 2686.9 against
  the zero-action statue's 3271.8 — **588 below the do-nothing policy** — with
  unsupported duty pinned at 0.1668, 8.3× the gate ceiling (issue #486).
  The cause is measured, not guessed: `ent_coef` reached its 0.001 floor at 3M
  and held for the remaining 7M, which across 21 action dims holds the policy
  std at equilibrium ~0.375 (`algo_std` 0.47 at 5.9M, 0.375 at 8.4M). Through
  `_scale_action`'s residual mapping and this model's ctrlranges that std is a
  1σ command noise of **24.4° at the hips and 18.8° at knee and ankle**, mean
  **20.6°** across the major leg joints, resampled every control step at
  100 Hz. A policy cannot learn to hold a pose it is being commanded to shake
  by 20°; PPO optimises expected return *under* that noise, so it converged to
  a 16.7 Hz limit cycle that survives its own tremor rather than to a still
  stance.
  The target is known reachable and known still: `action = 0` **is** the home
  keyframe under `home-keyframe-residual/v1` and scores 3274.4 ± 7.6 with duty
  0.0000 (measured locally with `stance_gate_report.py trex --stage 1
  --zero-action`). So this is a convergence failure, not a reward-shaping one,
  and the usual risk of zeroing entropy — premature convergence to a bad
  optimum — is unusually low here.
  Early exploration is deliberately unchanged: `ent_coef` still starts at
  0.005, and the first 3M now decays more slowly than before. The 3M anchor
  was sized for a 6M stage and never revisited when stage 1 became 10M.

### Fixed
- **Contact is now measured at every physics substep, not one in five — the
  stance-duty gate certifies airborne time instead of a sampled duty.**
  Physics runs at 500 Hz and control at 100 Hz (`frame_skip = 5`), and every
  contact-derived quantity — the `r/l_foot_contact` info keys the
  `stance_quality/v1` gate certifies from, the bilateral-support /
  foot-load-balance / support-conditioned-alive rewards, gait-symmetry
  touchdown detection, and floor-strike termination — read only the **last**
  substep's state. The seed-43 stage-1 run exposed the consequence
  mechanically: a 20 Hz control-clock-locked hop put exactly one unloaded
  sample in every five (`bilateral_support_duty` exactly 0.8000 in all 40
  panel episodes, every duty a multiple of 1/800), so the gated 0.1958 was a
  sample-phase statistic, not airborne time — and the false-PASS direction
  was open: a policy sharpening its unloading to fall *between* the 10 ms
  samples would have read duty 0.000 and passed the 0.02 ceiling while
  unloading every cycle. Reward, obs, and gate all shared the stroboscope,
  and the run proved policies phase-lock to it.
  Both backends now aggregate inside the substep loop, in lockstep: per-foot
  **MIN** touch force across the substeps (min of per-substep per-foot
  *sums* — never per-sensor minima, which under-report when load shifts
  between pad and digits) feeds the rewards and info keys, and the first
  floor strike is latched (SB3: contact-pair latch in `BaseDinoEnv.step`;
  MJX: the height-emulation checks OR-ed per substep through the `fori_loop`
  carry). One residual asymmetry is deliberate and documented rather than
  closed: SB3's explicit site/body **height** terminations (T-Rex
  `head_tip_z`/`skull_z`, dibothrosuchus `snout_tip_z`) remain
  boundary-sampled — SB3's any-substep coverage is the *contact* latch,
  MJX's is the *height* emulation, so a transient head dip that touches
  nothing can terminate on MJX/eval while surviving SB3. The divergence
  direction is MJX-stricter (conservative for training); folding SB3's
  height checks into the substep loop is left for the plant-revision PR.
  `evaluate_policy_cpu` aggregates identically and passes the
  substep-MIN through a new `aggregated_foot_forces` override on the reward
  composers, so stage-gate evaluation scores the same quantity training
  optimizes; the override defaults to single-state semantics, which is what
  the SB3-vs-JAX parity test pins. **Observations deliberately stay on the
  boundary sample**: the obs builders are source-fingerprinted by the plant
  contract, so aggregating them would bump every species' policy interface —
  and the defect lives in what the gate and reward *price*, not in what the
  policy senses.
  `stance_duty_validation.py` gains the regime that certifies this: a
  control-clock-locked 20 Hz hop (period 5 = the aliasing regime its 2.5 Hz
  driver could never express), with per-substep kinematic ground truth
  recorded through a new step-loop probe hook and a within-control-step
  aliasing column. A regression test drives the same hop and pins both that
  `info` equals the min of the per-substep sums exactly and that steps exist
  where the boundary sample read supported while a substep was airborne —
  structurally impossible to catch before this change. The statue is
  measurably unaffected (settled-stance substep ripple is below the
  1e-6-relative equality tests' tolerance; a statue-duty test pins
  unsupported 0.0 / bilateral 1.0), but **duty histories are pre/post
  apples-to-oranges for hopping policies** — the chatterer's 0.319 and the
  seed runs' 0.1958 were boundary-sampled and would measure higher now; the
  statue-derived constants are re-derived with the passive-toes plant
  revision that follows.
- **`obs_rms_decay_on_resume` and `ramp_attr` were silently dropped by both
  JAX library entry points — and the `[jax]` table now rejects unknown keys.**
  All eight stage-2/3 TOMLs set `obs_rms_decay_on_resume = 0.01` ("lets stats
  adapt to Stage 2's velocity distribution within ~2 updates"), and only the
  notebook honored it: `run_curriculum` carried the prior stage's observation
  statistics into the next stage with their full sample count — millions —
  so `update_running_stats` was nearly a no-op exactly when the obs
  distribution shifts, which is the pathology the key exists to prevent.
  `run_curriculum` now decays the carried count through
  `decay_running_stats` with the entered stage's configured factor (same
  0.01 default and `1.0` opt-out as the notebook's resume cell).
  `ramp_attr` was likewise unread by `_JAX_KEY_MAP` and the CLI; it now
  reaches `train_jax` → `JaxTrainer`, which honors the one value the MJX
  path can ramp (`forward_vel_weight` is wired through `env.step` as a
  runtime scale; other weights are baked into the jitted reward at trace
  time) and **refuses anything else at construction** — the same contract
  the notebook's `train()` path already enforced — instead of silently
  ramping nothing.
  The mechanism that let both keys rot is closed the way `[curriculum]`
  closed it: `validate_jax_kwargs` rejects unknown `[jax]` keys fail-closed
  on both library paths ("silently dropping a misspelled threshold disables
  it"), `_JAX_KEY_MAP` is hoisted to module level as the single source of
  truth, and a test pins every mapped name against `train_jax`'s real
  signature so the map and the function cannot drift apart. One key is
  grandfathered as known-but-inert: `[jax.policy_kwargs]` is declared in
  every TOML but no JAX path reads it — the network factory is fixed —
  which is now documented at the known-key set instead of discovered.
- **`warmup_ent_coef` never reached a gradient update: `EntCoefDecayCallback`
  overwrote the stage-entry boost on the very next step.** Every PPO stage-2/3
  config sets both mechanisms, and they fought: `StageWarmupCallback` set the
  configured boost (0.02) exactly once at training start and logged "warm-up
  active … ent_coef=0.020", while the decay callback reassigned
  `model.ent_coef` from its own schedule on **every** step — from an initial
  value captured before the warm-up ran — so the boost was gone long before
  PPO's `train()` first read the attribute, and the warm-up's restore at the
  end of the window was clobbered identically. Callback order could not save
  it: a per-step overwrite beats a once-at-start set in either order. The
  documented anti-catastrophic-forgetting boost therefore never happened on
  any species' stage-2/3 transition, while the log claimed it was active.
  The two now hand off through a marker on the model
  (`ENT_COEF_WARMUP_MARKER`): the warm-up stamps it while it owns `ent_coef`
  and clears it when it restores, and the decay callback stands aside while
  it is set, capturing its decay base on the first step after release — so
  the boost holds for the whole warm-up window and the configured decay
  continues exactly as before from there. On the model rather than between
  the callbacks because the two are constructed independently, in no
  guaranteed order, in all three launch paths (CLI `train`,
  `train_curriculum`, and the notebook) — which is also why the fix needs no
  wiring change anywhere. Pinned by tests that drive the two real callbacks
  together, in both construction orders. SAC is unaffected (no decay
  callback is built for it, and its warm-up writes `log_ent_coef`);
  stage-1 runs are unaffected (no warm-up).
- **A stance-gated run that passed every gate still could not publish.**
  `result_bundle.evidence` refused any complete bundle whose stage declares
  `stance_quality/v1`, because `evaluation_selected.csv` records per-episode
  reward and length but no unsupported duty, and certifying on `min_avg_reward`
  alone would pass the zero-action statue. The refusal was correct and it also
  made the stage-1 milestone unreachable: the refusal fires only once all three
  stages have passed, so the first genuinely successful run would train for ten
  hours and then fail at finalisation. Observed on run `20260803_012355`, whose
  `collected_results.csv` and `artifact_manifest.json` stop at stage 2 while
  stage 3's own artifacts (`stage_summary.txt`, `figures/`, `replays/`, both
  evaluation CSVs) were written in full.
  The stance panel now persists its per-episode measurements as
  `stance_panel_selected.csv` beside the report it already wrote, and the
  auditor **re-derives** the verdict from those episodes rather than trusting
  the `passed` recorded in `stance_gate_report.json` — an evidence file whose
  claim is not reproducible from the episodes behind it certifies nothing, and
  run `20260802_203215` is the counterexample that motivated it. Reduction and
  scoring both go through `curriculum.stance_gate`, so the auditor cannot drift
  from the gate the trainer applied. An unmeasurable duty is written blank and
  read as unmeasured, never as `0.0` — zero is the statue's score and the best
  attainable one, so coercing would turn missing evidence into a perfect
  result. A separate file rather than new columns because the panel is a
  different evaluation from the publication one: 40 episodes at the panel seeds
  against 30 at the publication seed, and `min_eval_episodes` is the sample
  size the bound's power is specified at.
- **The JAX gate cleared its thresholds on metrics it never measured.**
  `nan < threshold` is `False`, so `jax_eval.check_stage_gate` returned
  `(True, [])` for an evaluation that produced no usable episodes — and
  `jax_setup` writes that verdict straight into `publication_gate_passed`.
  This is the same fail-open closed on the SB3 path, in the function that
  gates JAX stages 2 and 3. The rule now lives once as
  `gate_schema.finite_gate_metric` and both backends read it. Confined to
  declared thresholds: an unset rail (`-inf`) and an unset floor (`0`) could
  not be failed before and still cannot, so no legitimate config changes
  verdict.

### Changed
- **The curriculum gate verdict is now durable.** `gate_failures` was computed
  and persisted nowhere — not in `stage_summary.txt`, not in
  `collected_results.csv`, not in the `stage_result.json` allowlist — so on a
  failure the notebook disconnected the runtime and raised, and the reasons
  went with it. `stage_summary.txt` now states the verdict and every failing
  criterion, and the reasons are persisted on both backends.
  `generate_stage_artifacts` writes the summary *after* evaluating the gate,
  which it previously did not, so the summary could never contain the verdict.
- **The collapse backstop says what it is doing.** The resolved settings are
  logged once at construction, and the disarmed → armed transition is announced
  once with the peak, the floor and the drop threshold. Neither appeared
  anywhere before, which is why diagnosing runs `20260802_203215` and
  `20260803_012355` required replaying their evaluation series against the TOML
  at each run's commit. A `collapse_peak_warmup_timesteps` at or beyond the
  stage's own budget — a plausible extra-zero typo — now warns rather than
  silently disabling the backstop for the whole run.
- **The collapse backstop armed on the untrained policy and killed stage 1 at
  14.5% of its budget.** With `home-keyframe-residual/v1` and
  `log_std_init = 0`, action = 0 commands the nominal stance, so an *untrained*
  T-Rex already scores near the reward optimum. Run `20260803_012355` peaked at
  2469.4 on its **second** evaluation (100k steps) — 75% of the 3271.8
  zero-action baseline — which cleared the 0.45 × 3271.8 = 1472.3 arming floor
  on initialisation. The ordinary exploration dip that follows then read as a
  collapse: training stopped at 1.45M of a 10M budget, and stages 2 and 3 spent
  9¼ hours on the near-statue checkpoint it left. The 6M run at the same seed
  passed through that same dip to new bests at 2.8M, 3.3M and 4.55M, so there
  was nothing to catch.
  No `peak_floor` fixes this, because on stage 1 "already good" and "hasn't
  started learning" are the same number: set the floor above initialisation
  (>0.75×) and it lands above what a learning policy passes through — exactly
  how the two absolute floors documented in `collapse_settings_from_config`
  failed to arm. `collapse_peak_warmup_timesteps` is a different axis: rolling
  windows containing any evaluation before it cannot **set** the peak.
  Eligibility is by window *start*, so no surviving window straddles the
  boundary — a median absorbs one contaminating sample out of five, not five.
  While nothing qualifies, the backstop stays disarmed, which is the fail-safe
  direction; if the evaluation timesteps are unavailable it also stays
  disarmed rather than aborting a multi-hour run on a signal it cannot read.
  Expressed in timesteps rather than reward, so unlike an absolute floor it
  survives a reward-function edit. Defaults to `0.0` — every existing stage
  behaves exactly as before; only trex 1a sets it, to **1.0M**.
  Note this is **not** a tightening of `drop_fraction`/`patience`, which
  `collapse_settings_from_config` documents at length as aborting healthy runs.
  The value and the diagnosis are both measured against that run's real
  29-evaluation series: replaying the detector over it reproduces the stopping
  point of exactly 1,450,000, and any warm-up from 150k up never arms. The
  margin was **2.3%** — the first rolling-median window came to 1506.5 against
  the 1472.3 floor, and the best of the other 24 windows is 580.5, so a
  35-point difference in one early evaluation decided a ten-hour run. The
  policy was also *recovering* when it was stopped (means rise 59.4 → 350.7
  from evaluation 14 to 27) against a threshold of 753.3 anchored to an
  untrained policy. 1.0M rather than a larger value because the floor already
  blocks arming through the trough, so the warm-up only has to clear the
  initialisation spike, which is spent by 200k; 1.0M leaves 88% of a 10M budget
  under the backstop where 2.5M would leave 72%.
- **The training notebook never enforced the stance gate, and advanced a
  failing stage on its reward rail.** `notebooks/sb3_training.ipynb` computed
  `publication_gate_passed` from its own checklist over `min_avg_reward` /
  `min_avg_episode_length` / `min_avg_forward_vel` / `min_success_rate`. It
  never called `CurriculumManager.should_advance`, where `stance_quality/v1`
  is evaluated — `_build_core_callbacks` constructs no `CurriculumCallback`,
  so on the Colab path the stance gate was **dead code**. Retiring
  `min_avg_episode_length` from the trex 1a config then made the checklist's
  only other criterion vacuous (`.get(..., 0.0)`), leaving *reward alone* —
  the one threshold the zero-action statue clears by 68%, and the exact
  reading `stance_quality/v1` exists to refute. Run `20260802_203215`
  recorded `stage_passed = publication_gate_passed = True` in
  `collected_results.csv` beside its own `stance_gate_report.txt` reading
  `GATE: FAIL`, `mean_unsupported_duty 0.2120 > 0.0200` — 10.6x the ceiling —
  and started stage 2 on that checkpoint. The rule now exists once, as
  `reporting.gates.evaluate_stage_gate`, dispatched on the declared
  `gate_kind` and called from `generate_stage_artifacts`: the entry point the
  notebook and the sweep trial worker already share, so no caller can advance
  on a checklist that has drifted away from the schema. It is fail-closed at
  every branch — an undeclared or unknown kind, `none/v1`, a missing stance
  panel, a verdict scored for a different gate, and a non-boolean verdict all
  fail with a stated reason, because "we could not check" must never read as
  "it passed". Replayed against the run's own config and measurements, the
  old checklist evaluates one criterion and returns `True`; the new gate
  returns `False` naming both duty failures. Stage 2 and 3 semantics are
  unchanged: thresholds still apply only when declared, and the velocity and
  success floors only when positive. Three fail-open holes were found by
  probing the new code and closed before it shipped — a NaN metric cleared
  every floor (`nan < threshold` is `False`, measured returning
  `(True, [])`), a `reward_and_length/v1` block with no threshold set was a
  vacuously-true empty conjunction, and a `failures` string was shredded into
  one-character reasons.
- **The evaluation diagnostic no longer contradicts the gate it reports on.**
  `evaluate_stance_gate` certifies on
  `ceil(min_eval_episodes x min_full_horizon_fraction)` duty episodes — 38 of
  40 for trex 1a — but `StageGatePlateauCallback` marked the duty metric
  usable only at the full panel size, so a 39-of-40 panel **passed the gate
  while `eval_gate_met` read 0**. That is the same curve-vs-gate disagreement
  the stance scalars were fixed for, reintroduced one layer up. Both now
  share `stance_gate.required_duty_episodes`. Two smaller companions:
  `iter_replay_files` guarded `is_file()` on the nested branch but not the
  legacy one, so a directory named `*.mp4` was handed to the GCS upload for
  one layout and not the other; and the report's `plant_validated` was
  `not allow_legacy_plant`, but the rollout environment is validated
  unconditionally and the flag only relaxes the *checkpoint* check — so for
  `--zero-action`, which has no checkpoint, `False` claimed something untrue.
  It is now `checkpoint_plant_validated`, `None` when there is no checkpoint.
- **Six smaller defects in the stance gate report.** (1) Its JSON now carries
  `episode_evidence` — per-episode length, reward, duty and the two ungated
  shares — which `write_stance_gate_report`'s docstring already claimed it
  did. `run_panel` computed those and threw them away, so the file held only
  a summary that cannot be re-checked. Note this does **not** by itself lift
  `result_bundle.evidence`'s refusal of stance-gated bundles, which reads
  `evaluation_selected.csv`; that docstring now says so instead of implying
  otherwise. (2) `run_panel` closes its environment, including when the
  rollout raises — a MuJoCo env holds native handles and the artifact path
  builds one per stage. (3) Its unused `stage` parameter is gone. (4)
  `stance_report_episodes` in `[curriculum]` overrides the report's panel
  size, and `0` skips the report: the 40-episode rollout costs a few minutes
  per stage *and* per sweep trial, which a fifty-trial sweep paid fifty times
  with no way to decline. Overriding downward is logged as not certifying
  what the gate claims. (5) `--episodes 0` is rejected rather than falling
  through `episodes or min_eval_episodes` to a full-size panel while also
  skipping the under-powered warning — it read as an override that silently
  did nothing. (6) The checkpoint and the rollout environment are validated
  against the species' current plant identity, as every other artifact path
  already does; `--allow-legacy-plant` scores a checkpoint that predates the
  contract — a real use, since the script exists partly to judge finished
  runs — and the report records `plant_validated` so the flag travels with
  the number.
- **`[curriculum]` keys that were read but unregistered are now settable.**
  `diagnostics_plateau_window`, `diagnostics_plateau_min_relative_variation`
  and `supplementary_episodes` are each read with a default in the trainer,
  which means each was intended to be configurable — but `validate_gate_config`
  rejects any key it does not know, so setting one in a TOML was fatal. The
  same trap `max_checkpoints` hit. Registered alongside the new
  `stance_report_episodes`.
- **A missing `mean_episode_return` is fatal instead of silently substituting
  the per-step reward.** `min_avg_reward` is an episode-level threshold shared
  with the SB3 TOMLs — trex stage 1 sets 1950.0 — while the MJX trainer's
  `mean_reward` is the mean *per-step* rollout reward, ~3.3 for a standing
  T-Rex. Three call sites substituted one for the other with three different
  behaviours: `reward_and_length` warned and defaulted to `0.0`, the stance
  branch fell back **silently** and defaulted to `-inf`, and
  `run_curriculum`'s log line printed a third variant *labelled "episode
  return"*. Substituting is always wrong — it compares numbers three orders
  of magnitude apart — and merely happens to give the right verdict
  sometimes; against the stance rail it gives 3.3 < 1950, so the stage never
  advances and the only clue is a rail failure indistinguishable from a
  policy that genuinely threw away its return. One resolver,
  `episode_return_for_gate`, now raises `GateSchemaError` naming both numbers
  when a finite reward criterion is configured and the episode return is
  absent — matching how this module already treats a declared
  `min_avg_episode_length` with no `mean_episode_length`. The rail is
  optional for `stance_quality/v1`, so an absent return with no rail declared
  stays fine. The live path is unaffected: `jax_trainer` always emits
  `mean_episode_return`, so only a hand-written `train_fn` can reach the
  raise — which is exactly who needs to be told which key to emit.
- **The JAX/MJX publication gate no longer certifies a stance-gated stage on
  the reward rail alone.** `jax_setup.run_stage_evaluation` called
  `jax_eval.check_stage_gate`, which reads four fixed thresholds and knows
  nothing about `gate_kind` — so for `stance_quality/v1` it certified on
  whichever of the four happened to be set. On trex stage 1 that is
  `min_avg_reward = 1950` alone, since `min_avg_episode_length` was retired
  when the stance gate replaced it, and the verdict is written straight into
  `stage_results["publication_gate_passed"]`. Measured: the zero-action
  statue (3271.8) and the chattering policy the gate exists to reject
  (2133.4, duty 0.319) **both passed**. The new
  `check_stage_gate_for_config` dispatches on the declared kind and
  reconstructs the stance panel from the CPU-eval foot traces, so the JAX
  path now enforces the same criteria the SB3 one does. Reconstruction is
  valid only where each step contributed one reading per side — the biped
  case — and the guard is a measurement of the data in hand
  (`len(diag_r_foot) == sum(lengths)`) rather than a species allow-list, so
  the known `diag_r_foot`/`diag_l_foot` interleaving defect fails the gate
  closed with its reason instead of silently mis-pairing feet.
- **One unmeasurable metric no longer switches the whole evaluation
  diagnostic off.** `StageGatePlateauCallback._process_evaluation` returned
  early if *any* configured metric was short of samples. Adding unsupported
  duty to the priority list then silenced the callback entirely for the case
  it is most needed: early stage-1 training, where episodes end before
  `settle_steps` and every duty is NaN. Measured — a flat, dying run produced
  one plateau warning under `reward_and_length/v1` and **zero** under the
  stance gate, with `eval_gate_met` and `eval_plateau_active` never recorded.
  It now follows the metrics the evaluation can support and names the ones it
  could not measure in the warning, so a partial panel cannot imply the
  metric it reports is the whole story. `eval_gate_met` is still only 1.0
  when every criterion was checkable.
- **The TensorBoard duty scalar is the number the gate compares.**
  `diagnostics/eval_unsupported_duty` averaged every episode while the gate
  averages full-horizon episodes only: on a panel of 35 clean episodes at
  0.01 plus 5 early failures at 0.60 the curve read **0.084 against a 0.02
  ceiling the policy was actually clearing**. It is now built from the same
  `stance_gate` reduction the curriculum runs. The three quantities that
  actually decide advancement reached no logger at all and are now published
  as `eval_unsupported_duty_ucb`, `eval_full_horizon_fraction` and
  `eval_duty_episodes` — the last even when zero, which is the early-training
  signal that nothing survived the settling window and otherwise reads
  identically to "not evaluated". `eval_bilateral_support_duty` moved to the
  same full-horizon episode set so the shares stay comparable.
- **A failed stance gate report no longer costs a finished run its
  artifacts.** `_load_policy` raised `SystemExit`, which derives from
  `BaseException` and sailed through the `except Exception` guard in
  `_write_stance_gate_report` that exists precisely so a diagnostic cannot
  sink a run — a truncated VecNormalize `.pkl` aborted artifact generation
  before the graphs and videos were written. It now raises
  `StanceGateReportError`, and `main` converts that to `SystemExit` at the
  CLI boundary where the behaviour belongs.
- **The report's three stance shares sum to 1 again.** Bilateral and single
  support were averaged over every episode that measured anything while the
  gated duty uses full-horizon episodes only, so the identity broke whenever
  an episode failed early — measured 0.90 with 5 failures in 40 — and it
  broke in the direction that hides the problem, folding the flailing
  episodes into bilateral and single but excluding them from unsupported.
  That identity is the stated reason the report prints those shares at all.
- **`stance_gate_report.json` is valid JSON.** `json.dumps` emitted bare
  `NaN` and `Infinity` for the gate's unmeasurable sentinels — Python reads
  them back, `jq`, `JSON.parse` and Go's `encoding/json` do not — so it was
  the *failing* panel, the one worth inspecting, that produced an unparseable
  file, while the docstring promises this form exists for tooling that does
  not parse prose. Non-finite values now serialize as `null` (with
  `allow_nan=False` as a backstop); the text form still prints `inf`.

### Changed
- **The replay video now shows the checkpoint the evidence CSV is evidence
  for.** `generate_stage_artifacts` recorded it from `best_model` and labelled
  it `best`, while `evaluation_selected.csv`, the next-stage handoff, and the
  stance gate report all resolve through `_select_handoff_checkpoint`, which
  prefers the risk-adjusted `robust_best_model`. Both checkpoints normally
  exist, so a stage directory held a video and an evidence CSV describing
  **different policies**, with nothing in either name saying so — the kind of
  mismatch that makes a published figure not match the numbers beside it.
  There were four private copies of the preference order (the artifact
  generator, `build_stage_results_from_eval_data`'s recorded `model_path`
  — which is what the sweep trial worker records, with nothing else to
  re-derive it — the stance gate report, and the SB3 notebook's
  `train_stage`); all four now call the one selector, and the replay is
  labelled `selected`
  (`<species>_<algo>_stage<N>_selected.mp4`). Because that selector requires
  the matched VecNormalize statistics, two silent-wrong-answer paths close
  with it: the stance gate report no longer passes `vecnorm_path=None` when
  the statistics are missing (scoring the policy on unnormalised observations
  — a different policy — and reporting the verdict as this one's), and the
  replay is skipped with a reason rather than showing footage that
  misrepresents the checkpoint. The notebook's "robust weights, best_model's
  statistics" hybrid case now falls back to `best_model` the way next-stage
  loading always did, instead of raising.
- **SB3 keeps only the newest `max_checkpoints` periodic checkpoints
  (default 5).** SB3's `CheckpointCallback` never deletes anything: at the
  shipped `save_freq = 500_000` a 6M-step stage 1 left **twelve** 4.02 MB
  policy zips plus their VecNormalize sidecars — 48 MB of the 60 MB `models/`
  measured on run `20260801_021545`, and ~200 MB for a three-stage run — on a
  Drive mount, for files nothing in this repository reads. They exist for
  manual rollback, which needs recent history, not all of it. The JAX backend
  already capped this at `max_checkpoints = 5`; the default here matches so a
  stage keeps the same rollback depth whichever backend produced it. At that
  default the measured stage-1 `models/` goes **60.4 MB → 32.2 MB** (30 files
  → 16), a three-stage run 181 MB → 97 MB. Note this reclaims *storage*, not
  write bandwidth: all twelve checkpoints are still written and seven are
  deleted again. Writing fewer would mean raising `save_freq`, which is a
  different and lossier decision. Pruning
  matches only the periodic `stage<N>_{steps}_steps.*` pattern, so
  `best_model`, `robust_best_model` and `stage<N>_final` are unreachable from
  it by construction rather than by an exclusion list that could fall out of
  date; a pruned step takes its matched `vecnormalize_` statistics with it;
  and ordering is by the step count parsed from the filename, not mtime,
  which on a Drive mount records when the upload finished rather than when
  the checkpoint was produced. It fires on the `CheckpointCallback` cadence,
  not every step — `_on_step` runs millions of times a stage and globbing a
  Drive mount that often would cost far more than the storage it reclaims.
  Set `max_checkpoints = 0` in `[curriculum]` to keep every one.
- **A stage's generated figures and replays are grouped into `figures/` and
  `replays/`, and render to local scratch before publishing.** A stage
  directory held 20 loose entries — 5 PNGs, 6 MP4s and 2 per-frame stance CSVs
  intermixed with the config, summary and npz files — written by three
  unrelated call sites, with the only reader
  (`config.upload_curriculum_artifacts`) finding the videos through a
  hand-written `glob("*.mp4")` that would have silently stopped uploading them
  the moment either side moved. `reporting.stage_layout` now owns the layout
  for both backends, and every accessor falls back to the legacy flat location
  so the existing runs on Drive keep resolving; nothing rewrites history.
  Rendering happens in local scratch and publishes in one batched pass of
  atomic copies, because a stage directory is normally a Drive or GCS-FUSE
  mount and both matplotlib's `savefig` and mediapy's encoder write
  incrementally — every flush of a 700 KB mp4 was a separate round trip
  interleaved with the encode. On run `20260801_021545` stage 1 that is
  **130 writes against the mount reduced to 13** for the same 8.15 MB; the
  time saved is a fraction of a second, so the durability is the real gain,
  and the ~30 model and vecnorm files under `models/` remain the dominant
  Drive cost and are untouched here. It also means a runtime that dies mid-encode
  leaves the previous complete artifact set in place rather than a truncated
  video, and that the replays which *did* render still land when a later one
  raises. `evaluations.npz`, `diagnostics.npz` and `evaluation_{selected,final}.csv`
  deliberately do **not** move: the first two are written on the training hot
  path and read by a dozen call sites including the sweep tooling, and the
  evaluation CSVs are the publication evidence contract `result_bundle.audit`
  names by fixed relative path. Filenames inside `replays/` are unchanged and
  still carry the redundant `<species>_<algo>_stage<N>` prefix the path already
  states; dropping it is a separate rename.
- **The stage-1 reward rails are sized to reject collapse, not to approximate
  competence** (0.60 × each statue's standing reward, superseding
  PLANT_VALIDATION §12's 0.89 ×): trex 2900 → **1950**, velociraptor and
  brachiosaurus 1550 → **1050 / 1040**, dibothrosuchus 2300 → **1560**. §12
  picked 0.89 to sit just below a competent policy, which is competence-bar
  reasoning that §9 refutes — the statue is the reward optimum, so no
  threshold separates competent from passive. Sized instead to the rail's one
  real job: the measured collapse (full-horizon 93% → 7%) bottomed at
  **888 = 0.27 × statue**, so 0.60 clears it by better than 2×, whereas 0.89 =
  2900 sat within ~2.4% of a competent policy's estimated ceiling (~2970 =
  statue − the measured 0.30/step smoothness cost, before energy and the
  posture terms a moving policy gives up) and risked rejecting the very policy
  it was meant to admit. `collapse_peak_floor` stays at 0.75 ×, which *should*
  sit near a good level since its job is arming a detector rather than
  admitting a policy.
- **`min_avg_episode_length` is now enforced on the JAX path.**
  `check_stage_gate` read only `min_avg_reward`, and the trainer's
  `eval_metrics` carried no length key at all — so the gate kind named
  `reward_and_length/v1` was fully enforced on SB3 and half-enforced on JAX,
  the exact "one backend silently ignores a gate the other enforces"
  divergence `gate_schema` exists to prevent. It became load-bearing when
  stage 1 began encoding its full-horizon ≥ 95% floor in that field. The
  trainer now tracks `episode_length_history` and emits `mean_episode_length`
  (NaN windows collapse to 0.0, which fails a length gate rather than passing
  it), and the gate enforces both halves; declaring the threshold without the
  metric raises rather than passing on reward alone.
- **The ground settle no longer probes geoms that cannot touch the ground.**
  Both backends included `contype=0, conaffinity=0` geoms — 2 on T-Rex, 5 on
  brachiosaurus, **14 of dibothrosuchus's 39** — when deciding where the floor
  is. Settling to one would hold the animal's real feet above the ground,
  which is the hover the settle exists to prevent. Measured latent (no phantom
  is currently the lowest geom at any shipped noise), so the filter is
  behaviour-preserving: resets are verified **bit-identical across all four
  species** after the change. A new invariant test pins that only collidable
  geometry can drive the settle.
- **Stage 1 moved to the 1a operating point and the reward gates were
  re-founded as sanity rails** (PLANT_VALIDATION §7/§9/§12; changes what
  stage 1 *is*, per decision §17.2). Every species' stage-1
  `reset_noise_scale` is now **0.05** (T-Rex was 0.10, dibothrosuchus 0.30):
  the noise sweep on the repaired plant showed reset noise does not make
  standing harder (statue standing reward nearly flat, 3288 → 3167 across
  0.01–0.20 on T-Rex) — it decides how often you get to stand at all
  (full-horizon 100% at ≤ 0.05, 65% at 0.10, 22% at 0.20). At the old
  operating points survival and stance quality were inseparable in every
  metric, which is why no scalar gate ever said anything useful; at 0.05 a
  statue survives 100% and stance-quality gates can measure stance quality.
  Robustness becomes stage 1b's job via declared `xfrc_applied`
  perturbations, where a lucky draw and a good controller are
  distinguishable — not reset noise. The four `min_avg_reward` values (all
  previously cleared by their own species' statue, certifying nothing) are
  now §12 sanity **rails** at 0.89 × each statue's standing reward at 0.05:
  trex 2900 (statue 3271.8 ± 12.0), velociraptor 1550 (1745.8 ± 5.0),
  brachiosaurus 1550 (1739.1 ± 1.2, on the repaired plant), dibothrosuchus
  2300 (2598.3 ± 0.9). A rail sits *below* the statue on purpose — above it
  is unreachable, since the statue is the reward optimum — and only rejects
  a policy that discarded most of the available return; `min_avg_episode_length`
  rises 750 → **950** on all four to encode §12's full-horizon ≥ 95% floor,
  and `collapse_peak_floor` moves to 0.75 × each statue's standing reward
  (still absolute; relative floors are §14 item 3 and wait for the fresh
  run). All values are provisional per §12 ("for review, not adopted") and
  the real 1a gate — the episode-level `stance_success` event over
  unsupported duty — remains unbuilt (tracked in KNOWN_ISSUES).
- **T-Rex home stance corrected to a flexed theropod limb** (breaking — plant
  change, physics revision 4 → 5 **and** policy interface revision 6 → 7; all
  existing T-Rex checkpoints are invalidated): the home keyframe stood the
  animal on a 172.1° interior knee, 7.9° from full extension and inside the
  singular region for leg-length control. Hip-to-ankle distance changed at
  only 0.024 m/rad there, so servicing stage 1's live height term
  (`height_weight = 1.0`) cost 23.7° of knee travel per centimetre of pelvis
  height — geometry alone accounting for the 31°-per-step knee excursions that
  three `smoothness_weight` escalations (0.1 → 0.7 → 2.0) reduced but could
  not remove. The keyframe now flexes to a **135.0°** interior knee: 45° of
  flexion from a fully columnar limb, the figure Hutchinson, Anderson, Blemker
  & Delp (2005), "Analysis of hindlimb muscle moment arms in *Tyrannosaurus
  rex* using a three-dimensional musculoskeletal computer model",
  *Paleobiology* 31(4):676–701, use for *T. rex* in Fig. 8, converted by their
  Table 1 rule ("for the knee subtract the angle here from 180°"). Leg-length
  authority rises to 0.134 m/rad, i.e. **4.3° of knee per centimetre**. Femur
  and tibia sit symmetrically 22.5° either side of vertical, the one
  inclination that holds the hip over the ankle at this plant's tibia:femur of
  1.000, so the CoM still sits 33.8% of the way heel-to-toe through the
  support polygon against 34.3% before. The three rotations sum to zero, so
  the metatarsus keeps its 21.8° digitigrade slope and the plantar pads and
  digits keep their 0.5 mm contact; segment vectors, masses and inertias are
  untouched. Leg `ctrlrange`s were re-centred on the new pose at unchanged
  width, preserving the `home-keyframe-residual/v1` invariant that the range
  midpoint *is* the home control — which incidentally cuts reachable knee
  hyperextension from 40.2° to 5.0°. Every dependent constant was re-measured
  on the new plant rather than adjusted by eye: `target_z` and
  `target_standing_z` 0.9757 → 0.9260, `natural_pitch` 0.05 → 0.027,
  `nosedive_termination_threshold` 0.47 → 0.493 (holding the absolute −0.520
  forward_z envelope), `healthy_z_range` (0.75, 1.6) → (0.70, 1.55), and
  `min_avg_reward` 1900 → 1840 from a re-run zero-action baseline of
  1743.73 ± 1275.54 (was 1800.56 ± 1267.66). `visual_revision` is deliberately
  unchanged: that layer fingerprints body-local geom/site/material/camera
  definitions, which a pose edit does not touch. See
  `docs/TREX_LEG_FLEXING_PLAN.md`.

- **Velociraptor knee actuators raised to 1.5×kp** (breaking — plant change,
  physics revision 1 → 2): the knee measured 0 % clip at the moderate
  2.5 Hz/0.8-amplitude contract gait but 30–46 % at sprint-like 3–4 Hz
  full-amplitude excitation while still capped at 0.8×kp (`forcerange`
  ±145 on kp=180) — the same clipped-torque signature that collapsed
  stage-2 locomotion at the hips. At ±270 the same sprint excitation
  measures 0 % knee clip; the moderate-regime contract and the
  home-control no-saturation guarantee are unchanged, and a new
  sprint-regime contract test pins the knee headroom.

- **Eval-collapse early-stop now uses a robust peak, raw-eval patience, a
  gate floor, and looser stage-1 settings**: the backstop takes the maximum
  full-window rolling median (default 5 evals,
  `collapse_smoothing_window`) of the per-evaluation mean rewards as its
  reference peak, then counts each newly arrived raw per-evaluation mean
  below `(1 − drop_fraction)` of that peak once toward `collapse_patience`.
  Detection waits for both `collapse_min_evals` and a full median window,
  and only arms once the robust peak clears the stage's `min_avg_reward`
  gate (overridable via `collapse_peak_floor`). Separating the peak
  estimator from patience rejects noise without multiplying it across
  overlapping windows: a variance-inflated early evaluation (run
  `20260720_203454`'s 50k eval, 261.79 ± 261.72) cannot dominate the median
  peak, while one low evaluation contributes exactly one patience strike.
  A healthy but bimodal gait transition is judged by its per-evaluation
  mean rather than `mean − std`, so intermittent healthy evaluations reset
  patience instead of reading as a sustained collapse.
  All three stage-1 TOMLs also gain the explicit
  `collapse_min_evals = 20` / `collapse_patience = 10` /
  `collapse_drop_fraction = 0.5` overrides stage 2 already had.
  (This supersedes the intermediate `mean − std` peak/current rule shipped
  earlier in this release, which correctly fixed the stage-1 abort but then
  aborted run `20260722_124556`'s stage 2 at 1.75M — 200k steps before it
  recovered to 2707 — because a bimodal transition deflates `mean − std` on
  the current side. The regression is pinned by tests built from that run's
  trace and the identical recovering trace `20260721_141523`.)

- **Velociraptor stage-3 budget raised 8M → 12M timesteps**: in run
  `20260721_141523` the best strike checkpoint landed at 7.9M of 8.01M
  with success still climbing (last evals 0.63 → 0.87 → 0.77); strike
  discovery produced no successes until ~5.5M, leaving only ~2.5M of
  effective hunting curriculum. 12M matches the brachiosaurus stage-3
  budget.

- **The four largest `environments/shared` modules are now packages**, each
  split along the seams its own docstring already described. Public surfaces
  are unchanged — every package `__init__` re-exports exactly what the module
  exported, so no import site outside the packages moved:
  - `reporting.py` (1,929 lines) → `formatting`, `gates`, `csv_output`,
    `text_summaries`, `summaries`, `bundles`, `stage_artifacts`
  - `plant_contract.py` (1,822) → one module per contract layer
    (`source_layer`, `policy_layer`, `physics_layer`, `visual_layer`) over
    `constants`/`errors`/`digests`/`identity`/`versions`/`introspection`, with
    `manifest`, `validation`, and a `__main__` for the existing
    `python -m environments.shared.plant_contract --check` entry point
  - `curriculum.py` (1,316) → `sb3_compat`, `manager`, `schedules`,
    `advancement`, `early_stopping`, `checkpoints`; the manager no longer
    shares a module with the SB3 integration
  - `result_bundle.py` (1,201) → `constants`, `errors`, `naming`, `hashing`,
    `provenance`, `manifest`, `evidence`, `audit`

  Three test patch points moved with the code they belong to: the plant
  contract's `configs/` paths are read through its `constants` module,
  `_SB3_AVAILABLE` through `curriculum.sb3_compat`, and the provenance helpers
  through `result_bundle.provenance`. Each is now a single patch point rather
  than one per consuming module.

- **Fixed: default provenance capture resolved the repository root one level
  too shallow.** `initialize_result_bundle` derived it as
  `Path(__file__).resolve().parents[2]`, correct while the code lived in
  `environments/shared/result_bundle.py` but off by one once it moved into
  `result_bundle/provenance.py`, where it resolved to `<repo>/environments`.
  `git status` reports repo-wide while `git ls-files --others` is scoped to its
  working directory, so an untracked file under `configs/` or the repository
  root set `repository_dirty` without entering `repository_patch_sha256` —
  materially different dirty trees could share one provenance hash. The root is
  now `REPOSITORY_ROOT` in `result_bundle/constants.py`, derived from a named
  `_SHARED_ROOT` anchor rather than a bare parent count, read through the
  `constants` module so there is one authoritative binding, and pinned by three
  tests — one on the constant, one on the initializer's default wiring, and one
  asserting a root-level untracked file changes the patch hash. Note the expression was byte-identical across the move, so an AST-level
  verbatim check could not see it; only evaluating it could.

- **The per-species script harnesses moved to `environments/shared/harnesses/`**
  and no longer use `test_` names: `test_env_base.py` → `harnesses/env_smoke.py`,
  `test_actuators_base.py` → `harnesses/actuators.py`, `view_model_base.py` →
  `harnesses/viewer.py`, with `test_actuators()` → `run_actuator_test()` and the
  internal `test_*` checks → `check_*`. These are hand-run smoke checks and
  viewers, but pytest collected them as tests — and errored on their unfillable
  `env_class`/`cfg` arguments — for any invocation that reached them, which the
  repository's `testpaths`/`norecursedirs` settings were the only thing
  preventing. The coverage `omit` entry added to work around the old naming is
  replaced by one covering the package.

- **CI now measures the coverage it actually produces.** Three jobs ran tests
  without `--cov` — `test-jax-cpu`, `test-sb3` and `plant-contract` — and those
  are precisely the jobs carrying the optional dependencies. The only job
  reporting a number for `environments/shared` was `test-shared`, which installs
  neither JAX nor SB3, so the modules needing them were omitted from the report
  as "not available in the standard CI test environment". That stopped being
  true when `test-jax-cpu` was added: 1,840 statements of JAX/MJX and 495 of
  `train_base.py` — 23% of the production code in `environments/shared` — were
  being exercised by CI and excluded from its number, which read 79.57% over the
  remaining 7,652 statements.

  Deleting the omits alone does not work; it drops `test-shared` to 66%, under
  the `fail_under = 70` gate, because that job genuinely cannot import those
  modules. So every test job now writes its own `.coverage.<job>` and uploads
  it, and a new `coverage` job combines them and gates the union. Per-job
  reports are suppressed with `--cov-fail-under=0`, since no single job's slice
  is meaningful alone. `relative_files` is enabled so the combine lines up
  across jobs. `harnesses/` and the mjlab adapter stay omitted, now with
  accurate reasons — hand-run, and GPU-only with no job able to reach it.

  The combined gate reports **80% over 11,071 statements**. On the comparable
  `environments/shared` slice that is 78.07% over 9,987 statements against the
  old 79.57% over 7,652 — the figure barely moves while the denominator grows
  31%, which was the point. `fail_under` stays at 70 rather than being tuned to
  the result. The combine log reads `Combined 15 files, skipped 3` against 18
  artifacts: coverage.py content-hashes each data file and skips one whose
  bytes match a file already combined, so three matrix runs that produced
  identical data to a sibling Python version contributed nothing new. No data
  is dropped.

### Removed
- **The top-level `Images/` directory** (25 MB): its three GIFs were
  byte-identical to the copies under `website/static/img/`, which are the ones
  the site and README actually reference. Nothing in the repository referred to
  `Images/` except the `.dockerignore` entry excluding it, now also dropped.

### Added
- **`stance_duty_validation.py`, which shows the support-duty metrics measure
  what they claim.** `unsupported_duty` and its siblings classify each step by
  thresholding the foot touch sensors at 0.1 N, and that reading is the evidence
  behind the claim that the stage-1 policy is off the ground ~21% of the time —
  a sensor that under-reports contact looks exactly like a foot in flight. The
  static check above covers one operating point; this sweeps driver policies
  producing **3% to 67%** airtime and compares the sensors against both
  `mj_contactForce` over foot-geom contacts and `mj_geomDistance` from every
  foot geom to the floor, the latter computed from geometry alone and so
  independent of the sensor path. The metric tracks kinematic truth to within
  **0.52%** of steps across the whole range (2.6% only under uniform-random
  actuation, which no policy occupies), and the two error directions nearly
  cancel. The decisive row is low-amplitude jitter — **16.07% true airtime
  against 16.21% reported**, straddling the figure under dispute. Combined with
  the 0.1 N threshold sitting against ~421 N per foot in quiet stance, the
  sensor-artifact explanation for `STAGE1_SPLIT_PLAN.md` §1.5 is exhausted and
  its hopping reading can be relied on. This validates the *instrument*, not the
  policy: the checkpoint was not replayed, and the causal half of §1.5 ("the
  reward caused the hop") still needs a counterfactual run.
- **`foot_sensor_report.py`, and a four-species audit of what the foot touch
  sensors actually measure.** A MuJoCo touch sensor sums only contacts on geoms
  belonging to its site's own body, so a site on a parent segment silently
  misses whatever the child geoms carry — a failure invisible from the reward
  trace, because a sensor reading zero looks exactly like a foot off the ground.
  Checking every species against `mj_contactForce` found T-Rex and dibothrosuchus
  correct (ratio **1.000**) and two species wrong: **velociraptor at 0.553**,
  missing its `metatarsus` (17.54 N) and lateral `toe_d4` (12.03 N) per foot
  because the site sits on `toe_d3` alone, and **brachiosaurus at 0.000**, blind
  to all 1699.2 N of its floor contact. Total floor reaction equals body weight
  on all four, so the contacts are real and these are sensor-scope defects, not
  physics ones. Neither reaches a stage-1 reward term today, but both reach the
  **observation**: brachiosaurus trains with four permanently zero input
  channels (indices 75–78 of 83) and the raptor's policy sees 55% of true
  per-foot load. Repairs are per-species MJCF changes and are *not* included
  here. Recorded in `docs/investigations/FOOT_SENSOR_VERIFICATION.md`, which
  also corrects the ground-reaction-force reading in `STAGE1_SPLIT_PLAN.md` §6.2:
  the T-Rex statue's 841 N is *exactly* body weight (840.9 N, ratio 1.002), and
  the 1483 N that makes it look low is `mj_getTotalmass` counting the 65.45 kg
  **prey body** — 43% of the model total. A GRF diagnostic must divide by the
  animal's kinematic subtree or it reports a false 0.57 on a plant standing in
  perfect equilibrium.

### Fixed
- **The stage-1 reward now charges the two things run `20260801_021545`
  proved it could not see** (PLANT_VALIDATION §14 items 1, 2 and 4). That run
  — T-Rex, 6.0M steps on the repaired plant at the 1a operating point — ended
  at **28.4% unsupported duty** against the statue's 0.000, versus 30–35% on
  the broken plant. §16's headline question ("was the airborne duty learned
  from the broken reset?") is therefore answered **no**: the plant repair
  moved it ~19% relative, so the chatter is a reward-design problem. Three
  changes follow:
  - `foot_load_balance_min_support_force` is **42.0** on T-Rex stage 1, 5% of
    the animal's own weight (85.72 kg excluding the prey prop = 840.9 N). The
    previous `0.0` was a *measured* no-op — two touch readings essentially
    never sum to exactly zero, so the airborne branch never fired once across
    709 logged rollouts. It now sits well above the duty metrics' 0.1 N/foot,
    so reward and diagnostics agree about what "unsupported" means.
  - A new `foot_load_balance_airborne_penalty` (0.3) makes the ordering
    **strictly monotone**: `both feet even +0.600 > single support −0.300 >
    airborne −0.600`. Previously airborne and single support both scored
    −0.300 — a flat region with no gradient out of the air, on the stage whose
    whole job is staying on the ground.
  - A new frequency-aware `action_jerk_weight` (1.0) penalises the **second**
    difference of actions. `smoothness_weight` charges first-difference
    magnitude and is blind to frequency: from the best to the final checkpoint
    of run `20260731_132102`, `action_delta` *fell* 12.0 → 10.5 and its penalty
    *improved* while toe-motion power above 4 Hz doubled. The new term inverts
    that — a slow ramp scores jerk 0.00 (though it has the *higher*
    `action_delta`) while a Nyquist-rate buzz scores 336.
  - `derive_stance_info` keys its airborne branch to the same
    `> contact_threshold` test the duty metrics use, so a foot at 0.001 N is
    unsupported in both instead of supported in one. It previously reported
    *perfect balance* for a truly airborne pair, and never fired anyway.
  All wired through the Gymnasium, MJX and JAX paths with matching parameters;
  every new weight defaults to `0.0`, so untouched species and stages are
  numerically unchanged.
- **`collapse_peak_floor` can be relative, because the absolute one failed to
  arm a second time.** Set to 2450 (0.75 × the statue) it sat above run
  `20260801_021545`'s best evaluation of **2347.67**, so the detector never
  armed and watched eval degrade 2347.67 → 1666.33 — the identical failure to
  the 2200-vs-1934.1 case in §11.4. Deriving the number from the statue was
  only half a fix: the statue bounds what is *achievable*, not what a
  *learning* policy passes through. `collapse_peak_floor_fraction` ×
  `collapse_peak_floor_reference` (0.45 × 3271.8 = 1472 on T-Rex stage 1)
  re-anchors whenever the reward changes; the failing run cleared it by ~2M
  steps while it still sits well above the 888 collapse bottom. An explicit
  `collapse_peak_floor` still takes precedence, so existing configs are
  unchanged, and a half-declared pair resolves to "never arm" rather than to a
  silently low floor.
- **Brachiosaurus stage 1 is a balance task again** (breaking — plant change,
  physics revision 2 → 4, policy interface revision 4 → 6, visual revision
  1 → 2; all existing brachiosaurus checkpoints are invalidated). The
  zero-action statue fell on 40 of 40 episodes at mean length 130.7, so no
  stage-1 result was interpretable (PLANT_VALIDATION §6, KNOWN_ISSUES). The
  collapse decomposed into two measured defects. First, BrachioEnv was the
  only species still on the inherited midpoint action mapping, and its home
  stance is not the ctrlrange midpoint: "do nothing" dragged the rear knees
  0.349 rad, the front knees 0.262 rad and all four ankles 0.175 rad away
  from the standing pose on every step; commanding the actual home controls
  tripled statue survival (mean length 141 → 455) on its own. Both backends
  now use `home-keyframe-residual/v1` like the other three species — on MJX
  this also moves the reset base pose from `qpos0`, which hovered 610 mm
  above the floor, to the home keyframe. Second, the leg position servos
  could not statically carry the animal: it sagged 71.6 mm under its own
  weight, putting the planted stance's lateral (roll) stiffness (~1.9·10³
  N·m/rad) at parity with the destabilising gravity stiffness m·g·h
  (~1.77·10³), so the statue tipped over in slow roll on 9 of 10 resets even
  when commanding the exact home pose — the same-signed slow roll to
  `fallen` visible in every trace. Leg kp is doubled (sag 11.1 mm), with
  forceranges scaled so each actuator keeps its documented
  force-to-stiffness sizing. Zero action, 40 episodes at stage-1 noise 0.05,
  before → after: reward 163.35 ± 81.40 → **1739.08 ± 1.17**, full-horizon
  **0/40 → 40/40**, terminations (fallen 34, tail_contact 6) → (truncated
  40). The ± 1.17 spread is the tightest of any species, making
  brachiosaurus the best-conditioned gate-prototyping plant — the property
  PLANT_VALIDATION §6 wanted. The full-horizon neutral-action test the
  KNOWN_ISSUES entry prescribed now exists on the brachiosaurus static
  balance suite.
- **Brachiosaurus can feel its feet.** All four foot touch sensors read
  exactly 0.0 N while `mj_contactForce` reported 98.8% of body weight over
  four real floor contacts (FOOT_SENSOR_VERIFICATION §4) — the policy
  trained with four permanently dead input dimensions, and the support-duty
  diagnostics scored the standing statue as 88.5% airborne (PLANT_VALIDATION
  §8). Two scope defects: the pad sites were r=0.03 spheres that did not
  contain the pad's floor contact points, and the metacarpus/metatarsus
  carries part of the load on the pad's *parent* body, invisible to any site
  on the pad body. The pad sites are now boxes enclosing the pad's contact
  zone, four meta touch sensors are appended at sensordata 26–29 so existing
  indices keep their positions, and both backends sum pad + meta per leg —
  the same repair shape as the T-Rex digit sensors (`aa87445`). Verified:
  summed sensors report **100.0%** of the independently measured
  `mj_contactForce` floor total in settled stance (fr/fl 313 N, rr/rl
  535 N), and statue stance quality at noise 0.05 moved from all-feet 0.000
  / unsupported 0.885 to **all-feet 0.998 / unsupported 0.000** — in line
  with the other three species. On attribution: with the stance repair above
  in place, the **pad-site enlargement alone** accounts for that 100.0% —
  the meta capsules no longer reach the floor and the meta sensors read
  0.0 N at home. They were carrying real load on the pre-repair sagging
  plant and remain correct cover for poses that put the meta back down, but
  nothing should read them as load-bearing today. Two gaps stay open:
  velociraptor's 45% under-read (same doc, §3), and the fact that only pads
  and metas are instrumented — a brachiosaurus kneeling on its shins
  (measured: shins carrying the full 1695 N with the feet clear) reads zero
  on every foot sensor, so support-duty scores it as airborne. That errs
  safely for a stance gate, but distinguishing "kneeling" from "airborne"
  would need shin instrumentation. With this and the stance repair, every species' §8
  stance-quality row is now interpretable, unblocking the §12 brachiosaurus
  gate row.
- **The MJX reset now settles the animal on the ground, closing the plant gap
  the PR #479 review found.** The `ca56f6c` ground settle landed only in
  `BaseDinoEnv.reset`; the MJX reset kept applying joint and yaw jitter with
  the root height fixed at the keyframe value — the same defect class, measured
  on T-Rex at the stage-1 noise (0.10) as spawns from **41.2 mm inside the
  floor to 5.2 mm above it**. The midpoint action mapping was worse: its reset
  base pose is `qpos0`, which on brachiosaurus hovers **610 mm** over the
  floor, so every MJX episode opened with a half-metre free fall. MJX has no
  `mj_geomDistance`, so the settle probes an analytic support extent per geom
  — exact for the sphere/capsule/cylinder/box/ellipsoid primitives the species
  use over a horizontal plane floor, and verified against `mj_geomDistance` to
  ~1e-9 across random poses on all four species. The settle target is the home
  keyframe's authored clearance (the `BaseDinoEnv.home_ground_clearance`
  semantics), probed once at construction through the reset's own kinematics
  path, so a noise-free home-residual reset computes a shift of exactly `0.0`
  and stays bit-identical to the pre-settle behaviour; only jittered and
  midpoint spawns move. Jittered spawns now land within 0.4 µm of the authored
  contact, deterministically in the PRNG key, with no extra RNG consumed. A
  new `test_mjx_reset_plant_invariants.py` suite pins penetration, hover,
  determinism, the bit-exact no-op, the brachiosaurus hover fix, and
  cross-backend agreement of the settle target — every assertion cross-checked
  with CPU `mj_geomDistance` rather than the implementation's own formula.
  Cost: one extra `mjx.kinematics` pass (the cheap position stage, no
  collision or constraint work) per reset, which the fused auto-reset step
  computes every step; the final full forward still runs once, on the settled
  pose. No plant revision bump: entry 6's bump already declared reset placement moved
  for every species, and no MJX checkpoint or baseline was produced between
  the two landings (addendum recorded in `configs/plant_versions.toml`).
- **A declared reward gate must now carry its reward threshold.** The gate
  schema only rejected *misplaced* threshold keys, so a config declaring
  `gate_kind = "reward_and_length/v1"` with no thresholds at all passed
  validation, produced no `threshold_fields`, and fell through to
  `StageThreshold`'s permissive defaults (`min_avg_reward = -inf`) on the SB3
  path — advancing on any evaluation — while the JAX path raised on the same
  config. That is precisely the backend divergence the schema exists to
  prevent. Each gate kind now declares required threshold fields
  (`reward_and_length/v1` requires `min_avg_reward`), enforced in
  `validate_gate_config` whenever the kind is declared, so both backends
  reject the shape identically; the JAX path's now-redundant local check is
  removed.
- **`BaseDinoEnv.lowest_ground_clearance` honours its `data` argument.** The
  parameter was accepted and silently ignored — the probe always measured
  `self.data`, and `home_ground_clearance` had to swap `self.data` out to use
  a scratch buffer. A caller passing a scratch pose got the live pose's
  clearance with no error. The probe now reads the passed `MjData` (pinned by
  a test probing the home pose shifted 0.1 m down), the swap hack is gone, and
  the noise-free reset state is verified bit-identical across the change.
- **The reset's root-height jitter is now documented and pinned as
  state-inert.** The ground settle overwrites the root height as a pure
  function of the sampled joint pose, which silently made
  `reset_height_noise_scale` and PR #478's `_bounded_reset_height_delta`
  dead — verified to one ULP with RNG streams identical across height scales.
  Behaviour is unchanged: the draw is deliberately kept so seeded resets and
  the freshly re-measured baselines stay anchored, the inertness is now stated
  at every definition site, `TestHeightJitterIsInertSinceGroundSettling`
  pins both the inertness and the stream alignment, and a KNOWN_ISSUES entry
  schedules the channel's removal for the next policy-interface revision.
- **Being airborne is no longer cheaper than honest single support.**
  `reward_foot_load_balance` computed `|R − L| / (R + L + 1e-8)`, which
  evaluates to **zero** when both feet read zero — so a plant off the ground
  scored the same as one standing evenly on both feet, and strictly better than
  one carrying its weight on a single foot, which pays the full penalty. On the
  stage-1 weights that ordering was `both feet down +0.600 > airborne 0.000 >
  single support −0.300`: flight was the second-best available state on the
  stage whose entire job is to stand still, and a policy off the ground cannot
  reject a disturbance at all. An unsupported pair now reports maximal
  imbalance, giving `both feet down +0.600 > single support = airborne −0.300`.
  A new optional `foot_load_balance_min_support_force` sets the total force
  below which the pair counts as unsupported; it defaults to `0.0`, which closes
  only the exact `[0, 0]` case and leaves **every loaded state numerically
  unchanged**, and can be raised during gate calibration to also deny credit for
  a token grazing contact. Wired identically through the Gymnasium, MJX and
  JAX paths, with the `[0, 0]` case now covered in both `test_reward_functions`
  and `test_trex_mjx_reward_parity` — neither covered it before, which is how
  the hole survived. Note the two failure states are now *tied* rather than
  airborne being strictly worst; separating them needs a term this fix does not
  add.
- **The curriculum advancement gate now fails closed** (breaking — every stage
  config must declare `gate_schema_version` and `gate_kind`). The gate plumbing
  failed *open* in three independent ways, so a stage could advance on evidence
  nobody had checked. `thresholds_from_configs` copied six known keys out of
  `[curriculum]` and **silently discarded everything else**, so a config
  carrying only new-style gate fields produced no thresholds at all and SB3 fell
  back to `StageThreshold`'s permissive defaults (`min_avg_reward = -inf`,
  length and success floors `0`) — which advance on any evaluation whatsoever.
  `jax_curriculum.check_stage_gate` logged a warning and returned `True` when
  `min_avg_reward` was absent. And neither backend rejected an unrecognised key,
  so a misspelled threshold name *disabled* that threshold instead of failing.
  Together these made "no gate" indistinguishable from "gate satisfied", with
  the permissive reading always winning — which is the wrong default for the
  mechanism that decides whether a policy is good enough to build the next stage
  on. A new `environments/shared/curriculum/gate_schema.py` makes the
  declaration explicit and versioned: unknown keys, unknown gate kinds and
  unsupported schema versions are **fatal** whenever advancement is enabled, and
  a threshold field its declared kind does not consume is fatal too, since it
  implies a gate that is not actually enforced. Running without a gate is still
  possible, but only by declaring `gate_kind = "none/v1"`, which is recorded in
  the config and *refuses* to advance rather than passing by default. All twelve
  stage configs declare `reward_and_length/v1`; the existing tests that asserted
  the permissive behaviour were updated in the same change, as they were the
  reason the defect survived. Effective thresholds for all four species are
  unchanged.
- **`collapse_peak_floor` no longer inherits `min_avg_reward`**, which coupled
  an early-stop backstop to an unrelated advancement threshold. The builder
  chained `collapse_peak_floor` → `min_avg_reward` → `0.0`, and only
  `configs/trex/stage1_balance.toml` set the key explicitly (1 of 12), so
  removing a stage's reward gate — which a state-capability gate would do —
  silently dropped its arming floor to `0.0` and armed collapse detection after
  *any* positive robust peak. A missing floor now means **never arm**, because a
  backstop that is not configured should not abort a run; the eleven configs
  that were relying on the fallback now set the value it produced, so every
  effective floor is bit-identical (`100.0` everywhere except T-Rex stage 1's
  `2200.0`). `collapse_smoothing_window` was also readable but undeclared, and
  is now part of the schema.
- **Reset can no longer generate an already-terminal episode** (breaking —
  policy interface revision bumps for the three home-keyframe-residual species:
  velociraptor 6 → 7, T-Rex 7 → 8, dibothrosuchus 3 → 4; brachiosaurus does not
  carry `home_reset` and is unchanged). The root-height jitter was the **only
  unbounded term in the reset** — every other one is a bounded uniform — so
  `BaseDinoEnv.reset` drew `normal(0, height_scale)` with nothing stopping it
  from placing the root outside `healthy_z_range` before the policy acted.
  T-Rex is the species it actually broke: a 0.926 m home pelvis sits 0.226 m
  above the 0.70 m floor, which at σ = 0.10 m is only **2.26σ**, so a predicted
  **1.19%** of spawns started sub-floor. Measured over seeds 3042–5041, 18/2000
  spawned below the floor and **16 of those terminated on step 1 whatever the
  policy did**; a wider scan of 3042–7041 found 43/4000. The draw is now
  truncated to the distance to the nearer end of `healthy_z_range` less a 0.02 m
  margin, which takes sub-floor spawns to **0/4000**. The bound is symmetric, so
  the mean spawn height is unchanged (0.9262 → 0.9261 m over 2000 seeds) while
  the standard deviation tightens 0.1007 → 0.0980. It binds at 2.07σ for T-Rex
  (3.9% of draws) against 3.6–3.8σ for velociraptor and dibothrosuchus (~0.02%),
  so those two move only in the far tail. Dibothrosuchus hit this same defect
  first and fixed it by decoupling `reset_height_noise_scale`; T-Rex was left
  coupled and nobody re-checked it.

  **This does not meaningfully move the zero-action baseline, and that is the
  point.** Re-measured on seeds 3042–3081 the statue is unchanged where it
  matters — still 23/40 full-horizon, `reward standing` still 3244.04 ± 23.55,
  unconditional mean 1971.57 → 1976.62 — because a statue fails those seeds
  anyway. What the defect capped was the **competent** policy: an episode that
  ends on step 1 regardless of the action is not a policy failure, and counting
  it as one put an unreachable ceiling on any reliability gate. Seed 3077 is
  precisely the first failure in the 39/40 evaluation panels that
  `docs/STAGE1_SPLIT_PLAN.md` §2.3.1 reports, and it is this bug rather than the
  policy. Existing checkpoints for the three bumped species were trained against
  a different reset distribution and must be re-baselined before any stage-1
  gate is calibrated against them.
- **T-Rex stages 2 and 3 no longer terminate at a different pitch angle on the
  MJX path than on the Gymnasium one**: `nosedive_termination_threshold` is the
  per-stage tunable pitch gate, and only `stage1_balance.toml` sets it. When a
  stage leaves it unset the two backends reached for different fallbacks —
  `TRexEnv.__init__`'s **0.62** on the Gymnasium path against the generic
  `weights.get("nosedive_termination_threshold", 0.5)` literal in `mjx_env` on
  the MJX one. With `natural_forward_z = -sin(0.027)` that is a termination at
  40.3° nose-down against 31.8°: MJX killed the episode **8.5° earlier**, in
  exactly the two stages whose configs deliberately ask for a head-forward
  running posture (`nosedive_weight` 1.5 → 0.5 → 0.2, `posture_weight` 1.5 →
  0.2 → 0.1). The comment at `environments/trex/mjx_config.py` asserted the two
  paths already agreed on 0.62, and the parity class that pins this category of
  parameter (`TestSB3MJXEnvelopeParity`, added for `healthy_z_range` and
  `max_tilt_angle`) did not cover it. The T-Rex registry now carries the value,
  making the generic fallback unreachable; registry weights are merged before
  the TOML overlay, so stage 1's calibrated 0.493 still wins. No plant
  fingerprint moves — reward and termination code are deliberately outside the
  policy-interface payload — so no revision bump is required and existing
  checkpoints stay valid. Found during the July 2026 T-Rex review; see
  `docs/investigations/TREX_REVIEW_2026_07.md` finding F2. The same shared 0.5
  fallback exists for the other three species, whose Gymnasium defaults also
  differ (Dibothrosuchus 0.55); those were out of the review's scope and are
  left for a general fix.

- **T-Rex neutral stance now has a viable training basin** (breaking —
  policy-interface revision 1 → 2, physics revision 1 → 2, visual revision
  1 → 2): the old `home` state embedded the feet as much as 9 cm into the
  floor, loaded the rear metatarsus behind the whole-body COM, made each leg
  spring pull toward numeric zero instead of the stance, and used raptor-scale
  servo gains on an 85 kg plant. A shallow plantar pad now supplies a real
  support surface; the reset is a 0.5 mm loaded contact; every leg spring
  references the stance; lower-body joint references preserve a small ankle
  gravity preload; and the hip/knee/ankle gains use mass scaling plus a tested
  stance margin while retaining bounded 1.5×kp headroom. The complete policy
  now uses the named-home residual mapping in both Gymnasium and MJX, so action
  zero commands all 21 home controls while ±1 still reach the actuator limits.
  The load-bearing plantar pads also own hidden box-shaped touch volumes:
  settled readings are ~419 N per foot, airborne readings are exactly zero,
  and four contact exclusions remove the former 28 mm adjacent-toe overlaps
  without changing sensor order or observation dimensions.
  The JAX factory also leaves the CPU evaluation model's XML solver and
  parent-contact options unchanged, eliminating evaluation-only plantar/toe
  self-contact and keeping checkpoint gates aligned with MJX training.
  The old plant nosedived at step ~77 even under its XML home command. The
  corrected plant survives all 1,000 Stage-1 steps in 50/50 noise-free probes,
  50/50 JAX-style probes at 0.05 rad joint noise, and 47/50 full SB3 resets
  (the raptor reference is 48/50), with peak actuator force at 48.1% of its
  bound. CPU-MJX tests pin the same complete home reset, residual action
  mapping, and live contact observations. Root-pitch, tail-spring,
  neck-stiffness, COM-shift, and static-control A/Bs were rejected before
  changing the stance mechanics. See
  `docs/investigations/TREX_HOME_EQUILIBRIUM.md`.

- **`metrics.json` now describes the promoted checkpoint**: the post-stage
  quality/velocity/success evaluation loaded SB3's mean-reward
  `best_model` while next-stage training loads `robust_best_model` when
  present, so the sidecar could describe a policy the curriculum never
  promoted (in run `20260720_203454`, the degenerate lucky-peak 50k
  checkpoint). Both paths now share one selection helper
  (`robust_best_model` → `best_model`, each only with its matched
  VecNormalize stats), and the sidecar records which checkpoint it
  evaluated as `quality_eval_checkpoint`.
- **Velociraptor foot touch sensors were dead during normal stance**
  (breaking — policy-interface revision 2 → 3, visual revision 1 → 2): the
  raptor is digitigrade and a lying toe capsule contacts the floor near its
  ends, but the r=0.02 touch sites sat at the toe midpoint, so both sensors
  — and the two foot-contact observation dims fed from them — read 0 while
  standing (verified: six active floor contacts, zero sensor signal). The
  sites now envelop the whole toe_d3 capsule. Fixing them exposed a second
  defect: the adjacent toe digits (d3/d4) interpenetrate at rest and the
  solver held a constant ~300 N phantom repulsion between them, which the
  enlarged sites would have summed even airborne; new contact excludes
  remove it. Verified post-fix: ~37 N per foot at settled stance, exactly
  0 N airborne, 48/50 standing-probe survival unchanged, and regression
  tests pin stance/airborne sensor behavior.
  Follow-up (visual revision 2 → 3): the enlarged sites rendered as large
  gray balls on the feet in run videos (first visible in run
  `20260721_141523`'s stage-1 video); they now live in site group 4,
  which default rendering hides, so they no longer draw — toggle site
  group 4 in an interactive viewer to visualize the touch volumes.

- **`BaseDinoEnv` reset-height jitter is separable from joint jitter**
  (`reset_height_noise_scale`, breaking — policy interface revision for the
  home-keyframe-residual species: velociraptor 4 → 5, trex 4 → 5,
  dibothrosuchus 1 → 2): `reset()` applied `reset_noise_scale` — a joint-angle
  scale in **radians** — as a root-height jitter in **metres**. That is nearly
  harmless on a metre-tall plant (T-Rex: 10 % of stance) and badly wrong on a
  short one. On Dibothrosuchus's 0.313 m stance the committed 0.14 spawned
  trunk heights across 0.050–0.691 m, putting **25 % of stage-1 episodes
  outside `healthy_z_range` before the first step**; all `fallen` and
  `too_high` terminations fired at step 1. The stage-1 baseline therefore read
  as "62 % full-horizon" while measuring unlearnable spawn noise rather than
  balance — genuine balance failures were only ~13 %.

  The new argument defaults to `None`, which reproduces the previous
  arithmetic exactly, so velociraptor, trex and brachiosaurus reset
  distributions are unchanged and their checkpoints remain valid; the revision
  bumps reflect the shared `home_reset` source moving, not a behaviour change.
  `reset_noise_scale` remains the master switch, so zero still gives a fully
  deterministic reset. Dibothrosuchus sets 0.03 m (~10 % of stance, 4.5σ clear
  of its healthy-height floor) and its stage-1 noise is recalibrated from 0.14
  to 0.30 rad on that basis. A regression test pins that no stage-1 seed spawns
  already terminated.

### Added
- **Tests for the unified training entry point** (`environments/shared/train.py`),
  which the newly-honest combined coverage showed at **0%** — the module was
  never imported by any test. It is the `--species` shim in front of
  `train_base.main`, so a break there takes out every documented
  `python -m environments.shared.train ...` invocation while the species
  packages stay green. 13 tests cover both validation exits, the argv rewrite
  (species pair at the front, middle and end; `argv[0]` preserved), the
  hand-off to `get_species_config`/`main`, and an unknown species propagating
  the registry's `ValueError` rather than being swallowed. The `sys.path`
  bootstrap is pinned against repository landmarks rather than its
  `parents[2]` arithmetic — the same depth-counting failure that produced the
  provenance-root bug on this branch — and the `__main__` guard is exercised
  end to end as a subprocess. 0% → 95%, the remainder being the guard body
  itself, which only runs out of process. All nine mutations of the module
  (both exit codes, the argv slice bounds, the parent depth, the guarded
  `sys.path` insert, and swallowing the registry error) turn the suite red.

- **New species: *Dibothrosuchus elaphros*** (77 obs / 27 actuators / 8.65 kg
  / nq=35, nv=34) — a sphenosuchian crocodylomorph and the first non-dinosaur
  in the project. It is a small, gracile quadruped that held its limbs erect
  and parasagittal rather than sprawled, with hindlimbs longer than its
  forelimbs, paired paramedian rows of dorsal osteoderms, and a narrow crested
  skull whose snout drives the Stage 3 "snap" contact proxy (no jaw
  articulation). Ships the MJCF plant, `DibothrosuchusEnv`, MJX registration,
  three curriculum stages plus PPO/SAC sweep spaces, 65 species tests, a
  `test-dibothrosuchus` CI job, and the model documentation page.

  Plant characterization, all measured rather than assumed: the stance is
  authored into the body offsets so **every hinge joint and the whole home
  control vector are 0** at the keyframe; holding that control for 1500 steps
  settles the trunk at 0.3129 m with forward_z +0.003, so `natural_pitch` is 0
  and posture/nosedive keep their world-vertical reference. All four pads
  carry the full 84.9 N of weight, split 71/29 rear/front. Hip pitch, knee and
  ankle carry 1.5×kp `forcerange` on **every** leg (a 1.24 m animal strides
  fast relative to its body length, so all three stance joints see gait-scale
  torque); measured clipping is 0.0 % under home-control settling, a 1.5 Hz
  walk and a 2.5 Hz trot, and 0.5 % worst case under 4 Hz full-amplitude
  excitation. Stage-1 `reset_noise_scale` is 0.30 rad, calibrated against the
  zero-action baseline with root-height jitter decoupled (see below): 0.05,
  0.10 and 0.14 all leave a do-nothing policy at 100 % full-horizon, while
  0.30 leaves 65 % with every failure a genuine topple (earliest step 40,
  median 58) and nothing spawning out of bounds.

- **Canonical result bundles for Colab/Google Drive training**: schema-v2
  summaries, runtime-captured provenance, immutable completed bundles,
  selected/terminal evaluation evidence, plant/config/model/VecNormalize
  hashes, partial-versus-promotion validation, idempotent manifests, and
  strict exporter tests now make failed runs auditable without presenting
  them as publishable results.
- **Velociraptor Stage-1 basin investigation**: records the pre-training
  50-seed physics probe, Commit A's A-only PPO run (`20260720_203454`), the
  natural-lean diagnosis, Commit B's reward geometry, targeted backend-routing
  lessons, and the falsifiable next-run plan.

### Changed
- **Velociraptor actions are residuals around the XML `home` controls**:
  `-1`/`0`/`+1` map to actuator minimum/home/maximum through a piecewise-linear
  transform. This is a breaking policy-interface revision (1 → 2); older PPO,
  SAC, and JAX checkpoints remain historical and cannot be resumed as current
  policies. XML physics and actuator reach are unchanged.
- **Velociraptor posture shaping targets its natural forward pitch** (0.35
  rad) instead of world vertical. The yaw-invariant, direction- and roll-aware
  squared-chord formulation keeps finite JAX gradients at the exact target;
  T-Rex and Brachiosaurus retain vertical posture targets.

### Fixed
- **Targeted natural-posture consistency across training and evaluation
  paths**: the shared NumPy/JAX primitive, direct MJX, JAX total and detailed
  reward paths, CPU evaluation, diagnostics, and runtime `natural_pitch`
  overrides resolve the same reward-only target. Absolute-tilt termination and
  the existing nosedive boundary remain unchanged, and the JAX notebook now
  binds reward functions only after environment creation. Comprehensive
  fixed-state, all-component SB3↔JAX parity remains open.

## [0.3.2] - 2026-07-21 — RL/GCP Review Fixes & Model Physics

### Changed
- **Plateau diagnostics now follow deterministic stage-gate evaluations**:
  rollout return warnings (which fired at different rates for PPO and SAC and
  repeated every rollout) are replaced by a warn-once/re-arm state machine.
  It monitors the currently blocking configured gate — balance duration,
  forward velocity, task success, then reward — and explains that training
  continues independently. SB3 evaluation now reports task success as `N/A`
  when the current stage has no success-rate gate, while preserving genuine
  Stage 3 `0%` results and success arrays.
- **Trex and brachiosaurus gait actuator headroom raised to 1.5×kp**
  (breaking — plant change): the same static-only forcerange sizing that
  broke velociraptor stage-2 measured 44–50 % hip / ~31 % ankle / 10–14 %
  knee gait-cycle clipping on trex (2.5 Hz) and 20–33 % hip clipping on
  brachiosaurus *at a 1.5 Hz walk*. Raised the clipped groups only (trex
  hip pitch/knee/ankle; brachio's four hip pitches) — post-fix clipping
  ≤0.5 %, root-z divergence vs an unbounded plant ≤2 mm. New
  `test_actuator_bounds.py` for both species pins the contract, built on
  shared helpers (`environments/shared/tests/actuator_bounds_helpers.py`);
  the velociraptor test now uses the same helpers. A species-generic
  saturation report lives at
  `environments/shared/scripts/actuator_saturation_report.py` (the
  velociraptor script delegates to it). Policies trained on the old caps
  will not transfer.
- **Entropy decay enabled for velociraptor stage 1** (`ent_coef_end =
  0.001`): run `20260711_235303` — which cleared stages 1–3 and set the
  stage-2 record (2705.93 ± 7.84 @ 2.75 m/s) — showed stage 1 destabilizing
  late (peak 1524 @ 2.45M → 385 @ 3.95M, bimodal falls) under constant
  `ent_coef`, the same pathology entropy decay fixed in stage 2. Staged
  (commented) `ent_coef_end` suggestions added to the trex and
  brachiosaurus stage 1/2 TOMLs for their next runs.
- **Raptor hip-pitch/ankle actuator headroom raised to 1.5×kp** (breaking —
  plant change): the blanket 0.8×kp `forcerange` from the July 9 model
  hardening clipped 34–40 % of hip and 22–25 % of ankle torque during a
  moderate gait cycle and collapsed stage-2 locomotion twice (runs
  `20260709_185946` and `20260711_165924`, bitwise-identical failures). At
  1.5×kp the same excitation measures 0 % clipping and zero pelvis-z
  divergence from an unbounded plant; `test_actuator_bounds.py` now pins
  the per-group ratios and asserts the gait regime stays *unclipped*.
  Velociraptor policies trained on the 0.8×kp plant will not transfer.
- **Velociraptor stage-2 config hardened against the fragile-fast-gait
  collapse**: `ent_coef_end = 0.001` (entropy decay — action std grew
  1.18→1.49 in both collapses), `forward_vel_max` 3.0 → 2.5 (saturate the
  speed incentive just past the 2.0 m/s gate), `fall_penalty` −50 → −150
  (late-fall hedge; the corrected break-even math is in
  `docs/investigations/STAGE2_RECOMMENDATIONS.md` §2.1).
- **Docs reorganized**: dated run analyses and reward investigations moved
  to `docs/investigations/` (TRAINING_REVIEW, TRAINING_REVIEW_JAX_STAGE1,
  REWARD_DISCREPANCY_INVESTIGATION, REWARD_SCALE_REDESIGN,
  STAGE2_INVESTIGATION, STAGE2_RECOMMENDATIONS); `docs/README.md` now maps
  the four doc categories (living reference / plans / investigations /
  reviews) and their lifecycles.
- **Python 3.11+ required; Python 3.13 supported** (breaking): dropped
  Python 3.10 support (`requires-python >= 3.11`), added 3.13 to the
  trove classifiers and the CI test matrix (3.11 + 3.13), and bumped the
  Docker base image to `python:3.13-slim`.  The `tomli` conditional
  dependency and import fallbacks are gone — stdlib `tomllib` is always
  available on 3.11+.  Ruff now targets py311; mypy targets 3.12 because
  numpy ≥ 2.3 stubs use PEP 695 syntax that mypy rejects for older targets.
- **Homepage light mode actually renders light**: the landing page palette
  (`index.module.css`) gained a `[data-theme='light']` variant — until now
  every section was hardcoded dark, so toggling the theme changed almost
  nothing.  Accent colors are darkened for WCAG AA contrast on light
  surfaces, the curriculum/feature icon colors follow the theme, and the
  forced-dark homepage navbar override (plus its logo-swap hack) in
  `custom.css` is removed.  The footer now follows the theme in both modes
  (Infima's `.footer--dark` element-level variables had silently overridden
  the intended colors, so it rendered `#303846` even in dark mode).  The
  code terminal mock-up intentionally stays dark in both themes.

### Fixed
- **Trex/brachio stage TOMLs: restored `[ppo.policy_kwargs]` headers lost
  in the entropy-decay staging** (PR #436): the dropped headers put
  `net_arch` into the top-level `[ppo]` table, crashing `PPO.__init__()`
  at stage start (trex run `20260712_185931`). New config regression
  tests now pin the structure for every species/stage: `net_arch` only
  under `policy_kwargs`, `[ppo]` keys validated against the PPO
  constructor signature plus harness schedule keys, and `[env]` keys
  validated against each species' env constructor signature.
- **JAX pipeline correctness** (July 2026 review, PR #426): curriculum gate
  now compares episode-level return against the TOML threshold (it compared
  per-step reward and never advanced past stage 1); observation-normalization
  stats carry across curriculum stages; `MJXDinoEnv` no longer mutates the
  live species registry (stage config leaked across stages); trex TOML
  `approach_weight` was shadowed by a legacy registry key; `vf_clip_range`
  now reaches the fused PPO update; CPU evaluation observes the real target
  body (was hardcoded to the world origin) and its reward includes the same
  approach/proximity/success/fall components as training; eval records
  per-episode success and gates on TOML `min_avg_forward_vel` /
  `min_success_rate`
- **SB3 pipeline**: eval-collapse callback re-read `evaluations.npz` every
  training step (a GCS network read on Vertex); `diagnostics.npz` was fully
  rewritten up to 3× per rollout-end (O(n²) I/O, severe for SAC); final
  model now saved before the post-training eval; `time_to_target` no longer
  averages a `-1` sentinel into the mean
- **Sweeps / Google Cloud**: sweep resume no longer submits duplicate Vertex
  HPT jobs after a swallowed timeout; Ray Tune stage-2/3 curriculum gates no
  longer fail unconditionally (metric-alias mismatch; NaN treated as
  missing); `--wandb` with a missing API key no longer kills headless
  training and relaunches resume the same W&B run; TensorBoard events sync
  to GCS periodically (spot-VM preemption no longer loses a stage's logs);
  `_wait_for_job` honors `--stage-timeout` during persistent API failures;
  effective per-trial seed and run identity recorded in `metrics.json`
- `ray_tune_sweep.ipynb` shipped with a `SyntaxError` in the
  save-search-space cell
- Broken sweep commands in `vertex-ai.md`; stale `configs/sweep_*.json`
  paths; nonexistent `wandb_project`/`WandbHook` in `jax.md`

### Changed
- **Brachiosaurus model physics (breaking for trained policies)**: leg
  springs now reference stance angles with stronger, force-bounded servos —
  the model can now statically stand at torso z≈1.13 (previously collapsed
  to z≈0.70, below the z=1.0 alive floor, even when commanded straight).
  Pre-existing brachiosaurus checkpoints will not transfer.
- All three models use `integrator="implicitfast"` and bounded `forcerange`
  on position actuators (raptor/trex bounds sized to leave settle behavior
  unchanged)
- Headless JAX runs with `--checkpoint-dir` now produce a per-update
  training CSV, rotating + final checkpoints, and a best-episode-return
  snapshot; `--curriculum` forwards the checkpoint dir per stage; the
  single-stage CLI passes previously-dropped TOML `[jax]` keys
  (`minibatch_size`, `warmup_*`, `ramp_*`, `num_envs`, `vf_coef`)
- New `gcp` extra (`google-cloud-aiplatform`, `google-cloud-storage`),
  installed in the Docker image
- Code reviews consolidated: `docs/KNOWN_ISSUES.md` is the living list of
  open findings; dated reviews archived under `docs/reviews/`

## [0.3.0] - 2026-07-09 — Codebase Consolidation & Training Results

### Added
- Velociraptor SAC training results — all 3 stages passed (90.0% strike success, 22M steps, 22:59:18)
- Brachiosaurus PPO training results — Stages 1-2 passed, Stage 3 (food_reach) at 16.7% success (target: 50%)
- Updated training review with Brachiosaurus Stage 2/3 analysis and Velociraptor SAC comparison
- `environments/shared/train_base.py` — shared training logic with `SpeciesConfig` dataclass (~1,100 lines), reducing each species' `train_sb3.py` from ~950 lines to ~44 lines
- `environments/shared/test_env_base.py` — shared test utilities for environment validation (~214 lines)
- `environments/shared/constants.py` — centralized simulation-wide constants (sensor layout, VecNormalize defaults, physics defaults)
- `environments/shared/tests/reward_test_helpers.py` — reusable reward assertion functions for cross-species test consistency
- Expanded T-Rex reward tests (nosedive, height, heading, spin, drift, backward velocity)
- Expanded Brachiosaurus reward tests (gait instability, speed penalty, food reach threshold)
- Consolidated training notebooks into single parameterized `notebooks/training.ipynb`
- Algorithm-specific training diagnostics persisted to `diagnostics.npz` for **both PPO and SAC** (previously only on TensorBoard): captured SB3 `train/*` metrics (PPO `clip_fraction`/`approx_kl`/`explained_variance`/`entropy_loss`, SAC `critic_loss`/`actor_loss`/`ent_coef`) under `algo_*` keys
- `diagnostics/action_saturation` and `diagnostics/action_abs_max` — fraction of action components pinned at the control limit and the peak magnitude, computed for both PPO and SAC (SAC previously had no action logging at all)
- `diagnostics/grad_norm` — PPO post-update (clipped) gradient norm, surfacing when the clip ceiling binds or the policy stalls
- `explained_variance` in the JAX PPO loss info dict, per-update CSV log, and console output (value-function fit diagnostic)

### Changed
- Species training scripts (`train_sb3.py`) are now thin wrappers around shared `train_base.py`
- `DiagnosticsCallback` plateau detection now averages every episode that completed during a rollout (accumulated per-step) instead of only episodes that happened to end on the rollout's final step — the old sampling was biased and sparse
- JAX PPO clamps the Gaussian policy's log-std to `[-5, 2]` (applied identically in `sample_action` and `ppo_loss`, preserving the importance ratio) so a collapsing/exploding policy can no longer drive the action std to 0/inf and NaN the log-prob and entropy terms
- Species test scripts use shared `test_env_base.py` utilities
- `BaseDinoEnv` now provides concrete helper methods for common reward computations, gait symmetry, and termination checks
- Raptor, T-Rex, and Brachio reward tests refactored to use shared helpers
- `CookieResetButton` now uses shared `resetConsentStatus()` instead of duplicating localStorage logic

### Removed
- ~3,000+ lines of duplicated code across training scripts, test utilities, and environment methods
- Unused `getConsentStatus()` export from CookieConsent component

## [0.2.0] - 2026-02-09

### Added
- TOML-based configuration for curriculum stage reward weights and hyperparameters (`configs/` directory)
- Config loader utility (`environments/shared/config.py`) with `load_stage_config()` and `load_all_stages()`
- Gymnasium namespace registration (`MesozoicLabs/Raptor-v0`, `MesozoicLabs/Brachio-v0`, `MesozoicLabs/TRex-v0`)
- Pre-commit hooks with Ruff (format + lint), mypy, and standard checks
- `pytest-cov` for test coverage reporting with 70% threshold
- `CONTRIBUTING.md` with development workflow guidelines
- `CHANGELOG.md` (this file)
- GitHub issue templates (bug report, feature request, new species proposal)
- Reward function unit tests and curriculum stage transition tests
- Package metadata in `pyproject.toml` (authors, license, classifiers, URLs)
- `mypy` and `ruff` configuration in `pyproject.toml`
- `CurriculumManager` class for automated multi-stage training (`environments/shared/curriculum.py`)
- `CurriculumCallback` SB3 callback that monitors evaluation and stops training when advancement thresholds are met
- `thresholds_from_configs()` helper to extract curriculum thresholds from TOML configs
- `[curriculum]` section in all TOML stage configs with `timesteps`, `min_avg_reward`, `min_avg_episode_length`, and `required_consecutive` fields
- `curriculum` subcommand in all three species' training scripts for automated end-to-end 3-stage training
- `LocomotionMetrics` class with gait symmetry, cost of transport, stride frequency, and time-to-target (`environments/shared/metrics.py`)
- `WandbCallback` for SB3 with per-component reward logging and config snapshots (`environments/shared/wandb_integration.py`)
- `WandbCallback` video recording of evaluation episodes (`video_env`, `video_freq` parameters)
- `wandb` added to `[train]` optional dependencies

### Changed
- Training scripts now load stage configs from TOML files instead of hardcoded dictionaries
- Config loader now parses `[curriculum]` section from TOML files into `curriculum_kwargs`
- Bumped version from 0.1.0 to 0.2.0
- Dev dependencies expanded: `pytest-cov`, `mypy`, `ruff`, `pre-commit`
- All `print()` calls in training scripts replaced with `logging` module
- Gymnasium environments auto-register on `import environments` (no longer requires `register_all()`)

## [0.1.0] - 2025-01-01

### Added
- Initial release
- Velociraptor bipedal locomotion environment with sickle claw strike
- Brachiosaurus quadrupedal locomotion environment with food reaching
- T-Rex bipedal locomotion environment with jaw bite
- 3-stage curriculum learning (balance, locomotion, behavior)
- PPO and SAC training via Stable-Baselines3
- MuJoCo MJCF models for all three species
- Colab-ready training notebooks
- Docusaurus documentation site
- GitHub Actions CI with per-species test jobs
