# The model-based trot on the adopted 128 mm capsule model: measurements (simulation)

27-28 Nov 2026. This is the first of D45's no-spend checks (`DECISION_LOG.md`, D45). The frozen trot runs its judge's
conditions and its sweep on the adopted design's model, with pushes both ways and delay points around 16-19 ms.

**Reported, not judged.** No pre-registration covers the trot on this model. `PREREG_WALK60.md`'s thresholds are
shown beside the measurements and are labelled as written for the 188 mm roll-ankle model. The data covers one
controller and two seed sets per condition.

**Kinds of evidence.** Every number is **simulation** of a model whose masses and contact are estimates. The drive
ratings and speed allowances are **desk** figures; their sources and dates are under Drive heat and Peak drive
speeds. Nothing is built, so nothing is **measured**.

- **Controller:** `walk/walk60.py` `TrotController` with `WalkParams()` defaults. The file is unchanged since its
  judged run (sha256 `5225c64a…`, at 0382ce56 and now). It uses leg odometry, not the simulator's body velocity
  (`walk60.py:10-12`).
  - **It reads the foot loads, which the robot's own sensors do not give.** The foot-load observation
    (`walk60.py:224`) picks which stance legs feed the leg odometry (`:239`; the threshold `stance_load` is at `:67`).
    Its other use, early touchdown (`:297`), is off by default (`:84`).
  - The core build has no foot load cells (`sim/build_model60.py:12`; `DECISION_LOG.md`, D46). So every result
    here assumes a foot-load signal the robot will not have, unless foot sensors are chosen after the D46 run.
  - The own-sensors option in `rl/trike_env.py` (`obs_foot_loads=False`, added for the D46 run alongside this check)
    drops that slot. The trot reads the slot by name, so as written it cannot run on that observation (from the
    code; not run).
- **Model:** `sim/generated/trike60_C_closed_capsule_fore128.xml`, the adopted design (`DECISION_LOG.md`,
  D25 and D41): 10 X4 and 2 X6 drives on 12S, the capsule sole, the 128 mm upper foreleg,
  12.751 kg.
- **Comparison models:**
  - the 188 mm capsule model `trike60_C_closed_capsule.xml` (12.694 kg; J1-J4, R1-R5 and four pushes);
  - the 188 mm roll-ankle model `trike60_C_closed_ankleroll.xml` (12.694 kg). The pre-registration was written for
    it, and the trot passed on it (`REPORT_WALK60.md:11-18`).

  The trot was developed on the roll-ankle foot and runs on the capsule as a drop-in, untuned
  (`sim/REPORT_FEET60.md:82`).
- **Tools:** `walk/fore128_walk60.py` and `walk/fore128_slow_gait.py`. This package changes none of `walk60.py`,
  `judge_walk60.py`, `sweep_walk60.py` or `rl/`. The windows ran with the D46 own-sensors change to
  `rl/trike_env.py` in the working tree, not committed at the time (see How these were run). The added parameters (model, J1 at extra delays, delay points, fore-aft
  pushes, first seed, peak drive speed) default to the existing tools' behaviour. The tests reproduce a stored
  judged episode and a stored sweep episode exactly (see Tests).
- **Output:** `results/walk60_fore128_2026-11-27/`:
  - `RESULTS.md` (every table in full) and `summary.json`;
  - the raw judged and sweep files;
  - `slow_gait/` (10-20 s windows);
  - `capsule188_check/` (the 188 mm capsule model's J1-J4 and R1-R5).

## Answer first

- **The trot does not fall in any of `PREREG_WALK60.md`'s conditions, at any delay, or after any fore-aft push.**
  - Judged conditions: no falls in 340 episodes (J1-J4, R1-R5 and J1 at 16-60 ms, seeds 101-110 and 201-210).
  - Delay: no falls in the sweep at any delay from 0 to 60 ms.
  - Fore-aft pushes up to 1.5 m/s either way: no falls in 380 episodes.
- **Sideways pushes knock it over** at:
  - 1.5 m/s both ways: 10 of 10 on both seed sets;
  - 1.0 m/s to the right: 5 of 10 on both sets;
  - 1.25 m/s, on seeds 300000 only: 3 of 10 to the right, 1 of 10 to the left.

  On the 188 mm roll-ankle model the sweep had no falls up to ±1.25 m/s and 8 of 8 at ±1.5 m/s
  (`REPORT_WALK60.md:131`; 8 seeds).
- **Against `PREREG_WALK60.md`'s thresholds, the slippery floor meets all four and the grippy floor does not.** The
  thresholds were written for the 188 mm roll-ankle model, where all four judged conditions passed.
  - J2 and J4 meet W1-W4 on both seed sets.
  - J3 misses W2 on both sets: mean 0.221 and 0.217 m/s, slowest episode 0.193 and 0.190 m/s.
  - J3 misses W3 on both sets: worst episode 1.10 and 1.13 of rating, a hind knee.
  - J1 misses W3 on seeds 101-110 (1.04, left hind knee) and meets all four on 201-210.
  - W1 and W4 are met everywhere. The worst foot slip ratio is 0.11-0.15.
  - The 188 mm capsule model misses W2 and W3 in J3 on seeds 101-110 only (0.241 m/s, slowest 0.193; 1.04).
- **Each of those misses is one state of the trot, called here the slow gait.**
  - The rule, in `fore128_slow_gait.py`, has two parts, both measured over 10-20 s of the episode:
    - the forward speed is under 0.225 m/s (90 % of the 0.25 m/s command);
    - the larger hind knee's RMS torque is at or over its continuous rating (6.75 N m, desk; source under Drive heat).
  - The rule was chosen after the data was seen. It is not pre-registered.
  - What the slow gait looks like:
    - the trot covers 0.17-0.21 m/s at the nominal mass and payload (0.168-0.225 m/s over all 479 such episodes);
    - the chassis rides 0.6-1.5 deg nose up, mostly 1.2-1.4 deg;
    - the hind knee runs at 1.00-1.41 of rating (1.01-1.24 at the nominal mass and payload).
  - At the 0.25 m/s command, every judged episode whose 2-20 s speed is under 0.20 m/s, or whose worst drive is at
    or over 1.0, is in it over 10-20 s. That holds at every delay. So is every J1-J4 episode under 0.225 m/s.
- **Where it appears.** Counts are episodes of 10, given as seed set 1 / seed set 2; the tables below have the
  detail.
  - Pushes back:
    - 0.3-1.25 m/s: 7-10 of 10 on both sets (10 / 10, except 7 / 8 at 0.75 m/s and 10 / 9 at 1.25 m/s);
    - 1.5 m/s: 6 / 6;
    - 0.2 m/s: 5 / 3;
    - 0.1 m/s: 0 / 2.
  - Pushes to the right:
    - 0.2-0.5 m/s: 8-10 of 10;
    - 0.1 m/s: 3 / 5;
    - 0.75 m/s: 4 / 6;
    - 1.0-1.25 m/s: 1-5 of the 5-10 that stayed up.
  - Pushes to the left:
    - 0.1-0.75 m/s: 0-2 of 10;
    - 1.0 m/s: 10 / 10;
    - 1.25 m/s: 4 of 10 / 6 of 9.
  - Pushes forward:
    - 0.1-0.75 m/s: 0-1 of 10;
    - 1.0-1.5 m/s: 1-4 of 10.
  - No push: 0 of 20 on the nominal model at 10 ms, and 0 of 20 in R5.
  - Randomised:
    - J1, grippy floor, 10 ms: 2 / 1;
    - J3, stiffer contact, 10 ms: 7 / 6;
    - slippery floor (J2 and J4 at 10 ms, R2 at 0 ms): 0 of 60.
  - Delay, J1 randomised:
    - 16 ms: 6 / 4;
    - 17 ms: 4 / 3;
    - 18 ms: 0 / 3;
    - 19 and 20 ms: 0 / 0;
    - 30 ms: 0 / 1;
    - 40 ms: 1 / 3;
    - 60 ms: 0 / 1.
  - Delay, the sweep's nominal model, no push:
    - 15 ms: 1 / 4;
    - 16 ms: 5 / 6;
    - 17 ms: 3 / 8;
    - 18 ms: 5 / 6;
    - 19 ms: 0 / 1;
    - 20 ms: 1 / 3;
    - 30 ms: 4 / 1;
    - 40 ms: 4 / 3;
    - 60 ms: 1 / 0.
  - At 0 ms (R1) none of 20 meet the rule, but that rests on the 1.0 cut-off. R1 seed 203 (0.219 m/s, knee 0.96,
    1.0 deg nose up) looks like the slow gait with the knee just under rating. So does J1 seed 203 at 10 ms
    (0.214 m/s, knee 0.98, 1.0 deg nose up).
  - kp x0.8, no push: 10 / 10 (0.176-0.182 m/s, knee 1.03-1.08). The sweep's kp break, 24 % below nominal, is this
    state.
  - Payload on the chassis, no push:
    - 0.5 kg: 3 / 4;
    - 1 kg: 4 / 6;
    - 2 kg: 9 / 9;
    - 3 kg: 10 / 10.
  - Mass scale, no push:
    - x1.2: 0 / 0;
    - x1.4: 2 / 4;
    - x1.6: 4 / 5.
  - The same nose-up slow state also appears with the knee under rating, so outside the rule:
    - mass x0.8-0.9, seeds 300000: 3 episodes at 0.200-0.204 m/s, 1.25-1.29 deg nose up, knee 0.90-0.97;
    - force limit x0.7: 1 episode;
    - floor friction 0.8: 1 episode.

    At force limit x0.4, five episodes slow to 0.157-0.218 m/s with the knee at 0.69-0.83.
- **The slow gait goes with the capsule sole.** This is an association, from one controller on three models.
  - With the same pushes and seeds, the 188 mm capsule model enters it at similar rates:
    - back 0.3 m/s: 6 / 7, against 10 / 10 on the 128 mm model;
    - back 1.0 m/s: 10 / 10 on both;
    - right 0.3 m/s: 10 / 10 on both;
    - left 1.0 m/s: 9 / 9, against 10 / 10.
  - The 188 mm roll-ankle model never entered it: 0 of 80 pushed episodes (the same four pushes on both sets),
    and 0 of 20 unpushed.
  - Randomised, it was more frequent on the 128 mm model than on the 188 mm capsule model: J3 7 / 6 against 2 / 0,
    and J1 2 / 1 against 0 / 0.
- **Drive heat.** The measure is RMS torque / continuous rating at the joint (`RESULTS.md:6`).
  - The ratings are desk figures, from the maker's datasheets as read on 7 Nov 2026 (`sim/data/drives.json`). The
    tools read them from `rl/trike_env.py` (`cont_torque_*`), which takes them from the mass ledger
    (`sim/generated/mass_ledger_trike60_C_closed_capsule_fore128.json`, `joint_thermal_eq_nm`).
    - X4 roll and pitch: 7.5 N m, the maker's temperature-rise table value (90 min, motor reaching 100 C, open
      air; `thermal_eq_nm`).
    - X4 hind knee: 6.75 N m = 7.5 x 0.90. The 0.90 is the transmission efficiency the model applies at the
      flexion joints' 1:1 linkages (`transmission_efficiency`, a desk figure; `sim/build_model60.py:412`).
    - X6 front elbow: 10.8 N m = 12.0 x 0.90. The 12.0 N m is the X6-50-L table at 0 rpm, used as a stand-in
      because the V4 X6-60 has no published table (`thermal_eq_note`).
  - In the slow gait the X4 hind knee runs over its rating over 10-20 s: 1.01-1.09 after pushes and 1.01-1.24 in
    the judged conditions.
  - Outside the slow gait, the larger hind knee runs at 0.54-0.85 in the judged conditions, 0.61-0.88 after pushes
    and 0.61-0.95 at the delay points.
  - The sweep's own 2-20 s heat figures dilute the slow gait with the 6 s before the push. In the push rows where
    the slow gait appears, the sweep's 2-20 s worst episode is 0.87-0.97 (0.69-0.97 over all push rows), and none
    reaches 1.0.
  - A small payload brings the slow gait on with no push. At 0.5-1 kg the worst episode is over rating on both seed
    sets (2-20 s: 1.02 and 1.06 at 0.5 kg; 1.08 and 1.10 at 1 kg).
  - At 2-3 kg, 8-10 of 10 episodes are over rating. At mass x1.6, 8 of 10 are (worst 1.32 and 1.36). At
    x1.4-1.6 the knee goes over even in episodes that keep their speed.
  - The 188 mm roll-ankle model's worst joint stayed at 0.74-0.88 of rating (as a mean) at the same points, 8 seeds
    (`results/sweep_walk60/summary_walk60.md:100-101`, `:114-115`):
    - mass x1.4: 0.78;
    - mass x1.6: 0.88;
    - payload 2 kg: 0.74;
    - payload 3 kg: 0.81;
    - worst episode at these points: 0.80-0.91 (`results/sweep_walk60/sweep_walk60.json.gz`, each point's
      `summary` `heat_worst_episode`; the md gives only the worst joint's mean).
  - F12's budget estimate put the hind knees at 1.00 of rating at 12.69 kg (`DECISION_LOG.md`, F12).
  - These are load ratios, not winding temperatures. No thermal model was run here. The heat rules now assume a
    35 C room (`DECISION_LOG.md`, D48).
- **Delay.** There are no falls and no sweep break from 0 to 60 ms, at 0.25 or 0.15 m/s, on either seed set.
  - 16-18 ms is where the slow gait is most frequent, in J1 and in the sweep.
  - At 19-20 ms J1 keeps its mean speed (0.233-0.244 m/s).
  - At 19, 30 and 40 ms, 19 J1 episodes slow to 0.209-0.225 m/s with the chassis 0.66-0.89 deg nose down and the
    knee at 0.64-0.77. The 23 other episodes at 30-40 ms that are not slow are also nose down (0.35-0.71 deg), at
    0.227-0.243 m/s. So this is the whole gait slowing with delay, split by the 0.225 cut-off, not a separate state.
- **Peak drive speeds.** Each peak is the largest joint speed from 2 s over the episodes that did not fall, from the
  1 ms log; the 99th percentiles are taken the same way (`RESULTS.md:7`). The budget's allowances are desk figures.
  - X4: 83 rpm, its rated-speed cap above 24 V. The source is a reseller's text quoting the maker (web search, 18 Nov
    2026; `DECISION_LOG.md` (E1), `hardware/triceratops_kickoff/PARTS_PREMIUM.md:45`).
  - X6 front elbow: 108 rpm, 0.70 x its 176 rpm no-load speed at 48 V, scaled to 42 V (`budget/size_budget.py:60`).
    The 176 rpm is from the maker's datasheet, read 7 Nov 2026 (`sim/data/drives.json`).
  - Peaks stay under the allowances in:
    - J1-J4 and R1-R5;
    - J1 up to 30 ms of delay, and the sweep up to 40 ms;
    - every friction, mass, payload, force-limit and kp point;
    - every backward push up to 1.5 m/s and every forward push up to 1.0 m/s;
    - sideways pushes up to 0.4 m/s to the left and 0.5 m/s to the right.
  - Peaks go over in each case below (rpm; seed set 1 / seed set 2 where they differ):
    - J1 at 40 ms: pitch 96 / 92.
    - J1 at 60 ms: pitch 166 / 120; front elbow 112 on seeds 101-110.
    - The sweep at 60 ms: pitch 113-125, at both 0.25 and 0.15 m/s.
    - Forward push of 1.25 m/s: pitch 89 / 98.
    - Forward push of 1.5 m/s: pitch 109 / 112; front elbow 112 / 112.
    - Left push of 0.5 m/s: pitch 88 / 84.
    - Left push of 0.75 m/s: pitch 107 / 107.
    - Left push of 1.0 m/s: pitch 122-124; front elbow 119; hind knee 95.
    - Left push of 1.25 m/s: pitch 134-138; front elbow 231-234; hind knee 122-123.
    - Right push of 0.75 m/s: hind knee 86 / 110; front elbow 108.2 on seeds 300000.
    - Right push of 1.0 m/s: pitch 144-148; front elbow 267-316; hind knee 151-156.
    - Right push of 1.25 m/s: pitch 130-132; front elbow 217-254; hind knee 236.
  - These peaks are brief: in the episodes that did not fall, every 99th percentile stays under its allowance. The
    largest are roll 31, pitch 67, front elbow 73 and hind knee 73 rpm.
  - In J1-J4 the pitch drives peak at 53-62 rpm on the 128 mm model, against 39-43 rpm on the 188 mm capsule model.
- **Heading and drift.**
  - Worst episode in J1-J4: 5.5-17.9 deg of heading change and 0.16-0.31 m of sideways drift over 18 s. The heading
    is 5.5-7.3 deg everywhere except 17.9 deg in J3, seeds 201-210.
  - Elsewhere the worst are 30.3 deg (J1 at 30 ms, seeds 201-210) and 21.4 deg (sideways push of 1.25 m/s to the
    right), then 20.5 and 20.2 deg at the sweep's 40 ms (seeds 200000) and 20 ms (seeds 300000) delay points.
  - With noise and randomisation off (R5), the worst is 1.1-1.2 deg and 0.01 m. So, as on the 188 mm model
    (5.6-5.9 deg, 0.21-0.25 m; `REPORT_WALK60.md:47`), the drift follows the simulated gyro noise and the heading
    estimate's random walk (`rl/trike_env.py:116`, `:123`), which the controller cannot see.
- **What it means (simulation, this controller).**
  - On the adopted model, on flat ground, the frozen trot stays up under every condition and delay tried, and
    after every fore-aft push.
  - It falls after sideways pushes of 1.0 m/s to the right and of 1.25-1.5 m/s.
  - Small disturbances can put it into a slower gait that holds the X4 hind knees over their continuous rating to
    the end of the 20 s episode. Such disturbances include a push of 0.2-0.5 m/s back or to the right, a 0.5 kg
    payload, 16-18 ms of delay, or a grippy floor with stiffer contact.
  - The trot was not developed on this foot. This is a measurement of a frozen controller, not a verdict on the
    design.

## How these were run

**Judged conditions.**
- Conditions: `PREREG_WALK60.md:30-43` (J1-J4, R1-R5) plus J1 at 16, 17, 18, 19, 20, 30, 40 and 60 ms.
- Randomisation and noise are on, as in the judged run.
- Seeds: 101-110 and the rerun set 201-210 (`PREREG_WALK60.md:47-48`).
- 10 episodes per condition, R1-R5 included (the pre-registration ran those on 5).

**Sweep.**
- The sweep's own points, physics changes, push, measures and break rule (`rl/robustness/sweep.py`; terrain not
  run, flat ground per `DECISION_LOG.md` D28).
- Nominal model, observation noise on, no randomisation.
- 10 episodes per point (the plan had 8, `PLAN_SWEEP_WALK60.md:24`). Seeds 200000-200009 and 300000-300009.
- Delay points: 0, 2, 5, 10, 15, 16, 17, 18, 19, 20, 30, 40 and 60 ms.

**Pushes.**
- The sweep's push: 0.1 s from 8 s, impulse = whole-body mass × free-body dv.
- Sideways is the sweep's own axis (+ left).
- Fore-aft goes along the chassis's forward axis at onset (+ forward). The sweep's recovery count is not defined for
  fore-aft pushes.

**10-20 s windows** (`fore128_slow_gait.py`).
- They measure the same episodes, checked seed by seed: the fall and the 2-20 s speed against the sweep; the
  whole-episode and 2-20 s speeds against the judged run.
- 1,740 episodes were windowed. The 1,564 that have a stored episode match it:
  - 128 mm model: all 120 sweep rows and all 30 judged groups that carry windows (1,500 episodes; `summary.json`);
  - 188 mm capsule model: its 4 judged groups, J1 and J3 on both seed sets (40 episodes;
    `capsule188_check/summary.json`);
  - 188 mm roll-ankle model: the 24 windows that have a stored episode in the 19 Nov sweep, on fall and 2-20 s
    speed (16 pushed and 8 unpushed; seeds 200000-200007, sideways pushes and no push).
- The other 176 have no stored episode to match:
  - the 188 mm capsule model's push-family windows (100 episodes, 20 of them unpushed; no trot sweep was stored on
    that model);
  - the roll-ankle windows outside the 19 Nov sweep's seeds 200000-200007 or its sideways axis (76, 12 of them
    unpushed).
- The 1,240 episodes windowed in a first run, the 176 among them, are identical in the rerun (the first run's files
  are not in the tree). Both runs used the same `rl/trike_env.py`, so this shows that the windows reproduce, not
  that they match a stored run.
- **Provenance of the windows.** They ran with `rl/trike_env.py` sha256 `766864a9…`: the working-tree file of
  27-28 Nov 2026, which carried the D46 own-sensors change, not committed at the time (`obs_foot_loads`, default
  True, so the observation is as before).
  - The judged and sweep runs used the committed file, `0f3a91e5…`, the same at aa150f89 and db43e595. Each sweep
    file records it (`env.trike_env_sha256`); the judged files, from the same batch just before the sweeps, record
    only the commit.
  - The judged and sweep runs used `walk/fore128_walk60.py` at sha256 `7ea9dcff…`, as each sweep file records
    (`driver.sha256`); the judged files predate that field. After the runs only its summary and Markdown functions
    changed; the committed file is `9a2f75e9…`. `walk/tests/test_fore128.py` reproduces a stored judged episode and
    a stored sweep episode with the committed file (15 passed, 28 Nov 2026).
  - The 1,564 checked episodes match their stored ones (above), so for them no number depends on the difference.
    For the other 176 this rests on the code, not on a run: with `obs_foot_loads` at its default the observation
    is built from the same values in the same order (the working-tree diff of `rl/trike_env.py` against
    db43e595).
- Before every push (2-8 s) the trot runs at 0.250-0.253 m/s with the knee at 0.62-0.69.

**Commands** (from the repository root; `W` = `hardware/triceratops_build/walk`, `R` = `$W/results/walk60_fore128_2026-11-27`):
- `python $W/fore128_walk60.py judged --seeds 101-110 --extra-delays 16 17 18 19 20 30 40 60 --workers 4 --out R`
  (and seeds 201-210; 7.0 and 6.4 min; commit aa150f89 with uncommitted new tools).
- `python $W/fore128_walk60.py sweep --seed0 200000 --seeds 10 --delays 0 2 5 10 15 16 17 18 19 20 30 40 60 --sweeps
  delay friction mass drive push --push-axis sideways --workers 4 --out R/sideways_200000 --date 2026-11-27`
  (21.2 min; seeds 300000: 14.9 min).
- `python $W/fore128_walk60.py sweep --seed0 200000 --seeds 10 --sweeps push --push-axis fore_aft --workers 4 --out
  R/fore_aft_200000 --date 2026-11-27` (5.4 min; seeds 300000: 3.0 min).
- `python $W/fore128_walk60.py judged --model trike60_C_closed_capsule.xml --seeds 101-110 --workers 4 --out
  R/capsule188_check` (and 201-210).
- `fore128_slow_gait.py` with `--judged`, `--dv all` or `--dv 0 --point '{...}'`, per file (60 files, 1,740
  episodes, 56.6 min; commit db43e595 with this package uncommitted; the same `fore128_walk60.py` and
  `fore128_slow_gait.py` for all). Each file records its command line and file hashes.
- `python $W/fore128_walk60.py summarise --raw R --out R --slow-gait R/slow_gait`, and the same for
  `R/capsule188_check`.

## PREREG_WALK60's judged conditions on three models

**How to read Tables 1-5.**
- Cells are seed set 1 / seed set 2.
- "Slow gait" counts episodes by the rule above. "(slow a / b)" adds the episodes under 0.225 m/s whatever the knee
  load, where that differs.
- Worst drive is the RMS / continuous rating of the worst joint in the worst episode, from 2 s (the
  pre-registration's W3 measure).
- Peak speeds are in rpm, from 2 s over the episodes that did not fall; **bold** is over the allowance.
- Sources: `RESULTS.md:14-32`, `:58-76` and `capsule188_check/RESULTS.md`; for the roll-ankle model,
  `results/walk60_2026-11-19/results.json`.

| Condition | Model | Falls | Speed, mean (slowest), m/s | Slow gait over 10-20 s, episodes | Worst drive RMS / rating, worst episode | Worst foot slip ratio | Heading change, worst, deg | Sideways drift, worst, m | Peak speed, pitch / front elbow / hind knee, rpm | PREREG_WALK60 thresholds (written for the 188 mm roll-ankle model) |
|---|---|---|---|---|---|---|---|---|---|---|
| J1 | 128 mm capsule (adopted) | 0 / 0 | 0.239 (0.201) / 0.244 (0.222) | 2 / 1 (slow 2 / 2) | 1.04 / 0.95 | 0.12 / 0.11 | 7.3 / 5.9 | 0.27 / 0.17 | 57 / 52 / 55; 57 / 51 / 55 | **not met: W3** / met |
| J1 | 188 mm capsule | 0 / 0 | 0.247 (0.244) / 0.246 (0.246) | 0 / 0 | 0.77 / 0.77 | 0.12 / 0.12 | 5.5 / 4.7 | 0.21 / 0.17 | 40 / 39 / 53; 41 / 39 / 55 | met / met |
| J1 | 188 mm roll-ankle (19 Nov, seeds 101-110 only) | 0 | 0.248 (0.247) | not windowed | 0.77 | 0.09 | 5.6 | 0.25 | not recorded | met (judged PASS) |
| J2 | 128 mm capsule (adopted) | 0 / 0 | 0.256 (0.253) / 0.257 (0.254) | 0 / 0 | 0.83 / 0.85 | 0.11 / 0.11 | 6.0 / 5.5 | 0.30 / 0.16 | 53 / 48 / 51; 55 / 49 / 50 | met / met |
| J2 | 188 mm capsule | 0 / 0 | 0.256 (0.252) / 0.257 (0.253) | not windowed | 0.80 / 0.83 | 0.12 / 0.11 | 5.7 / 5.5 | 0.28 / 0.15 | 39 / 37 / 52; 39 / 38 / 50 | met / met |
| J2 | 188 mm roll-ankle (19 Nov, seeds 101-110 only) | 0 | 0.252 (0.250) | not windowed | 0.78 | 0.16 | 5.8 | 0.23 | not recorded | met (judged PASS) |
| J3 | 128 mm capsule (adopted) | 0 / 0 | 0.221 (0.193) / 0.217 (0.190) | 7 / 6 (slow 8 / 7) | 1.10 / 1.13 | 0.13 / 0.15 | 6.0 / 17.9 | 0.30 / 0.31 | 56 / 52 / 55; 62 / 56 / 69 | **not met: W2, W3** / **not met: W2, W3** |
| J3 | 188 mm capsule | 0 / 0 | 0.241 (0.193) / 0.253 (0.252) | 2 / 0 | 1.04 / 0.91 | 0.11 / 0.11 | 20.7 / 4.8 | 0.23 / 0.17 | 40 / 46 / 57; 43 / 39 / 56 | **not met: W2, W3** / met |
| J3 | 188 mm roll-ankle (19 Nov, seeds 101-110 only) | 0 | 0.251 (0.249) | not windowed | 0.86 | 0.08 | 5.9 | 0.23 | not recorded | met (judged PASS) |
| J4 | 128 mm capsule (adopted) | 0 / 0 | 0.256 (0.252) / 0.257 (0.253) | 0 / 0 | 0.83 / 0.84 | 0.13 / 0.12 | 5.8 / 6.2 | 0.29 / 0.17 | 54 / 48 / 50; 55 / 48 / 50 | met / met |
| J4 | 188 mm capsule | 0 / 0 | 0.256 (0.251) / 0.257 (0.253) | not windowed | 0.79 / 0.81 | 0.13 / 0.12 | 5.5 / 5.6 | 0.27 / 0.15 | 39 / 38 / 49; 40 / 38 / 50 | met / met |
| J4 | 188 mm roll-ankle (19 Nov, seeds 101-110 only) | 0 | 0.252 (0.251) | not windowed | 0.77 | 0.16 | 5.6 | 0.21 | not recorded | met (judged PASS) |

Thresholds (`PREREG_WALK60.md:52-57`):
- W1: no fall.
- W2: mean at least 0.225 m/s, and no episode under 0.20 m/s.
- W3: every joint under 1.0 of its rating in every episode.
- W4: every foot's mean slip ratio at most 0.25.

The worst foot's slip ratio measured at the contacts is 0.05-0.10 in J1-J4 (`RESULTS.md:16-19`, `:60-63`).

**Reported cases, 128 mm model, 10 episodes per set.** No threshold applies to R3 and R4; for R1, R2 and R5 the
thresholds are for orientation only, and all four are met.

| Case | Falls | Speed, mean (slowest), m/s | Slow gait over 10-20 s | Worst drive RMS / rating, worst episode | Heading change, worst, deg | Sideways drift, worst, m | Peak speed, pitch / front elbow / hind knee, rpm |
|---|---|---|---|---|---|---|---|
| R1 J1 at 0 ms delay | 0 / 0 | 0.251 (0.248) / 0.250 (0.235) | 0 / 0 (slow 0 / 1) | 0.77 / 0.86 | 5.5 / 6.0 | 0.27 / 0.17 | 59 / 50 / 55; 58 / 50 / 56 |
| R2 J2 at 0 ms delay | 0 / 0 | 0.257 (0.255) / 0.257 (0.256) | 0 / 0 | 0.81 / 0.84 | 5.8 / 5.6 | 0.26 / 0.16 | 56 / 49 / 51; 55 / 49 / 50 |
| R3 J1 at the 0.15 m/s command | 0 / 0 | 0.139 (0.137) / 0.140 (0.138) | not windowed | 0.70 / 0.74 | 5.8 / 4.8 | 0.13 / 0.10 | 54 / 48 / 47; 51 / 48 / 48 |
| R4 J1 standing (0 m/s) | 0 / 0 | 0.000 / 0.000 | not windowed | 0.53 / 0.53 | 0.0 / 0.0 | 0.00 / 0.00 | 2 / 4 / 3; 2 / 4 / 3 |
| R5 nominal model, no randomisation or noise | 0 / 0 | 0.250 (0.250) / 0.250 (0.250) | 0 / 0 | 0.70 / 0.69 | 1.2 / 1.1 | 0.01 / 0.01 | 56 / 51 / 52; 56 / 51 / 52 |

## Delay

**Columns.**
- J1 columns: randomised, seeds 101-110 / 201-210. 0 ms is R1; 10 ms is J1 itself.
- Sweep columns: the nominal model at 0.25 m/s, no push, seeds 200000-9 / 300000-9 (`RESULTS.md:158`, `:280`).
- Sweep, 10 ms: the sweep's nominal point. These are the same episodes as the push families' 0 row, whose windows
  give its slow-gait cell (0 / 0). The same holds for the nominal friction row under Other sweep settings.
- Last column: the 188 mm roll-ankle sweep, 8 seeds, which had no 16-19 ms points. Speeds are from
  `results/sweep_walk60/summary_walk60.md:40-48`. Worst episodes are from `results/sweep_walk60/sweep_walk60.json.gz`
  (each point's `summary` `heat_worst_episode`), since the md gives only the worst joint's mean.

**Thresholds** (for orientation only; written for the 188 mm model).
- J1 misses W2 at 16, 17 and 40 ms on both sets, and at 18, 30 and 60 ms on seeds 201-210.
- J1 misses W3 at 16 and 17 ms on both sets, and at 18, 30, 40 and 60 ms on seeds 201-210.

**At 0.15 m/s** (`RESULTS.md:176`, `:298`): no falls and no break, down to 0.130 and 0.137 m/s at 60 ms.

| Delay, ms | J1: falls | J1: speed, mean (slowest), m/s | J1: slow gait, 10-20 s | J1: worst drive, worst episode | J1: heading, worst, deg | J1: peak speed, pitch / elbow / knee, rpm | Sweep: falls | Sweep: speed, mean (slowest), m/s | Sweep: slow gait, 10-20 s | Sweep: worst drive, worst episode; episodes at or over 1.0 | Sweep: peak speed, pitch / elbow / knee, rpm | 188 mm roll-ankle sweep: speed; worst drive, worst episode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 / 0 | 0.251 (0.248) / 0.250 (0.235) | 0 / 0 (slow 0 / 1) | 0.77 / 0.86 | 5.5 / 6.0 | 59 / 50 / 55; 58 / 50 / 56 | 0 / 0 | 0.253 (0.251) / 0.253 (0.251) | not windowed | 0.66 / 0.67; 0 / 0 | 58 / 49 / 52; 56 / 50 / 52 | 0.248; 0.68 |
| 2 | not run | | | | | | 0 / 0 | 0.253 (0.253) / 0.253 (0.253) | not windowed | 0.67 / 0.68; 0 / 0 | 57 / 49 / 52; 56 / 49 / 53 | 0.248; 0.67 |
| 5 | not run | | | | | | 0 / 0 | 0.251 (0.229) / 0.254 (0.253) | not windowed | 0.88 / 0.75; 0 / 0 | 55 / 50 / 52; 55 / 49 / 52 | 0.246; 0.67 |
| 10 | 0 / 0 | 0.239 (0.201) / 0.244 (0.222) | 2 / 1 (slow 2 / 2) | 1.04 / 0.95 | 7.3 / 5.9 | 57 / 52 / 55; 57 / 51 / 55 | 0 / 0 | 0.251 (0.250) / 0.251 (0.250) | 0 / 0 | 0.70 / 0.69; 0 / 0 | 57 / 51 / 53; 57 / 51 / 53 | 0.246; 0.69 |
| 15 | not run | | | | | | 0 / 0 | 0.243 (0.207) / 0.232 (0.189) | 1 / 4 (slow 2 / 5) | 1.00 / 1.07; 0 / 2 | 55 / 51 / 52; 55 / 56 / 52 | 0.245; 0.69 |
| 16 | 0 / 0 | 0.225 (0.194) / 0.226 (0.191) | 6 / 4 (slow 6 / 5) | 1.09 / 1.10 | 7.6 / 8.9 | 57 / 51 / 53; 55 / 52 / 55 | 0 / 0 | 0.223 (0.187) / 0.216 (0.187) | 5 / 6 (slow 5 / 7) | 1.08 / 1.09; 4 / 6 | 56 / 53 / 51; 55 / 53 / 52 | not in that sweep |
| 17 | 0 / 0 | 0.231 (0.195) / 0.233 (0.185) | 4 / 3 | 1.11 / 1.15 | 7.1 / 10.7 | 57 / 51 / 52; 56 / 53 / 55 | 0 / 0 | 0.235 (0.185) / 0.202 (0.183) | 3 / 8 | 1.10 / 1.10; 2 / 7 | 55 / 53 / 51; 55 / 55 / 52 | not in that sweep |
| 18 | 0 / 0 | 0.246 (0.242) / 0.233 (0.182) | 0 / 3 | 0.79 / 1.16 | 6.2 / 5.0 | 57 / 50 / 53; 56 / 52 / 55 | 0 / 0 | 0.219 (0.185) / 0.215 (0.184) | 5 / 6 (slow 6 / 6) | 1.10 / 1.10; 5 / 6 | 54 / 54 / 51; 54 / 54 / 52 | not in that sweep |
| 19 | 0 / 0 | 0.233 (0.217) / 0.234 (0.210) | 0 / 0 (slow 4 / 3) | 0.87 / 0.89 | 5.4 / 6.3 | 61 / 54 / 55; 73 / 57 / 56 | 0 / 0 | 0.247 (0.245) / 0.240 (0.184) | 0 / 1 | 0.77 / 1.09; 0 / 1 | 58 / 50 / 51; 58 / 53 / 51 | not in that sweep |
| 20 | 0 / 0 | 0.244 (0.237) / 0.242 (0.234) | 0 / 0 | 0.80 / 0.80 | 4.6 / 5.1 | 56 / 52 / 54; 64 / 55 / 54 | 0 / 0 | 0.243 (0.211) / 0.230 (0.185) | 1 / 3 | 1.00 / 1.10; 1 / 3 | 57 / 51 / 52; 57 / 54 / 52 | 0.245; 0.72 |
| 30 | 0 / 0 | 0.227 (0.217) / 0.226 (0.186) | 0 / 1 (slow 5 / 2) | 0.90 / 1.15 | 5.4 / 30.3 | 79 / 56 / 56; 78 / 61 / 66 | 0 / 0 | 0.222 (0.189) / 0.237 (0.191) | 4 / 1 | 1.12 / 1.09; 4 / 1 | 57 / 56 / 59; 56 / 54 / 51 | 0.238; 0.87 |
| 40 | 0 / 0 | 0.224 (0.203) / 0.222 (0.195) | 1 / 3 (slow 6 / 4) | 0.99 / 1.22 | 6.5 / 13.3 | **96** / 55 / 69; **92** / 60 / 65 | 0 / 0 | 0.218 (0.188) / 0.221 (0.196) | 4 / 3 (slow 4 / 4) | 1.16 / 1.13; 4 / 4 | 70 / 57 / 69; 68 / 57 / 71 | 0.240; 0.85 |
| 60 | 0 / 0 | 0.232 (0.202) / 0.230 (0.197) | 0 / 1 (slow 1 / 2) | 0.97 / 1.08 | 12.3 / 12.3 | **166** / **112** / 69; **120** / 87 / 77 | 0 / 0 | 0.230 (0.190) / 0.240 (0.231) | 1 / 0 (slow 1 / 1) | 1.09 / 0.82; 1 / 0 | **125** / 65 / 68; **113** / 66 / 63 | 0.243; 0.75 |

## Pushes

**Columns.**
- Seeds 200000-9 / 300000-9.
- "Slow gait / upright" is the episodes meeting the rule over the episodes that did not fall.
- Speed and worst drive are the sweep's 2-20 s figures, so they include the 6 s before the push.
- Peak speeds are from 2 s over the episodes that did not fall, so the rows where all fell have none.

**Sources:** `RESULTS.md:252`, `:374` (sideways) and `:102`, `:130` (fore-aft).

**The sweep's break rule:**
- sideways, low side: -1.0 m/s (5 of 10 fell) on both sets;
- sideways, high side: +1.5 m/s on seeds 200000 and +1.25 m/s (1 of 10 fell) on seeds 300000;
- fore-aft: no break.

**Sideways** (+ left, - right):

| Push, free-body dv, m/s | Falls | Speed 2-20 s, mean, m/s | Slow gait over 10-20 s / upright | Worst drive RMS / rating, worst episode, 2-20 s | Heading change, worst, deg | Peak speed, roll / pitch / front elbow / hind knee, rpm |
|---|---|---|---|---|---|---|
| -1.5 | 10 / 10 | all fell | | | | |
| -1.25 | 0 / 3 | 0.222 / 0.221 | 5/10 / 4/7 | 0.96 / 0.95 | 21.4 / 12.7 | 66 / **130** / **217** / **236**; 61 / **132** / **254** / **236** |
| -1 | 5 / 5 | 0.203 / 0.230 | 2/5 / 1/5 (slow 5 / 1) | 0.94 / 0.90 | 5.1 / 6.3 | 51 / **148** / **316** / **156**; 51 / **144** / **267** / **151** |
| -0.75 | 0 / 0 | 0.238 / 0.233 | 4/10 / 6/10 | 0.91 / 0.92 | 8.5 / 5.5 | 23 / 64 / 105 / **86**; 23 / 72 / **108** / **110** |
| -0.5 | 0 / 0 | 0.215 / 0.218 | 10/10 / 9/10 | 0.94 / 0.95 | 8.6 / 6.4 | 27 / 67 / 56 / 60; 27 / 67 / 60 / 73 |
| -0.4 | 0 / 0 | 0.212 / 0.219 | 10/10 / 8/10 | 0.94 / 0.95 | 8.7 / 8.1 | 23 / 57 / 56 / 53; 26 / 57 / 56 / 53 |
| -0.3 | 0 / 0 | 0.212 / 0.213 | 10/10 / 10/10 | 0.94 / 0.95 | 9.3 / 9.8 | 22 / 57 / 51 / 61; 25 / 57 / 56 / 64 |
| -0.2 | 0 / 0 | 0.215 / 0.215 | 10/10 / 10/10 | 0.93 / 0.94 | 7.6 / 5.4 | 20 / 57 / 51 / 53; 20 / 57 / 51 / 53 |
| -0.1 | 0 / 0 | 0.242 / 0.234 | 3/10 / 5/10 | 0.92 / 0.93 | 8.6 / 4.8 | 19 / 57 / 51 / 53; 20 / 57 / 51 / 53 |
| 0 | 0 / 0 | 0.251 / 0.251 | 0/10 / 0/10 | 0.70 / 0.69 | 8.4 / 3.8 | 6 / 57 / 51 / 53; 7 / 57 / 51 / 53 |
| +0.1 | 0 / 0 | 0.251 / 0.250 | 0/10 / 0/10 | 0.70 / 0.78 | 8.5 / 4.1 | 7 / 57 / 51 / 53; 20 / 57 / 51 / 53 |
| +0.2 | 0 / 0 | 0.250 / 0.247 | 0/10 / 1/10 | 0.70 / 0.88 | 8.4 / 4.1 | 10 / 57 / 53 / 53; 20 / 57 / 52 / 53 |
| +0.3 | 0 / 0 | 0.247 / 0.250 | 1/10 / 0/10 | 0.89 / 0.70 | 8.4 / 4.1 | 13 / 69 / 62 / 53; 13 / 64 / 62 / 53 |
| +0.4 | 0 / 0 | 0.242 / 0.250 | 2/10 / 0/10 | 0.90 / 0.72 | 8.4 / 3.8 | 20 / 74 / 68 / 53; 15 / 82 / 66 / 53 |
| +0.5 | 0 / 0 | 0.248 / 0.249 | 0/10 / 0/10 | 0.78 / 0.71 | 8.4 / 3.9 | 20 / **88** / 75 / 58; 19 / **84** / 75 / 57 |
| +0.75 | 0 / 0 | 0.249 / 0.246 | 0/10 / 1/10 | 0.77 / 0.87 | 8.2 / 3.9 | 35 / **107** / 87 / 66; 35 / **107** / 91 / 66 |
| +1 | 0 / 0 | 0.213 / 0.214 | 10/10 / 10/10 | 0.94 / 0.96 | 3.8 / 4.2 | 42 / **124** / **119** / **95**; 42 / **122** / **119** / **95** |
| +1.25 | 0 / 1 | 0.228 / 0.219 | 4/10 / 6/9 | 0.97 / 0.96 | 8.4 / 10.8 | 50 / **138** / **231** / **123**; 52 / **134** / **234** / **122** |
| +1.5 | 10 / 10 | all fell | | | | |

**Fore-aft** (+ forward, - back):

| Push, free-body dv, m/s | Falls | Speed 2-20 s, mean, m/s | Slow gait over 10-20 s / upright | Worst drive RMS / rating, worst episode, 2-20 s | Heading change, worst, deg | Peak speed, roll / pitch / front elbow / hind knee, rpm |
|---|---|---|---|---|---|---|
| -1.5 | 0 / 0 | 0.223 / 0.223 | 6/10 / 6/10 | 0.94 / 0.94 | 7.9 / 6.2 | 21 / 64 / 67 / 67; 20 / 66 / 72 / 71 |
| -1.25 | 0 / 0 | 0.212 / 0.215 | 10/10 / 9/10 | 0.93 / 0.94 | 7.9 / 6.0 | 23 / 57 / 64 / 72; 21 / 57 / 64 / 73 |
| -1 | 0 / 0 | 0.214 / 0.213 | 10/10 / 10/10 | 0.95 / 0.94 | 4.9 / 7.5 | 22 / 57 / 58 / 58; 22 / 57 / 60 / 57 |
| -0.75 | 0 / 0 | 0.224 / 0.220 | 7/10 / 8/10 (slow 7 / 9) | 0.93 / 0.94 | 8.1 / 5.8 | 24 / 57 / 56 / 53; 23 / 57 / 56 / 60 |
| -0.5 | 0 / 0 | 0.213 / 0.213 | 10/10 / 10/10 | 0.94 / 0.95 | 5.6 / 5.7 | 22 / 57 / 51 / 66; 25 / 57 / 54 / 63 |
| -0.4 | 0 / 0 | 0.214 / 0.215 | 10/10 / 10/10 | 0.93 / 0.95 | 8.1 / 8.6 | 20 / 57 / 51 / 53; 23 / 57 / 51 / 53 |
| -0.3 | 0 / 0 | 0.215 / 0.216 | 10/10 / 10/10 | 0.93 / 0.94 | 10.4 / 5.4 | 20 / 57 / 51 / 53; 20 / 57 / 51 / 53 |
| -0.2 | 0 / 0 | 0.235 / 0.241 | 5/10 / 3/10 | 0.91 / 0.93 | 10.0 / 4.7 | 20 / 57 / 52 / 53; 20 / 57 / 52 / 53 |
| -0.1 | 0 / 0 | 0.250 / 0.245 | 0/10 / 2/10 | 0.73 / 0.91 | 8.3 / 5.8 | 11 / 57 / 51 / 53; 20 / 57 / 51 / 53 |
| 0 | 0 / 0 | 0.251 / 0.251 | 0/10 / 0/10 | 0.70 / 0.69 | 8.4 / 3.8 | 6 / 57 / 51 / 53; 7 / 57 / 51 / 53 |
| +0.1 | 0 / 0 | 0.251 / 0.249 | 0/10 / 0/10 | 0.69 / 0.79 | 8.2 / 4.1 | 7 / 57 / 51 / 53; 12 / 57 / 51 / 53 |
| +0.2 | 0 / 0 | 0.251 / 0.248 | 0/10 / 1/10 | 0.69 / 0.87 | 8.3 / 4.0 | 6 / 57 / 51 / 53; 20 / 57 / 51 / 53 |
| +0.3 | 0 / 0 | 0.251 / 0.251 | 0/10 / 0/10 | 0.70 / 0.69 | 8.4 / 3.9 | 7 / 57 / 51 / 53; 8 / 57 / 51 / 53 |
| +0.4 | 0 / 0 | 0.250 / 0.250 | 0/10 / 0/10 | 0.69 / 0.69 | 8.4 / 4.1 | 9 / 57 / 53 / 53; 9 / 57 / 53 / 53 |
| +0.5 | 0 / 0 | 0.247 / 0.250 | 0/10 / 0/10 (slow 1 / 0) | 0.82 / 0.70 | 8.5 / 3.9 | 19 / 58 / 54 / 53; 10 / 58 / 54 / 53 |
| +0.75 | 0 / 0 | 0.250 / 0.251 | 0/10 / 0/10 | 0.73 / 0.69 | 8.3 / 4.0 | 14 / 63 / 59 / 53; 15 / 62 / 59 / 53 |
| +1 | 0 / 0 | 0.245 / 0.249 | 2/10 / 1/10 | 0.91 / 0.89 | 8.3 / 4.1 | 20 / 65 / 65 / 53; 20 / 68 / 64 / 62 |
| +1.25 | 0 / 0 | 0.244 / 0.253 | 4/10 / 1/10 | 0.91 / 0.91 | 9.6 / 4.1 | 24 / **89** / 93 / 53; 21 / **98** / 94 / 53 |
| +1.5 | 0 / 0 | 0.256 / 0.250 | 1/10 / 3/10 | 0.91 / 0.93 | 8.3 / 3.9 | 22 / **109** / **112** / 53; 20 / **112** / **112** / 56 |

**The same pushes on the 188 mm models** (slow gait over 10-20 s, seeds 200000-9 / 300000-9; `slow_gait/capsule_*.json`,
`slow_gait/walk60_*.json`):

| Push | 128 mm capsule | 188 mm capsule | 188 mm roll-ankle |
|---|---|---|---|
| none | 0 / 0 | 0 / 0 | 0 / 0 |
| back 0.3 m/s | 10 / 10 | 6 / 7 | 0 / 0 |
| back 1.0 m/s | 10 / 10 | 10 / 10 | 0 / 0 |
| right 0.3 m/s | 10 / 10 | 10 / 10 | 0 / 0 |
| left 1.0 m/s | 10 / 10 | 9 / 9 | 0 / 0 |

Of the 200 push-family windowed episodes on the 188 mm models, only 24 have a stored episode to check against
(roll-ankle model: sideways pushes and no push, seeds 200000-200007). The others stand on the windows alone (see How these
were run).

## The slow gait in detail

**Pushed episodes.** Both push families, nominal model, 10-20 s (after the push), seeds 200000-9 and 300000-9.
Each family's dv-0 row is included: 40 records of the same 20 unpushed episodes, identical in both files. Of the 706
records that stayed up:
- 291 meet the rule: 0.184-0.205 m/s, hind knee 1.01-1.09, 1.22-1.41 deg nose up.
- 5 are slow with the knee just under rating: 0.190-0.221 m/s, knee 0.94-1.00, 0.97-1.16 deg nose up. These are
  fore-aft +0.5 seed 200005; fore-aft -0.75 seed 300005; sideways -1.0 seeds 200003, 200005 and 200009.
- 410 are not slow: 0.226-0.256 m/s, knee 0.61-0.88, 0.71 deg nose up to 0.16 deg nose down. Twelve of these sit
  between the two states (0.226-0.239 m/s, knee 0.81-0.88; seeds 200000, 200005, 300001, 300009). That fits an entry
  partway through the window, but it was not checked step by step.

**Judged episodes.** The windowed judged conditions, 300 episodes: J1-J4, R1, R2, R5 and J1 at 16-60 ms. All are
randomised except R5.
- 42 meet the rule: 0.168-0.207 m/s, knee 1.01-1.24, 0.56-1.46 deg nose up.
- 7 are near misses: slow, with the knee under rating, at 0.214-0.225 m/s and knee 0.84-0.98.
  - Five are 0.67-1.00 deg nose up: J1 seed 203, R1 seed 203, J3 seeds 107 and 206, and J1 at 16 ms seed 210.
  - J1 at 60 ms seeds 105 and 204 are near level (0.32 and 0.04 deg nose up).
- 19 are the milder nose-down slowing at 19-40 ms, described under Delay.
- 232 are not slow: 0.227-0.259 m/s, knee 0.54-0.85.

**What is not known.**
- Whether the trot ever leaves the slow gait: the episodes end at 20 s.
- Why it enters it: not investigated.

## Other sweep settings

**Columns.**
- Nominal model, no push, 10 ms, 0.25 m/s; seeds 200000-9 / 300000-9.
- Worst drive: the worst joint's mean over the episodes, with the worst single episode in brackets.

**Sources:** `RESULTS.md:194-250` and `:316-372`. The table has every point of these families.

The 188 mm roll-ankle column comes from the 19 Nov sweep, 8 seeds. Its means and speeds are from
`results/sweep_walk60/summary_walk60.md`. The bracketed worst episodes are from
`results/sweep_walk60/sweep_walk60.json.gz` (`heat_worst_episode`).

**Break rule:** the only break is kp x0.8 (low side; 0.191 m/s, 24 % below nominal) on both sets. The roll-ankle
model had no break on any of these axes.

| Setting | Falls | Speed, mean (slowest), m/s | Slow gait over 10-20 s | Worst drive RMS / rating: worst joint's mean (worst episode) | Episodes at or over 1.0 (2-20 s) | 188 mm roll-ankle sweep: worst joint's mean (worst episode); speed |
|---|---|---|---|---|---|---|
| Floor friction 0.15 | 0 / 0 | 0.258 (0.257) / 0.258 (0.257) | not windowed | 0.80 (0.80) / 0.80 (0.80) | 0 / 0 | 0.80 (0.80); 0.260 m/s |
| Floor friction 0.2 | 0 / 0 | 0.257 (0.256) / 0.257 (0.255) | not windowed | 0.78 (0.78) / 0.78 (0.78) | 0 / 0 | 0.78 (0.78); 0.257 m/s |
| Floor friction 0.3 | 0 / 0 | 0.256 (0.255) / 0.256 (0.256) | not windowed | 0.74 (0.76) / 0.74 (0.76) | 0 / 0 | 0.74 (0.74); 0.254 m/s |
| Floor friction 0.4 | 0 / 0 | 0.257 (0.255) / 0.257 (0.256) | not windowed | 0.74 (0.76) / 0.73 (0.77) | 0 / 0 | 0.71 (0.72); 0.252 m/s |
| Floor friction 0.5 | 0 / 0 | 0.257 (0.256) / 0.257 (0.256) | not windowed | 0.73 (0.75) / 0.74 (0.76) | 0 / 0 | 0.69 (0.71); 0.250 m/s |
| Floor friction 0.6 | 0 / 0 | 0.256 (0.255) / 0.256 (0.253) | not windowed | 0.73 (0.75) / 0.72 (0.75) | 0 / 0 | 0.67 (0.70); 0.248 m/s |
| Floor friction 0.8 | 0 / 0 | 0.251 (0.239) / 0.251 (0.236) | 0 / 0 (slow 0 / 1) | 0.68 (0.77) / 0.66 (0.82) | 0 / 0 | 0.66 (0.67); 0.246 m/s |
| Floor friction, the model's own | 0 / 0 | 0.251 (0.250) / 0.251 (0.250) | 0 / 0 | 0.68 (0.70) / 0.67 (0.69) | 0 / 0 | 0.68 (0.69); 0.246 m/s |
| Floor friction 1.0 | 0 / 0 | 0.248 (0.247) / 0.248 (0.247) | not windowed | 0.71 (0.72) / 0.71 (0.72) | 0 / 0 | 0.73 (0.74); 0.248 m/s |
| Floor friction 1.2 | 0 / 0 | 0.247 (0.245) / 0.246 (0.245) | not windowed | 0.72 (0.75) / 0.74 (0.76) | 0 / 0 | 0.77 (0.80); 0.246 m/s |
| Mass x0.8 | 0 / 0 | 0.249 (0.249) / 0.245 (0.206) | 0 / 0 (slow 0 / 1) | 0.61 (0.62) / 0.62 (0.90) | 0 / 0 | 0.63 (0.64); 0.243 m/s |
| Mass x0.9 | 0 / 0 | 0.250 (0.249) / 0.242 (0.204) | 0 / 0 (slow 0 / 2) | 0.63 (0.65) / 0.66 (0.96) | 0 / 0 | 0.65 (0.67); 0.245 m/s |
| Mass x1.1 | 0 / 0 | 0.252 (0.251) / 0.252 (0.250) | not windowed | 0.70 (0.75) / 0.70 (0.77) | 0 / 0 | 0.71 (0.71); 0.248 m/s |
| Mass x1.2 | 0 / 0 | 0.250 (0.243) / 0.250 (0.243) | 0 / 0 | 0.77 (0.94) / 0.74 (0.90) | 0 / 0 | 0.73 (0.73); 0.248 m/s |
| Mass x1.4 | 0 / 0 | 0.247 (0.225) / 0.241 (0.220) | 2 / 4 | 0.93 (1.13) / 0.99 (1.19) | 3 / 4 | 0.78 (0.80); 0.248 m/s |
| Mass x1.6 | 0 / 0 | 0.238 (0.219) / 0.239 (0.217) | 4 / 5 | 1.12 (1.32) / 1.11 (1.36) | 8 / 8 | 0.88 (0.91); 0.247 m/s |
| Payload 0.5 kg | 0 / 0 | 0.240 (0.206) / 0.234 (0.199) | 3 / 4 | 0.76 (1.02) / 0.80 (1.06) | 2 / 2 | 0.69 (0.70); 0.247 m/s |
| Payload 1 kg | 0 / 0 | 0.236 (0.199) / 0.227 (0.199) | 4 / 6 | 0.82 (1.08) / 0.90 (1.10) | 2 / 3 | 0.69 (0.69); 0.247 m/s |
| Payload 2 kg | 0 / 0 | 0.219 (0.201) / 0.219 (0.205) | 9 / 9 | 1.04 (1.14) / 1.05 (1.13) | 8 / 8 | 0.74 (0.80); 0.246 m/s |
| Payload 3 kg | 0 / 0 | 0.213 (0.208) / 0.211 (0.205) | 10 / 10 | 1.15 (1.18) / 1.16 (1.21) | 10 / 10 | 0.81 (0.91); 0.244 m/s |
| Force limit x0.4 | 0 / 0 | 0.244 (0.211) / 0.241 (0.171) | 0 / 0 (slow 3 / 2) | 0.61 (0.69) / 0.60 (0.81) | 0 / 0 | 0.60 (0.61); 0.249 m/s |
| Force limit x0.55 | 0 / 0 | 0.252 (0.251) / 0.252 (0.251) | not windowed | 0.61 (0.67) / 0.61 (0.64) | 0 / 0 | 0.62 (0.64); 0.250 m/s |
| Force limit x0.7 | 0 / 0 | 0.251 (0.251) / 0.249 (0.226) | 0 / 0 (slow 0 / 1) | 0.65 (0.68) / 0.66 (0.86) | 0 / 0 | 0.63 (0.65); 0.250 m/s |
| Force limit x0.85 | 0 / 0 | 0.251 (0.250) / 0.251 (0.250) | not windowed | 0.67 (0.72) / 0.67 (0.72) | 0 / 0 | 0.67 (0.67); 0.249 m/s |
| kp x0.4 | 0 / 0 | 0.177 (0.174) / 0.177 (0.175) | not windowed | 0.94 (0.95) / 0.94 (0.95) | 0 / 0 | 0.59 (0.60); 0.216 m/s |
| kp x0.6 | 0 / 0 | 0.177 (0.175) / 0.176 (0.175) | not windowed | 1.02 (1.04) / 1.02 (1.04) | 10 / 10 | 0.60 (0.70); 0.233 m/s |
| kp x0.8 | 0 / 0 | 0.191 (0.183) / 0.191 (0.184) | 10 / 10 | 0.99 (1.04) / 0.98 (1.05) | 6 / 5 | 0.64 (0.64); 0.242 m/s |
| kp x1.2 | 0 / 0 | 0.251 (0.250) / 0.251 (0.250) | not windowed | 0.74 (0.76) / 0.73 (0.76) | 0 / 0 | 0.77 (0.78); 0.248 m/s |
| kp x1.5 | 0 / 0 | 0.250 (0.249) / 0.249 (0.248) | not windowed | 0.81 (0.83) / 0.81 (0.83) | 0 / 0 | 0.85 (0.86); 0.247 m/s |

Nominal row (mass x1, payload 0, force limit x1, kp x1, the model's friction): 0.251 (0.250) / 0.251 (0.250) m/s;
worst drive 0.68 (0.70) / 0.67 (0.69); no slow gait.

Peak drive speeds stay under the allowances at every point in this table. The largest are front elbow 78-82 rpm at
force limit x0.4 and kp x1.5, and pitch 65 rpm at kp x1.5.

## What this does not show

- **One controller.** It is frozen and was developed on the roll-ankle foot, with no tuning for the capsule sole or
  the 128 mm foreleg. The learned policies are the other controller, and they have not been run through these
  conditions on this model.
- **A rule fitted to the data.** The slow-gait rule's cut-offs (0.225 m/s and 1.0 of rating) were set after the data
  was seen. Nothing here is a verdict.
- **A foot-load signal the robot will not have.** The trot picks its odometry stance legs from the foot loads
  (`walk60.py:224`, `:239`). The core build has no foot load cells (D46), so every result here assumes a signal the
  robot's own sensors do not give.
- **Simulation only,** of a model whose masses and contact are estimates. Nothing is built or measured.
- **Heat as load, not temperature.** Heat is RMS torque / continuous rating over 2-20 s or 10-20 s windows, not a
  winding temperature. The thermal model was not run, at 35 C (D48) or at any other room temperature.
- **Peak speeds against desk figures.** Peak speeds are the largest joint speeds from 2 s over the episodes that did
  not fall, from the 1 ms log (as `RESULTS.md:7` defines them), against desk allowances. Episodes that fell are left
  out; their 99th percentiles, which include the fall, go over the allowances. The X4 cap rests on a reseller's text
  quoting the maker and is not confirmed (E1).
- **Short episodes.** Each lasts 20 s, so a longer walk is not measured.
- **Limited conditions:**
  - flat ground only (D28);
  - no turns;
  - pushes only at 8 s, with one push per episode;
  - no recovery count for fore-aft pushes.

## Next (proposals)

1. Find why the trot enters the nose-up slow gait on capsule-soled models. A controller change would need its own
   pre-registration; the frozen trot and these results stay as they are.
2. Measure the learned policies on the adopted model with the same fore-aft pushes and the 15-20 ms delay points
   (D45's Colab step), so that two controllers are compared before anything is called a result.
3. If the trot is to be judged on the adopted model, pre-register first. The pre-registration should fix the
   thresholds and a slow-gait rule, and run on new seeds.
4. Run the thermal model at 35 C (D48) on the slow gait's hind-knee torques.

## Tests

`python -m pytest walk/tests rl/robustness/tests`, run from `hardware/triceratops_build`: 33 passed (28 Nov 2026:
15 in `walk/tests/test_fore128.py`, 1 in `walk/tests/test_walk.py`, 17 in `rl/robustness/tests/test_robustness.py`).
The tests check:
- **Defaults leave existing results unchanged:**
  - at its defaults, on the walk's own model, the driver reproduces the stored judged episode (J1, seed 101) exactly;
  - it reproduces the stored sweep episode (+1.25 m/s, seed 200000) exactly;
  - added delay points keep every existing point's key;
  - the fore-aft push goes along the body.
- **The windows are of the stored episodes:**
  - a 10-20 s window episode reproduces the stored fore-aft sweep episode (-0.3 m/s, seed 200000) on the adopted
    model;
  - it also reproduces the stored judged episode (J1, seed 101);
  - one flipped fall or one changed speed reads as "not the same episodes";
  - a point-setting file attaches only to its own row.
- **The committed RESULTS.md and summary.json regenerate exactly.**

## Corrected after the independent check (27-28 Nov 2026)

The first draft of this report was checked independently before it was saved. These are the changes:
- **Back-push count:** 7 of 10 at 0.75 m/s on seeds 200000, not 9-10.
- **Drive speeds:** every case over an allowance is now listed. The first draft left out the 0.5-0.75 m/s sideways
  pushes, the forward pushes and two front-elbow cases.
- **Payload and mass:** payload 0.5-1 kg and mass x0.8-0.9 were added. Windows were run for payload, mass, delay,
  force-limit and friction points.
- **The slow-gait rule:** it is now written in `fore128_slow_gait.py` and shown in `RESULTS.md` as a second count.
  The 30-60 ms cases are named, and the 0 ms result is shown to rest on the cut-off.
- **Ranges:** widened, with their sources. The sentence on the capsule sole is now an association, not a cause.
- **Window checks:**
  - sweep rows: falls and 2-20 s speeds, seed by seed;
  - judged conditions: whole-episode and 2-20 s speeds, labelled as such;
  - point-setting files now sit beside their own rows.

  All 1,740 windowed episodes were rerun under these checks. The 1,564 that have a stored episode match it; the
  other 176 have none to match (see How these were run).

A second independent check (28 Nov 2026) found these, now corrected against the stored outputs:
- **Push-row heat:** the sweep's 2-20 s worst episode is 0.87-0.97 in the push rows with the slow gait, and
  0.69-0.97 over all push rows (it was given as 0.89-0.97).
- **Foot loads:** the trot reads them for its leg odometry. The robot will not have them (D46). This is now stated
  under the controller and under What this does not show.
- **Citations:** the plan's 8 episodes per point are at `PLAN_SWEEP_WALK60.md:24`. The 188 mm roll-ankle worst
  episodes come from the sweep's JSON, not its md.
- **Ranges and counts:**
  - the not-slow 30-40 ms episodes run at 0.227-0.243 m/s;
  - the roll-ankle model ran 80 pushed and 20 unpushed episodes;
  - two judged near misses are near level, not nose up.
- **Coverage:** the other-settings table now has every point (friction 0.2, 0.3, 0.5 and 0.6 and mass x1.1 added).
- **Provenance:** the window runs' `rl/trike_env.py` hash is now given.

The final check of this text also corrected:
- the worst foot's slip at the contacts (0.05-0.10, not 0.04-0.10);
- the delay of R2 (0 ms, not 10 ms);
- the 188 mm capsule check's conditions (J1-J4 and R1-R5);
- the next-largest heading changes;
- the sources and dates of the speed allowances.

A third independent check (28 Nov 2026) found these, now corrected:
- **Window checks:** the text said that all 1,740 windowed episodes pass the seed-by-seed checks. Only the 1,564
  with a stored episode can be checked, and they match; the other 176, push-family windows on the 188 mm models, have
  nothing to match. The 188 mm capsule model's 4 judged groups are now counted among the checks.
- **Ratings:** the continuous torque ratings now carry their source and date, with the X6's stand-in table.
- **Peak speeds:** they are now stated to be over the episodes that did not fall, from 2 s. In the episodes that
  fell, the 99th percentiles go over the allowances.

Also corrected then: the Tools bullet now says the tests reproduce one stored judged episode and one stored sweep
episode (it said the stored episodes).
