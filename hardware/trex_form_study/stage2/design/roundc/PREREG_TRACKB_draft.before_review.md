# Stage 2, Track B: rules fixed before the runs (R5, R6, R2L, the gates, the monitor, the notebook) - DRAFT

**Status (15 Nov 2026): a draft for the lead, not committed.** It is written before any look set is frozen, before
any Stage 2 closed-loop episode on a look set, before any R2L fit and before any training run. Simulation only; it
changes no body and adopts nothing without the owner (P24). It fixes everything that `ROUND_C_DESIGN.md` (the
design, same directory) leaves numeric or optional, in the style of `STAGE2_75_PLAN.md`. Paths are relative to
`hardware/trex_form_study/`. A rule changed after a result is labelled so. To be reviewed independently before its
commit as `stage2/tb/PREREG_TRACKB.md`. Check: the commit that adds that file is an ancestor of the commit in
every Track B record's `sessions[].commit` and of every look set's `freezes[].commit` that a Track B record uses.

## What had been run when these rules were drafted

- Everything listed in `STAGE2_75_PLAN.md` under the same heading.
- Four scratch experiments of 15 Nov on Stage 1's own run directory (`stage1/runs/2026-11-08`: stock references,
  no Stage 2 set), run while the design was made. They are not records and none may be cited as a result.

| Scratch experiment | Numbers (not records) | Bears on |
|---|---|---|
| The panels' open-loop row against a tracker in R2 mode with no feedback term and no ankle cap, stepped as an env steps it, on two separately built plants; `75-M-H-10`, 6 s, one seed | state digests equal at 3,000 of 3,000 physics steps at 0 ms and at 10 ms | Z1 |
| A stub env (Stage 1's env on that base, its own observation and reward) on `75-M-H-10` | 438-460 env steps/s on hard floor, 352-422 on robot_floor_carpet, one process; under action noise of 0.1 of the range the walk falls about every 200 env steps | O1, O2, Y11 |
| The same base's zero action on `60-N-T-10`, 5 episodes per task | walk up in 2 of 5 at 0 ms and in 2 of 5 at 10 ms; brisk walk and walking turn 0 of 5 at both; turn in place and stand 5 of 5 | M6 |
| A start from a frame inside the nominal stride with the delay queue filled from the reference; `75-M-H-10` | up 8 s from 6 frames at 0 and at 10 ms on hard floor; falls after 2.4-3.6 s on robot_floor_carpet | T3 |
| An archive saved with module-level schedule objects; a dummy env, 3 PPO updates | no bytecode member; fingerprint reproduced with difference 0.0; 70 of 120 minibatch steps ran | K6, O3 |

No constant below was set on a Stage 2 closed-loop result. Constants marked "(scratch)" lean on the table.

## In brief, for the owner

- **What is trained.** One policy per body, head and seed ("R5"): at every 20 ms it nudges the servo targets of
  the reference walk (by at most 0.35 rad on the legs and 0.15 rad on the head and tail) and the walk's clock (by
  at most 25 %). With no nudge it is the reference played open loop. A second kind ("R6", one per body) sets the
  targets alone, as a comparison. A balance controller refitted with delay in its mix ("R2L") is fitted here as a
  baseline.
- **Two checks before any training.** The policy's "do nothing" must reproduce the open-loop row of the agility
  bar exactly, step for step, at 0 and at 10 ms of delay. And the new readings (toe clearance, the look bands)
  must read as designed on the references themselves.
- **One go or no-go before Colab.** 2 million steps here on the 75 cm body; Colab starts only if the policy then
  does better than the open-loop row on 18 fixed episodes at 10 ms. Otherwise you are told first.
- **The 60 cm body is the hard case**: its open-loop walk falls within seconds. A longer local run (5 million
  steps) is made on it first; if its slow walk does not stay up in 2 of 3 episodes by then, you are told before any
  60 cm Colab hours are spent.
- **During a Colab run** the same 18 episodes are run every 2.5 million steps; they decide only when a run stops.
  Verdicts come later, from the 20-episode panels here.
- **What may come back to you.** A failed go or no-go; a failed 60 cm local run; a Colab rate too low for 15
  million steps in 20 hours; uploads of the input files that fail (then a fallback you perform or approve).

## Words, names and tags

As `STAGE2_75_PLAN.md`, with these additions. R5, R6, R0L, R2L are controllers (the plan's B.1-B.2). "The bar",
"the panels", "reading 1, 2, 3" as there. "The design" is `ROUND_C_DESIGN.md`. k is a servo delay in physics steps
of 2 ms. An "env step" is 20 ms. A "check" is a monitor run (M1). This text's rules are E1-E6, T1-T3, W1-W4,
J1-J5, O1-O4, M1-M6, U1-U5, Z1-Z2, Q1-Q6, K1-K7, Y1-Y11 and N1-N5; outside a rule's head "S1" is still Stage 1's
report and "G1"-"G9" the gates. Tags: **[plan]**, **[S1]**, **[NEW]**, **[differs: plan]**, **[differs: S1]** as in
`STAGE2_75_PLAN.md`; **[S2-n]** an amendment of the plan's section 9; **(scratch)** a constant leaning on the table
above. Every rule ends with "Check:".

## E. The controllers

**E1. R5.** [plan S2-3] `control.Tracker` in R2 mode with `terms=()`, `ankle_cap=()`, at the commanded clock 1.05,
on a frozen set's references; every 20 ms the policy's clipped, low-passed action a sets joint offsets
`a[:nj] x scale` (0.35 rad on leg joints, 0.15 rad on every other joint) added to the reference's servo angles,
and the clock factor `1 + 0.25 a[nj]`; rates and feedforward stay the reference's. A zero action with the head
neutral sends R0L's targets exactly. Check: `run.json` `settings.env_configs[].action_leg_rad`,
`action_other_rad`, `action_rate`; gate Z1.

**E2. R6.** [plan B.2] Joint targets `clip(home + a x span, range)` with Stage 1's span rule, no feedforward, its
own estimator; R5's inputs, commands, reward, starts, randomisation, settings and budget; one seed (42) per body
with the current head. Check: `run.json` `mode`, `settings`.

**E3. Inputs.** [plan; NEW widths] Actor: encoder angles and rates (nj each), gravity direction (3), gyro (3), foot
loads (2), gait phase (2), reference support (2), command (2), the last three actions (3 x (nj + 1)) and on `-X`
the head command (4): 87 inputs for 14 joints, 101 for 16. The critic also sees privileged quantities (design
3.1): 129 and 145 in all. Actions: 15 and 17. Check: `run.json` `layout`; `checkpoint_<n>.json` `layout`.

**E4. The command.** [NEW; differs: S1, whose input was the reference's own speed] The input is `(v, w)` times
1.05, piecewise constant on the reference clock: the gait's speed and yaw rate in a periodic gait; the next gait's
from the first frame of a start or change; zero from the first frame of a stop; in a finite turn of angle A at rate
w0 the walk's speed and `w0` for `A / |w0|` seconds of reference time from the turn's first frame, then zero.
Scales 0.5 m/s and 1.0 rad/s. Check: `tests/test_stage2_r5lib.py` (constant inside a segment; nothing before a
segment's first frame depends on it); the env description's `command_rule`.

**E5. The head command on `-X`.** [plan; NEW] `[yaw now, jaw now, yaw goal, jaw goal] / 0.6 rad`; the commanded
yaw and jaw angles are added to the reference's neck-yaw and jaw targets before the policy's offsets. Head neutral
is all zero. Check: the env description's `head_rule`; Z1 on the `-X` bodies with the head neutral.

**E6. Where the code lives.** [differs: plan B.4, which has the notebook run `designlab/stage1/train.py`, extended]
No file under `designlab/stage1/` is edited: the env, the trainer and the fit are new modules in
`designlab/stage2/`. S2-17 as ruled allows B.4's extension; this rule does not use it. Check: `git diff --stat
<state A>.. -- designlab/stage1` is empty; `stage2/regress/regress.json` is current.

## T. Tasks and starts

**T1. The task table.** [plan B.1; NEW weights] Each task is a plan over the run's set; `L` is `start`, one window
of `nominal`, `nominal_to_v0.4`. Weights are relative.

| Task | Plan | Weight |
|---|---|---|
| `stand` | a 12 s hold of the home stance (10 s played) | 1 |
| `walk_nominal` | `start`, `nominal` to the end | 2 |
| `walk_v0.4` | L, `v0.4` to the end | 3 |
| `walk_v0.3`, `walk_v0.35`, `walk_v0.45`, `walk_v0.5` | `start`, one window of `nominal`, the change to the rung, the rung to the end | 0.5 each |
| `turn_q90_left`, `turn_q90_right` | L, 2 (right: 2.5) strides of `v0.4`, `turn90_pi4_<dir>`, `v0.4` to the end | 1.5 each |
| `turn_<0.35, 0.5, 1>_<left, right>` | the same with `turn90_<rate>_<dir>` | 0.25 each |
| `stop` | L, 2.5 strides of `v0.4`, `stop2_v0.4`, a hold | 2 |
| `start_stop` | the `stop` task, a 2 s hold, `start`, 2 windows of `nominal` | 1 |
| `inplace_left`, `inplace_right` | `start_inplace_0.5_<dir>`, 3 strides of `inplace_0.5_<dir>`, `stop_inplace_0.5_<dir>` | 0.5 each |

Episodes last at most 30 s. Nudges are no task: pushes are drawn on every task (J3). Check: the env description's
`tasks`; `run.json` `settings.env_configs[].task_weights`.

**T2. Absent references.** [NEW; plan B.1 read] A task whose reference the set does not hold (`available: false`,
or not built) is dropped, its weight not redistributed by hand (the remaining weights are renormalised), and the
env description lists it under `absent` with the set's reason. "Walking turns at 0.35-1.0 rad/s" are the finite
90 degree turns; no periodic walking-turn gait is trained [differs: plan B.1's list, as read]. If the turn in
place is absent from a set, the plan's B.1 list is not met for that body and the report says so. Check: the env
description's `absent`; `set.json` `not_available`.

**T3. Starts.** [plan] Half of the episodes of every task but `stand` start mid-task: the reference time is
uniform between 0.2 s after the first non-hold frame and 2.0 s x 1.05 before the plan's end, snapped to a file
frame; the plant is set to that frame (rates x 1.05) with Stage 1's start noise (0.01 rad on the joints, 0.02 m/s
on the root's velocity), the estimator is reset from that state, and the delay queue holds the k commands the
zero action would have given before it. The others start from the settled stance with Stage 1's start noise.
(scratch) Check: `settings.env_configs[].mid_start == 0.5`; `tests/test_stage2_r5env.py` (the three mid-start
tests).

## W. Reward

**W1. Form.** [plan] `r = sum of the positive terms x their weights - sum of the penalties x their weights`, each
term in [0, 1], read on the true state at the end of every env step; formulas in design 3.4. Check: the env
description's `reward_rule`.

**W2. Weights and their source.** [S1 for Stage 1's terms; NEW for the others]

| Term | Weight | Source |
|---|---|---|
| joint imitation (scale 0.1 rad) | 0.55 | Stage 1 (`env.REWARD_RULE`) |
| speed and yaw-rate tracking, against the reference's own at the commanded clock | 0.35 | Stage 1 |
| foot imitation (both feet relative to the root, scale 0.10 H) | 0.15 | new |
| contact imitation (within 24 ms of the reference's support) | 0.10 | new |
| head attitude in the world (roll and pitch, scale 3 degrees) | 0.10 | new |
| duty, slip, impact, clearance, tilt | 0.02 each | Stage 1 |
| flight | 0.05 | Stage 1 |
| action rate | 0.03 | new |
| touchdown speed (over 0.15 m/s, full at 0.45 m/s) | 0.02 | new (the plan's "impact") |
| neck, tail, neck-yaw and jaw drives over their sustained rating (5 s exponential RMS; full at 1.5 x) | 0.05 | new |
| leg-to-leg interference (from 3 mm to contact) | 0.10 | new (W4) |

The new weights were set on 15 Nov on the recipe of `stage2/research/research_look.md` 3.3, before any run, so
that no new term outweighs Stage 1's joint imitation. They are the same for every body, head, seed and mode and are
not changed by M4's fix. Check: `settings.env_configs[].reward` in every `run.json` equals this table.

**W3. Two of Stage 1's terms read differently.** [differs: S1] `clearance` is read between 25 % and 75 % of the
planned swing time since the foot's **own** lift-off (the bar's reading), not on the reference's clock. `duty`'s
band is 0.55-0.72 while the commanded target speed is below 0.30 m/s (owner decision 2), else 0.55-0.65. Check:
the env description's `reward_rule`.

**W4. Leg-to-leg interference.** [NEW; `LESSONS_LEARNED.md` 2.8] Read at the end of every env step with
`legs.LegGap` (every left-leg against right-leg pair of the screened geometry, the reading of the bar's turn
item). In the reward (W2) and in the termination: an episode ends when the smallest distance is below zero at two
consecutive env-step ends. A leg against the trunk or the tail is not read. Check:
`settings.env_configs[].interference == "terminate"`; `info["interference"]` in the env's steps.

## J. Randomisation

**J1. Delay.** [S2-9] k uniform on the integers 0 to 12 per episode. Check: `env_configs[].latency_steps == [0, 12]`.

**J2. The plant.** [plan] Per episode: servo stiffness x 0.6-1.2; sliding friction 0.4-1.2; trunk mass and inertia
x 0.9-1.1 (`env.MASS_RULE`); joint damping x 0.5-1.5 and joint friction-loss x 0.5-1.5, one factor each for all
hinges [NEW precision]; IMU noise scale 0.5-1.5 [S1]. The controller's model and feedforward stay nominal. Check:
`env_configs[]`; `tests/test_stage2_r5env.py::test_factors_are_applied_and_restored`.

**J3. Pushes.** [plan; S1 for the pulse] With probability 0.5 in an episode longer than 4 s: one 0.1 s pulse at
the trunk's CoM at a time uniform in `[2 s, end - 2 s]`, direction uniform, `dv = u x` the body's Stage 0 forward
bound (its own envelope screen; no default value), u uniform on 0-0.5 if the reference walks at that time and on
0-1.0 if it stands. Check: `env_configs[].push_walking`, `push_standing`; the env description's `push_bound_m_s`.

**J4. Gaze and jaw events on `-X`.** [plan; NEW numbers, from `research_look.md` 5.1] With probability 0.5 a train
of gaze shifts for the whole episode: yaw target uniform on +-0.5 rad, minimum-jerk over 0.8-1.2 s, held 2-6 s.
With probability 0.25 one jaw event: open to 0.3-0.6 rad over 0.4-0.6 s, held 1.0-1.5 s, closed over the same
time. No head-first turn (F11) is built. Check: `env_configs[].gaze_prob`, `jaw_prob`; the env description's
`head_rule`.

**J5. Floors and seeds.** [plan; S1] Env i of 8 runs on hard floor for `i % 4` in 0-2 and on robot_floor_carpet
for `i % 4 == 3`; no pile. The task, start, factors, push and head events come from the env's own generator; the
start's and the sensors' noise from the episode seed (`env.EPISODE_SEED_RULE`); env i resets at boundary b of PPO
seed s with seed `100,000 s + 100 b + i` (K1). Check: `sessions[].floors`, `boundary_seeds`.

## O. PPO and the probes

**O1. Settings.** [S1 design 2; NEW for 8 envs] 8 envs in every run and session; `n_steps` 2048 (a rollout is
16,384 steps); batch 256; 5 epochs; gamma 0.99; lambda 0.95; clip 0.2; `target_kl` 0.03; networks (256, 256) with
the privileged critic; observation and reward normalisation clipped at 10 and 50; one torch thread; CPU. The
number of worker processes is the runtime's and changes no result. Every update's minibatch steps run, of the 320
possible, are recorded (lesson 16). Check: `settings.ppo`; `updates.jsonl`;
`tests/test_stage2_vecpool.py`.

**O2. Exploration.** [plan] Initial sigma 0.1 of each action's range (`log_std_init = ln 0.1`). The sigma probe is
reported and decides nothing: on each current-head body whose V6 set is frozen, the stance-start `walk_nominal` and
`walk_v0.4` tasks at k = 5 on the nominal plant, 10 episodes each (episode seeds 7000-7009), actions
`clip(sigma z)` with the same z for every sigma, sigma in 0, 0.025, 0.05, 0.1, 0.2: survivals and mean survival
time per sigma. (scratch) Check: `settings.log_std_init`; `stage2/tb/validation/sigma_probe.json`.

**O3. The learning rate.** [plan; decision 24's probe] For each body and head whose V6 set is frozen, for each of
1e-4, 3e-5, 1e-5: a fresh R5 model, seed 42, training's randomisation, sigma 0.1, the rate held constant, 6
updates. A rate *passes on a body* when all 320 minibatch steps ran in each of the 6 updates. The value used is
the largest that passes on every body and head probed; the schedule is linear from it to 0.3 x it over the run's
budget. If none passes, nothing trains and it goes to the lead. A body or head whose set is frozen after S2-4 is
probed before its first run; if that lowers the value, the Colab runs use the lower one, S2-4 is not repeated,
and the record says so. Check: `stage2/tb/validation/lr_probe.json`; `settings.lr`.

**O4. The action filter.** [plan: "re-chosen in validation"; NEW rule] Candidates: a first-order low-pass at 10 Hz
(Stage 1's) and at 20 Hz. Measured once, before S2-4, on the current-head bodies whose V6 set is frozen (at least
`75-M-H-10`): O2's episodes at sigma 0.1 under each cut-off. 20 Hz is used if its mean survival time, pooled over
those episodes, is at least 0.9 x that at 10 Hz; otherwise 10 Hz. One value for every run of the round; it is not
measured again. Check: `stage2/tb/validation/filter.json`; `env_configs[].filter_hz`.

## M. The monitor and the stop rules

**M1. Episodes.** [plan section 6; S2-10] 18 episodes: the bar's conditions `walk/nominal`, `walk/v0.4`,
`turn/q90_left`, `stop/two_step`, `nudge/forward`, `nudge/left` of the run's set at mu 0.9, each on episode seeds
6001, 6002, 6003 [NEW seeds: none is a panel, fit or validation seed], at 10 ms, as the bar builds and judges
them. A condition without a reference is "not run" and counts as not succeeding and as zero survival for both
rows. Check: `monitor2.CONDITIONS`, `SEEDS` (committed before any run); each monitor record's `episodes`.

**M2. Rows.** [NEW] The checkpoint's row `R5/<seed>@10ms` (or `R6/...`), loaded from its saved files through the
loader of K6, and once per run `R0@10ms` (R0L) on the same 18 episodes. A check runs at the first rollout start at
or past each multiple of 2,500,000 steps and on a run's final checkpoint. Check: `monitor/monitor_<n>.json`,
`monitor/r0l.json`.

**M3. A gain; S2-10's reading.** [plan section 6; S2-10 as ruled] N is the number of the 18 episodes succeeding
with the look (U4); S the mean over the 18 of the survival time (the fall time, or the episode's length). A check
shows a gain when `N >= N* + 2` or `S >= S* + 1.0 s`, N* and S* each the best value of an earlier check of the
run, and for the first check R0L's. S2-4 (2,000,000 steps asked, whole rollouts: 2,015,232; `75-M-H-10`, current
head, seed 42, training's delay draw) is read on its final checkpoint against R0L by this rule; earlier
checkpoints' checks are reported and decide nothing. A gain lets Colab start; no gain goes to the owner first. The
record also prints, beside N, the count succeeding but for A1 with the look bands kept, which decides nothing
[NEW, reported]. Check: `stage2/tb/runs/s2-4/S2_10.json` (both rows' 18 episodes, N, S, the two reported counts).

**M4. G8b and its one fix.** [plan section 6] At the first check at or past 5,000,000 steps: `walk/nominal` stays
up in at least 2 of its 3 episodes. Otherwise, once, inside the running job: the training weights of the tasks of
every monitor condition with fewer than 2 of 3 up are doubled, by this table: `walk/nominal`: `walk_nominal`;
`walk/v0.4`, `nudge/forward`, `nudge/left`: `walk_v0.4`; `turn/q90_left`: `turn_q90_left` and `turn_q90_right`;
`stop/two_step`: `stop` and `start_stop`. The reward's weights do not change. At the first check at or past the
fix + 5,000,000 steps the same reading; a second failure stops the run. Check: `run.json` `gates.g8b`, `fix`,
`gates.g8b_after_fix`, `stopped`.

**M5. Stopping without a gain; judging.** [plan section 6; NEW precision] Counting from the check at 20,000,000
steps: the run stops at the third consecutive check without a gain (25,000,000 at the earliest). No best-model
selection; a body and seed's two heads are compared at the step both reached, each run's upper end at its own last
checkpoint; the 220 runtime-hour and 8-day cap and the order of cuts are the plan's. Check: `run.json` `monitor[]`,
`stopped`.

**M6. A local 60 cm run before 60 cm Colab hours.** [NEW; `LESSONS_LEARNED.md` 5.2 item 7] R5 on `60-N-T-10`,
current head, seed 41 (no judged seed), 5,000,000 steps asked, here, with the rules above. It is read at its 5M
check by M4's first sentence. It starts and stops no Colab run by itself; if the reading fails, the owner is told,
with the options a delay curriculum, a feedback base, or launching under M4, before any 60 cm job is queued. Its
checkpoint is never judged on the panels. (scratch) Check: `stage2/tb/runs/local-60/jobs/R5-60-N-T-10-s41/run.json`.

## U. The look reading (reading 1)

**U1. The four measures.** [plan section 5; `research_look.md` 4.2; NEW definitions] From 50 Hz poses of an
episode: *bob*, the mean over the robot's own steps (touchdown to next touchdown of either foot, wholly inside the
window) of the range of the hip-midpoint's height; *head*, the RMS of the head's world pitch about its mean;
*tail phase*, the phase of the least-squares sinusoid of the tail-yaw angle on the robot's own phase (zero at each
of its left lift-offs); *toe-off*, the mean over its lift-offs of the lifting foot's pitch at that instant.
Contact is the 2 % load rule. Check: each episode's `measures.look`.

**U2. Window.** [NEW] A walk item: its judged last 10 s. A turn item: the 3 s after `t_c + at_s`. Nudge and stop
items: not read. Check: `measures.look.window`.

**U3. Bands.** [plan section 5] Against the reference's own values (U5): bob within +-30 %; head pitch at most
1.5 degrees RMS (not relative); tail phase within +-0.1 cycle; toe-off at least 70 %. The tail term is not read
when the reference's tail amplitude in the window is under 0.02 rad, the toe-off term when the reference's toe-off
is under 0.02 rad [NEW]; both then pass, labelled. Check: `measures.look.terms`.

**U4. The readings on a look set.** [plan] Reading 1 ("shown with the look"), the headline for walks and turns:
`success` and all four terms. For nudge and stop items reading 1 is reading 2, labelled [NEW precision: the plan
names walks and turns]. Readings 2 and 3 as `STAGE2_75_PLAN.md` C2. Printed beside and never a verdict: reading 3
with the bands kept. Check: `counts_look`, `counts`, `counts_but_A1`; the records' `reading_1` note.

**U5. The reference's values; errors.** [NEW] The same functions on a kinematic playback of the episode's own
track over the window's reference times at the nominal commanded clock. A window with fewer than 4 of the robot's
steps or 4 lift-offs in an episode that did not fall is an error of the episode (run again; never a failed term).
Check: `measures.look.reference`; the episode's `error`.

## Z. Gates

**Z1. G8a.** [plan B.3; NEW alignment and pass rule] For a set and a body or head: the R5 env with the zero action
(head neutral, nominal plant, the condition's friction, stance start with the row's start seed and Stage 1's start
noise, the condition's push, the delay through the bar's own wrapper, interference termination off) against the
bar's row `R0@0ms` and `R0@10ms`, on the eight gated mu 0.9 conditions the set has references for,
`walk/v0.4_mu0.54` and `walk/nominal`, panel seeds 3042 and 3043. Compared: sha256 of `qpos`, `qvel`, `ctrl`,
`sensordata` after every physics step the row ran, and the same end (if the row fell, the env ends in the env step
holding that step). **Pass: every episode identical.** Any other outcome: no run of that body and head. The
trainer repeats one condition (`walk/v0.4`, seed 3042, both delays) before every new run, here and on Colab; that
is the notebook's line 3. Check: `stage2/tb/gates/<set>/<body>/g8a.json` (`identical == of`); `run.json`
`gates.g8a`.

**Z2. G9.** [plan B.3; S2-11 as ruled] One record per body. It passes when: (a) every frozen reference played
kinematically at the commanded clock passes its own item's readings, the quick turn also reaching 80 degrees at the
slowest clock; (b) the look measures of each walk's playback agree with the look ladder's own numbers for that
rung (bob within 5 %, head pitch within 0.1 degree, tail phase within 0.01 cycle, toe-off within 0.02 rad) [NEW
limits]; (c) four degraded playbacks each fail exactly their term (bob x 0.5; a 3 degree neck sinusoid; the tail
0.2 cycle late; flat feet) and the unchanged playback passes all four [NEW]; (d) every episode of the body's
`R0@10ms` bar records of the set holds every reading of its item; (e) for `60-N-T-10` the re-run Stage 1b 10 ms
turn episodes are trace-identical to their committed records, with both toe-clearance readings reported; for
`75-M-H-10` every 10 ms bar record holds both. If (b) fails, U1's definitions go back to the lead before any
look-set record is made. Check: `stage2/tb/gates/<set>/<body>/g9.json`.

## Q. R2L

**Q1. What is fitted.** [plan B.2; S2-18; owner's allowance of the candidate] Stage 1's tracker (R2, 13 searched
gains, capture margin held) on the bar's tasks, seed 42, here: on set `V6noF5` for `75-M-H-10` and `60-N-T-10`
(current heads), and, labelled a candidate, on set `F25F9cl` for `75-M-H-10` if Q5 allows. `speedfit.py` is not
edited. Check: `stage2/tb/r2l/<set>/<body>/r2lfit/42/cma.json` `settings`.

**Q2. Mix and rotation.** [S2-18] Every evaluation runs five bar episodes at mu 0.9: `walk/v0.4`, a quick turn, the
two-touchdown stop, a nudge at 0.25 x the bound, `walk/nominal` (slots j = 0-4). In generation g: the episode seed
is `tune.training_order(42)[g]`; slot j runs at `k = (g + j) mod 6`; the turn is to the left for even g; the
nudge's direction is forward, left, backward, right for `g mod 4` = 0, 1, 2, 3. Every member of a generation runs
the same five episodes. The table for 30 generations is written before generation 0. Check: `rotation.json`
(its `written_at` before `sessions[0].started_at`); `evals.jsonl`'s delays equal it.

**Q3. Objective and search.** [S1b; NEW terms for the bar's items] `tune.score` (falls first, then tracking and
limit terms) with the item terms of design 3.8; CMA-ES, population 12, sigma0 0.1, from the body's Stage 1 G5
selection; at most 30 generations. Check: `cma.json` `objective`, `start`.

**Q4. Selection.** [S2-18] Candidates: the start and the mean after each third of the generations judged. Each
runs the five kinds on validation seeds 2000-2019 at k = 5. Ranked by `tune.selection_key` on those episodes
alone. Reported and not used: the selected candidate and the start at k = 0, then at k = 2 and k = 3, as far as Q5
reaches. Check: `r2lg5.json` `selection`, `by_delay`.

**Q5. Budget.** [plan section 6: 8 core-hours; NEW split] Counted as the sum of evaluation CPU seconds in
`ledger.json`. Each `V6noF5` fit: 3.2 core-hours (it ends at the generation that would pass it, or at 30) and 0.8
for its selection: 8.0 in all. The `F25F9cl` fit on `75-M-H-10` needs 4.0 more; it runs last and only if the lead
grants them, else it is reported "cut by the pre-registered order". A fit judged before generation 30 says so
(`judged_at_generation`). Check: `ledger.json`; `r2lg5.json` `judged_at_generation`.

**Q6. Rows and reading.** [plan] `R2L/42@<ms>ms` on the set it was fitted on, through the bar's runner; read at
10 ms as a baseline beside R0L and R5 (and at 0, 4 and 6 ms where the plan asks); one seed, so every verdict of
R2L is provisional. Check: the bar records' `controller.gains_info` (path and sha256 of `r2lg5.json`).

## K. Checkpoints, resume, versions

**K1. Boundaries.** [NEW; differs: S1, where each session drew new seeds] At the first rollout start at or past
each multiple of 250,000 steps a checkpoint is saved; then, in every run whether interrupted or not, everything
random is reseeded with `100,000 x seed + 100 x b` (b = that multiple / 250,000), every env is reset, and the
running returns of the reward normalisation are zeroed. A resume loads the newest intact checkpoint and does the
same before its first rollout. Check: `sessions[].boundary_seeds`; `tests/test_stage2_train2.py`.

**K2. Kept checkpoints.** [plan B.4] Each check's (M2) and the newest two; written locally, copied to Drive as
whole checkpoints (three files, then the record). Check: `run.json` `kept`; the files on Drive.

**K3. Kill and resume.** [plan B.3; NEW pass rule] On this machine, with one set of library versions: a run killed
with SIGKILL between two boundaries and resumed ends with every policy tensor, the optimizer's state, the
normalisation statistics and the fingerprint actions **equal** to those of the uninterrupted run of the same
settings (tolerance zero), also when the resumed session uses another worker count. If they are not equal, the
build stops and goes to the lead; no tolerance is set after a difference has been seen without a label. Across
machines or versions nothing is asserted. Check: `stage2/tb/validation/kill_resume.json`.

**K4. What a resume refuses.** [S2-13; S1] Changed settings, env configs or inputs (file hashes, the set's
freeze): always. Changed sources: unless `--allow-source-change`. Changed library versions (Python, numpy, MuJoCo,
scipy, torch without its build label, stable-baselines3, gymnasium, cloudpickle): unless `--allow-version-change`.
A checkpoint that fails its integrity check is skipped with its reason recorded; a resume needs one that passes
and whose fingerprint is reproduced within 1e-5. A flagged session records both sets and the flag, and every later
record of the run carries it. Check: `sessions[]` (`version_change`, `source_change`, `skipped_checkpoints`,
`fingerprint_max_abs_diff`).

**K5. One reference, one path, one queue.** [S2-13] A run keeps one REPO_REF, one RUN_ID, one bundle and one queue
for its whole life; the notebook refuses a RUN_ID that exists without RESUME, and RESUME with another commit,
bundle, queue or budget. Check: `QUEUE.json`; `SESSION_<k>.json`.

**K6. The fingerprint and the loader.** [S2-12; S2-14] Every checkpoint is saved with 256 observations from the
generator seeded `[20261115, observation width, actor width]` and the deterministic float32 actions its policy and
statistics give them. Every load, anywhere, goes through one loader that checks the files' hashes, passes the
schedule objects explicitly and requires the actions reproduced within 1e-5; a checkpoint that fails is not used
and, if it was to be judged, the owner is told. Schedules are instances of module-level classes; an archive holds
no bytecode. Check: `checkpoint_<n>.json`; the rows' `controller` (both version sets, `cross_version`,
`fingerprint_max_abs_diff`); `tests/test_stage2_s2load.py`.

**K7. Fine-tunes.** [plan S2-6] A new job directory from a parent's checkpoint and statistics, the parent's
`log_std`, a constant learning rate equal to the parent's schedule value at the parent's step, 5,000,000 steps,
its own boundaries and checks. Check: `run.json` `parent`.

## Y. The day-one smoke and the budget

On the chosen runtime type, one session, the notebook's `SMOKE` mode, on `75-M-H-10` with the current head.

| Rule | Item | Recorded | Pass |
|---|---|---|---|
| Y1 | setup | the versions against the pins (N2); Python's version | the pins installed; a Python minor other than 3.13 is reported (K6 covers it) |
| Y2 | the machine | vCPUs, RAM, CPU model | reported |
| Y3 | memory | resident memory per worker with 8 envs after one rollout | the total under 80 % of RAM |
| Y4 | the rate at 8 envs | env steps per wall second of an R5 job over its first three rollouts, updates included | Y11 gives at least 15,000,000 steps |
| Y5 | a test clip | an MP4 of 3 s from a monitor trace, rendered in a subprocess with OSMesa | reported (a clip is no stop condition) |
| Y6 | kill and resume on Drive | a tiny job killed after its first checkpoint is on Drive, resumed in a new process | the resume finds an intact checkpoint on Drive, reproduces its fingerprint, refuses nothing and reaches its budget |
| Y7 | an archive that loads here | the smoke's checkpoint, downloaded by the lead | the loader reproduces its fingerprint within 1e-5 in the study's venv |
| Y8 | the inputs rebuilt once | one body's models and arrays rebuilt on Colab, each sha256 against the manifest | reported; if every hash matches, the rebuild is an allowed fallback (N4) |
| Y9 | setup time | seconds from the setup cell to the first training step | reported |

**Y10.** [NEW] The smoke passes when Y1, Y3, Y4, Y6 and Y7 pass. Until it passes no training job is queued. Check:
`runs/<run_id>/smoke/SMOKE.json`; `stage2/tb/validation/colab_archive.json`.

**Y11. The budget (S2-8).** [plan section 6, as stated] Fixed once after the smoke for every run: 40,000,000 env
steps, or, if Y4's rate r cannot reach that in 20 h (r under about 556 env steps/s), `r x 72,000` rounded down to
a multiple of 5,000,000; never under 15,000,000 without the owner. Asked steps are run in whole rollouts. Check:
`QUEUE.json` `budget_steps`; `SMOKE.json` `rate`, `budget`.

## N. The notebook and the bundle

**N1. The four lines.** [plan section 6; NEW text] Exactly:

```
S2 LINE 1/4 commit: <40 hex> (REPO_REF <ref>; checkout clean: yes)
S2 LINE 2/4 inputs: bundle <bundle_id>, manifest <first 12 hex>, <n> of <n> files match their sha256: OK
S2 LINE 3/4 G8a <body> set <set>: zero action = R0 at 0 ms and 10 ms, <n> of <n> episodes identical (<steps> physics steps): PASS
S2 LINE 4/4 first rate <job>: <r> env steps/s (8 envs, <W> workers, <c> vCPUs); <B>M steps in about <h> h
```

Line 2 ending `FAILED (...)` or line 3 ending `FAIL (...)` stops the session before any training. Check: the
notebook's local run (N5) prints them; `STATUS.json` `lines`.

**N2. Pins.** [plan B.4; NEW values] The notebook installs the study venv's versions: mujoco 3.10.0, numpy 2.5.4,
scipy 1.18.1, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0 (the CPU wheel), cloudpickle 3.1.2. Training
processes run with `MUJOCO_GL=disable`, assigned by the entry point itself. Check: `SESSION_<k>.json` `versions`,
`pins_match`; `tests/test_stage2_nb.py::test_pins_equal_the_study_venv`.

**N3. The bundle.** [plan B.4] The model files and the frozen sets' arrays of the bodies and heads of a launch, in
plain tar parts of at most 20 MB, with a committed manifest of every file's and part's sha256. The notebook
verifies every part and file before anything is built; any mismatch stops the session. Check:
`stage2/tb/bundle/<bundle_id>/MANIFEST.json`; line 2.

**N4. Fallbacks if an upload fails.** [NEW] In this order: parts of at most 5 MB; the model parts alone with the
arrays rebuilt on Colab, allowed only after Y8 found every hash equal; with the owner's consent, the parts on a
data branch that is never merged. Nothing is rebuilt or fetched from anywhere else. Check: `SESSION_<k>.json`
`bundle.source`.

**N5. Before the owner's first session.** [plan B.3] A fresh clone of the pushed branch plus the bundle: every
hash verified, every env built, Z1's short form passed for each body and head. The notebook run here end to end on
a tiny budget, a local directory standing in for Drive, with one kill and one resume, printing N1's lines. Check:
`stage2/tb/bundle/<bundle_id>/DRYRUN.json`; `tests/test_stage2_nb.py` (slow) and its log.

## Not run in this round, by name

Periodic walking-turn gaits; head-first turns; the bar with gaze events (the head's "cost of using it") and its
head script; any R5 or R6 run on a set other than V6; a carpet verdict; the hardware levers and their fine-tunes;
third seeds. Each appears in the report as "not run".

## Where this text changes the plan's wording

1. E6: the trainer is a new module; `designlab/stage1/train.py` is not extended (plan B.4).
2. T2: "walking turns at 0.35-1.0 rad/s" are the finite 90 degree turns (plan B.1).
3. E4, W2: the policy's command input is piecewise constant, while the tracking reward keeps Stage 1's target.
4. W3: two of Stage 1's reward terms are read differently.
5. U4: reading 1 is reading 2 for nudge and stop items; a fourth count is printed beside the three readings.
6. M3: S2-10's record prints a count the ruling did not ask for (reported).
7. M5: "from 20M, three checks" is read as three consecutive checks counted from the one at 20M.
8. M6: a local 60 cm run and a tell-the-owner rule the plan does not have.
9. K1: every checkpoint is a boundary that resets the envs (the plan says "kill and resume").
10. O1: 16,384 steps per rollout (design 2 with 8 envs); O4: the filter's rule; N2: the pins' values.
11. Q5: the plan's 8 core-hours cover two fits; the candidate's fit is extra.

## Open for the lead (to be settled before this text is committed)

1. M3: whether "no gain" that rests on the tail servo's sustained limit alone goes to the owner as a no-go, or
   with the reported count beside it (design 7.1).
2. E6: confirm that no extension of `train.py` is wanted.
3. U4 and design 2.3: the reading hook in `bar.py`, before state A or at state B.
4. T1: the weights; T2: whether a set without the turn in place may be trained on.
5. M6: seed 41, 5,000,000 steps and the tell-the-owner rule; its 4.8 core-hours.
6. O3: one learning rate for all bodies and heads (the smallest), against one per body.
7. O4: the filter's rule and its 0.9.
8. Q5: 4.0 core-hours more for the candidate's fit, or cut.
9. W2: the new weights.
10. Y10: which smoke items must pass.
