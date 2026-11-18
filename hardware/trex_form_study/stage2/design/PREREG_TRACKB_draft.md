# Stage 2, Track B: rules fixed before the runs (R5, R6, R2L, the gates, the monitor, the notebook) - DRAFT 2

**Status (15 Nov 2026): the architect's second draft, after three independent reviews of the design; for the lead;
not committed.** It is written before any look set is frozen, before any Stage 2 closed-loop episode on a look set,
before any R2L fit and before any training run. Simulation only; it changes no body and adopts nothing without the
owner (P24). It is self-contained: every formula, scale and number a rule needs is in this text (the design,
`ROUND_C_DESIGN.md`, explains them and is not committed). Style and tags as `STAGE2_75_PLAN.md`. Paths are relative
to `hardware/trex_form_study/`. A rule changed after a result is labelled so. To be reviewed independently before its
commit as `stage2/tb/PREREG_TRACKB.md`. Check: the commit that adds that file is an ancestor of the commit in every
Track B record's `sessions[].commit` and of every look set's `freezes[].commit` that a Track B record uses.

**Rules marked [RULING NEEDED] change the plan's wording and are not final until the lead has ruled** (list at the
end). A rule the lead strikes is deleted from the committed text with the consequence this draft names.

## What had been run when these rules were drafted

- Everything listed in `STAGE2_75_PLAN.md` under the same heading.
- Scratch experiments of 15 Nov on Stage 1's own run directory (`stage1/runs/2026-11-08`: stock references, no
  Stage 2 set), by the architect and by the three reviewers of the design, on a stub env (Stage 1's env with the
  open-loop base in place of the tracker; Stage 1's observation and reward). 4-6 episodes per setting. They are
  not records and none may be cited as a result.

| Scratch experiment | Numbers (not records) | Bears on |
|---|---|---|
| The panels' open-loop row against a tracker in R2 mode with no feedback term and no ankle cap, stepped as an env steps it, on two separately built plants; `75-M-H-10`, 6 s, one seed | state digests equal at 3,000 of 3,000 physics steps at 0 ms and at 10 ms | Z1 |
| The stub env on `75-M-H-10`, one process | 411-460 env steps/s on hard floor, 352-422 on robot_floor_carpet | O1, Y11 |
| The base under action noise, 10 Hz filter, k = 5 | `75-M-H-10`: sigma 0 and 0.01 up 6 of 6; 0.02 up 1 of 6; 0.05 and 0.1 none (3.8 s, 3.3 s). `60-N-T-10`: sigma 0 up 1 of 6; 0.1 none (2.6 s). At equal filtered action RMS a 20 Hz filter survives 0.98 x as long as 10 Hz; at equal sigma 0.85 x | O2, O3, O4, M6 |
| PPO on the stub, sigma 0.1, 5 epochs, batch 256 | with a 16,384-step rollout lr 1e-4 ran 320 then 206 of 320 minibatch steps, 3e-5 all; at lower sigma far fewer run | O1, O3 |
| The zero action on `60-N-T-10`, 5 episodes per task | walk up in 2 of 5 at 0 ms and at 10 ms; brisk walk and walking turn 0 of 5 at both; turn in place and stand 5 of 5 | M6 |
| A clock held 25 % slow against the zero action | costs 1-7 % of Stage 1's reward per step; on `60-N-T-10` it survives longer | W5, M3 |
| The zero action's drift on `75-M-H-10`, 5 episodes | 0.08-0.22 m from the line in 10 s (limit 0.25 m), heading within about 1 degree | E3, W5 |
| A start from a frame inside the nominal stride with the delay queue filled from the reference; `75-M-H-10` | up 8 s from 6 frames at 0 and at 10 ms on hard floor; stance-foot load 0.8-1.4 body weights at reset; falls after 2.4-3.6 s on robot_floor_carpet | T3 |
| The leg-gap reading on `60-N-T-10` | the smallest gap is a fixed bracket pair at 4.0 mm in every pose | W4 |
| A body standing at home while the reference walks, under the first draft's reward | 0.39 of 1.25 per step | E2 |
| An archive saved with module-level schedule objects; a dummy env, 3 PPO updates | no bytecode member; fingerprint reproduced with difference 0.0; 70 of 120 minibatch steps ran | K9, O3 |
| A resume on a stub env with SB3 2.9.2 | loading with the env and letting SB3 reset: every tensor differs from the uninterrupted run; with `force_reset=False` and the boundary at the first rollout start: bit-identical. An env after 7 randomised episodes equals a fresh one after a seeded reset when its trackers were built on the nominal model | K1, K3 |

No constant below was set on a Stage 2 closed-loop result. Constants marked "(scratch)" lean on the table.

## In brief, for the owner

- **What is trained.** One policy per body, head and seed ("R5"): every 20 ms it nudges the servo targets of the
  reference walk (by at most 0.35 rad on the legs and 0.15 rad on the head and tail) and the walk's clock (by at
  most 25 %). With no nudge it is the reference played open loop. A second kind ("R6", one per body) sets the
  targets alone, as a comparison; it is expected to learn only to stand. A balance controller refitted with delay
  in its mix ("R2L") is fitted here as a baseline.
- **Two checks before any training.** The policy's "do nothing" must reproduce the open-loop row of the agility
  bar exactly, step for step, at 0 and at 10 ms of delay. And the new readings (toe clearance, the look bands) must
  read as designed on the references themselves.
- **One go or no-go before Colab.** 2 million steps here on the 75 cm body; Colab starts only if the policy then
  does better than the open-loop row on 18 fixed episodes at 10 ms. Otherwise you are told first, with a table of
  what failed.
- **Exploration starts rough.** The plan's starting noise knocks the open-loop walk over in about 3 seconds, on
  both sizes. Half of the training episodes therefore start in the middle of a task. This is expected.
- **The 60 cm body is the hard case**: its open-loop walk falls within seconds. A longer local run (5 million
  steps) is made on it first; if its slow walk does not stay up in 2 of 3 episodes by then, you are told before any
  60 cm Colab hours are spent, with the remedy the run's record points to.
- **During a Colab run** the same 18 episodes are run every 2.5 million steps; they decide only when a run stops.
  Verdicts come later, from the 20-episode panels here.
- **What may come back to you.** A failed go or no-go; a failed 60 cm local run; a Colab rate too low for 15
  million steps in 20 hours; uploads of the input files that fail (then a fallback you perform or approve).

## Words, names and tags

As `STAGE2_75_PLAN.md`, with these additions. R5, R6, R0L, R2L are controllers (the plan's B.1-B.2). "The bar",
"the panels", "reading 1, 2, 3" as there. k is a servo delay in physics steps of 2 ms. An "env step" is 20 ms. A
"check" is a monitor run (M1). nj is a body's number of hinges (14; 16 on `-X`); H its hip height; "rate" the
commanded clock 1.05; c the reference clock. This text's rules are E1-E6, T1-T3, W1-W5, J1-J5, O1-O4, M1-M6 and
M3b, U1-U6, Z1-Z2, Q1-Q6, K1-K9, Y1-Y11, N1-N5 and X1; outside a rule's head "S1" is still Stage 1's report and
"G1"-"G9" the gates. Tags: **[plan]**, **[S1]**, **[NEW]**, **[differs: plan]**, **[differs: S1]** as in
`STAGE2_75_PLAN.md`; **[S2-n]** an amendment of the plan's section 9; **[differs: C8]** a departure from
`STAGE2_75_PLAN.md` rule C8; **(scratch)** a constant leaning on the table above. Every rule ends with "Check:".

## E. The controllers

**E1. R5.** [plan S2-3] `control.Tracker` in R2 mode with `terms=()`, `ankle_cap=()`, at the commanded clock 1.05,
on a frozen set's references; every 20 ms the policy's clipped, low-passed action a sets joint offsets
`a[:nj] x scale` (0.35 rad on leg joints, 0.15 rad on every other joint) added to the reference's servo angles, and
the clock factor `1 + 0.25 a[nj]`; rates and feedforward stay the reference's. The offsets are not clipped to the
joint ranges (W5's `range` term prices them). A zero action with the head neutral sends R0L's targets exactly.
Check: `run.json` `settings.env_configs[].action_leg_rad`, `action_other_rad`, `action_rate`; gate Z1.

**E2. R6.** [plan B.2] Joint targets `clip(home + a x span, range)` with Stage 1's span rule, no feedforward, its
own estimator; R5's inputs, commands, reward, starts, randomisation, settings and budget; one seed (42) per body
with the current head. At a mid-task start its action history is set to the actions that give the reference's
targets. **Expected outcome, stated in advance:** Stage 1's policy-only controller learned on every body only to
stand still, and under this reward standing earns about a third of the positive weight (scratch); R6 standing or
shuffling is not news, and its G8b reading (M4) can pass by standing. Check: `run.json` `mode`, `settings`.

**E3. Inputs.** [plan; NEW widths; one input beyond the plan's list: RULING NEEDED] The actor's blocks, in this
order, with their first and last-plus-one index for 14 and for 16 joints:

| Block | 14 joints | 16 joints | Content |
|---|---|---|---|
| encoder angles, rates | 0:14, 14:28 | 0:16, 16:32 | the estimator's, plant joint order |
| gravity direction, gyro | 28:31, 31:34 | 32:35, 35:38 | |
| foot loads | 34:36 | 38:40 | over the nominal weight |
| gait phase | 36:38 | 40:42 | sine and cosine |
| reference support | 38:40 | 42:44 | |
| command | 40:42 | 44:46 | E4 |
| heading error | 42:43 | 46:47 | the estimator's yaw about home less the commanded heading (E4), over 0.5 rad, clipped to +-2 **[differs: plan B.1's input list]** |
| the last three actions | 43:88 | 47:98 | clipped, unfiltered, newest first |
| head command (`-X`) | - | 98:102 | E5 |

Actor 88 and 102 inputs. The critic also sees: root velocity (3), height (1), roll and pitch (2), DCM error (2),
joint errors (nj), heading error and friction (2), floor (2), push (3), plant factors (5: mass, stiffness, delay,
joint damping, joint friction), foot errors (6), the reference's contact 24 ms earlier (2): 130 and 146 in all.
Actions: 15 and 17. If the lead strikes the heading error: 87 and 101 (129 and 145), W5's `heading` term is struck
with it, and this rule reads: "R5 has no heading loop; the bar's `line` and `heading` readings are open-loop
readings and a failure there does not mean 'not learned'." Check: `run.json` `layout`; `checkpoint_<n>.json`
`layout`.

**E4. The command and the commanded heading.** [NEW; differs: S1, whose input was the reference's own speed] The
input is `(v / 0.5 m/s, w / 1.0 rad/s)` times 1.05, piecewise constant on the reference clock: the gait's speed and
yaw rate in a periodic gait; the next gait's from the first frame of a start or change; zero from the first frame
of a stop; in a finite turn of angle A at rate w0 the walk's speed and `w0` for `A / |w0|` seconds of reference
time from the turn's first frame, then zero. The commanded heading is the integral of w over reference time.
Nothing before a segment's first frame depends on that segment. Check: `tests/test_stage2_r5lib.py`; the env
description's `command_rule`.

**E5. The head command on `-X`, and the bar's head script.** [plan; NEW] `[yaw now, jaw now, yaw goal, jaw goal] /
0.6 rad`; the commanded yaw and jaw angles are added to the reference's neck-yaw and jaw targets before the
policy's offsets. Head neutral is all zero. The bar's "cost of using the head" (plan section 5) is run under the
row `R5G/<seed>` with head script H1: for episode seed s, from the generator `SeedSequence([s, 77])`, a train of
gaze shifts for the whole episode (yaw target uniform on +-0.5 rad, minimum-jerk over 0.8-1.2 s, held 2-6 s) and
one jaw event at a time uniform in the episode's middle half (open to 0.3-0.6 rad over 0.4-0.6 s, held 1.0-1.5 s,
closed over the same time); both always present. Head-first turns (F11) are deferred by the lead's ruling and are
not run [differs: plan section 5, labelled]. Check: the env description's `head_rule`; the rows' `controller.head`;
Z1 on the `-X` bodies with the head neutral.

**E6. Where the code lives.** [differs: plan B.4, which has the notebook run `designlab/stage1/train.py`, extended]
No file under `designlab/stage1/` and no module of the bar (`bar.py`, `readings.py`, `setio.py`, `s2io.py`) is
edited: the env, the trainer, the fit, the look reading and the rows are new modules in `designlab/stage2/`. S2-17
as ruled allows B.4's extension; this rule does not use it. Check: `git diff --stat <state A>.. --
designlab/stage1 designlab/stage2/bar.py designlab/stage2/readings.py designlab/stage2/setio.py
designlab/stage2/s2io.py` is empty; `stage2/regress/regress.json` is current.

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

Episodes last at most 30 s. Nudges are no task: pushes are drawn on every task (J3). Check: `settings.tasks` (the
resolved table: names, plans, weights) equals this table less the absent tasks.

**T2. Absent references; what is not trained.** [NEW; plan B.1 read] A task whose reference the set does not hold
(`available: false`, or not built) is dropped, the remaining weights renormalised, and the env description lists it
under `absent` with the set's reason. A freeze of the set made after a run's first session changes no task table of
that run (K4). "Walking turns at 0.35-1.0 rad/s" are the finite 90 degree turns; no periodic walking-turn gait is
trained **[differs: plan B.1's list, as read; RULING NEEDED with the no-loss reading below]**. Also not trained,
labelled: Stage 1's `carpet_walk` task (two of the eight envs run every task on robot_floor_carpet instead, J5);
stops from walks other than 0.40 m/s (the bar and the ladders hold none). If the turn in place is absent from a
set, the plan's B.1 list is not met for that body and the report says so.
*No loss for R5 and R6* [differs: plan section 5 "No loss"; RULING NEEDED]: Stage 1's twelve gated hard-floor
conditions are defined on Stage 1's stock references, and a policy is judged on the references it imitates, so
they cannot be run on the pseudo-set S1. They are run as look-set counterparts: stand; `walk/nominal`, `walk/v0.35`, `walk/v0.5`; the
finite 0.35 rad/s turn left and right in place of the periodic walking turn; the turn in place left and right;
start-stop; the standing pushes; at 0 ms, Stage 1's committed counts printed beside them. Check: the env
description's `absent`; `set.json` `not_available`.

**T3. Starts.** [plan] Half of the episodes of every task but `stand` start mid-task: the reference time is
uniform over the task's non-hold segments between 0.2 s after the first non-hold frame and 2.0 s x 1.05 before the
plan's end, snapped to a file frame (never inside a hold); the plant is set to that frame (rates x 1.05) with Stage
1's start noise (0.01 rad on the joints, 0.02 m/s on the root's velocity), the estimator is reset from that state,
and the delay queue holds the k commands the zero action would have given before it. The others start from the
settled stance with Stage 1's start noise. (scratch) Check: `settings.env_configs[].mid_start == 0.5`;
`tests/test_stage2_r5env.py` (the mid-start tests).

## W. Reward

**W1. Form.** [plan] `r = sum of the positive terms x their weights - sum of the penalties x their weights`, each
term in [0, 1], read on the true state at the end of every env step (c the reference clock then, c0 at its start).
A zero action on a kinematic playback scores every positive term 1 and every penalty 0. Check: the env
description's `reward_rule`; `tests/test_stage2_r5core.py`.

**W2. Terms, weights, scales and their source.** [S1 for Stage 1's terms; NEW for the others]

| Term | Weight | Formula | Source |
|---|---|---|---|
| joint imitation (+) | 0.55 | `exp(-mean_j ((q_j - q_ref_j(c)) / 0.1 rad)^2)` over all hinges | Stage 1 |
| speed and yaw-rate tracking (+) | 0.35 | Stage 1's Gaussian tracking reward of the CoM's speed along the previous heading and the trunk's yaw rate over the step, against the reference's own at c0 times 1.05 | Stage 1 |
| foot imitation (+) | 0.15 | `exp(-mean_f |r_f - r_ref_f(c)|^2 / (0.10 H)^2)`, r the foot's position less the root's in the root's heading frame | new (plan) |
| contact imitation (+) | 0.10 | the share, over both feet and the step's 10 physics steps, of steps in which the foot's contact (load above 2 % of the weight) equals the reference's support at the clock or 24 ms x 1.05 earlier | new (plan) |
| head attitude (+) | 0.10 | `exp(-((roll - roll_ref)^2 + (pitch - pitch_ref)^2) / (3 deg)^2)` of the head in the world | new (plan) |
| duty, slip, impact, tilt | 0.02 each | Stage 1's (W3 for the duty band) | Stage 1 |
| flight | 0.05 | Stage 1's | Stage 1 |
| clearance | 0.02 | W3 | Stage 1, read as the bar reads |
| action rate | 0.03 | `min(1, mean_i (a_i - a_prev_i)^2 / 0.25)` on the clipped actions | new (plan) |
| touchdown speed | 0.02 | at a foot's touchdown: `min(1, max(0, v_down - 0.15 m/s) / 0.30 m/s)` | new (the plan's "impact") |
| servo | 0.05 | per neck, tail, neck-yaw and jaw drive: a bias-corrected exponential mean of the squared force, time constant 1.0 s; `min(1, max_d max(0, RMS_d / sustained_d - 1) / 0.5)` | new (plan) |
| interference | 0.10 | W4 | new (plan) |
| progress | 0.10 | W5 | new, beyond the plan's list |
| heading | 0.05 | W5 | new, beyond the plan's list |
| range | 0.05 | W5 | new, beyond the plan's list |

The new weights were set on 15 Nov on the recipe of `stage2/research/research_look.md` 3.3, before any run, so that
no new term outweighs Stage 1's joint imitation. They are the same for every body, head, seed and mode and are not
changed by M4's fix. Check: `settings.env_configs[].reward` and the env description's `reward_scales` in every
`run.json` equal this table.

**W3. Two of Stage 1's terms read differently.** [differs: S1] `clearance` is the larger of (a) for a foot in the
air between 25 % and 75 % of the planned swing time since its **own** lift-off and lower than 0.04 H: the shortfall
over 0.04 H; (b) 1 if in any physics step of the env step a foot is loaded while the reference has that foot between
25 % and 75 % of its swing both at the clock and 24 ms x 1.05 earlier (a dragged foot or a stub step). `duty`'s
band is 0.55-0.72 while the commanded target speed is below 0.30 m/s (owner decision 2), else 0.55-0.65. Check: the
env description's `reward_rule`.

**W4. Leg-to-leg interference.** [NEW; `LESSONS_LEARNED.md` 2.8] Read at the end of every env step with
`legs.LegGap` (every left-leg against right-leg pair of the screened geometry, the reading of the bar's turn item).
Penalty: `clip((3 mm - g) / 3 mm, 0, 1)` with g the smallest distance over the **moving** pairs (pairs that stay at
one distance in every pose, such as the 60 cm bodies' hip brackets at 4.0 mm, carry no penalty and are listed in the
env description). Termination: the smallest distance over **all** pairs below zero at two consecutive env-step
ends. A one-pose contact is therefore penalised and not terminated, while the bar's turn item fails on one pose:
the training signal is weaker than the bar here, by choice. A leg against the trunk or the tail is not read.
(scratch) Check: `settings.env_configs[].interference == "terminate"`; the env description's `fixed_pairs`;
`info["interference"]`.

**W5. Three terms beyond plan B.1's list.** [differs: plan B.1; RULING NEEDED, each separately] (scratch)
`progress`: only inside a periodic walking segment, from one stride after its first frame, with a positive
command: `min(1, max(0, 1 - v_bar / (1.05 v_cmd)) / 0.10)`, v_bar the bias-corrected exponential mean (time
constant: the stride period over 1.05) of the CoM's speed along the commanded heading (the bar's walk item fails
4.8 % short of the commanded distance; Stage 1's tracking term charges 0.2 % of the reward for 5 %). `heading`:
`clip((|yaw - yaw_home - commanded heading(c)| - 3 deg) / 17 deg, 0, 1)` on the true yaw, zero in a hold (struck
with E3's input). `range`: `min(1, max_j max(0, required_j - margin_j) / 0.02 rad)`, margin the joint's distance to
its nearer limit; required, per joint: in a task without a finite turn `min(0.10 rad, the reference's own smallest
margin)` (0.10 rad is Stage 1's walk rule, `tune.WALK_RANGE_RAD`); in a task with one, decision 14's `max(min(0.10
rad, the reference's own smallest margin - 0.02 rad), 0.001 rad)`, as the bar computes it. The reference itself
therefore scores zero. A struck term's weight is not redistributed. Check: `settings.env_configs[].reward`.

## J. Randomisation

**J1. Delay.** [S2-9] k uniform on the integers 0 to 12 per episode. Check: `env_configs[].latency_steps == [0, 12]`.

**J2. The plant.** [plan] Per episode: servo stiffness x 0.6-1.2; sliding friction 0.4-1.2; trunk mass and inertia
x 0.9-1.1 (`env.MASS_RULE`); joint damping x 0.5-1.5 and joint friction-loss x 0.5-1.5, one factor each for all
hinges [NEW precision]; IMU noise scale 0.5-1.5 [S1]. The controller's model and feedforward stay nominal: every
tracker is built on the nominal model before any episode. Check: `env_configs[]`;
`tests/test_stage2_r5env.py::test_factors_are_applied_and_restored`, `::test_trackers_are_built_nominal`.

**J3. Pushes.** [plan; S1 for the pulse] With probability 0.5 in an episode longer than 4 s: one 0.1 s pulse at
the trunk's CoM at a time uniform in `[2 s, end - 2 s]`, direction uniform, `dv = u x` the body's Stage 0 forward
bound (its own envelope screen; no default value), u uniform on 0-0.5 if the reference walks at that time and on
0-1.0 if it stands. Check: `env_configs[].push_walking`, `push_standing`; the env description's `push_bound_m_s`.

**J4. Gaze and jaw events on `-X`.** [plan; NEW numbers, from `research_look.md` 5.1] With probability 0.5 a train
of gaze shifts for the whole episode: yaw target uniform on +-0.5 rad, minimum-jerk over 0.8-1.2 s, held 2-6 s.
With probability 0.25 one jaw event: open to 0.3-0.6 rad over 0.4-0.6 s, held 1.0-1.5 s, closed over the same
time. Check: `env_configs[].gaze_prob`, `jaw_prob`; the env description's `head_rule`.

**J5. Floors and seeds.** [plan; S1] Env i of 8 runs on hard floor for `i % 4` in 0-2 and on robot_floor_carpet
for `i % 4 == 3`; no pile. The task, start, factors, push and head events come from the env's own generator; the
start's and the sensors' noise from the episode seed (`env.EPISODE_SEED_RULE`); env i resets at boundary b of PPO
seed s with seed `100,000 s + 100 b + i` (K1). Check: `sessions[].floors`, `boundary_seeds`.

## O. PPO and the probes

**O1. Settings.** [S1 design 2; NEW for 8 envs] 8 envs in every run and session; `n_steps` 1024 (a rollout is
8,192 steps, Stage 1's size; 160 minibatch steps per update) [NEW: the first draft had 2048]; batch 256; 5 epochs;
gamma 0.99; lambda 0.95; clip 0.2; `target_kl` 0.03; networks (256, 256) with the privileged critic; observation
and reward normalisation clipped at 10 and 50; one torch thread; CPU. The number of worker processes is the
runtime's and changes no result. Every update's minibatch steps run, of the 160 possible, are recorded (lesson
16). Check: `settings.ppo`; `updates.jsonl`; `tests/test_stage2_vecpool.py`.

**O2. Exploration.** [plan] Initial sigma 0.1 of each action's range (`log_std_init = ln 0.1`). Known in advance
(scratch): the open-loop base tolerates about 0.01-0.02, so early training episodes on both sizes end in a fall
after about 3 s; mid-task starts (T3) are what shows the policy the whole task. The sigma probe is reported and
decides nothing: on each current-head body whose V6 set is frozen, the stance-start `walk_nominal` and `walk_v0.4`
tasks at k = 5 on the nominal plant, 10 episodes each (episode seeds 7000-7009), actions `clip(sigma z)` with the
same z for every sigma, sigma in 0, 0.01, 0.025, 0.05, 0.1, 0.2: survivals and mean survival time per sigma. Check:
`settings.log_std_init`; `stage2/tb/validation/sigma_probe.json`.

**O3. The learning rate, and updates cut by the KL limit.** [plan; decision 24's probe; NEW second sigma] For each
body and head whose V6 set is frozen, for each of 1e-4, 3e-5, 1e-5 from the largest down: a fresh R5 model, seed
42, training's randomisation, the rate held constant, 6 updates, once at sigma 0.1 and once with `log_std` set to
ln 0.03. A rate *passes on a body* when all 160 minibatch steps ran in each of the 6 updates at both sigmas; the
probe of a body stops at the first rate that passes. The value used is the smallest of the bodies' passing rates
(one value for every run); the schedule is linear from it to 0.3 x it over the run's budget. If none passes on a
body, nothing trains and it goes to the lead. A body or head whose set is frozen after S2-4 is probed before its
first run; if that lowers the value, the Colab runs use the lower one, S2-4 is not repeated, and the record says
so. During a run: if the median share of minibatch steps run over a check interval is under one half, STATUS.json
and `run.json` carry `flags.kl_cut` and the lead is told; it stops nothing. Check:
`stage2/tb/validation/lr_probe.json`; `settings.lr`; `run.json` `flags`.

**O4. The action filter.** [plan: "re-chosen in validation"; NEW rule] Candidates: a first-order low-pass at 10 Hz
(Stage 1's) and at 20 Hz. Measured once, before S2-4, on the current-head bodies whose V6 set is frozen (at least
`75-M-H-10`): O2's episodes with the noise scaled so that the **filtered** action has the same RMS under both
cut-offs (sigma 0.1 at 10 Hz; sigma 0.1 x sqrt((a10 / (2 - a10)) / (a20 / (2 - a20))) = 0.081 at 20 Hz, a = 1 -
exp(-2 pi f x 0.02 s)). 20 Hz is used if its mean survival time, pooled over those episodes, is at least 0.9 x that
at 10 Hz; otherwise 10 Hz. One value for every run of the round; it is not measured again. What this shows: whether
the faster filter costs the base stability at equal action amplitude. What it cannot show: which filter learns
better. (scratch) Check: `stage2/tb/validation/filter.json`; `env_configs[].filter_hz`.

## M. The monitor and the stop rules

**M1. Episodes and outcomes.** [plan section 6; S2-10; NEW seeds and outcomes] 18 episodes: the bar's conditions
`walk/nominal`, `walk/v0.4`, `turn/q90_left`, `stop/two_step`, `nudge/forward`, `nudge/left` of the run's set at mu
0.9, three episodes each, **every episode on its own seed**: condition i (0-5 in that order), repetition r (0-2):
seed `6001 + 3 i + r` (6001-6018; none is a panel, fit or validation seed), at 10 ms, as the bar builds and judges
them. Each episode has one outcome: *read*; *fell*; *unread* (an episode of a policy row, `R5`, `R6` or `R5G`,
that did not fall and that the bar could not read: not succeeding in any reading, recorded with the reference time
reached and its reason, `clock_behind` when its mean clock factor is below 1, so that its command or judged window
was not reached, else `unreadable`) **[differs: C8]**; *error* (an infrastructure fault, or an unreadable episode
of a row without a policy, for which C8 stands). The monitor never runs an episode twice. A
condition without a reference is "not run" and counts as not succeeding and as zero survival for both rows. Each
record holds the failures per term, the mean clock factor and the number of distinct start seeds. Check:
`monitor2.CONDITIONS`, `SEEDS` (committed before any run); each monitor record's `episodes[].outcome`.

**M2. Rows.** [NEW] The checkpoint's row `R5/<seed>@10ms` (or `R6/...`), loaded from its saved files through the
loader of K9, and once per run `R0L@10ms` on the same 18 episodes. A check runs at the first rollout start at or
past each multiple of 2,500,000 steps and on a run's final checkpoint; S2-4 checks every 500,000 steps (its earlier
checks are reported, S2-10). Check: `monitor/monitor_<n>.json` (it names its checkpoint's file hashes),
`monitor/r0l.json`.

**M3. A gain; S2-10's reading.** [plan section 6; S2-10 as ruled; NEW precision; one departure] N is the number of
the 18 episodes succeeding with the look (U4). S is the mean over the 18 of the **credited survival**: the survival
time (the fall time, or the episode's length) times `min(1, reference time covered / (1.05 x survival time))`,
which is the survival time itself for a row that does not slow its clock, R0L exactly **[differs: plan section 6's
"mean survival"; RULING NEEDED]** (scratch: a slowed clock can otherwise buy survival). A check shows a gain when
`N >= N* + 2` or `S >= S* + 1.0 s`, N* and S* each the best value over the run's earlier checks, taken separately
[NEW precision], and for the first check R0L's. S2-4 (2,000,000 steps asked, whole rollouts: 2,007,040;
`75-M-H-10`, current head, seed 42, training's delay draw) is read on its final checkpoint against R0L by this
rule; its earlier checks are reported and decide nothing. R0L's 18 episodes are run and committed before S2-4
starts. A gain lets Colab start; no gain goes to the owner first, with the per-term failure table of both rows.
The record also prints, beside N: the counts in readings 2 and 3 and the count succeeding but for A1 with the look
bands kept [NEW, reported]. Check: `stage2/tb/runs/s2-4/S2_10.json` (both rows' 18 episodes, N, S, the reported
counts, `failures_by_term`, `mean_clock_factor`).

**M3b. When the survival route is closed.** [differs: S2-10; RULING NEEDED; struck unless the lead rules it in
before S2-4 starts] If R0L's S is within 1.0 s of the mean planned length of the 18 episodes, a gain is also: at
least 2 more of the 18 succeeding in reading 3 with the look bands kept (every limit but a neck or tail servo's
sustained planning limit, and the four bands) than R0L. Why: on `75-M-H-10` R0L is expected to stay up (so S
cannot rise by 1 s) and the tail servo's sustained limit failed in 20 of 20 Stage 1 episodes at 0.35 m/s and
above, so N may be 0 for a reason no policy changes much. Check: `S2_10.json` `survival_route_closed`, `rule`.

**M4. G8b and its one fix.** [plan section 6] At the first check at or past 5,000,000 steps: `walk/nominal` stays
up in at least 2 of its 3 episodes. Otherwise, once, inside the running job: the training weights of the tasks of
every monitor condition with fewer than 2 of 3 up are doubled, by this table: `walk/nominal`: `walk_nominal`;
`walk/v0.4`, `nudge/forward`, `nudge/left`: `walk_v0.4`; `turn/q90_left`: `turn_q90_left` and `turn_q90_right`;
`stop/two_step`: `stop` and `start_stop`. The reward's weights do not change. At the first check at or past the
fix + 5,000,000 steps the same reading; a second failure stops the run. The decision is a function of the monitor
records alone, so a killed and resumed run takes the same one (K5). Check: `run.json` `gates.g8b`, `fix`,
`gates.g8b_after_fix`, `stopped`.

**M5. Stopping without a gain; judging; the cap.** [plan section 6; NEW precision] Counting from the check at
20,000,000 steps: the run stops at the third consecutive check without a gain (25,000,000 at the earliest). No
best-model selection; a body and seed's two heads are compared at the step both reached, each run's upper end at
its own last checkpoint; the 220 runtime-hour and 8-day cap and the order of cuts are the plan's; runtime-hours
are tallied per session from its start to its last STATUS.json. Check: `run.json` `monitor[]`, `stopped`;
`stage2/tb/hours.json`.

**M6. A local 60 cm run before 60 cm Colab hours.** [NEW; `LESSONS_LEARNED.md` 5.2 item 7] R5 on `60-N-T-10`,
current head, seed 41 (no judged seed), 5,000,000 steps asked (5,005,312), here, with the rules above. It is read
at its 5M check by M4's first sentence. Reported at each of its checks, to separate the causes of a failure: the
same 18 episodes at 0 ms for the policy and for R0L; sigma; the share of minibatch steps run; S against R0L. It
starts and stops no Colab run by itself; if the reading fails, the owner is told before any 60 cm job is queued,
with the remedy the record points to: the nominal walk up at 0 ms but not at 10 ms: a delay curriculum; up at
neither with sigma near 0.1 or most minibatch steps cut: exploration (a lower starting sigma or rate); up at
neither with a small sigma: a feedback base, or launching under M4. Its checkpoint is never judged on the panels.
(scratch) Check: `stage2/tb/runs/local-60/jobs/R5-60-N-T-10-s41/run.json`.

## U. The look reading (reading 1)

**U1. The four measures.** [plan section 5; `research_look.md` 4.2; NEW definitions] From 50 Hz poses of an
episode: *bob*, the mean over the robot's own steps (touchdown to next touchdown of either foot, wholly inside the
window) of the range of the hip-midpoint's height; *head*, the RMS of the head's world pitch about its mean;
*tail phase*, the phase of the least-squares sinusoid of the tail-yaw angle on the robot's own phase (zero at each
of its left lift-offs); *toe-off*, the mean over its lift-offs of the lifting foot's pitch at that instant.
Contact is the 2 % load rule with 20 ms debounce. Check: each companion episode's `look.own`.

**U2. Window.** [NEW] A walk item: its judged last 10 s. A turn item: the 3 s after `t_c + at_s`. Nudge and stop
items: not read. Check: `look.window`.

**U3. Bands.** [plan section 5] Against the reference's own values (U5): bob within +-30 %; head pitch at most
1.5 degrees RMS (not relative); tail phase within +-0.1 cycle; toe-off at least 70 %. The tail term is not read
when the reference's tail amplitude in the window is under 0.02 rad, the toe-off term when the reference's toe-off
is under 0.02 rad [NEW]; both then pass, labelled. Check: `look.terms`, `look.not_read`.

**U4. The readings on a look set.** [plan] Reading 1 ("shown with the look"), the headline for walks and turns:
the bar's `succeeded` and all four terms. For nudge and stop items reading 1 is reading 2, labelled [NEW precision:
the plan names walks and turns]. Readings 2 and 3 as `STAGE2_75_PLAN.md` C2. Printed beside and never a verdict:
reading 3 with the bands kept. Reading 1 counts on the sets V6 and V6noF5; on any other set the numbers are
recorded and its count is null. It is held in a companion record beside each bar record (U6); the bar's own
record keeps `counts_look: null`. Check: the companion's `counts_look`, `counts`, `counts_but_A1`, `reading_1`.

**U5. The reference's values; too few steps.** [NEW; differs: C8 in its second sentence] The same functions on a
kinematic playback of the episode's own track over the window's reference times at the nominal commanded clock.
When that playback has at least 4 steps and 4 lift-offs in the window and the robot has fewer, the bob, tail and
toe-off terms **fail** (labelled `too_few_steps`); it is not an error and the episode is not run again. When the
playback itself has fewer, it is an error of the definition, which gate Z2 must have caught. Check:
`look.reference`, `look.too_few_steps`.

**U6. The companion record.** [NEW] For every bar record of a Track B row (`R0L`, `R2L-<set>/<seed>`, `R5/<seed>`,
`R6/<seed>`, `R5G/<seed>`): one record that names the bar record by path, sha256 and key hash and holds, per
episode, the outcome (M1's four), the look numbers, both sides' values, the terms, the clock factor, and the three
readings' counts by the bar's own arithmetic, with an `unread` episode counted as survived and not succeeded
in each **[differs: C8]**; for a policy row also both version sets, `cross_version`, the largest fingerprint
difference and the tolerance (K9). The wrapper that makes it changes nothing an episode executes: Check:
`tests/test_stage2_lookbar.py` (`R0L` equals `R0` digest for digest and record for record);
`<run>/<set>/<body>/<block>/look/<row>/<condition>.json`.

## Z. Gates

**Z1. G8a.** [plan B.3; NEW alignment and pass rule] For a set and a body or head: the R5 env with the zero action
(head neutral, nominal plant, the condition's friction, stance start with the row's start seed and Stage 1's start
noise, the condition's push, the delay through the bar's own wrapper, interference termination off) against the
bar's row `R0@0ms` and `R0@10ms`, on the eight gated mu 0.9 conditions the set has references for,
`walk/v0.4_mu0.54` and `walk/nominal`, panel seeds 3042 and 3043 (40 episodes). The row runs `round(seconds /
2 ms)` physics steps; the env runs `ceil` of a tenth of that many env steps, so it covers them all. Compared:
sha256 of `qpos`, `qvel`, `ctrl`, `sensordata` after every physics step the row ran. Same end: if the row fell at
physics step j, the env ends in the env step holding j; if the row did not fall, the env does not end in any env
step lying wholly inside the row's steps (the env's last, partial step is not read). **Pass: every episode
identical.** Any other outcome: no run of that body and head. The trainer repeats one condition (`walk/v0.4`, seed
3042, both delays) before every new run, here and on Colab; that is the notebook's line 3. Check:
`stage2/tb/gates/<set>/<body>/g8a.json` (`identical == of`; `compared_steps == reference_steps` in every episode);
`run.json` `gates.g8a`.

**Z2. G9.** [plan B.3; S2-11 as ruled; NEW order and limits] Two records per body and head.
*G9-ref*, kinematic only; it must pass before the first closed-loop episode on that set and body: (a) every frozen
reference played kinematically at the commanded clock passes its own item's readings, the quick turn also reaching
80 degrees at the slowest clock (the bar's reference check, cited); (b) on each rung the look ladder holds for the
body, the look measures of the walk's playback agree with the ladder's own numbers (bob within 5 %, head pitch
within 0.1 degree, tail phase within 0.01 cycle, toe-off within 0.02 rad) and the set's look-format hash for that
rung equals `stage2/look/L1_FREEZE.json`'s; on a rung the ladder does not hold the four numbers are finite with at
least 4 steps in the window; on an `-X` body each walk's playback agrees with the plain body's playback of the
same rung within the same four limits; (c) four degraded playbacks each fail exactly their term (bob x 0.5; a
3 degree neck sinusoid; the tail 0.2 cycle late; flat feet), the unchanged playback passes all four with ratio 1,
and a playback cut to 3 steps reads `too_few_steps`; (f) each reference no bar item plays (the turn in place with
its start and stop; the changes to rungs other than 0.40 m/s) has its screen record cited, and every task's track
is continuous at its joins (joint targets within 1e-6 rad) and plays 2 s with finite targets. If (b) fails, U1's
definitions go back to the lead before any look-set episode.
*G9-run*, after R0L's records of that set and body: (d) every episode of the body's `R0L@10ms` records holds every
reading of its item and, where its window was reached, a finite look reading, and every 10 ms bar record holds
both toe-clearance readings; (e) for the 60 cm body and head the re-run Stage 1b 10 ms turn episodes of
`60-N-T-10` are trace-identical to their committed records (the bar's re-read record, cited). Check:
`stage2/tb/gates/<set>/<body>/g9_ref.json`, `g9_run.json`; `g9_ref.json`'s time before the first look-set episode
in `stage2/tb/dev_episodes.json` and in any record.

## Q. R2L

**Q1. What is fitted.** [plan B.2; S2-18; owner's allowance of the candidate] Stage 1's tracker (R2, 13 searched
gains, capture margin held) on the bar's tasks, seed 42, here: on set `V6noF5` for `75-M-H-10` and `60-N-T-10`
(current heads), and, labelled a candidate, on set `F25F9cl` for `75-M-H-10` if Q5 allows. `speedfit.py` is not
edited. Check: `stage2/tb/r2l/<set>/<body>/r2lfit/42/cma.json` `settings`.

**Q2. Mix and rotation.** [S2-18] Every evaluation runs five bar episodes at mu 0.9: `walk/v0.4`, a quick turn, the
two-touchdown stop, a nudge at 0.25 x the bound, `walk/nominal` (slots j = 0-4). In generation g, with block
`B = g // 6`: the episode seed is `tune.training_order(42)[g]`; slot j runs at `k = (g + j) mod 6`; the turn is to
the left for even B and to the right for odd B; the nudge's direction is forward, left, backward, right for
`B mod 4` = 0, 1, 2, 3. So within a block every slot meets all six delays in one direction, and every (kind,
direction) meets every delay, 10 ms included. Start and noise seeds are derived per slot (Stage 1b's rule: five
different starts per evaluation; unlike the bar's one seed slot, on purpose). Every member of a generation runs the
same five episodes. The table for 30 generations is written before generation 0. Check: `rotation.json` (its
`written_at` before `sessions[0].started_at`; per (kind, direction) the delays met); `evals.jsonl`'s delays and
directions equal it.

**Q3. Objective and search.** [S1b; NEW terms for the bar's items] `tune.score` (falls first, then tracking and
limit terms). Tracking terms: `tune.measure`'s over the item's judged window. Limit terms: `tune.measure`'s window
limits without the three "at nominal" ones, with toe clearance on the own-swing reading, plus the item's own, each
graded on [0, 1]: walk: the shortfall of the distance along the heading over 4.0 m, the line deviation over
0.25 m; turn: the heading error at the judged time and the largest in the 3 s after, each over 20 degrees; stop:
touchdowns beyond two over 2, the stop time over 1.5 strides; nudge: the recovery time over 3 s. CMA-ES,
population 12, sigma0 0.1, from the body's Stage 1 G5 selection; at most 30 generations. Check: `cma.json`
`objective`, `start`.

**Q4. Selection.** [S2-18; one partial departure] Candidates: the start and the mean after each third of the
generations judged. **Decisive, run first and always whole:** each candidate runs the five kinds on validation
seeds 2000-2019 at k = 5 (turn to the left for even seeds, to the right for odd; nudge direction forward, left,
backward, right for seed mod 4 = 0-3), ranked by `tune.selection_key` on those 100 episodes per candidate alone.
Reported and not used, as far as Q5 reaches, in this order: the selected candidate and the start at k = 0, 2, 4,
1, 3 on validation seeds 2000-2004 with the same directions (25 episodes per cell); a cell not reached reads "not
run: budget" **[differs: S2-18's "the others reported", partly]**. Check: `r2lg5.json` `selection`, `by_delay`.

**Q5. Budget.** [plan section 6: 8 core-hours; NEW split] Counted as the sum of evaluation CPU seconds in
`ledger.json`. Per set and body 4.0 core-hours: the fit at most 2.4 (it ends at the last whole block of six
generations inside that, or at 30); 1.2 reserved for the decisive cell, which finishes even if it needs more (the
overrun is recorded; it is the only one allowed); 0.4 for the reported cells. The two `V6noF5` fits: 8.0 in all.
The `F25F9cl` fit on `75-M-H-10` needs 4.0 more; it runs last and only if the lead grants them, else it is
reported "cut by the pre-registered order". Check: `ledger.json`; `r2lg5.json` `judged_at_generation`, `overrun`.

**Q6. Rows and reading.** [plan] Rows are named by their fit: `R2L-V6noF5/42@<ms>ms`, `R2L-F25F9cl/42@<ms>ms`,
resolved through the committed index `stage2/tb/r2l/index.json` (set, body, seed, the gains file's path and
sha256); a row refuses a condition of another set. Read at 10 ms as a baseline beside R0L and R5 (and at 0, 4 and
6 ms where the plan asks); one seed, so every verdict of R2L is provisional. If R5 fails G8b on both seeds of a
body, that body is reported with R2L at 0, 4, 6 and 10 ms (S2-9) and the owner is told before more Colab hours.
Check: the bar records' `controller.gains_info` (path and sha256 of `r2lg5.json`); `index.json`.

## K. Checkpoints, resume, versions

**K1. Boundaries.** [NEW; differs: S1, where each session drew new seeds] At step 0 and at the first rollout
start at or past each multiple of 250,000 steps a checkpoint is saved; then, in every run whether interrupted or
not, everything random is reseeded with `100,000 x seed + 100 x b` (b = that multiple / 250,000), every env is
reset, and the running returns of the reward normalisation are zeroed. A resume loads the newest intact checkpoint
without letting the library reset the envs, continues the step count and the schedule, and does the same boundary
before its first rollout. Check: `sessions[].boundary_seeds`; `tests/test_stage2_train2.py`.

**K2. Kept checkpoints.** [plan B.4; NEW precision for Drive] Locally: each check's (M2) and the newest two. To
Drive the launcher copies the checkpoints at multiples of 1,000,000 steps and each check's, each as a whole (three
files, then the record), and keeps each check's and the newest two of those; an older one is deleted on Drive only
after a newer one has stayed copied for one further copy cycle. A copy read back through the mount is "copied",
not verified. For each check's checkpoint one tar of its three binaries is also written. Check: `run.json` `kept`;
STATUS.json `last_checkpoint_copied`; the files on Drive.

**K3. Kill and resume.** [plan B.3; NEW pass rule and settings] On this machine, with one set of library versions,
on the real env: `75-M-H-10`, set V6, R5, seed 42, 8 envs, `n_steps` 1024, a checkpoint every 32,768 steps,
131,072 steps in all; one run uninterrupted with 2 workers; one run with 2 workers killed with SIGKILL (the whole
process group) when its progress reads 81,920 steps and resumed with 3 workers. Pass: every policy tensor, the
optimizer's state, the normalisation statistics (their count included) and the fingerprint actions of the two
final checkpoints are **equal** (tolerance zero). If they are not, the build stops and goes to the lead; no
tolerance is set after a difference has been seen without a label. Across machines or versions nothing is
asserted. Check: `stage2/tb/validation/kill_resume.json`.

**K4. What a resume refuses.** [S2-13; S1] Changed settings or env configs: always. Changed inputs (the sha256 or
the entry hash of a used file; the plant's inputs): always; a later freeze of the set is accepted when every used
file's and entry's hash and the resolved task table are unchanged, and both freeze indices are recorded. Changed
sources (a fixed list of files, hashed per session, children included): unless `--allow-source-change`. Changed
library versions (Python, numpy, MuJoCo, scipy, torch without its build label, stable-baselines3, gymnasium,
cloudpickle): unless `--allow-version-change`. A checkpoint that fails its integrity check is skipped with its
reason recorded; a resume needs one whose run id (`run_uuid`) and settings hash are the run's and whose
fingerprint is reproduced within 1e-5. A flagged session records both sets and the flag, and every later record of
the run carries it. In the notebook the two flags are the parameters `ALLOW_SOURCE_CHANGE` and
`ALLOW_VERSION_CHANGE`, set only on the lead's word. Check: `sessions[]` (`version_change`, `source_change`,
`skipped_checkpoints`, `fingerprint_max_abs_diff`, `freeze_index`).

**K5. The state a resume starts from.** [NEW] Every checkpoint's record holds the run state at its step: the task
weights in force, the fix, the monitor history, the no-gain count, the checks done, any stop. A resume rebuilds
the run's live record from it, moves later entries to `discarded_after_resume`, cuts `updates.jsonl` back to the
resumed step, runs any check that is due at that step and has no record naming that checkpoint, and then applies
M3-M5 to the records exactly as the uninterrupted run does. A job with a run record and no checkpoint restarts
from step 0 in place and is the same run (`restarted_from_zero`). Check: `checkpoint_<n>.json` `run_state`;
`sessions[].discarded_after_resume`; `tests/test_stage2_train2.py` (a kill during a check; after a fix).

**K6. One commit, one path, one queue.** [S2-13; NEW precision] A run keeps one RUN_ID, one bundle, one queue, one
budget and one commit for its whole life: REPO_REF is the 40-hex SHA of the launch commit (the notebook refuses
any other form for a launch), a resume takes it from `QUEUE.json`, and another commit is accepted only with
`ALLOW_SOURCE_CHANGE`. The notebook refuses a RUN_ID that exists without RESUME, and RESUME with another bundle,
queue or budget. Check: `QUEUE.json`; `SESSION_<k>.json`.

**K7. One writer.** [NEW] The launcher writes `LIVE.json` (its session's id, host, time) every 60 s and re-reads
it before every copy or delete on Drive; if the id is not its own it halts its job ("superseded"). A session with
RESUME is refused while `LIVE.json` is younger than 10 minutes, unless `TAKEOVER` is set on the lead's word. Check:
`LIVE.json`; `SESSION_<k>.json` `takeover`; `HALT.json` `reason`.

**K8. Fine-tunes.** [plan S2-6] A new job directory from a parent's checkpoint and statistics, the parent's
`log_std`, 5,000,000 steps, its own boundaries and checks, a constant learning rate: the parent's schedule value
at the parent's step, lowered to the largest of 1e-4, 3e-5, 1e-5 not above it at which all minibatch steps of 6
updates run from the parent's checkpoint [NEW: the parent's sigma is below 0.1]. Check: `run.json` `parent`,
`settings.lr`.

**K9. The fingerprint and the loader.** [S2-12; S2-14] Every checkpoint is saved with 256 observations from the
generator seeded `[20261115, observation width, actor width]`, the deterministic float32 actions its policy and
statistics give them, and the statistics as plain arrays. Every load, anywhere, goes through one loader that
checks the files' hashes, passes the schedule objects explicitly and requires the actions reproduced within 1e-5;
a load for judging also passes the spaces and reads the statistics from the plain arrays, so that only tensors and
the policy class cross library versions. A checkpoint that fails is not used and, if it was to be judged, the owner
is told. Schedules are instances of module-level classes; an archive holds no bytecode. A judged row is refused if
the code that builds its observations and applies its actions differs from the run's recorded hashes, unless a
flag records `source_change`. Check: `checkpoint_<n>.json`; the rows' `controller` (both version sets,
`cross_version`); the companion records' `policy` (`fingerprint_max_abs_diff`); `tests/test_stage2_s2load.py`.

## Y. The day-one smoke and the budget

On the chosen runtime type, one session, the notebook's `SMOKE` mode, on `75-M-H-10` with the current head. The
steps sheet gives the expected minutes of each item.

| Rule | Item | Recorded | Pass |
|---|---|---|---|
| Y1 | setup | the versions against the pins (N2), torch's build label, Python's version, ffmpeg | the pins installed; a Python minor other than 3.13 is reported (K9 covers it) |
| Y2 | the machine | vCPUs, RAM, CPU model | reported |
| Y3 | memory | resident memory of the trainer and of each worker with 8 envs after one rollout | the total under 80 % of RAM |
| Y4 | the rate at 8 envs | env steps per wall second of an R5 job over its first three rollouts, updates included | Y11 gives at least 15,000,000 steps |
| Y5 | a test clip | an MP4 of 3 s from a monitor trace, rendered in a subprocess with OSMesa | reported (a clip is no stop condition) |
| Y6 | kill and resume on Drive | a tiny job killed after its first checkpoint is on Drive; its local directory deleted; resumed in a new process | the resume finds an intact checkpoint on Drive, reproduces its fingerprint, refuses nothing and reaches its budget |
| Y7 | setup time | seconds from the setup cell to the first training step | reported |
| Y8 | the inputs rebuilt once (last; capped at 40 minutes) | one body's models and as many of its V6 arrays as the cap allows, each sha256 against the manifest | reported; "not finished" is an outcome; if every file compared matches and all were compared, the rebuild is an allowed fallback (N4) |
| Y9 | an archive that loads here | the smoke's checkpoint tar, downloaded by the lead | the loader reproduces its fingerprint within 1e-5 in the study's venv |

**Y10.** [NEW] `SMOKE.json` is written after Y7 and again after Y8. The smoke passes when Y1, Y3, Y4, Y6 and Y9
pass. Until it passes no training job is queued. Check: `runs/<run_id>/smoke/SMOKE.json`;
`stage2/tb/validation/colab_archive.json`.

**Y11. The budget (S2-8).** [plan section 6, as stated] Fixed once after the smoke for every run: 40,000,000 env
steps, or, if Y4's rate r cannot reach that in 20 h (r under about 556 env steps/s), `r x 72,000` rounded down to
a multiple of 5,000,000; never under 15,000,000 without the owner. Asked steps are run in whole rollouts.
Runtime-hours per run (steps over rate, plus setup) are tabled per runtime type. Check: `QUEUE.json`
`budget_steps`; `SMOKE.json` `rate`, `budget`.

## N. The notebook and the bundle

**N1. The four lines.** [plan section 6; NEW text] Exactly:

```
S2 LINE 1/4 commit: <40 hex> (REPO_REF is a commit: yes; checkout clean: yes)
S2 LINE 2/4 inputs: bundle <bundle_id>, manifest <first 12 hex of the content hash>, <n> of <n> files match their sha256: OK
S2 LINE 3/4 G8a <body> set <set>: zero action = R0 at 0 ms and 10 ms, <n> of <n> episodes identical (<steps> physics steps): PASS
S2 LINE 4/4 first rate <job>: <r> env steps/s (8 envs, <W> workers, <c> vCPUs); <B>M steps in about <h> h
```

Line 2 ending `FAILED (...)` or line 3 ending `FAIL (...)` stops the session before any training. Line 4 ends
` BELOW FLOOR <f>` when r is under the launch's floor, the rate at which the budget takes 22 h (`budget / 79,200`;
505 env steps/s for 40,000,000); the owner then tells the lead. Check: the acceptance run (N5) prints them;
`STATUS.json` `lines`.

**N2. Pins.** [plan B.4; NEW values] The notebook installs the study venv's versions: mujoco 3.10.0, numpy 2.5.4,
scipy 1.18.1, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0 (the CPU build, forced when another build is
installed), cloudpickle 3.1.2, pillow 13.0.0, PyOpenGL 3.1.10; ffmpeg and OSMesa from the system. There is no
unpinned fallback: a pin that does not install stops the session. Training processes run with
`MUJOCO_GL=disable`, assigned by the entry point itself. Check: `SESSION_<k>.json` `versions`, `pins_match`,
`torch_build`; `tests/test_stage2_nb.py::test_pins_equal_the_study_venv`.

**N3. The bundle.** [plan B.4] The model files and the frozen sets' arrays of the bodies and heads of a launch, in
plain tar parts of at most 20 MB, with a committed manifest of every file's and part's sha256. The manifest's
content block alone is hashed and quoted; upload receipts are kept in a separate file. The notebook verifies every
part and file before anything is built; any mismatch stops the session. Check:
`stage2/tb/bundle/<bundle_id>/MANIFEST.json` (`content_sha256`), `UPLOADS.json`; line 2.

**N4. Fallbacks if an upload fails.** [NEW] In this order: parts of at most 5 MB; the model parts alone with the
arrays rebuilt on Colab, allowed only after Y8 found every hash equal, the rebuilt arrays kept on Drive and
reused; with the owner's consent, the parts on a data branch that is never merged. Nothing is rebuilt or fetched
from anywhere else. Check: `SESSION_<k>.json` `bundle.source`.

**N5. Before the owner's first session.** [plan B.3; the plan's dry run and tiny notebook run, made one] The
acceptance run: a fresh clone of the launch commit from the remote, holding nothing git ignores, plus the bundle
in a directory standing in for Drive; the notebook's own cells: every hash verified, every env built, Z1's short
form passed for each body and head of the bundle, a tiny R5 job and a tiny R6 job; the launcher's process group
killed with SIGKILL while a copy to the stand-in Drive is in progress; the local run directory and the clone
deleted; a second clone; RESUME to DONE; N1's lines printed; a second launcher on the live run refused. Check:
`stage2/tb/bundle/<bundle_id>/ACCEPTANCE.json` (the remote's SHA, the clone's HEAD, the content hash, the lines,
the kill point, the sessions).

## X. Episodes that are no records

**X1. Development and check episodes.** [`STAGE2_75_PLAN.md` R4 as settled; NEW for the look sets] Until task 3's
first unit of the 75 cm round has run, no R2-mode, policy or training episode runs on V0, F25F9 or F25F9cl; the
build's tests and acceptances use Stage 1's own run directory (the read-only S1 view) or a synthetic set in a
temporary directory. On a look set, for each body and head: no closed-loop episode before G9-ref passed; between
G9-ref and S2-4's start only these: Z1's episodes, R0L's baseline and panel records, Z2's G9-run, the probes of
O2-O4 (zero action, noise, freshly initialised policies), the rate measurement, K3's two runs, and the acceptance
run's tiny jobs. None but Z1's, R0L's and Z2's is a record; each is listed in `stage2/tb/dev_episodes.json` with
its command, set, body, rows or mode, and count. Check: `stage2/tb/dev_episodes.json`;
`stage2/runs/2026-11-15-75cm/dev_episodes.json`.

## Not run in this round, by name

Periodic walking-turn gaits; head-first turns (F11); any R5 or R6 run on a set other than V6; the four no-loss
condition kinds the bar does not define on a set (stand, turn in place, start-stop, standing push: Track C's to
build); a carpet verdict; the hardware levers and their fine-tunes; third seeds. Each appears in the report as
"not run".

## Where this text changes the plan's wording

1. E6: the trainer is a new module and no bar module is edited (plan B.4; the look reading lives in companion
   records, U4, U6).
2. E3, W5: one input (the heading error) and three reward terms (`progress`, `heading`, `range`) beyond plan B.1's
   lists. [RULING NEEDED]
3. T2: "walking turns at 0.35-1.0 rad/s" are the finite 90 degree turns; "no loss" for R5 and R6 runs on look-set
   counterparts with the finite 0.35 rad/s turn (plan B.1; section 5). [RULING NEEDED]
4. M3: survival is credited as progress; M1, U5, U6: a policy row's unread episode and too few steps are "not succeeding", not
   C8's "run again", for rows that can slow the clock or stop stepping. [RULING NEEDED]
5. M3b: a second route to a gain when the survival route is closed (S2-10). [RULING NEEDED; struck by default]
6. E4, W2: the policy's command input is piecewise constant, while the tracking reward keeps Stage 1's target.
7. W3: two of Stage 1's reward terms are read differently.
8. U4: reading 1 is reading 2 for nudge and stop items; a fourth count is printed beside the three readings.
9. M5: "from 20M, three checks" is read as three consecutive checks counted from the one at 20M.
10. M6: a local 60 cm run and a tell-the-owner rule the plan does not have.
11. K1: every checkpoint is a boundary that resets the envs (the plan says "kill and resume"); K2: Drive holds
    every fourth checkpoint and each check's.
12. O1: 8,192 steps per rollout (`n_steps` 1024); O3: the probe's second sigma; O4: the filter's rule; N2: the
    pins' values.
13. Q4, Q5: the plan's 8 core-hours cover two fits; the reported delay cells may read "not run"; the candidate's
    fit is extra.
14. E5: head-first turns are not run in the head's "cost of using it".
15. N5: the plan's fresh-clone dry run and tiny notebook run are one acceptance run.

## Open for the lead (to be settled before this text is committed), most important first

1. M3b: rule it in or strike it; and whether a "no gain" that rests on the tail servo's sustained limit alone goes
   to the owner as a no-go or with the per-term table beside it.
2. E3 and W5: the heading input and each of the three terms.
3. T2: the no-loss counterparts and the finite turn for the periodic one (it decides whether the look work package
   must build a periodic walking-turn gait).
4. M3's credited survival; M1, U5, U6 in place of C8 for policy rows.
5. O2: sigma 0.1 as the plan has it, with the owner told what early episodes look like; or a lower start chosen by
   the probe (a labelled change to B.1).
6. O1: `n_steps` 1024 against 2048.
7. E6: confirm that no extension of `train.py` and no hook in `bar.py` is wanted.
8. E2: that R6's expected outcome may be stated in advance; E5: H1's numbers.
9. T1: the weights; T2: whether a set without the turn in place may be trained on.
10. M6: seed 41, 5,000,000 steps, the 0 ms episodes and the tell-the-owner rule; its 5.2 core-hours.
11. O3: one learning rate for all bodies and heads (the smallest), against one per body; O4's 0.9.
12. Q5: 4.0 core-hours more for the candidate's fit, or cut.
13. Y10: which smoke items must pass; K7's 10 minutes.
