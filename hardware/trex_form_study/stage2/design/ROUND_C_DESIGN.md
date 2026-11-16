# Round C design: R5, R6, R2L, the trainer, the gates, the notebook and the Drive bundle (Track B) - FINAL

Architect's final design, 15 Nov 2026, after three independent reviews (the review log is the last section). Scratch
file, not committed. Paths are relative to `/home/user/mesozoic-labs/hardware/trex_form_study` ("FS") unless absolute;
`REPO` is `/home/user/mesozoic-labs`. Code line numbers are of the main checkout at `89111c3`; `bar.py` is read from
`/home/user/mesozoic-wt/wp-b` (branch `wt/wp-b`, still moving on 15 Nov; cited by function name, not by line).
"C1"-"C11" are the lead's requirements, "D n" this document's decisions,
"S2-n" the plan's rules (`STAGE2_PLAN.md` section 9, merged at `89111c3`), "round A" is `ROUND_A_DESIGN.md`, whose
modules (`s2io`, `setio`, `readings`, `moves`, `sets`, `setscreen`, `bar`, `barreport`, `moves_look`) this design
builds on and does not edit: **round C changes no byte of any round-A module and of no file under
`designlab/stage1/`** (D18). Nothing in a checkout or worktree was edited and no pytest was run by the architect or
by the reviewers.

Every number marked **[scratch]** comes from a scratch experiment of this session (section 0). None is a record and
none may be cited as one. The rules to fix before the runs are in `PREREG_TRACKB_draft.md` (same directory); rule ids
such as "W2" or "K3" refer to it. One brief per work package is in `WPC_<name>.md` (same directory).

Changes of the plan's wording that this design proposes are tagged **[differs: plan]** and listed in section 7 for the
lead; nothing so tagged is to be built as final before the lead has ruled.

## 0. What was run, and what it settled

Environment of every command: cwd FS, `D=/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/roundc`,
`PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 .venv/bin/python`, one process at
a time, each under 3 minutes, load average 1.4-2.3 from the builders. Stock Stage 1 arrays only
(`stage1/runs/2026-11-08/<body>`: the read-only S1 view; no Stage 2 set was touched).

| # | Command | Result [scratch] |
|---|---|---|
| C1 | `$D/exp_c1_g8a.py 75-M-H-10 6.0` | **G8a's construction is exact.** The panels' R0 row (`Tracker` mode R1, `terms=()`, `ankle_cap=()`, rate 1.05, `control.run` from the settled stance, start noise 0.01 rad and 0.02 m/s, inside `speedfit.servo_delay(p, k)`) against a `Tracker` in **R2** mode with no term and no cap, stepped 10 physics steps per env step with `residual None` and `rate_factor 1.0`, on a **second, separately built plant** with the env's factor restore (`env.set_trunk_mass(..., 1.0, ...)`) before it: plan `start, nominal x 1, nominal_to_brisk, brisk x 8`, 6 s, seed 3042. sha256 of `qpos, qvel, ctrl, sensordata` after every physics step: 3,000 of 3,000 equal at k = 0 and at k = 5 (10 ms). The bare loop ran 684-700 env steps/s with the digests on |
| C2 | `$D/exp_c2_rate.py 75-M-H-10 hard 3000`; `... robot_floor_carpet 2000` | **Rate of a stub R5 env** (Stage 1's env with R5's base in place of R2: its observation and reward stand in for R5's; delay 0-12 steps, kp 0.6-1.2, friction 0.4-1.2): hard floor 438 (zero action) and 460 (action noise sigma 0.1) env steps/s in one process; robot_floor_carpet 422 and 352. Env build 5.2-5.4 s; resident memory 315 MB per env process, 354 MB with `legs.LegGap`. One `LegGap.at(qpos)` costs 0.19-0.20 ms (9 % of an env step at 50 Hz). **Under sigma 0.1 the 75 cm base falls about every 200 env steps** (15 falls in 3,000 steps; 1 fall with the zero action) |
| C2b | `$D/exp_c2_rate.py 60-N-T-10 hard 0 0`; `... 0 5` | **The 60 cm zero action** (5 episodes per task, nominal plant): at 0 ms the walk stays up in 2 of 5 (falls at 4.1-6.5 s), brisk 0 of 5 (4.1-10.3 s), the 0.35 rad/s walking turn 0 of 5 (4.1-8.5 s), the turn in place 5 of 5, the stand 5 of 5. **At 10 ms the counts are the same** (walk 2 of 5; falls at 4.3-13.1 s). The falls come seconds after the start, with or without delay |
| C3 | `$D/exp_c3_midstart.py 75-M-H-10 nominal hard`; `... robot_floor_carpet` | **A mid-task start needs no edit to `control.py` or `plant.py`.** A `Tracker` subclass lends the track another `q0`/`qd0` for one call of `Tracker.reset("frame")` (so `Plant.set_state` and `Estimator.reset` run as Stage 1 wrote them), sets `tau_ref` after, and refills the queue that `speedfit.servo_delay` yields with the k commands the reference gave before the start. Zero action from frames 0, 20, 50, 80, 110, 140 of the nominal stride, at k = 0 and k = 5: up for 8 s in 12 of 12 on hard floor (tilt at the end 0.1-0.6 deg). On robot_floor_carpet the zero action falls after 2.4-3.6 s from every frame (the open loop does not walk on that floor from a stance start either, decision 23) |
| C4 | `$D/exp_c4_archive.py` | **A module-level schedule gives an archive with no bytecode.** PPO (SB3 2.9.2, torch 2.15.0+cpu) on an -X sized layout (actor 97, observation 140, action 17) with `learning_rate=LinearSchedule(1e-4, 3e-5)` from an importable module: the repository's `inspect_sb3_archive` reads "bytecode members: none" (Stage 1's archives: `learning_rate`, `lr_schedule`); `load_sb3_model(..., custom_objects=...)` loads it; a fingerprint (256 seeded observations -> deterministic float32 actions through the saved normalisation) is reproduced with difference 0.0 after save and load, also with 2 torch threads. The archive is 2.41 MB. Wrapping `policy.optimizer.step` counts the minibatch steps PPO really ran: 70 of 120 in 3 updates (target_kl stopped the rest without a message, lesson 16), while `_n_updates` advanced as if all had run |
| - | `git check-ignore -v` on planned paths | `*.npz` is ignored at any depth (fingerprint files); `STATUS.json`, `run.json`, gate records, a `.tar` and `.mp4` under `stage2/` are **not** ignored until the new `stage2/tb/.gitignore` exists |

Run by the three reviewers on the same stub (scripts under `$D/review_learn`, `$D/review_prp`, `$D/review_cov`; 4-6
episodes per setting; scratch as above):

| # | Script | Result [scratch] |
|---|---|---|
| RV1 | `review_learn/rv5_noise.py` | **The base tolerates about a tenth of sigma 0.1.** `75-M-H-10` walk, k = 5, 10 Hz filter: sigma 0 and 0.01 up 6 of 6; 0.02 up 1 of 6; 0.05 none (3.8 s); 0.1 none (3.3 s). `60-N-T-10`: sigma 0 up 1 of 6 (8.4 s); 0.1 none (2.6 s). At equal filtered action RMS a 20 Hz filter survives 0.98 x as long as 10 Hz; at equal sigma 0.85 x |
| RV2 | `review_learn/rv6_ppo.py` | PPO on the stub, sigma 0.1, 5 epochs, batch 256: with a 16,384-step rollout lr 1e-4 ran 320 then 206 of 320 minibatch steps; 3e-5 ran all. With lower sigma far fewer run at the same rate |
| RV3 | `review_learn/rv2_exploit.py` | A clock held 25 % slow costs 1-7 % of Stage 1's reward per step and can survive longer than the zero action on `60-N-T-10` |
| RV4 | `review_learn/rv3_drift.py` | The zero action on `75-M-H-10` drifts 0.08-0.22 m from the line in 10 s (limit 0.25 m) |
| RV5 | `review_learn/rv4_midstart.py`, `rv7_gap.py`, `rv1_kin.py` | Mid-task starts are physically consistent on hard floor (stance-foot load 0.8-1.4 body weights at reset, tilt error under 0.2 degrees). On `60-N-T-10` the smallest leg gap is a fixed bracket pair at 4.0 mm in every pose. A body standing at home while the reference walks earns 0.39 of 1.25 per step under the draft's reward |
| RV6 | `review_prp/exp_resume.py` (re-run by the architect, same result) | **A resume that loads with `env=` and lets SB3 reset is not the uninterrupted run** (every tensor differs); with `force_reset=False` and the boundary in the callback it is bit-identical |
| RV7 | `review_prp/exp_history.py` | With trackers built once on the nominal model an env after 7 randomised episodes equals a fresh env after `reset(seed)` (2 seeds, 150 env steps each) |
| RV8 | `review_cov/rc1_import_rss.py`; re-run of C2 | A process that imports SB3 holds 358 MB before any plant (113 MB without); the stub's rate re-measured at 411-449 env steps/s; `LegGap.at` 0.36 ms per pose |

What these settle:

1. **G8a can be met bit for bit, across plant instances, at 0 and 10 ms** (C1), provided the env leaves
   `residual None` and `rate_factor 1.0` at a zero action, mirrors `env.Stage1Env`'s reset and step order, uses the
   bar's own delay wrapper, and is compared over the horizon both ran (3.7; the bar's lengths are not multiples of
   20 ms).
2. **R5's env is faster than Stage 1's R3 env** (C2, RV8). Every compute estimate of section 6 starts from 350 (hard)
   and 290 (carpet) env steps/s per process, to be replaced by the validation record.
3. **Mid-task starts and the delay queue need no edit to a key module** (C3, RV5); neither do 16 joints, commands,
   the trainer's modes or the schedule (section 2.3).
4. **On `60-N-T-10` the open-loop base falls as often at 0 ms as at 10 ms** (C2b). That says nothing about whether
   feedback is easier to learn through shorter delays; the delay curriculum stays a live option (D11).
5. **Sigma 0.1 makes every early training episode a 3 s noise fall, on both sizes** (RV1). Mid-task starts are what
   still shows the policy the whole task; the learning-rate probe must also pass at a lower sigma (D12).
6. **An exact resume needs three conditions** (RV6, RV7): no reset between the load and the boundary, trackers built
   on the nominal model before any randomisation, and nothing that depends on the process's history (D16).

## 1. Decisions

**D1. R5's env is a new `gymnasium.Env` in `designlab/stage2/r5env.py` that composes Stage 1's parts; it is not a
subclass of `env.Stage1Env`, and no file under `designlab/stage1/` is edited by this round.** It builds the plant
with `plant.build`, drives it with `control.Tracker(p, track, "R2", gains=None, terms=(), ankle_cap=(), rate=1.05,
model=nominal_model)` through the tracker's own `residual` and `rate_factor` hooks (`control.py:1273-1277`,
`1393-1394`, `1417-1418`), reads the tracker's estimator, and calls `env.py`'s helpers by name
(`env.set_trunk_mass`, `env.setconst_changes`, `env.foot_tables`, `env._gaussian_tracking_reward`,
`env._state_digest`, `env.PUSH_BOUND_M_S[body]`, `env.ENV_SUBSTEPS`, `env.RESIDUAL_RATE`). Its `reset` and `step`
follow `env.py:470-510` and `530-564` statement for statement where G8a needs it (section 3.2).
Rejected: (i) a subclass of `Stage1Env`: its `__init__` refuses any plant without 14 hinges (`env.py:361-364`),
builds its tracks from the literal `TASKS` (`:384-385`) and its spaces from module constants (`:398-399`); `reset`
zeroes 15-wide action buffers (`:499-500`) and always starts from the stance; a subclass would override every method
it has and inherit only the refusal. (ii) Generalising `env.py` in place behind defaults: any edit to `env.py` moves
`plant.sources_sha256` (the glob over `designlab/stage1/*.py`, `plant.py:267-275`), the speed fit's sources and the
R3/R4 rows' keys (map 05 section 0.1), and with them round A's committed `stage2/regress/regress.json`, which a fast
test compares with today's hashes; `tests/test_stage2_x.py:629-637` pins the refusal. With D1 that test stays true
and unchanged: Stage 1's env still refuses an `-X` plant, and Stage 2's env is the one that takes it.

**D2. The trainer is a new module `designlab/stage2/train2.py`; `designlab/stage1/train.py` and `policy.py` are not
edited.** [changes the plan's wording: B.4 says the notebook "runs `designlab/stage1/train.py`, extended with R5 and
R6 ...". S2-17 as ruled lets those extensions stand; it does not require them.] Why not extend: an edit to
`train.py` or `policy.py`, or a new file in `designlab/stage1/`, moves `plant.sources_sha256`, so `regress.json`
would have to be made again (about 1 core-hour) and every Stage 1 run would need `--allow-source-change` to resume;
`train.train` is one 270-line function whose mode checks, run-directory names, G6 logic and monitor are all R3's
(`train.py:395-396`, `158-159`, `519-523`, `567-594`). Shared by import, unchanged: `train.whole_rollouts`
(`:69-71`), `policy.DESIGN2` (`policy.py:35-36`), `policy.model_kwargs` (`:154-162`; its closure schedule is
replaced by keyword), `policy.PrivilegedCriticPolicy` (the archive stores the class by reference, so
`designlab.stage1.policy` must be importable wherever an archive loads: it is, in every clone). Mirrored in new
code: the settings and resume refusals (`train.py:426-465`), the checkpoint at a rollout start (`:551-566`), the
rate record (`:602-619`). Rejected: a `--mode R5` branch inside `train.train` (the same hash moves, and Stage 1's
tests pin its records' schema).

**D3. One reference library, indexed by task, not by a free command** (`designlab/stage2/r5lib.py`). An episode
plays a *script*: a plan of round A's track items (`("ref", name, windows)`, `("transition", name)`,
`("hold", seconds)`) over one frozen set directory, built with `setio.build_track` (so the track and its command
events are the bar's own). The command the policy sees is derived from the script, never from the reference's
instantaneous motion: it is piecewise constant and changes only at a segment's first frame (section 3.3), which is
the command instant of round A's D5 for causal moves and the file's first frame for starts and changes (before that
frame the track is the periodic walk, frame for frame). So the set of commands the policy can receive is the set
that has a reference: speeds 0, nominal and the frozen rungs of 0.30-0.50 m/s; yaw rates 0 and the frozen turns'.
Rejected: (i) a continuous command with the nearest or an interpolated reference: no reference exists between
rungs, two gaits of different stride periods cannot be blended frame by frame, and every bar and ladder condition
commands a rung; (ii) Stage 1's `_command` (`env.py:613-620`: the reference's own CoM speed and yaw rate at the
clock): it varies inside a stride and, on a non-causal start or change, moves before the command.
A bar condition is a script too: `Library.from_plan(plan)` builds the same side tables on the plan that
`bar.BarPlanner.resolve(cond)` returns, so the env (G8a, monitor) and the policy row read one code path.
For development and for every test before task 3's first unit (D23) the library also reads the read-only S1 view:
`Library(p, body, "S1")` takes `setio.stage1_view` and the small table `r5lib.S1_TASKS` (`stand`, `walk`, `brisk`,
`stop`: Stage 1's own plans of `env.TASKS`). The trainer accepts set `S1` only with `--tiny`.

**D4. "Walking turns at 0.35-1.0 rad/s" are round A's finite 90 degree turns from the 0.40 m/s walk**
(`turn90_<rate>_<dir>` for the four rates of `setio.TURN_RATES`); R5 trains on no periodic walking-turn gait.
**[differs: plan B.1, as read; needs the lead's ruling, section 7 item 3.]** Every turn item of the bar and of the
turn-rate ladder is such a move, each contains its own entry, hold and exit, and it is causal. Rejected: periodic
look turns with enter and exit moves (round A 3.9 calls them moderate to hard; no bar or ladder condition plays one).
The turn in place stays a periodic gait (D5).
Consequence, stated because the draft did not: Stage 1's twelve "no loss" conditions hold two periodic walking turns
at 0.35 rad/s, two turns in place, a stand, a start-stop, standing pushes and three walks, all defined on Stage 1's
stock references. An R5 policy is judged on the references it imitates (3.11), so its no-loss panel cannot run on
the pseudo-set S1. Proposed reading: no-loss for R5 and R6 runs on **look-set counterparts** built only from entries
D5 already asks for (stand; `walk/nominal`, `walk/v0.35`, `walk/v0.5`; the finite 0.35 rad/s turn left and right in
place of the periodic one; the turn in place left and right; start-stop; standing pushes), compared with Stage 1's
committed counts beside them. The training task table (T1) covers each. The four condition kinds that the bar does
not define on a set (stand, turn in place, start-stop, standing push) are Track C's to build as bar conditions of
their own module; round C guarantees the references, the tasks and that a Track B row runs any `BarCondition` of its
run's set.

**D5. What R5 needs from the sets that round A's `sets.items()` does not list** (for the lead's look work
package; nothing else is asked of round A, and nothing of `bar.py`):

| Needed | Why | Without it |
|---|---|---|
| `refs/inplace_0.5_left.npz`, `inplace_0.5_right.npz` (the periodic 0.5 rad/s turn in place under the set's laws), with `transitions/start_inplace_0.5_<dir>.npz` (stand to the turn) and `stop_inplace_0.5_<dir>.npz` (the turn to a stand) | plan B.1 names the turn in place and its start and stop; four of Stage 1's no-loss conditions need them (D4) | the two `inplace_*` tasks are dropped from the task table and recorded as absent (T2); the plan's B.1 list is then not met and the no-loss counterparts lose two conditions |
| Set `V6` built, screened and frozen on **each body and head trained**: `75-M-H-10`, `75-M-H-10-X`, `60-N-T-10`, `60-N-T-10-X` (the `-X` sets head neutral: neck yaw 0, jaw on its stop; generated on the `-X` model, whose mass and CoM differ) | G8a, G9 and training are per body and head | no run of that body or head |
| Set `V6noF5` on `75-M-H-10` and `60-N-T-10` | R2L (C6) | no R2L on that body |
| Every item of `sets.items()` on those: six walks, `start`, `nominal_to_v<v>`, `stop2_v0.4`, `turn90_<rate>_<dir>` | the task table (T1) | a task whose reference is `available: false` is dropped and recorded (ruling 4 expects this for the 60 cm quick turn) |
| Each walk file holds `phase` when it is a K-stride window (round A 3.9 lists it among the look keys); each move file holds `phase` (Stage 1's do) | the phase input and the swing clock (3.3) | a walk file without `phase` is read as uniform in its window (exact for one-stride files) |
| Each walk entry keeps its `look_sha256` (round A 3.6) | G9-ref compares it with `stage2/look/L1_FREEZE.json` (3.7) | that comparison reads "not checked" |
| Optional, not required: `stop2_nominal` and `v0.4_to_nominal` | Stage 1's task shapes | the nominal-walk task ends in the gait, as the bar's `walk/nominal` does; start-stop uses the stop from 0.40 m/s |

Before the look work package commits to four bodies (run plan step V0): gate G1 of both `-X` bodies on an idle
machine, the lead's decision on the jaw cover's overhang recorded (`stage2/x/HEAD_X.md` section 7: trimming it
changes every `-X` record), and one kinematic `sets build` of a walk and a start on an `-X` model (nobody has built a
set on 16 joints).

Names are proposals; they must come from `setio` (no transition may be named `stop` or `stop_in_place_*`, round A
D2; `stop_inplace_0.5_left` is not in `transitions.TRANSITIONS`, so `control.py:994` registers no Stage 1 stop
event for it). R5 reads from an entry only keys that are in round A's `WALK_SUBSET` and `MOVE_SUBSET` plus
`available`. No new entry key is asked for, and `set.json` needs no `look_bands` key (D17).

**D6. Mid-task starts are a wrap of `Tracker.reset`, and the delay is the bar's own wrapper** (C3).
`r5env.R5Tracker(control.Tracker).reset_at(t_ref, qpos, qvel, noise_seed)` lends the track `q0 = qpos`,
`qd0 = qvel` for one call of `super().reset("frame", noise_seed=...)` (which runs `p.set_state` and, in R2 mode,
`est.reset(qpos, qvel, reading)`, `control.py:1292-1296`, `1329-1337`), restores them, then sets `tau_ref = t_ref`.
The state comes from the segment's own file: the library keeps each file's full `q` and `qd` frames; the start is
snapped to a frame; the rates are multiplied by the commanded clock rate. The servo delay of every episode is
`speedfit.servo_delay(p, k)` entered by the env after the start (it yields its queue, `speedfit.py:324-362`); for a
mid-task start the env refills that queue with the k commands the zero action would have given over the k physics
steps before `t_ref`, sets `d.ctrl` to the oldest and calls `mj_forward` once. Rejected: (i) a start option in
`control.Tracker.reset` (a key-module edit); (ii) a fourth delay queue in Stage 2 (three exist, map 05 section 1.5;
using the bar's makes G8a's delay equal by identity of code); (iii) `env.Stage1Env._latency_on`'s queue (a closure
variable nobody can refill, `env.py:513-524`).

**D7. Layout per body from the plant's joint list** (`r5core.layout(joints)`; section 3.1). Actor: Stage 1's
deployable blocks (`env.py:138-139`) with `q` and `qd` nj wide, the command, **the heading error** (one number: the
estimator's yaw less the commanded heading), the last three clipped actions (3 x (nj + 1)) and, when the plant has
`neck_yaw` and `jaw`, a 4-wide head command. Critic: Stage 1's privileged blocks with `joint_error` nj wide, two more
plant factors, the foot errors and the reference's contact. 14 joints: actor 88, observation 130, action 15. 16
joints: actor 102, observation 146, action 17. Every index is by joint name (`-X` puts the legs from index 4).
The heading error is **[differs: plan B.1's input list; section 7 item 2]**: the bar judges the line (0.25 m) and the
heading (10 degrees), the actor of the plan's list has gravity and gyro but no yaw, and a network without memory
cannot integrate the gyro, so without it both readings rest on open-loop accuracy whatever is learned [scratch RV4].
The estimator already carries the yaw (the tracker's heading term used it), so the input is deployable. If the lead
keeps the plan's list, `layout(joints, heading=False)` drops the block (87/129 and 101/145), the reward's `heading`
term is struck with it, and rule E3 says in words that R5 has no heading loop and that `line` and `heading` are
open-loop readings. Rejected: one padded 16-joint layout for all bodies (the plain bodies' policies would carry
inputs that never move, and the head comparison would compare two layouts anyway).

**D8. Action: joint offsets and a clock rate on R0L's targets; R6 sets targets alone.** R5:
`tk.residual = a_f[:nj] * scale` (0.35 rad on `plant.leg_joints`, 0.15 rad on the others, the `-X` joints
included), `None` when every entry is zero and no head command is active; `tk.rate_factor = 1 + 0.25 a_f[nj]`;
`a_f` the clipped action through a first-order low-pass (cut-off by rule O4, 10 Hz until validation). The
feedforward and the rate targets stay the reference's (the tracker's own lines, `control.py:1396-1401`). R6
(`R6Env`): `q_target = clip(q_home + a_f[:nj] * span, lo, hi)` with Stage 1's span rule
(`env.POLICY_ONLY_SPAN_RULE`) over the env's scripts, `p.command(q_target)` alone, its own sensors and estimator,
as `env.PolicyOnlyEnv` (`env.py:738-790`); the same observation layout, commands, reward, starts and randomisation
as R5. On `-X` the head command is an overlay added to the reference's neck-yaw and jaw targets before the policy's
offsets (through `tk.residual`; R6: added to `q_target`'s reference part); with the head neutral it is exactly
zero and R5's zero action is R0L bit for bit.
R5's offsets are not clipped to the joint ranges (a clip would move the zero action's targets on a turn whose
reference uses the range to within 0.02 rad); the range enters the reward instead (D9). An R6 mid-task start sets the
filter state and the three-action history to the actions that reproduce the reference's joint targets at the three
env steps before the start, `clip((q_ref - q_home) / span, -1, 1)` with clock action 0, so that the target does not
jump toward home when the queue runs out. **R6 is expected to stand still or shuffle** (Stage 1's R4 did on every
body, `stage1/runs/2026-11-08/STAGE1_PROBE.md:423`; with this reward standing at home earns about 0.3 of the
positive weight [scratch RV5]); rule E2 says so, so that its outcome is not read as news. The contact and head terms
are not gated for R6's sake: R6 is "R4 with R5's reward" (plan B.2).

**D9. Reward: Stage 1's terms and weights unchanged, plus the plan's additions with their own weights, plus three
terms the plan does not list, all in the env config** (section 3.4, rules W1-W5). Stage 1's terms: two change
their reading and are labelled: the clearance penalty is read on the foot's own swing and also charges a foot that
is loaded while the reference is in mid-swing (so a dragged foot and a stub step are charged, as the bar fails them),
and the duty band follows the owner's decision below 0.30 m/s. The tracking term keeps Stage 1's target (the
reference's own speed and yaw rate at the commanded clock). The plan's additions: foot and contact imitation, head
attitude, action rate, touchdown speed, neck and tail drives over their rating, leg-to-leg interference. **Three
terms beyond plan B.1's list [differs: plan; section 7 item 2]**, each added because the bar judges something the
reward did not see [scratch RV3, RV4]: `progress` (the walk item fails 4.8 % short of the commanded distance, and
Stage 1's tracking term charges 0.2 % of the reward for a 5 % shortfall, so the clock action could trade speed for
survival almost free), `heading` (with D7's input), `range` (decision 14's joint-range margin, which the offsets can
use up). Source of the weights: Stage 1's where the term is Stage 1's; the others are this design's, set before any
run on [look] 3.3's recipe, sized so that no new term outweighs Stage 1's joint imitation (W2 gives each number and
scale). None was tuned on a result.

**D10. Leg-to-leg interference enters the reward and the termination** (`LESSONS_LEARNED.md` 2.8). At the end of
every env step (50 Hz) `legs.LegGap(body).at(qpos)` gives the signed distance of every left-leg to right-leg pair of
the screened geometry, the reading the bar's turn item judges. Penalty: on the **moving pairs** (`LegGap.moving`:
pairs whose home distance is at least the report cap), 1 at contact falling to 0 at `clearance.TIGHT` (3 mm); the
fixed pairs (on the 60 cm bodies two hip brackets 4.0 mm apart at every pose, `legs.py`'s docstring) carry no ramp
and are listed in the env description. Termination: the smallest distance over **all** pairs below zero at two
env-step ends in a row (40 ms). The env build stops with the pair's name if a fixed pair's home distance is below
zero. Stated in W4: a one-pose contact is penalised, not terminated, while the bar's turn item fails on one pose; the
training signal is weaker than the bar there, by choice (a graze of one pose would end episodes the bar only fails on
turns). G8a runs the env with the termination off and records when it would have fired. Cost about 0.4 ms per env
step [scratch RV8]. Not covered, stated in the env's rules: a leg against the trunk or tail.

**D11. The 60 cm zero-action problem: one recipe for both sizes, and a local run on `60-N-T-10` that tells which
remedy the evidence points to before any 60 cm Colab hour.** The facts: R0 keeps 3 of 20 nominal walks up at 0 ms
(S1 section 4); the open-loop falls come 4-9 s into an episode and are as frequent at 10 ms as at 0 ms [scratch
C2b]; under sigma 0.1 both sizes fall after about 3 s [scratch RV1], so early training looks alike on both; Stage
1b's fitted tracker walks this body at 0 ms and falls in nearly every walk at 10 ms (S1b section 4), so feedback is
harder to hold through the delay on this body. Options:

| Option | What it does | Verdict |
|---|---|---|
| (a) Same recipe as 75 cm | half of the episodes start mid-task in reference states, so the policy sees every part of every task although episodes are short; `policy.LOG_STD_RULE` cannot be applied (the plan fixes sigma 0.1 for that reason) | **kept as the default** |
| (b) A delay curriculum (k uniform on 0-2 at first, widened to 0-12 by a fixed schedule) | feedback may be learnable at short delays first; **not tested by C2b** (which shows only that the open loop is indifferent to delay) | a live option, not the default: it adds a schedule both sizes and heads would have to share; named to the owner if (e) points to it |
| (c) More mid-task starts or starts only late in the walk | the base falls within seconds from a stance start already; a body-specific start mix breaks "one recipe" | rejected |
| (d) A feedback base (the tracker's terms) under the 60 cm policy | S2-3 chose R5 without a tracker | a plan change: owner only |
| (e) **A local 60 cm run to 5M steps** (`60-N-T-10`, current head, seed 41, no judged seed), read by G8b's rule, **with the evidence that separates the causes** | at each of its checks the monitor's 18 episodes also run at 0 ms for the policy and for R0 (reported); the record holds sigma, the share of minibatch steps run and S against R0L beside the 2-of-3 reading | **recommended**; pre-registered as M6 |

M6 starts no Colab run and stops none. If the nominal walk stays up in fewer than 2 of 3 at 5M, the owner is told
before a 60 cm job is queued, with the reading that the record supports: up at 0 ms but not at 10 ms points to (b);
up at neither with sigma still near 0.1 or most minibatch steps cut points to exploration (D12's remedies); up at
neither with sigma small points to (d) or to launching under G8b. S2-4's go or no-go stays the 75 cm reading.

**D12. PPO: design 2 on 8 envs with Stage 1's rollout size, sigma 0.1, a learning rate that passes its probe at two
sigmas, schedules as objects** (rules O1-O3). `n_envs 8`, `n_steps 1024` (a rollout is 8,192 steps, Stage 1's size;
S2-4 then has 245 updates, not 123), `batch_size 256`, `n_epochs 5` (160 minibatch steps per update), `target_kl
0.03`, `log_std_init ln 0.1`, `learning_rate = schedules.LinearSchedule(lr, 0.3 lr)` over the run's budget,
`clip_range = 0.2` (a float: SB3 wraps it in its own schedule class; scratch C4's archive holds no bytecode). `lr`
is the largest of 1e-4, 3e-5, 1e-5 at which every minibatch step of 6 updates runs **at sigma 0.1 and at sigma
0.03** (the second on a model whose `log_std` is set to ln 0.03: `log_std` is learned, KL per step grows as sigma
shrinks, and `target_kl` then cuts updates without a message, lesson 16), on every body and head of the launch (one
value for all runs, so two heads never differ in it). Every update's minibatch steps run, of the possible, go into
`updates.jsonl` (the optimizer's `step` wrapped, C4); if their median share over a check interval is under one
half, STATUS.json flags `kl_cut` and the lead is told (O3; it stops nothing). Sigma 0.1 stays the plan's: the
sigma probe is reported, and the lead and owner are told now what it will show (section 7 item 5).

**D13. 8 envs in W worker processes: `designlab/stage2/vecpool.PoolVecEnv`.** Each of W child processes hosts its
share of the 8 envs (contiguous blocks, `numpy.array_split(range(8), W)`) and runs a 40-line loop that mirrors
SB3's `DummyVecEnv.step_wait` and `reset` statement for statement (auto-reset, `terminal_observation`,
`TimeLimit.truncated`, the seed given to the next reset) **without importing SB3 or torch** (a worker then holds
about 245 MB less [scratch RV8]). The children are started as `python -m designlab.stage2.train2 worker <address>`
(a subprocess that connects back over `multiprocessing.connection`), not by `multiprocessing` spawn: their command
line then carries `designlab.stage2.train2`, which `tools/memguard.sh` can protect, and the entry point assigns
`MUJOCO_GL` and the thread variables itself. An env is described to its worker by an importable factory name and
keyword arguments, never by a pickled closure. `W = min(8, vCPUs)` unless `--workers` says otherwise; `W = 0` runs
SB3's `DummyVecEnv` in the trainer's process. Because every env has its own generator and plant and no state is
shared, a rollout does not depend on W (tested bit for bit at W = 0, 1, 2, 4, and 8 behind the slow flag), so
"PPO's batch and floor mix never depend on the runtime" holds by construction. Env i's floor is `("hard", "hard",
"hard", "robot_floor_carpet")[i % 4]`. Rejected: SB3's `SubprocVecEnv` (one process per env: 8 processes and 2.8 GB
on any runtime, unable to leave a core to the panels here).

**D14. A checkpoint is four files with one record, and the policy fingerprint (S2-12) is in it** (`s2load.py`;
section 3.6). Written in this order, each through a temporary name: `model_<n>.zip`, `vecnormalize_<n>.pkl`,
`fingerprint_<n>.npz` (the 256 seeded observations, the deterministic float32 actions, **and the normalisation
statistics as plain arrays**), `checkpoint_<n>.json` (the three files' sha256 and bytes, the versions, the step
count, the run's `run_uuid` and settings hash, and the **run state** a resume starts from, D16). The JSON is written
last: its presence and matching hashes are what "an intact pair" means, on local disk and on Drive. Every load, in
the trainer, the monitor, the rows and the smoke, goes through `s2load.load_checkpoint`, which checks the hashes,
runs the scaffold's `checkpoint_pair_problem`, loads through `environments.shared.policy_loading.load_sb3_model`
with the schedule objects passed (S2-14), and reproduces the fingerprint within 1e-5. **An inference load unpickles
as little as it can**: it passes the two spaces (rebuilt from the record's layout) and `None` for the archive's
last-observation members as `custom_objects`, and normalises from the plain arrays, so only torch tensors and the
policy class reference cross library versions; the `.pkl` is read only by a training resume. For every checkpoint
of a check the launcher also writes `ckpt_<n>.tar` (the three binaries, deterministic, about 2.6 MB: the size class
the Drive tools are proven on), which `s2load import` unpacks here and verifies against the record. Rejected:
observations sampled from env rollouts (they would tie the fingerprint to one machine's floating point before the
archive exists).

**D15. The monitor is the bar's own code on a saved checkpoint, in a child process** (`monitor2.py`, rules M1-M6).
Its 18 episodes are six bar conditions of the run's set (`walk/nominal`, `walk/v0.4`, `turn/q90_left`,
`stop/two_step`, `nudge/forward`, `nudge/left`; mu 0.9), **each on its own seed** (`6001 + 3 x condition index +
repetition`: 6001-6018; with one seed slot and one lead-in per seed in the bar, three shared seeds would have made
the go or no-go rest on three starts), under `R5/<seed>@10ms` built from the checkpoint's files, judged by the
bar's judge and the look reading (D17). The same 18 under `R0L@10ms` are run once per run. The trainer saves the
checkpoint, starts `python -m designlab.stage2.train2 monitor ...` as a subprocess, waits, and reads its record; a
monitor that crashes is recorded and never stops training by itself; no episode is ever run twice. Each episode has
one of four outcomes: read; fell; **`unread`** (a policy row's episode that did not fall and that the bar could not
read: not succeeding in any reading, with its reason, `clock_behind` when the policy ran its clock slow so that the
command or the judged window was not reached, else `unreadable`, and the reference time reached); error (an
infrastructure fault only). Each record holds the failures per term, the mean clock factor and
the number of distinct start seeds. Survival is credited as progress: an episode's S is its survival time times
`min(1, reference time covered / (1.05 x survival time))`, which is the survival time itself for every row that
does not slow its clock (R0L exactly) **[differs: plan section 6's "mean survival"; section 7 item 4]**. Rejected:
Stage 1's in-process monitor on the training env (other conditions than the bar's, no look reading).

**D16. Kill and resume: every checkpoint is a boundary, in an uninterrupted run too, so a resumed run is the
uninterrupted one bit for bit, and a checkpoint carries the whole run state** (rules K1-K5). At the first rollout
start at or past each multiple of 250,000 steps (and at step 0) the trainer saves the checkpoint and then: seeds
SB3, torch, numpy and the vec env with `100,000 x seed + 100 x b` (b the multiple's index; env i resets with that
+ i), resets every env, zeroes `VecNormalize`'s running returns and sets the model's last observation and episode
starts from that reset. A resume loads the newest intact checkpoint **with `force_reset=False`**, calls
`learn(total_timesteps=budget - n, reset_num_timesteps=False)`, lets nothing reset or step the vec env in between,
and performs the same boundary at its first rollout start [scratch RV6]. Three more conditions make the equality hold
on the real env: every tracker is built at construction on the nominal model (a lazily built one would copy a
randomised damping, `control.py:1199-1200`, and make an episode depend on the process's history) [scratch RV7];
`PYTHONHASHSEED=0` in the trainer and its workers, and no iteration over a set in `r5lib`, `r5core`, `r5env`; a
"reset is history-free" test. **The run state** (task weights in force, the fix record, the monitor history, the
no-gain counter, the checks done, the settings hash) is written into `checkpoint_<n>.json`; a resume rebuilds
`run.json`'s live fields from it, moves anything later to `discarded_after_resume`, truncates `updates.jsonl` to
the resumed step, and runs any check that is due at the resumed step and has no record before it trains; every
decision after a check (the gain, G8b, the fix, a stop) is a function of the records alone, so a kill during a
monitor or after a fix replays to the same state. A job with a `run.json` and no checkpoint restarts from step 0 in
place, which by this construction is the same run. Cost: at most 8 x 1,500 env steps of cut episodes per 250,000
(under 5 %). Across machines or library versions nothing is asserted (S2-13 applies). Rejected: a stated tolerance
(any nonzero difference grows through PPO; there is no principled number).

**D17. The look reading is two modules outside the bar: `lookread.py` (the four measures on 50 Hz poses) and
`lookbar.py` (a row wrapper and a companion record); `bar.py` is not edited** (rules U1-U6; sections 3.5, 3.11).
The draft asked for a hook, 50 Hz poses and a trace key in `bar.py`; S2-17 forbids it after state A, WP B is built
without it, and `bar.reference_recorder` records no poses (`rec.poses = None`). Instead:
- Every row Track B runs is a registered row wrapped in `lookbar.LookRow`: `R0L` (the panels' R0 row), `R2L-<set>/42`,
  `R5/<seed>`, `R6/<seed>`, `R5G/<seed>`. The wrapper lends the built condition a recorder with 50 Hz poses for the
  one episode (an instance attribute on the worker's own `BarBuilt`, removed in a `finally`; the panels' turn item
  records the same poses already), runs the inner row unchanged, and computes the look reading from the recorder
  before the bar's judge sees it. Physics, seeds and the bar's episode record are those of the unwrapped row
  (tested: `R0L@10ms` against `R0@10ms`, digest per physics step and equal episode records).
- The reading goes to a sidecar file per episode, written before the bar writes its partial, and
  `lookbar.collect` joins the bar's finished record with its sidecars into a **companion record** beside it
  (`look/<row>/<condition>.json`, schema `s2-look-1`): the bar record's path, sha256 and key hash, per episode the
  four numbers, the reference's, the terms, the outcome and `succeeded_look`, then `counts_look` (reading 1) with
  the bar's own `counts` arithmetic, and for a policy row the version sets and fingerprint check (3.11). The bar's
  record keeps `counts_look: null`; reading 1 is read from the companion.
- The reference's values are the same functions applied to a kinematic playback of the episode's own track over the
  same window at the nominal commanded clock, through `bar._Played(track, run_dir).q(t_ref)` and `.loads` (the
  bar's own placement of the files in one world frame; a test pins the two methods). `_Played` refuses a periodic
  turning gait, so a track with the turn in place is never played; no item with a look reading holds one.
- Bob, tail phase and toe-off are segmented by the robot's own touchdowns and lift-offs (the 2 % load rule).
  "Success with the look" is the bar's `succeeded` and all four bands, for walk and turn items; for nudge and stop
  items reading 1 is reading 2, labelled. Reading 1 counts on the look sets `lookbar.LOOK_SETS = ("V6", "V6noF5")`;
  on any other set the numbers are recorded and `counts_look` stays null.
- Too few of the robot's own steps where the reference has enough is a **failed** bob, tail and toe-off term, not
  an error; an `unread` episode of a policy row (D15) is not succeeding. Both replace `STAGE2_75_PLAN.md` C8's "run again" for rows whose
  policy can stop stepping or slow the clock **[differs: C8 as extended by S2-16; section 7 item 4]**.

**D18. Code state B changes no module of any existing record key.** Round C adds files under `designlab/stage2/`
and edits none: not `bar.py`, `readings.py`, `setio.py`, `s2io.py`, `speedfit.py`, nothing under
`designlab/stage1/`. So `bar_sources()` and `panels.panel_sources()` are the same before and after state B, the
75 cm panels can run across it, a state-A record stays comparable with a state-B record of another row on the
same set (R2L on `F25F9cl` beside `R2/42`, `R2/43` and `R0`: same bar sources, same panel sources, same inputs),
and nothing has to be run again under S2-17's last sentence. State B is the commit at which `lookread.py`,
`lookbar.py`, `r5lib.py`, `r5core.py`, `s2load.py`, `s2rows.py`, `r2lfit.py` and `g8a.py` are frozen: they enter
the `row_sources` of Track B's rows. From state B until the Colab phases end none of them changes without the
affected rows being run again into a new directory. A fast test asserts that no round-A or Stage 1 file differs
from state A (`git diff --stat <state A> -- designlab/stage1 designlab/stage2/{bar,readings,setio,s2io,moves,sets,
setscreen,barreport}.py` is empty).

**D19. The notebook is thin and runs nothing heavy in the kernel** (`stage2/stage2_training.ipynb`; section 3.9).
Each code cell calls one step of `python -m designlab.stage2.nb` in a subprocess whose environment the cell passes
in full (`MUJOCO_GL=disable`, `PYTHONPATH=.`, the three thread variables at 1, `PYTHONHASHSEED=0`) with cwd FS, and
prints its lines. So the freshly pinned packages are the ones every job imports, the kernel never loads OSMesa or
EGL, a segfault in a child is a return code and not a dead kernel, and the same steps run here without Colab. The
kernel imports only the standard library and `google.colab`: the final cell's flush and disconnect are twelve
lines in `nb.py` (`nb.release_runtime`), with the repository's `disconnect_runtime` tried first inside a `try`
whose `except` still flushes Drive and unassigns the runtime (importing `environments` in the kernel pulls
gymnasium and numpy after the pins replaced them on disk). **Copied from the scaffold, adapted:** the setup cell's
Colab detection, legacy-gym purge and Git block (`notebooks/sb3_training.ipynb` cell 3, lines 36-58, with
`designlab` added to the "restart" refusal) and the Drive mount (cell 8, lines 8-21, with the Stage 2 root).
**Imported from the repository inside subprocesses, unchanged:** `environments.shared.curriculum.checkpoints.
checkpoint_pair_problem`, `environments.shared.policy_loading.{inspect_sb3_archive, load_sb3_model,
sb3_archive_load_preflight}`; the repository root is put on `sys.path` by one helper (`tbenv.repo_on_path`,
appended, never at position 0), because `environments` is not importable from FS with `PYTHONPATH=.`. **Not
taken:** everything bound to `ModelId`, stage manifests, gate verdicts and the result bundle;
`newest_intact_periodic_pair`; the kernel-wide `MUJOCO_GL=egl`; the GPU metadata. Rejected: training in the kernel
as the scaffold does.

**D20. The Drive bundle is a few plain `.tar` parts of at most 20 MB with a committed JSON manifest, built
deterministically** (`bundle.py`; section 3.10). One part per (group, body): the model files of a case, the arrays
of a set and body (split when over 20 MB). Tars are written in `tarfile.USTAR_FORMAT` with sorted names, mtime 0,
uid and gid 0, empty user and group names, mode 0644 and no compression, so the same files give the same bytes in
any process and under any umask. The manifest's **content block** (bundle id, bodies, models, sets, files, parts
with sha256 and bytes) is what is hashed, printed and quoted; the build's time, commit and versions sit beside it,
outside the hash, and upload receipts go to a separate `UPLOADS.json`, so the quoted hash never changes after the
build. Before the tool is final the lead tries the route once (run plan V0b): a 20 MB and a 5 MB dummy tar and a
170 kB file up and down through the connector. Fallbacks if an upload of that size fails, in order: (1) the same
content in parts of at most 5 MB (`--part-mb 5`); (2) only the model parts (3 MB in all) and the arrays **rebuilt
on Colab** by `sets build`, accepted only if every sha256 equals the manifest's and only after the smoke's Y8 found
them equal; a rebuild's verified arrays are kept on Drive (`inputs/<bundle_id>/rebuilt/`) and reused, not rebuilt at
each of the 13 launches; (3) with the owner's consent, the parts force-added to a data branch that is never merged
(`BUNDLE_SOURCE = "git:<ref>"`). Rejected: `.zip`; one archive.

**D21. R2L is a new module on the bar's conditions; `speedfit.py` is imported, not edited** (`r2lfit.py`; section
3.8, rules Q1-Q6). It reuses `tune.CMA`, `tune.Pool` (with its own worker factory), `tune.score`,
`tune.training_order`, `tune.selection_key`, `tune.to_unit`, `from_unit`, `repair_penalty`,
`speedfit.servo_delay`, `speedfit.start_record` and `speedfit.gains_vector`. Its evaluator builds five bar episodes
per evaluation through `bar.BarContext` and judges them with `bar.judge`; the delays and directions come from a
rotation table in blocks of six generations, written before generation 0 (S2-18); the selection's 10 ms cell is
counted first and always finishes. Its rows are named by their fit (`R2L-V6noF5/42`, `R2L-F25F9cl/42`) and resolved
through a committed index, because a row factory is given a body and a name, never a set (`bar.rows`).

**D22. Track B's outputs: live files under one ignored directory, finished records collected and committed.**
`stage2/tb/.gitignore` holds `work/`, `*.tar`, `*.zip`, `*.npz`, `*.pkl`, `*.mp4`, `*.tmp`, `*.part`, `.lock`,
`logs/`, `partial/`, `traces/`, `checkpoints/`, `parts/`. Every job that runs longer than a minute (a training run,
R2L, a look-set bar run, the stand-in Drive) writes under `stage2/tb/work/`, so the checkout's `git status` does not
change while it runs and no concurrent round-A record reads a dirty tree (round A's provenance counts untracked
files, `s2io.py:400-420`). When a job ends, `nb collect` copies its finished text records to their committed place
(`stage2/tb/runs/<run_id>/`, `stage2/tb/r2l/...`, `stage2/tb/bar/<run>/...`), verifies each by sha256, writes
`COLLECTED.json`, and they are committed at once. Short commands (gates, probes) write their one record directly.
Fresh clones are made outside the checkout. Never committed: archives, statistics, fingerprints, tars, clips, logs,
traces.

**D23. Development episodes** (rule X1; `STAGE2_75_PLAN.md` R4 as settled). Until task 3's first unit of the 75 cm
round has run, no R2-mode, policy or training episode runs on V0, F25F9 or F25F9cl: every test and acceptance of
round C that needs closed-loop episodes runs on the read-only S1 view or on a synthetic set in a temporary
directory. On a look set no closed-loop episode runs before G9-ref passed for that set and body; between then and
S2-4 the only closed-loop episodes on a look set are the gates', the R0L baseline's, the probes' (zero action,
noise, freshly initialised policies) and the acceptance run's tiny jobs, each listed in
`stage2/tb/dev_episodes.json` and none a record.

**D24. One commit per launch series; REPO_REF is its 40-hex SHA** (rules K6, N1). Commit C1 holds the code at
state B, the notebook, the frozen sets' JSON, the committed rules and `MANIFEST.json`. The acceptance run (3.10)
runs on C1 from a fresh clone of the remote. Commit C2 adds `ACCEPTANCE.json` and the steps sheet, both naming C1.
The sheet gives REPO_REF as C1's SHA (and the lightweight tag `s2-launch-1` on it for the notebook's URL); the
notebook refuses a non-smoke launch whose REPO_REF is not 40 hex digits, and on RESUME takes REPO_REF from
`QUEUE.json`. The working branch may move freely afterwards. A second launch commit is accepted by a run only with
`ALLOW_SOURCE_CHANGE`, a moved Colab image only with `ALLOW_VERSION_CHANGE` (both parameters of cell 1, default
False, recorded in the session).

**D25. One writer per run directory** (rule K7). The launcher writes `LIVE.json` (a session uuid, the host, the
time) with STATUS.json every 60 s and re-reads it before every copy or delete on Drive; if the uuid is no longer
its own it stops its job with HALT ("superseded"). A session with RESUME refuses when `LIVE.json` is younger than
10 minutes ("still running on another runtime"), unless `TAKEOVER = True` is set on the lead's word. With
background execution a lost browser does not end a runtime, so "disconnected" alone never means "resume".

## 2. Module layout

### 2.1 New files (round C edits no existing file)

| File | Purpose | Enters a record key as |
|---|---|---|
| `designlab/stage2/tbenv.py` | standard library only: the environment of every Track B child process, the `MUJOCO_GL` assignment, the repository root on `sys.path` | no |
| `designlab/stage2/schedules.py` | `LinearSchedule`, `ConstantSchedule`: module-level classes (S2-14) | no (training only) |
| `designlab/stage2/s2load.py` | the one loader; checkpoint files, integrity, fingerprint, version sets, `import` of a downloaded checkpoint | Track B rows' sources |
| `designlab/stage2/r5lib.py` | scripts over a frozen set: tracks, command, heading, phase, start states, reward references; the task table | Track B rows' sources |
| `designlab/stage2/r5core.py` | layout, observation, action mapping and filter, head overlay, reward terms, interference, the R6 tracker: shared by the env and the rows | Track B rows' sources |
| `designlab/stage2/r5env.py` | `R5Config`, `R5Tracker`, `R5Env`, `R6Env`, randomisation, starts, `make`, `throughput` | no (the rows never import it) |
| `designlab/stage2/g8a.py` | G8a's alignment and digest comparison (the trainer, the gates, the notebook and the tests call one function) | no |
| `designlab/stage2/lookread.py` | the look measures and bands on 50 Hz poses | Track B rows' sources |
| `designlab/stage2/lookbar.py` | `LookRow`, the `R0L` row, the companion record, in-process episodes for the monitor; CLI | Track B rows' sources |
| `designlab/stage2/s2rows.py` | policy rows `R5/<seed>`, `R6/<seed>`, `R5G/<seed>` for the bar runner | its own (row sources) |
| `designlab/stage2/vecpool.py` | `PoolVecEnv` and its worker loop | no |
| `designlab/stage2/train2.py` | the trainer and its probes; CLI (also the entry of workers and of the monitor child) | no |
| `designlab/stage2/monitor2.py` | the 18-episode monitor, gain rules, G8b, S2-10's record | no |
| `designlab/stage2/clip2.py` | one closed-loop clip from a monitor trace (run with `MUJOCO_GL=osmesa`; imports no torch) | no |
| `designlab/stage2/gates2.py` | G8a's and G9's records and the validation records; CLI | no |
| `designlab/stage2/r2lfit.py` | R2L's fit, selection and rows; CLI | R2L rows' sources |
| `designlab/stage2/bundle.py` | the Drive bundle: build, verify, unpack; CLI | no |
| `designlab/stage2/nb.py` | the notebook's steps, the queue launcher, `collect`, `hours`, the acceptance run; CLI; **standard library only at import** | no |
| `stage2/stage2_training.ipynb` | the notebook (9 cells) | |
| `stage2/tb/.gitignore`, `colab_steps.txt`, `requirements-colab.txt`, `run_trackb.sh`, `r2l/index.json`, `dev_episodes.json` | ignore rules; the steps sheet; the pins; the run chain; R2L's row index; the listed check episodes | |
| `stage2/tb/PREREG_TRACKB.md` | the rules (from the draft, after the lead's review; the lead commits it) | |
| `tests/test_stage2_{tbenv,schedules,s2load,r5lib,r5core,r5env,g8a,lookread,lookbar,s2rows,vecpool,train2,monitor2,gates2,r2lfit,bundle,nb}.py` | section 4 | |

Every module with a CLI uses round A's skeleton (thread variables before numpy, `main()` of the importable copy).
`train2.py`, `monitor2.py`, `gates2.py`, `g8a.py`, `r2lfit.py` and `lookbar.py` start with

```python
from __future__ import annotations
from designlab.stage2 import tbenv
tbenv.assign_gl("disable")     # os.environ["MUJOCO_GL"] = "disable" (assigned, never setdefault); pops PYOPENGL_PLATFORM;
                               # raises if mujoco is already imported with another backend
```

before any other import (`python -m designlab.stage2.train2` imports `designlab/__init__.py` first, which only
sets defaults, `designlab/__init__.py:15-16`; nothing has imported mujoco yet). `clip2.py` starts with
`tbenv.assign_gl("osmesa")`, which also sets `PYOPENGL_PLATFORM=osmesa`: mujoco's OSMesa backend raises unless that
variable is unset or `osmesa`, and a child of a `disable` process inherits `disable` otherwise.

### 2.2 Public APIs

```python
# tbenv.py   (standard library only)
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
def assign_gl(backend: str) -> None                     # "disable" | "osmesa"; see 2.1
def child_env(gl: str = "disable", base=None) -> dict   # a copy of os.environ with MUJOCO_GL (and PYOPENGL_PLATFORM for osmesa),
                                                        # the three thread variables = "1", PYTHONHASHSEED = "0", PYTHONPATH = "."
def repo_on_path() -> str                               # appends the repository root (FS/../..) to sys.path once; returns it
def study_dir() -> str                                  # FS, from this file's location

# schedules.py
class LinearSchedule:   def __init__(self, start: float, end: float); def __call__(self, progress_remaining: float) -> float
class ConstantSchedule: def __init__(self, value: float);             def __call__(self, progress_remaining: float) -> float
def describe(s) -> dict                      # {"class", "start", "end"} | {"class", "value"}: what settings record

# s2load.py   (imports torch and SB3 inside functions; calls tbenv.repo_on_path() before importing environments.*)
VERSION_KEYS = ("python", "numpy", "mujoco", "scipy", "torch", "stable_baselines3", "gymnasium", "cloudpickle")
FINGERPRINT_ROWS, FINGERPRINT_TOL = 256, 1e-5
def versions() -> dict                       # the imported modules' own __version__; torch's "+build" label kept here
def comparable(v: dict) -> dict              # torch's build label dropped (S2-12); what is compared, pins included
def checkpoint_files(job_dir, steps) -> dict # {"model", "vecnormalize", "fingerprint", "record"} paths
def list_checkpoints(job_dir) -> list[int]   # step counts that have a checkpoint_<n>.json, sorted by the number in the name
def fingerprint_obs(layout: dict) -> np.ndarray          # [256, obs_dim] float32, from default_rng([20261115, obs_dim, actor_dim])
def fingerprint(model, stats: dict, obs) -> np.ndarray   # deterministic actions, float32; stats = plain normalisation arrays
def stats_of(vecnorm) -> dict                            # {"obs_mean", "obs_var", "obs_count", "clip_obs", "epsilon"} plain arrays
def normalize(stats: dict, obs) -> np.ndarray            # clip((obs - mean) / sqrt(var + eps), -clip, clip), float32: VecNormalize's formula
def save_checkpoint(job_dir, steps, model, vecnorm, layout, run: dict, state: dict) -> dict
    # the four files, in order, each atomic; run = {run_uuid, run_id, job, settings_sha256}; state = the run state (3.6);
    # refuses an archive in which inspect_sb3_archive reports a bytecode member
def pair_problem(job_dir, steps) -> str | None           # missing file, sha256 or bytes mismatch, checkpoint_pair_problem's reason
def newest_intact(job_dir) -> tuple[int | None, list[dict]]   # newest step count without a problem; [{steps, reason}] of the skipped
def load_model(path, layout, env=None, lr: tuple | None = None)
    # load_sb3_model(path, algorithm="ppo", device="cpu", env=env, force_reset=False, custom_objects=...):
    #   training (env given): {learning_rate, lr_schedule: LinearSchedule(*lr), clip_range: ConstantSchedule(0.2)};
    #   inference (env None): those with ConstantSchedule(0.0), plus observation_space and action_space built from
    #   layout and None for _last_obs, _last_original_obs, _last_episode_starts, ep_info_buffer, ep_success_buffer
def load_vecnormalize(path, venv)                        # VecNormalize.load on a venv: a training resume only
def load_checkpoint(job_dir, steps, env=None, lr=None, tol=FINGERPRINT_TOL, expect_run: dict | None = None) -> dict
    # {"model", "stats", "vecnorm" (training only), "record", "versions_saved", "versions_here", "cross_version",
    #  "fingerprint_max_abs_diff"}; raises CheckpointError on a pair problem, on a run_uuid or settings hash other than
    #  expect_run's, or on a fingerprint beyond tol ("not judged": S2-12)
def keep(steps_all: list[int], monitor_every: int) -> list[int]      # K2: each check's and the newest two
def pack(job_dir, steps, out) -> dict        # ckpt_<n>.tar: the three binaries, deterministic (bundle's tar rules)
def import_tar(record, tar, into) -> dict    # unpack, check the three sha256, reproduce the fingerprint; the validation record
# CLI: python -m designlab.stage2.s2load {check JOBDIR N | pack JOBDIR N --out DIR |
#        import --record checkpoint_<n>.json --tar PATH --into JOBDIR [--write stage2/tb/validation/colab_archive.json]}

# r5lib.py
TASKS: tuple[TaskDef, ...]                   # name, plan builder, weight, monitor conditions it stands for (T1)
S1_TASKS: tuple[TaskDef, ...]                # stand, walk, brisk, stop on the S1 view (development and tests only)
class Script:                                # one plan on one plant: the track and its side tables. Every table is defined for
                                             # any t_ref >= 0 and continues past t_end exactly as Track.joint and Track.info do
                                             # (a periodic last segment loops, a hold holds: control.py:1048-1061)
    plan: list; track: control.Track; t_end: float; events: list[dict]     # events: setio.add_events' (causal commands)
    def command(self, t_ref: float) -> tuple[float, float]   # (v, w) at the references' own clock: section 3.3
    def heading(self, t_ref: float) -> float                 # the commanded heading about home: the integral of w over reference time
    def phase(self, t_ref: float) -> float                   # gait phase in [0, 1); 0 at a left lift-off
    def stride_s(self, t_ref: float) -> float;  def walking(self, t_ref: float) -> bool
    def steady(self, t_ref: float) -> bool                   # inside a periodic walking segment, one stride or more after its first frame
    def swing_s(self, t_ref: float) -> float                 # the planned swing time of the gait in force (reference s)
    def support(self, t_ref: float) -> np.ndarray            # [2] bool, the reference's
    def mid_swing(self, t_ref: float) -> np.ndarray          # [2] bool: the reference's foot between 25 % and 75 % of its swing
    def foot_rel(self, t_ref: float) -> np.ndarray           # [2, 3] feet relative to the root in the heading frame (FK of the file's q)
    def head_rp(self, t_ref: float) -> np.ndarray            # [2] the head body's world roll and pitch about home (FK of the file's q)
    required_margin: np.ndarray                              # [nj] decision 14's margin for this script, as bar.BarContext.built computes it
    def start_state(self, t_ref: float) -> tuple[float, np.ndarray, np.ndarray]   # (snapped t_ref, qpos, qvel at the files' own rate)
    def mid_spans(self) -> list[tuple[float, float]]         # reference-time intervals a mid-task start may take (T3); holds left out
class Library:
    def __init__(self, p: plant.Plant, body: str, set_name: str, rate: float, root=setio.ROOT, require_frozen=True)
    inputs: dict                             # files -> sha256 and entry sha256, the freeze index and manifest sha256, plant_inputs
    omega: float; crouch_frac: float         # control.omega0(p, q_home, crouch_frac=the set's): round A D16
    tasks: dict                              # name -> {"plan", "weight", "available", "reason", "seconds"}: the RESOLVED task table
    def script(self, task: str) -> Script;   def from_plan(self, plan: list) -> Script
    def describe(self) -> dict               # the resolved task table, absent tasks with their reasons, used files with hashes

# r5core.py
def layout(joints: list[str], heading: bool = True) -> dict   # {"blocks": {name: (i0, i1)}, "actor_dim", "obs_dim", "action_dim", "head_x", "nj"}
REWARD_WEIGHTS: dict;  REWARD_SCALES: dict;  REWARD_RULE: str # W1-W5, copied into every env description
class Core:                                  # everything the env and a policy row share, on one plant, tracker and script
    def __init__(self, p, cfg_core: dict, lg: legs.LegGap | None)     # cfg_core: action scales, rate, filter_hz, heading, reward
    def begin(self, clock, est, script: Script, head=None) -> None    # clock: () -> t_ref;  est: () -> dict(q, qd, gravity, gyro, foot_load, yaw)
    def set_history(self, actions3: np.ndarray, filter_state: np.ndarray) -> None    # an R6 mid-task start (D8)
    def actor_obs(self) -> np.ndarray        # float32 [actor_dim]
    def critic_obs(self, factors, push, floor) -> np.ndarray          # the env only
    def filter(self, action) -> np.ndarray   # clip, low-pass; shifts the three-action history
    def offsets(self, a_f) -> tuple[np.ndarray | None, float]         # (tk.residual or None, rate_factor)  (R5)
    def targets(self, a_f) -> tuple[np.ndarray, float]                # (q_target, rate_factor)             (R6)
    def substep(self) -> None                # per physics step: loads, flight, contact match, drag, touchdown speed, servo mean squares
    def reward(self, clock0: float) -> tuple[float, dict]             # the env only; section 3.4
    def interference(self) -> tuple[float, bool]                      # (penalty in [0, 1] on the moving pairs, any pair below zero)
    def clock_stats(self) -> dict            # mean, min of the clock factor so far; reference time covered
class HeadScript:                            # the -X head command: events (kind, t0, duration, hold, target); value(t) -> (yaw, jaw, yaw_goal, jaw_goal)
    @classmethod
    def drawn(cls, rng, seconds, gaze_prob, jaw_prob) -> "HeadScript | None"    # training (J4)
    @classmethod
    def fixed(cls, seed: int, seconds: float) -> "HeadScript"         # the bar's head script H1 (rule E5): a gaze train and one jaw event
class R6Tracker:                             # R6's own sensors, estimator, clock and p.command(q_target): as policy_rows.PolicyOnlyTracker
    def __init__(self, p, track, rate, model);  def reset(...);  def reset_at(...);  def step(self) -> None

# r5env.py
@dataclass(frozen=True)
class R5Config: ...                          # section 3.2; describe() -> plain dict (no path: the set by name and freeze hash)
class R5Tracker(control.Tracker):  def reset_at(self, t_ref, qpos, qvel, noise_seed) -> None     # D6
class R5Env(gym.Env):                        # mode "R5"
    def reset(self, *, seed=None, options=None)   # options: task | plan (+ seconds), start ("stance" | "mid" | float t_ref),
                                                  #   factors, push, head, episode_seed, seeds {"start", "noise"}, terminate_on_interference
    def step(self, action)
    def describe(self) -> dict;  def process_cpu(self);  def episode_stream(self);  def set_task_weights(self, w) -> None
class R6Env(R5Env)                           # mode "R6"
def make(mode: str, cfg: R5Config) -> R5Env  # the factory vecpool names: "designlab.stage2.r5env:make"
def env_configs(mode, body, set_name, n_envs=8, overrides=None) -> list[R5Config]     # env i's floor by D13
def throughput(mode, cfg, env_steps=2000, sigma=0.1, seed=0) -> dict

# g8a.py
def compare(row_digests: list[str], row_fell_step: int | None, env_digests: list[str], env_term_step: int | None) -> dict   # 3.7's rule
def stock(body, condition="walk/brisk", seed=3042, k=0, seconds=None) -> dict
    # the env on Library(p, body, "S1").from_plan(the stock condition's plan) against panels' R0 row inside speedfit.servo_delay
    # (no bar import): the development form and the fast tests' form
def bar_form(body, set_name, conditions, seeds, delays, cut_s=None) -> dict   # the env against bar.rows(body, ["R0@<ms>ms"]); imports bar lazily
def short(body, set_name) -> dict            # bar_form on walk/v0.4, seed 3042, k = 0 and 5: the trainer's and the notebook's line 3

# lookread.py   (numpy, mujoco for FK; imports metrics for the contact rule only)
BANDS = motion.BANDS                         # bob_rel 0.30, head_world_pitch_rms_deg 1.5, tail_phase_cycles 0.1, toe_off_rel 0.70
MIN_STEPS = 4                                # own steps and own lift-offs a window needs
def series(p, poses, loads, dt, every) -> dict            # t50, hip_mid_z, head_pitch_world, tail, foot_pitch [.., 2] by FK; contact, lift-offs, touchdowns
def reading(ser: dict, ta: float, tb: float) -> dict      # bob_pp_m, head_world_pitch_rms_deg, tail {phase_cycles, amp_rad}, toe_off_rad, steps, lift_offs
def compare(own: dict, ref: dict) -> dict    # ratios, differences, terms {bob, head, tail, toe_off}, not_read, too_few_steps, ok
def playback(p, track, run_dir, ta_ref: float, tb_ref: float) -> dict    # the reference's series over reference times, through bar._Played
def window(b, rec) -> tuple[float, float] | None                         # U2: a walk item's last 10 s; a turn item's 3 s after t_c + at_s; else None
def sources() -> dict

# lookbar.py   (registers "R0L" with panels.register_rows at import)
LOOK_SETS = ("V6", "V6noF5")
class LookRow(panels.Row):                   # wraps any row; runs/gated/hard_only are the inner row's
    def __init__(self, inner: panels.Row, sidecar: str | None = None, name: str | None = None)
    last: dict | None                        # the last episode's look block (in-process callers)
    def episode(self, ctx, cond, seed, relax=())           # 3.5: poses lent, inner.episode, reading, sidecar; returns the inner's value
    def describe(self) -> dict;  def sources(self) -> dict;  def spec(self) -> dict
    @classmethod
    def from_spec(cls, s: dict) -> "LookRow"
def rows(body, names, sidecar=None, stage1_run=bar.STAGE1_RUN) -> dict[str, panels.Row]   # bar.rows' names; every inner row a LookRow
def episodes(body, set_name, rows: dict, pairs: list[tuple[str, int]], traces=None) -> list[dict]
    # in-process: bar.context_for, bar.run_episode per (condition key, seed); catches bar.Unreadable; one companion episode each
def run(set_name, body, out, row_names, conditions=None, block=bar.HEADLINE_BLOCK, procs=2, **kw) -> dict
    # bar.evaluate(set_name, body, out, rows=rows(..., sidecar=<out>/.../look_partial), ...) then collect
def collect(out, set_name, body, block=bar.HEADLINE_BLOCK) -> dict     # the companion records (schema s2-look-1) and their summary
# CLI: python -m designlab.stage2.lookbar {run|collect} --set S --body B --rows R0L@10ms,R5/42@10ms --out DIR [--conditions ...] [--procs 2]

# s2rows.py   (registers "R5", "R6", "R5G" with panels.register_rows at import; never imports r5env or train2)
class S2Policy(panels.Policy)                # begin(tk, built, seed) / act(tk, k) through r5core.Core; load() via s2load.load_checkpoint
class R5Row(panels.TrackerRow)               # TrackerRow(name, "R2", gains=None, terms=(), ankle_cap=(), policy=S2Policy(...))
class R6Row(panels.Row)                      # its own episode: r5core.R6Tracker under control.run with the acting hook
def configure(policies: dict) -> None        # row name -> {"job_dir", "steps"}; default from $STAGE2_POLICIES (a JSON file)
def equivalence_check(job_dir, steps, body, set_name, condition, seed, k) -> dict   # a row episode against the env, digest per step
                                                                                    # (imports r5env inside the function only)

# vecpool.py   (the worker loop imports gymnasium and numpy only)
class PoolVecEnv(VecEnv):
    def __init__(self, specs: list[tuple[str, dict]], workers: int, worker_cmd: list[str] | None = None)
        # specs[i] = ("pkg.module:factory", kwargs); workers 0: SB3's DummyVecEnv in this process;
        # worker_cmd default [sys.executable, "-m", "designlab.stage2.vecpool", "--worker"]; train2 passes its own
def worker_main(address: str) -> int         # connect, receive its specs, build, serve reset/step/seed/env_method/get_attr/close

# monitor2.py
CONDITIONS = ("walk/nominal", "walk/v0.4", "turn/q90_left", "stop/two_step", "nudge/forward", "nudge/left")
SEEDS = {c: (6001 + 3 * i, 6002 + 3 * i, 6003 + 3 * i) for i, c in enumerate(CONDITIONS)}     # 6001-6018, one episode each
S1_CONDITIONS = ("walk/nominal", "walk/brisk")           # stock conditions for tiny runs on the S1 view (development only)
def run(job_dir, steps, body, set_name, out, k=5, tiny=False, also_k=()) -> dict    # the checkpoint's episodes; monitor_<n>.json
def baseline(body, set_name, out, k=5, tiny=False, also_k=()) -> dict               # R0L on the same episodes; r0l.json
def survival_credit(e: dict) -> float                    # M3's S of one episode
def gain(now: dict, earlier: list[dict], baseline: dict) -> dict                    # M3
def g8b(mon: dict) -> dict;  def fix_weights(mon: dict, weights: dict) -> dict      # M4
def decide(history: list[dict], baseline: dict, state: dict, rules: dict) -> dict   # everything read after a check, from records alone
def s2_10(final: dict, earlier: list[dict], baseline: dict) -> dict                 # S2-10's one record
# CLI (also reachable as "train2 monitor ..."): python -m designlab.stage2.monitor2 {run|baseline|s2-10} --job DIR --steps N ...

# train2.py
SOURCE_FILES: tuple[str, ...]                # the fixed tuple of 3.6 (every Track B module, bar.py, readings.py, setio.py, s2io.py,
                                             # motion.py, and the four repository files the reward and the loader execute)
def sources() -> dict                        # {files: {path: sha256}, sha256} over SOURCE_FILES + plant.sources_sha256() + panels.panel_sources()
def train(mode, body, seed, set_name="V6", out=None, job=None, steps=None, resume=False, workers=None,
          checkpoint_every=250_000, monitor_every=2_500_000, monitor_also_k=(), lr=None, config=None, ppo=None, gates=True,
          allow_version_change=False, allow_source_change=False, parent=None, tiny=False, run_id=None, log=print) -> dict
def probe_lr(bodies, set_name, out, rates=(1e-4, 3e-5, 1e-5), sigmas=(0.1, 0.03), updates=6, workers=None, parent=None) -> dict   # O3
def probe_sigma(bodies, set_name, out, sigmas=(0.0, 0.01, 0.025, 0.05, 0.1, 0.2)) -> dict        # O2 (reported)
def filter_choice(bodies, set_name, out, cutoffs=(10.0, 20.0)) -> dict                           # O4
# CLI: python -m designlab.stage2.train2 run --mode R5|R6 --body B --seed N [--set V6] --out ROOT [--job NAME] [--steps N]
#        [--workers W] [--resume] [--allow-version-change] [--allow-source-change] [--from-checkpoint JOBDIR:STEPS]
#        [--lr X] [--monitor-every N] [--monitor-also-k K,...] [--no-gates] [--tiny] [--run-id ID]
#      python -m designlab.stage2.train2 {probe-lr|probe-sigma|filter|preflight|worker ADDRESS|monitor ...}

# gates2.py
G8A_CONDITIONS: tuple                        # Z1's ten condition keys
def g8a(body, set_name, record=None) -> dict                     # g8a.bar_form over Z1's 40 episodes; the committed record
def g9_ref(body, set_name, record=None) -> dict;  def g9_run(body, set_name, bar_out, stage1b_reread=None, record=None) -> dict
def rates(body, set_name, floors=("hard", "robot_floor_carpet"), workers=(1, 2, 3, 4), steps=16_384, record=None) -> dict
def kill_resume(body, set_name, out, record=None) -> dict        # K3's settings, on the real env
def gl_start(out, record=None) -> dict;  def r6_start(body, set_name, record=None) -> dict
# CLI: python -m designlab.stage2.gates2 {g8a|g9-ref|g9-run|rates|kill-resume|gl-start|r6-start} --body B --set S --record PATH ...

# r2lfit.py   (registers one controller per entry of stage2/tb/r2l/index.json at import)
def rotation(seed: int, generations: int = 30) -> dict       # Q2's table: per generation the episode seed and, per slot, k and direction
class Evaluator;  def _evaluator(...)                         # tune.Pool's factory (top level)
def fit(body, set_name, seed=42, out=None, workers=1, resume=False, budget_core_h=2.4, max_generations=30, log=print) -> dict
def select(body, set_name, seed=42, out=None, workers=1, reported_core_h=0.4) -> dict     # Q4: the 10 ms cell first, always whole
def publish(body, set_name, seed=42, out=None, index="stage2/tb/r2l/index.json") -> dict   # copies the records, adds the index entry
def r2l_row(body, run_dir, name) -> panels.Row                # "R2L-<set>/<seed>": gains from the index entry's r2lg5.json (sha256 checked)
# CLI: python -m designlab.stage2.r2lfit {rotation|fit|select|publish} --body B --set S [--seed 42] --out stage2/tb/work/r2l [--workers 2] [--resume]

# bundle.py
def build(bundle_id, bodies, sets=("V6",), out=None, part_mb=20, root=None) -> dict   # parts + MANIFEST.json
def content_sha256(manifest: dict) -> str                                     # s2io.sha256_json of the content block alone
def verify(manifest, root, parts_dir=None) -> dict;  def unpack(manifest, parts_dir, root) -> dict
# CLI: python -m designlab.stage2.bundle {build|verify|unpack} ...

# nb.py   (standard library only at import; every step is also a CLI subcommand)
def session(run_id, drive, local, queue, repo_ref, commit, resume, params) -> dict     # SESSION_<k>.json; LIVE.json; K6's and K7's refusals
def inputs(bundle_id, drive, source="drive") -> dict                                   # line 2
def preflight(queue, set_name) -> dict                                                 # archive preflight, GL start, envs, G8a per body: line 3
def launch(run_id, drive, local, queue, set_name, budget, workers, tiny=False, clips=True,
           monitor_every=None, monitor_also_k=()) -> dict                              # the queue; STATUS.json; line 4 per job; the two
                                                                                       # monitor options go to every train2 run and into QUEUE.json
def smoke(run_id, drive, local) -> dict;  def finish(run_id, drive) -> dict
def release_runtime(reason: str, in_colab: bool, auto: bool, flush_drive: bool) -> None   # the final cell's flush and unassign (D19)
def collect(src, dst, kind) -> dict                                                    # D22: "run" | "bar" | "r2l"; COLLECTED.json
def hours(runs_root, out="stage2/tb/hours.json") -> dict                               # runtime-hours per run and session against the 220 h cap
def local_run(notebook, drive_dir, params, record=None) -> dict                        # the cells in this checkout (development; S1 view, --tiny)
def acceptance(notebook, ref, parts_dir, scratch, record) -> dict                      # 3.10: fresh clone of the remote + the bundle; kill; resume
```

### 2.3 What Stage 1's env and tracker cannot do today, and how each is got without editing a key module

| Map 05's item | How, without an edit | Where |
|---|---|---|
| Mid-task starts: `Tracker.reset` sets `tau_ref = 0` and starts from the track's first frame (`control.py:1292-1303`) | `R5Tracker.reset_at` (D6; scratch C3, RV5) | `r5env.py` |
| A track keeps the full state of its first frame only (`control.py:963`, `1021`) | the library keeps each file's `q`, `qd` (they are in every reference and transition file: `q [n, nq]`, `qd [n, nv]`) | `r5lib.py` |
| No delay wrapper covers `Plant.set_state` | the env refills the queue `speedfit.servo_delay` yields (D6) | `r5env.py` |
| 14-joint literals (`env.py:138-142`, `284-286`, `361-364`, `499-500`) | a new env with a layout per joint list (D1, D7); `env.py` untouched, its refusal and the test that pins it stay | `r5core.py`, `r5env.py` |
| The command is the reference's (`env.py:613-620`) | `Script.command` (D3) | `r5lib.py` |
| `TASKS` literals, `omega` from `refs/nominal.npz` at 0.03 H (`env.py:78-99`, `380`) | the task table over a set; `control.omega0(..., crouch_frac=the set's)` as the bar | `r5lib.py` |
| A stop event only for names in `transitions.TRANSITIONS` (`control.py:994`) | `setio.build_track` appends round A's command events | round A |
| Reward weights are module constants (`env.py:146-148`) | `R5Config.reward` | `r5env.py` |
| Joint damping and friction not randomised; `Tracker.damp` and `.fric` copied from the plant's live model at construction (`control.py:1199-1200`) | the env scales `m.dof_damping` and `m.dof_frictionloss` per episode and restores them; **every tracker is built at construction, on the nominal arrays**, and each reset asserts that the chosen tracker's `damp` and `fric` equal the nominal copies, so the feedforward keeps the nominal values: a deliberate model error, as the mass (`env.MASS_RULE`) | `r5env.py` |
| `PUSH_BOUND_M_S.get(body, 0.5)` gives 0.5 m/s to unlisted bodies (`env.py:465`) | `env.PUSH_BOUND_M_S[body]` (a `KeyError` stops the build) | `r5env.py` |
| Modes, run directories, G6 and monitor hard-coded in `train.py` | a new trainer (D2) | `train2.py` |
| No `MUJOCO_GL` in `train.py` | assigned at the top of every Track B entry point (2.1) | `tbenv.py` |
| The closure schedule (`policy.model_kwargs` returns `learning_rate=linear_schedule(...)`, `policy.py:154-162`), plain `PPO.load` (`train.py:497`; `policy_rows.py:601`) | `kw = policy.model_kwargs(...)`; `kw.update(learning_rate=LinearSchedule(...), n_steps=1024)`; one loader (D14); the trainer asserts before the first save that no schedule is a function | `schedules.py`, `s2load.py`, `train2.py` |
| Versions refused, no flag (`train.py:457-460`); `check_versions` reads torch another way (`policy_rows.py:412-419`) | `s2load.versions` and `comparable` on both sides; `--allow-version-change` (K4) | `s2load.py`, `train2.py` |
| `policy_rows` built around R3/R4 and env.py's widths | new rows on `panels.TrackerRow(policy=)` and `panels.Row` (the panels' own extension points, `panels.py:751-801`, `865-871`) | `s2rows.py` |
| `speedfit.Evaluator` hard-wired to five stock episodes (`speedfit.py:434-446`) | a new evaluator on the bar's conditions (D21) | `r2lfit.py` |
| `motion.BANDS` is only written out; no closed-loop trace adapter (map 07 item 7) | `lookread` on a recorder's 50 Hz poses; `lookbar.LookRow` lends the recorder (D17) | `lookread.py`, `lookbar.py` |
| `bar.reference_recorder` records no poses (its last statement sets `rec.poses = None`); `bar.judge` has no hook; `counts_look` is `None` in three places | the reference by `bar._Played`; the reading in a row wrapper and a companion record (D17) | `lookbar.py` |
| `tools/memguard.sh` protects one substring of a command line (default `designlab.stage1 train`) | `PROTECT="designlab.stage2.train2"`: the trainer, its workers and its monitor child all carry it (D13, D15) | run plan |

**No edit to any existing module is needed: none in a Stage 1, panels, tuner, speed-fit or bar key, and none in
round A's files.** What is relied on, as round A built it (pinned by tests in round C's files, so a later change is
seen): `bar.rows` wrapping any registered row in `bar.Delayed`; `bar.evaluate(rows=<dict of built
rows>)`; `bar.run_episode`, `bar.context_for`, `bar.Unreadable`, `bar.counts`, `bar.conditions`;
`bar.BarPlanner.resolve(cond)["plan"]`; `bar._Played.q` and `.loads`; `BarBuilt.recorder` being looked up on the
instance; `setio.build_track`, `read_inputs`, `add_events`, `stage1_view`, `frozen`; `s2io.write_json`,
`sha256_json`, `Lock`, `git_state`, `require_one_thread`. `bar._check_rule` restricts only the 75 cm round's three
sets, so a look-set run needs no `--task`. From the look work package: D5's entries, nothing else.

## 3. Algorithms and schemas

Notation: `dt` 0.002 s (a physics step); `dt_env` 0.02 s (10 physics steps, `env.ENV_SUBSTEPS`); `rate` 1.05 (the
commanded clock, `tune.COMMAND_RATE`); `c` the reference clock `tk.tau_ref`; `nj` the number of hinges (14 or 16);
`H` the hip height; `k` a servo delay in physics steps.

### 3.1 Observation and action layout (`r5core.layout(p.joints)`)

| Block | 14 joints | 16 joints (`-X`) | Content |
|---|---|---|---|
| `q` | 0:14 | 0:16 | the estimator's encoder angles, plant joint order |
| `qd` | 14:28 | 16:32 | encoder rates |
| `gravity` | 28:31 | 32:35 | `-est.R[2]` |
| `gyro` | 31:34 | 35:38 | |
| `foot_load` | 34:36 | 38:40 | over the nominal weight |
| `phase` | 36:38 | 40:42 | `sin, cos` of `2 pi Script.phase(c)` |
| `support` | 38:40 | 42:44 | the reference's support flags at `c` |
| `command` | 40:42 | 44:46 | `[v_cmd / 0.5, w_cmd / 1.0]`, `Script.command(c)` times `rate` |
| `heading` | 42:43 | 46:47 | `clip(wrap(yaw_est - yaw_home - Script.heading(c)) / 0.5 rad, -2, 2)`; `yaw_est` is `control._yaw` of the estimator's rotation **[differs: plan]** (D7) |
| `prev_actions` | 43:88 | 47:98 | the last three clipped, unfiltered actions, newest first (3 x (nj + 1)); zeros at a reset (R6 mid-task start: D8) |
| `head_command` | - | 98:102 | `[yaw now, jaw now, yaw goal, jaw goal] / 0.6` of the head script; zeros with the head neutral |
| **actor** | **88** | **102** | deployable inputs only |
| `root_vel` (3), `height` (1), `roll_pitch` (2), `dcm_error` (2) | 88:96 | 102:110 | as `env.py:652-654` |
| `joint_error` | 96:110 | 110:126 | true joints less the reference's (with the head overlay) |
| `heading_error` (1), `friction` (1) | 110:112 | 126:128 | as Stage 1 (the true yaw less the reference's) |
| `floor` (2) | 112:114 | 128:130 | one-hot hard, robot_floor_carpet |
| `push` (3) | 114:117 | 130:133 | as Stage 1 |
| `plant_factors` (5) | 117:122 | 133:138 | mass, kp, `k dt / 0.024`, joint damping factor, joint friction factor |
| `foot_error` (6) | 122:128 | 138:144 | `r_f - r_ref_f` of section 3.4, both feet |
| `ref_contact` (2) | 128:130 | 144:146 | the reference's support at `c - 0.024 rate` |
| **observation** | **130** | **146** | |
| **action** | **15** | **17** | `a[:nj]` joints in plant order, `a[nj]` the clock rate; `Box(-1, 1)` |

`PrivilegedCriticPolicy` takes any widths (`policy.py:69-141`); `actor_dim` goes to `policy.model_kwargs`. Without
the heading block (the lead's ruling, D7): 87/129 and 101/145, every later block one lower.

### 3.2 `R5Config`, reset and step

```python
@dataclass(frozen=True)
class R5Config:
    body: str;  set: str = "V6";  mode: str = "R5";  floor: str = "hard"
    rate: float = 1.05;  max_seconds: float = 30.0
    tasks: tuple = ();  task_weights: tuple | None = None       # (): every available task; describe() holds the RESOLVED table
    mid_start: float = 0.5                                      # T3
    randomize: bool = True
    friction: tuple = (0.4, 1.2);  mass: tuple = (0.9, 1.1);  kp: tuple = (0.6, 1.2)
    latency_steps: tuple = (0, 12);  noise_scale: tuple = (0.5, 1.5)
    joint_damping: tuple = (0.5, 1.5);  joint_friction: tuple = (0.5, 1.5)
    push_prob: float = 0.5;  push_walking: tuple = (0.0, 0.5);  push_standing: tuple = (0.0, 1.0);  push_s: float = 0.1
    head_events: bool = True;  gaze_prob: float = 0.5;  jaw_prob: float = 0.25      # -X plants only
    start_joint_noise: float = 0.01;  start_root_velocity_noise: float = 0.02       # tune's constants (asserted equal)
    action_leg_rad: float = 0.35;  action_other_rad: float = 0.15;  action_rate: float = 0.25;  filter_hz: float = 10.0
    heading_input: bool = True                                   # D7
    reward: tuple = REWARD_WEIGHTS_ITEMS                         # ((name, weight), ...): W2
    interference: str = "terminate"                              # "terminate" | "reward" | "off"
    set_freeze: str | None = None                                # the freeze's manifest sha256 the run started on
```

`describe()` holds no path: the set is named and pinned by its freeze, and `setio.set_dir` resolves it under
`s2io.STUDY`, so a run's settings are the same on local disk, on Colab and on Drive (map 05 hazard 4). It holds the
**resolved** task table (names, plans, weights, absent tasks with reasons) and, per used file, its sha256 and entry
sha256, so the same settings cannot resolve to another table after a later freeze appended an entry (K4).

**Construction** (mirrors `env.py:352-406`): `plant.build(body, floor)`; `nominal_model = deepcopy(m)`;
`env.setconst_changes(m)` must be empty; the nominal weight; the nominal copies of `geom_friction`,
`actuator_gainprm[:, 0]`, `actuator_biasprm[:, 1]`, `dof_damping`, `dof_frictionloss`;
`lib = r5lib.Library(p, body, set, rate)`; **one `R5Tracker(p, script.track, "R2", gains=None, terms=(),
ankle_cap=(), rate=rate, model=nominal_model)` per task of the resolved table, all built here, while the model is
nominal**; `legs.LegGap(body)`, with the fixed pairs' home distances checked (D10);
`core = r5core.Core(p, cfg, lg)`; spaces from `layout(p.joints, cfg.heading_input)`. A tracker for a plan given by
`options["plan"]` (the gates, the monitor's equivalence check) is built on first use **after** the nominal arrays
were restored (step 2 below) and cached by the plan.

**`reset(seed, options)`**, in this order (the Stage 1 line each step mirrors):

1. `super().reset(seed=seed)`; a seeded reset restarts the episode stream (`env.py:471-473`).
2. **The model's five nominal arrays are restored** (so nothing below, and no tracker built below, sees the previous
   episode's factors).
3. The script: `options["plan"]` (a bar condition's, with `options["seconds"]`), else `options["task"]`, else a
   draw by the task weights (`:476-477`). The chosen tracker's `damp` and `fric` are asserted equal to the nominal
   copies.
4. The start: `options["start"]`, else "mid" with probability `mid_start` when the script has `mid_spans()`, else
   "stance"; for "mid" the reference time is uniform on the union of the spans, then snapped to a frame.
5. `seconds = min((t_end - t_start) / rate, max_seconds)`; `n_steps = round(seconds / dt_env)` (`:478-481`). For a
   plan given by options: `n_steps = ceil(round(options["seconds"] / dt) / 10)`, so that the env covers every
   physics step a bar row runs (the bar's lengths are not multiples of 20 ms; 3.7).
6. The factors, drawn in this order when `randomize`: friction, mass, kp, latency_steps (integers, inclusive),
   noise_scale (Stage 1's five and order, `:437-445`), joint_damping, joint_friction; `options["factors"]`
   overrides (`:482-485`). Applied: Stage 1's scaling (`:447-458`), then `m.dof_damping[p.dofs] = damping0 * f`,
   `m.dof_frictionloss[p.dofs] = friction0 * f` (`mj_setConst` does not read them).
7. The push (`options["push"]`, else J3's draw) and, on an `-X` plant, the head script (`options["head"]`, else
   J4's draw) (`:487`).
8. The episode seed and its start and noise seeds (`:488-497`: `1,000,000 x reset seed + k`;
   `SeedSequence([ep, 0])`, `([ep, 1])`), unless `options["seeds"]` gives them (the gates and the monitor do).
9. Counters, the three-action history, the filter state and the reward's accumulators to zero (`:498-501`).
10. Leave the previous episode's delay (`:502`).
11. The start (`:503`): "stance": `tk.reset("stance", joint_noise=cfg.start_joint_noise, seed=seeds["start"],
    root_velocity_noise=cfg.start_root_velocity_noise, noise_seed=seeds["noise"])` after `tk.noise =
    SensorNoise(scale=noise_scale)`, `tk.residual, tk.rate_factor = None, 1.0` (`:714-719`). "mid":
    `tk.reset_at(t, qpos, qvel, seeds["noise"])` with `qpos` the file's frame plus uniform joint noise and `qvel`
    the frame's rates times `rate` plus uniform root-velocity noise, both from `default_rng(seeds["start"])`.
12. The delay (`:504-505`): `speedfit.servo_delay(p, k)` entered when `k > 0`; for a "mid" start its queue is
    refilled (D6).
13. The baselines for speed, yaw and the reference's yaw (`:506-509`); `core.begin(...)`; the observation.

**`step(action)`**, in this order (`env.py:530-564`):

1. `a = clip(action)`; `a_f += alpha (a - a_f)`, `alpha = 1 - exp(-2 pi filter_hz dt_env)` (`:531-532`).
2. R5: `off = a_f[:nj] * scale + head overlay`; `tk.residual = off if off.any() else None`;
   `tk.rate_factor = 1.0 + 0.25 * float(a_f[nj])` (`:723-725`). A zero action with the head neutral leaves `None`
   and exactly `1.0`.
3. Ten times: the push (`control._apply_push`, `:542-543`); `tk.step()` (`:544`); the step hook (`:545-546`: the
   gates' digests); `core.substep()` (loads, flight, contacts as `:547-552`; contact match, drag, touchdown speed,
   the servo mean squares).
4. The fall and any MuJoCo warning, read once (`:553-555`); the interference reading (D10).
5. Counters, the contact history (`:556-557`); the reward (`:558`); the action history shifts (`:559`).
6. `terminated` on a fall, or on interference at two env-step ends in a row when `cfg.interference` is
   "terminate" and the episode did not switch it off (`options["terminate_on_interference"] = False`: the gates);
   `truncated` at `n_steps` (`:560-561`).
7. `info` (Stage 1's keys plus `start`, `t_ref`, `clock_factor`, `interference`, `reward_terms`), the observation
   (`:562-564`).

R6 differs in steps 11 and 2-3 only, as `env.PolicyOnlyEnv` (`env.py:758-783`), through `r5core.R6Tracker`: its own
`NoisySensors` and `Estimator`, `p.command(q_target)`, its own clock (the track's tables continue past its end,
3.3); a "mid" start sets the plant, resets the estimator from the known state, fills the queue with the reference's
joint targets and sets the action history and filter state by D8.

**Randomisation rules** (J1-J5 in the draft): delay k uniform on the integers 0 to 12 (S2-9); stiffness x 0.6-1.2;
friction 0.4-1.2; trunk mass x 0.9-1.1 (`env.MASS_RULE`); joint damping and friction-loss, one factor each for all
hinges, x 0.5-1.5; IMU noise scale 0.5-1.5; a push with probability 0.5 in an episode longer than 4 s, at a time
uniform in `[2, seconds - 2]`, of `dv = u x env.PUSH_BOUND_M_S[body]`, u uniform on 0-0.5 if the reference walks at
that time and on 0-1.0 if it stands, direction uniform, force `true mass x dv / 0.1 s`; on `-X`: with probability
0.5 a train of gaze shifts (yaw target uniform on +-0.5 rad, minimum-jerk over 0.8-1.2 s, held 2-6 s, [look] 5.1)
and with probability 0.25 one jaw event (open to 0.3-0.6 rad over 0.4-0.6 s, held 1.0-1.5 s, closed the same way).
Everything but the start's and the sensors' noise comes from the env's own generator, as Stage 1
(`env.EPISODE_SEED_RULE`). Env i of PPO seed s resets at boundary b with seed `100,000 s + 100 b + i` (K1).

### 3.3 A script's side tables (`r5lib.Script`)

Built once per plan from the set's files (the same files `setio.read_inputs` hashed), one table per track segment.
**Every table answers any `t_ref >= 0` and continues past `t_end` exactly as `Track.joint` and `Track.info` do**
(`control.py:1048-1061`: a periodic last segment loops, a hold holds), because a clock up to 25 % fast passes the
end of a plan that the bar sized for a clock 5 % fast; tested at clock factor 1.25 on a plan ending in a periodic
segment and on one ending in a hold.

- **Command** `(v, w)` in the references' own units; the observation multiplies by `rate`. A hold: (0, 0). A
  periodic walk or turn in place: the entry's (`speed_m_s`, `yaw_rate_rad_s`). A start, a change, a start into the
  turn in place: gait b's command from the segment's first frame. A stop: (0, 0) from its first frame (the command
  instant of a causal stop). A finite turn of angle A at rate w0 from walk a: `v` stays a's; `w = sign x |w0|` for
  reference times in `[t0, t0 + A / |w0|)` and 0 after (a rectangular pulse whose integral is A). Nothing before
  a segment's first frame depends on that segment.
- **Commanded heading**: the integral of `w` over reference time from the track's start (piecewise linear; A after
  a finite turn). The `heading` input and the `heading` penalty read it.
- **Phase**: a periodic segment reads the file's `phase` at the local time, modulo 1, when the file holds it (a
  K-stride window), else `(tl / stride_s) mod 1`, `stride_s` the entry's `period_s`; a finite segment interpolates
  the file's `phase` over its `t`, modulo 1; a hold reads 0. `stride_s(t)`, `swing_s(t) = (1 - duty) x stride_s`,
  `walking(t)`, `steady(t)` and `mid_swing(t)` come from the same tables.
- **Start states**: the file's `q` and `qd` per frame (root included); `start_state(t)` returns the frame at or
  before `t`. `mid_spans()` are the non-hold segments' intervals inside `[first non-hold segment's t0 + 0.2 s,
  t_end - 2.0 s x rate]`: a mid-task start never lands in a hold (a hold has no file frame; the stop task's final
  hold and the start-stop task's 2 s hold are reached only by playing into them).
- **Reward references**, by forward kinematics of each file frame on the plant's nominal model at library build:
  `foot_rel [n, 2, 3]` (each foot body's position less the root's, rotated into the root's heading) and
  `head_rp [n, 2]` (the head body's roll and pitch in the world about home), interpolated linearly; `support`.
- **Required joint-range margin** `[nj]`: for a script without a finite turn `min(tune.WALK_RANGE_RAD (0.10 rad),
  the track's own smallest margin)` per joint; for a script that holds a finite turn, `max(min(tune.WALK_RANGE_RAD,
  the track's own smallest margin - tune.TURN_RANGE_SLACK_RAD (0.02 rad)), tune.AT_LIMIT_RAD (0.001 rad))` per
  joint, the expression of `bar.BarContext.built`. It never exceeds what the reference itself keeps, so a playback
  of the reference scores no `range` penalty.

### 3.4 Reward (`r5core.Core.reward`; rules W1-W5)

`r = 0.55 I_joint + 0.35 T + 0.15 I_foot + 0.10 I_contact + 0.10 I_head - sum_k w_k pen_k`, every term in [0, 1],
read on the true state at the env step's end (`c` the clock then, `c0` at its start):

| Term | Weight | Formula | Source |
|---|---|---|---|
| `I_joint` | 0.55 | `exp(-mean_j ((q_j - q_ref_j(c)) / 0.1 rad)^2)` over all nj hinges, the reference with the head overlay | Stage 1 (`env.py:578`) |
| `T` | 0.35 | `gaussian_tracking_reward(|v - v_ref|, |w - w_ref|)`: the CoM's speed along the previous heading and the trunk's yaw rate over the step, against the reference's own at `c0` times `rate` | Stage 1 (`:579-588`, `613-620`) |
| `I_foot` | 0.15 | `exp(-mean_f |r_f - foot_rel_f(c)|^2 / (0.10 H)^2)`, `r_f` the foot body's position less the root's in the root's heading frame | new (plan B.1 "foot imitation") |
| `I_contact` | 0.10 | the share, over both feet and the step's 10 physics steps, of steps in which the foot's contact (load above 2 % of the true weight) equals the reference's support at the clock **or** at the clock 24 ms x `rate` earlier (the largest randomised delay) | new ("contact imitation") |
| `I_head` | 0.10 | `exp(-((roll - roll_ref)^2 + (pitch - pitch_ref)^2) / (3 deg)^2)` of the head body in the world, about home | new ("head attitude in the world") |
| `duty` | 0.02 | Stage 1's (`:592-594`) with the band 0.55-0.65, or 0.55-0.72 while the commanded target speed is under 0.30 m/s | Stage 1; band by owner decision 2 **[differs: S1]** |
| `slip`, `flight`, `impact`, `tilt` | 0.02, 0.05, 0.02, 0.02 | Stage 1's (`:597-602`, `607-609`) | Stage 1 |
| `clearance` | 0.02 | the larger of (a) for a foot in the air (load under 2 %) between 25 % and 75 % of the planned swing time since **its own** lift-off and lower than 0.04 H: the shortfall over 0.04 H, the larger of the two feet; (b) 1 if, in any physics step of this env step, a foot is loaded while the reference has that foot in mid-swing both at the clock and at the clock 24 ms x `rate` earlier (a dragged foot, a stub step) | Stage 1's term, read as the bar reads **[differs: S1]** |
| `action_rate` | 0.03 | `min(1, mean_i (a_i - a_prev_i)^2 / 0.25)` on the clipped actions | new |
| `touchdown` | 0.02 | in a step in which a foot's contact begins: `min(1, max(0, v_down - 0.15 m/s) / 0.30 m/s)`, `v_down` the foot body's downward speed one physics step before (the plan's "impact": quiet feet) | new |
| `servo` | 0.05 | per drive that is not a leg drive (`~plant.leg_drives(p.drives)`: neck, tail, neck yaw, jaw): a bias-corrected exponential mean of the squared actuator force, time constant 1.0 s (`m_t / (1 - exp(-t / 1.0 s))`, so that it reads the mean so far from the first step and after a mid-task start); `min(1, max_d max(0, sqrt(mean_d) / sustained_d - 1) / 0.5)` ("neck and tail torque over their limit") | new |
| `interference` | 0.10 | `clip((3 mm - g) / 3 mm, 0, 1)`, `g` the smallest distance over the moving left-to-right pairs (D10) | new |
| `progress` | 0.10 | only while `Script.steady(c)` and `v_cmd > 0`: `min(1, max(0, 1 - v_bar / (rate x v_cmd)) / 0.10)`, `v_bar` the bias-corrected exponential mean (time constant: the stride period over `rate`) of the CoM's speed along the commanded heading; full at 10 % short | new, **[differs: plan]**: the walk item's 4.0 m |
| `heading` | 0.05 | `clip((|wrap(yaw - yaw_home - Script.heading(c))| - 3 deg) / 17 deg, 0, 1)` on the true yaw; zero in a hold | new, **[differs: plan]**: the bar's line and heading |
| `range` | 0.05 | `min(1, max_j max(0, required_j - margin_j) / 0.02 rad)`, `margin_j` the true joint's distance to its nearer limit, `required` the script's (3.3) | new, **[differs: plan]**: decision 14 |

The weights and the scales are `R5Config.reward` and `r5core.REWARD_SCALES`, written into every env description,
`run.json` and the settings a resume compares. G8b's fix does not touch them (M4). A zero action on a kinematic
playback scores every imitation term 1 and every penalty 0 (tested).

### 3.5 The look reading (`lookread.py`, `lookbar.LookRow`; rules U1-U6)

- **Where it runs.** `LookRow.episode(ctx, cond, seed, relax)`: `b = ctx.built(cond)`; for a bar walk or turn item
  it sets `b.recorder = lambda: metrics.Recorder(legs_every=tune.LEGS_EVERY)` on the instance (50 Hz full `qpos`;
  `BarBuilt.recorder` gives poses to turn items only), calls `self.inner.episode(...)`, deletes
  the attribute in a `finally`, computes the reading from the returned recorder, keeps it in `self.last`, writes
  the sidecar when it has a sidecar root, and returns the inner row's `(b, out, rec)` unchanged, so
  `bar.run_episode` judges and traces exactly what it would have. The wrapper never touches the plant.
- **Series** (`lookread.series`). From the recorder's 50 Hz poses, by `mj_kinematics` on the plant's model:
  `hip_mid_z` (the mean height of the two hip joints' anchors, `look.py:600-624`), `head_pitch_world` (as
  `look.arrays`, `look.py:546`, `:646`), each foot's pitch about its own lateral axis (toe down positive, the sign
  of `foot_pitch_solved`), and the tail-yaw angle; contact and its lift-offs and touchdowns from the recorder's
  500 Hz loads by the 2 % rule with 20 ms debounce (`readings.contact_flags`).
- **Window** (U2): a walk item's judged last 10 s; a turn item's 3 s of walking after `t_c + at_s`; nudge and
  stop items are not read.
- **Four numbers** over the window: (1) *bob*: the mean, over the steps that lie wholly inside it (a step runs
  from one touchdown of either foot to the next), of max less min of `hip_mid_z`; (2) *head*: the RMS of
  `head_pitch_world` about its mean, in degrees; (3) *tail phase*: the least-squares fit `tail = A sin(2 pi phi +
  psi) + c` (`look._fit_phase`'s form) with `phi` the robot's own phase, 0 at each of its left lift-offs and linear
  in time between them; `psi / 2 pi` in cycles and `A`; (4) *toe-off*: the mean, over the lift-offs inside the
  window, of the lifting foot's pitch at the lift-off instant (interpolated between poses).
- **The reference's numbers** (`lookread.playback`): the same functions on the poses `bar._Played(track,
  run_dir).q(t_ref)` gives at 50 Hz over the reference times the window covers at the nominal commanded clock, with
  contact from `_Played.loads`; cached per condition in the worker.
- **Terms** (U3): `bob`: `|own / ref - 1| <= 0.30`; `head`: `own <= 1.5 deg`; `tail`: the wrapped difference of
  the two phases at most 0.1 cycle, not read (passes, labelled) when the reference's amplitude is under 0.02 rad;
  `toe_off`: `own >= 0.70 x ref`, not read when the reference's is under 0.02 rad (a set without F5).
- **Too few steps** (U5): when the reference's playback of the window has at least 4 steps and 4 lift-offs and the
  robot has fewer, `bob`, `tail` and `toe_off` **fail** (`too_few_steps: true`); `head` is still read. When the
  reference itself has fewer, it is a definition fault: an error, which G9-ref must have caught.
- **Outcome of an episode** (U6), decided by `lookbar` from the bar's episode and the sidecar: `read`; `fell`;
  `unread` when the bar reports the episode unreadable (`bar.Unreadable`: the trace ends before the command or the
  window, or the window cannot be read), it did not fall, and the row is a policy row (`R5`, `R6`, `R5G`): not
  succeeding in any reading, with `reason: "clock_behind"` when its mean clock factor is below 1 and
  `"unreadable"` otherwise, and `t_ref_reached`; `error` otherwise (an infrastructure fault, or an unreadable
  episode of a row without a policy, `R0L` or `R2L`, for which C8 stands unchanged).
- **Readings** (U4): `look_ok` is the conjunction of the four terms; `succeeded_look = succeeded and look_ok` on
  walk and turn items, `= succeeded` on nudge and stop items, with `reading_1: "defined for walks and turns"`.
  Printed beside, never a verdict: `succeeded_but_A1 and look_ok`.

**Sidecar** (`<sidecar root>/<row>/<condition key>/<seed>.json`, written atomically before the bar's partial):
`{schema: "s2-look-ep-1", row, condition, seed, steps, fell, survival_s, clock {mean_factor, min_factor,
t_ref_reached, t_ref_planned}, window, own, reference, terms, not_read, too_few_steps, look_ok, policy_check}`.
**Companion record** (`<out>/<set>/<body>/<block>/look/<row>/<condition>.json`, schema `s2-look-1`): `{bar_record
{path, sha256, key_sha256, status}, set, body, block, row, condition, look_set: bool, bands, rules, episodes
[{seed, outcome, succeeded, succeeded_but_A1, succeeded_look, succeeded_but_A1_look, look {...}, clock {...}}],
counts_look, counts, counts_but_A1 (each by bar.counts on these episodes, unread episodes counted as survived
and not succeeded), failures_by_term, policy {versions_trained, versions_judged, cross_version,
fingerprint_max_abs_diff, fingerprint_tol, checkpoint, source_change}, sources, at}`. `collect` refuses an episode
whose sidecar is missing or whose `steps`, `fell` or `survival_s` differ from the bar's episode ("delete that
condition's partials and run again").

### 3.6 The trainer (`train2.py`)

**Run directory** (the same layout locally and on Drive; `<root>` is `--out`):

```
<root>/jobs/<job>/run.json                    job = "<mode>-<body>-s<seed>", e.g. R5-75-M-H-10-X-s42; a fine-tune names its own
<root>/jobs/<job>/updates.jsonl               one line per PPO update (lesson 16)
<root>/jobs/<job>/progress.json               rewritten at every rollout start (the launcher reads it)
<root>/jobs/<job>/gates/g8a.json, rate.json
<root>/jobs/<job>/checkpoints/model_<n>.zip, vecnormalize_<n>.pkl, fingerprint_<n>.npz, checkpoint_<n>.json
<root>/jobs/<job>/monitor/r0l.json, monitor_<n>.json, traces/..., clip_<n>.mp4, clip_<n>.json
```

**Start.** Settings = `{mode, body, seed, set, set_freeze {index, manifest_sha256}, inputs (the library's: each
used file's sha256 and entry sha256, and panels.key_context(body)["plant_inputs"]), tasks (the resolved table),
n_envs 8, ppo (design 2's values as used, with n_steps 1024), lr {start, end}, log_std_init, checkpoint_every,
monitor_every, budget, env_configs [8 x R5Config.describe()], layout, parent, tiny, validation}`; rule texts are
stored under `rules`, outside the settings (map 05 hazard 3). A new run draws a `run_uuid`, then: G8a's short form
(`g8a.short`: `walk/v0.4`, seed 3042, k = 0 and 5; about 15 s; "G8a fails: no run": the job exits with code 3 and
writes `stopped`); the vec env (D13); `VecNormalize(norm_obs, norm_reward, clip_obs 10, clip_reward 50, gamma
0.99)`; then

```python
kw = policy.model_kwargs(actor_dim, log_std_init=math.log(0.1))     # returns n_steps, clip_range and a closure learning_rate
kw.update(learning_rate=schedules.LinearSchedule(lr, 0.3 * lr), n_steps=1024, clip_range=0.2)
model = PPO(env=venv, seed=seed, device="cpu", **kw)
assert not inspect.isfunction(model.learning_rate) and not inspect.isfunction(getattr(model.lr_schedule, "__wrapped__", model.lr_schedule))
```

`torch.set_num_threads(1)`; the monitor's R0L baseline (once); the callback's first rollout start saves checkpoint 0
and performs boundary 0, so a fresh start and a resume from step 0 are the same code path.

**Sources.** `train2.SOURCE_FILES` is a fixed tuple: every Track B module of 2.1, `bar.py`, `readings.py`,
`setio.py`, `s2io.py`, `motion.py`, and the repository files the reward and the loader execute
(`environments/shared/direction_commands.py`, `command_frame.py`, `policy_loading.py`,
`curriculum/checkpoints.py`). Its hash with `plant.sources_sha256()` and `panels.panel_sources()` is taken once per
session, recorded, passed to the monitor child, and each child refuses to run if its own hash differs (the trainer
does not import what its children run; a local run's checkout must not move under it).

**Resume** refuses, in this order, each with a message naming the flag if one exists: no `run.json`; a stopped
run; other settings; other inputs (a used file's sha256 or entry hash, or the plant inputs: no flag; a **later
freeze** of the set is accepted when every used file and entry hash and the resolved task table are unchanged, and
both freeze indices are recorded); other sources (`--allow-source-change`); other library versions
(`s2load.comparable` against the last session's; `--allow-version-change`); a checkpoint whose `run_uuid` or
settings hash is not the run's; a fingerprint that is not reproduced within 1e-5 (always checked; with a version
change it is S2-13's condition). No intact checkpoint is not a refusal: the job restarts from step 0 in place
(`restarted_from_zero: true` in the session). Otherwise:

1. `n, skipped = s2load.newest_intact(job)`; `ck = load_checkpoint(job, n, env=venv, lr=(start, end))`, which
   loads with `force_reset=False`; `VecNormalize` from its `.pkl`.
2. **The run state** is taken from `ck["record"]["run_state"]`: `run.json`'s `monitor`, `fix`, `gates.g8b*`,
   `stopped`, the no-gain counter and the env configs' task weights are rebuilt from it; entries of `run.json` with
   `at_steps > n` move to `sessions[k].discarded_after_resume`; `updates.jsonl` is truncated to lines with
   `steps <= n` (the cut lines go to `updates.discarded.s<k>.jsonl`).
3. If n is a check's step and `monitor/monitor_<n>.json` is missing, or does not name this checkpoint's three file
   hashes, the monitor runs now; then `monitor2.decide` is applied to the records, exactly as the uninterrupted run
   applies it after that check (so a kill during the monitor, or after a fix and before the next checkpoint,
   replays to the same weights and the same `fix` record).
4. `model.learn(total_timesteps=budget - n, reset_num_timesteps=False, callback=...)`; nothing resets or steps the
   vec env before the callback's first rollout start, which performs boundary `n / checkpoint_every` (D16).

A flagged session records both version sets or both source hashes and `version_change` / `source_change: true`,
and every later record of the run carries the flag.

**The callback, at a rollout start with n steps done:**

1. If n has reached the next multiple of `checkpoint_every` (0 included): `s2load.save_checkpoint` (four files; the
   record holds the run state as it is before this step's check), then delete what `s2load.keep` no longer lists
   (K2), then `run.json`.
2. If n has reached the next multiple of `monitor_every`: `train2 monitor` on that checkpoint in a subprocess
   (`tbenv.child_env("disable")`, the session's sources hash); append its record; `monitor2.decide`: the gain (M3);
   at the first check at or past 5M G8b (M4) and, if it fails, the fix in place:
   `venv.env_method("set_task_weights", w2)`, `doc["fix"] = {rule, at_steps, weights_before, weights, tasks,
   monitor}`, the settings' env configs replaced by the fixed ones with the earlier ones kept under the fix; at the
   first check at or past the fix + 5M G8b again, a second failure sets `stopped`; from 20M M5's stop rule; the
   `kl_cut` flag (O3). The trainer waits for the monitor and for nothing else: clips are the launcher's (3.9).
3. If step 1 saved, or this is a resumed session's first rollout: the boundary (D16, K1).

`_on_step`: after the first rollout, `gates/rate.json` (`first_rollout`: env steps per wall second, CPU seconds of
the trainer and of the workers, vCPUs, workers); after the third rollout `first_three` (two updates included:
S2-8's rate). After `learn` returns: a final checkpoint if n has none, a final monitor, `status`.

**`updates.jsonl`** (one line per update): `{session, update, steps, minibatch_steps_run,
minibatch_steps_possible, approx_kl, clip_fraction, explained_variance, value_loss, std_mean, lr, ep_len_mean,
ep_rew_mean, wall_s}`; `minibatch_steps_run` from the wrapped `optimizer.step` [scratch C4].

**`run.json`** (`schema: "s2-run-1"`): `run_uuid, run_id, job, mode, body, head ("current" | "X"), seed, set,
settings, settings_sha256, rules {...texts}, layout, sessions [{index, started_at, ended_at, commit, dirty (own
outputs excluded), versions, sources, workers, vcpus, torch_threads, resumed_from, restarted_from_zero,
skipped_checkpoints [{steps, reason}], fingerprint_max_abs_diff, version_change, source_change, freeze_index,
boundary_seeds, discarded_after_resume}], checkpoints [n], kept [n], gates {g8a, rate, g8b, g8b_after_fix}, monitor
[{at_steps, record, checkpoint_sha256, N, S, N_reading2, N_reading3, N_but_A1_look, mean_clock_factor,
failures_by_term, gain}], baseline {record, ...}, fix, flags {kl_cut}, stopped {at_steps, reason}, parent {job,
run_uuid, steps, files {sha256}, fingerprint_max_abs_diff}, updates_file, status ("running" | "done" | "partial" |
"stopped_g8a" | "stopped_g8b" | "stopped_no_gain"), budget, num_timesteps`.

**`checkpoint_<n>.json`** (`schema: "s2-ckpt-2"`): `{run_uuid, run_id, job, settings_sha256, steps, files {model |
vecnormalize | fingerprint: {name, bytes, sha256}}, versions, layout {actor_dim, obs_dim, action_dim, nj, heading},
fingerprint {rows 256, seed [20261115, obs_dim, actor_dim], obs_sha256, tol 1e-5}, run_state {task_weights, fix,
monitor, gates, no_gain_count, checks_done, stopped, flags}, saved_at}`. `fingerprint_<n>.npz`: `obs [256, obs_dim]
float32`, `actions [256, action_dim] float32`, `obs_mean`, `obs_var` (float64), `obs_count`, `clip_obs`, `epsilon`.
The observation table: per row, each actor block drawn from a fixed distribution of its kind (angles 0.3 z, rates
2 z, gravity (0, 0, -1) + 0.05 z renormalised, gyro 0.5 z, loads uniform on 0-1, phase a point of the unit circle,
support bits, commands uniform on their ranges, the heading 0.3 z, previous actions 0.1 z, the head command
uniform), the critic's columns zero (the actor does not read them). The record also counts how many of the 256
rows the normalisation clips.

**Run level.** `train2 run` writes its job's files and exits 0 (done or stopped by a rule), 3 (G8a failed) or
another code (a failure). The run-level files (QUEUE.json, SESSION files, STATUS.json, LIVE.json, DONE.json or
HALT.json, the logs) are `nb launch`'s, which also runs here with a local directory as its "Drive" (S2-4 and the
local 60 cm run go through it, under `stage2/tb/work/`). `--tiny` (tests, the acceptance run, the smoke's kill and
resume): `n_steps` 64, 4,096 steps, a checkpoint every 1,024, a check every 2,048 **with each condition's first
seed and full-length episodes** (6 + 6 judged episodes: under a minute), G8b's reading point scaled to the first
check and its second reading to the next, G8a's short form cut to 2 s; `tiny: true` is in the settings, so a tiny
run can never be resumed as a real one. On set `S1` (development, `--tiny` only) the monitor runs
`monitor2.S1_CONDITIONS` judged by the panels' judge.

**A fine-tune** (`--from-checkpoint JOBDIR:STEPS --job NAME --steps 5000000`): a new job directory and `run_uuid`;
the parent's checkpoint through `load_checkpoint` (same layout required); its normalisation statistics loaded and
updated further; `log_std` as saved; a constant learning rate: the parent's schedule value at the parent's step,
lowered to the largest probed rate at which every minibatch step of 6 updates runs from the parent's checkpoint
(`probe_lr(parent=...)`, K8: the parent's sigma is smaller than 0.1); `parent` recorded. The new job has its own
settings, boundaries and monitor.

### 3.7 G8a and G9 (`g8a.py`, `gates2.py`)

**G8a's alignment** (per episode: a set, a body, a bar condition, a panel seed, a delay k):

| Quantity | The bar's row `R0@<2k>ms` | The R5 env with the zero action |
|---|---|---|
| plant | `ctx.built(cond).p` (hard floor; the condition's friction) | its own `plant.build(body, "hard")`; factors `{friction: cond's, mass 1, kp 1, damping 1, friction-loss 1, latency_steps k, noise_scale 1}`; `randomize False` |
| track | `b.track`, from `BarPlanner.resolve(cond)["plan"]` | `lib.from_plan(the same plan)`: the same `setio.build_track`, files and omega rule |
| start | `control.run(start="stance", seed=s["start"], joint_noise=0.01, root_velocity_noise=0.02)`, `s = panels.episode_seeds(cond, seed)` | `options["start"] = "stance"`, `options["seeds"] = {start: s["start"], noise: s["noise"]}`; the config's start noise asserted equal to `tune`'s |
| sensor noise | none (R1 mode reads the true state and sets nothing) | R2 mode's sensors and estimator, which only read the plant |
| clock | `rate = panels.RATE` | `cfg.rate = 1.05`, `rate_factor` exactly 1.0 |
| delay | `bar.Delayed`: `speedfit.servo_delay(p, k)` around the episode; the queue refilled with the reset's targets | the same function, entered after the reset: k copies of the same targets |
| push | `b.push(seed)` | `options["push"]` = that tuple |
| length | `n_row = round(seconds / dt)` physics steps (`control.py:2302`), or to the fall, read after every physics step | `ceil(n_row / 10)` env steps (3.2 step 5); falls read at env-step ends; interference termination off, its would-be step recorded |
| digest | sha256 of `qpos, qvel, ctrl, sensordata` after every physics step (`env._state_digest`), through a wrapper lent to the built condition's `recorder` for the episode | the same, through the env's step hook |

An episode is **identical** (Stage 1's shorter-horizon reading, `policy_rows.EQUIVALENCE_RULE`, because the bar's
lengths are not multiples of 20 ms): (1) no digest differs over the physics steps both ran, which by the env's
length is every step the row ran; (2) the same end: if the row fell at physics step j, the env terminates in the
env step that holds j; if the row did not fall, the env does not terminate in any env step that lies wholly inside
the row's `n_row` steps (the env's last, partial step, up to 9 physics steps past the row's end, is not read).
**G8a passes for a body and set when every episode is identical** over: the eight gated mu 0.9 conditions the set
has references for, `walk/v0.4_mu0.54` and `walk/nominal`; panel seeds 3042 and 3043; k = 0 and k = 5 (40 episodes,
rule Z1). Record `stage2/tb/gates/<set>/<body>/g8a.json`: `{rule, body, set, set_freeze, episodes [{condition,
seed, k, reference_steps, env_steps, compared_steps, first_difference, reference_fell_step, env_terminated_step,
ends_alike, identical, interference_first_env_step}], identical, of, not_run [...], sources {panels, bar, track_b},
key_context, versions, blas_threads, commit, dirty, at}`. The trainer's own G8a and the notebook's line 3 are
`g8a.short` (one condition, one seed, both delays), the same function.

**G9** (S2-11 as ruled) is two records per body and head, in a fixed order (rule Z2):

*G9-ref* (`stage2/tb/gates/<set>/<body>/g9_ref.json`; kinematic only; **must pass before the first closed-loop
episode on that set and body**):
(a) `bar refcheck` (S2-11 (c): every frozen reference played kinematically at the commanded clock passes its own
item's readings; the quick turn also reaches 80 degrees at the slowest clock), cited by path and sha256;
(b) the look reading agrees with the look ladder: for each rung of `stage2/look/metrics.json` that the body has
(the two plain bodies: nominal, v040, fr05 where in the ladder), `lookread`'s numbers of that walk's playback against
the ladder's own (bob within 5 %, head pitch RMS within 0.1 deg, tail phase within 0.01 cycle, toe-off within
0.02 rad), and each such walk entry's `look_sha256` equals `stage2/look/L1_FREEZE.json`'s hash for that body and
rung (the set's rungs are the files the owner picked); for the rungs the ladder does not hold (v0.3, v0.35, v0.45,
v0.5 where absent) the playback's four numbers are recorded and must be finite with at least 4 steps in the window;
for an `-X` body, whose rungs are in no ladder: the `-X` walk's playback against the plain body's playback of the
same rung within the same four limits (the head is neutral, so the look is the plain body's), else "differs" with
the numbers, which fails (b);
(c) negative controls on a playback, so that each band is shown to bite: the pelvis height's oscillation scaled by
0.5 fails `bob` only; a 3 degree sinusoid added to the neck pitch fails `head` only; the tail delayed by 0.2 cycle
fails `tail` only; the foot pitch zeroed fails `toe_off` only; the unchanged playback passes all four with ratio 1
and difference 0; a playback cut to 3 steps gives `too_few_steps`;
(f) references no bar item plays (the turn in place with its start and stop, the changes to rungs other than
0.40 m/s): their `setscreen` records cited, and each task script of the library builds with joints continuous at
every join (the largest jump of the track's joint targets across a join under 1e-6 rad at the join's own frames)
and plays 2 s under the zero action kinematically (the tracker's targets finite); the turn in place is **not**
played by `bar._Played` (it refuses a periodic turning gait) and has no look reading.
If (b) fails, U1's definitions go back to the lead before any look-set episode.

*G9-run* (`g9_run.json`; after the R0L records of that set and body exist):
(d) every episode of the body's `R0L@10ms` companion records holds each reading of its item and, for walk and turn
items that reached their window, a finite look reading; both toe-clearance readings are present in every 10 ms bar
record (S2-11 (b), asserted for every body and head, `-X` included);
(e) for `60-N-T-10` and `60-N-T-10-X`: round A's `bar reread` record of Stage 1b's 10 ms turn episodes of
`60-N-T-10` (each trace identical to its committed sha256), cited by path and sha256 (S2-11 (a); the `-X` body has
no Stage 1b episodes and cites the plain body's).

### 3.8 R2L (`r2lfit.py`; rules Q1-Q6)

- **Mix**: five bar episodes per evaluation on the fit's set (mu 0.9), in slots j = 0-4: `walk/v0.4`,
  `turn/q90_<left|right>`, `stop/two_step`, `nudge/<direction>`, `walk/nominal`; each built by `bar.BarContext`
  and run as the bar's R2 row runs it (`panels.TrackerRow` mode R2 with the candidate's gains) inside
  `speedfit.servo_delay`, judged by `bar.judge`.
- **Rotation** (S2-18; `rotation()` writes `rotation.json` before generation 0), in **blocks of six generations**
  (block `B = g // 6`): generation g runs episode seed `tune.training_order(seed)[g]`; slot j runs at delay
  `k = (g + j) mod 6` (within a generation five different delays; within a block every slot meets all six delays);
  the turn's side is `("left", "right")[B mod 2]`; the nudge's direction is `("forward", "left", "backward",
  "right")[B mod 4]`. So every (kind, direction) meets every delay, 10 ms included: left and right turns in any two
  consecutive blocks, the four nudge directions in any four (24 generations). The draft's `g mod 2` and `g mod 4`
  tied direction to the delay's parity (right turns and sideways nudges never met 10 ms: lesson 17's fault). A fit
  cut by its budget ends at a whole block (Q5). Start and noise seeds: `tune.derived_seed(ep, j, 0 | 1)` per slot,
  Stage 1b's rule: unlike the bar, which gives every item of a seed the walk's slot 0 and one lead-in (B7), the fit
  varies the lead-in across slots on purpose (five different starts per evaluation); stated in Q2.
- **Objective** `stage2-r2l-v1`: `tune.score` over the five episodes (falls, then tracking and limit terms).
  Tracking terms: `tune.measure`'s over the item's judged window. Limit terms: `tune.measure`'s window limits
  without the three "at nominal" ones, with the toe clearance on the own-swing reading, plus the item's own, each
  graded on [0, 1]: walk: the shortfall of `along` over its 4.0 m target, `line` over 0.25 m; turn: the heading
  error at `at_s` and the largest in the hold, each over 20 degrees; stop: touchdowns beyond two over 2, the stop
  time over 1.5 strides; nudge: the recovery time over 3 s.
- **Search**: `tune.CMA`, population 12, sigma0 0.1 in the [0, 1] scaling, from the body's Stage 1 G5 selection
  (`speedfit.start_record`), seed 42; 30 generations or the fit's share of the budget.
- **Selection** (Q4): candidates the start and the mean after each third of the generations judged. **The decisive
  cell first, and whole**: each candidate runs the five kinds on validation seeds 2000-2019 at k = 5 (turn side
  left for even seeds, right for odd; nudge direction `("forward", "left", "backward", "right")[seed mod 4]`; the
  same at every reported delay); ranked by `tune.selection_key` on those 100 episodes per candidate (fewer falls,
  then the lower mean S, P, J, then the earlier candidate). Then, reported and not used, as far as the share
  reaches: the selected candidate and the start at k = 0, 2, 4, 1, 3 in that order, on validation seeds 2000-2004
  (25 episodes per cell); a cell the share does not reach reads "not run: budget" **[differs: S2-18's "the others
  reported", partly]**.
- **Budget** (Q5), counted as summed evaluation CPU seconds in `ledger.json`: per set and body 4.0 core-hours: the
  fit at most 2.4 (it ends at the last whole block of six generations inside it, or at 30), 1.2 reserved for the
  decisive cell (4 candidates x 100 episodes at 4.4-11 core-seconds each is 0.5-1.2 core-hours; if it needs more
  it still finishes and the overrun is recorded: the one allowed overrun), 0.4 for the reported cells.
- **Rows**: named by their fit, `R2L-<set>/<seed>` (`R2L-V6noF5/42`, `R2L-F25F9cl/42`); `publish` writes the entry
  `{set, body, seed, r2lg5 {path, sha256}}` into `stage2/tb/r2l/index.json`, and importing `r2lfit` registers one
  controller per indexed set with `panels.register_rows`. The factory returns `lookbar.LookRow` around a tracker
  row in R2 mode with the indexed gains (sha256 checked); the row refuses a condition whose set is not its fit's
  (`cond.set`); `bar.rows` wraps it for `@<ms>ms`. `sources()`: `r2lfit.py`, `lookbar.py`, `lookread.py`; the
  gains file's path and sha256 are in `describe()`.
- **Records**: live under `stage2/tb/work/r2l/<set>/<body>/`: `start.json`, `r2lfit/42/{rotation.json, cma.json,
  evals.jsonl, ledger.json}`, `r2lg5.json`, `r2lg5_evals.jsonl` (resumable per generation and per validation
  evaluation, as `speedfit.RESUME_RULE` and `G5_RESUME_RULE`); `publish` copies them to
  `stage2/tb/r2l/<set>/<body>/`.

### 3.9 The notebook, its lines and the Drive layout

**Cells of `stage2/stage2_training.ipynb`** (metadata: no accelerator; every code cell parses as Python; the
kernel imports the standard library and `google.colab` only):

| # | Type | Content |
|---|---|---|
| 0 | md | What this is, the link to `stage2/tb/colab_steps.txt`, "nothing here trains in the notebook's own process" |
| 1 | code | **Parameters**: `REPO_REF` (`# @param`; 40 hex digits, D24), `RUN_ID`, `QUEUE` (a list such as `["R5:75-M-H-10:42", "R6:75-M-H-10:42"]`), `BUNDLE_ID`, `BUNDLE_SOURCE = "drive"`, `RESUME = False`, `TAKEOVER = False`, `SMOKE = False`, `BUDGET_STEPS`, `WORKERS = 0` (0: one per vCPU, at most 8), `CLIPS = True`, `ALLOW_VERSION_CHANGE = False`, `ALLOW_SOURCE_CHANGE = False`, `USE_GOOGLE_DRIVE = True`, `AUTO_DISCONNECT = True`. If the environment variable `S2_PARAMS` names a JSON file, its values replace these (the local runs and the tests) |
| 2 | code | **Setup** (the scaffold's cell 3, adapted): Colab detection; no `MUJOCO_GL` for the kernel; `pip uninstall -y gym gym-notices shimmy`; the pins of `stage2/tb/requirements-colab.txt` applied when the installed versions differ (compared with torch's build label dropped): `mujoco==3.10.0 numpy==2.5.4 scipy==1.18.1 gymnasium==1.4.0 stable-baselines3==2.9.2 cloudpickle==3.1.2 pillow==13.0.0 PyOpenGL==3.1.10`, and `torch==2.15.0+cpu` from `https://download.pytorch.org/whl/cpu` whenever the installed build label is not `cpu` (pip takes an installed `2.15.0+cu...` as satisfying `==2.15.0`); `apt-get install -y libosmesa6 ffmpeg` (failure reported, not fatal; `shutil.which("ffmpeg")` printed); the Git block (clone without checkout, fetch `REPO_REF`, the two refusals, detached checkout; a non-smoke launch refuses a `REPO_REF` that is not 40 hex digits); **line 1**. There is no unpinned fallback: if a pin does not install, the cell stops and the owner sends its output |
| 3 | code | **Storage and session**: Drive mount (the scaffold's lines); `DRIVE = /content/drive/MyDrive/mesozoic-labs/stage2` (or `S2_PARAMS`' local directory); `nb session`: SESSION_<k>.json, LIVE.json, the run-directory line, the refusals of K6 and K7; a stale `*.part` sweep |
| 4 | code | **Inputs**: `nb inputs`: the parts from `DRIVE/inputs/<BUNDLE_ID>/` copied to local disk, each part's sha256 against the committed manifest, unpacked at study paths, every file's sha256 (`bundle.verify`), then `setio.read_inputs(require_frozen=True)` per set and body (every file against its entry and the freeze manifest; `sets verify` needs the look-format twins, which the bundle leaves out, and stays the lead's check); **line 2**; a failure halts |
| 5 | code | **Preflight**: `nb preflight`: the repository's archive-load preflight (its two lines), a policy built with `MUJOCO_GL` assigned (`train2 preflight`), every env of the queue's bodies built (both floors, the job's mode), G8a's short form for each body; **line 3** per body; a failure halts ("G8a fails: no run") |
| 6 | code | **Smoke** (only when `SMOKE`): `nb smoke` (rules Y1-Y8; Y9 is the lead's download, Y10 the pass rule): one line per item, `S2 SMOKE Y<i> <item>: <value>: OK` (or `FAILED`, or `REPORTED`), and `smoke/SMOKE.json` written after Y7 and again at the end; then cell 8's finish; no training job |
| 7 | code | **Launch**: reprints lines 1-3 from STATUS.json (so the owner reads all four in one place), then `nb launch`: the queue, one job at a time on all workers; **line 4** per job; STATUS.json, LIVE.json and the Drive copies every 60 s; never raises (a failure becomes HALT.json) |
| 8 | code | **Finish**: `nb finish` (DONE.json or HALT.json, the last copy, the summary); then `nb.release_runtime(...)` inside `try ... finally` (D19) |

**The four lines, exact text** (N1; each starts at column 0 and is one line):

```
S2 LINE 1/4 commit: <40 hex> (REPO_REF is a commit: yes; checkout clean: yes)
S2 LINE 2/4 inputs: bundle <bundle_id>, manifest <first 12 hex of the content hash>, <n> of <n> files match their sha256: OK
S2 LINE 3/4 G8a <body> set <set>: zero action = R0 at 0 ms and 10 ms, <n> of <n> episodes identical (<steps> physics steps): PASS
S2 LINE 4/4 first rate <job>: <r> env steps/s (8 envs, <W> workers, <c> vCPUs); <B>M steps in about <h> h
```

Failing forms end `: FAILED (<first path or reason>)` (line 2) and `: FAIL (<condition> seed <s> at <ms> ms,
first difference at step <i>)` (line 3); line 1 with a dirty checkout reads `checkout clean: no`; line 4 below the
launch's floor (the rate at which the budget takes 22 h: `budget / 79,200` env steps/s, 505 for 40M; the sheet
states the number) ends ` BELOW FLOOR <f>`. Lines 3 and 4 repeat per body and per job. After a resume line 4 is
printed again for the session, and the session line reads `Run directory: <path> (resuming session <k>: <job> from
<n> steps; <m> jobs done)`; a new run reads `Run directory: <path> (new run)`; a refused live run reads `Run
directory: <path>: still running (LIVE.json written <s> s ago); nothing to do`.

**The launcher** (`nb launch`), per loop of 60 s: re-read `LIVE.json` (not its own uuid: stop the job, write HALT
"superseded by session <k>", exit); STATUS.json and LIVE.json; copy new text records; copy checkpoints (below);
start one clip child if a new monitor trace exists and none is running (`clip2` with `tbenv.child_env("osmesa")`,
a 300 s limit, its failure recorded and ignored; the trainer never waits for a clip; the clip shows the newest
check's `walk/v0.4` episode of seed 6004). Job rules: a job that ends with exit 0 or 3 (done, or stopped by a
rule) lets the next job start; a crash (another code, or a signal) is retried once from its newest intact
checkpoint, and a second crash halts the run; finished jobs of the queue are skipped on a resume; a job with a
`run.json` and no checkpoint is restarted from step 0 in place.

**Drive** (`MyDrive/mesozoic-labs/stage2/`; a local directory stands in for it here):

```
inputs/<bundle_id>/<part>.tar ..., rebuilt/...            uploaded by the lead; a verified rebuild's arrays (D20)
runs/<run_id>/QUEUE.json                                  the queue, budget, bundle, REPO_REF and commit as first given
runs/<run_id>/SESSION_<k>.json, LIVE.json
runs/<run_id>/STATUS.json                                 rewritten every 60 s
runs/<run_id>/DONE.json | HALT.json
runs/<run_id>/logs/launcher.s<k>.log, <job>.s<k>.log
runs/<run_id>/jobs/<job>/...                              the trainer's run directory (3.6): run.json, updates.jsonl, gates/,
                                                          copied checkpoints with their records, ckpt_<n>.tar per check, monitor/ with clips
runs/<run_id>/smoke/SMOKE.json, ...                       the day-one smoke's files
```

The trainer writes to local disk (`/content/s2runs/<run_id>/`). The launcher copies to Drive the checkpoints at
multiples of 1,000,000 steps and every check's (**[NEW precision]**: K2's "newest two" on Drive are the newest two
of these, so at most about 1M steps, 20-30 minutes, are lost to a dead runtime, and Drive's trash takes about
150 MB per 40M run instead of 400 MB; a tiny run copies every checkpoint): the three files each as `<name>.part`,
then renamed, then `checkpoint_<n>.json` last; a copy read back through the mount with the right sha256 is called
**copied**, not verified (the mount reads its own cache; Drive uploads are asynchronous), and an older checkpoint
is deleted on Drive only after a newer one has stayed copied for one further cycle. On a resume the launcher
copies the job's directory back from Drive (the local directory of a new runtime is empty), and the trainer's own
checks choose the checkpoint.

**`STATUS.json`** (`schema: "s2-status-2"`): `{run_id, run_uuids, session, session_uuid, written_at, started_at,
commit, bundle_id, state ("running" | "done" | "halted"), runtime {vcpus, ram_gb, disk_free_gb}, lines [...the S2
lines printed so far], queue [{job, state ("queued" | "running" | "done" | "stopped" | "failed"), steps, budget,
env_steps_per_s, eta_h, last_checkpoint, last_checkpoint_copied, last_monitor {at_steps, N, S, mean_clock_factor,
gain}, g8a, g8b, fix, flags {kl_cut}, minibatch_steps_last [run, possible], retries, last_error}]}`.
**`SESSION_<k>.json`** (`"s2-session-2"`): `{run_id, session, session_uuid, started_at, last_status_at, resume,
takeover, repo {ref, commit, dirty}, runtime {in_colab, vcpus, affinity, cpu_model, ram_gb, disk_free_gb, gpu},
versions, torch_build, pins, pins_match, python_minor_matches, ffmpeg, bundle {id, content_sha256, files_ok,
source}, queue, budget_steps, workers, allow_version_change, allow_source_change, setup_seconds, params}`.
**`DONE.json` / `HALT.json`**: `{run_id, session, at, reason, jobs {job: {state, steps, last_checkpoint, stopped}},
lines}`. `nb hours` sums, per run, each session from `started_at` to `last_status_at` into
`stage2/tb/hours.json`, read against the 220 runtime-hour cap.

### 3.10 The bundle (`bundle.py`) and the acceptance run (`nb acceptance`)

`build` reads what to pack from committed records only: for each body its case's model files (`plant.model_files`:
`round2/refine/<case>/models/<id>.xml` and `.json`; a clone lacks only these two per body among the plant's
inputs) with the hashes the records hold, and for each set and body the files of the latest freeze manifest
(`refs/*.npz`, `transitions/*.npz`; the look-format twins are not packed). Parts: `models-<body>.tar`,
`<set>-<body>-<i>.tar`, written by D20's tar rules. `MANIFEST.json` (committed at `stage2/tb/bundle/<bundle_id>/`;
`schema: "s2-bundle-2"`):

```
{content: {bundle_id, bodies [...], models {body: {case, xml {path, sha256}, record {path, sha256}}},
           sets {set: {body: {freeze_index, manifest_sha256, files: n}}},
           files {<study-relative path>: {sha256, bytes, part}},
           parts {<name>.tar: {sha256, bytes, files: n}}},
 content_sha256,                                        # s2io.sha256_json(content): what line 2, the sheet and ACCEPTANCE.json quote
 provenance: {made_at, commit, dirty, rule, tool_sha256, versions}}
```

`UPLOADS.json` (beside it, written by the lead after the uploads): per part the Drive file id, the bytes, and for
at least one part a re-download whose sha256 equals the manifest's. `verify(manifest, root)` re-hashes every file
under `root`; `unpack` refuses a member that is not in `files`, a path that leaves `root`, or a part whose sha256
differs.

**The acceptance run** replaces the draft's separate dry run and local notebook run; it is the one test that runs
the notebook's own steps where the owner's session will run them: in a fresh clone that holds nothing git ignores.
`nb acceptance --ref <40 hex> --parts <dir> --scratch <dir outside the checkout> --record
stage2/tb/bundle/<bundle_id>/ACCEPTANCE.json`:

1. `git ls-remote origin` must list the ref's commit on the pushed branch; `git clone --no-checkout <remote URL>`
   into the scratch directory, `git checkout --detach <ref>` (if the remote cannot be reached from this container:
   a clone of the local repository's remote-tracking ref after `git fetch`, recorded as `clone_source`).
2. A stand-in Drive directory with the parts under `inputs/<bundle_id>/`.
3. The notebook's cells 1-5 and 7-8, executed as Python with cwd in the clone and the study venv's interpreter,
   parameters by `S2_PARAMS`: queue `["R5:75-M-H-10:42", "R6:75-M-H-10:42"]` on set V6, `--tiny`; preflight covers
   every body of the bundle (each env built, G8a's short form per body and head).
4. During the first job: SIGKILL of the launcher's whole process group at the moment a `.part` file exists on the
   stand-in Drive (a copy in progress). Then the local run directory and the clone are deleted (a dead VM loses
   both).
5. A second clone; RESUME: the `.part` sweep, the checkpoint copied back from the stand-in Drive, the run to DONE.
6. A third session on the finished run is refused (DONE.json); a second launcher started while LIVE.json is fresh
   is refused; `ALLOW_VERSION_CHANGE` and `ALLOW_SOURCE_CHANGE` are each exercised with a faked session record.
7. Assertions: the four lines by regular expression, every hash, both SESSION files, STATUS.json's schema,
   `run.json`'s sessions, `resumed_from` and `run_uuid`, the copied checkpoints with records, a monitor record with
   6 judged episodes and a gain record, G8b's scaled reading, a real clip (the file exists, is not empty and
   `ffprobe` reads it; OSMesa and ffmpeg are installed here), DONE.json, the R6 job's record, and that `git status
   --porcelain` of the main checkout is unchanged from before the run.

`ACCEPTANCE.json`: the remote URL, `ls-remote`'s SHA, the clone's HEAD, the content hash, counts of verified files,
the G8a counts per body, the printed lines, the kill point, the sessions, seconds, versions. Its tiny training
episodes on V6 are listed in `stage2/tb/dev_episodes.json` (D23). `nb local-run` (the same cells in the main
checkout on the S1 view with `--tiny`) remains as a slow development test that needs no bundle.

### 3.11 Policy rows for the bar runner (`s2rows.py`; C4)

`R5Row` is a `panels.TrackerRow(name, "R2", gains=None, terms=(), ankle_cap=(), policy=S2Policy(...),
controller="R5: ...")`: the panels' own row with their own policy hook (`panels.py:751-801`, `841-852`), so the
tracker, the start, the seeds, the recorder and the judge are exactly those of the R0 row beside it. The registered
factory returns it inside `lookbar.LookRow`, and `bar.rows(body, ["R5/42@10ms"], ...)` wraps that in `bar.Delayed`
like any registered row. `S2Policy.begin(tk, built, seed)` builds `lib.from_plan(built.plan)` on the built track
and a `r5core.Core` on the tracker's estimator and clock, zeroes the action history and the filter, and acts once;
`act(tk, k)` after every tenth physics step: `a = clip(model.predict(s2load.normalize(stats, actor_obs),
deterministic=True))`, the filter, then `tk.residual, tk.rate_factor = core.offsets(a_f)`: the env's step 1 and 2
through the same `Core` methods (the critic's columns are not computed: `PrivilegedCriticPolicy.obs_to_tensor`
pads them, `policy.py:131-141`). `R6Row` is a `panels.Row` whose episode runs `r5core.R6Tracker` (its own reset
and physics step, as `policy_rows.PolicyOnlyTracker`, `policy_rows.py:492-541`) under `control.run` with an acting
hook of its own. **`R5G/<seed>`** is the R5 row of an `-X` body with the bar's head script H1
(`r5core.HeadScript.fixed(episode seed, seconds)`: a gaze train and one jaw event with J4's ranges, both always
present, drawn from `SeedSequence([seed, 77])`): the plan's "cost of using it" row (rule E5); head-first turns
(F11) stay deferred by the lead's ruling 1 and are not built.

A row is built from a checkpoint's files (`configure`, or the JSON file `$STAGE2_POLICIES` names: row name to job
directory and step count). Building it:

1. loads the checkpoint once through `s2load.load_checkpoint(..., expect_run=run.json's)`, so a checkpoint whose
   fingerprint is not reproduced within 1e-5, or whose `run_uuid` or settings hash is not its `run.json`'s, raises
   before any episode ("not judged"; S2-12);
2. compares the run's recorded hashes of `r5core.py`, `r5lib.py`, `lookread.py`, `designlab/stage1/policy.py`,
   `env.py` and the panels' sources with today's; unequal: refused, or, under `STAGE2_ALLOW_SOURCE_CHANGE=1`, run
   with `source_change: true` and both hash sets in `describe()` (the code that turns the tracker's state into
   observations and actions into offsets must be the code the policy was trained with);
3. accepts the run's set under a later freeze only by K4's rule (unchanged files, entries and task table).

`describe()`, which is the record's `controller` and enters its key, holds what an episode executes and no
measured value: `{row, controller, mode, body, ppo_seed, checkpoint {steps, files {name, sha256}}, run {run_uuid,
job, set, set_freeze, settings_sha256}, layout, action {scales, rate, filter_hz}, versions_trained,
versions_judged (s2load.comparable), cross_version, deterministic: true, head: "neutral" | "H1", source_change}`.
The library versions are in the key on purpose: they are executed, and one record's episodes must share them; S2-12
asks that every policy-row record hold both sets, and without an edit to `bar.py` the key's `row` block is where
the bar's own record can hold them. The **measured** fingerprint difference and its tolerance are not in the key:
they go to the sidecar's and the companion record's `policy` block (3.5). `sources()`: the sha256 of `s2rows.py`,
`r5core.py`, `r5lib.py`, `s2load.py`, `lookbar.py`, `lookread.py` and of `designlab/stage1/policy.py` and
`env.py`; not `r5env.py`, `train2.py` or `schedules.py`, which no row executes (a test asserts that building and
running a row imports none of them). `spec()` carries the paths and hashes; a worker checks the hashes again and
loads lazily. The set a row is asked to run on must be the run's set (`cond.set`; refused otherwise: a policy is
judged on the references it imitates; D4 says what that means for "no loss").

### 3.12 The steps sheet (`stage2/tb/colab_steps.txt`; C10)

Plain text, wrapped at 115 columns, addressed to the owner, in the order of
`docs/investigations/stance_2026_10/stance3/colab_steps.txt`. It quotes every expected line with its values and
says, after each, what to do if it differs. Outline:

1. *What runs*: REPO_REF as the launch commit's 40-hex SHA (and the tag on it), the bundle id and the first 12 hex
   of its content hash, the rules file and its sha256, the runtime type (High-RAM CPU) and why. If High-RAM is not
   offered: stop and tell me (the plan's next choice, L4, costs about twenty times as much; I confirm it first).
2. *Before day one*: the input parts are uploaded by me through the Drive connector; each upload asks for your
   approval (how many, and what each is called).
3. *Day one, the smoke (one runtime; the minutes of each item, measured here where they could be, are in the
   sheet; Y8 is capped at 40 minutes and may read "not finished")*: open the notebook from the tag
   (`https://colab.research.google.com/github/kuds/mesozoic-labs/blob/<tag>/hardware/trex_form_study/stage2/stage2_training.ipynb`),
   save a copy in Drive, Runtime > Change runtime type, the parameter cell's values with `SMOKE = True`, Run all,
   allow Drive (if Drive access is refused: nothing was written; run the cell again and allow it, or tell me).
   Check: line 1 (another commit: stop and tell me); line 2 (FAILED: stop; nothing was trained; tell me the line);
   line 3 (FAIL: stop); the eight `S2 SMOKE` lines, each with what it means and what to do if it reads FAILED;
   where the files are on Drive. "There is nothing to send: I read them from Drive."
4. *A launch (per runtime)*: the table of runtimes, each with its RUN_ID and QUEUE, `SMOKE = False`,
   `BUDGET_STEPS` as fixed after the smoke; Run all; the four lines with their expected values, all four repeated
   together above the launch (line 4 "within about 3 minutes of line 3", with the floor below which it prints
   BELOW FLOOR and you tell me); how long to leave it; what the end looks like (DONE.json, the runtime
   disconnects by itself).
5. *If the browser shows "disconnected"*: with background execution the runtime may still be running. Look at
   `runs/<RUN_ID>/STATUS.json`'s `written_at`: if it is less than 10 minutes old, the run is alive: do nothing.
   If `DONE.json` or `HALT.json` exists: finished, do not resume. Otherwise (STATUS.json older than 10 minutes):
   the same copy, a fresh runtime, the same parameters with `RESUME = True`, Run all; the expected lines
   (`resuming session <k>: <job> from <n> steps`; any `WARNING: skipping checkpoint <n>: <reason>`; `restarting
   <job> from step 0` when the cut came before its first checkpoint: that is expected and loses nothing else). A
   run that answers `still running` was alive after all: close the new runtime.
6. *Messages that mean stop and tell me* (never restart, never delete, never change RUN_ID or REPO_REF, never set
   TAKEOVER or an ALLOW parameter on an error's advice; I tell you when one applies): a refused resume (settings,
   inputs, sources, versions), `HALT.json`, `G8a ... FAIL`, `fingerprint not reproduced`, `BELOW FLOOR`, a job
   retried twice, Drive full (empty Drive's trash first: deleted checkpoints stay there).
7. *Fallbacks*: if line 2 fails because a part is missing or cut short: which part I upload again (you approve);
   if uploads keep failing: `BUNDLE_SOURCE = "rebuild"` (only after the smoke's Y8 passed; the first launch then
   takes longer, the later ones reuse its arrays) or `"git:<ref>"` (only if you agreed to the data branch); if
   clips fail: nothing to do; if the pins do not install: stop and send the cell's output.
8. *Which lines were measured here and which are inferred*: every line the acceptance run printed is quoted from
   its record; Colab-only lines (the mount, the disconnect, the runtime releasing itself, Drive's 10 minutes) are
   marked inferred.

## 4. Tests and gates

Fast tests run in the default suite; slow ones behind `STAGE2_SLOW=1`; a test that needs arrays, models, torch or
a built set skips without them; no test writes inside the checkout (each uses `tmp_path`). **Until task 3's first
unit no test or acceptance runs an R2-mode, policy or training episode on V0, F25F9 or F25F9cl** (D23): closed-loop
tests use the S1 view (`Library(..., "S1")`, `g8a.stock`, `monitor2.S1_CONDITIONS`) or a synthetic set in
`tmp_path`.

| What must be proven | Test or gate | Default |
|---|---|---|
| **G8a** | `gates2.g8a` (the committed record per body and set); `test_stage2_g8a.py::test_zero_action_is_r0_on_the_s1_view` (`g8a.stock`, 1 s, k = 0 and 3: identical); `::test_a_length_that_is_no_multiple_of_20_ms` (seconds 1.013: `compared_steps == reference_steps`, identical); `::test_compare_rule` (tables of digests and ends: a row fall inside the env's last partial step; an env termination the row did not have) | fast (needs Stage 1 arrays) |
| A zero action is not "almost" zero | `test_stage2_r5env.py::test_any_offset_breaks_the_digest` (one action entry at 1e-6: the first env step differs); `::test_zero_action_leaves_residual_none_and_rate_one` | fast |
| Reset is history-free | `::test_reset_is_history_free` (an env after 6 randomised episodes against a fresh env: `reset(seed)` plus fixed actions, digest per physics step; hard and carpet; stance and mid starts; 14 and 16 joints); `::test_trackers_are_built_nominal` (every tracker's `damp`, `fric` equal the nominal arrays after randomised episodes; a plan-by-options tracker first used right after one) | fast |
| Mid-task start and queue | `::test_mid_start_state_is_the_files_frame`; `::test_mid_start_queue_holds_the_references_commands` (k = 3: `ctrl` equals an undelayed replay's shifted by 3); `::test_mid_start_never_lands_in_a_hold`; `::test_mid_start_walks_on` (slow, 75 cm, S1 view: up 4 s from 3 frames) | fast / slow |
| 14 and 16 joints | `test_stage2_r5core.py::test_layouts` (88/130/15 and 102/146/17; 87/129 and 101/145 without the heading; blocks contiguous; actor first); `::test_x_env_steps`; joints addressed by name | fast |
| Commands, heading, phase, tables past the end | `test_stage2_r5lib.py`: the command is constant inside a segment and changes at first frames only; the turn's pulse integrates to 90 degrees and `heading` ends at it; nothing before a segment's first frame depends on it; the phase is 0 at the reference's left lift-offs; every table at clock factor 1.25 past `t_end` on a plan ending in a periodic segment and on one ending in a hold equals `Track.joint`'s continuation; the resolved task table and its absent tasks; the S1 table | fast (synthetic set, S1 view) |
| Reward | `test_stage2_r5core.py`: each term in [0, 1]; on a kinematic playback every imitation term is 1 and every penalty 0; each term moves when its cause is injected (a foot offset; a contact mismatch beyond 24 ms; a head pitch; a tail servo over its rating from the first step; a moving-pair gap under 3 mm while a fixed pair at 2.9 mm adds nothing; a foot loaded through the reference's mid-swing; a clock 10 % slow in a steady walk; a 10 degree heading error; a joint inside its required margin); the weights and scales reach `describe()` and equal the rule text's table | fast |
| R6 | `test_stage2_r5env.py::test_r6_steps_on_14_and_16_joints` (targets inside the ranges; the span rule recorded); `::test_r6_mid_start_history` (after k steps the target is the reference's, not home); `test_stage2_train2.py::test_tiny_r6_run_resumes` (slow); `test_stage2_s2rows.py::test_r6_row_equals_env` (slow, digest at k = 5); `gates2.r6_start` (the record `r6_start.json`) | fast / slow / record |
| Randomisation restored | `::test_factors_are_applied_and_restored` (the next nominal episode's model equals the nominal one bit for bit, damping and friction-loss included) | fast |
| The look reading and its wrapper | `test_stage2_lookread.py`: synthetic series for each band on both sides of its limit; own-touchdown segmentation ignores a 10 ms shift of the whole motion; "not read" under 0.02 rad; **too few own steps fails bob, tail and toe-off and is no error**; too few in the reference is an error. `test_stage2_lookbar.py`: `R0L` equals `R0` (S1 view: digest per physics step, equal episode records, the instance's `recorder` restored after an exception); the sidecar precedes the partial; `collect` on synthetic bar records (counts by `bar.counts`; an `unread` episode survived and not succeeded; a missing sidecar refused); `bar._Played` has `q` and `loads`; no round-A file differs from state A | fast |
| Policy rows equal the env | `s2rows.equivalence_check` (a row episode under `bar.Delayed` against the env driven by the same checkpoint: digest per physics step at k = 5, on a plain and an `-X` body); `test_stage2_s2rows.py` with a freshly initialised policy saved through `s2load`; a row imports neither `r5env` nor `train2`; a changed `r5core.py` hash is refused without the flag | slow |
| **Cross-version loading** | `test_stage2_s2load.py`: an archive **saved by `train2` itself** has no bytecode member; `load_checkpoint` reproduces the fingerprint exactly in a fresh process; an inference load unpickles neither space nor a last-observation member (asserted by loading an archive whose members were replaced by objects that raise on unpickling); the plain statistics normalise as `VecNormalize` does; a changed byte, a missing sidecar, a truncated zip, a foreign `run_uuid` and a wrong fingerprint are each refused with their reason; `comparable` drops torch's build label; `pack` then `import_tar` round-trips. The real cross-version case is the smoke's Y9 | fast (needs torch) |
| **Kill and resume** | `test_stage2_train2.py::test_a_killed_run_resumes_bit_for_bit` (stub env: an uninterrupted run against one killed with SIGKILL between boundaries and resumed: every policy tensor, the optimizer's state, `obs_rms` with its count at the first rollout after the resume, `ret_rms` and the fingerprint equal; also with another worker count); `::test_kill_during_the_monitor`, `::test_kill_after_the_fix` (a stub monitor forces a G8b failure: fix, kill, resume, equality with the uninterrupted forced-fix run; `fix` recorded once); `::test_resume_when_run_json_is_ahead_of_the_checkpoint` (later entries move to `discarded_after_resume`; `updates.jsonl` holds no step twice); `::test_restart_from_zero_is_the_fresh_run`; `::test_resume_refusals` (settings, inputs, a later freeze with and without changed files, sources and versions with and without their flags, a broken newest pair skipped with its reason); `gates2.kill_resume` (K3's settings on the real env: the validation record) | fast (stub) / record |
| The worker count does not matter | `test_stage2_vecpool.py`: rollouts of a seeded stub env equal `DummyVecEnv`'s at W = 1, 2, 4 (8 behind the slow flag); a dead worker raises; `env_method` and `seed` keep env order; a worker's `sys.modules` holds neither torch nor stable_baselines3; its command line holds the given `worker_cmd` | fast |
| Lesson 16 | `::test_minibatch_steps_are_counted`; `::test_kl_cut_flag` | fast |
| `MUJOCO_GL` and the clip path | `::test_the_trainer_assigns_mujoco_gl` (a subprocess started with `MUJOCO_GL=osmesa` builds a policy and runs one rollout: exit 0, `disable` in it and in a worker); `test_stage2_tbenv.py` (the clip child's environment holds `PYOPENGL_PLATFORM=osmesa` when its parent holds `disable`; `repo_on_path` appends; importing `s2load` in a process that never built an env finds `environments`); `clip2` renders a real clip here and its process holds no torch | fast |
| Monitor, G8b, stop rules | `test_stage2_monitor2.py`: 18 distinct seeds; the gain rule on tables (two more; one second longer; against the best earlier check; the first check against R0L); **a policy that stands, a policy whose clock lags (`unread`, reason `clock_behind`), a too-few-steps episode: each counted as not succeeding, never an error, never re-run**; the survival credit (a clock held at 0.75 earns 0.75 of its time; R0L its full time); G8b's 2 of 3; the fix doubles the mapped tasks' weights only; a second failure stops; three checks without a gain from 20M stop at the third; `decide` gives the same state from records alone; S2-10's record with its earlier checkpoints and its per-term table | fast |
| R2L | `test_stage2_r2lfit.py`: the rotation table **per (kind, direction)**: each meets every delay within the stated number of blocks; written before generation 0; a resume refuses another table; every member of a generation gets the same episodes; the decisive cell runs first and whole; a reported cell beyond the share reads "not run: budget"; the fit ends at a whole block; the row's name resolves through the index and refuses another set; `speedfit.py`'s sha256 is unchanged by importing `r2lfit` | fast (stub evaluator) |
| Bundle | `test_stage2_bundle.py`: two builds of the same files in two processes and under another umask give byte-identical parts; the content hash does not change when the file is rewritten with CRLF, another indentation or an added provenance field; a changed file, an extra member and a path leaving the root are refused; `git check-ignore` on every output path, live files included (`work/...`, a `.tar`, a checkpoint's files, a clip, a log, `LIVE.json` under `work/`) | fast |
| **The notebook** | `test_stage2_nb.py::test_notebook_cells_parse`, `::test_kernel_cells_import_only_stdlib`, `::test_pins_equal_the_study_venv` (every pin, PyOpenGL and pillow included); launcher tests on a stub trainer (LIVE.json fencing: a second launcher is refused, a superseded one halts; a crashed job is retried once; a rule-stopped job lets the next start; `.part` sweep; the copy cadence; `collect`; `hours`); `::test_local_run_with_a_kill_and_a_resume` (slow: `nb local-run` on the S1 view with `--tiny`); **`nb acceptance`** (3.10: the record `ACCEPTANCE.json`) | fast / slow / record |
| No checkout dirt | `test_stage2_nb.py::test_a_local_job_leaves_git_status_unchanged` (a stub job under `stage2/tb/work/` in a scratch repository) | fast |

**What a reviewer can run in under 10 minutes** (from FS, round A's environment line):

```
$PY -m pytest -p no:cacheprovider tests/test_stage2_tbenv.py tests/test_stage2_schedules.py tests/test_stage2_s2load.py \
    tests/test_stage2_r5lib.py tests/test_stage2_r5core.py tests/test_stage2_r5env.py tests/test_stage2_g8a.py \
    tests/test_stage2_lookread.py tests/test_stage2_lookbar.py tests/test_stage2_vecpool.py tests/test_stage2_train2.py \
    tests/test_stage2_monitor2.py tests/test_stage2_r2lfit.py tests/test_stage2_bundle.py tests/test_stage2_nb.py   # about 7 min
$PY -m designlab.stage2.gates2 g8a --body 75-M-H-10 --set V6 --conditions walk/v0.4 --seeds 3042 --record /tmp/x.json   # 20 s
$PY -m designlab.stage2.bundle verify --manifest stage2/tb/bundle/<id>/MANIFEST.json --root .      # seconds
$PY -m designlab.stage2.r2lfit rotation --body 75-M-H-10 --set V6noF5 --seed 42                     # prints the table; equals the record
git status --porcelain; git check-ignore -v stage2/tb/work/x stage2/tb/bundle/x/parts/a.tar
```

**What the independent reviewer of the finished build verifies:**

1. No file under `designlab/stage1/` and no round-A module changed since state A (`git diff --stat`);
   `stage2/regress/regress.json` is current; no bar-key module changed between state A and the end of tasks 3-4.
2. `stage2/runs/2026-11-15-75cm/dev_episodes.json` and `stage2/tb/dev_episodes.json` list every check episode; no
   R2-mode or policy episode on V0, F25F9 or F25F9cl before task 3's first unit; the commit adding
   `PREREG_TRACKB.md` is an ancestor of every Track B record and of every look-set freeze a record uses.
3. G8a records exist for all four bodies and heads with 40 identical episodes each (or their "not run" list with
   the set's reasons); a sample recomputes; the notebook's line 3 comes from the same function; the "any offset
   breaks it" test fails when the env's zero test is weakened; a length that is no multiple of 20 ms is covered.
4. The env's reset and step order against `env.py` line by line (3.2); trackers are built at construction; the
   history-free test.
5. G9's two records carry times before and after the first look-set episode as ruled; the degraded playbacks fail
   exactly their term; the set's rungs equal `L1_FREEZE.json`.
6. The four look numbers recompute from a trace's poses on episodes that pass and that fail; the too-few-steps case
   fails rather than errors; `R0L` equals `R0` step for step; a companion record's counts recompute from its
   episodes and its bar record's hash matches.
7. Every reward scale and weight in the code equals the committed rule text, and `run.json` holds them; G8b's fix
   changes task weights only; the three terms beyond the plan's list are as the lead ruled.
8. Kill and resume on the real env: equality of weights, optimizer, statistics and fingerprint; a kill during the
   monitor and after the fix; no step twice in `updates.jsonl`; a resume from a Drive copy whose `run.json` is
   newer than its checkpoint; a restart from zero.
9. An archive saved by `train2` itself has no bytecode member (`policy.model_kwargs` injects a closure by default,
   `policy.py:162`); grep finds no `PPO.load`, `VecNormalize.load` or `pickle.load` of a checkpoint outside
   `s2load`; the fingerprint reproduces in a fresh process and after a Drive round trip; how many of its rows the
   normalisation clips; a bar record made from a checkpoint with other versions holds both version sets and
   `cross_version`.
10. Row against env by digest at k = 5 with a trained checkpoint, on a plain and an `-X` body; a row refuses a
    checkpoint of another run and changed observation code.
11. `monitor2.CONDITIONS` and `SEEDS` were committed before S2-4 started; `S2_10.json`'s two counts and two means
    recompute from its 36 episodes; the standing and lagging cases are counted, not errors.
12. R2L: `rotation.json` precedes the first generation; every evaluation's delays and directions equal it; the
    selection read only k = 5; the ledger's sum; the gains file's sha256 in every R2L bar record; `speedfit.py`
    unchanged.
13. R6: the env on 14 and 16 joints, a tiny run with a resume, the row against the env, `r6_start.json`.
14. Bundle and acceptance: the acceptance run's commit is on the remote; every hash; every env builds from the
    clone alone; the content hash equals the sheet's and line 2; one part downloaded again from Drive matches
    (`UPLOADS.json`).
15. Notebook: the acceptance run's printed lines equal the sheet's quotations character for character; refusals
    for an existing RUN_ID, another queue, a live run, a changed version without its flag, a REPO_REF that is no
    commit; `git status` stays unchanged during and after a local run.
16. The steps sheet: every check is a line the acceptance run or the smoke printed; every failure has an
    instruction; REPO_REF is a commit.
17. After the smoke: `SMOKE.json` against Y1-Y10, the budget's arithmetic (Y11), `colab_archive.json`.
18. `git check-ignore` on every output path; `rates.json` was measured on an idle machine or is labelled
    provisional, and section 6's estimates were replaced by it; `hours.json` sums to the sessions.

## 5. Work packages

Each builder works in its own worktree on its own branch; no two packages own a file; at most two agents build at
once, each with one heavy process (round A's rule). Names carry a "C" so that they do not collide with round A's.
One brief per package: `WPC_<name>.md`.

| Wave | WP | Owns (exclusively) | Needs from round A | Acceptance |
|---|---|---|---|---|
| 1, can start now | **CP** policy files | `tbenv.py`, `schedules.py`, `s2load.py`, their three test files | nothing (torch, SB3, the repository's `policy_loading`) | its tests; `s2load check`, `pack` and `import` on a scratch checkpoint in `tmp` |
| 1, now | **CV** workers | `vecpool.py`, its test | nothing | its tests (stub env) |
| 1, now | **CB** bundle | `bundle.py`, its test, `stage2/tb/.gitignore` | nothing for the tool (tested on Stage 1's files in `tmp`); the real bundle needs the frozen look sets | its tests; a build, verify and unpack of a stand-in bundle of the two picks' models and Stage 1 arrays |
| 1, now | **CK** look measures | `lookread.py`, its test | the look ladder's arrays (`restore_inputs.py copy-from ... --group look`); synthetic series. `playback` needs `bar._Played` (WP B's branch; its test skips until `bar` imports) | its tests; the measures on a kinematic playback of a look-ladder rung against `stage2/look/metrics.json` |
| 2, after round A's R is merged | **CE** env | `r5lib.py`, `r5core.py`, `r5env.py`, `g8a.py`, their four test files | `setio`, `s2io` (R); Stage 1 arrays (it develops on the S1 view); `g8a.bar_form` needs `bar` (lazy import; tested by CG) | its tests; scratch C1 and C3 reproduced through the real env on the S1 view (`g8a.stock`); `throughput` on both floors |
| 3, after CE and CP | **CT** trainer | `train2.py`, `monitor2.py`, `clip2.py`, their test files | CE, CP, CV; the monitor's real episodes need CW's `lookbar.episodes` (built against a stub of it) | its tests; a tiny R5 run and a tiny R6 run with a kill and a resume **on the S1 view** |
| 3, after CE, CP, CK and round A's B | **CW** rows | `lookbar.py`, `s2rows.py`, their two test files | CE, CP, CK, `bar.py` | its tests; `R0L` against `R0` and `equivalence_check` at k = 5 **on the S1 view**; on a real set only after task 3's first unit |
| 3, after round A's B and S | **CF** R2L | `r2lfit.py`, its test, `stage2/tb/r2l/index.json` (empty) | `bar.py`; CW's `lookbar.LookRow` for the row | its tests (stub evaluator); a 2-generation smoke fit with the real evaluator on V0 **only after task 3's first unit** (the lead says when; no record kept, listed as a dev episode) |
| 4, after CT, CW | **CG** gates | `gates2.py`, its test, `stage2/tb/run_trackb.sh`, the records under `stage2/tb/gates/` and `stage2/tb/validation/` | everything above; the frozen V6 sets for the committed records | its tests; then the run plan's records, in its order |
| 4, after CT and CB (its launcher half starts in wave 1 against a stub trainer) | **CN** notebook | `nb.py`, `stage2/stage2_training.ipynb`, `stage2/tb/colab_steps.txt`, `stage2/tb/requirements-colab.txt`, its test | CT's CLI, CB, CP's `tbenv` | its tests; `nb local-run` on the S1 view; then the acceptance run (with the lead: it needs the pushed commit and the bundle) |

The lead owns `stage2/tb/PREREG_TRACKB.md`, `stage2/tb/dev_episodes.json`, the uploads and `UPLOADS.json`, and
the look work package's entries of D5.

What can start before round A is merged: CP, CV, CB, CK and the launcher half of CN (five packages; two at a
time). CE needs only round A's R package (committed on `wt/wp-r`). Nothing in round C waits for the look work
except the committed gate records, R2L's real fits, S2-4 and the acceptance run.

Easy to miss: CE's `Core` and `Script` are what CW's policy calls, so their interface (2.2) is frozen by this
document; `train2` calls `g8a.short` and `monitor2` calls `lookbar.episodes` through lazy imports, so CT builds and
tests against stubs of both and the packages do not import each other at module level; `train2.SOURCE_FILES` names
files of every package, so they are frozen together at state B; `lookread.py` and `lookbar.py` enter the row
sources of every Track B row; `nb.py` and `tbenv.py` are standard library only at import (a test imports each in a
process with numpy blocked).

## 6. Run plan

From FS; `PY=.venv/bin/python`; `export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 PYTHONHASHSEED=0`; every long command through `tools/lowprio.sh`; `tools/memguard.sh` with
`PROTECT="designlab.stage2.train2"` (the trainer, its workers and its monitor child carry that string; the
launcher is small and is never the largest process). `TB=stage2/tb`, `W=$TB/work` (ignored). Each step is one line
of an idempotent chain (`$TB/run_trackb.sh`, round A's `step` pattern with done-markers); every command resumes by
itself; every long job writes under `$W` and is collected with `nb collect` when it ends (D22).

Rates used below [scratch C2 and RV8, with the interference check and the added reward terms; to be replaced by
step V3's record]: 350 env steps/s per process on hard floor, 290 on robot_floor_carpet. With 8 envs (6 hard, 2
carpet) in W workers and about 15 % for PPO: W = 2 about 570 env steps/s, W = 3 about 850, W = 4 about 1,140 on
idle cores. Bar episodes: R0 0.08 s of wall time per simulated second, R2 0.2-0.45, a policy row about 0.12.

| # | Step | Command | Estimate |
|---|---|---|---|
| V0 | `-X` preconditions (D5) | G1 of both `-X` bodies on an idle machine; the jaw cover decision recorded; one kinematic `sets build` of a walk and a start on an `-X` model | the lead's; minutes of compute |
| V0b | the upload route (D20) | the lead uploads and re-downloads a 20 MB tar, a 5 MB tar and a 170 kB file; outcome in `$TB/validation/drive_route.json` | minutes; the owner approves each upload |
| V1 | G9-ref, per body and head (4) | `gates2 g9-ref --body B --set V6 --record $TB/gates/V6/B/g9_ref.json` | kinematic; 2-3 min each |
| V2 | G8a, per body and head (4) | `gates2 g8a --body B --set V6 --record $TB/gates/V6/B/g8a.json` | 40 episodes, about 3 min each body; 0.2 core-h |
| V3 | R0L's records and G9-run | `lookbar run --set V6 --body B --rows R0L@10ms --out $W/bar/r0l` (16 gated conditions and `walk/nominal`), `nb collect`, `gates2 g9-run ...` | 0.15 core-h per body and head; G9-run minutes; the Stage 1b re-run is round A's |
| V4 | rates and memory | `gates2 rates --body 75-M-H-10 --set V6 --record $TB/validation/rates.json` (2 floors x 1, 2, 3, 4 workers x 16,384 steps; RSS per worker and of the trainer) | 15 min; **last, alone**, or recorded provisional (lesson 10) |
| V5 | sigma probe (reported), filter rule | `train2 probe-sigma ...`; `train2 filter ...` (both sizes, current heads) | about 400 short episodes, 15 min, 0.25 core-h |
| V6 | learning-rate probe | `train2 probe-lr --bodies <the four> --set V6 --workers 2` (from 1e-4 down, stopping at the first rate that passes both sigmas; 6 updates of 8,192 steps per cell) | 98k steps per rate tried: 3-9 min per body at W = 2; at most 0.6 h wall, 1.2 core-h |
| V7 | a training start with `MUJOCO_GL` set against it; R6's start; kill and resume | `gates2 gl-start`; `gates2 r6-start --body 75-M-H-10 --set V6`; `gates2 kill-resume --body 75-M-H-10 --set V6` (K3: 131,072 steps twice) | 15 min at 2-3 workers; 0.6 core-h |
| V8 | **S2-4** | `nb launch --run-id s2-4 --queue R5:75-M-H-10:42 --set V6 --budget 2000000 --monitor-every 500000 --workers 3 --local $W/local --drive $W/drive`, `nb collect`, then `monitor2 s2-10 ...` | 2,007,040 steps: about 40 min of stepping at W = 3 plus four checks and the baseline: 50 min (65 at W = 2 beside the panels); 2.7 core-h |
| V9 | the 60 cm local run (D11, M6; recommended) | `nb launch --run-id local-60 --queue R5:60-N-T-10:41 --set V6 --budget 5000000 --monitor-also-k 0 --workers 3 ...` (its checks also at 0 ms) | 5,005,312 steps: about 1.7 h at W = 3; 5.2 core-h |
| V10 | R2L | `r2lfit rotation`, `fit`, `select`, `publish` for (V6noF5, 75-M-H-10), (V6noF5, 60-N-T-10), then if granted (F25F9cl, 75-M-H-10); `--workers 2`, `--out $W/r2l` | 4.0 core-h per set and body by the ledger (Q5): 8.0 for the two; 4.0 more for the candidate |
| V11 | the bundle | `bundle build --bundle-id <id> --bodies <trained> --sets V6 --out $TB/bundle/<id>`; commit C1 (D24); push; the uploads (the lead, one approval per part); `UPLOADS.json` | build 2 min |
| V12 | **the acceptance run** (3.10) | `nb acceptance --ref <C1> --parts $TB/bundle/<id>/parts --scratch /home/user/mesozoic-wt/tb-accept --record $TB/bundle/<id>/ACCEPTANCE.json` | about 12 min, 2 workers |
| V13 | the steps sheet | commit C2: `$TB/colab_steps.txt` with C1's SHA, the bundle id and content hash, the queue per runtime, the lines as V12 printed them, the minutes of each smoke item; `ACCEPTANCE.json` | - |

Order on 4 cores shared with the 75 cm panels (two processes): V1 and V2 first (V1 gates every later look-set
episode); V3, V5 and V7 run one or two processes beside the panels at any time; V6 and V8 at `--workers 2` beside
them, or 3 once they are done; V4 last, alone, or recorded provisional; V10 at two workers after V8, about 4 h of
wall time for the two fits; V9 beside V10's second half or overnight; V11-V13 need the frozen sets and the pushed
commit. Sum without the candidate fit: validation and S2-4 about 6.5 core-h, the 60 cm run 5.2, R2L 8.0; the
plan's estimate for this CPU has 10 for validation and 8 for R2L.

On-disk layout of the outputs:

```
stage2/tb/work/                                       ignored: everything live (local/, drive/, bar/, r2l/)
stage2/tb/gates/<set>/<body>/g8a.json, g9_ref.json, g9_run.json
stage2/tb/validation/rates.json, sigma_probe.json, filter.json, lr_probe.json, gl_start.json, r6_start.json,
                     kill_resume.json, drive_route.json, colab_archive.json (after the smoke)
stage2/tb/runs/s2-4/{QUEUE.json, SESSION_*.json, STATUS.json, DONE.json, COLLECTED.json, S2_10.json,
                     jobs/R5-75-M-H-10-s42/{run.json, updates.jsonl, gates/, monitor/*.json, checkpoints/checkpoint_*.json}}
stage2/tb/runs/local-60/..., stage2/tb/runs/<colab run_id>/... (collected from Drive by the lead)
stage2/tb/bar/<run>/<set>/<body>/<block>/{eval/..., look/...}      the bar's records and their companions
stage2/tb/r2l/index.json, <set>/<body>/{start.json, r2lfit/42/..., r2lg5.json}
stage2/tb/bundle/<bundle_id>/MANIFEST.json, UPLOADS.json, ACCEPTANCE.json, parts/*.tar (ignored)
stage2/tb/colab_steps.txt, requirements-colab.txt, PREREG_TRACKB.md, dev_episodes.json, hours.json, run_trackb.sh
```

## 7. Risks and open questions for the lead, most important first

**Rulings needed before the build (each changes a rule's wording; the draft marks them):**

1. **S2-10's gain may be out of reach on `75-M-H-10` for a reason that is not learning.** R0L walks there, so its
   survival is at the cap and the "+1 s" route is closed; reading 1 contains reading 2, and the tail servo's
   sustained limit failed in 20 of 20 Stage 1 episodes at 0.35 m/s and above, so R0L's N may be 0 and R5 can gain
   only by pulling the tail under its rating in two episodes. Proposed (rule M3b, **[differs: S2-10]**, struck
   unless the lead rules it in before S2-4): when R0L's S is within 1.0 s of the mean planned length, a gain is
   also "at least 2 more of the 18 succeeding in reading 3 with the look bands kept". Whatever is ruled, R0L's 18
   episodes run first (V3 precedes V8), every monitor record holds failures per term, and a "no gain" goes to the
   owner with that table. Please rule on M3b.
2. **Three reward terms and one input beyond plan B.1's lists** (D7, D9): `progress`, `heading`, `range`; the
   heading error as an actor input. Without them the reward is nearly blind to what the bar's walk and turn items
   read (distance, line, heading, range margin) and the policy has no heading loop. Please accept or strike each;
   striking the input strikes the `heading` term and makes rule E3 say that line and heading are open-loop.
3. **"No loss" for R5 and R6, and "walking turns"** (D4): accept the look-set counterparts with the finite
   0.35 rad/s turn in place of the periodic one (and Track C builds the four missing condition kinds), or name the
   no-loss conditions that R5 will not run. This fixes whether the look work package must also build a periodic
   walking-turn gait.
4. **Three readings that change wording**: the progress-credited survival S (D15, M3); too few own steps as a
   failed look term and an unread episode of a policy row (`clock_behind`, `unreadable`) as "not succeeding" in place of C8's "run again" for policy rows (D17, U5,
   U6). Without the last two the monitor has no outcome for a policy that stands or lags, which R6 will do.
5. **Sigma 0.1 is about ten times the action noise the open-loop base tolerates** [scratch RV1]: every early
   training episode on both sizes is a 3 s noise fall. The design keeps the plan's 0.1 (mid-task starts still show
   the whole task; the learning rate must pass its probe at 0.1 and at 0.03; `kl_cut` is flagged). The lead should
   know this before S2-4 and may prefer a lower start chosen by the probe (a labelled change to B.1); the owner
   should be told either way.
6. **`n_steps` 1024** (rollouts of Stage 1's 8,192 steps; 245 updates in S2-4 instead of 123): my choice on the
   reviewers' advice; confirm or keep 2048.
7. **No edit under `designlab/stage1/` and none in round A's modules**, so the trainer is
   `designlab.stage2.train2` and not an extended `train.py` (D2; changes B.4's wording), and reading 1 lives in
   companion records beside the bar's, whose own `counts_look` stays null (D17). If the lead would rather have the
   hook in `bar.py`, it must go into WP B before state A; this design does not need it.
8. **R6 is expected to stand** (D8). Its G8b reading can pass by standing; nothing in the plan asks more of R6.
   Confirm that the rules may say so in advance.
9. **The head's cost of use**: the bar's head script H1 and row `R5G` are pre-registered now (E5, 3.11);
   head-first turns stay deferred (ruling 1), labelled. Confirm H1's numbers (J4's ranges, both events always on).

**Risks that need no ruling:**

10. **The look sets R5 needs are more than round A lists** (D5): the turn in place with its start and stop; V6 on
    both `-X` bodies and on `60-N-T-10`; V6noF5 on both picks. If the 60 cm body gets no quick turn (ruling 4), its
    R5 trains without the turn tasks and its bar turn items are "not run". Nobody has built a set on a 16-joint
    model, G1 of `-X` is provisional and the jaw cover is undecided: step V0.
11. **The 60 cm policy must learn balance from a base that falls** (D11): the local 5M run (V9, M6) costs 5.2
    core-hours that the plan's estimate does not list; its record is built to say which remedy to offer the owner.
12. **Every checkpoint is a boundary that resets all envs** (D16). It buys an exact resume; it cuts up to 5 % of
    episodes short. If the equality test fails on the real env, the build stops and comes back to the lead: no
    tolerance is set after seeing a difference.
13. **Colab's image decides whether the pins install.** There is no unpinned fallback; a failed pin stops the
    smoke and a new launch commit carries other pins. A Python minor other than 3.13 still loads here (no
    bytecode; inference loads unpickle almost nothing) but makes every Colab checkpoint `cross_version`.
14. **Uploads of tens of MB through the Drive connector are untried** (V0b tries them); each part needs the
    owner's approval. Fallbacks: 5 MB parts, a rebuild on Colab checked by hash and kept on Drive, a data branch.
15. **Drive is not a transactional store**: a read-back verifies the mount's cache; two runtimes see each other's
    files minutes late. The design never trusts an unrecorded copy (the JSON is last, the trainer re-checks), fences
    a run by LIVE.json with a 10 minute threshold, and the sheet tells the owner how to tell a lost browser from a
    lost runtime. The 10 minutes are inferred, not measured.
16. **R2L's budget holds two fits, not three** (Q5); the cost per evaluation is uncertain by a factor of two, so a
    fit may end at its third or fourth block of six generations; the reported delay cells may read "not run".
17. **`PoolVecEnv`'s worker loop re-implements `DummyVecEnv`'s 40 lines** to keep torch out of the workers; its
    equality with SB3's own class is tested bit for bit at every worker count.
18. **The row wrapper lends a recorder through an instance attribute** and the reference playback uses
    `bar._Played`, a private name. Both are pinned by tests and frozen with state A; if round A renames either
    before state A, `lookbar` follows.
19. **The reward's new weights are untested** (W2); G8b's one fix changes task weights only.
20. **robot_floor_carpet stays unsolved for the base** [scratch C3], so a quarter of the envs give short episodes;
    the floor is reported, never gated.
21. **The fingerprint's observations are synthetic** (D14); the row equivalence check covers real observations.
22. **Estimates rest on a stub env**; step V4 replaces them, measured alone or labelled provisional.

**For the owner** (through the lead): the approvals of the uploads (V0b's three trial files, then the bundle's
parts); consent only if the data-branch fallback is needed; the day-one smoke; to be told before S2-4 that
exploration starts with short episodes by the plan's sigma (item 5); to be told before 60 cm Colab hours if the
local 60 cm run fails G8b's reading, with the remedy its record points to; to be told first if S2-4 shows no gain,
with the per-term table; the jaw cover decision before any `-X` set (V0).

## Review log

One line per finding of the three reviews: id, accepted or rejected, what changed. "Checked" names what the
architect verified in the code or by re-running the reviewer's script.

| Id | Verdict | What changed |
|---|---|---|
| L1 | accepted (checked: `control.py:2302`, `policy_rows.py:169-188`, `env.py:845-856`) | G8a is read by Stage 1's shorter-horizon rule; for a plan the env runs `ceil(n_row / 10)` steps so every row step is compared; the "same end" rule ignores the env's last partial step; a fast test with a length that is no multiple of 20 ms (3.2 step 5, 3.7, Z1) |
| L2 | accepted (checked: `bar._judge` raises `Unreadable` when a trace ends before the command or the window) | four outcomes per episode; a policy row's unread episode (`clock_behind` or `unreadable`) and too-few-steps are "not succeeding", never errors; the monitor never re-runs (D15, D17, 3.5, M1, U5, U6); tagged as differing from C8 for policy rows |
| L3 | accepted | reward term `progress` (weight 0.10, full at 10 % short, steady walking only) [differs: plan]; survival credited as progress in M3; the mean clock factor recorded per monitor episode (D9, D15, 3.4) |
| L4 | accepted as a proposal for the lead | the heading error as a deployable actor input (88/130, 102/146) and a `heading` penalty [differs: plan]; the fallback wording if the lead keeps the plan's list (D7, 3.1, 3.4, E3) |
| L5 | accepted in part | the learning-rate probe must pass at sigma 0.1 and 0.03; `kl_cut` flag from `updates.jsonl`; the fine-tune's rate is probed at the parent's sigma (K8); `n_steps` 1024. Sigma 0.1 itself is kept (the plan's and the lead's requirement) and put to the lead and owner with the measurement (D12, section 7 item 5); the sigma probe stays reported |
| L6 | accepted | 18 distinct seeds, `6001 + 3 x condition index + repetition`; each record counts its distinct start seeds (D15, M1) |
| L7 | accepted | reward term `range` on decision 14's required margin (offsets stay unclipped so the zero action stays exact); the servo mean is bias-corrected with a 1.0 s constant; R0L's 18 episodes run before S2-4; failures per term in every monitor record; M3b proposed for the "survival route closed" case (D8, D9, 3.4, M3, section 7 item 1) |
| L8 | accepted (checked by arithmetic: slot 1 with even g gives odd k only) | directions change by block of six generations; the test reads the table per (kind, direction); validation directions fixed by seed; a cut fit ends at a whole block (3.8, Q2, Q4, Q5) |
| L9 | accepted | D11 (b) reworded: the curriculum is a live option that C2b does not test; M6's checks also run at 0 ms for the policy and R0 and record sigma, the minibatch share and S, so a failure points to a remedy |
| L10 | accepted as a question for the lead | D4 states the consequence; proposed look-set counterparts of the twelve no-loss conditions, the finite 0.35 rad/s turn for the periodic one; the row still refuses other sets (D4, 3.11, section 7 item 3) |
| L11 | accepted (checked: `control.py:1199-1200`) | every tracker built at construction on the nominal model; the nominal arrays restored before the script is chosen; `damp`/`fric` asserted at every reset; tests (3.2, 2.3, section 4) |
| L12 | accepted | O4 compares the cut-offs at equal filtered action RMS and says what the measurement cannot show |
| L13 | accepted | `clearance` also charges a foot loaded through the reference's mid-swing (drag, stub step); the label "read as the bar reads" now holds (3.4, W3) |
| L14 | accepted in part | R6's mid-start sets the action history and filter state to the reference's targets; R6's expected outcome is stated in the rules. Gating the contact and head terms by tracking: rejected (R6 is "R4 with R5's reward" by the plan; gating would give R5 and R6 different rewards) |
| L15 | accepted (checked: `legs.py:43`, `LegGap.moving`) | the 3 mm ramp on moving pairs, "below zero" on all pairs, fixed pairs listed, the build stops on a fixed pair below zero; W4 states that a one-pose contact is penalised, not terminated (D10) |
| L16 | accepted (checked: `control.py:1048-1061`) | every table continues past `t_end` as `Track.joint` does, tested at clock factor 1.25; `mid_spans()` leaves holds out (3.3) |
| P1 | accepted, by removing the edit | no edit to `bar.py`: the look reading is `lookread.py` plus a row wrapper and companion records in `lookbar.py`; no bar key moves at state B, so state-A baselines stay comparable and nothing is re-run (D17, D18, 2.3) |
| P2 | accepted (checked: re-ran `review_prp/exp_resume.py`, same three outcomes) | `force_reset=False`; `learn(total_timesteps=budget - n, reset_num_timesteps=False)`; nothing resets between load and boundary; `obs_rms.count` asserted; `PYTHONHASHSEED=0`; no set iteration (D16, 3.6) |
| P3 | accepted | as L11; a history-free reset test; `gates2.kill_resume` uses K3's settings (32,768 steps per boundary, 131,072 in all) so envs finish several randomised episodes on both sides of the kill |
| P4 | accepted | the run state is in `checkpoint_<n>.json`; a resume rebuilds from it, discards later entries, truncates `updates.jsonl`, runs a due check that has no record, and replays `monitor2.decide`; tiny mode has a scaled G8b; tests for a kill during the monitor and after the fix (D16, 3.6) |
| P5 | accepted (checked: `s2io.py:400-420`) | all live files under the ignored `stage2/tb/work/`; `nb collect` publishes finished records; dirty excludes own outputs; a test that `git status` does not change (D22) |
| P6 | accepted | one acceptance run replaces the dry run and the in-checkout notebook run: a clone of the remote, the bundle, the notebook's own cells, a process-group SIGKILL during a Drive copy, the local directory and the clone deleted, a second clone, RESUME (3.10, V12, N5); Y6 deletes the local job directory before its resume |
| P7 | accepted | commit order C1 then C2; REPO_REF is C1's 40-hex SHA, refused otherwise for a non-smoke launch; a resume reads REPO_REF from QUEUE.json (D24, K6, N1) |
| P8 | accepted | LIVE.json heartbeat with fencing; RESUME refused on a live run unless TAKEOVER; a superseded launcher halts; 10 minutes, not 5, because Drive syncs between runtimes late (D25, K7, 3.9) |
| P9 | accepted | a job without a checkpoint restarts from step 0 in place; checkpoint 0 is saved at the first rollout start; the "_b" rule is gone from the sheet (D16, 3.6, 3.12) |
| P10 | accepted in part | row build compares the run's recorded hashes of the observation and action code with today's; the measured fingerprint difference and tolerance leave the key for the companion record; `r5env.py` and `schedules.py` leave the row's sources and the R6 tracker moves to `r5core.py`. Kept in `describe()`: both version sets and `cross_version` (they are executed, not measured, and without an edit to `bar.py` the key's row block is the only place the bar's own record can hold what S2-12 asks) (3.11) |
| P11 | accepted | settings hold the resolved task table, each used file's and entry's hash, the freeze index and the plant inputs; a later freeze is accepted when those are unchanged (3.2, 3.6, K4) |
| P12 | accepted | `train2.SOURCE_FILES` is a fixed tuple including what the children run and the four repository files; hashed per session, checked by each child (3.6) |
| P13 | accepted (checked: `designlab/__init__.py:15-16`) | `tbenv.child_env("osmesa")` sets `PYOPENGL_PLATFORM`; `clip2` imports no torch and uses ffmpeg on PATH; the acceptance run requires a real clip here |
| P14 | accepted | `tbenv.repo_on_path` (appended); the final cell's release is in `nb.py` with a guarded import; the cell passes the thread variables and `PYTHONHASHSEED`; pins compared with the build label dropped and the CPU torch forced when the label differs (D19, 3.9) |
| P15 | accepted | the content block alone is hashed and quoted; provenance outside it; `UPLOADS.json` separate; USTAR, empty names, mtime 0; the two-process and umask test (D20, 3.10) |
| P16 | accepted | inference loads pass the spaces and `None` for the last-observation members; the statistics are plain arrays in `fingerprint_<n>.npz`; `run_uuid` and the settings hash in `run.json`, every checkpoint and monitor record; a row refuses a foreign checkpoint (D14, 3.6, 3.11) |
| P17 | accepted | "copied", not "verified"; an older checkpoint is deleted on Drive one cycle after a newer one is copied; Drive gets every 1M steps and each check's; the trash is named in the sheet (3.9, K2) |
| P18 | accepted (checked: `policy.py:154-162`) | `kw = model_kwargs(...); kw.update(...)`; the trainer asserts that no schedule is a function; `save_checkpoint` refuses an archive with a bytecode member (3.6, 2.3) |
| COV-01 | accepted (checked: `STAGE2_75_PLAN.md` R4, `bar._check_rule`) | every acceptance before task 3's first unit runs on the S1 view or a synthetic set; real-set acceptances wait; rule X1 for look sets; `dev_episodes.json` (D23, sections 4 and 5) |
| COV-02 | accepted, option (b) (checked in `bar.py`: `BarBuilt.recorder`, `reference_recorder`, `condition_record`, `_Played.__init__`) | as P1; the false "no change asked" sentence is gone; the reference comes from `bar._Played`; tracks with a periodic turning gait are never played and have no look reading (D17, 2.3, 3.5) |
| COV-03 | accepted | G9-ref (kinematic, before the first look-set episode) and G9-run (after R0L), each per body and head; what (b) means for rungs outside the ladder and for `-X`; the L1 hash check; references no bar item plays get check (f); S2-11's per-body records (3.7, Z2) |
| COV-04 | accepted | as L2: too few own steps fails the three terms when the reference has enough; an error only when the reference has too few (3.5, U5) |
| COV-05 | accepted as a question for the lead | as L10 |
| COV-06 | accepted | the bar's head script H1 and the row `R5G` are pre-registered now; head-first turns labelled as deferred (3.11, E5) |
| COV-07 | accepted | G8a is its own module `g8a.py`, owned by CE, with a stock form that needs no `bar`; `train2`, `monitor2` and `lookbar` meet through lazy imports and stubs (2.1, section 5) |
| COV-08 | accepted | tiny checks run full-length episodes on each condition's first seed; tiny mode has a scaled G8b; the acceptance run asserts 6 judged episodes and a gain record (3.6, 3.10) |
| COV-09 | accepted | as P4 |
| COV-10 | accepted | as P9; the launcher's rule: a rule-stopped job lets the next start, a crash is retried once and then halts the run (3.9) |
| COV-11 | accepted | as P8 |
| COV-12 | accepted | as P7; `ALLOW_VERSION_CHANGE` and `ALLOW_SOURCE_CHANGE` are parameters of cell 1, recorded and tested (D24, 3.9) |
| COV-13 | accepted | SMOKE.json is written after Y7, before the rebuild; Y8 is capped at 40 minutes with a "not finished" outcome; the sheet carries each item's minutes, the V6 build time measured when the set is built (Y table, 3.12) |
| COV-14 | accepted | Colab and the acceptance run verify with `bundle.verify` and `setio.read_inputs(require_frozen=True)`; `sets verify` stays the lead's; `environments` importable through `tbenv.repo_on_path` (3.9) |
| COV-15 | accepted | as P5 |
| COV-16 | accepted | the decisive cell is counted first and always finishes (1.2 core-h reserved); the other delays get 0.4 and may read "not run: budget", tagged as differing from S2-18; rows named by their fit and resolved through `index.json` (3.8, Q4-Q6) |
| COV-17 | accepted | the rules text carries its own formulas: the reward table with every scale, the layout, R2L's item terms, K3's settings, the rate floor, the clip's episode; M3's "best earlier check" tagged [NEW precision] |
| COV-18 | accepted | R6 tests on 14 and 16 joints, the mid-start, a tiny run with a resume, the row against the env, and the record `r6_start.json` (section 4, V7) |
| COV-19 | accepted | `s2load import` writes `colab_archive.json`; the acceptance run writes `ACCEPTANCE.json`; `UPLOADS.json`; `nb hours` writes `hours.json` |
| COV-20 | accepted | `ckpt_<n>.tar` per check on Drive; `s2load import`; V0b tries the 170 kB and 20 MB cases before the format is frozen (D14, V0b) |
| COV-21 | accepted | as P14 |
| COV-22 | accepted (checked: `tools/memguard.sh:11-20`) | workers and the monitor are subprocesses of `-m designlab.stage2.train2`, so one PROTECT string covers them; the worker loop imports no SB3; rates lowered to 350 and 290; the fast worker test stops at W = 4; clips are the launcher's and never hold the trainer (D13, 3.9, section 6) |
| COV-23 | accepted (checked: `tools/requirements-study.txt`) | PyOpenGL 3.1.10 and pillow 13.0.0 pinned; imageio-ffmpeg dropped; ffmpeg checked on PATH; no unpinned fallback (3.9, N2) |
| COV-24 | accepted | run plan step V0: G1 of `-X`, the jaw cover decision, one kinematic set build on an `-X` model (D5) |
| COV-25 | accepted | S2-4 checks every 500,000 steps so earlier kept checkpoints are reported; Q2 says which seed slots the fit uses and why; (3) as P10 |
| COV-26 | accepted | owners named (CG: `run_trackb.sh`; the lead: the rules file); clones outside the checkout, checked against `ls-remote`; the reviewers' runs are in "what had been run"; `carpet_walk` and stops from other walks labelled in T2 |
| COV-27 | accepted | V0b's trial uploads; a rebuild's arrays kept on Drive and reused; the sheet's instructions for High-RAM not offered, Drive refused, a FAILED smoke line, the upload approvals; cell 7 reprints lines 1-3 |
| COV-28 | noted | a reference table; no change of its own |
| COV-29 | accepted | the reviewer's list at the end of section 4 now holds its twenty points, adapted to the no-edit look reading |
