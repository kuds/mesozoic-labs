# Round C design: R5, R6, R2L, the trainer, the gates, the notebook and the Drive bundle (Track B)

Architect's design, 15 Nov 2026, for the lead. Scratch file, not committed. Paths are relative to
`/home/user/mesozoic-labs/hardware/trex_form_study` ("FS") unless absolute; `REPO` is `/home/user/mesozoic-labs`. Code
line numbers are of the main checkout at `e0ad284`. "C1"-"C11" are the lead's requirements, "D n" this document's
decisions, "S2-n" the plan's rules (section 9 as in `/home/user/mesozoic-wt/prereg`), "round A" is
`ROUND_A_DESIGN.md`, whose APIs (`s2io`, `setio`, `readings`, `moves`, `sets`, `setscreen`, `bar`, `barreport`,
`moves_look`) this design builds on and does not redesign. Nothing in a checkout or worktree was edited and no pytest
was run (`git status --porcelain` of the main checkout was the same before and after the experiments).

Every number marked **[scratch]** comes from a scratch experiment of this session (section 0). None is a record and
none may be cited as one: they show that a construction works and what it costs. The companion file of rules to fix
before the runs is `PREREG_TRACKB_draft.md` (same directory as this file); rule ids such as "W2" or "K3" refer to it.

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

What these settle:

1. **G8a can be met bit for bit, across plant instances, at 0 and 10 ms** (C1), provided the env leaves
   `residual None` and `rate_factor 1.0` at a zero action, mirrors `env.Stage1Env`'s reset and step order, and uses the
   bar's own delay wrapper.
2. **R5's env is about twice as fast as Stage 1's R3 env** (C2: about 440 against 358 env steps/s on hard floor,
   `STAGE1_PLAN.md` decision 26), because the tracker's feedback terms were most of an env step. Every compute
   estimate of section 6 starts from 400 (hard) and 330 (carpet) env steps/s per process, to be replaced by the
   validation record.
3. **Mid-task starts and the delay queue need no edit to a key module** (C3); neither do 16 joints, commands, the
   trainer's modes or the schedule (section 2.3 goes through map 05's list).
4. **On `60-N-T-10` a delay curriculum would not help the base**: it falls as often at 0 ms as at 10 ms (C2b).
5. **Lesson 16's count is cheap to keep** (C4), and a boundary design (D16) can make a resumed run the uninterrupted
   one bit for bit.

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

**D4. "Walking turns at 0.35-1.0 rad/s" are round A's finite 90 degree turns from the 0.40 m/s walk**
(`turn90_<rate>_<dir>` for the four rates of `setio.TURN_RATES`); R5 trains on no periodic walking-turn gait.
[a reading of plan B.1, stated.] Every turn item of the bar and of the turn-rate ladder is such a move, each
contains its own entry, hold and exit, and it is causal. Rejected: periodic look turns with enter and exit moves
(round A 3.9 calls them moderate to hard; no bar or ladder condition plays one). The turn in place stays a periodic
gait (D5).

**D5. What R5 needs from the sets that round A's `sets.items()` does not list** (for the lead's look work
package; nothing else is asked of round A's set code):

| Needed | Why | Without it |
|---|---|---|
| `refs/inplace_0.5_left.npz`, `inplace_0.5_right.npz` (the periodic 0.5 rad/s turn in place under the set's laws), with `transitions/start_inplace_0.5_<dir>.npz` (stand to the turn) and `stop_inplace_0.5_<dir>.npz` (the turn to a stand) | plan B.1 names the turn in place and its start and stop; Stage 1's no-loss conditions hold both in-place turns | the two `inplace_*` tasks are dropped from the task table and recorded as absent (T1); the plan's B.1 list is then not met |
| Set `V6` built, screened and frozen on **each body and head trained**: `75-M-H-10`, `75-M-H-10-X`, `60-N-T-10`, `60-N-T-10-X` (the `-X` sets head neutral: neck yaw 0, jaw on its stop; generated on the `-X` model, whose mass and CoM differ) | G8a, G9 and training are per body and head | no run of that body or head |
| Set `V6noF5` on `75-M-H-10` and `60-N-T-10` | R2L (C6) | no R2L on that body |
| Every item of `sets.items()` on those: six walks, `start`, `nominal_to_v<v>`, `stop2_v0.4`, `turn90_<rate>_<dir>` | the task table (T1) | a task whose reference is `available: false` is dropped and recorded (round A's "not run" rule; ruling 4 expects this for the 60 cm quick turn) |
| Each walk file holds `phase` when it is a K-stride window (round A 3.9 lists it among the look keys); each move file holds `phase` (Stage 1's do) | the phase input and the swing clock (section 3.3) | a walk file without `phase` is read as uniform in its window (exact for one-stride files) |
| Optional, not required: `stop2_nominal` (lets the nominal-walk task end in a stand, as Stage 1's `walk` task) and `v0.4_to_nominal` | Stage 1's task shapes | the nominal-walk task ends in the gait, as the bar's `walk/nominal` does |

Names are proposals; they must come from `setio` (no transition may be named `stop` or `stop_in_place_*`, round A
D2; `stop_inplace_0.5_left` is not in `transitions.TRANSITIONS`, so `control.py:994` registers no Stage 1 stop
event for it). R5 reads from an entry only keys that are in round A's `WALK_SUBSET` and `MOVE_SUBSET` plus
`available`: `period_s`, `window_s`, `strides`, `speed_m_s`, `yaw_rate_rad_s`, `crouch_frac`; `move.kind`,
`move.causal`, `move.command_phase`, `turn.angle_rad`, `turn.rate_rad_s`, `turn.direction`, `gaits.a`, `gaits.b`.
No new entry key is asked for.

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
deployable blocks (`env.py:138-139`) with `q` and `qd` nj wide, the command, the last three clipped actions
(3 x (nj + 1)) and, when the plant has `neck_yaw` and `jaw`, a 4-wide head command. Critic: Stage 1's privileged
blocks with `joint_error` nj wide, two more plant factors, the foot errors and the reference's contact. 14 joints:
actor 87, observation 129, action 15. 16 joints: actor 101, observation 145, action 17. Every index is by joint
name (`-X` puts the legs from index 4). Rejected: one padded 16-joint layout for all bodies (the plain bodies'
policies would carry inputs that never move, and the head comparison would compare two layouts anyway).

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

**D9. Reward: Stage 1's terms and weights unchanged, plus the plan's additions with their own weights, all in the
env config** (section 3.4, rules W1-W4). Two terms of Stage 1's change their reading and are labelled: the
clearance penalty is read on the foot's own swing (the bar's reading; Stage 1's is the reference-clock reading that
misreads a late lift-off), and the duty band follows the owner's decision below 0.30 m/s. The tracking term keeps
Stage 1's target (the reference's own speed and yaw rate at the commanded clock), so it never fights the imitation
of a start, change or stop; only the *input* is the piecewise-constant command. Source of the weights: Stage 1's
where the term is Stage 1's; the new ones are this design's, set before any run on [look] 3.3's recipe (BD-X and
Olaf: imitation of joints, feet and contact timing, action rate, impact, head, a thermal-style servo term), sized
so that no new term outweighs Stage 1's joint imitation (W2 gives each number). None was tuned on a result.

**D10. Leg-to-leg interference enters the reward and the termination** (`LESSONS_LEARNED.md` 2.8). At the end of
every env step (50 Hz) `legs.LegGap(body).at(qpos)` gives the smallest signed left-leg to right-leg distance of the
screened geometry, the reading the bar's turn item judges (`metrics.leg_gap`, `legs.py:29-61`). Penalty: 1 at
contact, falling to 0 at `clearance.TIGHT` (3 mm). Termination: a distance below zero at two env-step ends in a
row (40 ms). G8a runs the env with the termination off and records when it would have fired (the check only reads
`qpos`). Rejected: reward only (a policy could still pass a foot through the other shank once per turn and keep the
reward of the turn); termination on the first sample (a graze of one pose would end an episode the bar would only
fail on turns). Cost 0.2 ms per env step [scratch C2]. Not covered, stated in the env's rules: a leg against the
trunk or tail.

**D11. The 60 cm zero-action problem: one recipe for both sizes, and a local run on `60-N-T-10` to G8b's own
reading point before any 60 cm Colab hour.** The facts: R0 keeps 3 of 20 nominal walks up at 0 ms (S1 section 4);
the falls come 4-9 s into an episode and are as frequent at 10 ms as at 0 ms [scratch C2b]; the stand and the turn
in place stay up. What that changes, and the options:

| Option | What it does | Verdict |
|---|---|---|
| (a) Same recipe as 75 cm | Episodes of the base last 200-450 env steps before a fall, so every rollout holds dozens of falls with their seconds of walking before them; half of the episodes start mid-task in reference states; `policy.LOG_STD_RULE` cannot be applied (the plan fixes sigma 0.1 for that reason) | **kept** |
| (b) A delay curriculum (start at 0-4 ms, widen to 0-24 ms) | Removes nothing the base falls from [scratch C2b: same falls at 0 and 10 ms]; adds a schedule that the two heads and the two sizes would have to share | rejected as a default; named as an option for the owner if (e) fails |
| (c) More mid-task starts or starts only late in the walk | The base falls within seconds from a stance start already; mid-task starts are half the episodes by the plan; a body-specific start mix breaks "one recipe" | rejected |
| (d) A feedback base (the tracker's terms) under the 60 cm policy | S2-3 chose R5 without a tracker, and the tracker falls in nearly every 60 cm walk at 10 ms (S1b section 4) | a plan change: owner only |
| (e) **A local 60 cm run to 5M steps** (`60-N-T-10`, current head, seed 41, which is no judged seed) with the 5M monitor read by G8b's rule | First evidence on the hard body before Colab; about 1.6 h of wall time on 3 cores at the measured rate (section 6) | **recommended**; pre-registered as M6: it starts no Colab run and stops none, but if the nominal walk stays up in fewer than 2 of 3 at 5M, the owner is told with options (b), (d) and "launch anyway under G8b" before a 60 cm job is queued |

S2-4's go or no-go stays the 75 cm reading, as the plan has it.

**D12. PPO: design 2 with 8 envs, sigma 0.1, the probed learning rate, schedules as objects** (rules O1-O3).
`n_envs 8`, `n_steps 2048` (a rollout is 16,384 steps; Stage 1's was 8,192 with 4 envs), `batch_size 256`,
`n_epochs 5`, `target_kl 0.03`, `log_std_init ln 0.1`, `learning_rate = schedules.LinearSchedule(lr, 0.3 lr)`
over the run's budget, `clip_range = 0.2` (a float: SB3 wraps it in its own schedule class; Stage 1's archives hold no
bytecode in `clip_range`, map 08 section 2, and scratch C4's archive none at all). `lr` is the largest of 1e-4, 3e-5, 1e-5 at which every minibatch step of 6 updates runs, **on every
body and head of the launch** (one value for all runs, so two heads never differ in it). Every update's minibatch
steps run, of the possible, go into the run's `updates.jsonl` (the optimizer's `step` wrapped, C4).

**D13. 8 envs in W worker processes: `designlab/stage2/vecpool.PoolVecEnv`.** Each of W spawned processes hosts a
`stable_baselines3` `DummyVecEnv` of its share of the 8 envs (contiguous blocks, `numpy.array_split(range(8), W)`, which puts the two
carpet envs in two workers whenever W is 2 or more); the parent forwards
`step`, `reset`, `seed`, `env_method`, `get_attr` and returns results in env order. Auto-reset, the terminal
observation and seeding are `DummyVecEnv`'s own code. `W = min(8, vCPUs)` unless `--workers` says otherwise;
`W = 0` runs a plain `DummyVecEnv` in the trainer's process. Because every env has its own generator and plant, a
rollout does not depend on W (tested bit for bit at W = 0, 1, 2, 4, 8), so "PPO's batch and floor mix never depend
on the runtime" holds by construction. Env i's floor is `("hard", "hard", "hard", "robot_floor_carpet")[i % 4]`:
six envs on hard floor, two on robot_floor_carpet. Rejected: SB3's `SubprocVecEnv` (one process per env: 8
processes and 2.8 GB on any runtime, 4 times oversubscribed on a 2-vCPU runtime and unable to leave a core to the
panels here).

**D14. The policy fingerprint (S2-12) is made from a fixed, layout-seeded observation table and saved beside every
checkpoint** (`designlab/stage2/s2load.py`; section 3.6). A checkpoint is a set of four files written in this
order, each through a temporary name: `model_<n>.zip`, `vecnormalize_<n>.pkl`, `fingerprint_<n>.npz`,
`checkpoint_<n>.json` (the three files' sha256 and bytes, the versions, the step count). The JSON is written last:
its presence and matching hashes are what "an intact pair" means, on local disk and on Drive. Every load, in the
trainer, the monitor, the rows and the smoke, goes through `s2load.load_checkpoint`, which checks the hashes, runs
the scaffold's `checkpoint_pair_problem`, loads through `environments.shared.policy_loading.load_sb3_model` with
the schedule objects passed (S2-14), and reproduces the fingerprint within 1e-5. Rejected: observations sampled
from env rollouts (they would tie the fingerprint to one machine's floating point before the archive exists; the
table is stored in the file either way).

**D15. The monitor is the bar's own code on a saved checkpoint, in a child process** (`monitor2.py`, rules M1-M5).
Its 18 episodes are six bar conditions of the run's set (`walk/nominal`, `walk/v0.4`, `turn/q90_left`,
`stop/two_step`, `nudge/forward`, `nudge/left`; mu 0.9) on episode seeds 6001-6003, under `R5/<seed>@10ms` built
from the checkpoint's files through `s2rows` and `bar.Delayed`, judged by `bar.judge` with the look reading
(D17). The same 18 episodes under `R0@10ms` are run once per run. The trainer saves the checkpoint, starts the
monitor as a subprocess (`MUJOCO_GL=disable`), waits, and reads its record; a monitor that crashes is recorded and
never stops training by itself. So the stop rules, G8b and S2-10 are read on the path that judges the policy
later, here and on Colab. Rejected: Stage 1's in-process monitor on the training env (`train.py:338-374`): other
conditions than the bar's, no look reading, and the bar's context would live in the trainer's process.

**D16. Kill and resume: every checkpoint is a boundary, in an uninterrupted run too, so a resumed run is the
uninterrupted one bit for bit** (rules K1-K4). At the first rollout start at or past each multiple of 250,000
steps the trainer saves the checkpoint and then: seeds SB3, torch, numpy and the vec env with
`100,000 x seed + 100 x b` (b the multiple's index; env i resets with that + i), resets every env, zeroes
`VecNormalize`'s running returns and sets the model's last observation and episode starts from that reset. A
resume loads the newest intact checkpoint and performs the same boundary before its first rollout. Stage 1 gave
each *session* new seeds (`train.SEED_RULE`), so a resumed run was never the uninterrupted one. Cost: at most
8 x 1,500 env steps of cut episodes per 250,000 (under 5 %). The validation asserts equality of every policy
tensor, the optimizer's state and the normalisation statistics between an uninterrupted and a killed run on this
machine; across machines or library versions nothing is asserted (S2-13 applies). Rejected: a stated tolerance
(any nonzero difference grows through PPO; there is no principled number).

**D17. The look reading is one module, `designlab/stage2/lookread.py`, read on a `metrics.Recorder` whether it was
filled by a closed-loop episode or by a kinematic playback** (rules U1-U5; section 3.5). The reference's values
are the same functions applied to round A's `bar.reference_recorder` playback of the episode's own track over the
same window at the nominal commanded clock, so the two sides of each band share their definition. Bob, tail phase
and toe-off are segmented by the robot's own touchdowns and lift-offs (the 2 % load rule), not by the reference's
clock. "Success with the look" is `success` (reading 2) and all four bands, for walk and turn items; for nudge and
stop items reading 1 is reading 2, labelled. It needs 50 Hz poses in the recorder and a hook in `bar.judge`:
section 2.3 says exactly what, and D18 when.

**D18. Code state B.** Round C adds modules and asks one change of a round-A module that is in the bar's key:
`bar.py` gets a reading hook, 50 Hz poses for sets with look bands and a trace key (section 2.3). If the lead can
put it in before state A, no key ever moves; otherwise it lands at state B, the commit at which `lookread.py`,
`r5lib.py`, `r5core.py`, `r5env.py`, `s2rows.py`, `s2load.py`, `schedules.py` and that `bar.py` are frozen, before
any look-set row (R0L, R2L, R5, R6) runs. State-A records (the 75 cm track's three sets) are never re-run under
state B; they are cited. From state B until the Colab phases end, none of those files changes without the
affected baseline rows being run again into a new directory (S2-17's rule, extended to them).

**D19. The notebook is thin and runs nothing heavy in the kernel** (`stage2/stage2_training.ipynb`; section 3.8).
Each code cell calls one step of `python -m designlab.stage2.nb` in a subprocess with `MUJOCO_GL=disable`,
`PYTHONPATH=.` and cwd FS, and prints its lines. So the freshly pinned packages are the ones every job imports
(the kernel may hold an older numpy in memory), the kernel never loads OSMesa or EGL, a segfault in a child is a
return code and not a dead kernel, and the same steps run here without Colab. **Copied from the scaffold, adapted:**
the setup cell's Colab detection, legacy-gym purge and Git block (`notebooks/sb3_training.ipynb` cell 3, lines
36-58, statement for statement, with `designlab` added to the "restart" refusal) and the Drive mount (cell 8,
lines 8-21, with the Stage 2 root). **Imported from the repository, unchanged:**
`environments.shared.notebook_runtime.disconnect_runtime`, `environments.shared.curriculum.checkpoints.
checkpoint_pair_problem`, `environments.shared.policy_loading.{inspect_sb3_archive, load_sb3_model,
sb3_archive_load_preflight}`. **Not taken:** everything bound to `ModelId`, stage manifests, gate verdicts and the
result bundle; `newest_intact_periodic_pair` (it walks SB3's `_steps` names; ours are `model_<n>.zip`); the
kernel-wide `MUJOCO_GL=egl`; the "install only if a module is missing" condition (the pins are applied whenever
`pip freeze` differs from them); the GPU metadata. Rejected: training in the kernel as the scaffold does (one job,
no queue, the kernel's GL backend and numpy).

**D20. The Drive bundle is a few plain `.tar` parts of at most 20 MB with a committed JSON manifest, built
deterministically** (`designlab/stage2/bundle.py`; section 3.9). One part per (group, body): the model files of a
case, the arrays of a set and body (split when over 20 MB). Tars are written with sorted names, zero mtime, uid and
gid, mode 0644 and no compression, so the same files give the same tar bytes and a fresh clone can rebuild a part
and compare its sha256. The manifest lists every file and every part with sha256 and bytes; the notebook prints the
sha256 of the manifest's canonical JSON (`s2io.sha256_json`), which no line-ending rule can change. Fallbacks if an
upload of that size fails, in order: (1) the same content in parts of at most 5 MB (`--part-mb 5`; more owner
approvals); (2) only the model parts (3 MB in all, the size class verified on 14 Nov) and the arrays **rebuilt on
Colab** by `sets build`, accepted only if every sha256 equals the manifest's (the smoke runs this once anyway);
(3) with the owner's consent, the parts force-added to a data branch that is never merged, which the notebook
fetches (`BUNDLE_SOURCE = "git:<ref>"`; about 50 MB per body and head, permanent in the repository's history).
Rejected: `.zip` (ignored by git, and its member timestamps make it irreproducible); one archive (a failed upload
of 200 MB repeats 200 MB).

**D21. R2L is a new module on the bar's conditions; `speedfit.py` is imported, not edited**
(`designlab/stage2/r2lfit.py`; section 3.7, rules Q1-Q6). It reuses `tune.CMA`, `tune.Pool` (with its own worker
factory), `tune.score`, `tune.training_order`, `tune.selection_key`, `tune.to_unit`, `from_unit`,
`repair_penalty`, `speedfit.servo_delay`, `speedfit.start_record` and `speedfit.gains_vector`. Its evaluator
builds five bar episodes per evaluation through `bar.BarContext` and judges them with `bar.judge`; the delays come
from a rotation table written before generation 0 (S2-18); the selection reads the 10 ms cells.

**D22. Track B's outputs live under `stage2/tb/`** with its own `.gitignore` (`checkpoints/`, `drive/`, `clone/`,
`parts/`, `*.tar`, `*.mp4`, `*.tmp`, `.lock`, `logs/`, `partial/`, `traces/`), so no rule of round A's
`stage2/runs/.gitignore` changes. Committed: gate and validation records, `run.json`, monitor records, R2L's
records, the bundle manifest and dry-run record, the steps sheet. Never committed: archives, statistics,
fingerprints, tars, clips, logs.

## 2. Module layout

### 2.1 New files

| File | Purpose | In a record key |
|---|---|---|
| `designlab/stage2/schedules.py` | `LinearSchedule`, `ConstantSchedule`: module-level classes (S2-14) | R5/R6 rows' sources |
| `designlab/stage2/s2load.py` | the one loader; checkpoint files, integrity, fingerprint, version sets | R5/R6 rows' sources |
| `designlab/stage2/r5lib.py` | scripts over a frozen set: tracks, command, phase, start states, reward references; the task table | R5/R6 rows' sources |
| `designlab/stage2/r5core.py` | layout, observation, action mapping and filter, head overlay, reward terms, interference: shared by the env and the rows | R5/R6 rows' sources |
| `designlab/stage2/r5env.py` | `R5Config`, `R5Tracker`, `R5Env`, `R6Env`, randomisation, starts, `make`, `throughput` | R5/R6 rows' sources |
| `designlab/stage2/s2rows.py` | policy rows `R5/<seed>` and `R6/<seed>` for the bar runner | its own (row sources) |
| `designlab/stage2/lookread.py` | the look measures and bands on a recorder | **the bar's key for look sets (state B)** |
| `designlab/stage2/vecpool.py` | `PoolVecEnv` | no |
| `designlab/stage2/train2.py` | the trainer and its probes; CLI | no |
| `designlab/stage2/monitor2.py` | the 18-episode monitor, gain rules, G8b, S2-10's record; CLI | no |
| `designlab/stage2/clip2.py` | one closed-loop clip from a monitor trace (run with `MUJOCO_GL=osmesa`) | no |
| `designlab/stage2/gates2.py` | G8a, G9 and the validation records; CLI | no |
| `designlab/stage2/r2lfit.py` | R2L's fit, selection and rows; CLI | R2L rows' sources |
| `designlab/stage2/bundle.py` | the Drive bundle: build, verify, unpack, dry run; CLI | no |
| `designlab/stage2/nb.py` | the notebook's steps and the queue launcher; CLI; **standard library only at import** | no |
| `stage2/stage2_training.ipynb` | the notebook (8 cells) | |
| `stage2/tb/.gitignore`, `stage2/tb/colab_steps.txt`, `stage2/tb/requirements-colab.txt` | ignore rules; the steps sheet; the pins | |
| `stage2/tb/PREREG_TRACKB.md` | the rules (from the draft, after the lead's review) | |
| `tests/test_stage2_{schedules,s2load,r5lib,r5core,r5env,s2rows,lookread,vecpool,train2,monitor2,gates2,r2lfit,bundle,nb}.py` | section 4 | |

Every module with a CLI uses round A's skeleton (thread variables before numpy, `main()` of the importable copy).
`train2.py`, `monitor2.py`, `gates2.py` and `r2lfit.py` start with

```python
from __future__ import annotations
import os, sys
_gl_before = os.environ.get("MUJOCO_GL")  # designlab/__init__.py:15 has defaulted it to osmesa or egl by now; a Colab kernel may export egl
os.environ["MUJOCO_GL"] = "disable"       # assigned, never setdefault
os.environ.pop("PYOPENGL_PLATFORM", None)
if "mujoco" in sys.modules and _gl_before != "disable":
    raise RuntimeError("mujoco was imported with a GL backend before this entry point set MUJOCO_GL=disable")
```

before any other import (`python -m designlab.stage2.train2` imports `designlab/__init__.py` first, which only sets
the variable; nothing has imported mujoco yet). Spawned workers inherit the environment and assert it.

### 2.2 Public APIs

```python
# schedules.py
class LinearSchedule:   def __init__(self, start: float, end: float); def __call__(self, progress_remaining: float) -> float
class ConstantSchedule: def __init__(self, value: float);             def __call__(self, progress_remaining: float) -> float
def describe(s) -> dict                      # {"class", "start", "end"} | {"class", "value"}: what settings record

# s2load.py   (imports torch and SB3 inside functions)
VERSION_KEYS = ("python", "numpy", "mujoco", "scipy", "torch", "stable_baselines3", "gymnasium", "cloudpickle")
FINGERPRINT_ROWS, FINGERPRINT_TOL = 256, 1e-5
def versions() -> dict                       # the imported modules' own __version__; torch's "+build" label kept here
def comparable(v: dict) -> dict              # torch's build label dropped (S2-12); what is compared
def checkpoint_files(job_dir, steps) -> dict # {"model", "vecnormalize", "fingerprint", "record"} paths
def list_checkpoints(job_dir) -> list[int]   # step counts that have a checkpoint_<n>.json, sorted by the number in the name
def fingerprint_obs(layout: dict) -> np.ndarray          # [256, obs_dim] float32, from default_rng([20261115, obs_dim, actor_dim])
def fingerprint(model, vecnorm, obs) -> np.ndarray       # deterministic actions, float32
def save_checkpoint(job_dir, steps, model, vecnorm, layout, extra: dict) -> dict   # the four files, in order, each atomic
def pair_problem(job_dir, steps) -> str | None           # missing file, sha256 or bytes mismatch, checkpoint_pair_problem's reason
def newest_intact(job_dir) -> tuple[int | None, list[str]]   # newest step count without a problem; reasons of the skipped
def load_model(path, env=None, lr: tuple | None = None)  # load_sb3_model(path, algorithm="ppo", device="cpu", env=env,
                                                         #   custom_objects={learning_rate, lr_schedule, clip_range}); lr None: inference
def load_vecnormalize(path, venv=None)                   # VecNormalize.load on a venv, or the bare statistics for inference
def load_checkpoint(job_dir, steps, env=None, lr=None, tol=FINGERPRINT_TOL) -> dict
    # {"model", "vecnorm", "record", "versions_saved", "versions_here", "cross_version", "fingerprint_max_abs_diff"};
    # raises CheckpointError on a pair problem or a fingerprint beyond tol ("not judged": S2-12)
def keep(steps_all: list[int], monitor_every: int, checkpoint_every: int) -> list[int]   # K2: each check's and the newest two

# r5lib.py
TASKS: tuple[TaskDef, ...]                   # name, plan builder, weight, monitor tasks it stands for (T1)
class Script:                                # one plan on one plant: the track and its side tables
    plan: list; track: control.Track; t_end: float; events: list[dict]     # events: setio.add_events' (causal commands)
    def command(self, t_ref: float) -> tuple[float, float]   # (v, w) at the references' own clock: section 3.3
    def phase(self, t_ref: float) -> float                   # gait phase in [0, 1); 0 at a left lift-off
    def stride_s(self, t_ref: float) -> float;  def walking(self, t_ref: float) -> bool
    def swing_s(self, t_ref: float) -> float                 # the planned swing time of the gait in force (reference s)
    def support(self, t_ref: float) -> np.ndarray            # [2] bool, the reference's
    def foot_rel(self, t_ref: float) -> np.ndarray           # [2, 3] feet relative to the root in the heading frame (FK of the file's q)
    def head_rp(self, t_ref: float) -> np.ndarray            # [2] the head body's world roll and pitch about home (FK of the file's q)
    def start_state(self, t_ref: float) -> tuple[float, np.ndarray, np.ndarray]   # (snapped t_ref, qpos, qvel at the files' own rate)
    def mid_span(self) -> tuple[float, float] | None         # reference times a mid-task start may take (T3)
class Library:
    def __init__(self, p: plant.Plant, body: str, set_name: str, rate: float, root=setio.ROOT, require_frozen=True)
    inputs: dict                             # setio.read_inputs' first value: file hashes and the freeze's manifest sha256
    omega: float; crouch_frac: float         # control.omega0(p, q_home, crouch_frac=the set's): round A D16
    tasks: dict                              # name -> {"plan", "weight", "available", "reason", "seconds"}
    def script(self, task: str) -> Script;   def from_plan(self, plan: list) -> Script
    def describe(self) -> dict               # the task table as built, absent tasks with their reasons

# r5core.py
def layout(joints: list[str]) -> dict        # {"blocks": {name: (i0, i1)}, "actor_dim", "obs_dim", "action_dim", "head_x", "nj"}
REWARD_WEIGHTS: dict;  REWARD_RULE: str      # W1-W4, as text copied into every env description
class Core:                                  # everything the env and a policy row share, on one plant, tracker and script
    def __init__(self, p, cfg: "R5Config", lg: legs.LegGap | None)
    def begin(self, clock, est, script: Script, head=None) -> None    # clock: () -> t_ref;  est: () -> dict(q, qd, gravity, gyro, foot_load)
    def actor_obs(self) -> np.ndarray        # float32 [actor_dim]
    def critic_obs(self, factors, push, floor) -> np.ndarray          # the env only
    def filter(self, action) -> np.ndarray   # clip, low-pass; shifts the three-action history
    def offsets(self, a_f) -> tuple[np.ndarray | None, float]         # (tk.residual or None, rate_factor)  (R5)
    def targets(self, a_f) -> tuple[np.ndarray, float]                # (q_target, rate_factor)             (R6)
    def substep(self) -> None                # per physics step: loads, flight, contact match, touchdown speed, servo mean squares
    def reward(self, clock0: float) -> tuple[float, dict]             # the env only; section 3.4
    def interference(self) -> tuple[float, bool]                      # (penalty in [0, 1], distance < 0)
class HeadScript:                            # the -X head command: events (kind, t0, duration, hold, target); value(t) -> (yaw, jaw, yaw_goal, jaw_goal)

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
def make(mode: str, cfg: R5Config) -> R5Env
def env_configs(mode, body, set_name, n_envs=8, overrides=None) -> list[R5Config]     # env i's floor by D13
def throughput(mode, cfg, env_steps=2000, sigma=0.1, seed=0) -> dict

# vecpool.py
class PoolVecEnv(VecEnv):  def __init__(self, env_fns: list, workers: int, start_method="spawn")   # workers 0: in-process

# s2rows.py   (registers "R5" and "R6" with panels.register_rows at import)
class S2Policy(panels.Policy)                # begin(tk, built, seed) / act(tk, k) through r5core.Core; load() via s2load.load_checkpoint
class R5Row(panels.TrackerRow)               # TrackerRow(name, "R2", gains=None, terms=(), ankle_cap=(), policy=S2Policy(...))
class R6Row(panels.Row)                      # its own episode: r5env's R6 tracker under control.run with the acting hook
def configure(policies: dict) -> None        # row name -> {"job_dir", "steps"}; default from $STAGE2_POLICIES (a JSON file)
def equivalence_check(job_dir, steps, body, set_name, condition, seed, k) -> dict   # a row episode against the env, digest per step

# lookread.py   (numpy, mujoco for FK; imports metrics for the contact rule only)
BANDS = motion.BANDS                         # bob_rel 0.30, head_world_pitch_rms_deg 1.5, tail_phase_cycles 0.1, toe_off_rel 0.70
def series(model, rec) -> dict               # t50, hip_mid_z, head_pitch_world, tail, foot_pitch [.., 2] from rec.poses by FK; contact from rec
def reading(model, rec, ta: float, tb: float) -> dict     # bob_pp_m, head_world_pitch_rms_deg, tail {phase_cycles, amp_rad}, toe_off_rad, counts
def compare(own: dict, ref: dict) -> dict    # ratios, differences, terms {bob, head, tail, toe_off}, ok
def episode_reading(b, out, rec, seed) -> dict | None     # the hook bar.judge calls: window by the item (U2), reference by playback (cached per condition)
def sources() -> dict

# monitor2.py
CONDITIONS = ("walk/nominal", "walk/v0.4", "turn/q90_left", "stop/two_step", "nudge/forward", "nudge/left"); SEEDS = (6001, 6002, 6003)
def run(job_dir, steps, body, set_name, out, cut_s=None, procs=1) -> dict     # the checkpoint's 18 episodes; record monitor_<n>.json
def baseline(body, set_name, out, cut_s=None) -> dict                         # R0@10ms on the same 18; record r0l.json
def gain(now: dict, earlier: list[dict], baseline: dict) -> dict              # M3
def g8b(mon: dict) -> dict;  def s2_10(final: dict, baseline: dict) -> dict   # M4; S2-10's record
# CLI: python -m designlab.stage2.monitor2 {run|baseline|s2-10} --job DIR --steps N --body B --set S --out DIR [--cut-s X]

# train2.py
def train(mode, body, seed, set_name="V6", out=None, job=None, steps=None, resume=False, workers=None,
          checkpoint_every=250_000, monitor_every=2_500_000, lr=None, config=None, ppo=None, gates=True, clips=False,
          allow_version_change=False, allow_source_change=False, parent=None, tiny=False, log=print) -> dict
def probe_lr(bodies, set_name, out, rates=(1e-4, 3e-5, 1e-5), updates=6, workers=None) -> dict     # O3
def probe_sigma(bodies, set_name, out, sigmas=(0.0, 0.025, 0.05, 0.1, 0.2)) -> dict                # O2 (reported)
def filter_choice(bodies, set_name, out, cutoffs=(10.0, 20.0)) -> dict                             # O4
def sources() -> dict                        # sha256 over the Stage 2 modules the job imports + plant.sources_sha256() + panels.panel_sources()
# CLI: python -m designlab.stage2.train2 run --mode R5|R6 --body B --seed N [--set V6] --out ROOT [--job NAME] [--steps N]
#        [--workers W] [--resume] [--allow-version-change] [--allow-source-change] [--from-checkpoint JOBDIR:STEPS]
#        [--lr X] [--clips] [--no-gates] [--tiny]
#      python -m designlab.stage2.train2 {probe-lr|probe-sigma|filter|preflight} ...

# gates2.py
def g8a(body, set_name, conditions=G8A_CONDITIONS, seeds=(3042, 3043), delays=(0, 5), cut_s=None) -> dict
def g9(body, set_name, stage1b_run=None, bar_run=None) -> dict
def rates(body, set_name, floors=("hard", "robot_floor_carpet"), workers=(1, 2, 3, 4), steps=16_384) -> dict
def kill_resume(body, set_name, out, workers=2) -> dict;  def gl_start(out) -> dict
# CLI: python -m designlab.stage2.gates2 {g8a|g9|rates|kill-resume|gl-start} --body B --set S [--record PATH] ...

# r2lfit.py
def rotation(seed: int, generations: int = 30) -> dict       # Q2's table: per generation the episode seed and, per kind, k and direction
class Evaluator;  def _evaluator(...)                         # tune.Pool's factory (top level)
def fit(body, set_name, seed=42, out=None, workers=1, resume=False, budget_core_h=3.2, max_generations=30, log=print) -> dict
def select(body, set_name, seed=42, out=None, workers=1, budget_core_h=0.8) -> dict     # Q4
def r2l_row(body, run_dir, name) -> panels.Row                # "R2L/<seed>"; gains from <out>/<set>/<body>/r2lg5.json
# CLI: python -m designlab.stage2.r2lfit {rotation|fit|select} --body B --set S [--seed 42] --out stage2/tb/r2l [--workers 2] [--resume]

# bundle.py
def build(bundle_id, bodies, sets=("V6",), out=None, part_mb=20) -> dict      # parts + MANIFEST.json
def verify(manifest, root, parts_dir=None) -> dict;  def unpack(manifest, parts_dir, root) -> dict
def dry_run(manifest, parts_dir, ref, repo=None, scratch=None) -> dict        # fresh clone + parts -> hashes, envs, G8a; DRYRUN.json
# CLI: python -m designlab.stage2.bundle {build|verify|unpack|dry-run} ...

# nb.py   (standard library only at import; every step is also a CLI subcommand)
def session(run_id, drive, local, queue, repo_ref, commit, resume, params) -> dict     # SESSION_<k>.json; the run-directory line
def inputs(bundle_id, drive, source="drive") -> dict                                   # line 2
def preflight(queue, set_name) -> dict                                                 # archive preflight, GL start, G8a per body: line 3
def launch(run_id, drive, local, queue, set_name, budget, workers, tiny=False) -> dict # the queue; STATUS.json; line 4 per job
                                                                                       # (drive == local: no copies; the local runs)
def smoke(run_id, drive, local) -> dict;  def finish(run_id, drive) -> dict
def local_run(notebook, drive_dir, params, kill_after_checkpoint=True) -> dict         # the notebook's cells, here, with a kill and a resume
```

### 2.3 What Stage 1's env and tracker cannot do today, and how each is got without editing a key module

| Map 05's item | How, without an edit | Where |
|---|---|---|
| Mid-task starts: `Tracker.reset` sets `tau_ref = 0` and starts from the track's first frame (`control.py:1292-1303`) | `R5Tracker.reset_at` (D6; scratch C3) | `r5env.py` |
| A track keeps the full state of its first frame only (`control.py:963`, `1021`) | the library keeps each file's `q`, `qd` (they are in every reference and transition file: `q [n, nq]`, `qd [n, nv]`) | `r5lib.py` |
| No delay wrapper covers `Plant.set_state` | the env refills the queue `speedfit.servo_delay` yields (D6) | `r5env.py` |
| 14-joint literals (`env.py:138-142`, `284-286`, `361-364`, `499-500`) | a new env with a layout per joint list (D1, D7); `env.py` untouched, its refusal and the test that pins it stay | `r5core.py`, `r5env.py` |
| The command is the reference's (`env.py:613-620`) | `Script.command` (D3) | `r5lib.py` |
| `TASKS` literals, `omega` from `refs/nominal.npz` at 0.03 H (`env.py:78-99`, `380`) | the task table over a set; `control.omega0(..., crouch_frac=the set's)` as the bar | `r5lib.py` |
| A stop event only for names in `transitions.TRANSITIONS` (`control.py:994`) | `setio.build_track` appends round A's command events | round A |
| Reward weights are module constants (`env.py:146-148`) | `R5Config.reward` | `r5env.py` |
| Joint damping and friction not randomised; `Tracker.damp` and `.fric` copied at construction (`control.py:1199-1200`) | the env scales `m.dof_damping` and `m.dof_frictionloss` per episode and restores them; the trackers are built on the nominal model, so the feedforward keeps the nominal values: a deliberate model error, as the mass (`env.MASS_RULE`) | `r5env.py` |
| `PUSH_BOUND_M_S.get(body, 0.5)` gives 0.5 m/s to unlisted bodies (`env.py:465`) | `env.PUSH_BOUND_M_S[body]` (the body's own envelope screen; a `KeyError` stops the build) | `r5env.py` |
| Modes, run directories, G6 and monitor hard-coded in `train.py` | a new trainer (D2) | `train2.py` |
| No `MUJOCO_GL` in `train.py` | assigned at the top of every Track B entry point (2.1) | |
| The closure schedule, plain `PPO.load` (`policy.py:144-151`; `train.py:497`; `policy_rows.py:601`) | schedule objects by keyword; one loader (D14) | `schedules.py`, `s2load.py` |
| Versions refused, no flag (`train.py:457-460`); `check_versions` reads torch another way (`policy_rows.py:412-419`) | `s2load.versions` and `comparable` on both sides; `--allow-version-change` (K4) | `s2load.py`, `train2.py` |
| `policy_rows` built around R3/R4 and env.py's widths | new rows on `panels.TrackerRow(policy=)` and `panels.Row` (the panels' own extension points, `panels.py:751-801`, `865-871`) | `s2rows.py` |
| `speedfit.Evaluator` hard-wired to five stock episodes (`speedfit.py:434-446`) | a new evaluator on the bar's conditions (D21) | `r2lfit.py` |
| `tools/memguard.sh` protects `designlab.stage1 train` | run it with `PROTECT="designlab.stage2.train2"` (its own variable) | run plan |

**No edit to a module in a Stage 1, panels, tuner or speed-fit key is needed.** One edit to a round-A module that is
in the bar's key is needed, and nothing else outside this round's own files:

| File | Change | Why the default is identical | When |
|---|---|---|---|
| `designlab/stage2/bar.py` (round A, WP B) | (1) `bar.register_reading(name, fn, sources)`: `judge` calls `fn(b, out, rec, seed)` for an episode of a walk or turn item of a set whose `set.json` holds `look_bands`; the result goes to `measures.<name>`; when it holds `ok`, `success_look = success and ok` fills `counts_look` (round A reserved the block, D10); for nudge and stop items of such a set `success_look = success`. (2) `BarBuilt.recorder()` gives `metrics.Recorder(legs_every=tune.LEGS_EVERY, ...)` for every item of such a set (50 Hz poses; today only turns). (3) the trace gains `poses50` (float32) when the recorder holds poses. (4) `bar_sources()` adds a registered reading's sources for such a set only | a set without `look_bands` (V0, F25F9, F25F9cl) takes none of the four paths: recorder, judge output, trace keys and source list are as before; only `bar.py`'s own bytes differ | before state A if the lead can still add it to WP B's brief (then no bar key ever moves); otherwise at state B (D18), before any look-set row |
| `designlab/stage2/sets.py` (look work package) | `set.json` of `V6` and `V6noF5` holds `look_bands` (the four numbers of `motion.BANDS`) and the entries of D5 | new sets only | with the look sets |
| `tests/test_stage2_x.py:629-637` | none (the refusal it pins stays true) | | |

Also relied on, as round A specifies them (no change asked): `bar.reference_recorder(p, track, seconds)` filling
loads from the reference's support and poses at 50 Hz; `bar.rows` wrapping any registered row in `bar.Delayed`;
`bar.BarPlanner.resolve(cond)["plan"]`; `setio.build_track`, `setio.read_inputs`, `setio.add_events`;
`s2io.write_json`, `sha256_json`, `Lock`, `built_by`.

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
| `prev_actions` | 42:87 | 46:97 | the last three clipped, unfiltered actions, newest first (3 x (nj + 1)); zeros at a reset |
| `head_command` | - | 97:101 | `[yaw now, jaw now, yaw goal, jaw goal] / 0.6` of the head script; zeros with the head neutral |
| **actor** | **87** | **101** | deployable inputs only |
| `root_vel` (3), `height` (1), `roll_pitch` (2), `dcm_error` (2) | 87:95 | 101:109 | as `env.py:652-654` |
| `joint_error` | 95:109 | 109:125 | true joints less the reference's (with the head overlay) |
| `heading_error` (1), `friction` (1) | 109:111 | 125:127 | as Stage 1 |
| `floor` (2) | 111:113 | 127:129 | one-hot hard, robot_floor_carpet |
| `push` (3) | 113:116 | 129:132 | as Stage 1 |
| `plant_factors` (5) | 116:121 | 132:137 | mass, kp, `k dt / 0.024`, joint damping factor, joint friction factor |
| `foot_error` (6) | 121:127 | 137:143 | `r_f - r_ref_f` of section 3.4, both feet |
| `ref_contact` (2) | 127:129 | 143:145 | the reference's support at `c - 0.024 rate` |
| **observation** | **129** | **145** | |
| **action** | **15** | **17** | `a[:nj]` joints in plant order, `a[nj]` the clock rate; `Box(-1, 1)` |

`PrivilegedCriticPolicy` takes any widths (`policy.py:69-141`); `actor_dim` goes to `policy.model_kwargs`.

### 3.2 `R5Config`, reset and step

```python
@dataclass(frozen=True)
class R5Config:
    body: str;  set: str = "V6";  mode: str = "R5";  floor: str = "hard"
    rate: float = 1.05;  max_seconds: float = 30.0
    tasks: tuple = ();  task_weights: tuple | None = None       # (): every available task of r5lib.TASKS, its weights
    mid_start: float = 0.5                                      # T3
    randomize: bool = True
    friction: tuple = (0.4, 1.2);  mass: tuple = (0.9, 1.1);  kp: tuple = (0.6, 1.2)
    latency_steps: tuple = (0, 12);  noise_scale: tuple = (0.5, 1.5)
    joint_damping: tuple = (0.5, 1.5);  joint_friction: tuple = (0.5, 1.5)
    push_prob: float = 0.5;  push_walking: tuple = (0.0, 0.5);  push_standing: tuple = (0.0, 1.0);  push_s: float = 0.1
    head_events: bool = True;  gaze_prob: float = 0.5;  jaw_prob: float = 0.25      # -X plants only
    start_joint_noise: float = 0.01;  start_root_velocity_noise: float = 0.02       # tune's constants (asserted equal)
    action_leg_rad: float = 0.35;  action_other_rad: float = 0.15;  action_rate: float = 0.25;  filter_hz: float = 10.0
    reward: tuple = REWARD_WEIGHTS_ITEMS                         # ((name, weight), ...): W2
    interference: str = "terminate"                              # "terminate" | "reward" | "off"
    set_freeze: str | None = None                                # the freeze's manifest sha256; the trainer fills and compares it
```

`describe()` holds no path: the set is named and pinned by its freeze, and `setio.set_dir` resolves it under
`s2io.STUDY`, so a run's settings are the same on local disk, on Colab and on Drive (map 05 hazard 4).

**Construction** (mirrors `env.py:352-406`): `plant.build(body, floor)`; `nominal_model = deepcopy(m)`;
`env.setconst_changes(m)` must be empty; the nominal weight; `lib = r5lib.Library(p, body, set, rate)`; the
nominal copies of `geom_friction`, `actuator_gainprm[:, 0]`, `actuator_biasprm[:, 1]`, `dof_damping`,
`dof_frictionloss`; one `R5Tracker(p, script.track, "R2", gains=None, terms=(), ankle_cap=(), rate=rate,
model=nominal_model)` per script, built lazily while the model is nominal; `legs.LegGap(body)`;
`core = r5core.Core(p, cfg, lg)`; spaces from `layout(p.joints)`.

**`reset(seed, options)`**, in this order (the Stage 1 line each step mirrors):

1. `super().reset(seed=seed)`; a seeded reset restarts the episode stream (`env.py:471-473`).
2. The script: `options["plan"]` (a bar condition's, with `options["seconds"]`), else `options["task"]`, else a
   draw by the task weights (`:476-477`).
3. The start: `options["start"]`, else "mid" with probability `mid_start` when the script has a `mid_span()`, else
   "stance"; for "mid" the reference time is uniform on the span, then snapped to a frame.
4. `seconds = min((t_end - t_start) / rate, max_seconds)`; `n_steps = round(seconds / dt_env)` (`:478-481`).
5. The factors, drawn in this order when `randomize`: friction, mass, kp, latency_steps (integers, inclusive),
   noise_scale (Stage 1's five and order, `:437-445`), joint_damping, joint_friction; `options["factors"]`
   overrides (`:482-485`). Applied: Stage 1's restore and scaling (`:447-458`), then
   `m.dof_damping[p.dofs] = damping0 * f`, `m.dof_frictionloss[p.dofs] = friction0 * f` (restored to the nominal
   arrays first; `mj_setConst` does not read them).
6. The push (`options["push"]`, else J3's draw) and, on an `-X` plant, the head script (`options["head"]`, else
   J4's draw) (`:487`).
7. The episode seed and its start and noise seeds (`:488-497`: `1,000,000 x reset seed + k`;
   `SeedSequence([ep, 0])`, `([ep, 1])`), unless `options["seeds"]` gives them (the gates and the monitor do).
8. Counters, the three-action history, the filter state and the reward's accumulators to zero (`:498-501`).
9. Leave the previous episode's delay (`:502`).
10. The start (`:503`): "stance": `tk.reset("stance", joint_noise=cfg.start_joint_noise, seed=seeds["start"],
    root_velocity_noise=cfg.start_root_velocity_noise, noise_seed=seeds["noise"])` after `tk.noise =
    SensorNoise(scale=noise_scale)`, `tk.residual, tk.rate_factor = None, 1.0` (`:714-719`). "mid":
    `tk.reset_at(t, qpos, qvel, seeds["noise"])` with `qpos` the file's frame plus uniform joint noise and `qvel`
    the frame's rates times `rate` plus uniform root-velocity noise, both from `default_rng(seeds["start"])`.
11. The delay (`:504-505`): `speedfit.servo_delay(p, k)` entered when `k > 0`; for a "mid" start its queue is
    refilled (D6).
12. The baselines for speed, yaw and the reference's yaw (`:506-509`); `core.begin(...)`; the observation.

**`step(action)`**, in this order (`env.py:530-564`):

1. `a = clip(action)`; `a_f += alpha (a - a_f)`, `alpha = 1 - exp(-2 pi filter_hz dt_env)` (`:531-532`).
2. R5: `off = a_f[:nj] * scale + head overlay`; `tk.residual = off if off.any() else None`;
   `tk.rate_factor = 1.0 + 0.25 * float(a_f[nj])` (`:723-725`). A zero action with the head neutral leaves `None`
   and exactly `1.0`.
3. Ten times: the push (`control._apply_push`, `:542-543`); `tk.step()` (`:544`); the step hook (`:545-546`: the
   gates' digests); `core.substep()` (loads, flight, contacts as `:547-552`; contact match, touchdown speed, the
   servo mean squares).
4. The fall and any MuJoCo warning, read once (`:553-555`); the interference reading (D10).
5. Counters, the contact history (`:556-557`); the reward (`:558`); the action history shifts (`:559`).
6. `terminated` on a fall, or on interference at two env-step ends in a row when `cfg.interference` is
   "terminate" and the episode did not switch it off (`options["terminate_on_interference"] = False`: the gates);
   `truncated` at `n_steps` (`:560-561`).
7. `info` (Stage 1's keys plus `start`, `t_ref`, `interference`, `reward_terms`), the observation (`:562-564`).

R6 differs in steps 10 and 2-3 only, as `env.PolicyOnlyEnv` (`env.py:758-783`): its own `NoisySensors` and
`Estimator`, `p.command(q_target)`, its own clock capped at the track's end; a "mid" start sets the plant, resets
the estimator from the known state and fills the queue with the reference's joint targets.

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

Built once per plan from the set's files (the same files `setio.read_inputs` hashed), one table per track segment:

- **Command** `(v, w)` in the references' own units; the observation multiplies by `rate`. A hold: (0, 0). A
  periodic walk or turn in place: the entry's (`speed_m_s`, `yaw_rate_rad_s`). A start, a change, a start into the
  turn in place: gait b's command from the segment's first frame. A stop: (0, 0) from its first frame (the command
  instant of a causal stop). A finite turn of angle A at rate w0 from walk a: `v` stays a's; `w = sign x |w0|` for
  reference times in `[t0, t0 + A / |w0|)` and 0 after (a rectangular pulse whose integral is A: 2.0 s of reference
  time for the quick turn, 1.905 s of simulated time at the commanded clock, at 1.05 x pi/4 rad/s). Nothing before
  a segment's first frame depends on that segment.
- **Phase**: a periodic segment reads the file's `phase` at the local time, modulo 1, when the file holds it (a
  K-stride window), else `(tl / stride_s) mod 1`, `stride_s` the entry's `period_s`; a finite segment interpolates
  the file's `phase` over its `t`, modulo 1; a hold reads 0. `stride_s(t)`, `swing_s(t) = (1 - duty) x stride_s`
  and `walking(t)` (not the track's `stand` flag) come from the same tables.
- **Start states**: the file's `q` and `qd` per frame (root included); `start_state(t)` returns the frame at or
  before `t`. `mid_span()` is `[first non-hold segment's t0 + 0.2 s, t_end - 2.0 s x rate]`.
- **Reward references**, by forward kinematics of each file frame on the plant's nominal model at library build:
  `foot_rel [n, 2, 3]` (each foot body's position less the root's, rotated into the root's heading) and
  `head_rp [n, 2]` (the head body's roll and pitch in the world about home), interpolated linearly; `support`.

### 3.4 Reward (`r5core.Core.reward`; rules W1-W4)

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
| `clearance` | 0.02 | a foot in the air (load under 2 %) between 25 % and 75 % of the planned swing time since **its own** lift-off, lower than 0.04 H: the shortfall over 0.04 H, the larger of the two feet | Stage 1's term on the bar's own-swing reading **[differs: S1]** |
| `action_rate` | 0.03 | `min(1, mean_i (a_i - a_prev_i)^2 / 0.25)` on the clipped actions | new |
| `touchdown` | 0.02 | in a step in which a foot's contact begins: `min(1, max(0, v_down - 0.15 m/s) / 0.30 m/s)`, `v_down` the foot body's downward speed one physics step before (the plan's "impact": quiet feet) | new |
| `servo` | 0.05 | per drive that is not a leg drive (`~plant.leg_drives(p.drives)`: neck, tail, neck yaw, jaw): an exponential mean of the squared actuator force, time constant 5 s; `min(1, max_d max(0, sqrt(mean_d) / sustained_d - 1) / 0.5)` ("neck and tail torque over their limit") | new |
| `interference` | 0.10 | `clip((3 mm - g) / 3 mm, 0, 1)`, `g` the smallest left-to-right leg distance (D10) | new |

The weights are `R5Config.reward`, written into every env description, `run.json` and the settings a resume
compares. G8b's fix does not touch them (M4).

### 3.5 The look reading (`lookread.py`; rules U1-U5)

- **Series.** From the recorder's 50 Hz poses, by `mj_kinematics` on the plant's model: `hip_mid_z` (the mean
  height of the two hip joints' anchors, `look.py:600-624`), `head_pitch_world` (as `look.arrays`, `look.py:546`,
  `:646`), each foot's pitch about its own lateral axis (toe down positive, the sign of `foot_pitch_solved`), and
  the tail-yaw angle; contact and its lift-offs and touchdowns from the recorder's loads by the 2 % rule with
  20 ms debounce (`readings.contact_flags`).
- **Window** (U2): a walk item's judged last 10 s; a turn item's 3 s of walking after `t_c + at_s`; nudge and
  stop items are not read.
- **Four numbers** over the window: (1) *bob*: the mean, over the steps that lie wholly inside it (a step runs
  from one touchdown of either foot to the next), of max less min of `hip_mid_z`; (2) *head*: the RMS of
  `head_pitch_world` about its mean, in degrees; (3) *tail phase*: the least-squares fit `tail = A sin(2 pi phi +
  psi) + c` (`look._fit_phase`'s form) with `phi` the robot's own phase, 0 at each of its left lift-offs and linear
  in time between them; `psi / 2 pi` in cycles and `A`; (4) *toe-off*: the mean, over the lift-offs inside the
  window, of the lifting foot's pitch at the lift-off instant (interpolated between poses).
- **The reference's numbers**: the same function on `bar.reference_recorder(p, track, seconds)` of the episode's
  own track, over the reference times the window covers at the nominal commanded clock; cached per condition.
- **Terms** (U3): `bob`: `|own / ref - 1| <= 0.30`; `head`: `own <= 1.5 deg`; `tail`: the wrapped difference of
  the two phases at most 0.1 cycle, not read (passes, labelled) when the reference's amplitude is under 0.02 rad;
  `toe_off`: `own >= 0.70 x ref`, not read when the reference's is under 0.02 rad (a set without F5). A window
  with fewer than 4 steps or 4 lift-offs is an error of the episode, not a failed term (round A's C8 kind).
- **Readings** (U4): `look_ok` is the conjunction; `success_look = success and look_ok` on walk and turn items,
  `= success` on nudge and stop items, with `reading_1: "defined for walks and turns"` in those records. Printed
  beside, never a verdict: `success_but_A1 and look_ok`.

### 3.6 The trainer (`train2.py`)

**Run directory** (the same layout locally and on Drive; `<root>` is `--out`):

```
<root>/jobs/<job>/run.json                    job = "<mode>-<body>-s<seed>", e.g. R5-75-M-H-10-X-s42; a fine-tune names its own
<root>/jobs/<job>/updates.jsonl               one line per PPO update (lesson 16)
<root>/jobs/<job>/progress.json               rewritten at every rollout start (the launcher reads it)
<root>/jobs/<job>/gates/g8a.json, rate.json
<root>/jobs/<job>/checkpoints/model_<n>.zip, vecnormalize_<n>.pkl, fingerprint_<n>.npz, checkpoint_<n>.json
<root>/jobs/<job>/monitor/r0l.json, monitor_<n>.json, trace_<n>.npz, clip_<n>.mp4, clip_<n>.json
```

**Start.** Settings = `{mode, body, seed, set, set_freeze, inputs (file -> sha256, the library's), n_envs 8,
ppo (design 2's values as used, with n_steps), lr {start, end}, log_std_init, checkpoint_every, monitor_every,
budget, env_configs [8 x R5Config.describe()], layout, parent}`; rule texts are stored under `rules`, outside the
settings (map 05 hazard 3). A new run: G8a for the body and set on `walk/v0.4`, seed 3042, at k = 0 and 5
(`gates2.g8a`, about 15 s; "G8a fails: no run": the job exits with code 3 and writes `stopped`); the vec env
(D13); `VecNormalize(norm_obs, norm_reward, clip_obs 10, clip_reward 50, gamma 0.99)`; `PPO(env, seed,
device="cpu", **model_kwargs(actor_dim, log_std_init=ln 0.1), learning_rate=LinearSchedule(lr, 0.3 lr),
clip_range=0.2, n_steps=2048)`; `torch.set_num_threads(1)`; the monitor's R0L baseline (once).

**Resume** refuses, in this order, each with a message naming the flag if one exists: no `run.json`; a stopped run;
other settings; other inputs (a file's sha256 or the freeze changed: no flag); other sources (`train2.sources()`
against the first session's; `--allow-source-change`); other library versions (`s2load.comparable` against the
last session's; `--allow-version-change`); no intact checkpoint (`s2load.newest_intact`: the skipped ones and their
reasons go to the session record and the log); a fingerprint that is not reproduced within 1e-5 (always checked;
with a version change it is S2-13's condition). Then `load_checkpoint(..., env=vecnorm venv, lr=(start, end))`.
A flagged session records both version sets or both source hashes and `version_change` / `source_change: true`,
and every later record of the run carries the flag.

**The callback, at a rollout start with n steps done:**

1. If n has reached the next multiple of `checkpoint_every`: `s2load.save_checkpoint` (four files), then delete
   what `s2load.keep` no longer lists (K2), then `run.json`.
2. If n has reached the next multiple of `monitor_every`: `monitor2.run` on that checkpoint in a subprocess; append
   its record; read the gain (M3); at the first check at or past 5M read G8b (M4) and, if it fails, apply the fix
   in place: `venv.env_method("set_task_weights", w2)`, `doc["fix"] = {rule, at_steps, weights_before, weights,
   tasks, monitor}`, the settings' env configs replaced by the fixed ones with the earlier ones kept under the fix
   (a later resume compares against the fixed ones); at the first check at or past the fix + 5M read G8b again, a
   second failure sets `stopped`; from 20M apply M5's stop rule. With `--clips`: `clip2` in a subprocess with
   `MUJOCO_GL=osmesa`, a 300 s limit, its failure recorded and ignored.
3. If step 1 saved, or this is a resumed session's first rollout: the boundary (D16, K1).

`_on_step`: after the first rollout, `gates/rate.json` (`first_rollout`: env steps per wall second, CPU seconds of
the trainer and of the workers, vCPUs, workers); after the third rollout `first_three` (two updates included:
S2-8's rate). After `learn` returns: a final checkpoint if n has none, a final monitor (S2-10's reading when the
budget is under 2.5M), `status`.

**`updates.jsonl`** (one line per update): `{update, steps, minibatch_steps_run, minibatch_steps_possible,
approx_kl, clip_fraction, explained_variance, value_loss, std_mean, lr, ep_len_mean, ep_rew_mean, wall_s}`;
`minibatch_steps_run` from the wrapped `optimizer.step` [scratch C4].

**`run.json`** (`schema: "s2-run-1"`): `job, mode, body, head ("current" | "X"), seed, set, settings, rules
{...texts}, layout, sessions [{index, started_at, ended_at, commit, dirty, versions, sources, workers, vcpus,
torch_threads, resumed_from, skipped_checkpoints [{steps, reason}], fingerprint_max_abs_diff, version_change,
source_change, boundary_seeds}], checkpoints [n], kept [n], gates {g8a, rate, g8b, g8b_after_fix}, monitor
[{at_steps, record, success_look, success, success_but_A1, mean_survival_s, gain}], baseline {record, ...}, fix,
stopped {at_steps, reason}, parent {job, steps, files {sha256}, fingerprint_max_abs_diff}, updates_file,
status ("running" | "done" | "partial" | "stopped_g8a" | "stopped_g8b" | "stopped_no_gain"), budget,
num_timesteps`.

**`checkpoint_<n>.json`**: `{schema: "s2-ckpt-1", job, steps, files {model | vecnormalize | fingerprint: {name,
bytes, sha256}}, versions, layout {actor_dim, obs_dim, action_dim, nj}, fingerprint {rows 256, seed
[20261115, obs_dim, actor_dim], obs_sha256, tol 1e-5}, saved_at}`. `fingerprint_<n>.npz`: `obs [256, obs_dim]
float32`, `actions [256, action_dim] float32`. The observation table: per row, each actor block drawn from a
fixed distribution of its kind (angles 0.3 z, rates 2 z, gravity (0, 0, -1) + 0.05 z renormalised, gyro 0.5 z,
loads uniform on 0-1, phase a point of the unit circle, support bits, commands uniform on their ranges, previous
actions 0.1 z, the head command uniform), the critic's columns zero (the actor does not read them).

**Run level.** `train2 run` writes `run.json`, `updates.jsonl` and `progress.json` for its job and exits 0 (done or
stopped by a rule), 3 (G8a failed) or another code (a failure). The run-level files (QUEUE.json, SESSION files,
STATUS.json, DONE.json or HALT.json, the logs) are `nb launch`'s, which also runs here with a local directory as
its "Drive" (S2-4 and the local 60 cm run go through it, so every run has them). `--tiny` (tests, the notebook's
local run, the smoke's kill and resume): `n_steps` 64, 4,096 steps, a checkpoint every 1,024, a check every 2,048
with one seed and episodes cut to 3 s, G8a cut to 2 s; `tiny: true` is in the settings, so a tiny run can never be
resumed as a real one.

**A fine-tune** (`--from-checkpoint JOBDIR:STEPS --job NAME --steps 5000000`): a new job directory; the parent's
checkpoint through `load_checkpoint` (same layout required); its normalisation statistics loaded and updated
further; `log_std` as saved; a constant learning rate at the parent's schedule value at the parent's step;
`parent` recorded. The new job has its own settings, boundaries and monitor.

### 3.7 G8a and G9 (`gates2.py`)

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
| length | `round(seconds / dt)` physics steps, or to the fall, read after every physics step | `round(seconds / dt_env)` env steps; falls read at env-step ends; interference termination off, its would-be step recorded |
| digest | sha256 of `qpos, qvel, ctrl, sensordata` after every physics step (`env._state_digest`), through a wrapper put on the built condition's `recorder` for the episode | the same, through the env's step hook |

An episode is *identical* when no digest differs over the steps the row ran and the env ends where the row did
(`env.ZERO_RESIDUAL_RULE`'s reading: if the row fell, the env terminates in the env step that holds that physics
step; if not, it runs as many steps and does not terminate). **G8a passes for a body and set when every episode
is identical** over: the eight gated mu 0.9 conditions the set has references for, `walk/v0.4_mu0.54` and
`walk/nominal`; panel seeds 3042 and 3043; k = 0 and k = 5 (40 episodes, rule Z1). Record
`stage2/tb/gates/<set>/<body>/g8a.json`: `{rule, body, set, set_freeze, episodes [{condition, seed, k,
reference_steps, compared_steps, first_difference, reference_fell, fall_env_step, ends_alike, identical,
interference_first_env_step}], identical, of, not_run [...], sources {panels, bar, r5}, key_context, versions,
blas_threads, commit, dirty, at}`. The trainer's own G8a (one condition, one seed, both delays) uses the same
function and is what the notebook prints.

**G9** (S2-11 as ruled; one record per body, `stage2/tb/gates/<set>/<body>/g9.json`, rule Z2):
(a) *references*: `bar refcheck` (every frozen reference played kinematically at the commanded clock passes its own
item's readings; the quick turn also reaches 80 degrees at the slowest clock), and for each walk the look reading
of its playback against the look ladder's own numbers for the same rung (`stage2/look/metrics.json`): bob within
5 %, head pitch RMS within 0.1 deg, tail phase within 0.01 cycle, toe-off within 0.02 rad; (b) *R0L*: every
episode of the body's `R0@10ms` bar records of the look set holds each reading of its item and, for walk and turn
items that reached their window, a finite look reading; (c) *negative controls on a playback*, so that each band
is shown to bite: the pelvis height's oscillation scaled by 0.5 fails `bob`; a 3 degree sinusoid added to the neck
pitch fails `head`; the tail delayed by 0.2 cycle fails `tail`; the foot pitch zeroed fails `toe_off`; the
unchanged playback passes all four with ratio 1 and difference 0; (d) for `60-N-T-10`: round A's `bar reread`
record of Stage 1b's 10 ms turn episodes (each trace identical to its committed sha256), cited by path and
sha256; for `75-M-H-10`: that every 10 ms bar record holds both toe-clearance readings for every episode.

### 3.8 R2L (`r2lfit.py`; rules Q1-Q6)

- **Mix**: five bar episodes per evaluation on the fit's set (mu 0.9), in slots j = 0-4: `walk/v0.4`,
  `turn/q90_<left|right>`, `stop/two_step`, `nudge/<direction>`, `walk/nominal`; each built by `bar.BarContext`
  and run as `bar.rows`' `R2` row runs it (`panels.TrackerRow` mode R2 with the candidate's gains) inside
  `speedfit.servo_delay`, judged by `bar.judge`.
- **Rotation** (S2-18; `rotation()` writes `rotation.json` before generation 0): generation g runs episode seed
  `tune.training_order(seed)[g]` (start and noise seeds `tune.derived_seed(ep, j, 0 | 1)`); slot j runs at delay
  `k = (g + j) mod 6` (0-10 ms: within a generation five different delays, and over any six generations every kind
  at every delay once); the turn's side is left for even g; the nudge's direction is
  `("forward", "left", "backward", "right")[g mod 4]`. Every member of a generation runs these same five episodes.
- **Objective** `stage2-r2l-v1`: `tune.score` over the five episodes (falls, then tracking and limit terms).
  Tracking terms: `tune.measure`'s over the item's judged window. Limit terms: `tune.measure`'s window limits
  without the three "at nominal" ones, with the toe clearance on the own-swing reading, plus the item's own, each
  graded on [0, 1]: walk: the shortfall of `along` over its target, `line` over 0.25 m; turn: the heading error
  at `at_s` and the largest in the hold, each over 20 degrees; stop: touchdowns beyond two over 2, the stop time
  over 1.5 strides; nudge: the recovery time over 3 s.
- **Search**: `tune.CMA`, population 12, sigma0 0.1 in the [0, 1] scaling, from the body's Stage 1 G5 selection
  (`speedfit.start_record`), seed 42; 30 generations or the fit's share of the budget, whichever comes first,
  counted as summed evaluation CPU seconds in `ledger.json` (Q5).
- **Selection** (Q4): candidates the start and the mean after each third of the generations judged; each runs the
  five kinds on validation seeds 2000-2019 at k = 5; ranked by `tune.selection_key` on those 100 episodes (fewer
  falls, then the lower mean S, P, J, then the earlier candidate). Reported, as far as the share reaches and in
  this order: the selected candidate and the start at k = 0, then at k = 2 and 3.
- **Rows**: `R2L/42` registered with `panels.register_rows("R2L", r2l_row)`: a tracker row in R2 mode with
  `r2lg5.json`'s selected gains for the bar run's set and body, refused unless judged; `bar.rows` wraps it for
  `@<ms>ms`. `sources()` is `r2lfit.py`'s sha256; the gains file's path and sha256 are in `describe()`.
- **Records** under `stage2/tb/r2l/<set>/<body>/`: `start.json`, `r2lfit/42/{rotation.json, cma.json,
  evals.jsonl, ledger.json}`, `r2lg5.json`, `r2lg5_evals.jsonl` (resumable per generation and per validation
  evaluation, as `speedfit.RESUME_RULE` and `G5_RESUME_RULE`).

### 3.9 The notebook, its lines and the Drive layout

**Cells of `stage2/stage2_training.ipynb`** (metadata: no accelerator; every code cell parses as Python):

| # | Type | Content |
|---|---|---|
| 0 | md | What this is, the link to `stage2/tb/colab_steps.txt`, "nothing here trains in the notebook's own process" |
| 1 | code | **Parameters**: `REPO_REF` (`# @param`), `RUN_ID`, `QUEUE` (a list such as `["R5:75-M-H-10:42", "R6:75-M-H-10:42"]`), `BUNDLE_ID`, `BUNDLE_SOURCE = "drive"`, `RESUME = False`, `SMOKE = False`, `BUDGET_STEPS`, `WORKERS = 0` (0: one per vCPU, at most 8), `CLIPS = True`, `USE_GOOGLE_DRIVE = True`, `AUTO_DISCONNECT = True`. If the environment variable `S2_PARAMS` names a JSON file, its values replace these (the local run and the tests) |
| 2 | code | **Setup** (the scaffold's cell 3, adapted): Colab detection; no `MUJOCO_GL` for the kernel; `pip uninstall -y gym gym-notices shimmy`; the pins applied when `pip freeze` differs: `mujoco==3.10.0 numpy==2.5.4 scipy==1.18.1 gymnasium==1.4.0 stable-baselines3==2.9.2 cloudpickle==3.1.2 pillow imageio-ffmpeg` and `torch==2.15.0` from `https://download.pytorch.org/whl/cpu` (the study venv's versions, each requirement quoted); `apt-get install -y libosmesa6` (failure reported, not fatal); the Git block (clone without checkout, fetch `REPO_REF`, the two refusals, detached checkout); `print(f"Repository ref: {REPO_REF}; commit: {requested_commit}")`; **line 1** |
| 3 | code | **Storage and session**: Drive mount (the scaffold's lines); `DRIVE = /content/drive/MyDrive/mesozoic-labs/stage2` (or `S2_PARAMS`' local directory); `nb session`: SESSION_<k>.json, the run-directory line, the refusals of K5 |
| 4 | code | **Inputs**: `nb inputs`: the parts from `DRIVE/inputs/<BUNDLE_ID>/` copied to local disk, each part's sha256 against the committed manifest, unpacked at study paths, every file's sha256, `sets verify` per set and body; **line 2**; a failure halts |
| 5 | code | **Preflight**: `nb preflight`: the repository's archive-load preflight (its two lines), a policy built with `MUJOCO_GL` assigned (`train2 preflight`), G8a for each body of the queue; **line 3** per body; a failure halts ("G8a fails: no run") |
| 6 | code | **Smoke** (only when `SMOKE`): `nb smoke` (rules Y1-Y9): one line per item, `S2 SMOKE Y<i> <item>: <value>: OK` (or `FAILED`, or `REPORTED` for an item that is only recorded), and `smoke/SMOKE.json`; then cell 8's finish; no training job |
| 7 | code | **Launch**: `nb launch`: the queue, one job at a time on all workers; **line 4** per job; STATUS.json and the Drive copies every 60 s; never raises (a failure becomes HALT.json) |
| 8 | code | **Finish**: `nb finish` (DONE.json or HALT.json, the last copy, the summary); then `disconnect_runtime(reason, in_colab=IN_COLAB, auto=AUTO_DISCONNECT, flush_drive=USE_GOOGLE_DRIVE)` inside `try ... finally` |

**The four lines, exact text** (N1; each starts at column 0 and is one line):

```
S2 LINE 1/4 commit: <40 hex> (REPO_REF <ref>; checkout clean: yes)
S2 LINE 2/4 inputs: bundle <bundle_id>, manifest <first 12 hex>, <n> of <n> files match their sha256: OK
S2 LINE 3/4 G8a <body> set <set>: zero action = R0 at 0 ms and 10 ms, <n> of <n> episodes identical (<steps> physics steps): PASS
S2 LINE 4/4 first rate <job>: <r> env steps/s (8 envs, <W> workers, <c> vCPUs); <B>M steps in about <h> h
```

Failing forms end `: FAILED (<first path or reason>)` (line 2) and `: FAIL (<condition> seed <s> at <ms> ms,
first difference at step <i>)` (line 3); line 1 with a dirty checkout reads `checkout clean: no`. Lines 3 and 4
repeat per body and per job. After a resume line 4 is printed again for the session, and the session line reads
`Run directory: <path> (resuming session <k>: <job> from <n> steps; <m> jobs done)`; a new run reads
`Run directory: <path> (new run)`.

**Drive** (`MyDrive/mesozoic-labs/stage2/`; a local directory stands in for it here):

```
inputs/<bundle_id>/MANIFEST.json, <part>.tar ...          uploaded by the lead
runs/<run_id>/QUEUE.json                                  the queue, budget and bundle as first given (a resume must match)
runs/<run_id>/SESSION_<k>.json
runs/<run_id>/STATUS.json                                 rewritten every 60 s: its modified time is the run's liveness
runs/<run_id>/DONE.json | HALT.json
runs/<run_id>/logs/launcher.s<k>.log, <job>.s<k>.log
runs/<run_id>/jobs/<job>/...                              the trainer's run directory (3.6): run.json, updates.jsonl, gates/,
                                                          kept checkpoints with their fingerprints and records, monitor/ with clips
runs/<run_id>/smoke/SMOKE.json, model_<n>.zip, ...        the day-one smoke's files
```

The trainer writes to local disk (`/content/s2runs/<run_id>/`). The launcher copies to Drive: a checkpoint's three
files each as `<name>.part`, then renamed, then `checkpoint_<n>.json` last; after a copy is verified by sha256 it
deletes on Drive what K2 no longer keeps; text records and logs every 60 s. On a resume the launcher copies the
job's directory back from Drive, and the trainer's own checks choose the checkpoint.

**`STATUS.json`** (`schema: "s2-status-1"`): `{run_id, session, written_at, started_at, commit, bundle_id, state
("running" | "done" | "halted"), runtime {vcpus, ram_gb, disk_free_gb}, lines [...the S2 lines printed so far],
queue [{job, state ("queued" | "running" | "done" | "stopped" | "failed"), steps, budget, env_steps_per_s, eta_h,
last_checkpoint, last_checkpoint_on_drive, last_monitor {at_steps, success_look, success, mean_survival_s, gain},
g8a, g8b, fix, minibatch_steps_last [run, possible], last_error}]}`.
**`SESSION_<k>.json`** (`"s2-session-1"`): `{run_id, session, started_at, resume, repo {ref, commit, dirty},
runtime {in_colab, vcpus, affinity, cpu_model, ram_gb, disk_free_gb, gpu}, versions, pins, pins_match,
python_minor_matches, bundle {id, manifest_sha256, files_ok, source}, queue, budget_steps, workers,
setup_seconds, params}`. **`DONE.json` / `HALT.json`**: `{run_id, session, at, reason, jobs {job: {state, steps,
last_checkpoint, stopped}}, lines}`.

### 3.10 The bundle (`bundle.py`)

`build` reads what to pack from committed records only: for each body its case's model files (`plant.model_files`:
`round2/refine/<case>/models/<id>.xml` and `.json`) with the hashes the records hold, and for each set and body
the files of the latest freeze manifest (`refs/*.npz`, `transitions/*.npz`; the look-format twins are not packed).
Parts: `models-<body>.tar`, `<set>-<body>-<i>.tar`. `MANIFEST.json` (committed at
`stage2/tb/bundle/<bundle_id>/`; `schema: "s2-bundle-1"`):

```
{bundle_id, made_at, commit, dirty, rule, tool_sha256, versions,
 bodies [...], models {body: {case, xml {path, sha256}, record {path, sha256}}},
 sets {set: {body: {freeze_index, manifest_sha256, files: n}}},
 files {<study-relative path>: {sha256, bytes, part}},
 parts {<name>.tar: {sha256, bytes, files: n, drive_file_id}}}
```

`verify(manifest, root)` re-hashes every file under `root`; `unpack` refuses a member that is not in `files`, a
path that leaves `root`, or a part whose sha256 differs. `dry_run` makes a fresh clone of the pushed branch at the
manifest's commit in a scratch directory (`git clone --no-checkout`, `git checkout --detach`), unpacks the parts,
verifies every hash, runs `sets verify` per set and body, builds every env (each body, both floors, R5 and R6),
runs G8a's notebook form per body, and writes `DRYRUN.json` beside the manifest (the clone's commit, counts, the
G8a records' identical counts, seconds, versions).

### 3.11 Policy rows for the bar runner (`s2rows.py`; C4)

`R5Row` is a `panels.TrackerRow(name, "R2", gains=None, terms=(), ankle_cap=(), policy=S2Policy(...),
controller="R5: ...")`: the panels' own row with their own policy hook (`panels.py:751-801`, `841-852`), so the
tracker, the start, the seeds, the recorder and the judge are exactly those of the R0 row beside it, and
`bar.rows(body, ["R5/42@10ms"], ...)` wraps it in `bar.Delayed` like any registered row. `S2Policy.begin(tk, built,
seed)` builds `lib.from_plan(built.plan)` on the built track and a `r5core.Core` on the tracker's estimator and
clock, zeroes the action history and the filter, and acts once; `act(tk, k)` after every tenth physics step:
`a = clip(model.predict(vecnorm.normalize_obs(actor_obs), deterministic=True))`, the filter, then
`tk.residual, tk.rate_factor = core.offsets(a_f)`: the env's step 1 and 2 through the same `Core` methods (the
critic's columns are not computed: `PrivilegedCriticPolicy.obs_to_tensor` pads them, `policy.py:131-141`). The
head is neutral unless a head script is given. `R6Row` is a `panels.Row` whose episode runs `r5env`'s R6 tracker
(its own reset and physics step, as `policy_rows.PolicyOnlyTracker`, `policy_rows.py:492-541`) under
`control.run` with an acting hook of its own.

A row is built from a checkpoint's four files (`configure`, or the JSON file `$STAGE2_POLICIES` names: row name
to job directory and step count). Building it loads the checkpoint once through `s2load.load_checkpoint`, so a
checkpoint whose fingerprint is not reproduced within 1e-5 raises before any episode ("not judged"; S2-12).
`describe()`, which is the record's `controller` and enters its key: `{row, controller, mode, body, ppo_seed,
checkpoint {steps, files {name, sha256}}, run {job, set, set_freeze, settings sha256}, layout, action {scales,
rate, filter_hz}, versions_trained, versions_judged, cross_version, fingerprint_max_abs_diff, fingerprint_tol,
deterministic: true, head: "neutral"}`; `cross_version` compares `s2load.comparable` of the two sets (torch's
build label dropped). `sources()`: the sha256 of `s2rows.py`, `r5core.py`, `r5lib.py`, `r5env.py`, `s2load.py`,
`schedules.py` and of `designlab/stage1/policy.py` and `env.py`. `spec()` carries the paths and hashes; a worker
checks the hashes again and loads lazily. The set a row is asked to run on must be the run's set and freeze
(refused otherwise: a policy is judged on the references it imitates).

### 3.12 The steps sheet (`stage2/tb/colab_steps.txt`; C10)

Plain text, wrapped at 115 columns, addressed to the owner, in the order of
`docs/investigations/stance_2026_10/stance3/colab_steps.txt`. It quotes every expected line with its values and
says, after each, what to do if it differs. Outline:

1. *What runs*: the branch and its full commit, the bundle id and the first 12 hex of its manifest, the rules
   file and its sha256 (`sha256sum`), the runtime type (High-RAM CPU) and why.
2. *Day one, the smoke (about an hour, one runtime)*: open the notebook from the branch
   (`https://colab.research.google.com/github/kuds/mesozoic-labs/blob/<branch>/hardware/trex_form_study/stage2/stage2_training.ipynb`),
   save a copy in Drive, Runtime > Change runtime type, the parameter cell's values with `SMOKE = True`, Run all,
   allow Drive. Check: the `Repository ref` line and line 1 (another commit: stop and tell me); line 2 (FAILED:
   stop; nothing was trained; tell me the line); line 3 (FAIL: stop); the nine `S2 SMOKE` lines with what each
   means; where the files are on Drive. "There is nothing to send: I read them from Drive."
3. *A launch (per runtime)*: the table of runtimes, each with its RUN_ID and QUEUE, `SMOKE = False`,
   `BUDGET_STEPS` as fixed after the smoke; Run all; the four lines with their expected values (line 4 "within
   about 3 minutes of line 3", with the rate below which to tell me); how long to leave it; what the end looks
   like (DONE.json, the runtime disconnects by itself).
4. *If Colab disconnects*: first tell a finished run from a cut one (`runs/<RUN_ID>/DONE.json` or `HALT.json`
   exists: finished, do not resume). Otherwise: the same copy, a fresh runtime, the same parameters with
   `RESUME = True`, Run all; the expected lines (`resuming session <k>: <job> from <n> steps`; any
   `WARNING: skipping checkpoint <n>: <reason>`); when no checkpoint exists yet, the same RUN_ID with
   `RESUME = False` is refused, so use `<RUN_ID>_b` and tell me.
5. *Messages that mean stop and tell me* (never restart, never delete, never change RUN_ID or REPO_REF on an
   error's advice): a refused resume (settings, inputs, sources, versions), `HALT.json`, `G8a ... FAIL`,
   `fingerprint not reproduced`, a rate under the stated floor, Drive full.
6. *Fallbacks*: if the inputs' line fails because a part is missing or cut short: which part to re-upload (the
   lead does it; you approve); if uploads keep failing: set `BUNDLE_SOURCE = "rebuild"` (only after the smoke's
   Y8 passed) or `"git:<ref>"` (only if you agreed to the data branch); if clips fail: nothing to do; if the pins
   do not install: stop and send the cell's output.
7. *Which lines were measured here and which are inferred* (as `stance5/colab_steps.txt` does): every line the
   local run printed is quoted from its log; Colab-only lines (the mount, the disconnect) are marked inferred.

## 4. Tests and gates

Fast tests run in the default suite; slow ones behind `STAGE2_SLOW=1`; a test that needs arrays, models, torch or
a built set skips without them; no test writes inside the checkout (each uses `tmp_path`).

| What must be proven | Test or gate | Default |
|---|---|---|
| **G8a** | `gates2.g8a` (the committed record per body and set); `test_stage2_r5env.py::test_zero_action_is_r0_on_the_s1_view` (the env on `setio.stage1_view` against `panels`' R0 row, 1 s, k = 0 and 3: identical); `::test_g8a_on_a_built_set` (slow: one bar condition of V0) | fast (needs Stage 1 arrays) / slow |
| A zero action is not "almost" zero | `::test_any_offset_breaks_the_digest` (one action entry at 1e-6: the first env step differs); `::test_zero_action_leaves_residual_none_and_rate_one` | fast |
| Mid-task start and queue | `::test_mid_start_state_is_the_files_frame` (qpos, scaled qvel, `tau_ref`, the estimator's pose); `::test_mid_start_queue_holds_the_references_commands` (with k = 3 the first three physics steps apply the commands of `t - 3..1` steps: `ctrl` equals an undelayed replay's shifted by 3); `::test_mid_start_walks_on` (slow, 75 cm: up 4 s from 3 frames) | fast / slow |
| 14 and 16 joints | `test_stage2_r5core.py::test_layouts` (87/129/15 and 101/145/17; blocks contiguous; actor first); `::test_x_env_steps` (the `-X` env builds and steps where `env.Stage1Env` still raises "pending"); joints addressed by name (a permuted joint list gives the permuted blocks) | fast |
| Commands and phase | `test_stage2_r5lib.py`: the command is constant inside a segment and changes at first frames only; the turn's pulse integrates to 90 degrees; nothing before a segment's first frame depends on it (two scripts that differ after a frame agree before it); the phase is 0 at the reference's left lift-offs; a K-stride file's `phase` is used | fast (synthetic set) |
| Reward | `test_stage2_r5core.py`: each term in [0, 1]; on a kinematic playback every imitation term is 1 and every penalty 0; each term moves when its cause is injected (a foot offset, a contact mismatch beyond 24 ms, a head pitch, a tail servo over its rating, a leg gap under 3 mm); the weights reach `describe()` | fast |
| Randomisation restored | `test_stage2_r5env.py::test_factors_are_applied_and_restored` (the next nominal episode's model equals the nominal one bit for bit, damping and friction-loss included); the tracker's feedforward keeps the nominal damping | fast |
| Policy rows equal the env | `s2rows.equivalence_check` (a row episode under `bar.Delayed` against the env driven by the same checkpoint: digest per physics step at k = 5); `test_stage2_s2rows.py` with a freshly initialised policy saved through `s2load` | slow |
| **Cross-version loading** | `test_stage2_s2load.py`: an archive saved with `schedules` has no bytecode members (`inspect_sb3_archive`); `load_checkpoint` reproduces the fingerprint exactly in a fresh process; a changed byte, a missing sidecar, a truncated zip and a wrong fingerprint are each refused with their reason; `comparable` drops torch's build label; a row built on a checkpoint whose record names other versions has `cross_version: true` and both sets in `describe()`. The real cross-version case is the smoke's Y7 (a Colab-saved archive loaded here) | fast (needs torch) |
| **Kill and resume** | `test_stage2_train2.py::test_a_killed_run_resumes_bit_for_bit` (tiny settings on a stub env: an uninterrupted run against one killed with SIGKILL between boundaries and resumed: every policy tensor, the optimizer's state, `obs_rms`, `ret_rms` and the fingerprint equal; also with another worker count); `::test_resume_refusals` (settings, inputs, sources with and without the flag, versions with and without the flag, a broken newest pair skipped with its reason); `gates2.kill_resume` (the same on the real env: the validation record) | fast (stub) / record |
| The worker count does not matter | `test_stage2_vecpool.py`: rollouts of a seeded stub env equal `DummyVecEnv`'s at W = 1, 2, 4, 8; a dead worker raises; `env_method` and `seed` keep env order | fast |
| Lesson 16 | `::test_minibatch_steps_are_counted` (a learning rate that trips `target_kl`: `minibatch_steps_run < possible` in `updates.jsonl`) | fast |
| `MUJOCO_GL` | `::test_the_trainer_assigns_mujoco_gl` (a subprocess started with `MUJOCO_GL=osmesa` in its environment builds a policy and runs one rollout: exit 0, and `os.environ` reads `disable` in it and in a worker); `gates2.gl_start` records it | fast |
| Monitor, G8b, stop rules | `test_stage2_monitor2.py`: the gain rule on tables of counts and survivals (two more; one second longer; against the best earlier check; the first check against R0L); G8b's 2 of 3; the fix doubles the mapped tasks' weights and leaves the reward weights; a second failure stops; three checks without a gain from 20M stop at the third; S2-10's record | fast |
| The look reading (**G9**) | `test_stage2_lookread.py`: synthetic series for each band on both sides of its limit; own-touchdown segmentation ignores a 10 ms shift of the whole motion; "not read" under 0.02 rad; fewer than 4 steps is an error; `gates2.g9` (the record) | fast / record |
| R2L | `test_stage2_r2lfit.py`: the rotation table (every kind at every delay once per six generations; written before generation 0; a resume refuses another table); every member of a generation gets the same episodes; the selection reads only the k = 5 cells; the ledger stops a fit at its share; `speedfit.py`'s sha256 is unchanged by importing `r2lfit` | fast (stub evaluator) |
| Bundle | `test_stage2_bundle.py`: two builds of the same files give byte-identical parts; the manifest's canonical sha256 does not change when the JSON file is rewritten with CRLF or other indentation; a changed file, an extra member and a path leaving the root are refused; `git check-ignore` on every output path (`parts/x.tar`, `drive/...`, `clone/...`, a checkpoint's four files, a clip, a log) | fast |
| **The notebook, here** | `test_stage2_nb.py::test_notebook_cells_parse` and `::test_pins_equal_the_study_venv` (fast); `::test_local_run_with_a_kill_and_a_resume` (slow, about 5 min: `nb local-run` on set V0 or V6 of `75-M-H-10` with `--tiny`: session 1 is killed after its first checkpoint reaches the stand-in Drive, session 2 resumes; asserts the four lines by regular expression, both SESSION files, STATUS.json's schema, `run.json`'s sessions and `resumed_from`, the kept checkpoints of K2 with records and fingerprints, a monitor record, a clip or its recorded failure, DONE.json, and that nothing was written inside the checkout) | fast / slow |

**What a reviewer can run in under 10 minutes** (from FS, round A's environment line):

```
$PY -m pytest -p no:cacheprovider tests/test_stage2_schedules.py tests/test_stage2_s2load.py tests/test_stage2_r5lib.py \
    tests/test_stage2_r5core.py tests/test_stage2_r5env.py tests/test_stage2_vecpool.py tests/test_stage2_train2.py \
    tests/test_stage2_monitor2.py tests/test_stage2_lookread.py tests/test_stage2_r2lfit.py tests/test_stage2_bundle.py \
    tests/test_stage2_nb.py                                                    # the fast set: about 6 min
$PY -m designlab.stage2.gates2 g8a --body 75-M-H-10 --set V6 --conditions walk/v0.4 --seeds 3042   # 20 s; "2 of 2 identical"
$PY -m designlab.stage2.bundle verify --manifest stage2/tb/bundle/<id>/MANIFEST.json --root .      # seconds
$PY -m designlab.stage2.r2lfit rotation --body 75-M-H-10 --set V6noF5 --seed 42                     # prints the table; equals the record
git status --porcelain; git check-ignore -v stage2/tb/runs/x/jobs/j/checkpoints/model_1.zip stage2/tb/bundle/x/parts/a.tar
```

The independent reviewer of the finished build should verify: (1) no file under `designlab/stage1/` changed
(`git diff --stat`), `regress.json` still current; (2) G8a's records recompute for a sample of episodes and the
"any offset breaks it" test fails when the env's zero test is weakened; (3) the env's reset and step order against
`env.py` line by line (section 3.2); (4) each reward term's formula and weight against W2, and that the weights in
a `run.json` are the config's; (5) the command is causal (the two-script test) and equals the bar's command
instants; (6) the fingerprint is checked on every load path (grep for `PPO.load` finds none outside `s2load`);
(7) a killed run equals the uninterrupted one, and a resume refuses each changed thing without its flag;
(8) `updates.jsonl` counts real optimizer steps; (9) the monitor's episodes and seeds are the fixed ones and its
gain arithmetic matches M3 on the records; (10) the look bands recompute from a trace's `poses50`, and G9's
negative controls fail as designed; (11) R2L's rotation is in the record before generation 0 and the selection
used only 10 ms cells; (12) the dry-run record names the pushed commit and every hash; (13) the notebook's local
run prints the four lines exactly as the steps sheet quotes them.

## 5. Work packages

Each builder works in its own worktree on its own branch; no two packages own a file; at most two agents build at
once, each with one heavy process (round A's rule). Names carry a "C" so that they do not collide with round A's.

| Wave | WP | Owns (exclusively) | Needs | Acceptance |
|---|---|---|---|---|
| 1, can start now | **CP** policy files | `schedules.py`, `s2load.py`, their two test files | nothing of round A (torch, SB3, the repository's `policy_loading`) | its tests; `python -m designlab.stage2.s2load check <job> <n>` on a scratch checkpoint |
| 1, now | **CV** workers | `vecpool.py`, its test | nothing | its tests (stub env) |
| 1, now | **CB** bundle | `bundle.py`, its test, `stage2/tb/.gitignore` | nothing for the tool (it is tested on Stage 1's files); the real bundle needs the frozen sets | its tests; a build and a dry run of a stand-in bundle of the two picks' models and Stage 1 arrays |
| 1, now | **CK** look reading | `lookread.py`, its test | the look ladder's arrays (`restore_inputs.py copy-from ... --group look`); synthetic recorders. The `bar.py` hook is the lead's (2.3) | its tests; the adapter on a kinematic playback of a V6 rung against `stage2/look/metrics.json` |
| 2, after round A's R is merged | **CE** env | `r5lib.py`, `r5core.py`, `r5env.py`, their three test files | `setio`, `s2io` (R); Stage 1 arrays (it develops on `setio.stage1_view`); round A's V0 set for its slow tests | its tests; scratch C1 and C3 reproduced through the real env on the S1 view; `throughput` on both floors |
| 3, after CE and CP | **CT** trainer | `train2.py`, `monitor2.py`, `clip2.py`, their test files | CE, CP, CV; the monitor's real run needs round A's B and CW (it is built against a stub row) | its tests; a tiny run with a kill and a resume on set V0 |
| 3, after CE, CP and round A's B | **CW** rows | `s2rows.py`, its test | CE, CP, `bar.py` | its tests; `equivalence_check` on V0 at k = 5 |
| 3, after round A's B and S | **CF** R2L | `r2lfit.py`, its test, `stage2/tb/r2l/run_r2l.sh` | `bar.py`, a built set (V0 for development) | its tests; a 2-generation smoke fit on V0 with `cut_s` (no record kept) |
| 4, after CT, CW, CK | **CG** gates | `gates2.py`, its test, the record paths under `stage2/tb/gates/` and `stage2/tb/validation/` | everything above; the frozen V6 sets for the committed records | G8a on V0 (development), then the run plan's records |
| 4, after CT and CB | **CN** notebook | `nb.py`, `stage2/stage2_training.ipynb`, `stage2/tb/colab_steps.txt`, `stage2/tb/requirements-colab.txt`, its test | CT's CLI (it starts in wave 1 against a stub trainer: the launcher, STATUS, the copies, the kill), CB | its tests; the local run with a kill and a resume |

What can start before round A is merged: CP, CV, CB, CK and the launcher half of CN (five packages; two at a
time). CE needs only round A's R package (already committed on `wt/wp-r`). Nothing in round C waits for the look
work except the committed gate records, R2L's real fits and S2-4.

Easy to miss: CE's `Core` is what CW's policy calls, so their interface (section 2.2) is frozen by this document;
CT's trainer records `train2.sources()`, which hashes CE's, CP's and CV's files, so those are frozen with state B;
CK's reading enters the bar's key for look sets, so it is frozen with state B too; CN must not import numpy in the
kernel (`nb.py` is standard library only at import; a test imports it in a process with numpy blocked).

## 6. Run plan

From FS; `PY=.venv/bin/python`; `export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1`; every long command through `tools/lowprio.sh`; `tools/memguard.sh` with
`PROTECT="designlab.stage2.train2"`. `TB=stage2/tb`. Each step is one line of an idempotent chain
(`$TB/run_trackb.sh`, round A's `step` pattern with done-markers); every command resumes by itself.

Rates used below [scratch C2, less about 10 % for the interference check and the added reward terms; to be
replaced by step V3's record]: 400 env steps/s per process on hard floor, 330 on robot_floor_carpet. With 8 envs
(6 hard, 2 carpet) in W workers and about 15 % for PPO: W = 2 about 650 env steps/s, W = 3 about 870, W = 4 about
1,250 on idle cores. Bar episodes: R0 0.08 s of wall time per simulated second, R2 0.2-0.45 (Stage 1b's fit
against round A's review figure), a policy row about 0.12.

| # | Step | Command | Estimate |
|---|---|---|---|
| V1 | G8a, per body and head trained (4) | `gates2 g8a --body B --set V6 --record $TB/gates/V6/B/g8a.json` | 40 episodes, about 3 min each body; 0.2 core-h in all |
| V2 | G9, per body | `bar run --set V6 --body B --rows R0@10ms ...` (the R0L records; they are also the baseline rows), then `gates2 g9 --body B --set V6 --record ...` | R0L records 0.15 core-h per body and head; G9 itself minutes; the Stage 1b re-run is round A's (0.4 core-h) |
| V3 | rates and memory | `gates2 rates --body 75-M-H-10 --set V6 --record $TB/validation/rates.json` (2 floors x 1, 2, 3, 4 workers x 16,384 steps; RSS per worker) | 15 min; provisional unless the machine is otherwise idle (lesson 10) |
| V4 | sigma probe (reported), filter rule | `train2 probe-sigma ...`; `train2 filter ...` (both sizes, current heads) | 280 short episodes, about 12 min, 0.2 core-h |
| V5 | learning-rate probe | `train2 probe-lr --bodies <the four> --set V6 --workers 2` (3 rates x 6 updates x 16,384 steps each) | 295k steps per body and head: about 8 min at W = 2; 0.5 h wall, 1.1 core-h |
| V6 | a training start with `MUJOCO_GL` set against it; kill and resume | `gates2 gl-start`; `gates2 kill-resume --body 75-M-H-10 --set V6 --workers 2` | 5 min |
| V7 | **S2-4** | `nb launch --run-id s2-4 --queue R5:75-M-H-10:42 --set V6 --budget 2000000 --workers 3 --local $TB/runs/s2-4 --drive $TB/runs/s2-4`, then `monitor2 s2-10 ...` | 2,015,232 steps: about 40 min at W = 3 (55 min at W = 2 beside the panels); 2.3 core-h with the monitors |
| V8 | the 60 cm local run (D11, M6; recommended) | `nb launch --run-id local-60 --queue R5:60-N-T-10:41 --set V6 --budget 5000000 --workers 3 --local $TB/runs/local-60 --drive $TB/runs/local-60` | 5,013,504 steps: about 1.6 h at W = 3; 4.8 core-h |
| V9 | R2L | `r2lfit rotation`, `fit`, `select` for (V6noF5, 75-M-H-10), (V6noF5, 60-N-T-10), then if granted (F25F9cl, 75-M-H-10); `--workers 2` | 3.2 core-h per fit and 0.8 per selection by the ledger: 8.0 core-h for the two; 4.0 more for the candidate (Q5) |
| V10 | the bundle and its dry run | `bundle build --bundle-id <id> --bodies <trained> --sets V6 --out $TB/bundle/<id>`; push the branch; `bundle dry-run --manifest ... --parts ... --ref <branch>` | build 2 min; dry run about 6 min; then the uploads (the lead, one approval per part) |
| V11 | the notebook, here | `nb local-run --notebook stage2/stage2_training.ipynb --drive $TB/drive --tiny` | about 5 min, 2 workers |
| V12 | the steps sheet | commit `$TB/colab_steps.txt` with the commit, the bundle id and manifest hash, the queue per runtime and the lines as V11 printed them | - |

Order on 4 cores shared with the 75 cm panels (two processes): V1, V2, V4 and V6 run one process each beside
them at any time; V5 and V7 at `--workers 2` beside them, or 3 once they are done; V3 last, alone, or it is
recorded provisional; V9 at two workers after V7, about 4 h of wall time for the two fits; V8 beside V9's second
half or overnight; V10-V12 need the frozen sets and the pushed branch. Sum without the candidate fit: validation
and S2-4 about 6 core-h, the 60 cm run 4.8, R2L 8.0; the plan's estimate for this CPU has 10 for validation and 8
for R2L.

On-disk layout of the outputs:

```
stage2/tb/gates/<set>/<body>/g8a.json, g9.json
stage2/tb/validation/rates.json, sigma_probe.json, filter.json, lr_probe.json, gl_start.json, kill_resume.json,
                     colab_archive.json (after the smoke)
stage2/tb/runs/s2-4/jobs/R5-75-M-H-10-s42/{run.json, updates.jsonl, gates/, monitor/, checkpoints/ (ignored)}
stage2/tb/runs/s2-4/S2_10.json                        S2-10's one record: both rows' 18 episodes, the counts, the survivals
stage2/tb/runs/local-60/jobs/R5-60-N-T-10-s41/...
stage2/tb/r2l/<set>/<body>/{start.json, r2lfit/42/..., r2lg5.json}
stage2/tb/bundle/<bundle_id>/MANIFEST.json, DRYRUN.json, parts/*.tar (ignored)
stage2/tb/colab_steps.txt, requirements-colab.txt, PREREG_TRACKB.md
stage2/tb/drive/ (ignored: the stand-in Drive of local runs), stage2/tb/clone/ (ignored: dry-run clones)
```

## 7. Risks and open questions for the lead, most important first

1. **S2-10's gain may be out of reach for a reason that is not learning.** "Succeeding with the look" contains
   reading 2, and on `75-M-H-10` the tail servo's sustained limit failed in 20 of 20 Stage 1 episodes at 0.35 m/s
   and above. If R0L succeeds in 0 of 18 for that term and stays up in all 18, R5 can show a gain only by pulling
   the tail under its rating in two episodes (its `servo` penalty and its +-0.15 rad of tail authority aim at
   that) or by a second of survival that is not there to gain. The ruling fixed reading 1; I recommend that
   S2-10's record also print the counts of reading 3 with the look bands, as a reported number, and that the lead
   decide now whether "no gain for the tail's sake" goes to the owner as a no-go or with that number beside it.
2. **No edit under `designlab/stage1/`, so the trainer is `designlab.stage2.train2` and not an extended
   `train.py`** (D2). It changes the plan's B.4 wording; S2-17's ruling allows the extension but does not need it.
   Please confirm, or name what must live in `train.py`.
3. **`bar.py` needs the reading hook, 50 Hz poses and a trace key for look sets** (2.3). If WP B can still take it
   before state A, no key moves; otherwise it is state B's one change to a round-A key module, and every look-set
   row waits for it. Reading 1 is undefined in the bar until then.
4. **The look sets R5 needs are more than round A lists** (D5): the turn in place with its start and stop; V6 on
   both `-X` bodies and on `60-N-T-10`; V6noF5 on both picks. If the 60 cm body gets no quick turn (ruling 4), its
   R5 trains without the turn tasks and its bar turn items are "not run". Whether `sets.py` and `moves.py` build
   on a 16-joint model has not been tried by anyone.
5. **The 60 cm policy must learn balance from a base that falls** (D11). The scratch runs say a delay curriculum
   would not change that. I recommend the local 5M run (V8, M6) and telling the owner before any 60 cm Colab job
   if it fails G8b's reading; it costs 4.8 core-hours that the plan's estimate does not list.
6. **Sigma 0.1 knocks the 75 cm base over about every 4 s** [scratch C2]. That is the plan's choice and PPO's
   usual start, but it means early rollouts are short episodes on both sizes, and the learning-rate probe must be
   run at that sigma on every body and head (O3). The sigma probe is reported and decides nothing (O2).
7. **Every checkpoint is a boundary that resets all envs** (D16). It buys an exact kill-and-resume test; it also
   cuts up to 5 % of episodes short in every run. If the equality test fails on the real env for a reason inside
   SB3, the build stops and comes back to the lead: no tolerance is set after seeing a difference.
8. **Colab's image decides whether the pins install.** numpy 2.5.4, scipy 1.18.1 and torch 2.15.0+cpu must have
   wheels for Colab's Python; a Python minor other than 3.13 still loads here (no bytecode, S2-12's rule) but
   makes every Colab checkpoint `cross_version`; replacing Colab's CUDA torch by the CPU wheel costs a download at
   every setup. Only the smoke settles it; the fallback is the image's own versions, recorded, under S2-12.
9. **Uploads of tens of MB through the Drive connector are untried**, and each part needs the owner's approval.
   The parts are 20 MB; the fallbacks are 5 MB parts, a rebuild on Colab checked by hash (unproven on Colab's
   BLAS; the smoke's Y8 tells), and a data branch (the owner's consent; it stays in the repository's history).
10. **R2L's budget holds two fits, not three.** 8.0 core-hours cover the V6noF5 fits on both picks with their
    selections (Q5). The F25F9cl candidate on `75-M-H-10` needs 4.0 more or is reported "cut by the pre-registered
    order". The cost per evaluation is uncertain by a factor of two (0.2 against 0.45 s per simulated second), so
    a fit may end before 30 generations; the ledger decides, and the record says at which generation.
11. **`PoolVecEnv` is new code under every run.** It is thin (each worker runs SB3's own `DummyVecEnv`) and its
    equality with one process is tested bit for bit; the alternative is SB3's `SubprocVecEnv` with 8 processes on
    every machine.
12. **The action filter's rule is mine** (O4: 20 Hz if the base survives action noise about as long as at 10 Hz,
    else 10 Hz). It is cheap and decided before training; it does not show which filter learns better.
13. **The reward's new weights are untested** (W2). They are recorded and fixed for all runs; G8b's one fix
    changes task weights only, so a badly sized term cannot be repaired inside the round without a labelled
    change.
14. **Reading 1 is read on 50 Hz poses** and on the robot's own steps; the reference's numbers come from a
    playback of the same window. G9's item (a) compares the adapter with the look ladder's own numbers; if they
    disagree by more than its limits, the definitions of U1 need the lead before any look-set record.
15. **robot_floor_carpet stays unsolved for the base** [scratch C3: it falls in 2-4 s there from any start], so a
    quarter of the envs give short episodes; the floor is reported, never gated, as the plan has it.
16. **The head overlay goes through `tk.residual`** (angle only; the reference's rates and feedforward do not
    know the gaze shift). Head-first turns (F11) are not built. The bar's "cost of using it" rows will need a head
    script in `s2rows`, which has the hook but no script yet.
17. **The fingerprint's observations are synthetic** (D14); they exercise the network away from saturation but
    are not states the robot visits. The row equivalence check (4) covers real observations.
18. **Estimates rest on a stub env** (Stage 1's observation and reward on R5's base). Step V3 replaces them.

**For the owner** (through the lead): the approvals of the bundle's uploads; consent only if the data-branch
fallback is needed; the day-one smoke (about an hour); to be told before 60 cm Colab hours if the local 60 cm run
fails G8b's reading; to be told first if S2-4 shows no gain. Nothing else in round C needs an owner decision.
