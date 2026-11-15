# Work package CE: the env: `r5lib.py`, `r5core.py`, `r5env.py`, `g8a.py`

You are building one part of a robotics simulation study (a robotic T. rex biped; everything is simulation). You
need no other context than this brief and the two texts it points into:
- the design: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_C_DESIGN.md` (read its header, the decisions and sections named under "Read first" below);
- the rules fixed before the runs: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/PREREG_TRACKB_draft.md` (rule ids such as "K3" or "W2" refer to it).
Round A's design, where a round-A API is named: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/design/ROUND_A_DESIGN.md`.

**Read first:** design D1, D3-D11, D16 (the three conditions on the env), D23; sections 2.2 (`r5lib`, `r5core`, `r5env`, `g8a`), 2.3, 3.1, 3.2, 3.3, 3.4, 3.7 (G8a's alignment and the "identical" rule); rules E1-E5, T1-T3, W1-W5, J1-J5, Z1. Code: `designlab/stage1/env.py` (all of it: you mirror its reset and step), `control.py` (`Tracker.__init__`, `reset`, `step`, `Track.joint`, `Track.info`, `run`), `speedfit.py` (`servo_delay`), `policy_rows.py:492-541` (`PolicyOnlyTracker`), `legs.py`.

## Goal

R5's and R6's training env for 14-joint and 16-joint plants, the reference library it plays (scripts over a frozen set, or over Stage 1's run directory for development), the core that the env and the policy rows share (layout, observation, action mapping, reward), and gate G8a's comparison. The env composes Stage 1's parts and edits none of them.

## Files you own (and no others)

- `designlab/stage2/r5lib.py`, `designlab/stage2/r5core.py`, `designlab/stage2/r5env.py`, `designlab/stage2/g8a.py`
- `tests/test_stage2_r5lib.py`, `tests/test_stage2_r5core.py`, `tests/test_stage2_r5env.py`, `tests/test_stage2_g8a.py`

Do not edit any other file: nothing under `designlab/stage1/`; no round-A module (`designlab/stage2/bar.py`,
`readings.py`, `setio.py`, `s2io.py`, `moves.py`, `sets.py`, `setscreen.py`, `barreport.py`, `moves_look.py`); no
other package's file; no committed record; nothing at all inside `/home/user/mesozoic-labs` (the main checkout:
you work in your own worktree). If you believe an edit elsewhere is unavoidable, stop and tell the lead: the design
claims none is needed (its section 2.3).

## Setup

Worktree: the absolute path the lead gives you; work only there.
```
cd <your worktree>/hardware/trex_form_study
PY=/home/user/mesozoic-labs/hardware/trex_form_study/.venv/bin/python
export PYTHONPATH=. MUJOCO_GL=disable OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
$PY tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays x
```
Versions: Python 3.13.18, torch 2.15.0+cpu, stable-baselines3 2.9.2, gymnasium 1.4.0, mujoco 3.10.0, numpy 2.5.4.
Needs from other packages: round A's `setio.py` and `s2io.py` (package R, branch `wt/wp-r`; the lead merges it into your branch). `g8a.bar_form` and `g8a.short` need round A's `bar.py`: import it inside those functions; their tests skip until `bar` imports. Wave 2.

## Machine-sharing rules (4 cores, shared with other builders and with running panels)

- One heavy process at a time, always `nice -n 15` (or through `tools/lowprio.sh`); at most 2 worker processes in
  anything you start; nothing longer than about 10 minutes without telling the lead first.
- Never run the full test suite (`pytest tests/` or `pytest` without a file): run your own test files only, slow
  ones (`STAGE2_SLOW=1`) one file at a time.
- Every job you write must be resumable and idempotent (the container restarts without warning) and bounded in
  processes and memory.
- **Episodes (design D23, rule X1):** until the lead tells you that task 3's first unit of the 75 cm round has
  run, no R2-mode, policy or training episode on the sets V0, F25F9 or F25F9cl; no closed-loop episode on a look
  set (V6, V6noF5) before its gate G9-ref has passed. Use Stage 1's own run directory through the read-only S1
  view (`setio.stage1_view`, `stage1/runs/2026-11-08/<body>`) or a synthetic set in a temporary directory.

## Engineering rules

New code only under `designlab/stage2/` (and the paths you own). Records are JSON with their rule texts,
provenance (commit, dirty with your own outputs excluded, versions) and sha256, written atomically (round A's
`s2io.write_json`, `sha256_json`, `git_state` once `s2io` is on your branch). Names are formatted with `%g`.
`git check-ignore` must report every path your code writes that is not a committed record. Tests write only under
`tmp_path`; a test that needs arrays, models or torch skips without them. Fast tests in your test file(s); slow
ones behind `STAGE2_SLOW=1`. Every CLI module follows round A's skeleton (thread variables before numpy; `main()`
of the importable copy) and, where the design's 2.1 says so, starts with `tbenv.assign_gl(...)` before any other
import.

## API (frozen: packages CT and CW build against it)

Exactly the listing of the design's section 2.2 under `r5lib.py`, `r5core.py`, `r5env.py` and `g8a.py`, and
`R5Config` as in 3.2. The interface of `Script`, `Library` and `Core` may not change without the lead.

## What to build, with pointers

1. **`r5lib`** (design 3.3, D3, D5): `Library(p, body, set_name, rate)` reads a frozen set through
   `setio.read_inputs(setio.set_dir(set_name, body), files, require_frozen=True)` and builds each task of
   `TASKS` (rule T1's table) as a plan of round A's track items with `setio.build_track(p, set_dir, plan, omega,
   tr_doc)`; `omega = control.omega0(p, q_home, crouch_frac=the set's)`. `set_name == "S1"`: `setio.stage1_view(
   stage1/runs/2026-11-08, body)` and `S1_TASKS` (`stand`, `walk`, `brisk`, `stop`: the plans of Stage 1's
   `env.TASKS` with those names). A task whose reference is absent or `available: false` is dropped and listed with
   its reason; `tasks` is the **resolved** table and `inputs` holds each used file's sha256 and entry sha256, the
   freeze index and manifest sha256 and `panels.key_context(body)["plant_inputs"]`. `Script`: every table of 3.3,
   each defined for any `t_ref >= 0` and continuing past `t_end` as `Track.joint` / `Track.info` do; `mid_spans()`
   without holds; `required_margin` as 3.3's last bullet. Iterate no `set` anywhere (rule: `PYTHONHASHSEED`
   independence).
2. **`r5core`** (3.1, 3.4, D7-D10): `layout(joints, heading=True)`; `Core` with the observation, the filter
   (`alpha = 1 - exp(-2 pi f x 0.02)`), `offsets` (returns `None` and exactly `1.0` for a zero filtered action with
   the head neutral), `targets` (R6), the per-physics-step accumulators and the reward of 3.4 with
   `REWARD_WEIGHTS`, `REWARD_SCALES` and `REWARD_RULE` equal to rule W2's table (a test compares the dicts with a
   literal copy of that table); `interference` on `legs.LegGap` (moving pairs for the ramp, all pairs for "below
   zero"); `HeadScript.drawn` (rule J4) and `HeadScript.fixed` (rule E5's H1); `R6Tracker` (as
   `policy_rows.PolicyOnlyTracker`, with `reset_at` for a mid-task start and the queue filled with the reference's
   joint targets). `r5core` must not import `r5env`.
3. **`r5env`** (3.2, D6, D11): construction with **every task's tracker built there, on the nominal model**;
   `reset` in 3.2's thirteen steps (nominal arrays restored before the script is chosen; the chosen tracker's
   `damp` and `fric` asserted equal to the nominal copies); `step` in 3.2's seven; `R5Tracker.reset_at` (D6: lend
   the track `q0`, `qd0` for one `super().reset("frame", ...)`, restore, set `tau_ref`); the delay through
   `speedfit.servo_delay(p, k)` with its queue refilled for a mid start; randomisation J1-J5 from the env's own
   generator in the stated order; `set_task_weights`; `describe()` without any path; `env_configs` (floors by D13);
   `throughput`. `R6Env` per D8 (the mid-start action history).
4. **`g8a`** (3.7): `compare` (the pure rule on two digest lists and two end steps); `stock(body, condition,
   seed, k, seconds)`: the env on `Library(p, body, "S1").from_plan(<the stock condition's plan from
   panels.Planner>)` with the zero action against `panels.tracker_rows(body, run_dir, ("R0",))["R0"]`'s episode
   inside `speedfit.servo_delay(b.p, k)`, digests through a wrapper lent to the built condition's `recorder` (an
   instance attribute for the one episode, removed in a `finally`); `bar_form` and `short` the same against
   `bar.rows(body, ["R0@<ms>ms"])` on a set (lazy `bar` import). Alignment exactly as 3.7's table; the env's
   length for a plan is `ceil(round(seconds / dt) / 10)` env steps.

## Tests to write

The rows of the design's section 4 table that name your four test files: G8a (S1 view; a length that is no
multiple of 20 ms; the compare rule on tables); a zero action is not "almost" zero; **reset is history-free**
(an env after 6 randomised episodes against a fresh env, digest per physics step, hard and carpet, stance and mid
starts, 14 and 16 joints) and trackers built nominal; mid-task start and queue (three tests) and never in a hold;
layouts (88/130/15, 102/146/17; 87/129, 101/145 without the heading; joints by name); `-X` env steps where
`env.Stage1Env` still refuses; commands, heading, phase, tables past the end at clock factor 1.25; the resolved
task table and absent tasks on a synthetic set in `tmp_path` (build a tiny frozen set from Stage 1 arrays with
`setio`'s own helpers, kinematic only); the reward's rows (each term in [0, 1]; a kinematic playback scores
imitation 1 and penalties 0; each term moves when its cause is injected; weights and scales equal the rule's
table); R6 on 14 and 16 joints and its mid-start history; factors applied and restored. Slow (`STAGE2_SLOW=1`,
S1 view only): a mid start walks on for 4 s from 3 frames on `75-M-H-10`.

## Acceptance

```
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_r5lib.py tests/test_stage2_r5core.py
nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_r5env.py tests/test_stage2_g8a.py
nice -n 15 $PY -c "from designlab.stage2 import g8a; r = g8a.stock('75-M-H-10', 'walk/brisk', 3042, 5, 6.0); print(r['identical'], r['compared_steps'])"   # True 3000
nice -n 15 $PY -c "from designlab.stage2 import r5env; c = r5env.env_configs('R5', '75-M-H-10', 'S1')[0]; print(r5env.throughput('R5', c, 2000))"            # env steps/s, hard floor; then the carpet env (index 3)
git status --porcelain            # only your eight files
```
Report the two throughput numbers and the resident memory of one env process; they replace the design's
estimates of 350 and 290 env steps/s.

## Commit

Commit on your own branch in your worktree, messages ending with the attribution lines your session's system
reminder gives, exactly; name no model anywhere else. Do not merge, do not push unless the lead says so. Final
report: files changed, each acceptance command with its exit code, every place where you read the design or a rule
in one of two possible ways (say which you chose: these are rules fixed before the runs, so say so rather than
decide silently), and anything the design asked for that you could not build.

---

## LEAD ADDENDUM 2, 15 Nov: RULINGS AFTER THE REVIEW (the fixer follows these)

These settle the points the reviewer handed up. Where a ruling and the reviewer's proposed fix differ, the ruling
wins. The lead writes the matching words into `STAGE2_TRACKB_PLAN.md` (still a draft, in force only once settled and
merged; no Track B run has been made); you do not edit that file.

**CE-R1. The reward's four readings stand as coded** (finding 1, blocker). The lead amends W1, W2, W3 (a) and W5 to
the code's readings, tagged [lead, before any Track B run]: the duty term reads a foot only where the reference's
own share over the same stride lies inside the band; progress is min(1, max(0, (v_ref_bar - v_bar) / (1.05 v_cmd)) /
0.10) with v_ref_bar as `r5core.REWARD_RULE` defines it, restarted at the segment's first frame and read after one
stride; clearance (a) times the swing by the planned length of the reference's swing in progress, on the reference
clock; positions are those of the last physics step's start. The reason is one the rules already state: a playback of
the reference scores every penalty zero. Your part: add the last sentence to `r5core.REWARD_RULE`, and make W2's
check an equality of the env description's `reward_rule` text (a test that fails when the formula text and the rule
text differ). No formula changes.

**CE-R2. The commanded heading in a turn in place** (finding 2). Take the proposal: inside a finite segment that is
no finite turn and that enters or leaves a gait with a yaw rate, the command stays E4's and the commanded heading
follows the reference's own yaw; the next segment's `heading0` is its value at the segment's end. Add the in-place
plan on S1 arrays to the playback test. For the finite turns nothing changes in the code: the bar judges the heading
against the command's own integral, so a playback carries the heading penalty of its ramps. Add one test that
measures that penalty on a finite turn of the S1 arrays and pins it, and give the number in your report; the lead
writes the sentence into W1.

**CE-R3. A later freeze of the same set** (finding 3). Take the fix as written (rule K4).

**CE-R4. Scripted head targets keep the bar's joint margin** (finding 4). Rule: no scripted target of the head lies
within 0.15 rad of a joint limit (the bar's 0.10 rad margin plus 0.05 rad for tracking). Compute the numbers from
each -X body's own ranges and report them; for the jaw's [0, 0.6] rad that gives `JAW_OPEN_RAD = (0.3, 0.45)`, and
the gaze target is cut the same way from the neck yaw's range. Add the test on both -X bodies (a playback with head
events at their extremes scores range 0 and meets the bar's 0.10 rad). This narrows how far the jaw opens in a
script; it is a look choice the lead reports to the owner.

**CE-R5. The rest.** Findings 5, 6 and 7 as the reviewer proposes; for 6 take a rung's target speed from its name, as
the bar does, so that training and the bar apply the same duty band at 0.30 m/s. Finding 8: no code change; the lead
adds the numbers to T1 and J4. Finding 9: add the refusal as written (`allow_75cm_round=False`, rule X1).
