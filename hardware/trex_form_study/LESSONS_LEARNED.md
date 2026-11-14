# Lessons learned, 7–14 Nov 2026

Written on 14 Nov 2026, at the end of the study's first long working session (rounds one and two, Stage 1, Stage 1b
and Stage 2's first phase). It is for whoever continues the work, the next session first, and for the owner. It covers
what the work found about the robot, how the work was done, what went wrong and how it was caught, and how to work in
the cloud container. It summarises and points to sources. The records and reports it cites are the evidence, and where
they disagree with this file, they win.

Paths are relative to `hardware/trex_form_study/`. "a / b" figures are 60 cm / 75 cm unless stated. S1 is the Stage 1
report (`stage1/runs/2026-11-08/STAGE1_PROBE.md`) and S1b the Stage 1b report
(`stage1/runs/2026-11-10-speed/STAGE1B_SPEED.md`). P, F, E and A rows are in `DECISION_LOG.md`. The next session's
starting prompt is `NEXT_SESSION_PROMPT.md`.

## 1. Where things stand

| Step | What it did | Main document | State on 14 Nov |
|---|---|---|---|
| Round one (7 Nov) | 4 sizes, 3 widths, 3 body families, 3 postures, sized from sourced parts and screened | `ROUND1_REVIEW.md` | done; the owner answered P10–P14 |
| Round two (7–8 Nov) | 20 candidates at 60 and 75 cm, an overnight refinement, three picks | `ROUND2_REVIEW.md`, `round2/SHORTLIST.md` | done; the picks were adopted (P16–P23) |
| Stage 1 (8–10 Nov) | closed-loop probe of the picks, controllers R0–R4 (§2.1) | `STAGE1_PLAN.md`, S1 | done; S1 corrected in place on 10 Nov |
| Stage 1b (10 Nov) | the tracker refitted for speed (R2s) on the 60 cm picks and on a faster-knee variant; servo delay | `STAGE1B_PLAN.md`, S1b | done |
| Stage 2 plan (10–11 Nov) | character plus "somewhat agile": the agility bar, R5, Colab training | `STAGE2_PLAN.md` | owner decisions 3–6 taken (P29) |
| Stage 2, Track A, phase 1 (11 Nov) | look features in the gait generator, the V0–V6 ladder, its Stage 0 screen, the tail-phase check, clips, the owner's look review page | `stage2/look/SCREEN.md`, `stage2/look/RENDERS.md` | awaiting the owner's L1 picks |
| Stage 2, S2-2 (11 Nov) | the expressive head (`-X`: neck yaw and jaw) built and costed | `stage2/x/HEAD_X.md`, `stage2/x/X_COSTS.md` | done; its G1 is provisional (§5) |

The bodies are `60-N-T-10` (the 60 cm Tapered), `60-N-H-10` (the 60 cm Hybrid, dropped for Stage 2 by P29) and
`75-M-H-10` (the 75 cm Hybrid with the targeted drives and bigger feet), plus the variants `-K` (a faster knee, Stage
1b) and `-X` (the expressive head). Nothing from Stage 1 onward has been adopted (P24). The draft decision-log rows
wait for the owner (§7). All of this is on the branch `claude/mesozoic-trex-design-iteration-azm2fs`, whose pull request
is kuds/mesozoic-labs#667 (https://github.com/kuds/mesozoic-labs/pull/667).

## 2. What the robot can and cannot do (in simulation)

### 2.1 The controllers

R0 is the reference walk's feedforward with PD servos. R1 adds a balance tracker on the true centre of mass, and R2 the
same tracker on the estimated one, with 13 gains fitted by CMA-ES; each body has two fits. R3 adds a PPO residual on R2,
and R4 is a policy alone. R2s is Stage 1b's speed refit of R2. Stage 2 plans R5, an imitation policy under speed and
turn commands whose zero action is R0L, and R6, a policy alone. "Shown" means that all 20 episodes stay up and at least
17 meet every limit, under both fits or seeds; with one, the verdict is provisional.

### 2.2 What is shown

- **At no servo delay and the model's grip (μ 0.9), R2 shows all three picks on the same 14 of 20 gated conditions.**
  These are standing on hard floor, robot_floor_carpet and low pile; the nominal walk (0.218 / 0.243 m/s) at μ 0.9
  and 0.54; the 0.35 rad/s walking turn and the 0.5 rad/s turn in place; starting and stopping; and standing shoves at
  half the Stage 0 bound (S1 §2; draft F26).
- **The 75 cm body walks and turns even open loop** (R0) and is the most tolerant body on most margins (S1 Answers).
- **The fastest walk shown is the nominal one at both sizes.** It is the walk the owner found too slow (S1b Answers).

### 2.3 Where it stops

- **Speed.**
  - At 60 cm the brisk walk (0.315 m/s) is not shown. The Tapered speed refit came closest, one success short: 19
    of 20 up and 17 meeting every limit.
  - The animal-like walk (Froude 0.05, 0.446 m/s) stays up in 16 of 20 after the refit, but about 3 % short of its
    target speed.
  - At 75 cm the brisk walk (0.35 m/s) stays up in every Stage 1 episode and misses only the tail servo's planning
    limit, 1.04–1.07 × A1 (S1, S1b).
  - A faster knee did not change the 60 cm answer: the refit, not the knee, raised survival (S1b Q2).
- **Servo delay is the largest unknown.**
  - At 10 ms of command delay every 60 cm walk and walking turn falls in all 20 episodes, refitted or not, on every
    body, with one exception, a Hybrid nominal walk (S1b Answers, §4).
  - The 75 cm nominal walk stays up to 8 ms and falls at 10 ms (S1's correction).
  - Faster walks tolerate less: about 4 ms for brisk and about 2 ms for Froude 0.05 (hints, S1b §4).
  - Nobody has measured the real servos' delay (draft A17).
- **Walking shoves.** At half the Stage 0 bound, 0–8 of 20 walks stay up per direction. Walking shoves are recovered
  only up to 0.07–0.15 / 0.10–0.17 m/s. The tracker's fit never saw a shove (S1 Answers).
- **Thin carpet.** Every walking start falls, on every body, in the unmeasured carpet model (S1).
- **Learning so far.**
  - R3 added no new pass and used about 1 % of its authority at σ 0.02 (S1 N3). R4 learned only to stand still.
  - Budgets were small: R3 trained to at most 2.5M steps.
  - Character policies of the kind Disney publishes train for about 22 GPU-days each (`stage2/research/research_look.md`
    §2), so "not learned" at this scale says nothing about the body.
- **Fit-to-fit spread.** Two fits of one body differed by 15–17 of 20 at the brisk walk (S1b Answers). One fit is a
  narrow test, which is why Stage 2 requires two seeds.

### 2.4 What limits it

- **Not the leg drives, as far as the model shows.**
  - Removing their torque limits never helped significantly where enough walks fell to test it, and only falls and
    single stumbles pass a drive's rated speed (S1b §5).
  - In the reference walks the binding limit is the knee's speed: 0.93–0.95 of it at Froude 0.05.
  - Lighter materials buy torque headroom and runtime, not speed headroom (F22).
- **The tail servo (STS3215) is near its sustained planning limit.**
  - Closed loop, it sits at 0.97–1.03 of A1 at the 60 cm Froude 0.05 walk and 1.04–1.07 at the 75 cm brisk walk.
  - With the plant's joint damping and friction it is over A1 at the 75 cm animal pace: cont up to 1.071, peak up to
    1.178 (`stage2/look/SCREEN.md`).
- **Today's tail swings in the wrong phase.**
  - It adds 8–12 % to the stance foot's peak yaw moment.
  - F9's phase cuts the friction need by 14–18 % and roughly halves the tail servo's load, because the body's sway
    then drives the tail and the servo brakes it (SCREEN.md, "In brief").
- **The 60 cm neck is close to its heat budget.**
  - The STS3250 neck runs at 0.83–0.84 of A1 in the closed-loop nominal walk (S1 §3).
  - With the look's head bob and the plant's damping, its peak reaches up to 1.17 at the animal pace (SCREEN.md).
  - The `-X` jaw servo in the head would take the closed-loop hold to about 0.93, an estimate (X_COSTS.md).
- **Joint damping and friction dominate the small servos' loads.** The values are the MJCF defaults on the neck and tail
  (0.2 N·m·s/rad and 0.15 N·m), unsourced (A16). A bench measurement could move these loads more than any part change.
- **Grip.**
  - Stance-foot spin, the pad's yaw moment, is the friction limit that binds (F21).
  - At 60 cm the straight walk at 0.40 m/s or faster needs more than μ 0.54 on every look rung, and more than 0.9 on
    several (SCREEN.md).
  - The treads' estimated grip is 0.42–0.8 on vinyl and laminate (F22).
- **Size.**
  - At the same gait, speed grows with the square root of size: v = √(Fr·g·H).
  - The 75 cm body is 11–12 % faster than the 60 cm one, and 120 cm would be about 40 % faster: bigger buys speed
    slowly (S1b Answers).
  - Froude speeds at 60 cm (H 0.405 m): nominal 0.218, brisk 0.315 and Fr 0.05 0.446 m/s. At 75 cm (H 0.502 m):
    0.243, 0.351 and 0.496 m/s.

### 2.5 The expressive head (`-X`)

- It adds 0.150 kg and moves the centre of mass 5.2 / 4.4 mm forward.
- The one-foot hip-pitch hold rises from 0.55 to 0.65 / 0.76 to 0.83 of its limit.
- With the new joints held still, the nominal and Froude 0.05 walks stay within every drive limit. The knee's speed
  goes from 0.93 to 0.97 / 0.93 to 0.96.
- A 0.6 rad gaze shift fits in 0.5 s at 60 cm and in 0.6 s at 75 cm.
- A 0.6 rad jaw opening in 0.4 s runs the servo at its rated speed; in 0.5 s it keeps both margins. A quick snap is out
  of reach.
- The cost to watch is the 60 cm neck's heat budget (§2.4; `stage2/x/X_COSTS.md`).

### 2.6 What none of this shows

- **Simulation only.** The servos are ideal PD behind a torque-speed envelope and a delay. There is no backlash, no
  heat beyond A1 and no fall damage.
- **Unmeasured inputs.** Joint friction and damping, the carpet, grip and delay are unmeasured.
- **Untestable flag.** The tracker's pad-edge flag could not be tested (S1 N1).
- **No hardware.** No hardware has been built or measured.
- **Reference walks are not controllers.** Kinematic clips and Stage 0 screens show prescribed motions with perfect
  tracking.

## 3. How the work was done: method lessons

1. **Pre-register the pass rules, stop rules, seeds and budgets before any run, and label any later change as such.**
   An example is decision 39: a tolerance set after seeing a 17.997 s against 18.000 s miss is marked in every place it
   is used. Plans carry numbered decisions (`STAGE1_PLAN.md` has 39) so that a report can cite the rule it applied.
2. **Run two fits or seeds.** One is a narrow test (§2.3), and the plan's verdicts are provisional with one.
3. **Keep prospective design, simulation evidence and measured results apart, in words as well as files.** No result
   so far is measured. Kinematic clips carry "kinematic reference, not a simulation" on every frame.
4. **Have every build, report and owner page reviewed by an independent agent.** Give the reviewer the code, the records
   and the rule, never the lead's conclusion, then fix and re-check. The reviews caught real errors:
   - S1 said the 75 cm walk survives at least 20 ms of delay. Its own records showed falls at 10 ms (corrected 10 Nov;
     lesson 5).
   - The `-X` gates were first recorded "all pass" while G1's step rate had been timed on 2 of 4 cores. G1 is now
     provisional, and the cores are recorded (`stage2/x/HEAD_X.md` §3).
   - The look review page's blind clips leaked their labels. A separate viewers' page was made, and the mechanical
     "tells" list was corrected.
   - The Stage 1b report mixed measured 75 cm speeds with targets. It now gives targets.
5. **Do not trust a search's reading; check every point it evaluated.**
   - The stay-up search evaluated 0 and 20 ms only. It assumed that a walk falling at some delay falls at every
     larger one, and so read "works across the interval".
   - The falls at 10 ms were in the same records, from another search.
   - No delay search ran 2 ms or 12–18 ms, and the grip searches skipped μ 0.10–0.20 (S1, correction note).
6. **Change the generator only behind options that default to today's behaviour, and test that bit for bit.**
   - G2: with default GaitSpec fields, Stage 1's references regenerate byte for byte.
   - Every existing model keeps its recorded `xml_sha256` (`tests/test_stage2_x.py`, `tests/test_stage2_look.py`).
7. **Hash the inputs into the records, and make reports from the records.**
   - Every Stage 1 record key holds the sha256 of the model files the plant reads (`panels.key_context`).
   - The tables are machine-made (`python -m designlab.stage1 report`, `python -m designlab.stage2.screen md`,
     `python -m designlab.stage2.motion md`), and reports cite record paths.
   - The cost: the git-ignored inputs must be restored exactly (§4.1).
8. **Read a metric's definition before reading its verdict.**
   - The speed term passes only when the judged 10 s mean reaches the target; the command is 5 % above it.
   - Toe clearance read on the reference's clock misreads a foot that lifts 10 ms late (S1b N5).
   - Sideways drift went unjudged: a median 0.62 m in 10 s with the heading gain near zero (S1b N3).
   - Stage 2's bar fixes both of the last two (`STAGE2_PLAN.md` §5).
9. **Fall attribution is a hint (decision 20).**
   - It needs at least 5 falls.
   - It lifts relaxable limits and applies a sign test.
   - Its labels name a likely cause, not a proof. The pad-edge candidate cannot be tested.
10. **Timing gates need an idle machine with one process per core.** G1's step rate failed on arm K's bodies and on the
    pick alike because the machine was slower that day (`STAGE1B_PLAN.md` decision 11). On a shared machine a timing is
    provisional, never a pass.
11. **Price before rendering, and render only what the owner must judge.** The Stage 0 screen (drives, centre of
    pressure, sway, accelerations, clearance, passive neck and tail loads, grip) ran on every rung before any clip was
    made (`stage2/look/SCREEN.md`).
12. **Communicate with the owner plainly.**
    - Answer the question first, in plain words, then give the evidence.
    - Never mix measured values with targets.
    - Correct a report in place, with a dated note at the top that lists every changed place.
    - Ask decisions as questions with a recommended option.
    - Propose decision-log rows as drafts; the owner adds them (P24).
    - Publish owner pages as private claude.ai artifacts, checked at desktop and phone width.
    - For blind ratings, make sure no file name, caption, alt text or order reveals a variant.

## 4. Working in the cloud container

### 4.1 A new session starts from a fresh clone

Nothing outside git reaches a new session. Each piece comes back in its own way:

- **The venv:** `tools/setup_venv.sh`.
- **The git-ignored model files and arrays:** `tools/restore_inputs.py`, which regenerates them from the committed
  records and checks each sha256.
- **Stage 1's PPO checkpoints and VecNormalize statistics:** no restore.
  - They are under `stage1/runs/2026-11-08/<body>/{ppo,policy_only}/*/checkpoints/`, about 150 MB.
  - Only retraining makes them again, and `run.json` records their steps, not their hashes.
  - Stage 2 does not need them.
- **The evaluation traces:** no restore. They are about 1.2 GB under `<body>/eval/traces/` and back the committed
  evaluation rows.
- **The clips' media:** they are in the published look pages and regenerate with `designlab.stage2.clips`.
- **The scratchpad's scripts:** the useful ones are now in `tools/`.

Without the model files the plant rebuilds models elsewhere, with another ledger. Every record key then differs, and
`eval` and `sweep` refuse the checkout (`STAGE1_PLAN.md` decision 33, "Where it runs"). Restore first, then run the
tests.

### 4.2 Restarts, keep-alive and resuming

- **The container restarts without warning.**
  - Restarts came at about 02:23 and 06:38 UTC on 11 Nov.
  - Work resumed on 14 Nov in a container that had just restarted.
  - Each time, the filesystem (checkout and scratchpad) survived and every process died.
- **Every long job is an idempotent chain that resumes from its own checkpoints.** A watchdog starts any chain that is
  neither done nor running (`stage1/runs/2026-11-10-speed/run_speed_round.sh` and its watchdog). After any pause, check
  `uptime` first.
- **Keep the container alive.**
  - Run `tools/keepalive.sh` with the Bash tool's `run_in_background`. The session then tracks it and is re-invoked
    when it ends after about 115 min; re-arm it then.
  - As a backstop, hourly scheduled tasks wake the session after a restart (create_trigger, send_later). Delete them
    when the work is done; none were active on 14 Nov.
- **Workflows resume with `resumeFromRunId`.**
  - Completed agents return their cached results, and an agent that was running starts again from the beginning.
  - Edit its prompt to add a "RESUMPTION NOTE" saying what it left behind (as in the `-X` build, which a restart cut
    off).
  - A workflow runs at most min(16, CPUs − 2) agents at once: 2 on this 4-core machine.

### 4.3 Worktrees

- **Give each builder agent its own git worktree, and name it by absolute path in the prompt.**
- **A worktree has none of the git-ignored inputs.**
  - Link or copy them in and check their sha256 (`stage2/x/x_inputs.py copy-from`,
    `stage1/runs/2026-11-10-speed/arm_k_inputs.py`).
  - Exclude the link directories through `.git/info/exclude`; the round-two refine model links showed as untracked
    files.
  - After a merge, copy the generated ignored files back the same way.
- **Do not `cd` into a worktree in the lead's own shell.** It moved the primary working directory once. Use `git -C`
  or a subshell.
- **Remove merged worktrees.** Each holds 0.2–0.5 GB; thirteen held about 3.5 GB on 14 Nov.

### 4.4 CPU, memory and disk

- **The machine.** 4 cores and 15 GB.
- **Training speed.** One R3 training job (4 envs in one process, PPO) runs about 200 env steps/s. MuJoCo is only
  14–20 % of an env step; the Python tracker and estimator are most of the rest
  (`stage2/research/research_pipeline.md` §3).
- **Priority and threads.**
  - Run side jobs through `tools/lowprio.sh`; the kernel's autogroup makes `nice` alone ineffective across shells.
  - Use `tools/memguard.sh` when training and evaluation share the machine.
  - Set `OMP_NUM_THREADS=1` in worker processes.
- **Process counts.**
  - `nproc` reads `OMP_NUM_THREADS`: `run_x.sh g1`'s old default timed 1 process instead of 4.
  - `PYTHON_CPU_COUNT` does not change `os.sched_getaffinity`, which G1 now records.
- **Disk is a fixed allowance per session.**
  - On 14 Nov about 4 GB were free, beside a 6 GB venv (torch from PyPI brings its CUDA libraries) and 1.2 GB of
    traces.
  - `TORCH=cpu bash tools/setup_venv.sh` installs the CPU-only wheel of the same torch instead.
  - Deleting files frees space at once.

### 4.5 MuJoCo, OpenGL and torch

- **MuJoCo picks its GL backend when it is first imported.** `designlab/__init__.py` sets `MUJOCO_GL` only if it is
  unset: EGL when `/dev/dri` exists, else OSMesa. This container has no `/dev/dri`.
- **With libOSMesa loaded, torch segfaults while building a stable-baselines3 policy, whichever is imported first.**
  This was seen on 11 Nov after a restart (exit 139).
- **Shared test processes and training run with `MUJOCO_GL=disable`.** `tests/conftest.py` sets it.
- **Rendering runs in a child process with `MUJOCO_GL=osmesa`.** `tests/test_stage2_clips.py` re-runs its two rendering
  tests that way.
- **The R5 and R6 entry points must set `MUJOCO_GL=disable` themselves.** Stage 1's `train.py` relies on the default
  (`STAGE2_PLAN.md` S2-5).
- **The first suite crash was GLFW asking for a display.** A test module had imported mujoco before designlab set the
  backend.

### 4.6 Tests, git and the tools

- **The full suite is `MUJOCO_GL=disable PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider tests` from this
  directory.** On 11 Nov it gave 575 passed, 107 skipped and 0 failed in the main checkout. Write pytest's own exit
  code into the log (`; echo PYTEST_EXIT=$? >> LOG`). A pass was once reported from a wrapper's exit code; it was
  corrected the same day.
- **The Bash tool's safety check can refuse a long heredoc that contains `rm`.** Write the script to a file with the
  Write tool and run the file. Use Edit or Python, not sed, for multi-line edits.
- **A stop hook requires every turn to end with all changes committed and pushed.**
  - Push with `git push -u origin <branch>`, retrying up to 4 times on network errors (2, 4, 8, 16 s).
  - Never force-push or rewrite history.
  - Commit messages end with the session's attribution lines and never name a model.
- **GitHub goes through the GitHub MCP tools.** `gh` is only for search.
  - Comment only when the owner asks, with the attribution footer.
  - Create no pull request unless asked; kuds/mesozoic-labs#667 already carries the branch.
- **Owner pages are private claude.ai artifacts.**
  - Build them with `tools/report_page.py`, or `designlab/stage2/look_page.py` for the look pages.
  - Check them with `node tools/shot_page.js` at 1280 and 390 px.
  - Republishing to the same URL needs the same file path in the same session, or the artifact's `url` (read it
    first).
- **The owner's Google Drive is readable with the Google Drive tools.** That is how Colab results come back
  (`STAGE2_PLAN.md` §6). Large binaries cannot pass through a tool call.

## 5. Traps and open issues

1. **Restore before anything else in a fresh clone** (§4.1).
2. **G1 of the `-X` bodies is provisional.** To finish it, run `bash stage2/x/run_x.sh g1` on an idle machine. Arm K's
   G1 failed its step rate on the K bodies and on the pick alike, so `-X` may fail it too (HEAD_X.md §3).
3. **The Stage 1 env assumes 14 joints.** Its action and observation widths (`designlab/stage1/env.py:138-142,
   284-286`) need generalising before R5 can drive `-X`'s neck yaw and jaw. New joints take the MJCF's default damping
   and friction, which are unsourced.
4. **`designlab/stage1/env.py` imports `environments.shared.direction_commands` from the main repository.** It adds the
   repository root to `sys.path` if needed. A Colab bundle therefore needs the repository checkout, not the study alone.
5. **`stage1/runs/2026-11-08/STAGE1_TABLES.md` still reads "works to 10 steps (20 ms)" for the 75 cm delay.** That is
   the search's own machine-made reading. The corrected reading is in S1's correction note.
6. **Duty and F3.** A fixed swing time puts the nominal walk's duty at about 0.69, over the spec's 0.65. Widening the
   band to 0.72 below 0.30 m/s is owner decision 2.
7. **The 75 cm V6 is over the knee's speed limit at the animal pace** (1.012; V6S 1.009). By the plan's rule it keeps
   its features and loses that speed rung unless the owner decides otherwise.
8. **Clearance of the new joints.** The `-X` joint ranges stop where the clearance screen allows, but the screen
   cannot see every seam (HEAD_X.md §4).
9. **Out of scope.** Eleven registry and contract tests fail on PR #662's clean checkout, and merging PR #664 conflicts
   in 4 files (`REPRODUCTION.md`). They belong to the main repository's integration, not to this study.

## 6. Next steps toward long training runs

This is `STAGE2_PLAN.md` §7 in order, with what a new session does first.

1. **Environment.** Run `tools/setup_venv.sh`, then `tools/restore_inputs.py restore` and `check`, then the full test
   suite. Compare the counts with §4.6 and with `tools/README.md`.
2. **The owner's look picks (L1).**
   - The owner picks the features, decides whether a feature may cost a speed rung, and settles the duty band
     (decisions 1, 2; `STAGE2_PLAN.md` §7).
   - Then freeze the approved reference by hash (A.5).
   - The look review page and the viewers' blind sheet are published; see `NEXT_SESSION_PROMPT.md` for their links.
3. **R5** (B.1).
   - Build: generalise the env for `-X`; add commands, the references at nominal and 0.30–0.50 m/s, turns and stops,
     the reward terms and the randomisation.
   - Gates: G8a (R5's zero action reproduces the panels' R0 exactly, at 0 and 10 ms) and G9 (the new measures read as
     designed on the references).
4. **R2L**, the tracker refitted as Stage 1b's speed fit on the bar's tasks, here (B.2).
5. **Validation on this CPU** (B.3).
   - Measure rates and memory per floor and process count.
   - Check kill and resume, and a fresh-clone dry run of the Colab bundle.
   - Run the σ and learning-rate probes.
   - Run the notebook on a tiny budget, and start a training run with `MUJOCO_GL=disable`.
6. **S2-4.** Train 2M steps here on `75-M-H-10` with training's delay draw. Colab starts only if R5's 10 ms monitor
   shows a gain over R0L; otherwise tell the owner first.
7. **S2-5, the notebook.**
   - Write `stage2/stage2_training.ipynb`, on the scaffold `notebooks/sb3_training.ipynb`.
   - Write its steps sheet and a sha256-manifested Drive bundle.
   - Then comes the owner's day-one smoke session (about an hour).
8. **S2-8.** Fix the budget once after the smoke: 40M env steps per run, or fewer if the smoke's rate cannot reach that
   in 20 h. Then run Colab's three phases (High-RAM CPU, up to 3 runtimes, 24 h sessions, a 220 runtime-hour cap,
   P29).
9. **Panels, levers and report.**
   - Run the panels here on the judged checkpoints from Drive (Track C).
   - Run the hardware levers (Track D; `STAGE2_PLAN.md` §4 D), each labelled "untrained transfer" until fine-tuned.
   - Then write the report.

**The hardware questions that matter most** are the bench tests: each drive type's command-to-motion delay and
stiffness (draft A17), the STS3215 tail servo at the speed walks' loads (A1), the treads' grip on the owner's floors
(P21), and the joints' friction and damping (A16). The delay alone decides whether anything at 60 cm walks.

## 7. Waiting for the owner

- **Stage 2's open decisions** (`STAGE2_PLAN.md` §7):
  - 1: the look features, and whether one may cost a speed rung.
  - 2: the duty band.
  - 7: raters for L1 (3–5 new viewers) and L2 (at least 12).
  - 8: when the bench tests run.
- **Draft decision-log rows, not yet in `DECISION_LOG.md`:**
  - S1 §7: F26 (with the 10 Nov correction from S1b N11), the A15 update and the A16 update.
  - S1b §8 and N11: F27, the A1 update (bench-test the tail servo first) and A17 (new: the servo delay and stiffness
    assumption).
- **P25 stays open.** After Stage 1, the owner asked what covering the remaining gap at 60 cm would take.
