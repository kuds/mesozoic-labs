# Prompt for the next working session

Written on 14 Nov 2026 at the end of the first long session. To start the next session, open a new Claude Code session
on kuds/mesozoic-labs and paste everything inside the fence below.

- Fill in or delete the "OWNER'S ANSWERS" block at the end; the session asks for whatever is still open.
- The last line, `ultracode`, turns on multi-agent workflows with independent verification, as in the first session.
  Delete it to run without them. The first session used up the account's weekly usage limit (`LESSONS_LEARNED.md`
  §4.7).

The repository holds what the session needs beyond the prompt: `LESSONS_LEARNED.md` first, then `STAGE2_PLAN.md` and
`JETSON_PLEM_REVIEW.md`.

```text
You are continuing the Mesozoic Labs robotic T. rex form study in the repository kuds/mesozoic-labs, directory
hardware/trex_form_study/ (paths below are relative to it unless they start with "the repository root"). A first
session (7-14 Nov 2026) did the design rounds, Stage 1 (closed-loop probe), Stage 1b (speed round) and the first
phase of Stage 2 (a lifelike reference walk and an expressive head). Its work is pull request kuds/mesozoic-labs#667
(branch claude/mesozoic-trex-design-iteration-azm2fs). Pull request kuds/mesozoic-labs#668 (branch
jetson-orin-plem-study, stacked on #667, unmerged) studies the 75 cm build with a Jetson Orin Nano Super. Its results
are reviewed, not verified, in JETSON_PLEM_REVIEW.md.

GOAL OF THIS SESSION
Keep refining the configuration and hardware of the robot in simulation, understand what it can and cannot do, and get
to the point where long training runs can start on Colab. The owner asked on 14 Nov for the 75 cm build (75-M-H-10)
combined with a Jetson Orin Nano to be explored first. The 60 cm body (60-N-T-10) stays in the plan unless the owner
narrows the scope.

The milestone is "ready for the owner's day-one Colab smoke". It is reached when each item below has been reviewed
independently:
- the 75 cm reference question settled under the study's own rules (75 CM + JETSON, tasks 1-4);
- the delay bench sheet written for the owner;
- G8a and G9 passed on every body and head being trained;
- R2L fitted;
- each validation item of STAGE2_PLAN.md B.3 recorded;
- S2-4's pre-registered monitor showing a gain over R0L, or the owner told and the owner's decision recorded;
- R6's entry point built, or deferred by the owner;
- the notebook run here end to end on a tiny budget, with a kill and resume;
- the Drive bundle uploaded with its manifest and checked by a fresh-clone dry run;
- the steps sheet committed.

READ FIRST, in this order:
1. LESSONS_LEARNED.md: state, findings, method, container operations (section 4), traps (section 5), next steps
   (section 6). Sections 4 and 5 will save you hours.
2. JETSON_PLEM_REVIEW.md: the 75 cm + Jetson review, its track (section 5) and its owner decisions (section 6).
3. STAGE2_PLAN.md: the plan you are carrying out. Its owner decisions are in section 7; P28-P29 are in DECISION_LOG.md.
4. DECISION_LOG.md. Then the Answers of stage1/runs/2026-11-10-speed/STAGE1B_SPEED.md and of
   stage1/runs/2026-11-08/STAGE1_PROBE.md, with the latter's correction note at the top.
5. stage2/look/SCREEN.md ("In brief"), stage2/x/HEAD_X.md, stage2/x/X_COSTS.md, stage2/research/research_pipeline.md
   and tools/README.md.
Look features (F1-F14 of stage2/research/research_look.md) share their numbers with the decision log's F rows. Write
"look feature F3" where the two could be confused.

FIRST STEPS
1. Branch.
   - Your session names its own branch. Create it from origin/main if main already holds the study (the file
     hardware/trex_form_study/LESSONS_LEARNED.md exists there: PR #667 has been merged); otherwise from
     origin/claude/mesozoic-trex-design-iteration-azm2fs. Fetch branches by name.
   - Then merge origin/jetson-orin-plem-study into it. It adds files only, and merged cleanly on 14 Nov.
   - Push only to your designated branch, and tell the owner which branch carries the work.
   - Open no pull request unless the owner asks.
2. Keep-alive, before any long job (the restore and the test suite are the first).
   - From hardware/trex_form_study: sh tools/keepalive.sh logs/keepalive.log
   - Start it with the Bash tool's run_in_background and timeout 7200000; the default timeout stops it after 30 min.
     Re-arm it when it ends.
   - logs/ is git-ignored at any depth.
   - The container restarts without warning: files survive, processes do not. Check uptime after every pause, and
     write every long job so that it resumes from its own checkpoints.
3. Environment. The container is fresh: nothing outside git exists.
   - From hardware/trex_form_study: bash tools/setup_venv.sh (with TORCH=cpu in front: about 4 GB of disk were free in
     the first container, and the default torch build needs about 6 GB).
   - Then: PYTHONPATH=. .venv/bin/python tools/restore_inputs.py restore
   - Then the same with check. It must report every input matching.
   - If any restored sha256 differs from its record, stop and report it to the owner: every record key hashes these
     files.
   - Then the full suite, with no -q (pyproject.toml already adds one, and a second hides the counts):
     mkdir -p logs; MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python -m pytest -p no:cacheprovider tests >
     logs/pytest.log 2>&1; echo PYTEST_EXIT=$? >> logs/pytest.log
   - Compare the counts with the fresh-clone counts in tools/README.md.
   - Always use .venv/bin/python, including PY=.venv/bin/python for the run scripts. The system python has no MuJoCo.
4. Test the Drive route early (LESSONS_LEARNED.md section 4.6).
   - Up: Google Drive create_file takes a local file by reference; the owner approves each call.
   - Down: download_file_content returns base64 inline.
   - Upload a small archive and download a 2 MB .zip; check both sha256. If either fails, plan a fallback the owner
     performs.
5. Ask the owner, in one question round (the question tool, at most 4 questions per call, labelled "Q1 of N",
   recommended option first), whatever below is still open, unless it is answered at the end of this prompt:
   - the decisions in JETSON_PLEM_REVIEW.md section 6, above all compute, the real-time layer, the judged delay, the
     references, the scope (75 cm only or both bodies) and the budget;
   - the look picks for L1. The look review page is https://claude.ai/artifact/MG1yh5ztqJwVdFPs6GmkkY, and the viewers'
     blind sheet is https://claude.ai/artifact/QVktxN3pHsov2LTVs6d69c;
   - whether a look feature may cost a speed rung;
   - the duty band (STAGE2_PLAN.md section 7, decision 2);
   - raters, and when the bench tests run;
   - approval of the draft decision-log rows: STAGE1_PROBE.md section 7 (F26 with its 10 Nov correction, A15, A16),
     STAGE1B_SPEED.md section 8 and N11 (F27, A1, A17), and the joint damping and friction row proposed in
     LESSONS_LEARNED.md section 7.
   Read every free-text answer before acting on it.

75 CM + JETSON (JETSON_PLEM_REVIEW.md section 5; do these first)
1. Treat PR #668's numbers as unverified. Its 20-episode closed-loop table (its section 2.3) has no code or record.
   Its headline speeds are read at 0-2 ms of delay, and its "safe" speeds are prescribed-walk ceilings.
2. Record its references for 75-M-H-10:
   - F25 + F9, and F25_F9_cl (duty 0.59, lift 0.095 H);
   - at the nominal speed and at 0.30-0.50 m/s;
   - with starts, changes and two-step stops from the official builders;
   - each hashed and run through the Stage 0 screen (designlab.stage2.screen).
3. Run the official panels on the 0.40 m/s walk.
   - Walk F25_F9_cl, with F25 + F9 and V0 as controls, under Stage 1's R2/42 and R2/43 unchanged.
   - Seeds 3042-3061, commanded 5 % above target, mu 0.9 and 0.54.
   - Delays 0, 2, 4, 6, 8 and 10 ms (whole 2 ms physics steps).
   - Fall attribution where there are 5 or more falls, and Stage 2's readings.
   - Label the duty and lift margin as set after results.
4. Run the rest of the bar on 75 cm at 10 ms: the 90 degree turn (new references, thigh twist checked first), four
   nudges and the stop.
5. Do these after the owner's decisions:
   - a Jetson body as its own refine case, with every round-two screen;
   - a latency budget and bus plan per architecture;
   - the delay bench sheet (draft A17) on both paths;
   - the budget rebuilt with designlab/cost.py at catalog prices;
   - a ToF speed-governor table.

STAGE 2 TRACK B (STAGE2_PLAN.md section 4 B and section 7; LESSONS_LEARNED.md section 6)
- Before any run, settle the open rules in LESSONS_LEARNED.md section 5.3, labelling each change:
  - whole-step delays: the plan's 5, 15 and 25 ms cannot run as stated;
  - S2-4's reading point: a 2M-step run never reaches the plan's 2.5M-step monitor;
  - G9's 10 ms turn traces, which a fresh clone lacks;
  - how Colab checkpoints are judged here: the policy rows refuse other library versions;
  - a resume across library versions;
  - a picklable learning-rate schedule: archives segfault across Python minor versions.
- Freeze the approved look reference by hash (plan A.5) as soon as L1 is in.
- R5 (plan B.1).
  - Generalise designlab/stage1/env.py beyond its 14 joints for -X (env.py:138-142 and 284-286).
  - Add look-aware turns and transitions beside the old ones.
  - Add the commands, references, reward terms and randomisation the plan lists.
  - Pass G8a and G9 before any training.
- The R5 and R6 training entry points assign MUJOCO_GL=disable (not setdefault). With libOSMesa loaded, torch
  segfaults, and the Colab scaffold sets egl.
- R2L (plan B.2), here, with fixed delay draws per kind.
- Validation (plan B.3) and S2-4 (2M steps here on the 75 cm body); Colab only if the monitor shows a gain over R0L,
  otherwise tell the owner first.
- S2-5, the notebook: stage2/stage2_training.ipynb, built on the repository root's notebooks/sb3_training.ipynb.
  - It lives outside notebooks/, because main-repository tests scan that folder.
  - With it: a steps sheet, a sha256-manifested Drive bundle and a kill-and-resume test.
  - Runs save to MyDrive/mesozoic-labs/stage2/runs/<run_id>/.
- G1 of the -X bodies is provisional. Finish it on an otherwise idle machine: PY=.venv/bin/python bash stage2/x/run_x.sh g1

STANDING RULES (the owner's, from the decision log and the first session)
- Simulation only.
  - Keep prospective design, simulation evidence and measured results separate, in words and in files.
  - Nothing is adopted without the owner (P24): propose decision-log rows as drafts.
- Pre-registration.
  - Fix pass rules, stop rules, seeds and budgets before the runs.
  - Label any rule changed after a result.
  - Use two seeds or fits for any verdict; with one, it is provisional.
  - Judge the agility bar at 10 ms of servo delay until a bench value exists (P29); report other delays beside it.
- Independent review.
  - Have every build, report and owner page checked by an independent reviewer. Give the reviewer the code and the
    records, not your conclusion. Fix, then re-check.
  - Never report a pass you have not seen in the tool's own output.
  - Report a count only once every check behind it has returned.
- Bit-identity and provenance.
  - Existing models and references must regenerate bit for bit (sha256 tests). New behaviour goes behind options that
    default to today's.
  - Never rerun or extend Stage 1 or 1b records with newer code.
  - Before relying on a new output, run git check-ignore -v on it: the root rules ignore logs/, models/, *.zip, *.pkl
    and *.npz at any depth.
- Bounded compute.
  - This machine has 4 cores.
  - Colab is capped at 220 runtime-hours on High-RAM CPU, with up to 3 runtimes at once and 24 h sessions (P29).
  - Size workflows to the question; the account has a weekly usage limit.
- Git.
  - Commit in small steps with clear messages, ending with the attribution lines exactly as your session's system
    reminder gives them.
  - Name no model anywhere else, in a commit message or in code.
  - Push before every turn ends; a stop hook enforces it.
  - Never force-push or rewrite history.
- Talking to the owner.
  - Answer first, with plain numbers and units ("0.22 m/s (0.5 mph)"), then the evidence.
  - Never mix measured values and targets.
  - Ask with the question tool, not lettered lists.
  - "Help me think" means discussion, not new runs.
  - Correct a report in place, with a dated note.
  - Publish owner pages as private claude.ai artifacts, checked at phone width.
  - Send a push notification when long work finishes.

OWNER'S ANSWERS (optional; fill in before sending, or delete):
- Compute and real-time layer for the 75 cm build:
- Scope (75 cm only, or both bodies):
- Look picks for L1, and whether a look feature may cost a speed rung:
- Duty band below 0.30 m/s (widen to 0.72, or look feature F3 from the brisk walk):
- Draft decision-log rows approved:
- Raters, and when the bench tests run:

ultracode
```
