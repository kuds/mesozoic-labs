# Prompt for the next working session

Written on 14 Nov 2026 at the end of the first long session. To start the next session, open a new Claude Code session
on kuds/mesozoic-labs and paste everything inside the fence below.

- Fill in or delete the "OWNER'S ANSWERS" block at the end; the session asks for whatever is still open.
- The last line, `ultracode`, turns on multi-agent workflows with independent verification, as in the first session.
  Delete it to run without them.

What the session needs beyond the prompt is in the repository: `LESSONS_LEARNED.md` first, then `STAGE2_PLAN.md`.

```text
You are continuing the Mesozoic Labs robotic T. rex form study in the repository kuds/mesozoic-labs, directory
hardware/trex_form_study/ (all paths below are relative to it). A previous session (7-14 Nov 2026) did rounds one and
two of the design study, Stage 1 (closed-loop probe), Stage 1b (speed round) and the first phase of Stage 2 (a
lifelike reference walk and an expressive head). Its work is on the branch claude/mesozoic-trex-design-iteration-azm2fs,
in pull request kuds/mesozoic-labs#667 (https://github.com/kuds/mesozoic-labs/pull/667), open and not merged.

GOAL OF THIS SESSION
Keep refining the configuration and hardware of the two Stage 2 bodies in simulation, understand what they can and
cannot do, and get to the point where long training runs can start on Colab. The bodies are 60-N-T-10 (60 cm Tapered)
and 75-M-H-10 (75 cm Hybrid), each also with the expressive head (-X: neck yaw and jaw). The plan is STAGE2_PLAN.md,
Track B.
This session's milestone is "ready for the owner's day-one Colab smoke":
- R5, the imitation policy under commands, built and through its gates G8a and G9;
- the R2L baseline fitted;
- validation on this CPU done (plan B.3), including S2-4, the 2M-step run on 75-M-H-10;
- stage2/stage2_training.ipynb ready, with its steps sheet and a sha256-manifested Drive bundle.
If the owner's look picks (gate L1) have not arrived, do everything that does not depend on them first, and ask.

READ FIRST, in this order:
1. LESSONS_LEARNED.md: where things stand, what the robot can and cannot do in simulation, method lessons, how to
   work in this cloud container, traps, and next steps. Sections 4 and 5 will save you hours.
2. STAGE2_PLAN.md: the plan you are carrying out. Owner decisions are in its section 7, and P28-P29 in DECISION_LOG.md.
3. DECISION_LOG.md. Then the Answers sections of stage1/runs/2026-11-10-speed/STAGE1B_SPEED.md and of
   stage1/runs/2026-11-08/STAGE1_PROBE.md, with the latter's correction note at the top.
4. stage2/look/SCREEN.md ("In brief"), stage2/x/HEAD_X.md, stage2/x/X_COSTS.md and
   stage2/research/research_pipeline.md.
5. tools/README.md.

FIRST STEPS
1. Branch.
   - Your session names its own branch. Create it from origin/claude/mesozoic-trex-design-iteration-azm2fs (fetch that
     branch by name) so that it holds all the previous work, unless you are told to continue on that branch itself.
   - Push only to your designated branch, and tell the owner which branch carries the work.
   - Open no pull request unless the owner asks.
2. Environment. The container is fresh: nothing outside git exists yet.
   - Run: bash hardware/trex_form_study/tools/setup_venv.sh (with TORCH=cpu in front if disk is short).
   - Then, from hardware/trex_form_study: PYTHONPATH=. .venv/bin/python tools/restore_inputs.py restore
   - Then the same with check. It must report every input matching.
   - If any restored sha256 differs from its record, stop and report it to the owner: every record key hashes these
     files.
   - Then the full suite: MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python -m pytest -q -p no:cacheprovider tests >
     LOG 2>&1; echo PYTEST_EXIT=$? >> LOG
   - Compare the counts with tools/README.md.
3. Keep-alive.
   - Before any job longer than a few minutes, start tools/keepalive.sh with the Bash tool's run_in_background, and
     re-arm it when it ends.
   - The container restarts without warning: files survive, processes do not. Check uptime after every pause, and
     write every long job so that it resumes from its own checkpoints.
4. Ask the owner, in one question round with recommended options first, whatever below is still open, unless it is
   answered at the end of this prompt.
   - The look picks for L1. The look review page is https://claude.ai/artifact/MG1yh5ztqJwVdFPs6GmkkY, and the
     viewers' blind sheet is https://claude.ai/artifact/QVktxN3pHsov2LTVs6d69c.
   - Whether a look feature may cost a speed rung.
   - The duty band (STAGE2_PLAN.md section 7, decision 2).
   - Raters for L1 and L2.
   - When the bench tests run.
   - Approval of the draft decision-log rows:
     - STAGE1_PROBE.md section 7: F26, with its 10 Nov correction, and the A15 and A16 updates;
     - STAGE1B_SPEED.md section 8 and N11: F27, the A1 update and the new A17.

THE WORK, in the plan's order (STAGE2_PLAN.md section 4 B and section 7)
- Freeze the approved look reference by hash (plan A.5) as soon as L1 is in.
- R5 (plan B.1).
  - Generalise designlab/stage1/env.py beyond its 14 joints for -X (env.py:138-142 and 284-286).
  - Add the commands and the references: nominal and 0.30-0.50 m/s walks, walking turns, the turn in place, starts,
    two-step stops and changes.
  - Add the reward terms and the randomisation the plan lists.
  - Pass G8a and G9 before any training.
- Set MUJOCO_GL=disable in the R5 and R6 training entry points. With libOSMesa loaded, torch segfaults.
- R2L (plan B.2), here: the tracker refitted as Stage 1b's speed fit on the bar's tasks, current heads, one seed per
  body.
- Validation (plan B.3) and S2-4: 2M steps here on 75-M-H-10.
  - Colab starts only if R5's 10 ms monitor shows a gain over R0L.
  - Otherwise tell the owner first.
- S2-5, the notebook: stage2/stage2_training.ipynb, on the scaffold of notebooks/sb3_training.ipynb at the repository
  root.
  - It goes outside notebooks/, because main-repository tests scan that folder.
  - With it: its steps sheet, a sha256-manifested Drive bundle and a kill-and-resume test.
  - Runs save to MyDrive/mesozoic-labs/stage2/runs/<run_id>/, which you read with the Google Drive tools.
- G1 of the -X bodies is provisional. Finish it on an otherwise idle machine: bash stage2/x/run_x.sh g1
- Hardware: Track D (model only) comes after a judged R5 exists. Meanwhile, if it helps, draft a bench-test plan the
  owner could run with a few servos. Its items: servo delay and stiffness (A17), the STS3215 tail at the speed walks'
  loads (A1), tread grip on the owner's floors (P21) and joint friction and damping (A16). The servo delay alone decides
  whether anything at 60 cm walks: at 10 ms nothing does in the model.

STANDING RULES (the owner's, from the decision log and the first session)
- Simulation only.
  - Keep prospective design, simulation evidence and measured results separate, in words and in files.
  - Nothing is adopted without the owner (P24): propose decision-log rows as drafts.
- Pre-registration.
  - Fix pass rules, stop rules, seeds and budgets before the runs.
  - Label any rule changed after a result.
  - Use two seeds or fits for any verdict; with one, it is provisional.
- Independent review.
  - Have every build, report and owner page checked by an independent reviewer. Give the reviewer the code and the
    records, not your conclusion. Fix, then re-check.
  - Never report a pass you have not seen in the tool's own output.
- Bit-identity. Existing models and references must regenerate bit for bit (sha256 tests). New behaviour goes behind
  options that default to today's.
- Bounded compute. This machine has 4 cores. Colab is capped at 220 runtime-hours on High-RAM CPU, with up to 3
  runtimes at once and 24 h sessions (P29).
- Git.
  - Commit in small steps with clear messages that end with your session's attribution lines.
  - Push before every turn ends; a stop hook enforces it.
  - Never force-push or rewrite history.
  - Never name a model in a commit or in code.
- Talking to the owner.
  - Answer first, then give the evidence.
  - Never mix measured values and targets.
  - Correct a report in place, with a dated note.
  - Publish owner pages as private claude.ai artifacts, checked at phone width.

OWNER'S ANSWERS (optional; fill in before sending, or delete):
- Look picks for L1:
- May a look feature cost a speed rung:
- Duty band below 0.30 m/s (widen to 0.72, or F3 from the brisk walk):
- Draft decision-log rows approved:
- Raters, and when the bench tests run:

ultracode
```
