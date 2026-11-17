# build:CT3 (roundC, 17 Nov): the agent's final report

Working paper. Nothing here is a rule.

I did not do the CT build step. There was no merge, no file edits, no commits and no test runs. The relayed user request only asks for "a quick rundown on all you've done these last couple of days" and "an update on how soon I can start training". The computed task would turn that read-only question into merges, code changes and commits. The harness says the computed task cannot extend the request and the request wins, so I only read things. Below is what I found, for answering the user. CT's third follow-up (rulings 10 and 11 of 17 Nov) is still open and is listed under "what is left".

STATE (all read-only)
- Worktree /home/user/mesozoic-wt/wpc-ct, branch wt/wpc-ct, head 2fca27b. `git status --porcelain` is empty. No commits this round.
- Two commits are on the branch but not on the working branch: 57b5bf3 and 2fca27b. Both end with attribution lines (the earlier model's). Their content is already on the working branch as 0ad2072 and bf9bc43.
- Working branch claude/mesozoic-trex-stage2-refine-4jffre head: 32ff20b (17 Nov 01:13 UTC). It has 395 commits since 14 Nov.

RUNDOWN, 15 to 17 Nov (from git log and stage2/design/checkpoint/STATUS.md)
15 Nov:
- Round A of Stage 2 merged:
  - R: readings and record helpers
  - B: the bar's runner and judge
  - C: set screen
  - G: the one Stage 1 edit, plus the regression check
  - M: starts, stops and turns builder
  - S: three 75 cm reference sets
  - T: bar tables
  - V: turning-path and clip views (owner decision P35)
- Regression record made again on the merged code: 256 of 256 episode runs match.
- Owner decisions P32 to P36 recorded (a build chosen in simulation is the main focus; carpet side-check later).
- Policy files package (CP) merged.
- Track B rules draft started (25 rulings written in).

16 Nov:
- Owner decisions P37 to P41:
  - answers on the 75 cm report
  - usage line and "training path first"
  - staged buy gate with a harsher simulation before any purchase
  - gate 1's speed and numbers
  - gate 1's motor-robustness condition
- Servo model design note written, with two independent checks. Five questions for the maker of the leg drives are drafted. The owner has not yet said whether to send them.
- Training tooling merged, each one built, independently reviewed and fixed:
  - the training env (CE, plus two fix rounds)
  - the worker pool (CV)
  - the Drive bundle tool (CB)
  - the policy files' actions_at_bound
  - the look reading (CK, with the tail amplitude band)
  - the trainer, monitor and clip (CT, plus two follow-ups: a monitor refusal exits 8 and the trainer halts without counting it)
- Track B rules draft (STAGE2_TRACKB_PLAN.md) merged with five lists of the lead's rulings written in. The sixth list (TRACKB_RULINGS6_16NOV.md) is ruled but not written in.
- Work paused at 97.3 % of the usage allocation. Cut mid-run:
  - rows package CW (fix round)
  - notebook and steps sheet CN (fix round)
  - look sets L (build)

17 Nov 01:13 UTC:
- The session resumed on the new model. The lead wrote ROUND_C_RULINGS_17NOV.md: fix rulings for CN (Drive copy-back, what is final versus resumable, the resume-from-0 guard), for CW (a rule text lost from row descriptions, minor points, warned episodes count as falls), and CT's third follow-up.
- The cut work is still uncommitted:
  - wpc-cw (9952f5b): 4 modified files (lookbar.py, s2rows.py and their two tests)
  - wpc-cn (b54c0aa): clean, 12 commits ahead
  - wp-l (b72c61c): 17 commits ahead, plus an untracked stage2/sets/V6/60-N-T-10/. V6noF5 on 75-M-H-10 is built and screened but not frozen.

HOW SOON TRAINING CAN START (my reading; nothing measured)
No training run, not even a smoke, can start yet. Work still to do:
1. Code still to finish:
   - CW fix round, then merge
   - CN fix round, then merge
   - CT third follow-up (rulings 10 and 11: pass --allow-source-change to the row; s2rows.RowError counts as a refusal; missing run.json, a malformed checkpoint record or a fingerprint mismatch count as refusals; the --retry-check halt text; a SIGKILL-around-refusal test)
   - the look sets: finish the build, then review, then the lead freezes V6
   - the R2L refit tool (CF): build, review, fix
   - the gates (CG): build, review, fix
   - write the sixth list into the Track B draft
2. Then the run plan in ROUND_C_DESIGN.md section 6, V0 to V13. The design's own estimates:
   - validation plus S2-4 (the first real local R5 training, 2M steps, about 50 to 65 min wall): about 6.5 core-hours
   - 60 cm local run: 5.2 core-hours
   - R2L: 8 core-hours, about 4 h wall
   - bundle, upload (the owner approves each part), acceptance run (about 12 min), steps sheet
3. Only then can the owner run the day-one Colab smoke from the steps sheet.

My estimate, not a plan figure: about 1 to 2 more working sessions of build, review and fix for the packages in item 1, if the usage allocation allows. Then about a day of validation on the shared 4 cores. So the first local training run (S2-4) is at least a day or two away, and the owner's Colab start comes after the acceptance run.

Open with the owner:
- whether to send the five maker questions
- gate 1 condition 4, and the numbers for gates 2 and 3
- draft decision-log rows F28, F29, E and F30
- how much of the new model's allocation is left (meter zero taken as $2,701.94)

Files:
- /home/user/mesozoic-labs/hardware/trex_form_study/stage2/design/checkpoint/STATUS.md
- /home/user/mesozoic-labs/hardware/trex_form_study/stage2/design/ROUND_C_RULINGS_17NOV.md
- /home/user/mesozoic-labs/hardware/trex_form_study/stage2/design/ROUND_C_DESIGN.md (sections 5 and 6)
- /home/user/mesozoic-labs/hardware/trex_form_study/stage2/design/checkpoint/reviews/roundC__review_followups2.md

For the lead: to run the CT third follow-up, send it with a user request that asks for that work. I left it undone on purpose.
