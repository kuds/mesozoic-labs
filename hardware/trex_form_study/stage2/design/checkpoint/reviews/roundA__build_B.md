# build:B (roundA): the agent's final report

WP B report (every number below is smoke, not a result)

Commits on wt/wp-b (worktree clean): 5827816, 96b80df, 8b03584, 4a55ae2, 69004ea, 3cedf2a, 85f9b98. Merged: wt/wp-r (ff), wt/wp-m (c67d14e), working branch for STAGE2_75_PLAN.md (57904e2). wt/wp-s has no commits: no V0 exists.

Acceptance
1. pytest tests/test_stage2_bar.py: exit 0, "37 passed, 2 skipped in 129.21s" (skips: STAGE2_SLOW; they pass on the scratch set below).
2. run --set S1 ... R0@0ms,R2/42@0ms walk/brisk: exit 0; 4 of 4 episodes equal Stage 1's committed records in every field but wall_s, trace sha256 included.
3. reread (60-N-T-10, 40 episodes): exit 0, 40 of 40 trace sha256 equal. Both clearance readings exist for the 20 in-place-turn episodes: own-swing 0.047-0.050 H, reference-clock 0.037-0.040 H. The 20 walking-turn episodes fell (as committed), so Stage 1 judged no window.
4. V0 commands: NOT RUN (no set; no dev_episodes.json written). Substitute: scratch set D0 under logs/ (Stage 1's nominal and brisk walks under bar names, causal stop and turns built with moves.py), R0 rows only: dry-run exit 0 (writes nothing); run R0@10ms,R0@0ms x 16 gated x 2 episodes, procs 2: exit 0, 32 records, every reading present; killed at 14 records + 2 partial episodes, resumed exit 0, 0 of 32 records differ but for wall times and sessions; refcheck exit 1: stop 2 touchdowns, turns 85.45 deg at 2.0 s, clearance 0.045-0.051 H; its "v0.4" walk covers 3.68 m only because it is Stage 1's 0.351 m/s walk.
5. git status --porcelain: empty, exit 0.

Numbers: R0 about 2.0 s wall per episode (0.10 s per simulated s); R2 0.23-0.25 s per simulated s (S1 walk/brisk at 10 ms, two 8 s episodes, both fell). Recorder convention: 545 touchdown rows, none above the floor, 480 within one step's travel, the rest deeper (to -0.52 mm, soft contact); STEP_OFFSET stays 0.

Deviations
- bar.py refuses a run on V0/F25F9/F25F9cl without --task unless it is a check (V0, R0 rows, 1 episode, --dev-episodes); the brief's R2/42@10ms V0 smoke is therefore impossible, by the ruling.
- CLI additions: --set-root, --unit, --cost, --control, --supersede-reason (required), --dev-episodes/--dev-reason, --no-traces-for, condition groups @walk @rest @speed-ladder @ladders.
- Trace also holds ref_swing100 (reading (a) is re-readable from it; tested).
- Convention check "holds" is one-sided (not above the floor); the literal count is recorded.
- run_75cm.sh: modes list/cost/all; exit 3 marks a step cut and the chain continues so later units are listed as cut. Only "list" was run.

Ambiguous, decided and flagged
- Nudge length: latest onset taken at the slowest clock, (t_steady + 3T)/(0.95 x 1.05) + 5.1 s.
- Turn: drive speed, joint range and leg interference read over the episode (design 3.8c); C5's sentence could mean the window.
- stop/one_step: 2 strides before Stage 1's stop file (unspecified).
- along target: the rung name's speed (4.0 m); nominal from its entry.

Not done: any episode on a real set; barreport (WP T) is only called by the chain.

Files: /home/user/mesozoic-wt/wp-b/hardware/trex_form_study/designlab/stage2/bar.py, tests/test_stage2_bar.py, stage2/runs/2026-11-15-75cm/run_75cm.sh
