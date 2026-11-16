# build:CT (roundC, wave 3): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

WORK PACKAGE CT (the trainer): BUILD REPORT

Worktree /home/user/mesozoic-wt/wpc-ct, branch wt/wpc-ct, study dir /home/user/mesozoic-wt/wpc-ct/hardware/trex_form_study. Simulation only; nothing here is a result. No episode on a look set or a 75 cm panel set; everything real ran on the Stage 1 view (S1) of 75-M-H-10 or on a stub env. Nothing pushed, no history rewritten, main checkout untouched.

1. MERGES OF THE WORKING BRANCH (four, all `git merge --no-ff -F <file>`, no conflict in any; none touched a file of this package)
- 6c64b00: the env's fix round (7b4df13, 67cb6f5) and ROUND_C_RULINGS' 15:40 section. After it I dropped my fallback for the old env: the trainer reads the env's own keys (task_names, tasks, task_files, set_freeze_started) and requires configs that name their tasks.
- 69c3c17: servo-model papers.
- 3d44abf: 9417ca9 (STAGE2_TRACKB_PLAN.md on the working branch, TRACKB_RULINGS5_16NOV.md, the bundle/s2load follow-ups). I then built the fifth list's rulings 1 and 2 (below).
- bcaea22: fb421ba (maker questions; papers only).
Registered hash line after the last merge: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (unchanged).

2. WHAT I BUILT (five new files, no other file changed)
- designlab/stage2/train2.py (1,801 lines). `train(...)` = one job dir `<out>/jobs/<job>`: run.json (s2-run-1), updates.jsonl, progress.json, gates/g8a.json, gates/rate.json, checkpoints/, monitor/. PoolVecEnv with `train2 worker` as worker entry; PPO exactly as 3.6's block; every checkpoint a boundary (seed 100,000 s + 100 b, reset, returns zeroed, last obs set); run state in each checkpoint record; resume in the rule's order with nothing written before the last refusal; checks with attempts logged before the monitor child starts; `monitor2.decide` on records; fix applied in place by `env_method("set_task_weights")`; optimizer-step counter (lesson 16) and `kl_cut`; fine-tune (K8); `probe_lr`, `probe_sigma`, `filter_choice`; `preflight`; `sources()`; CLI `run | probe-lr | probe-sigma | filter | preflight | worker ADDRESS | monitor ...`. Tests: tests/test_stage2_train2.py (33 fast, 3 slow): `test_a_killed_run_resumes_bit_for_bit` (SIGKILL at 896 steps between boundaries, 2 workers, resumed with 1: policy tensors, optimizer state, obs_rms incl. count at the first checkpoint after the resume, ret_rms, fingerprint, updates.jsonl equal), `test_kill_during_the_monitor` (external killpg while the monitor child runs), `test_kill_after_the_fix`, `test_resume_when_run_json_is_ahead_of_the_checkpoint`, `test_restart_from_zero_is_the_fresh_run`, `test_a_resume_from_checkpoint_zero`, `test_resume_refusals_{settings,inputs_and_a_later_freeze,sources_and_versions,checkpoints}`, `test_minibatch_steps_are_counted`, `test_kl_cut_flag`, `test_the_trainer_assigns_mujoco_gl`, `test_an_archive_saved_by_the_trainer_has_no_bytecode`, `test_the_schedule_follows_the_step_count_across_a_resume`, the M1 attempt tests, G8a fail (exit 3), second G8b failure stops, failed baseline halts, failed set_task_weights ends the session, probe_lr and fine-tune on the stub; slow: `test_tiny_r5_run_resumes`, `test_tiny_r6_run_resumes`, `test_tiny_probes_on_the_s1_view`.
- designlab/stage2/monitor2.py (835 lines). CONDITIONS, SEEDS (6001-6018), S1_CONDITIONS; `run`, `baseline` (records monitor_<n>.json, r0l.json; a baseline holding an error is kept as r0l.failed.<i>.json and raises); `survival_credit`, `gain`, `g8b`, `fix_weights`, `decide`, `s2_10`, `no_record`, `is_record`, `baseline_equal`; CLI `run | baseline | s2-10`. One indirection `_episodes(body, set, row, pairs, policies, traces)` -> `s2rows.configure`, `lookbar.rows`, `lookbar.episodes` (imported by name inside it). Tests: tests/test_stage2_monitor2.py (21): every item of section 4's row "Monitor, G8b, stop rules", plus "not run", the sources refusal, the baseline rule.
- designlab/stage2/clip2.py (189 lines). `--trace --body --set --out [--seconds 3] [--view] [--strict]`; `assign_gl("osmesa")`, no torch; reuses barviews.pose_of/draw_frame and clips.encode; shows the trace's last seconds; writes `<out stem>.json` in every case; exit 0 unless --strict. Tests (in test_stage2_monitor2.py): a real 1 s clip from a Stage 1 R0 trace rendered in a child (passed here: OSMesa and ffmpeg present), and a missing trace -> record with reason, exit 0. I looked at one frame: robot in frame, labels on.

3. ACCEPTANCE (final tree bcaea22; env line as the standing rules; all through `bash tools/lowprio.sh`, which runs nice -n 19 instead of the brief's nice -n 15)
- `$PY -m pytest -p no:cacheprovider tests/test_stage2_monitor2.py` -> exit 0, `21 passed in 16.57s`
- `$PY -m pytest -p no:cacheprovider tests/test_stage2_train2.py` -> exit 0, `33 passed, 3 skipped, 1 warning in 150.92s (0:02:30)` (the warning is SB3's batch-size note in the stub probe)
- `STAGE2_SLOW=1 $PY -m pytest -p no:cacheprovider tests/test_stage2_train2.py -k "tiny"` -> exit 0, `4 passed, 32 deselected in 262.19s (0:04:22)`. This holds the tiny R5 run killed (SIGKILL of the process group) after checkpoint 1,024 and resumed, equal bit for bit to the uninterrupted tiny run at 4,096 steps (2 workers), and the tiny R6 run killed and resumed.
- `$PY -m designlab.stage2.train2 run --mode R5 --body 75-M-H-10 --seed 42 --set S1 --tiny --workers 2 --out <scratch>/ct/tiny` -> exit 0; last line `R5-75-M-H-10-s42: done at 4096 steps`; run.json status 'done', num_timesteps 4096.
- A tiny R6 run by hand through the CLI (before the last two commits): killed with `kill -9` of its group at checkpoint 1,024 (0 surviving processes), `--resume` -> exit 0, `resuming session 1: R6-75-M-H-10-s42 from 1024 steps`, `done at 4096 steps`, fingerprint difference 0.0, updates 512..4096 each once.
- `git status --porcelain` -> empty.
- Rates of the tiny R5 run at 2 workers (three runs, machine load 1.5-3 from other jobs, so provisional): first rollout 416-453 env steps/s; first three rollouts (two updates in) 386-448. Resident memory: trainer 517-526 MB; one worker (4 envs) 524-544 MB.
- Outputs: tests under tmp_path; my runs under /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/ct/ (outside the repository, so `git check-ignore` answers "outside repository"). The designed in-repo place is ignored: `git check-ignore -v stage2/tb/work/ct/jobs/x/run.json` -> `stage2/tb/.gitignore:1:work/`.

4. FINAL STATE
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
 hardware/trex_form_study/designlab/stage2/clip2.py | 189
 hardware/trex_form_study/designlab/stage2/monitor2.py | 835
 hardware/trex_form_study/designlab/stage2/train2.py | 1801
 hardware/trex_form_study/tests/test_stage2_monitor2.py | 674
 hardware/trex_form_study/tests/test_stage2_train2.py | 1388
 5 files changed, 4887 insertions(+)
`git status --porcelain`: empty. The working branch's tip (fb421ba) is an ancestor of HEAD.

5. FOR THE LEAD
Readings I chose (each could be read otherwise):
a. Settings are never changed by a fix. Design 3.6 says the settings' env configs are replaced by the fixed ones. I kept the settings and their hash for the run's life (K4 and the loader need every checkpoint's settings hash to be the run's); the fix record holds weights before/after and the fixed configs; weights in force are run state.
b. G8b's second reading is at the check of the multiple 10M (rule M4 as corrected), not design 3.6's "fix + 5M", and only when the first reading failed. A failure at 10M after a pass at 5M has no consequence.
c. The check on the final checkpoint applies neither a fix nor a stop: the budget ended the run. They are recorded in run.json `final_check` (`fix_not_applied`, `would_have_stopped`); status stays "done". For the local 60 cm run this means its 5M reading is in `gates.g8b` and no `fix` is recorded.
d. Resume order: inputs are checked by building the envs from the stored configs before sources and versions, so a refused resume costs the env build; nothing is written before the last refusal.
e. On a resume every argument given must be the run's; `--steps` and `--lr` may be left out, but non-None defaults are compared (S2-4's `--monitor-every 500000` must be repeated).
f. `--no-gates`: no G8a short form, no baseline, no check; it is a setting.
g. Learning rate without `--lr`: rule O3 as settled (one rate per body, smallest over its probed heads), read from stage2/tb/validation/lr_probe.json `rates_by_body`; the brief's and design's single value is also in the record (`value`). Without a record a tiny run takes 3e-5, any other run is refused. filter.json's choice is read the same way.
h. Probes: every rate tried runs both sigmas, each cell all 6 updates. K8's probe uses the parent's own sigma and n_steps, rates p then 1e-4, 3e-5, 1e-5, 3e-6 below p.
i. `ep_len_mean`/`ep_rew_mean` are over the episodes that ended in that update's rollout (null if none): SB3's 100-episode window would differ after a resume.
j. gates/rate.json leaves out the seconds spent at later rollout starts (checkpoint, check, boundary).
k. SOURCE_FILES taken literally (all 18 Track B modules, bar, readings, setio, s2io, motion, four repository files). Six are not on the branch yet (lookread, lookbar, s2rows, gates2, r2lfit, nb) and are recorded as absent; when they arrive every running job sees "other sources".
l. `monitor_every` must be a multiple of `checkpoint_every` (refused otherwise).

Beyond the frozen API (each needed by a rule or ruling later than the brief):
- `--config JSON` on `run` (the servo-model ruling: any env config field passes through).
- `--retry-check` / `train(retry_check=)` (fifth list, 2): after the halt for two never-closed attempts, one more attempt per resume, logged `after_halt`, recorded in the session.
- `monitor2.run/baseline(sources_sha256=)`, `baseline(job_dir=)`, CLI `s2-10 --committed-baseline`.
- Exit codes 4 (refused: nothing was run) and 5 (halt for the lead: baseline could not be made, or two never-closed attempts; reason in run.json `halt`) besides 0, 3, other. The launcher should not retry 4 or 5.
- A fifth episode value "not_run" accepted from lookbar (fifth list, 1).

A stand-in you should rule on: lookbar is not on the branch, and under rule M1 a run cannot start without a baseline, so the plain CLI tiny run could not reach "done". In `monitor2._episodes`, only when `designlab.stage2.lookbar` cannot be imported and only on set S1, `_stand_in` runs the training env itself on the stock condition's plan (zero action for R0L, the checkpoint's deterministic actions otherwise). Nothing is judged there (every reading False, N always 0), and each record says so in `episodes_by`. On any other set a missing lookbar fails the monitor. Once CW is merged it is never reached; delete it or keep it, your call.

Not built / not done:
- Rule M3's "S2-4's r0l.json must equal the committed baseline, else the run stops before its first rollout": the trainer has no input for the committed file. `monitor2.baseline_equal` and `s2-10 --committed-baseline` record it afterwards. A flag such as `--expect-baseline PATH` would do it.
- monitor2 against the real `lookbar.episodes` and `s2rows.configure`: tested only with a stub of the brief's schema.
- `g8a.short`/`bar_form` through the trainer (only the S1 stock form ran); probes on a real set (no validation record written); fine-tune on the real env; `-X` bodies; worker counts above 2 on the real env; clip2 on a policy row's trace.
- The monitor child has no timeout (the design says the trainer waits).
- Commit e52b153's title says "and their tests"; the monitor's tests are in 71c30ca.
- I ran only my two test files, as the brief says; s2load changed in the last merge and my tests exercise it.

Observations:
- The tiny run's final checkpoint record reads `fingerprint.clipped_rows: 256` of 256 (`actions_at_bound: 0`): after 4,096 steps the normalisation clips every fingerprint row in at least one actor column. Worth a look on a long run before relying on the fingerprint's strength.
- R0L's S differs from its plain mean survival in the 13th digit (float division in the credit formula); the rule's "R0L exactly" holds only to rounding. It also needs `clock.t_ref_reached` to be the reference time at the instant `survival_s` is read.

For the next packages:
- CW: `_episodes` calls `s2rows.configure({"R5/42": {"job_dir", "steps"}})`, `lookbar.rows(body, ["R5/42@10ms"])`, `lookbar.episodes(body, set, rows, pairs, traces=<job>/monitor/traces/<n>)`; one dict per pair, in the pairs' order. On S1 the pairs are walk/nominal 6001 and walk/brisk 6004. Optional keys `start_seed` and `state_digest` are used when present. R6's span is in run.json `sessions[-1].env.core.span`; the env configs in `settings.env_configs`.
- CG: `train(...)` takes `checkpoint_every` (K3's 32,768) only as a function argument; `train2.sources()`; `gates/rate.json` per session.
- CN: the trainer refuses without PYTHONHASHSEED=0 or one BLAS thread (exit 4); progress.json (s2-progress-1) holds steps, budget, status, env_steps_per_s, eta_h, last_checkpoint, last_monitor, g8a, g8b, fix, flags, stopped, minibatch_steps_last; clip traces are `<job>/monitor/traces/<n>/<row>/<behaviour>/<condition>/<seed>.npz`; clip2 writes `clip_<n>.json` beside the mp4.

6. COMMITS (oldest first)
6c64b00 Merge the working branch: the env's small fix round and the rulings added after it
e52b153 the monitor's rules and records (monitor2)
f514780 the trainer of R5 and R6 (train2) and its tests on a stub env
71c30ca one clip from a monitor trace (clip2); the monitor's and the clip's tests
0e1c3f5 the trainer builds on the merged env; sessions hold what their envs execute
b2587aa the rate leaves rollout-start work out; tests of the CLI and of the real-env probe
fba113a the monitor's records hold the library versions and where their traces are
69c3c17 Merge the working branch: the servo model's papers
3d44abf Merge the working branch: the Track B rules draft, the fifth list of rulings, the follow-ups
d6f820a the lead's fifth list of rulings and rule K4 as it now stands
bcaea22 Merge the working branch: the questions for the maker of the leg drives
Every message ends with the two attribution lines and names no model elsewhere.
