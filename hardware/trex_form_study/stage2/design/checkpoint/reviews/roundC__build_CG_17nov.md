# build:CG (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

## Report: work package CG (gates and validation records), worktree /home/user/mesozoic-wt/wpc-cg, branch wt/wpc-cg

I built and tested all three files. No record was written under `stage2/tb/gates/` or `stage2/tb/validation/`, nothing ran on a look set, nothing was pushed, and the main checkout was not touched.

### 1. Merge of the working branch
- The working branch moved from a8f62e12 to fccebe42 while I worked. The only change was `stage2/design/checkpoint/STATUS.md`.
- I merged it as 8069eda9. There were no conflicts. The message was written to a file and passed with `-F`, and it ends with the two attribution lines.

### 2. What was built
All code is in `/home/user/mesozoic-wt/wpc-cg/hardware/trex_form_study/designlab/stage2/gates2.py` (with `tests/test_stage2_gates2.py` and `stage2/tb/run_trackb.sh`). Each gate writes one record with its rule text, the hashes of its inputs, sources, versions, commit, dirty (its own outputs left out), time and its own sha256. Exit codes: 0 pass, 1 fail, 2 error or refusal.

- **Rule X1 refusal** (`require_g9_ref`): `g8a`, `rates`, `kill-resume` and `r6-start` refuse a look set unless `g9_ref.json` is a valid, passed record of that set, body and the set's latest freeze.
  - Test: `test_closed_loop_gates_refuse_a_look_set_without_a_passed_g9_ref` (missing, failed, other freeze, other body, edited record).
- **`g8a`** (Z1): uses `g8a.bar_form` with `G8A_CONDITIONS` (10 keys), seeds 3042 and 3043, delays 0 and 5. On S1 it uses the stock form. `differences` names condition, seed, delay and first differing step; a run with fewer conditions, seeds or delays is marked `complete: false`.
  - Tests: S1 record and exit codes (a forced 1e-6 action gives exit 1 and names the step); the bar form on synthetic T0 with a condition not run; the 10 keys equal `g8a.conditions` on T0 and V0.
- **`g9_ref`** (Z2, kinematic only):
  - (a) cites round A's refcheck: the one the latest freeze cites, else the set directory's if its files match, else `bar.refcheck` is run and its record written beside the gate's.
  - (b) ladder rungs against `metrics.json` and `L1_FREEZE.json`; other walks must be finite with at least 4 steps; an `-X` body or V6noF5 is compared with a comparand set. Every walk and turn item's window is listed with steps, lift-offs, its reference toe-off and its side of 0.02 rad (ruling 7).
  - (c) uses `lookread.controls` plus the fifth control from ruling 2, `tail_tenth`, which fails the tail alone (0.012 rad, under the 0.02 rad band). Flat feet only on V6.
  - (f) cites `screen.json` for references no refcheck item plays; for every task it records the joint jump at each join and checks 2 s of `R5Tracker.zero_command` targets are finite.
  - Tests: (c) on the V6 ladder arrays; fr05's flat feet cannot fail; the join rule on tables and on S1; the whole gate on V0 (record in tmp_path); (a) running refcheck on T0; (b) against a comparand.
- **`g9_run`** (d, e): reads the R0L@10ms companions and the bar records they name (sha256 checked). Checks each episode's readings against its record's terms, finite look numbers where the window was reached, and both toe-clearance readings in every 10 ms record. (e) cites a reread record for 60-N-T-10 and its `-X` head.
  - Test: synthetic companions over the committed V0 R0@10ms walk/v0.4 record, with each failure type injected.
- **`rates`**: `r5env.throughput` per floor, then 8 envs at 1-4 workers with a fresh PPO policy collecting rollouts and no update; CPU and resident memory of parent and each worker; `provisional` unless the 1-minute load was under 0.5.
  - Test: stub env, 2 workers.
- **`kill_resume`** (K3): two `train2` runs through a `_train` child, a SIGKILL of the whole process group when `progress.json` reads 81,920, a resume with 3 workers. Compared: policy, optimizer and torch-variable tensors, VecNormalize statistics (via `s2load.load_vecnormalize`; no `pickle.load` outside s2load), and every fingerprint array. Reports the first difference with index and both values. It is resumable, remakes an interrupted run, and refuses while a run's trainer is alive.
  - Tests: stub env pass, resumable, perturbed-tensor report, a kill not as ruled is redone.
- **`gl_start`**: `train2 preflight` in a child started with `MUJOCO_GL=osmesa`. Test included.
- **`r6_start`**: R6Env with 200 zero and 200 random steps, a mid start, then a tiny R6 run of 2 rollouts. Slow test on S1.
- **`dev-episode`**: rule X1's list, keyed by step, set and body. Test included.
- **`run_trackb.sh`** (POSIX sh): round A's step pattern with markers under `stage2/tb/work/done/`, V1-V10, `nb collect` after the long jobs, X1 listing for V4, V5, V6 and V7's kill-resume. V0, V0b and V11-V13 are comments naming the lead.
  - Tests: `sh -n`; dry run prints 45 steps in order and writes nothing; markers make a second run a no-op; a failing step stops the chain with no marker; `git check-ignore` on live and record paths.

### 3. Test summary lines and acceptance outputs
- `nice -n 15 $PY -m pytest -p no:cacheprovider tests/test_stage2_gates2.py`: **29 passed, 1 skipped in 75.19s (0:01:15)**, exit 0.
- `STAGE2_SLOW=1 … -k r6_start` (run before the last two tests were added): **1 passed, 27 deselected in 22.20s**.
- `TB_DRY=1 sh stage2/tb/run_trackb.sh`: exit 0, 45 steps, ends "dry run (TB_DRY): nothing was run or written". `sh -n`: exit 0.
- Record-key hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered pair).
- Command smokes (outputs in the scratchpad's `wave5/cg/`, none is a record):

| Command | Result | Exit |
|---|---|---|
| `g9-ref --set V0` | (a) True, (b) False, (c) False, (f) False: FAIL, as expected on V0 (its files are not the V6 pick, V0's head pitch RMS 2.03 deg is over the 1.5 deg band, joins over 1e-6 rad); 17 s | 1 |
| `g8a --set S1 --seeds 3042 --cut-s 1.013` | 4 of 4 identical | 0 |
| `gl-start` | PASS | 0 |
| `rates --set S1 --workers 1,2 --steps 2048` | W=1 320, W=2 522 env steps/s; provisional (load 1.76) | 0 |
| tiny `kill_resume` on S1, real env, workers 2→1 (2-worker limit) | pass; polled kill at 2560; resumed from 2048; checkpoints 3072 and 4096 equal | 0 |

### 4. Worktree state
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
```
 .../trex_form_study/designlab/stage2/gates2.py     | 1867 ++++++++++++++++++++
 hardware/trex_form_study/stage2/tb/run_trackb.sh   |  229 +++
 .../trex_form_study/tests/test_stage2_gates2.py    |  832 +++++++++
 3 files changed, 2928 insertions(+)
```
`git status --porcelain`: empty.

### 5. Commits on wt/wpc-cg, oldest first
- f84f1ced (gates2 and the chain)
- 3a0983a0 (tests)
- 8069eda9 (merge of the working branch)
- b13bcd4c (kill-resume guard, rate and filter in its record, two tests)

Head: **b13bcd4c**.

### For the lead (decisions needed)
1. **V3 cannot run as built.** The bar's runner refuses any row whose controller is not R0 on a set outside the 75 cm round's three (`bar._check_rule`, `RULES["checks"]`). So `lookbar run --rows R0L@10ms --set V6` is refused, and so would R5, R6 and R2L rows be on the look sets. `bar.py` is round A's and part of a record key, so changing it moves `bar_sources`. This needs your ruling.
2. **G9-ref (f)'s 1e-6 rad join rule fails on Stage 1's own arrays and on V0.** The jumps are at the joins from a walk into a move: S1 walk→stop 3.77e-5 rad, brisk→brisk_to_nominal 1.11e-4; V0 up to 5.33e-5. I built the rule as written, with no tolerance. If V6's moves join the same way, (f) fails on V6. The alternatives are a limit tied to one frame's motion or moves that start exactly on the walk's frame; either is a rule change and must be labelled.
3. **My readings of the rules** (each is marked in the rule texts after `MODULE_READING`):
   - (b) plays each walk alone over the reference times [2.0, 12.5] s. A rung's walk is found by `ladder_twin`, else by name. The fr05 rung, which the set does not hold, is listed but does not fail; a missing nominal or v040 rung fails.
   - (a): a failing walk/nominal or walk/v0.4 item blocks the body. Any other failing item's reference is listed as not available, and G9-ref can still pass.
   - G9-run: "both toe-clearance readings" means the own-swing reading and Stage 1's reference-clock reading. A record where no episode reached its window is listed, not failed.
   - K3: both runs have gates off and `monitor_every` 131,072. The rate and filter come from `lr_probe.json` and `filter.json`. A run interrupted any other way is redone.
   - `rates` uses 16,384 env steps per floor as well; the brief gives no count for that part.
   - r6-start and gl-start run on S1, as X1 rules; the run plan names V6 for r6-start.
4. **The chain's choices:**
   - V4 is placed after V7, before S2-4. The alternative is after V10.
   - V9 is not listed in `dev_episodes.json`; I treated it as a run a rule names (M6). The alternative is to list it.
   - V8 runs `train2 run` directly, not `nb launch`. Your rewrite to the notebook's lines (ruling 6) would replace it.
   - V8 makes R0L_BASELINE.json with the `monitor2.baseline` function, because its CLI requires `--job`.
   - V10 calls `r2lfit` (CF), which is not on the branch yet.
   - G9-run (e) needs `TB_REREAD`, round A's reread record, which is not committed. Without it the 60 cm steps fail.
5. **The trainer builder's four open points** (none can be built in my files):
   - `--expect-baseline` (M3): the chain checks S2-4's baseline after the run with `s2-10 --committed-baseline`, not before its first rollout. I recommend the trainer gain `--expect-baseline PATH` and `monitor2 baseline` work without a job.
   - The S1 stand-in in the monitor: never reached by my gates. I recommend deleting it at the trainer's next change.
   - No timeout on the monitor child: I recommend keeping none; rule M1 already treats a kill mid-record as an interruption. My kill-resume children have a 4 h bound.
   - "R0L exactly" to rounding: I recommend `survival_credit` return the survival time itself when the clock covered 1.05 × it, so R0L's S equals its mean survival exactly.
6. **Beyond the frozen API:**
   - Keyword-only arguments on the frozen functions: `conditions`, `seeds`, `delays`, `cut_s`, `root`, `g9_ref`, `refcheck`, `block`, `tiny`, `settings`, `child_cmd`, `child_env`, `hold`, `n_steps`.
   - Public helpers: `part_a`–`part_f`, `controls`, `tail_tenth`, `joins`, `compare_*`, `checkpoint_items`, `require_g9_ref`, `dev_episode`.
   - CLI: `dev-episode`, the internal `_train` child, `--g9-ref`, `--set-root`.
7. **Not done:**
   - No gate or validation record was made.
   - G9-run was not exercised on real `lookbar` output (synthetic companions only).
   - The `-X` and V6noF5 forms of G9-ref were tested only through the comparand test on V0.
   - K3's real settings were not run.
   - The stub kill-resume test resumes with 3 workers; the stub env has no plant and is light.
   - Rates ran only at W=1 and 2.
   - The chain was only dry-run, plus the marker and failure tests.
