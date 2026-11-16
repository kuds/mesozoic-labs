# build:CV (roundC, wave 2a): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

WP CV (vecpool) - builder's report. Worktree /home/user/mesozoic-wt/wpc-cv, branch wt/wpc-cv, head ca7800a. Simulation only; nothing trained, no policy, no look set touched.

1. MERGE OF THE WORKING BRANCH
- First merge (5c8abf7): claude/mesozoic-trex-stage2-refine-4jffre at 26b057e, 233 commits. "Automatic merge went well": no conflict (the working branch does not touch my two files).
- The working branch moved by 2 commits while I worked (STAGE2_75_PLAN.md, stage2/design/TRACKB_RULINGS3_16NOV.md; the rulings do not mention the pool). Second merge (ca7800a) at bb5b63d, again no conflict. `git rev-list --count HEAD..<working branch>` = 0 at the end.
- Record-key hashes after each merge and at the end: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1 (the registered ones).

2. WHAT WAS THERE, WHAT I CHANGED
The earlier builder's three commits already held a complete pool and test file: after the merge and before any edit the brief's test command gave "40 passed, 1 skipped in 33.99s". I read both files in full against the brief, SB3 2.9.2's dummy_vec_env.py / base_vec_env.py and design D13, D16, 2.2, 3.6, 4. The worker loop (_Block.step / reset) is DummyVecEnv's statement for statement. Every item of the brief's API, start-up, failure and test list is present. Changes (commit 30af723):
- designlab/stage2/vecpool.py
  a. tbenv is on the branch since the merge: child_env(), study_dir() and THREAD_VARS are now tbenv's; the inline copy of the child environment (and its `except ImportError` fallback) is gone. Test: test_child_env_is_the_track_b_child_environment (now unconditional).
  b. A gap at start-up: a child that connected and left, or answered the challenge with another key, raised BrokenPipeError / AuthenticationError out of the constructor (I saw both before the fix). _accept now drops such a connection and the constructor raises the RuntimeError naming the worker and its exit code, as for a child that never connected. Test (new): test_a_child_that_fails_the_handshake_fails_the_constructor_by_its_name.
- tests/test_stage2_vecpool.py: the new test; three waits that assumed an idle machine made load-independent (the SIGTERM-deaf worker now writes a mark when it is deaf instead of a 0.5 s guess; "the workers left by themselves" is shown by exit code 0 instead of a 5 s bound; a dead start is bounded by half the connect limit; _gone waits up to 60 s). No behaviour of the pool changed by these.

3. ACCEPTANCE (run from the worktree's study directory with the standing environment; through tools/lowprio.sh, i.e. nice 19, instead of the brief's nice -n 15)
- `bash tools/lowprio.sh $PY -m pytest -p no:cacheprovider tests/test_stage2_vecpool.py` at head ca7800a: "41 passed, 1 skipped in 14.52s", exit code 0 (under 2 minutes; at most 4 workers; the skip is W=8). Same at 30af723: "41 passed, 1 skipped in 15.21s".
- Slow flag, once, at 30af723: `STAGE2_SLOW=1 bash tools/lowprio.sh $PY -m pytest -p no:cacheprovider tests/test_stage2_vecpool.py`: "42 passed in 16.08s", exit code 0 (W=8 included).
- After the runs: no vecpool-* directory in /tmp, no worker process left, `git status --porcelain` empty.
- Real env, once (asked beyond the brief). Script (not committed, scratch): /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/cv/realenv_check.py; records under .../scratchpad/cv/out, out_short, out_final. Command: `bash tools/lowprio.sh $PY <script> --envs 8 --steps 400 --workers 2 --out <scratch>`. Setup: R5, body 75-M-H-10, set "S1" (Stage 1's view), r5env.env_configs' 8 envs (6 hard, 2 carpet), seed(4,200,000), reset, 400 vec steps of a zero action = 3,200 env steps; reference = SB3's own DummyVecEnv in one process; everything returned and held (obs, rewards, dones, full infos, reset_infos) compared at every step by type, dtype, shape and bytes.
  Final code (30af723), load average about 1.2, 4 vCPUs:
  - DummyVecEnv, one process: 10.09 s wall, 317.1 env steps/s (CPU 9.96 s); build 20.0 s; RSS 1,087 MB.
  - PoolVecEnv, 2 workers (blocks 0-3, 4-7): 5.89 s wall, 543.2 env steps/s; workers' CPU 5.21 + 5.05 s, trainer process 0.25 s; build 11.2 s; worker RSS 521 and 534 MB; no torch library mapped in either worker (/proc/<pid>/maps); spaces equal; first_difference None, equal_bit_for_bit True; value digest of both runs 74a28220ff5c89aa6d228e2766f1de9f3f10a40ece22557f62ba82b8652ada1a.
  - The rollout had 3 terminations (tilt), 0 truncations; tasks seen: brisk, stand, stop, walk.
  Earlier runs of the same check (stepping code identical): one process 334.1; 2 workers 466.4 (in that run the parent was also collecting the reference's closed envs: 1.45 s of its CPU, an artefact of my script, fixed with gc.collect) and 510.1 from a fresh parent process (same digest); 1 worker 309.2 (equal bit for bit too). So W=2 gave 466-543 env steps/s, 1.5-1.7 times the one-process run.
  - Variant with max_seconds 2.0 (300 steps; 24 truncations with auto-reset, 0 terminations): equal bit for bit, digest c034401d... both; 304.5 against 490.6 env steps/s.
  - Where the time goes: a 4-env reply is 4,470 bytes, 0.068 ms to pickle and 0.051 ms to load, against 11.1 ms median (12.3 mean, 17.3 p90) for the block's step. The pool's own cost is about 1 %; the rest of the gap to 2x is the spread of step times (each vec step waits for the slower block) and the workers' CPU per step rising a few percent side by side.

4. END STATE
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
 .../trex_form_study/designlab/stage2/vecpool.py    | 861 ++++++++++++++++++++
 .../trex_form_study/tests/test_stage2_vecpool.py   | 894 +++++++++++++++++++++
 2 files changed, 1755 insertions(+)
`git status --porcelain`: empty. Nothing pushed. Nothing written in the main checkout.
Git-ignored files added to my worktree: the picks' models and Stage 1 arrays, by `tools/restore_inputs.py copy-from /home/user/mesozoic-labs/hardware/trex_form_study --group picks-models picks-arrays` (90 files, "all here with their listed sha256"; git check-ignore confirms round2/.gitignore models/ and stage1/.gitignore refs/, *.npz). The real-env check needed them.

5. FOR THE LEAD
Readings chosen (the first six are the earlier builder's, kept; say if another is wanted):
- workers=0 is not a pure delegation to DummyVecEnv: kwargs, options, set_attr values and env_method arguments are pickled per env and get_attr / env_method results and the spaces are copies at W=0 too, and unpicklable arguments are refused at W=0. Reason: D16, nothing may depend on W. Alternative: the brief's literal "delegate everything" (then a closure or a shared mutable object works at W=0 only).
- env_is_wrapped answers False at W >= 1 (brief) but DummyVecEnv's true answer at W=0. The study's envs are not wrapped.
- The socket directory is removed as soon as all workers are connected, not at close() (brief: close removes it). After close there is none either way.
- A worker does not `assert` its environment: it reports MUJOCO_GL / thread variables in its hello, exits 3, and the constructor raises RuntimeError with the values.
- Design 2.2 lists `seed` among what worker_main serves: seeds stay in the parent (SB3's VecEnv.seed) and travel with `reset`; env i gets seed + i by pool index. set_attr is served too.
- Beyond the brief: an env's exception is re-raised in the parent with the worker's traceback as a note and the pool goes on; a call interrupted between send and reply (Ctrl-C) breaks the pool for good; workers > number of envs is refused.
- Mine: handshake failures at start-up become the named RuntimeError (above).
Disagreements in the texts:
- Brief's machine rule "at most 2 worker processes in anything you start" against its acceptance "at most 4 workers" and its W=8 slow test. I followed the acceptance, and ran the slow file once on the sentence "slow ones (STAGE2_SLOW=1) one file at a time" (8 stub workers for a few seconds at nice 19).
- The brief points to design texts in the scratchpad; I read the copies under stage2/design in the main checkout, as the task said.
- Design run plan assumes 350 / 290 env steps/s per process and "W = 2 about 570" with 15 % for PPO (about 670 stepping alone). Measured here, zero action on the S1 view at nice 19: 317-334 in one process, 466-543 at W=2 stepping alone. The estimate looks high by roughly a fifth; V3's record should replace it.
Not done:
- No committed real-env test (the brief and design 4 name a stub env; the check is a scratch script that a cleanup can remove). If wanted, it fits as a slow test that skips without arrays, but it would tie my test file to r5env's API.
- Real env only at W=1 and 2, 8 envs, body 75-M-H-10, S1 view, zero action; not at W=3, 4, 8, not R6, not a frozen set.
- The trainer's own command line (`-m designlab.stage2.train2 worker`) is not tested: train2 does not exist yet; a stand-in command line is.
- No run of the tests on a heavily loaded machine.
For the next package (CT, train2):
- `vecpool.PoolVecEnv([("designlab.stage2.r5env:make", {"mode": mode, "cfg": cfg}) for cfg in env_configs], workers, worker_cmd)`; an R5Config travels by pickle. Importing vecpool imports no SB3; the class is made on first access of `vecpool.PoolVecEnv`.
- The `worker ADDRESS` entry must call `sys.exit(vecpool.worker_main(address))` before anything imports torch or SB3; the pool cannot enforce that. The worker refuses unless MUJOCO_GL=disable and the three thread variables are "1" (the pool gives it tbenv.child_env("disable") and the study directory as cwd).
- Workers' CPU for rate.json: `venv.env_method("process_cpu")` gives (pid, seconds) per env, one pid per worker; also pool.worker_pids, .workers, .blocks, .worker_cmd.
- After a dead worker or an interrupted call every later call raises RuntimeError: close and build a new pool. close() is idempotent; clean-up also runs at interpreter exit; workers leave when the trainer is killed.
- Memory seen: a worker with 4 real envs about 525 MB, with 8 about 880 MB, with 1 about 242 MB; workers build their envs side by side (8 envs: 11 s at W=2, 20 s in one process).

6. COMMITS (oldest first)
ceff954, 69eed3b, c51c0db (the earlier builder's), 5c8abf7 (merge of the working branch), 30af723 (tbenv, handshake, test waits), ca7800a (merge of the working branch's two later commits).
