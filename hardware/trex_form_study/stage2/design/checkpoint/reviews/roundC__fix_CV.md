# fix:CV (roundC, wave 2a): the agent's final report, 16 Nov 2026

Working paper. Nothing here is a rule.

WP CV (vecpool) - fixer's report. Worktree /home/user/mesozoic-wt/wpc-cv, branch wt/wpc-cv, head d1b3de8. Simulation only; nothing trained, no look set touched, nothing pushed, nothing written in the main checkout.

Files: V = /home/user/mesozoic-wt/wpc-cv/hardware/trex_form_study/designlab/stage2/vecpool.py ; T = /home/user/mesozoic-wt/wpc-cv/hardware/trex_form_study/tests/test_stage2_vecpool.py

0. MERGE
The working branch had moved by 3 commits (b134773, b57fdd0, 82ba1ad: servo model note, its review, checkpoint status). Merged first as ecdbf92, no conflict (none touches my two files). `git rev-list --count HEAD..claude/mesozoic-trex-stage2-refine-4jffre` = 0 at the end. Record-key hashes after the merge and at the end: 45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1 (the registered ones).

1. PER FINDING (all seven checked against the code; all real; the verdict was "merge as it stands" but it lists defects, so I fixed them)

Evidence that the new tests fail without the fixes: the new/changed tests run against the unfixed pool gave "6 failed, 7 passed, 36 deselected in 103.76s" (the five tests of finding 1 and the "no handshake" case of finding 2). The rest was checked by mutations on a scratch export (never the worktree), each caught; lines below.

Finding 1 (env exception in step/reset leaves a usable pool in a W-dependent state) - FIXED.
- V: `_Pool._exchange(ends_pool=...)`, new `_Pool._spent`, `_Pool.reset`, `_Pool.step_wait`, `_Pool._local`, `_Pool.close`. An env's exception in reset or step is raised (the env's own exception, first in env order, as DummyVecEnv) and then every later call raises RuntimeError "an env raised in `step` (...); ... the pool cannot continue". close still asks the workers to close their envs (they are in step with the pool) and they leave with exit code 0.
- The reviewer's second point in the same finding (which exception when several workers fail in env_method): `_Block._each`, `_Raised`, `_pack_error(at)`, `_serve`, `_Pool._per_env(place=...)`: the exception raised is now DummyVecEnv's, the first in the call's order (was the first in worker order). get_attr / set_attr / env_method errors stay non-fatal, as the reviewer asked (has_attr still works; tested).
- Tests: test_an_envs_exception_in_reset_or_step_ends_the_pool_at_every_worker_count [0-reset, 0-step, 2-reset, 2-step]; TestOnePool::test_an_envs_exception_is_raised_here_and_the_pool_goes_on (rewritten against DummyVecEnv's answer, five index/tag cases). Mutations: no `_spent` at W>=1 "2 failed, 2 passed"; none at W=0 "2 failed, 2 passed"; close not polite after it "2 failed, 2 passed"; `place` ignored "1 failed, 11 passed"; place always 0 "1 failed, 11 passed".
- Small addition of mine so that the new rule has no W-dependent corner: step_wait before any step_async is the caller's AttributeError at every W and ends no pool (test_a_step_wait_before_any_step_async_reaches_no_env_at_any_worker_count; mutation "1 failed").

Finding 2 (a child that connects and says nothing holds the constructor past the limit) - FIXED.
- V: `_Pool._accept`: a threading.Timer at CONNECT_TIMEOUT_S kills (SIGKILL: a stopped child does not act on SIGTERM) the children still not connected, under a lock so that a worker is taken or killed, never both; the blocked handshake read then ends and the constructor raises "vecpool workers [..] did not connect within N s". The wait for a connected child's hello is cut at the same deadline (it was a second full limit). Docstrings say what holds.
- Test: test_a_child_that_connects_and_says_nothing_does_not_hold_the_constructor [no handshake, no hello]. Unfixed pool: "no handshake" failed after 90 s with "left before it connected: exit code 0". Mutation "timer does nothing": "1 failed, 2 passed ... in 94.57s".

Finding 3 (two behaviours held by no test) - FIXED (tests).
- (a) atexit hook: T `test_workers_leave_with_their_parent[exit]` now asserts the workers are gone the moment the parent is; the stub's close lingers 2 s (new `linger` kwarg) so that workers left to themselves are provably still there. Mutation `self._finalizer.atexit = False`: "1 failed, 1 passed, 48 deselected in 8.19s".
- (b) connect limit: new test_children_that_do_not_connect_fail_the_constructor_at_the_limit (limit 1 s, two sleeping children; message, no child left, no socket directory). Mutations: loop check removed "1 failed ... in 3.04s"; check removed and timer idle "1 failed ... in 92.17s".

Finding 4 (tests assume a short temporary directory of their own) - FIXED (tests).
- T: module fixture `own_temp_dir` (autouse): a directory `vpt-*` made under the temp dir if short enough, else /tmp; tempfile.tempdir and TMPDIR point to it for this module only (pytest's base temp is made first); removed at the end. `_socket_dirs()` therefore sees only this module's pools; the listener probe takes its path from it.
- Evidence, TMPDIR of 156 characters: the unfixed files on a scratch export "2 failed, 39 passed, 1 skipped in 15.55s" (the reviewer's two); the fixed worktree "49 passed, 1 skipped in 23.15s".

Finding 5 (two clean-up gaps on a kill, not stated) - FIXED AS THE REVIEWER ASKED (docstring only, no code, no test): V's module docstring now states both (a trainer killed while its workers connect leaves the socket directory; a forked copy of the trainer keeps the workers alive) and "no fork in a trainer while it has a pool open". The sentence for CT's brief is not mine to write: see "for the lead".

Finding 6 (PPO test leaves SB3-<time> folders in /tmp) - FIXED (test): SB3_LOGDIR set to tmp_path/sb3, and the test asserts no SB3-* in the temp dir. Mutation without the setenv: "1 failed, 49 deselected in 3.11s".

Finding 7 (ruff) - FIXED: `# noqa: F822` on `__all__` with a comment; imports of T in ruff's order. `ruff check --no-cache designlab/stage2/vecpool.py tests/test_stage2_vecpool.py`: "All checks passed!" (ruff 0.16.9, repository configuration; no cache left behind).

2. TEST SUMMARY LINES AFTER THE CHANGES (worktree's study directory, standing environment, through tools/lowprio.sh)
- `bash tools/lowprio.sh $PY -m pytest -p no:cacheprovider tests/test_stage2_vecpool.py`: "49 passed, 1 skipped in 24.60s", exit 0 (was 41 passed, 1 skipped in about 15 s; the skip is W=8).
- `STAGE2_SLOW=1 ...` once: "50 passed in 25.12s", exit 0.
- Same file under a 156-character TMPDIR: "49 passed, 1 skipped in 23.15s".
- After the runs: no /tmp/vecpool-* or /tmp/vpt-* directory, no worker process, no .ruff_cache.
The tree tested is the tree committed (no edit between the last run and the commit).

3. END STATE
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
 .../trex_form_study/designlab/stage2/vecpool.py    |  954 ++++++++++++++++++
 .../trex_form_study/tests/test_stage2_vecpool.py   | 1052 ++++++++++++++++++++
 2 files changed, 2006 insertions(+)
`git status --porcelain`: empty (worktree and main checkout).

4. COMMITS OF THIS ROUND (oldest first)
ecdbf92 (merge of the working branch's three later commits), d1b3de8 (the seven findings: one commit, both files; I did not split it because the tests and the code they hold belong together). Head: d1b3de8. Both end with exactly the two required lines.

5. FOR THE LEAD
Readings I built (say if another is wanted):
- Finding 1 at W = 0: I made an env's exception in reset/step end the pool at W = 0 as well, by the module's own rule "at every W, 0 included" (D16). Alternative: W = 0 stays DummyVecEnv pure and usable after such an exception (the brief's literal "delegate everything"); then a trainer that catches and carries on works at W = 0 only. Recommendation: keep.
- After an exception in env_method or set_attr the pool goes on, and one W-dependence remains (now stated in the docstring): envs later in the call's order that live in another worker have been called; at W = 0 or in the same worker they have not. Options: keep; make these errors end the pool too (get_attr must stay non-fatal for has_attr); or run per-env commands one worker-run at a time in the caller's order (exactly DummyVecEnv, no parallelism for env_method). Recommendation: keep, and tell CT not to carry on after a failed env_method that changes env state.
- Not changed: Ctrl-C during a call breaks the pool at W >= 1 (in flight) but leaves a W = 0 pool usable.
- Finding 2's residue: the limit is enforced by killing the unconnected children; a child whose own descendant keeps the socket open would still hold the read. Not covered; a real worker never does this.
- Finding 5(b): I documented it only. A code option exists (os.register_at_fork closing the pool's connections in a forked copy, about 15 lines plus a test); recommend it only if some trainer-side code turns out to fork.
Sentences for CT's brief (not my file): an exception out of reset or step ends the pool, close it and build a new one; no os.fork and no fork-start multiprocessing while a pool is open; clamp W to the number of envs (the reviewer's point).

Not done:
- No real-env run after these changes. The worker's step and reset loop (`_Block.step`, `_Block.reset`) is untouched; the parent's reply handling in `_exchange` changed, and the stub rollouts still equal DummyVecEnv bit for bit at W = 0, 1, 2, 3, 4 and 8.
- No run on a heavily loaded machine (load was about 0.1 to 1 during mine).
- One internal change to note: a worker's error reply carries a fourth element (the failing env's place); nothing outside vecpool reads it.

Scratch (not committed): /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/cv_fix/mutate.py, /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/cv_fix/mutate_limit2.log (and mutate_limit.log, an earlier run whose test selection missed the connect-limit test; superseded by mutate_limit2.log).
