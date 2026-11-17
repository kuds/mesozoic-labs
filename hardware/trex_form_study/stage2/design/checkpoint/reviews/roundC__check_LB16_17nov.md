VERDICT: merge as it stands. I found no defect that blocks the merge. One rule question is for the lead (finding 1). Three low points can go in the next change to lookbar.py.

FINDINGS (most serious first)

1. Rule question for the lead, medium (designlab/stage2/lookbar.py:754, :971, :1246). On a look set, R0@<ms> rows go through the bar's own check, which lets R0 run with no G9-ref. So `lookbar run --set V6 --rows R0@10ms` runs closed-loop episodes on a look set before G9-ref, and so does `lookbar.episodes` with an R0@ row.
   - The test `test_lookbar_run_and_episodes_go_through_the_rule` asserts this, as the ruling and brief word it ("every other set and row goes through the bar's own check unchanged").
   - Rule X1 says "no closed-loop episode before G9-ref passed" on a look set. Between G9-ref and S2-4 it allows only a list, and R0@ rows are not on it. So the literal build leaves an X1 gap.
   - `python -m designlab.stage2.bar run --set V6 --rows R0@10ms` has the same gap. It is outside lookbar, and no lookbar change can close it.
   - Options:
     - (a) keep it as built;
     - (b) on a look set, refuse every row outside the four families, in `_decide` and `check` (the builder's alternative);
     - (c) require the G9-ref for R0 rows too.
   - I would take (b) or (c). Either way, the bare `bar run` gap stays.
   - The builder's other points for the lead also stand, and I agree they are the lead's choices:
     - R5G is not admitted, but plan line 1348 lists R5G among the Track B rows that get bar records;
     - R2L must name the evaluated set exactly;
     - reading 1 counts only on the exact names, not on versions;
     - gates2.require_g9_ref lets a version through;
     - the monitor (monitor2._episodes) lets LookSetRefused propagate;
     - the stale texts.
   - The context checks only X1's first clause, the G9-ref. It does not enforce the "between G9-ref and S2-4's start only these" list. The ruling did not ask for that.

2. Low: state leak when contexts end out of order (lookbar.py:778). On exit, the finally sets `bar._check_rule = rule.own` unconditionally.
   - My probe entered two contexts (V6, then V6noF5) and exited them in the wrong order. Output: "out-of-order exit leaves the bar's own check: False | left: the first context's bar_check".
   - Two threads would do the same. In ordinary `with` use, nothing does this today.
   - What stays installed still demands a passed G9-ref and the exact set, body, root and rows. But it stays in force outside any context.
   - Fix: restore only if `bar._check_rule == rule.bar_check`, and otherwise raise LookBarError. Or keep a module-level stack under a lock.

3. Low: two cases have no test (code right, tests do not pin them).
   - Mutation "drop the `caller.f_code is _EVALUATE_CODE` test" (lookbar.py:727) SURVIVED. The test's direct calls (test_stage2_lookbar.py:1579, :1588) come from frames without a `body` local, so they cannot tell. My probe confirms the real code refuses a non-evaluate caller that holds the context's body and root ("the bar's own refusal: BarError ..."). Fix: one test calling `bar._check_rule` from a helper with `body`/`root` locals inside the context.
   - Mutation "`pass_ is not True` -> `pass_ is False`" (lookbar.py:671) SURVIVED. The tests use only pass_=False. My probe shows pass_=1 is refused as it should be. Fix: add pass_ missing, None and 1 to the bad-record list.
   - A trivial third survivor: `cited` is not reset before a failed check.

4. Very low, the same in gates2: a G9-ref record whose own sha256 is intact but whose `set_freeze` is not an object raises AttributeError, not LookSetRefused. Both lookbar and gates2 printed "AttributeError 'list' object has no attribute 'get'". It fails closed, but as a traceback.

5. For the lead, low: lookbar.require_g9_ref (lookbar.py:653) re-implements gates2's check on gates2's private `_read` and `_latest_freeze`, because gates2's version returns None for a version of a look set.
   - The word-for-word test guards against drift.
   - The gates fix (wt/wpc-cg, bd29fe88) does not touch require_g9_ref, SCHEMAS or those helpers.
   - The cleaner fix: gates2.require_g9_ref covers versions, and lookbar calls it. That is the gates package's file.

CHECKED AND FOUND RIGHT
- Scope: `git diff 3d261a79 HEAD --stat` shows only lookbar.py (+241), tests/test_stage2_lookbar.py (+392) and tests/test_stage2_s2rows.py (9). bar.py is not edited.
- Hash pair at head c4b191dc and on the merged tree: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` (the registered pair).
- D18 on the merged tree: `git diff --name-only c377ba1 <merged tree>` over designlab/stage1 and stage2/{bar,readings,setio,s2io,moves,setscreen}.py lists nothing.
- Every point of the job is built and tested:
  - the context lets only R0L, R5/<seed>, R6/<seed> and R2L-<set>/<seed> run (delay optional, checked by bar.ms_steps), on LOOK_SETS or a version of one;
  - only for evaluations that bar.evaluate itself makes, of the context's set, body, root and rows;
  - the G9-ref is read at the evaluation's check. A record written after the context was entered was honoured.
  - Every other call goes to the bar's own check. Nothing is replaced when the rule does not apply.
  - The context restores on an exception, on a refusal and when nested. vars(bar) are identical afterwards.
  - `lookbar run` and `episodes` use it. RULES["look_sets"] is in every row's description (rule_texts).
  - A refused `lookbar run` creates nothing under --out (probe: "out exists: False"). The CLI exits 1.
- Mutations: 29 single mutations of lookbar.py, applied in a scratch export of the merged tree, file restored and compared after. 26 were caught by the rule's 8 tests, including:
  - no restore;
  - no G9-ref check;
  - body, root or set ignored;
  - any() for all() in covers and in applies;
  - episodes or run unchecked;
  - a stale, unfrozen, wrong-schema, wrong-sha256, wrong-set or wrong-body record accepted;
  - R2L of any set;
  - delay unchecked;
  - versions ignored;
  - nested hand-on removed;
  - the refusal not a BarError;
  - the wrong root in require, episodes or run;
  - the g9_ref path ignored;
  - the RULES key missing.

  The 3 that survived are in finding 3.
- Commits not in the working branch: 9b60d803, 03043319 and c4b191dc. Each ends with exactly the two attribution lines, and none names a model elsewhere.
- Ruff on the three files: "All checks passed!"
- The worktree's tracked and ignored file lists are the same before and after my runs, and `git status` is empty in the worktree and in the main checkout.

TEST SUMMARY LINES (tests/test_stage2_lookbar.py tests/test_stage2_s2rows.py)
- Branch head c4b191dc, in the worktree: "60 passed, 11 skipped in 132.32s (0:02:12)". All 11 skips are STAGE2_SLOW=1.
- Merged tree (code, tests, tools, stage2/sets and design exported from the merge of HEAD into 9889291c; everything else symlinked to the worktree): "59 passed, 12 skipped in 137.18s (0:02:17)".
  - The extra skip is the D18 test ("git cannot tell: no repository here"). I checked D18 by git diff instead (above).
  - The first attempt there gave "2 failed, 45 passed, 12 skipped, 12 errors". The cause was my export: it lacked the repository's `environments` package, giving ModuleNotFoundError. I added the symlink and reran.

WHAT TO BRING IN
- The working branch moved during the check, to 81a395e1. The two new commits change only documents: rulings 28-30, the plan, and review files. None of them touches ruling 16.
- `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre c4b191dc` gives tree 1fb5343632a6a36e7e955b2011e48aace435c556 with no conflict. Against 9889291c it gave 1ca98e5270b1ba7d7aaaf9e5f21f21cc8c43ced7.
- Merge the tip c4b191dc375efd0fbee9c373dcccf9eeb0306c95 (wt/wpc-cw). No cherry-pick is needed.

NOT CHECKED
- The slow tests (STAGE2_SLOW).
- The other modules' tests on the merged tree: monitor2, gates2, lookread, r2lfit. The builder reports "141 passed, 33 skipped"; I did not rerun them.
- The merged tree at the newest tip, 81a395e1. I tested the merge against 9889291c; the newer commits touch only documents.
- The CF use of the context that ruling 16 names (not this package).
- Thread behaviour beyond the out-of-order probe.
- No look-set episode or G9-ref run was made, and nothing was committed or pushed.

Scratch files in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave7/check_lb16/: mutate.py, mutate.log, probe.py, head_tests.log, merged_tests.log, merged_hash.txt. The merged export and pytest temp directories are deleted, because the disk is about 99% full.
