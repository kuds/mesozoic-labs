check:CF2 (ruling 35): independent check of wt/wpc-cf at f5973876. Simulation only. I changed no tracked file, committed nothing and ran no physics episode. My scripts and outputs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave8/check_cf2/.

VERDICT: merge as it stands. All four points are done as ruling 35 says. Points 1 to 3 each have a test that fails without the change. I found no defect in the package. One integration point for the lead (finding 1) comes from LT, which is being built in parallel. It does not block this merge, but it must be settled before R2L's first fit on a look set.

FINDINGS (most serious first)

1. Major, integration with LT, for the lead (not a CF2 defect against ruling 35 as written).
   - The problem: once LT lands, R2L's fit and selection on a look set will still place turns and stops the bar's way (whole windows), not at join rule J1's leave point (ruling 21).
   - Where: r2lfit.py:785-807 (`gate`, `look_rule`) and the Evaluator at r2lfit.py:578-588 (`bar.BarPlanner(...).resolve(c)`).
   - Why: `look_rule` opens `lookbar.look_set_rule` only to ask `check()`, then closes it. Pools and Evaluators are built afterwards (`_pool`, r2lfit.py:995), outside any context. LT's placement (`_placed_resolve`) acts only while a context holds the set's key, or a worker holds it for life (`hold_for_life`).
   - Evidence: my probe on the merged tree CF2+LT. After `r2lfit.gate("V6noF5", ...)` returns, `lookbar.placed()` is `[]` and `BarPlanner._resolve` is the bar's own.
   - Effect: the fit would tune gains on turn and stop episodes that ruling 21 calls a code error (about 20 s of walk, then a 0.42 rad jump). The row's bar records (`lookbar run`, inside the context) are placed at J1. So slot 0 would no longer equal the bar's R2L row's episode on a look set.
   - Fix (a later CF and LT change, after LT is merged): build the Evaluator inside the set's placement, in-process and in each pool worker (for example `lookbar.hold_for_life(placement)` in `_evaluator`). Add a test that a fit's turn plan on a look set equals `lookbar.leave_resolve`'s.

2. Low, report precision only.
   - The builder's note d says "off the look sets the records gain only g9_ref: null".
   - In fact `rules` (it carries GATE_RULE's new text) and `sources()` (r2lfit's sha256) also change in every new cma.json and r2lg5.json, V0, F25F9 and F25F9cl included.
   - No such record exists (stage2/tb/r2l/index.json has no fits), so no existing record changes. Episodes and decisions off the look sets are unchanged in the code.

3. Low, test gap (equivalent mutant).
   - Mutant M2d keeps the context but replaces `rule.check()` with a direct `lookbar.require_g9_ref(...)` call. It passes every test.
   - It is equivalent for every valid name: `R2L-<set>/<seed>` is always admitted.
   - No fix is needed. Optionally, assert that the context's `rule.applies`/`cited` is what `look_rule` returns.

FOR THE LEAD (the builder's choices; I agree with each)
- a. `select` is gated as well as `fit`. Rule X1 requires this: the selection runs about 400 closed-loop episodes on the set.
- b. G9-ref is read by gates2 and by lookbar's context, with a same-file check. Mutant M2c (the check removed) is caught.
- c. A version (V6noF5.1) is gated, but it is a development fit because FITS names V6noF5. This needs a ruling if the rebuilt sets come as versions.
- d. A fit started under the old r2lfit cannot be resumed (its sources moved). Only scratch smoke runs exist.

CHECKED AND FOUND RIGHT
- Point 1:
  - `gate` decides "look set" with `lookbar.look_set_of`, versions included, then calls `gates2.require_g9_ref(body, set, path, root=SET_ROOT)`.
  - It refuses on a GateError, on no citation (None) and on a missing gates2. Nothing is copied.
  - On the working branch, a version is refused (gates2 gives None), so it fails closed. On CF2+LT, the version is let in with the record's citation (probe: "V6noF5.1 gate: cited ... row R2L-V6noF5.1/42 look_set V6noF5").
- Point 2:
  - `look_rule` enters `lookbar.look_set_rule(set, body, ["R2L-<set>/<seed>"], SET_ROOT, g9_ref=path)` and asks `check()`.
  - A LookBarError or BarError becomes an R2LError, and a missing citation is refused.
  - `bar._check_rule` is restored. In the probe it was the bar's own after every call, on the working branch and on CF2+LT.
  - The same calls work against LT's changed `look_set_rule` and `check()` (merged tree tests pass).
- Point 3:
  - `resume_flag <dir> [file]` with `F=$W/r2l/V6noF5/$B/r2lfit/42`, which matches `fit_dir`.
  - A fresh fit writes cma.json at session start, before any evaluation (r2lfit.py:1019-1022). So "cma.json exists" is the right condition.
  - V10-collect is gone; `publish` is the collect. The step count is 45.
- Point 4: the V3 note matches ruling 16, LB16 as built (`lookbar run` uses `look_set_rule`; R5G and refusals per 31 (1)) and X1. The header and V10 notes are also corrected. No other stale "no --task" text remains (grep).
- Off the look sets, `gate` returns None for V0, F25F9, F25F9cl and S1 (probe), and fit and select are otherwise unchanged.
- Only the package's four files differ from the merge base 3da0c6a8: r2lfit.py, run_trackb.sh, test_stage2_r2lfit.py, test_stage2_gates2.py (its chain section only: the chain's test). The merge d6dd3d10's tree equals 3da0c6a8's. bar.py, moves.py, lookbar.py and gates2.py are untouched.
- Record-key hashes, before and after my runs: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair.
- `git status --porcelain` is empty in the worktree and in the main checkout. `ruff check --no-cache` on the three Python files: "All checks passed!".

REVERT AND MUTANT CHECK
- Method: a scratch export (symlinks plus copies of the changed file and the test files) for each variant. The worktree was not touched.
- Results: mutants_r2l.out and mutants_chain.out. The r2l control run gave "36 passed, 1 skipped". The chain control's only failure is test_every_output_path_of_the_chain_is_ignored_or_a_record, which needs git and fails in the export.
- Point 1:
  - M1a (gate by LOOK_SETS name only): fails "DID NOT RAISE R2LError" (version test).
  - M1b (record accepted through lookbar's rule instead of gates2): 4 tests fail.
  - M1c (base r2lfit.py): 3 tests fail.
- Point 2:
  - M2a (no context): 3 tests fail.
  - M2b (select ungated): "DID NOT RAISE".
  - M2c (no same-file check): "DID NOT RAISE".
  - M2e (context's refusal swallowed): "DID NOT RAISE".
  - M2d: survives (equivalent; finding 3).
- Point 3:
  - M3a (--resume always): "assert [False, False] == [True, True]".
  - M3b (V10-collect back): "assert 46 == 45", plus the new test.
  - M3c (resume_flag reads run.json): caught.
  - M3d (cma.json read at the set/body directory): caught.
  - M3e (base run_trackb.sh): 2 tests fail.

TEST SUMMARY LINES (my runs)
- In the worktree, `bash tools/lowprio.sh $PY -m pytest -p no:cacheprovider tests/test_stage2_r2lfit.py -rs`: exit 0, "36 passed, 1 skipped in 78.01s (0:01:18)". The skip is the slow test.
- In the worktree, the same on tests/test_stage2_gates2.py (holds the chain's tests): exit 0, "41 passed, 4 skipped in 99.25s (0:01:39)". Three skips need the look ladder's arrays; one is STAGE2_SLOW.
- `bash -n stage2/tb/run_trackb.sh`: exit 0, no output. `sh -n`: exit 0.
- Merged with the working branch's tip fe605133 (tree 9f020f91), scratch export:
  - r2lfit tests: "36 passed, 1 skipped in 76.20s (0:01:16)".
  - gates2 tests (-k "not output_path"): "40 passed, 4 skipped, 1 deselected in 103.42s (0:01:43)".
- Merged with LT (`git merge-tree HEAD wt/wpc-cw` gives tree 77be39de, which also holds fe605133), scratch export:
  - r2lfit tests: "36 passed, 1 skipped in 78.69s (0:01:18)".
  - gates2 tests: "1 failed, 42 passed, 4 skipped, 1 deselected in 95.10s (0:01:35)".
  - The failure is test_g8a_record_and_exit_codes_on_the_s1_view, a sha256 mismatch of gates2.py in the record's sources. It fails the same way on LT's tree without CF2 (merged_ltonly.out). It is an artefact of my export: s2io.STUDY resolves to the worktree, which has the old gates2.py. It is not caused by CF2.

WHAT TO BRING IN
- `git merge-tree --write-tree claude/mesozoic-trex-stage2-refine-4jffre HEAD` (the branch is at fe605133): tree 9f020f91, exit 0, no conflict.
- The branch moved after the builder's merge (3da0c6a8 to fe605133: moves_look, sets, reports). That is not a fast-forward, so merge wt/wpc-cf at f5973876c5e4bbe2c10cbe1d51fb6e47ab007a25 with a merge commit.
- Commits not in the working branch, oldest first: d6dd3d10 (merge), 1977a6db, f5973876. Each ends with exactly the two attribution lines and names no model elsewhere. None lacks them.
- Merge-tree of wt/wpc-cf with wt/wpc-cw, wt/wpc-ct, wt/wpc-cn, wt/wpc-cg and wt/wp-l: exit 0 each. Only wpc-cw also edits tests/test_stage2_gates2.py, and the merge is clean.

WHAT I DID NOT CHECK
- The slow tests (STAGE2_SLOW=1). lookbar's, r5lib's or any other package's test files.
- A real fit or selection on a look set: none is frozen, and look-set arrays are not in this checkout.
- A numeric demonstration of finding 1's placement difference: no look-set arrays. It rests on reading the code and the probe's `placed()` result.
- A run of the chain beyond its tests (dry and fake-python runs).
