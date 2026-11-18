CF follow-up (ruling 35). Worktree /home/user/mesozoic-wt/wpc-cf, branch wt/wpc-cf. Simulation only. I pushed nothing, rewrote no history and amended nothing. I ran no physics episode and no fit.

1. MERGE OF THE WORKING BRANCH
- d6dd3d10 merges claude/mesozoic-trex-stage2-refine-4jffre at 3da0c6a8 (`--no-ff -F`). There were no conflicts. The merge changed none of the package's files: r2lfit.py, run_trackb.sh and test_stage2_r2lfit.py on the branch were as CF's earlier merge left them. It brought in gates2, lookbar (LB16), the look sets and rulings 21-35.
- The branch tip 3da0c6a8 is an ancestor of HEAD, and the branch had not moved at the end of my work.

2. PER POINT

(1) G9-ref only through gates2.require_g9_ref, versions included. File: designlab/stage2/r2lfit.py.
- New `look_set_of`, which calls `lookbar.look_set_of`, so V6noF5.1 counts as V6noF5. I did not copy it.
- `gate(set, body, seed)` now calls `gates2.require_g9_ref(body, set, path=GATES/<set>/<body>/g9_ref.json, root=SET_ROOT)` for every look set and every version of one. Before, it called it only when the name was exactly one of LOOK_SETS, so a version had no gate at all.
- It refuses on a GateError, on no citation, and when gates2 is missing.
- On the working branch, gates2 returns None for a version, so a fit on a version is refused for now ("gave no citation ... a version of the look set V6noF5"). It fails closed.
- Against the parallel change, the gates2.py and lookbar.py of wt/wpc-cw at about 08:07 UTC (now committed there, 9ebc87b6..37e864e1), in a scratch export since deleted:
  - the 5 gate tests passed: "5 passed, 32 deselected in 8.33s";
  - a direct `gate("V6noF5.1", ...)` cited the record (`R2L-V6noF5.1/42`, sha256 equal).
- Test: `test_a_version_of_a_look_set_waits_for_g9_ref_through_the_gates_package`. It uses a gates2 stub that covers versions, then the real gates2, and branches on what the real one returns.
- Without the change it fails:
  - only LOOK_SETS by name: "Failed: DID NOT RAISE R2LError";
  - record read through lookbar instead of gates2: "Regex pattern did not match";
  - HEAD's whole r2lfit.py: AttributeError (no look_set_of). This last one is trivial; the first two are the real checks.

(2) Look-set records through lookbar's context. File: r2lfit.py, new `look_rule` and `row_name`.
- `look_rule` enters `lookbar.look_set_rule(set, body, ["R2L-<set>/<seed>"], SET_ROOT, g9_ref=path)` and asks `rule.check()`, as `lookbar.episodes` does, before anything is written or run. A LookBarError or BarError becomes an R2LError.
- `gate()` calls it after gates2. Both must cite the same path and sha256.
- The fit session's `gate` now holds `look_set_rule` (checked_by, row, look_set, cited).
- `select()` is now gated too; before, it was not gated at all. r2lg5.json records the citation as `g9_ref`.
- R2L rows' bar records made with `lookbar run` already go through the context, and lookbar's own tests cover R2L rows, so nothing changed there.
- Tests:
  - `test_fit_and_selection_run_on_a_look_set_only_as_lookbars_look_set_rule_lets_their_row`: lookbar refuses while gates2 passes; the two cite different files; the selection is refused afterwards; the bar's check is put back; nothing is asked on V0 or F25F9cl.
  - `test_the_gate_calls_the_real_gates_package` now also checks the real lookbar rule's citation.
- Without the change it fails: no context, "DID NOT RAISE R2LError"; select ungated, the same; HEAD's r2lfit.py, the same.

(3) The chain resumes V10 only on cma.json. File: stage2/tb/run_trackb.sh.
- `resume_flag <dir> [file]`, default run.json. V10's fit passes `$(resume_flag "$W/r2l/V6noF5/$B/r2lfit/42" cma.json)`, checked body by body.
- The V10-collect step is removed.
- The chain's test lives in tests/test_stage2_gates2.py. I edited only its chain section:
  - the step count goes from 46 to 45;
  - new `test_r2ls_fit_resumes_only_when_its_cma_json_exists_and_publish_is_its_collect` (a dry run, and a real run with a python that records its calls).
- Without the change it fails: "--resume" always gives "assert [False, False] == [True, True]"; V10-collect put back gives a step-list mismatch; HEAD's whole chain gives a step-list mismatch.
- Revert script: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave8/cf/revert_check.py. Output: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave8/cf/revert_check.out. It ends "every variant failed its test: True", with the files restored.

(4) The stale V3 note of run_trackb.sh is rewritten to what ruling 16 and LB16 built.
- The bar's check is unchanged and bar.py is not edited. Design 2.3's "no --task" sentence was wrong.
- `lookbar run` evaluates inside `lookbar.look_set_rule`: R0L, R5, R6 and R2L-<set> rows run on V6 or V6noF5 (or a version) only after G9-ref passes. As LB16 built it, other rows go to the bar's own check; ruling 31 (1) adds R5G and refuses the rest.
- So R0L runs after V1 with no --task. `bar run` on a look set is not that rule, and no step calls it.
- Also changed (same file):
  - the V10 note "r2lfit is not on the working branch yet" is replaced;
  - the header's resume and collect sentence now says R2L's fit resumes on cma.json and `publish` is R2L's collect.
- There is no test for (4); it is text only.

FOR THE LEAD
a. I gate `select` as well as `fit`, reading "an R2L record on a look set needs a passed G9-ref" to cover r2lg5.json, since the selection runs over 400 episodes on the set. The alternative is to gate the fit only, as before. I recommend keeping it.
b. G9-ref is now read twice: gates2 directly, which point 1 requires, and lookbar's context, which point 2 requires and which calls gates2 itself once LT lands. The alternative is the context alone, citing `rule.cited`. I recommend keeping both and the same-file check.
c. Rule Q1 names V6noF5 by name (FITS). So a fit on a version is gated but is a development fit, and it is no record. If the rebuilt look sets come as versions, FITS and role_of need a ruling.
d. Effects of the new keys and the edit:
   - Off the look sets the records gain only `g9_ref: null` in r2lg5.json. Episodes and decisions on V0, F25F9 and F25F9cl are unchanged.
   - `gate()` now imports lookbar on every set.
   - Editing r2lfit.py moves sources(), so a fit started earlier cannot be resumed (only scratch smokes exist). It also moves train2's recorded source set.
e. tests/test_stage2_gates2.py is shared with LT. `git merge-tree --write-tree HEAD wt/wpc-cw` gives tree b613fd95 with exit 0 and no conflict. I did not run the merged tests.

3. TESTS, OUTPUTS, HASHES
- `bash tools/lowprio.sh $PY -m pytest -p no:cacheprovider tests/test_stage2_r2lfit.py -rs`: "36 passed, 1 skipped in 74.72s (0:01:14)", exit 0. The skip is the slow test (STAGE2_SLOW=1).
- Same on tests/test_stage2_gates2.py, which holds the chain's tests: "41 passed, 4 skipped in 101.22s (0:01:41)", exit 0.
  - Three skips are "needs the look ladder's arrays and the round-two models": those inputs are not restored in this worktree.
  - One skip is "real env and a tiny training run on the S1 view; set STAGE2_SLOW=1".
- `bash -n stage2/tb/run_trackb.sh`: no output, exit 0. `sh -n` also exits 0.
- `ruff check --no-cache` on r2lfit.py, test_stage2_r2lfit.py and test_stage2_gates2.py: "All checks passed!"
- Hashes: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`, the registered pair.
- Not run: the slow tests, lookbar's tests, other packages' tests.
- I created no new output path in the repository. Pytest temp directories and the scratch export are deleted.

4. DIFF AND STATUS
`git diff claude/mesozoic-trex-stage2-refine-4jffre --stat`:
 .../trex_form_study/designlab/stage2/r2lfit.py     | 105 ++++++++++----
 hardware/trex_form_study/stage2/tb/run_trackb.sh   |  34 +++--
 .../trex_form_study/tests/test_stage2_gates2.py    |  58 +++++++-
 .../trex_form_study/tests/test_stage2_r2lfit.py    | 153 ++++++++++++++++++++-
 4 files changed, 313 insertions(+), 37 deletions(-)
- `git status --porcelain`: empty, in the worktree and in the main checkout.
- bar.py and moves.py are not touched, and neither is lookbar.py or gates2.py.

5. COMMITS, OLDEST FIRST
Each ends with exactly the two attribution lines and names no model anywhere else.
- d6dd3d10 Merge the working branch (look sets, LB16, CG2, rulings 21-35)
- 1977a6db r2lfit: G9-ref only through gates2.require_g9_ref, versions included; fit and selection through lookbar's look-set rule (points 1, 2)
- f5973876 Track B chain: R2L's fit resumes only when its cma.json exists; no V10-collect; the V3 and V10 notes as built (points 3, 4)

Branch head: f5973876c5e4bbe2c10cbe1d51fb6e47ab007a25 (wt/wpc-cf).
