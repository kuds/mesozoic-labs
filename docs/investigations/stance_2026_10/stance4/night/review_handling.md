# Review handling: replicate study (N4/spec.md), 2026-10-20, 01:36-01:50 UTC

Paths: SP = scratchpad,
N4 = SP/foot/stance4/night. All changes were made before any training step of this study. Report only: nothing here
changes the stance4 criteria (sha256 978a93da..., measured unchanged at 01:49 UTC) or any stance4 verdict.

Nine items: eight findings that ask for action and one scope note. All eight were checked and accepted. None was
rejected.

| # | severity | finding | checked how | decision | change |
|---|---|---|---|---|---|
| 1 | blocking | Reading line (1) cannot be met (K2a cannot separate completely); the (a)/(b) fork was open | Measured: readout dry run gives K2a A 0, Colab no-change 0, local no-change 15. perm_test([0,0,0],[15,0,3,2]): complete False, weak True, p 4/35 | Accepted. **Option (a) chosen** | spec section 7: decision with UTC time and reasons; spec section 0 updated. readout.py line logic unchanged; it adds a qualifier to line (2) when the other readout is weakly separated |
| 2 | blocking | The 35-split p overstates the evidence: last night's three runs generated the hypothesis | Measured by enumeration (own script): with A < Colab < local held fixed, complete separation occurs in 72 of 840 orderings = 3/35 = 0.086. New runs alone (2 vs 2): floor 1/6 | Accepted | spec 6.4 and 0 amended. readout.py: cond_perm_test() (p conditional on the old three's order, ties kept as observed) and a new-runs-only test with within-seed signs for seeds 152 and 252; both report only, shown in the primary table; markdown text rewritten |
| 3 | minor | A restart between judge_E.json and judge_E.exp.json leaves an orphan the judge refuses to overwrite; the run would end post_failed | Code read: judge_stance3.py line 1007 refuses an existing --out, line 1135 opens it with "x"; judge_exp.py writes .exp.json after J.main returns; judged() needs both | Accepted | queue.sh post_one: if judge_E.json exists and .exp.json is missing or empty, move it to judge_E.orphan.<UTC>.json, log it, and judge again. Tested in queue_sim2 |
| 4 | minor | The ended-run sensitivity filters on state == "ok", so it would drop a run whose R2 did not reproduce from the K2a test too | Code read: lines 222-225 of the pre-amendment readout.py | Accepted | readout.py: the sensitivity uses the main test's filter and drops only ended runs |
| 5 | minor | Stagger sd over full-horizon episodes can be NaN or 0, or readouts() can raise; max/min with NaN depend on order | Code read: analyze_night.readouts() uses np.std over length >= T episodes (0 for one episode, NaN for none) and np.min (raises on none). The reviewer's max/min example is standard Python behaviour | Accepted, slightly widened | readout.py (last night's tools untouched): fewer than 2 full-horizon episodes in the judge file, or a non-finite value, ranks the run worst on stagger sd (K2a counts as measured); an exception from r2_set is caught and recorded instead of stopping the readout; the full-horizon count per run is shown under the primary table. Spec section 8 states the rule |
| 6 | minor | A prune interrupted by a restart is never re-run | Code read: prune was only called at the end of post_one | Accepted | queue.sh: every poll runs prune for runs in state done (idempotent; logs only when it removes a file). Placed before the disk check, so it also runs when disk is low. Tested in queue_sim2 |
| 7 | minor | The queue's judge/R2 is a process outside the stated budget; the CPU check is a one-time pcpu check | Code read: busy_count is a lifetime-average pcpu count taken once, before a post job starts | Accepted | queue.sh: a post job also requires that pgrep finds no queue_controls.sh or panels4.py process (QUEUE_PANELS_RE). Spec section 9: the replicate judge/R2 takes the panels slot only when no other panels job holds it; other panels jobs should check N4/logs/post.lock. Wall-time estimate updated (inferred about 2 h 50 min to 3 h). Tested in queue_sim2 |
| 8 | minor | Nothing checks the new build files at run time; R1.json has only 12-character prefixes | Code read: queue.sh PIN covered only last night's six files | Accepted | queue.sh PIN now also holds the full sha256 of judge_rep.sh, roll_rep.py and readout.py (9 files; dry run: preflight ok). Spec section 11 lists all four build files with full sha256 and a timestamp |
| 9 | none | Scope note: checks that passed | Read only | No action | none |

## Why option (a) and not (b)

- The orchestrator fixed the lines. (b) lowers the K2a bar below what was intended.
- (b) would set the bar after last night's K2a values were seen. That is finding 2's problem again.
- Under (a) the outcomes are still informative: line (2) through stagger sd, line (3), or "incomplete". Line (2) with
  complete separation on stagger sd and weak separation on K2a is the best the design allows. readout.py names it.

## Checks run (no training)

- readout.py dry run on last night's three runs: unchanged numbers (stagger sd 7.46 / 25.93 / 9.67 mm; K2a 0 / 15 / 0);
  no .pyc written. N4/build/checks/review_fix/readout_dry/.
- Conditional and new-runs-only tests on hypothetical values: N4/build/checks/review_fix/condcheck.txt. Measured:
  complete separation on stagger sd gives 1/35 = 0.029 (35 splits), 72/840 = 0.086 (conditional), 1/6 = 0.167 (new
  only). The K2a best case gives 4/35 = 0.114, 144/864 = 0.167 and 1/6.
- queue.sh --dry-run: preflight ok with 9 pinned files; commands identical to the first dry run.
  N4/build/checks/queue_dry_run_v2.txt.
- queue.sh state-machine simulation with stand-ins (N4/build/checks/queue_sim2/): T1 ran all four runs with one
  simulated crash, and posts waited while a fake panels4.py was alive. T2 moved an orphan judge file aside and judged
  again. T3 pruned leftover files of finished runs on the first poll. A first T1 attempt is kept as t1_artifact.out:
  the pattern matched the test shell's own command text, which contained "panels4.py". That is a test artifact. It also
  means that any shell whose command line contains "queue_controls.sh" or "panels4.py" makes the replicate posts wait
  while it lives. That errs on the safe side.

## Final sha256 (measured 01:49 UTC)

- queue.sh eb11c48af8a387153ac36fea00f108be467971812935739a2f7462919722760b
- judge_rep.sh ac144468549c9e93ab7a7cef40086da40d8805115bd94774010d1d6c4cfbc537 (unchanged)
- roll_rep.py 166e1b0d5a1ad24722e86e5ce9bb9d7b8edc8bc0b15123c83c2764af279959dd (unchanged)
- readout.py 6ef7ec98a5f10f36ddd2c16e450a9db69f3eb100d093e49a12f8904a164872e3
- spec.md: see N4/spec.sha256 (frozen, mode 444)
- spec.md was frozen at 2026-10-20T01:48:58Z (sha256 24af508a363c382a51768da2c4e8b924ef026b6f3607a78c9794f23f4b56f24a); its header's "01:50" was written a minute before.
