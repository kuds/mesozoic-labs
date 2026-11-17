# Round C (training tooling): the lead's rulings of 17 Nov 2026, 01:13 UTC

Working paper for the fixers' briefs. Nothing here is a result. The rulings of 16 Nov
(`ROUND_C_RULINGS_16NOV.md`) stand; these follow the reviews of the rows package (CW) and the notebook (CN),
whose fix rounds were cut by the usage limit at 21:19 UTC on 16 Nov
(`checkpoint/reviews/roundC__{build,review}_{CW,CN}.md`).

## The notebook and launcher (CN)

1. **A job's last copy to Drive (review, major 1).** At a job's end the copy is tried again, fenced and with a
   growing wait, for at most 20 minutes, until the final checkpoint and each check's archive read back from Drive.
   If they do not: no `DONE.json` is written, `finish` returns false, the runtime is **not** released, and the
   notebook prints one plain paragraph telling the owner what happened and to run the finish cell again later.
   The hours tally counts that time. A trained job lost to a failed copy costs more than idle minutes. `finish`
   never raises on a Drive error.
2. **What is final and what is resumable (review, major 2).** As the reviewer proposes:
   - trainer exit 4 (a refusal) writes no `HALT.json`: the session ends and `LIVE.json` is closed, as for an
     interruption;
   - trainer exit 5 (a halt) and any error of the launcher itself write a halt marked resumable; `session`
     accepts a resume of it only with the remedy the halt names, and then keeps the old halt as `HALT_<k>.json`;
   - only "crashed twice" stays final.
   The remedy comes from **one small file the lead writes on Drive** beside the run (`LEAD.json`), which
   `session` reads and records in the session's record: `retry_check`, `allow_source_change`,
   `allow_version_change`, `budget_by` and `budget_set_on` (rule Y10). The owner is never asked to set these.
   `_say` never raises; `copy_back` and the first beat handle a Drive error. Then build step 6 of the acceptance
   run (design 3.10).
3. **A resume that would start from step 0 (review, major 3).** As the reviewer proposes: the copy back is all or
   nothing (each checkpoint checked against its record, then a marker; it is skipped only on the marker); a
   `--resume` start is refused when Drive holds a readable checkpoint and the local directory holds none; the
   sheet says "restarting from step 0" is right only when the Run directory line said "from 0 steps", and
   otherwise to stop and tell the lead.
4. **The code's minor points:** all fixed as the reviewer proposes. Also: `S2_NB_SEAMS` is refused on Colab and in
   any run that is not tiny; a Colab launch is refused when tracked files of the checkout are modified (untracked
   files are only reported).
5. **The steps sheet:** every point of the review is taken. `QUEUE` becomes a text field. The rules file is named
   by a field (today `STAGE2_TRACKB_PLAN.md`). Colab behaviour that was not seen here is marked "expected, not
   yet seen", to be confirmed by the acceptance run.
6. **The builder's points a to k:** kept as built, as the reviewer chose (b: the resume right after a kill needs
   TAKEOVER; c: `session` writes `QUEUE.json`; f: checkpoint 0 is not copied; i: the rate on line 4; j:
   `REBUILD_OK.json`; k: a smoke is not resumed). Point h is answered by ruling 2. Retries are counted per job
   across sessions. Fine-tunes (rule K8) get no queue entry until it is decided where they run. The run plan's V8
   command is rewritten by the lead to the notebook's real lines.

## The rows for the bar runner (CW)

7. **A rule text lost from every row description (review, major 1).** `lookbar`'s key `reference` is renamed
   `reference_use`; an assertion that the two tables share no key; a test that every row description holds the
   look reading's own text. Before any Track B record: the description is in the key.
8. **The review's minor points 2 to 6:** all fixed as the reviewer proposes (the state-A test; a test of rule K4's
   check at the episode; a refused record's stale companion moved aside and kept as a cell of the summary; a fault
   recorded as `Type: message` without this machine's paths; `set_check` refuses a `run.json` without plant
   inputs or task names).
9. **An episode during which MuJoCo warned (review, point 7, a rule reading).** It is a fall at its start, as
   `tune.warned_fall` has it: the companion episode carries `fall_time_s` 0 and a `warned` flag, and the
   monitor credits it no survival.

## The trainer (CT), a third small follow-up

10. **What the rows package's review found the trainer owes:** the trainer passes its `--allow-source-change` to
    the row (in the monitor child's environment), and the monitor treats a row's refusal before any episode
    (`s2rows.RowError`) as a refusal of the monitor (ruling 6 of the fifth list; ruling 3 of the sixth list).
11. **Owed from the review of the second follow-ups and the sixth list:** a missing `run.json`, a checkpoint record
    that cannot be read or has the wrong shape, and a fingerprint not reproduced at the row's load are refusals;
    the halt message after a refusal under `--retry-check` names that parameter; one test kills the trainer
    around a refusal.

## Commits from 17 Nov

The session runs on another model from 17 Nov. Every commit message ends with the two lines the session's
instructions now give. The saved workflow scripts of 16 Nov carry the earlier lines inside their prompts and are
not resumed as they are: the remaining stages run from new scripts.

## Added 03:11 UTC, after the rows package (CW) was merged (`f64094e2`)

Its second fix round did every finding and rulings 7 to 9; the independent check: merge as it stands (33 of 35
reverts caught). On the merged tree the rows', the look reading's and the monitor's tests give "75 passed, 12
skipped". Left for the next change to `lookbar.py`, with the check's three low points: a test of the path cut at the
two other fault sites; `_no_paths` keeps no path's first part; a refusal's text in `look/summary.json` carries no
absolute path.

## Added 03:34 UTC, after the notebook (CN) and the trainer's third follow-up (CT3) were merged (`8348a3a8`, `ff04355a`)

Both checks: merge as they stand. On the merged tree the notebook's, the trainer's, the monitor's, the rows' and the
look reading's tests give "189 passed, 16 skipped". The acceptance run is not done.

12. **The follow-up before the acceptance run** takes every minor point of the two checks
    (`checkpoint/reviews/roundC__check_CN_17nov_2.md`, `roundC__check_CT3_2.md`) and the three low points of the
    rows package's check, with these rulings:
    - `LEAD.json`'s `retry_check` applies only to the session that resumes the halt whose remedy names it; a later
      plain resume carries no `--retry-check`.
    - The steps sheet never tells the owner to delete a runtime that the finish cell kept on purpose.
    - Cell 8 releases the runtime only on finish's exit codes 0 and 5; on anything else it keeps it and says so.
      Cell 8 tries `drive.mount(force_remount=True)` once when the mount itself has died, and says so.
    - The acceptance run's step 6 exercises the real trainer's source check and asserts that
      `--allow-source-change` reaches the trainer.
    - **An allowance for moved sources holds for the rest of the run for the same sources**: the trainer records
      which source hashes were allowed and passes the allowance on at every later session whose sources equal them;
      a halt for moved sources names the parameter that resolves it.
    - A checkpoint record without a `layout` or `fingerprint` object is a refusal of the monitor.

## Added 05:27 UTC: the follow-up and the gates package merged; rulings on what the look sets' and the gates' builders found

Merged: the follow-up before the acceptance run (`e8ff4f60`, `dfd2fbc2`, `a5aac7a8`; the notebook's, trainer's,
monitor's, rows' and look reading's tests on the merged tree "196 passed, 16 skipped") and the gates package
(`3a42df8a`; "41 passed, 1 skipped"). The record-key hashes are the registered pair.

13. **The steps sheet's 5.3 (follow-up check, finding 1).** Before 5.4, the sheet tells the owner to reopen the
    notebook tab; if the last box's output cannot be seen, the owner does not do 5.4 and tells the lead. Two test
    gaps of that check go in with it.
14. **G9-ref while part (a) fails for a reference that is still available (gates check, major 2).** Option A:
    G9-ref fails (exit 1) while a failing item's reference is still available in the set, and its record names
    those references. The lead marks them not available, freezes again and runs G9-ref again. Owed before any
    record.
15. **G9-ref (f)'s join rule (gates builder, point 2).** The rule as written, "a join's joint jump at most
    1e-6 rad", fails on Stage 1's own arrays (up to 1.1e-4 rad) and on V0 (up to 5.3e-5 rad), where nothing is
    wrong. **Changed after a result** (those arrays' numbers, seen on 17 Nov; before any look set is frozen and
    before any closed-loop episode on one): a join's jump at a joint must not exceed the largest change between two
    consecutive frames of the walk it leaves, at that joint. A join then never adds a step larger than the walk's own
    motion. The gate records both numbers.
16. **The bar's set check refuses every Track B row on a look set (gates builder, point 1).** `bar._check_rule`
    lets only R0 rows run on a set that is not one of the 75 cm round's three; the design's sentence that "a look-set
    run needs no --task" (`ROUND_C_DESIGN.md` 2.3) is wrong, the lead's error in the design. `bar.py` is not edited:
    that would move the bar's hash and break D18 (state-A baselines stay comparable). Instead **`lookbar` carries its
    own rule for the look sets**: a narrow, documented context in `lookbar.py` lets only the rows R0L, R5, R6 and
    R2L run on `lookbar.LOOK_SETS`, and only after the set's G9-ref has passed (rule X1); every other set and row
    goes through the bar's own check unchanged. The refit tool (CF) uses the same context. Owed as a change to the
    rows package (and to CF), reviewed, before any look-set record.
17. **The gates builder's other points:** its readings stand; the chain's order stands (V4 after V7; V9 a run rule
    M6 names, not a development episode; V8 runs `train2` directly for the local run, the notebook being Colab's).
    The trainer gains `--expect-baseline PATH` (rule M3, checked before the first rollout) and `monitor2 baseline`
    works without a job; the monitor's S1 stand-in is deleted; the monitor child keeps no timeout;
    `survival_credit` returns the survival time itself when the clock covered 1.05 times it, so R0L's S is exactly
    its mean survival. Owed as the trainer's next change.
18. **The look sets' start (look sets builder, point (b)).** On every set and body the look start passes its entry
    gates and goes over the knee's Stage 0 speed limit (75 cm: 1.126; 60 cm: 1.234; limit 1.0). The cause: its first
    swing lifts a flat foot from an almost straight stand with the look walk's short swing and low lift peak.
    **Changed after a result** (these numbers, 17 Nov; before any look set is frozen and before any closed-loop
    episode on one): a start's lift law, and its duty, mix the stand's law (Stage 1's) into the look walk's by the
    start's amplitude ramp, so the first swings are the stand's and the last are the look's. The builder's
    diagnostic gives a knee peak of 0.85 of the limit on V6/75-M-H-10 with the lift law alone (scratch, no record).
    The moves of all six sets are built again. Any set whose start still fails, and the 60 cm bodies' change to
    0.40 m/s (over the knee limit too, not diagnosed), go to the owner as a question with their numbers. **The 75 cm
    sets come first** (P33).
19. **The V6 turns on 75-M-H-10 fail join_command at events.** The independent review of the join code (L12) is
    ordered before any cut 3; the builder's note that `join_command` may read frames that depend on the command frame
    goes to that review.
20. **The turn in place.** `setscreen` reads every entry as a walk, so a set holding the periodic turn cannot be
    screened. The turn in place stays out of the look sets until `setscreen` learns it, which is a round-A module
    and waits; rule T2 already allows a missing task, and `absent` lists it with its reason.
