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
