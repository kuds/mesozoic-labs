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

## The look sets' check (`checkpoint/reviews/roundC__check_L_17nov.md`), ruled 17 Nov 2026, ~06:20 UTC

Ruled before any look set is frozen and before any closed-loop episode on one.

21. **Where a look set's turns and stop begin (check, finding 1, major).** T1 and B6 say "2 (2.5) strides of
    `v0.4`" before a turn or the stop, and J1 builds the look set's moves to leave the walk there
    (`look.window_a`: a_phase 2.0 or 2.5 strides, a_time_s, a_windows). The env (`r5lib.py` `TURN_LEAD`,
    `STOP_LEAD`) and the bar count that lead in whole windows of the walk's file. On the round's three sets a
    window is one stride, so the two readings agree; on a look set a window is K strides (K = 12 for `v0.4` on
    V6noF5/75-M-H-10), so the env plays about 20 s of walk and then jumps 0.42 rad into a move built for a
    different instant. **This is a code error, not a rule change: the rules as written stand.** On a set whose
    `set.json` carries `look.window_a`, the env and `lookbar`'s look-set context (ruling 16) end the walk at J1's
    leave point: they play the walk for `a_windows` of its window (the move's own `a_time_s` at the commanded
    clock) and start the move at the frame it was built from. Every other set keeps today's code, bit for bit.
    The phase check for a look set counts strides inside the window (left at a whole stride, right at a half),
    in `lookbar`, because `bar.check_phase` reads a lead mod 1 and `bar.py` is not edited. The bar's own record
    of a look-set turn or stop keeps its whole-window placement and is printed as "the bar's reading (C7)" beside
    the companion's, as ruling 5 of the third Track B list says. A test plays each look-set turn and the stop
    through the env and asserts that the largest joint change at the join is within the walk's own largest
    frame-to-frame change (ruling 15's limit). Owed as a small package after the `lookbar` rule (16) is merged,
    reviewed, before G9-ref is run on any look set. Cut 3 is not taken for this.
22. **The turn in place's numbers (check, finding 2).** Ruling 20 is amended: `setscreen` lists a turn in place
    that is not available without screening it, so the rebuild under ruling 18 builds `inplace_0.5_left` and
    `inplace_0.5_right` for each set and records them as "not available" with their numbers (75 cm, as the check
    measured them on a copy: thigh-twist use 0.9114 and 0.9115 against 0.87; hip-yaw margin 0.0266 rad against
    0.05). They stay out of training: a turn in place that is not available is never a task (T2).
23. **`blocked` names every failing group (check, finding 3).** `set.json` `blocked` keeps a list, so a 60 cm set
    whose `start` and `nominal_to_v0.4` both fail names both. In the rebuild.
24. **The sets builder's scope (check, finding 4).** Accepted as built: the look sets live in their own registry
    (`LOOK_SETS`), `sets.py` is in no record key, and the round's three sets verify as before (the check ran them).
25. **The reach of a toes-up swing after a change (check, finding 9).** The builder's reading is accepted and
    written into L10 with the label "settled 17 Nov 2026, after the look work package built sets (prescribed
    motion; none frozen), before any Track B run": a swing that lands at or before the change's instant keeps
    gait a's reach; a later swing has a reach only when both its leaving and its landing stance follow a walk's
    law. Otherwise it has none.
26. **The other minor points (check, findings 5 to 8).** All go into the rebuild under ruling 18: `join_events`
    uses the moves' join limit (4.0 mm) and pads instead of wrapping; turn reasons name only the gates that
    failed; `tail_phase.json` is rebuilt on a clean tree before anything is frozen (nothing is frozen from a
    dirty tree); the moves test writes into its tmp path. Finding 10 (the 60 cm change to 0.40 m/s) is not
    diagnosed and goes to the owner with its numbers after the rebuild if it still fails (ruling 18).
27. **`sets.py` leaves D18's state-A test (on merging the look sets, 17 Nov 2026, ~06:45 UTC).** The look sets'
    package adds the look sets' registry (`LOOK_SETS`) and their builder to `sets.py`, which D18's fast test
    (`tests/test_stage2_lookbar.py`) lists among the files that must equal state A. D18 exists so that no record
    key moves and state-A records stay comparable. `sets.py` is in no record key: the bar's and the panels' source
    hashes stay the registered pair, and `sets verify` on V0, F25F9 and F25F9cl gives the same counts as before
    (ruling 24). So `sets.py` leaves the test's list with this reason, as `barreport.py` did. The other files of
    the list stay; `moves.py` is unchanged by the package. D18's text in `ROUND_C_DESIGN.md` carries the
    amendment. The next independent check of a round C package reads this change.

## The join-code review (`checkpoint/reviews/roundC__review_JR_17nov.md`, sha256 `b3ff457673d537f65d8821c1910c27b6bebc7eb27acd43f0d0e0e96d9424212a`), ruled 17 Nov 2026, ~06:55 UTC

This is the independent review of the join code that L12 requires, ordered by ruling 19. Its record is the file
named above. Every rebuilt look set names it, by path and sha256, in `set.json` `rules`. Ruled before any look
set is frozen and before any closed-loop episode on one.

28. **The preview grid of a causal move is made causal on the look sets (review, gate 3).** A6 says that the
    preview controller first sees the command at the command frame. But `moves.com_path_causal` takes its grid,
    and so its gains, preview length and switch frame, from the smallest frame interval of the whole plan, and on
    the V6 turns that interval lies after the command (the turn's quicker steps, or window b's fade-in). So the
    warm-up, and `join_command`'s warm-up reading, depend on what comes after the command. On the V6 turns of
    75-M-H-10 this is the whole spread of 5.54, 4.40 and 3.54 mm (left) at stride factors 1.0, 0.8 and 0.6.
    **This is a code error against A6, not a rule change.** On the look sets, a causal move's controller run
    takes its grid from the frames up to the command frame alone, `min(diff(t[:k_cmd + 1]))` (5.2244 ms on
    every V6 move). Everything else stays as coded, and every gate of the move, after the command included, is
    read again on the rebuilt arrays. `moves.py` is not edited (D18). The look path passes this grid from
    `moves_look.py`; if no path exists without a change to `moves.py`, the builder stops and reports it to
    the lead. The round's three frozen sets keep their records: they were built with the same grid, and their
    warm-up readings are all within 4.0 mm (V0: 1.349, 1.289 and 2.114 mm; F25F9: 2.083 and 3.485 mm; F25F9cl:
    1.249, 2.321 and 1.307 mm; joints at the command within 1e-3 rad). A6 gains a dated note that records this
    and says the records stand. The review's scratch reading on the causal grid: the left turns read 4.621 mm
    (over 4.0 mm, at every stride factor), the right turns and the stop read 3.868 mm.
29. **Cut 3 is not taken where it does not lower the failing reading.** **Changed after a result**: the review's
    scratch probes of 17 Nov, prescribed motion only, before any look set was frozen and before any Track B run.
    L12 takes cut 3 for a causal move that fails `join_command`. On 75-M-H-10, at cut 3's leave point (the
    window's boundary) and on the causal grid, the same reading is 4.038 mm for the left turns and 4.361 mm for
    the right turns. So cut 3 would fail every turn, and by the same reading the stop too, where J1's leave point
    fails only the left turns. The cause is not J1's leave point: the controller re-runs the walk, the CoM differs
    from the window's by at most 0.066 mm, and two differences in time turn that into about 4 mm of ZMP. That is
    what A7's P34 paragraph says of the round's sets. So L12 gains a sentence: cut 3 is not taken for a body when
    the join-code review's probe, on the causal grid, gives the failing reading at cut 3's leave point as over
    its limit too. The move that fails at J1's leave point is then "not available" with its numbers (A11 for a
    turn; H3 for the stop). No limit is changed. The 4.0 mm limit is the owner's (P34), and whether to change it
    for the look sets' left turns goes to the owner as a question with the rebuilt numbers, not before.
30. **What runs next.** After the start's law (ruling 18, building now) is checked, one rebuild of the 75 cm
    look sets carries rulings 18, 22, 23, 26 and 28 together and names this review in `set.json` `rules`.
    Then comes an independent check, then the 60 cm sets. Nothing is frozen until the lead has read the rebuilt
    numbers and, where a question goes to the owner, the owner's answer.

## The checks of LB16, CG2, CT4 and CN3, ruled 17 Nov 2026, ~07:30 UTC

LB16, CG2 and CN3 are merged as they stand. CT4 goes to a fix round.

31. **The look sets' rule in `lookbar` (check of LB16).**
    - (1) On a look set (or a version of one), `lookbar` refuses every row outside the families R0L, R5/<seed>,
      R6/<seed>, R5G/<seed> and R2L-<set>/<seed>, inside the context and in `episodes`. Every admitted row needs a
      passed G9-ref, so a closed-loop R0@ row on a look set before G9-ref is refused (X1). R5G is admitted: the
      rules draft lists it among the Track B rows with bar records (rule at line 1348). `python -m
      designlab.stage2.bar run` on a look set stays open, because `bar.py` is not edited. X1 forbids it, and no
      step of the chain calls it; X1's text says so.
    - (2) `R2L-<set>` names the evaluated set exactly, as built.
    - (3) A version of a look set (V6.2) is a look set everywhere: in the context, in `episodes`, in `collect`
      (reading 1) and in `gates2.require_g9_ref`, which covers versions. `lookbar` then calls the gates' check
      instead of its own copy.
    - (4) The context restores the bar's check only if its own is the one installed, and raises otherwise. A G9-ref
      record whose `set_freeze` is not an object is refused cleanly, in both modules. The check's two test gaps
      (a non-`evaluate` caller; `pass_` missing, None or 1) get tests.
    - (5) The companion record carries the G9-ref citation (path and sha256) under which its episodes ran.
    - (6) The monitor treats `LookSetRefused` as a refusal of the monitor (exit 8), not a crash.
    - (7) The context enforces X1's first clause only; "between G9-ref and S2-4's start only these" stays with the
      chain.
32. **G9-ref (check of CG2).**
    - (A) Ruling 14's procedure cannot finish today, because `sets.freeze` refuses to drop a frozen entry.
      Option (i): `sets freeze` may drop an entry that the set marks not available, and the freeze records it with
      the G9-ref record (path and sha256) that named it. A `sets` command marks an entry not available, with its
      reason and that record. Built with the look sets' rebuild (ruling 30), in `sets.py`.
    - (B) "The walk it leaves" (ruling 15) is the walk's whole window, as built. On a look set fewer frames are
      played before a turn, but the window is the walk's own law, which the policy meets in every walk.
    - (C) The builder's readings stand: a join that leaves no walk is held to the walk the track played last
      before it (else the first after it; the stand task to 0), and the wrap from the window's last frame to its
      first counts as a consecutive pair.
    - (D) A turn or stop item's reference is read from the refcheck item's `plan`, not from the sorted file list.
      The printed lines that name a failing reference get a test.
    - (E) The lead writes rulings 14 and 15's words into the old texts: the rules draft (the 1e-6 rad sentence,
      and "treats that reference as not available") and the design's section on G9-ref.
33. **The trainer (check of CT4).** Findings 1 and 2 are fixed as the check proposes. `baseline_equal` requires
    digests only for pairs in which at least one episode ran. A digest counts as compared only when both sides hold
    a non-empty string. The rule's text says "in every episode that ran". Tests: an equal baseline with a not-run
    condition lets the run train, in `train2` and in `monitor2`. Point 3 stands as built: a mismatch halts (exit 5).
34. **The steps sheet (check of CN3).**
    - 5.3 (e) becomes a positive test: go on to 5.4 only if what the last box printed ends with "Disconnecting
      runtime in 5 seconds..."; if the box is still running, or it printed anything else, do not do 5.4 and tell
      the lead.
    - Before Run all, from the second session on, the sheet has the owner clear all outputs ("Edit > Clear all
      outputs"; expected, not yet seen).
    - 5.3 (a) names the smoke's copy, "s2 smoke-1".
35. **The CF follow-up**, as owed. `r2lfit` accepts a G9-ref record through `gates2.require_g9_ref` (CG's format,
    versions included) and runs its look-set records through `lookbar`'s context (ruling 16). The chain resumes V10
    with `--resume` only when `cma.json` exists, and the V10-collect step goes. The stale V3 note of
    `run_trackb.sh` is corrected.

## The start's law (`checkpoint/reviews/roundC__{build,check}_LS_17nov.md`), ruled 17 Nov 2026, ~08:00 UTC

Brought in by cherry-pick: the check found no defect, and with the new start law every start passes its gates on a
scratch build (75 cm: knee 0.80, 0.85 and 0.79 of the limit; 60 cm: 0.92, 0.98 and 0.90).

36. **The builder's points**, all accepted as built:
    - The turn in place's start keeps its own law (L13), so it is not mixed.
    - The start's mixed duty also times the bob's trough and the roll's phase in the first stride, as it does in a
      change.
    - `start_v0.4` takes the same mix from Stage 1's start into nominal (Stage 1 has no start into `v0.4`).
    - The turn reasons are per direction on the look sets only.
    - `blocks_in_records` stays, and so does the rule-text merge of 403987c6.
    - The check's two low points go into the rebuild's code: `verify` also names a stop that `blocks_in_records`
      shows and an older `blocked` leaves out, and a test builds with two processes so that the start and the
      change are both named.

## Wave 8 (`checkpoint/reviews/roundC__{build,check}_{LT,CT5,CF2,CN4}_17nov.md`), ruled 17 Nov 2026, ~09:05 UTC

CT5, CN4 and CF2 are merged as they stand. LT is merged on its check's condition: its two latent faults are fixed
before anything runs on a look set, and nothing can run there until G9-ref has passed.

37. **The look sets' placement, everywhere it is read.**
    - (a) LT's reading of ruling 21 is confirmed. The bar's own runner, outside `lookbar`'s context, keeps its
      whole-window placement. A record made inside the context (`lookbar run` on a look set) holds the placed plan,
      its seconds and `command.leave`, because a record describes the episodes it ran.
    - (b) Every reader that plays a look set's turns or stop plays them at J1's leave point. `lookbar` exposes a
      placement-only context (placement without the row rule or the G9-ref requirement, because G9-ref itself runs
      before G9-ref has passed). These run inside it on a look set:
      - G9-ref part (a) (`gates2._refcheck_record` and `bar.refcheck`);
      - G8a (`gates2.g8a`);
      - `s2rows.equivalence_check`;
      - R2L's fit and selection (`r2lfit`: its evaluators are built inside the set's placement, in process and in
        every pool worker; check of CF2, finding 1).
      A test shows that a fit's turn plan on a look set equals `lookbar.leave_resolve`'s. `bar dry-run` keeps the
      bar's own placement: it estimates cost and decides nothing, and its overstatement (about 20 s per turn and
      stop) is written beside its numbers.
    - (c) and (d) accepted as built: `LookRow.sources` leaves out `r5lib.py`, because the key carries the placed
      plan; the companion always holds `g9_ref`. The optional hardening is taken: `leave_point` checks `a_time_s`
      against the walk file's own `phase` array.
    - The check's F1 and F2 are fixed:
      - F1: on entry, the context releases the process's cached bar context for the set.
      - F2: on exit, the context resets the rows it stamped.
      Each fix gets the test the check proposes.
38. **The other points.**
    - CT5: the refusal kind `look_set`, exit 1 at a further delay (M6), and a pair whose digests sit under different
      keys counting as not compared are all accepted.
    - CF2: `select` is gated as well as `fit` (X1).
    - CF2: G9-ref is read by the gates and by the context, with the same-file check.
    - CF2: a version of `V6noF5` counts as `V6noF5` for Q1 (`FITS` and `role_of` read `look_set_of`), as
      ruling 31 (3) says.
    - CN4: readings (a) to (d) are accepted. Its check's optional test gap ("Run all" stands only in 3.5, 4.4 and
      5.4 (d)) goes into the next change to the sheet.
    - Rule M3 gains ruling 33's words, "in every episode that ran".

## LT2 (`checkpoint/reviews/roundC__{build,check}_LT2_17nov.md`), ruled 17 Nov 2026, ~10:20 UTC

Merged as it stands.

39. **The last readers of a look set's turns, and the builder's choices.**
    - (a) Option (i) is accepted: G9-ref part (a) reads a record from a file only if that record holds the placed
      plan; otherwise it runs its own placed check, and a given unplaced record blocks.
    - (b) The two remaining readers of a look set's turns run inside the placement-only context:
      - G9-ref part (b)'s playback of walk and turn items (`gates2.part_b`);
      - the owner's views (`barviews.SetView.planned`).
      A placed refcheck command (in `gates2`, since `bar.py` is not edited) gives the look sets' freeze a placed
      record. The look sets' freeze text in `sets.py` names it once the rebuild is merged.
    - (c) The env's own task placement passes the walk file to `leave_point` (`r5lib.place`,
      `Library.place`), so ruling 37 (c)'s check covers the env too.
    - (d) and (e) are accepted: the rule texts that enter records stay unchanged and the docstrings carry the
      change; the tolerance is one frame of the walk file.
    - `bar dry-run`'s overstatement on a look set is written into the cap rule (Q5) and beside each cap's printed
      total.
    - A V0 comparison test for part (a) and G8a (`needs_v0`) is added where the arrays exist.

## LT3 (`checkpoint/reviews/roundC__{build,check}_LT3_17nov.md`), ruled 17 Nov 2026, ~11:50 UTC

Merged as it stands.

40. **LT3's points.**
    - The builder's (a) to (g) are accepted as built:
      - `place` requires the set's directory;
      - the refcheck record keeps bar refcheck's own format, with no "placed" key;
      - in a rendering process a look set's plan is made by a child process.
    - Owed as one small tidy-up package, with a check, before G9-ref runs on any look set:
      - the check's finding 1: the placed refcheck command removes the record it wrote when it refuses an
        unplaced item, with a test;
      - finding 2: `gates2._SetView` refuses a set.json that names another body, as it refuses another set;
      - the stale docstrings: `r5lib.leave_point`'s "None: not checked"; `lookbar.look_set_placement`'s list of
        readers, which gains part (b), the views and the placed refcheck.
    - The look sets' freeze text in `sets.py` names the placed refcheck in the six places the check lists (lines
      about 63, 145, 2478, 2662, 2768 and the `--refcheck` help at 2906). This is done by the next change to
      `sets.py` after the rebuild is merged, in the words the check suggests.

## The 75 cm freezes and the first validation (`checkpoint/reviews/roundC__{do,check}_{freeze75,validate75}_17nov.md`), ruled 17 Nov 2026, ~18:00 UTC

V6/75-M-H-10, V6noF5/75-M-H-10 and V6/75-M-H-10-X are frozen (freeze 0). G9-ref passed on each; the check re-ran it
and found every part identical. G8a passed on V6/75-M-H-10 (32 of 32 episodes identical). R0L's records ran
(13 conditions; the four quick-turn conditions not run, their references being unavailable). G9-run failed on part
(d), and the validation stopped there.

41. **The mixed freezes are accepted.** The three freezes used `--allow-mixed`: P42 rebuilt only the stop and turn
    groups, so the entries come from two source states. Rule S2S-freeze allows a mixed freeze when it is recorded,
    and each freeze records `mixed_sources: true` and `dirty: false`. The moves.py condition (A4) held. The P42
    check found every rebuilt entry's arrays bit-equal where nothing else changed. The freezes stand. The check's
    two small corrections are noted: in the -X form of (b), toe-off differs by 1e-9 to 3e-9 rad, not by exactly 0.0;
    and a frozen entry cannot be rebuilt in place.
42. **G9-run (d) and conditions that are not run (code error, not a rule change).** `gates2.part_d` reads every
    companion under `look/R0L@10ms/` and fails any bar record whose status is not "done". The four quick-turn
    conditions have no available reference, so the bar records them as "not run" with no episode. Rule Z2 (d) speaks
    of the episodes of the records, and H3 says that a reference marked not available leaves its conditions "not run".
    G9-ref (a), G8a and `_expected_conditions` already treat such a condition as "not run, ok". `part_d` is changed
    to read only the expected conditions' companions and to list the not-run ones as "not run", with a test. The
    companion of a not-run condition also carries the G9-ref citation (ruling 37 (c)). The failed record is kept
    beside the new one as `g9_run.first.json`. The new run is labelled "settled after G9-run's first result: a code
    error, the rule unchanged". Validation then goes on from G9-run: V5, V6, V7, then V4 alone.

## S2-4's first start (`checkpoint/reviews/roundC__{do,check}_s2-4_17nov.md`), ruled 17 Nov 2026, ~19:35 UTC

R0L's baseline was made, copied and committed (58109a17, sha256 f33e4eb1…0543) before the run, as M3 says. The
trainer then stopped with exit 1 after 32 s, before G8a's record and before any rollout. The cause:
`train2._g8a` passes `at=` beside `**res`, and on a look set G8a's bar form already returns `at`
(`TypeError: dict() got multiple values for keyword argument 'at'`). Training on a look set had never been run, so
the validation steps did not reach this path.

43. **The fix, and S2-4 started afresh.**
    - **Code error, rule unchanged:** `train2._g8a` keeps the form's own time as `form_at` and stamps the record's
      `at` itself. A test on a synthetic look set in a temporary directory makes a bar-form G8a record and fails
      without the fix. The builder also reads `train2.py` for any other `**` spread beside explicit keys on the
      look-set path, and reports or fixes each one with a test.
    - **M3's "S2-4 is run once":** the failed start does not count as S2-4's run. It took 0 steps, ran no rollout and
      wrote no G8a record. Its `run.json`, `progress.json` and log are kept as a record under
      `stage2/tb/runs/s2-4/failed_start_17nov/`. The ignored job directory under `stage2/tb/work/local/s2-4` is then
      cleared, and S2-4 starts as a new run with the fixed code. The committed baseline stays: its commit is an
      ancestor of the new run's first session.

## The launch (`checkpoint/reviews/roundC__{do,check}_bundle_17nov.md`), ruled 17 Nov 2026, ~21:45 UTC

The bundle s2-b1 (75-M-H-10, V6 freeze 0; content 9227d8a0bf0b…; three parts, 33,812,480 bytes) is checked sound.
The launch commit C1 is b914088b. Its parts are on the data branch `claude/mesozoic-trex-stage2-bundle-s2-b1`
(ecf95a81, P46), and a fresh fetch gives each part's sha256 as built.

44. **The first launch series and the -X head.**
    - Launch series 1 trains 75-M-H-10, the plain head, alone (P33, P43, P44). The -X head waits for D5's
      preconditions (V0: G1 of the -X bodies on an idle machine, and the jaw cover decision), so it is not in s2-b1.
      A later bundle with its own id adds it (K6).
    - Rule O says that a body's -X head is probed before any Colab job of that body is queued. That is met now:
      rule O3's learning-rate probe runs on 75-M-H-10-X on V6 (frozen, G9-ref passed; X1 allows the probes between
      G9-ref and S2-4) and its record joins `lr_probe.json`, before the steps sheet names the queue. If the jaw cover
      later changes the -X records, the probe is run again then.
    - D24's committed rules are named in the steps sheet by path and by sha256 at C1:
      `STAGE2_TRACKB_PLAN.md`, `stage2/design/ROUND_C_DESIGN.md` and this file.
