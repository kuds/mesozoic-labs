# check:TB6 (17 Nov): the agent's final report

Working paper. Nothing here is a rule.

VERDICT: merge after the listed fixes. Fix 1 should go in before the merge. Fixes 2 to 4 are small and recommended. Everything else I checked is right. The writing-in is faithful, labelled and complete, and the code reading for ruling 3 matches monitor2.py and train2.py.

FINDINGS (most serious first)

1. Medium, a defect. STAGE2_TRACKB_PLAN.md l.1011-1012 (M1's note) and l.3057 ("What else changed").
   - The problem: the note defines ruling 3's "wrong shape" as only the shape of the record's `files`, which is all the code checks ("the shape being that its `files` name each of the three files with a whole number of bytes and a sha256"). l.3057 says the same.
   - What it contradicts: the fourth amendment calls a record without a `layout` object, or with a `fingerprint` entry that is no object, "one wrong shape ... not yet held" (l.2998 and l.3030-3036). The lead's ruling 12 of 17 Nov (ROUND_C_RULINGS_17NOV.md, last bullet) makes that record a refusal.
   - Why it matters: M1's Check reads refusals "on any ground of rulings 6 and 3 (M1's note names the code's kinds)". A reviewer reading the note would take the case below as correct under the rule, while the amendment and ruling 12 say it is a refusal the code does not yet hold.
   - Evidence: monitor2._checkpoint checks only `files`. s2rows.build then does `record["layout"]`, which gives a KeyError: exit 1 and a counted attempt (roundC__check_CT3_2.md finding 1).
   - Fix: reword the note to "the code checks the shape of its `files` alone (...); a record without a `layout` object, or whose `fingerprint` entry is no object, is a wrong shape too (the lead's ruling 12 of 17 Nov 2026), which the code does not yet refuse (the fourth amendment)". Change l.3057 to match.

2. Low, a defect (a fact). l.186: "Branch `wt/wp-l` (at `2237ba9c`, its head when the writing-in began)".
   - 2237ba9c was committed at 04:23:11 UTC. The writing-in began at 04:19 UTC (merge 3f7d32de at 04:19:35, and l.183 says "from 04:19 UTC").
   - At 04:19 the head was a024cbe8 (04:04:10). `git diff a024cbe8 2237ba9c -- stage2/sets/` is empty, so the six sets read are the same either way.
   - Fix: "(at `2237ba9c`, its head when it was read, 04:23 UTC on 17 Nov)".

3. Low, a defect (ruling not carried to M5). l.1189-1190 (M5's pointer paragraph): "After the first of the two the lead has the fault repaired and resumes with M1's recorded parameter."
   - It does not say that after the second halt (a refusal of the attempt that the parameter let run) the resume needs the parameter too. M1's note (l.1039-1044), K5 (l.1736-1739) and the closing paragraph do say it.
   - Read alone, M5 suggests no parameter after a refusal.
   - Fix: add "; after the second, where the refused attempt was the one that parameter let run, again with it (M1's note, K5)".

4. Low, a defect (labelling). l.2167-2168 (L12, ruling 4).
   - The ruled sentence reads "the set whose `start` or `nominal_to_v0.4` fails". The lead wrote "whose start fails".
   - The widening is defensible: the confirmed reading said "the set that fails", and the paragraph above names both failures. But the mark does not say the words were changed, and the report calls it "the lead's sentence".
   - Fix: say so in the mark, or use the lead's words exactly.

5. Low, for the lead (a pointer). l.1780 (K9): "A checkpoint that fails is not used and, if it was to be judged, the owner is told."
   - It has no pointer to M1. At a monitor check, a fingerprint not reproduced at the row's load is now a refusal (ruling 3): the trainer halts and the run goes to the lead.
   - It is not a strict contradiction. The choices: (a) add "(at a check of the monitor: a refusal, M1)"; (b) leave it.

6. Nit. l.989: "What follows from the three paragraphs above". Six settled paragraphs now stand above it; the count was already loose before this item. A possible fix: "the paragraphs above".

7. Nit. l.1009-1010: "Ruling 3's three: a `run.json` that is missing, cannot be read or is no JSON object". The ruling names only a missing run.json; "cannot be read" and "no JSON object" are the code's wider kind `run`. A possible fix: say so.

On the author's "for the lead" items:
- Item 1 (`s2rows.RowError` named only in M1's note) and item 2 (the halt names --retry-check only for the attempt marked `after_halt`) are written as the code reads them. Both are the lead's choice.
- Item 3: the two changes owed in other modules are right. The `_checkpoint` layout/fingerprint fix belongs to the follow-up before the acceptance run, by ruling 12 of 17 Nov. The fifth control (the tail scaled to a tenth) goes to the gates package's brief; it is not in `lookread.DEGRADED`, which holds bob_half, neck_3deg, tail_late and flat_feet.

WHAT I CHECKED AND FOUND RIGHT

Merge and diff:
- wt/prereg-b was an ancestor of the working branch. 3f7d32de has parents f8067851 and fccebe42, and its tree equals the working branch's.
- The draft at 3f7d32de is byte-identical to the main checkout's copy (cmp).
- `git diff claude/...4jffre HEAD --stat`: only STAGE2_TRACKB_PLAN.md, "1 file changed, 325 insertions(+), 83 deletions(-)".

Ruling 1:
- Rule l.952-955, head tags, Check l.1049-1051, with the old sentence quoted.
- `monitor2.baseline_problem` takes read, fell or not_run and nothing else.

Ruling 2:
- Z2 head, (c) "five" with the old sentence quoted, the note, the Check, and the V6noF5 count of four.
- The note's mechanism holds in the code. `lookread._fit_phase` is a linear least-squares fit of `A sin(2 pi phi + psi) + c` on the tail-yaw joint angle. Scaling the tail about its mean therefore scales A by 0.1 and leaves the phase unchanged. Bob, head and toe-off are read from the hips, the head and the feet.
- The 0.120-0.125 rad range matches the reports: smallest 0.11998 rad (roundC__build_followups2.md l.76) or 0.119983 rad (roundC__review_followups2.md l.51), largest 0.125268 rad, from kinematic playbacks only, no record.
- The fourth amendment's account of the tail band in lookread.compare matches the code: NOT_READ_RAD and TAIL_AMP_MIN_RAD are both 0.02, compared without slack; an amplitude that cannot be computed fails; and it records tail_amp_rad, tail_amp_side, tail_amp_min_rad, tail_amp_ratio and the reference's amplitude and side.

Ruling 3, read against the merged code:
- The refusal kinds are ("sources", "checkpoint", "other_run", "run", "row"), and EXIT_REFUSAL is 8.
- `_job_doc(refuse=True)` covers a run.json that is missing, cannot be read or is no object. `_checkpoint` covers a missing record, an unreadable record, a record that is no object, a `files` of the wrong shape, and a file that is not the one its record names.
- A row's refusal raised while the row is built is a refusal: `_row_refusal` in `_episodes`. lookbar.rows calls s2rows._factory, which calls build and then load_checkpoint, so the fingerprint is checked within 1e-5 before any episode.
- A refusal at a further delay (`_also`) becomes MonitorError and exit 1.
- main() unlinks the note before it runs. `_note_refusal` writes monitor_<n>.refused.json, or r0l.refused.json for the baseline.
- train2: the attempt is marked refused, with its refusal, and left out of the `closed` count. Halt exit is 5. The baseline halts with "R0L baseline could not be made". The attempts never closed stay open after an after_halt refusal, and only then does the halt name --retry-check.
- monitor_env passes STAGE2_ALLOW_SOURCE_CHANGE only with the session's flag. nb.py on exit 5 writes remedy ["retry_check"] when the halt names "--retry-check", and on exit 4 writes no HALT.json.

Ruling 4, the frame and the facts:
- Ruling 4: rule, head, Check, brief l.284 and item 30.
- Frame: "four amendments" in Status, brief, Tags and the change-list opening; all the pointers in the third and fifth amendments.
- Commits and times cited: 34afe717, 0ad2072d and bf9bc434 at 19:52 on 16 Nov; 695e8429, 9c348ca7 and eee35062; ff04355a at 03:27:29 and 8348a3a8, both on 17 Nov; bc7ca202; fccebe42. The list itself was committed at 19:56:01 (fdb31d58).
- The six set.json files on 2237ba9c all have `freezes` empty and the `blocked.by` values as stated. 33652cb is at 19:15:57, and the five others run from 20:53:15 to 04:04:10. The working branch holds no V6 set and has only three files in stage2/tb/.
- The V0 check episodes (49 + 8) match roundC__build_CW.md l.39 and roundC__review_CW.md l.64.
- Every cited review file is tracked.

State and commits:
- The record-key hashes are the registered pair: `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1`.
- The worktree is clean, with no ignored outputs (`git status --porcelain --ignored` is empty). The main checkout is clean.
- All six commits (3f7d32de, d8eca747, 14fe4d7c, eb9ec6f8, 0373c4ee, ac86c4d5) end with the two given lines and name no model anywhere else. None lacks them.

Mutation probes, on a scratch copy built from `git archive` (study dir, environments/, configs/, conftest.py, pyproject.toml); every probe was caught:
- monitor: run.json refusal off: "1 failed, 25 deselected"; files-shape check off: "1 failed, 25 deselected"; row refusal not recognised: "1 failed, 25 deselected"; baseline rejecting not_run: "1 failed, 25 deselected".
- train2: halt never naming --retry-check: "1 failed, 52 deselected in 26.46s"; refused attempts counted: "1 failed, 52 deselected in 26.90s".

TESTS (in the worktree, through tools/lowprio.sh, standing environment, -p no:cacheprovider)
- tests/test_stage2_monitor2.py, whole file: "24 passed, 2 skipped in 19.73s". The two skips: "lookbar is merged: the stand-in is no longer reached" and "Stage 1's arrays are not restored".
- tests/test_stage2_train2.py -k "refuses_to_run_is_no_attempt or kill_around_a_refusal or refusal_without_its_note or passes_its_allow_source_change": "4 passed, 49 deselected in 54.57s".
- The author's selection `-k "refus or exits or baseline"` collects 6 tests, as reported.

WHAT TO BRING IN
- After fix 1 (and preferably 2-4): fast-forward the working branch to the new tip of wt/prereg-b. As it stands now, the tip is ac86c4d5. The working branch tip fccebe42 is an ancestor. `git merge-tree --write-tree` gives tree 91de77e3, which equals ac86c4d5's tree: no conflict, and only STAGE2_TRACKB_PLAN.md differs.
- No commit needs cherry-picking separately.

NOT CHECKED
- The full test suite, and the train2 tests beyond the four named.
- The worktrees of wt/wpc-cg and wt/wp-l, beyond the two set.json keys and the commit dates.
- The CN and CT3 reports beyond spot checks.
- The design ROUND_C_DESIGN.md against the new Z2 (c).
- Whether a frozen set's reference tail amplitude will stay under 0.2 rad.
- No episode, fit, training or dry run.

Scratch: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave6/check_tb6/ (plan.diff, mutate.sh, monitor2_tests.log, train2_refusal_tests.log, mutate_monitor.log, mutate_train.log, and the scratch copy under mut/).
