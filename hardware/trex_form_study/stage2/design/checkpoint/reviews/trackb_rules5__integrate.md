<!-- Working paper, 16 Nov 2026: the integrator's report on writing the lead's fifth list of rulings into the Track B rules draft. Nothing here is a rule. -->

## Fifth list written into `STAGE2_TRACKB_PLAN.md` (worktree `/home/user/mesozoic-wt/prereg-b`, branch `wt/prereg-b`)

Merge first: the working branch (`ffecc2c`) was ahead of `wt/prereg-b`; merged with `--no-ff` (`01c8341`). `git diff claude/mesozoic-trex-stage2-refine-4jffre -- .../STAGE2_TRACKB_PLAN.md` was empty after the merge: the two copies were identical. Now only the draft differs from the working branch. I ran no episode, fit, training, dry run or test; only git, grep, text scripts and the hash line, which printed the registered pair `45719d03...7528d c0027992...1be1c1`. Line numbers are the file's at `0710a90` (2,753 lines). Every new wording carries "[settled 16 Nov 2026, before any Track B run]" and "ruling n of the fifth list".

### Per ruling

1. **M1, a record with a "not run" condition** (M1 rule 902-907, note 921-923, Check 953-958; head tags 837-839).
   - Old: "A record of a check is a `monitor_<n>.json` that names the checkpoint's hashes and holds an outcome for all 18 episodes." (quoted once in the mark).
   - New: "... and holds, for each of the 18 episodes, an outcome or, for a condition without a reference, "not run"".
   - Note: "or holds fewer than 18 outcomes" -> "or that holds neither an outcome nor "not run" for one of the 18 episodes". Check: "holds 18 outcomes" -> the new phrase (`episodes[].outcome` is then `not_run`, the monitor's name).
2. **After the halt for two attempts never closed** (M1 909-911, note 926-930, Check 959-962; K5 1564-1573, Check 1578-1579; pointer paragraph in M5 1068-1072).
   - New in M1, the lead's words: "After the halt for two attempts at one check that were never closed, the lead has the fault repaired and resumes with a recorded parameter, which lets the check run once more per resume. A run that cannot be resumed is `unfinished: <reason>`."
   - M1's note lost "and what follows it is the lead's to say" (quoted). K5's author's sentence "the check is not run a third time: the run halts and goes to the lead (M1)" now reads "a resume without M1's recorded parameter does not run the check a third time ..." (old quoted).
   - Checks name `sessions[].retry_check` and the attempt's `after_halt`, as `train2.py` has them.
3. **Q5, S2-9 asking late** (Q5 head 1322-1324, reading 1376-1392, new paragraphs 1394-1407, Check 1451-1469; brief 248-251; change list item 15).
   - Old author's reading (quoted once): "if S2-9 asks later, the dry run is made again with all four delays and committed before the first record at another delay; 1.5 times its total is the row's cap from then on, and what the 10 ms unit spent counts in it".
   - New: "Where S2-9 asks for the three other delays after a row's cap is committed, those delays get a cap of their own, with its own dry run, run directory and `budget.json`."
   - Check worded per cap; its mark names each changed phrase ("the row's first panel record" -> "the first panel record the cap covers"; "each cap holding the units of its own delays" added). Brief: "before that row's first record" -> "before the first record it covers".
4. **L12, the `-X` head's sets** (L12 head 1960-1961, rule 1985-1988, note 1989-1994, Check 2021-2024; brief 234-236; change list item 30).
   - New: ""That body's other look sets" takes in the sets of the body's `-X` head: the paragraph above itself takes a cut for every look set of that body and of its `-X` head." No sentence replaced.
5. **The tail's amplitude band** (U3 head 1110-1112, rule 1125-1128, reading 1130-1138, Check 1140-1143; U1 1100-1105; list of words 319-320; change list item 24).
   - New in U3, the lead's words: "The tail term fails when the robot's fitted tail amplitude is under 0.02 rad while the reference's amplitude is read." Tagged NEW and "differs: plan section 5, whose tail term reads the phase alone"; the mark says the ruling came after the review's probe on synthetic series (no record).
   - U1 names the amplitude of its own sinusoid as the "fitted tail amplitude". Words: "reading 1's four limits" -> "four terms" (old quoted).
   - Code today, said in the closing account only: `lookread.compare` reads the phase alone, records `tail_amp_ratio`, and its rule text says "no band reads it".
6. **A monitor that refuses to run** (M1 913-916, note 932-945, Check 962-964; pointers in M4 1029-1031, M5 1068-1072, K5 1573-1575).
   - New in M1, the lead's words: "A monitor that refuses to run (its sources have moved, the checkpoint is not intact, or the checkpoint is another run's) is no attempt: the trainer halts, and the refusal does not count toward the two attempts after which a check counts as without a gain."
   - The note names the three as `monitor2.py` does: "the sources differ"; "checkpoint <n> is not intact"; "another run's checkpoint is not judged".
   - Code today, in the closing account only: all three raise `MonitorError`, the monitor exits 1 like any failure, and the trainer counts the attempt. The change is owed.

**Confirmations.** Q5: a cap's run directory is `stage2/tb/bar/<run>/` with a `DRY_RUN.json` at its top (rule 1397, Check 1453-1455; the old sentence "The rulings name no path ..." quoted once). The candidate's two caps are made again (mark at 1383-1384). E4's mark says the lead confirmed its sentences (455). K4: no word changed.

**Frame.** Status (45-51), "What had been run" (132-148), brief (276-282), Tags entry (375-379), change list opening, a third closing amendment (2672-2753) with what the code holds today, and pointers in six places of the two earlier amendments.

### Points of wording I settled
- "the two amendments" -> "the three amendments" (Status, brief, Tags entry, change list opening).
- M1's note: a checkpoint whose record is missing is read as "not intact" (`s2load.py` defines intact so); the refused attempt stays in `monitor[]`, closed with the monitor's own exit code (the 19:01 UTC round C ruling); on Colab the retry parameter is set on the lead's word.
- M1 and K5: the earlier paragraphs on a check without a record count attempts in which the monitor ran; after a refusal's halt the resumed run makes the check at the step it resumes from, or when it reaches that step again.
- Q5: the later cap's dry run takes `--rows R2/42@0ms,R2/42@4ms,R2/42@6ms --conditions @gated,walk/nominal`; units 0, 4, 6 ms then attribution; nothing spent under one cap counts in the other.
- L12's note: "The ruling" -> "Ruling 7 of the fourth list".

### Handed back to the lead (each changes what a rule decides; listed in the amendment's last paragraph)
1. **M1, R0L's baseline.** "each must end *read* or *fell*" names no condition without a reference. To the letter, a body without a quick turn has no usable baseline. (a) Add "or, for a condition without a reference, is "not run""; (b) leave. I recommend (a); `monitor2.baseline_problem` already reads it so. Not written in.
2. **Z2 (c).** No degraded playback fails the tail term by its new limit. (a) Add a fifth control, a tail held still, that fails exactly the tail term; (b) leave it to `lookread`'s test. I recommend (a), when the band goes into the code. Not written in.
3. **A fourth refusal.** `monitor2.run` also leaves before any episode when `run.json` is missing; ruling 6 does not list it. (a) No attempt, halt; (b) a counted attempt. I recommend (a). Not written in.
4. **L12, the converse case.** When the failing set is the `-X` head's own, does the lead's decision take in the plain head's sets? Written in as the author's reading: yes. To confirm.

Seen and left: the trainer writes a halt's reason to `run.json` `halt`; `HALT.json`, which M1, M5 and K7 name, is the launcher's file in the design. Three older split "[lead, 15 Nov, ...]" labels remain (O3, M6, Q1 heads).

### Commits (not pushed, no amend, no history rewritten; all end with the two lines)
- `01c8341` merge of the working branch
- `9c36b17` rulings 1, 2, 6 (M1, M4, M5, K5)
- `ef983c6` ruling 3 and the confirmations (Q5, E4's mark)
- `7afea7e` ruling 4 (L12)
- `e847b08` ruling 5 (U3, U1)
- `2c56d1b` the frame and the third amendment
- `0710a90` pointers from the earlier amendments

`git status --porcelain`: empty in the worktree and in the main checkout. Edit specs and helper scripts: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trackb_r5/`.
