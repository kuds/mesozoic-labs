<!-- Working paper, 16 Nov 2026: the independent review of that integration. Nothing here is a rule. -->

## Review of the fifth list's writing-in (`STAGE2_TRACKB_PLAN.md`, `wt/prereg-b` at `0710a90`)

**Verdict: stands after the listed fixes.** All six rulings are written in faithfully, with the right labels and the replaced sentences quoted. No finding changes what a rule decides. Finding 1 is a stale fact in "What had been run" and should be fixed before the draft goes on. Findings 2 to 4 close gaps in three Checks. I changed no tracked file and committed nothing.

### Findings, most serious first

**1. "What had been run", lines 145-148: the look branch moved while the writing-in was being committed. Wording alone (a fact of the account).**
- The draft says branch `wt/wp-l` (at `de96fc0`) held no look set's `set.json`.
- `git log wt/wp-l` shows `33652cb`, committed 19:15:57 UTC. It adds `stage2/sets/V6/75-M-H-10/set.json`.
- That file has `freezes: []`, `blocked.by: "start"` and `not_available.start.reason: "no transition within the gates"`. Its commit message says "no episode of any kind" and "Nothing is frozen".
- The draft's commits: `9c36b17` 19:14:10, `ef983c6` 19:15:00, `7afea7e` 19:15:31, `e847b08` 19:16:12, `2c56d1b` 19:19:25 (this bullet), `0710a90` 19:20:10. So the sentence was no longer true when it was committed.
- "No look set frozen" still holds. The omitted set is a case of L12 (a `start` that fails), the rule ruling 4 is about.
- Replacement for the clause on `wt/wp-l`: "branch `wt/wp-l` held no look set's `set.json` at `de96fc0`; at 19:15:57 UTC, while the writing-in was being committed (19:14 to 19:20 UTC), the look work package committed `33652cb` there: set `V6` of `75-M-H-10`, built and screened by prescribed motion with no episode, `freezes` empty, recorded as blocked by its `start`. Ruling 4 was given at 16:28 UTC. That commit and the package's worktree were not read for the writing-in." The author should confirm the last sentence.

**2. M1's Check for a refusal, lines 962-964. Wording alone (checkability).**
- "The monitor's exit code for a refusal" names no code. `monitor2.main` returns 1 for every `MonitorError` today (lines 841-843).
- After a refusal for a checkpoint that is not intact, the resume starts from an earlier checkpoint. `train2._rollback` (lines 1063-1066) then moves the entry to that session's `discarded_after_resume`. It is no longer in `monitor[]`.
- The rule says "no attempt"; the Check says "that attempt".
- Replacement: "For a monitor that refused to run: `HALT.json` `reason`; the entry logged for it, closed with the exit code that `monitor2.py` names for a refusal (a constant of that module; no number is ruled), in `monitor[]` or, where the resume started from an earlier checkpoint, in that session's `discarded_after_resume`; and no entry closed with that code among the two closed attempts of a check counted without a record."

**3. Ruling 2's Checks and K5's reading, lines 959-961, 1568-1571, 1578. Wording alone (checkability).**
- K5 allows a resume without the parameter, which halts again. So "the resumed session's recorded parameter" does not hold for every resumed session.
- K5 says what follows when the allowed attempt is never closed. It does not say what follows when it closes without a record. The code then counts it and halts in the same session (`train2.py` 1257-1263: `after_halt` still true, `retry_used` true).
- Replacement in M1's Check: "every attempt at a check made after two attempts at it were never closed is marked `after_halt`, lies in a session whose `retry_check` is true, and is that session's only such attempt".
- Addition to K5's reading: "If that attempt is closed without a record, it counts toward M1's two and the session halts again: the check's next run needs the next resume with the parameter."

**4. Q5, lines 1377, 1466-1469, 1406, 2187. Wording alone (checkability).**
- Line 1377 reads "before the first dry run is made"; ruling 3 reads "after a row's cap is committed". The time between made and committed has no rule. Replace "is made" with "is committed".
- The Check's last item turns on when "S2-9 asked". No record field holds that. Replacement: "for a row with a record at 0, 4 or 6 ms whose first cap's `DRY_RUN.json` names 10 ms rows alone: a second cap with the units of 0, 4 and 6 ms, each cap with its own `DRY_RUN.json` and `budget.json`". This reads the case from files. It agrees with the rule as long as the builder follows line 1377.
- `DRY_RUN.json` has no key names (line 1406), and `bar dry-run` writes no file. A program cannot read it until the keys are named. Name three keys here or in the builder's brief.
- Change list item 15 (line 2187) still says "before the row's first panel record"; the brief was changed to "before the first record it covers".

**5. Ruling 1's Check, lines 953-955. Wording alone.**
- The rule allows "not run" only "for a condition without a reference". The Check names no field that says which conditions those are. `monitor2.is_record` accepts `not_run` for any episode.
- Add where it is read, for example "(the conditions the set records as without a reference, H3)". The author should name the field.

**6. Small points of wording. Wording alone.**
- Line 133: "no Track B run, fit or training run had been made" is unqualified. Lines 143-144 call the tiny runs "training runs of the tooling", and the Status (line 47) says "on a Stage 2 set". Add "on a Stage 2 set" at line 133.
- Line 1131: "fitted over the window". The code fits between the window's first and last left lift-off (`lookread.RULES["measures"]`). Say "from the same fit as the tail phase".
- E4 (line 455): "the lead confirmed these sentences" (three). The list names "rule E4's sentence on a plan given as a plan".
- Tags entry (lines 375-377): the mark is said to stand "on what that list confirms"; K4 carries none.

### For the lead (each changes what a rule decides)

1. **R0L's baseline with a "not run" condition.** Options: (a) add "or, for a condition without a reference, is 'not run'"; (b) leave. I recommend (a). M1's closing paragraph ("for both rows") and M3b ("as it does in S") already read so.
2. **Z2 (c), a control for the new tail limit.** Options: (a) a fifth control; (b) leave it to the module's test. I recommend (a), but not "a tail held still". My probe on synthetic numbers through `lookread._fit_phase` and `compare` (no model, no episode):
   - a tail held at 0.0 already fails today by phase (difference +0.27 cycle);
   - a tail held at 0.03 got a round-off phase (difference -0.14);
   - a tail scaled by 0.1 about its mean keeps its phase (difference 0.0000, amplitude 0.0125 rad) and passes today.
   
   The scaled tail would fail by the new limit alone, so that is the control to add.
3. **Exits of the monitor before any episode that ruling 6 does not list.** The integrator named a missing `run.json`. Two more:
   - a checkpoint record that cannot be read (`s2io.read_json` raises, not `MonitorError`);
   - a checkpoint whose fingerprint is not reproduced at the row's load (`s2load.CheckpointError`, which `monitor2.main` does not catch). I saw this on the `S1` stand-in path only; `lookbar` and `s2rows` are not on the branch.
   
   Options: (a) each is a refusal, no attempt, halt; (b) each is a counted attempt. I recommend (a).
4. **L12, the converse case** (the failing set is the `-X` head's own). Written in as the author's reading: yes. I recommend confirming it. The case is live: see finding 1.

### Checked and found right

- **The diff.** Only this file differs from the working branch (407 insertions, 154 deletions). A word-level diff shows no removed word outside the described changes. The merge `01c8341` has the working branch's tree exactly.
- **Rulings 1 to 6 and the confirmations**, against `TRACKB_RULINGS5_16NOV.md`: wording, marks, quoted old sentences, head tags, brief, change list items 15, 24 and 30, and the pointers in the two earlier amendments.
- **Contradictions.** I searched "not run", "18 outcomes", "halt", "refus", "attempt", "exit code", "tail", "four bands", "look bands", "other look sets", "S2-9", "dry run" and "budget.json". I found none beyond the findings above. M1's 16 Nov sentence on closed attempts is read with ruling 6 by the author's note and by ruling 6's own words.
- **The tail band against `lookread.py`.**
  - `compare` reads the phase alone, records `tail_amp_ratio`, and its rule text says no band reads it.
  - `reading` holds the robot's amplitude as `tail.amp_rad`.
  - `compare`'s `reference` holds the reference's amplitude and its side of 0.02 rad.
  - The limit can be added in `compare` from values it already has.
  - "Under 0.02 rad" matches the code's strict comparison.
  - The ladder's V6 tail amplitude is 0.125 rad on the rungs I read, far from the threshold.
- **The refusal against `monitor2.py` and `train2.py`.**
  - The three messages are quoted as the code has them, and all three are raised before any episode.
  - All exit 1 today, and the trainer's loop counts them.
  - `--retry-check`, `sessions[].retry_check`, `after_halt` and one such attempt per session are as the draft says.
  - The trainer writes `halt` to `run.json` and `progress.json`; `HALT.json` is the launcher's in `ROUND_C_DESIGN.md`.
  - `s2load.py` defines "intact" as the draft's note says.
- **"What had been run".** The S1 episodes, the tiny runs, the kills on the stub and the "40 of 200" probe are in the round C reports as quoted. The working branch at `ffecc2c` holds no look set, and `stage2/tb/` holds one ignore file.
- **Commits.** Seven, each ending with the two lines and naming no model elsewhere. The reflog shows no amend. No remote branch holds them. `git status --porcelain` is empty in the worktree and the main checkout.
- **The hash line** printed the registered pair: `45719d03...7528d c0027992...1be1c1`.

### Not checked

- No test, episode, fit, training run or dry run. My one probe is pure functions on synthetic numbers.
- `lookbar` and `s2rows` (not on the branch), so the monitor's real episode path and the companion's `look.own` and `look.reference` keys.
- `bar.py`'s `budget.json` and unit rule against Q5's wording.
- The notebook's name for the retry parameter.
- Whether earlier builds of the look sets also failed their `start`.
- Sections of the draft that the diff does not touch, beyond the term searches.
- The integrator's line numbers, beyond spot checks (all matched).

Files are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_trackb5/:
- plan.diff
- tail_probe.py
