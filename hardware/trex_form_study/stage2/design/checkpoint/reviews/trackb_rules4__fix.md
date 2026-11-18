<!-- Working paper, 16 Nov 2026: the fixer's report on applying that review. Nothing here is a rule. -->

## Result

The review is applied to `/home/user/mesozoic-wt/prereg-b/hardware/trex_form_study/STAGE2_TRACKB_PLAN.md`. Six findings are fixed in whole or in part, and four points go to the lead unchanged. I ran no episode, fit, training, dry run or test; only git, grep and small text scripts. Line numbers are the file's at `e4d9e84`. Nothing here is a hardware measurement.

The working branch had moved from `67cb6f5` to `355e91a` (the lead's 15:40 UTC rulings on the env's fix round, and working papers). I merged it first; only the draft differs from it.

## Per finding

1. **M1, a record on a set with a "not run" condition (864-865, 871, 874, 880-881): for the lead.** Confirmed: the sentence is the lead's own (ruling 4, the earlier reviewer's text word for word), and M1's four outcomes do not include "not run". Nothing changed.
2. **After the halt (M1 865-866, K5 1425-1426, M5 972-973): for the lead.** Confirmed: M5 says a halted run is resumed, and K5 says the resume halts again. Nothing changed.
3. **E4 against the merged code (396-421): fixed.** I did not leave this to the lead, because the lead had already ruled it: `stage2/design/checkpoint/STATUS.md`, 16 Nov 03:02 UTC, ruling 4: "`from_plan` gains a heading argument before any row plays a turn in place".
   - Code read: `r5lib.py` 839-860 takes `heading`; no caller outside the tests passes it; `r5env._plan` passes none.
   - New text (405-408): a plan given as a plan keeps the integral unless its caller names the reference's yaw. No rule of this text runs such a row (T2).
   - The mark quotes entry 5a's old last sentence and names `011f6e2`. The draft now holds 22 of the wording file's 23 "New" texts word for word (my check script); 5a's last sentence is the one departure.
4. **Q5, S2-9 asking late (1261-1266): the choice is for the lead; two points of wording fixed.**
   - "Is known only after" now reads "may be known only after" (1262).
   - The case where S2-9 has asked before the first dry run is written (1263-1264).
   - The late case still stands as reading (a), and the Check still does not allow for its second dry run.
5. **L12 (1832-1836, Check 1864-1866): place fixed; one case for the lead.**
   - Place: an author's note and the Check name "an amendment of this text", by the text's own rule for a later answer of the lead. I did not add `set.json` `rules`, which would bind the look package's files.
   - A set already frozen: not a defect once the place is named. The same rule says an answer given after a result is labelled so.
   - Whether the `-X` head's sets are meant: for the lead.
6. **Q5, no path or task key (1272-1274, Check 1321): fixed without naming a path.** The report names each cap's run directory and the file and commit of its dry run, and the Check reads them there.
   - I did not take the suggested `stage2/tb/runs/r2l-<set>-<body>/`: the design (`ROUND_C_DESIGN.md`, D22) puts the bar's records under `stage2/tb/bar/<run>/`.
   - The earlier review left "[directory]" blank and the lead ruled without filling it.
   - `bar.py`'s `--task` takes 3 or 4 only, so the key's name is the builder's.
7. **W5 `range` (707-711) and W2's joint row (602, note at 623-626): fixed.** One parenthesis each, as the review proposed. Code read: `r5core.py` 737-742 adds the head overlay to the joint reference; `metrics.joint_margins` leaves out the jaw's closed end. `REWARD_RULE` says both.
8. **Labels: fixed.**
   - L11's two marks carry the look-set label, and the second names ruling 6 (1779-1781, 1789-1791).
   - Check marks name their rulings: U4 (1078, ruling 8), Q5 (1326, rulings 1 and 2).
   - L11's Check mark says the L9 item was moved and quotes the old wording (1802-1804).
   - Line 108: `wt/wp-l` was at `e2696b2`, then at the merge commit `c9b8819` from 15:34 UTC. Git shows the look-set files equal in both.
   - Y10's Check: "a UTC date and time" (1498). The study's records write UTC (`s2io.now_utc`).
   - Split labels: five made whole (Status 22-23, the Tags entry, both ruling sections' openings, the second list's closing paragraph); no word changed.
   - Left: three older splits of "[lead, 15 Nov, ...]" in the heads of O3, M6 and Q1 (788, 988, 1168). They predate this round.
9. **`r5core.py` line 32 ("every penalty 0"): not a defect of the draft.** Confirmed in the code. It belongs to the env package's next change of that file. `r5lib.COMMAND_RULE` likewise does not mention `from_plan`'s `heading`.

**The integrator's points 2 and 3** (the candidate's caps made again; K4's entry 8 kept): no defect. Both stand as written, and the reviewer and I agree with them.

**The frame.**
- "What had been run" (120-126) says what the review and these fixes ran.
- The Tags entry (341-343) names the three places where wording goes beyond the wording file.
- The second amendment closes with what was changed after the review and what is with the lead (2477-2500).

## For the lead

1. **M1: what a record holds for a condition without a reference.**
   - (a) "... and holds, for each of the 18 episodes, an outcome or, for a condition without a reference, "not run"", in the rule, the author's note and the Check.
   - (b) Leave as is.
   - I recommend (a). Read to the letter, a body without a quick turn would fail every check.
2. **How a run goes on after the halt for two attempts never closed.**
   - (a) The lead has the fault repaired and resumes with a recorded parameter, which lets the check run once more per resume; a run that cannot be resumed is `unfinished: <reason>`.
   - (b) The check then counts as one without a record (18 errors).
   - I recommend (a).
3. **Q5, S2-9 asking after a row's cap is committed.**
   - (a) The dry run is made again and the cap replaced (as written); the Check then needs two exceptions and a rule for the attribution of the 10 ms records.
   - (b) The three other delays get a cap of their own, with its own dry run, run directory and `budget.json`.
   - I recommend (b): the Check holds per cap as written.
4. **L12: does "that body's other look sets" take in the `-X` head's sets?** (a) Yes; (b) no. I recommend (a): L12 itself says a cut is taken for "every look set of that body and of its `-X` head".
5. **To confirm, no change needed if yes.**
   - E4 as I fixed it under the 03:02 UTC ruling.
   - The candidate's two caps are made again.
   - K4 keeps entry 8.
   - Q5 names no path. If names are wanted, I recommend `stage2/tb/bar/<run>/` with a `DRY_RUN.json` at its top.

## Commits (branch `wt/prereg-b`, not pushed, no history rewritten, no amend)

- `77fda84` merge of the working branch (at `355e91a`)
- `60cd741` E4, W2, W5 (findings 3 and 7)
- `7b50156` Q5, L12, Y10 (findings 4, 5, 6)
- `48b3eeb` the labels (finding 8)
- `e4d9e84` the frame

Branch head: `e4d9e84`. All 24 commits on the branch that are not in the working branch end with the two required lines; none lacks them, and none names a model elsewhere. `git status --porcelain` in the worktree is empty. I wrote nothing in the main checkout.

Scripts are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trackb_fix4/` (`ed.py`, `rewrap.py`, `splits.py`, `wcheck.py`, and the edit specs `specA.py` to `specD3.py`).
