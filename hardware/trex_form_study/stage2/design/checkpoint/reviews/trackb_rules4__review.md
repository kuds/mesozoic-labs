<!-- Working paper, 16 Nov 2026: the independent review of that integration. Nothing here is a rule. -->

## Verdict

The draft's text for these rules can stand as the pre-registration **after the listed fixes**. Both sets are written in faithfully. Four points change what a rule decides and need the lead (findings 1 to 4); the rest is wording.

I ran no episode, fit or training. I ran one bar dry run as a spot check. Draft reviewed: `/home/user/mesozoic-wt/prereg-b/hardware/trex_form_study/STAGE2_TRACKB_PLAN.md` at `0ea2114`, against `c4a0ff8`. Line numbers are the file's at `0ea2114`. Nothing here is a hardware measurement.

## Findings, most serious first

**1. M1, lines 839-840, 846, 855-856 (ruling 4): a set without one of the six monitor references can never leave a record. Changes what a rule decides (one case).**
- "A record ... holds an outcome for all 18 episodes." M1's outcomes are four (read, fell, unread, error). Line 849 says a condition without a reference is "not run", which is none of them.
- Read to the letter, on a body whose quick turn or stop is "not available" (L12 allows it) every check counts as without a gain and as a failed G8b reading.
- Replacement: "... and holds, for each of the 18 episodes, an outcome or, for a condition without a reference, "not run"." The same words in the author's note (846) and the Check (856).

**2. M1 lines 840-841, K5 lines 1396-1397, M5 lines 947-949 (ruling 4): after the halt nothing says how the run goes on. Changes what a rule decides.**
- M5: a halted run "is resumed until one of the first four ends it". K5: a resume does not run that check a third time and halts. So every resume halts again.
- Options: (a) the lead has the fault repaired and resumes with a recorded parameter (as `TAKEOVER` is), which lets the check run once more per resume; a run that cannot be resumed is `unfinished: <reason>` (M5); (b) the check then counts as one without a record (18 errors). I recommend (a): M1 calls a dead session an interruption.

**3. E4, lines 397-398, against the merged code. Changes what a rule decides (a row on a turn-in-place plan).**
- Draft: "so does a plan that is given as a plan (a bar condition's): only a task's definition names the reference's yaw".
- Code: `r5lib.Library.from_plan(plan, track, heading)` (`r5lib.py` lines 839-860, commit `011f6e2`) takes `heading="reference"`; its docstring says a row that plays a turn-in-place plan passes it. Test: `test_a_plan_given_as_a_plan_can_name_the_references_yaw`. The env's own `_plan` passes none.
- The integrator copied wording entry 5a faithfully; the entry is behind the code.
- Replacement: "A walking turn keeps the integral. A plan given as a plan (a bar condition's) keeps it unless its caller names the reference's yaw (`from_plan`'s `heading`), as a row that plays a turn-in-place plan does." The mark then also names `011f6e2`. No rule of this round runs such a row yet (T2: "not run").

**4. Q5, lines 1235-1239 against its Check, lines 1292-1296 (the integrator's point 1). Changes what a rule decides.**
- Under the reading written in (a), the second dry run is committed after the row's first panel record. The Check asks for a commit that is an ancestor of that record, and for the order "10, 0, 4, 6 ms and then attribution". Both fail in that case.
- "Known only after the row's 10 ms records" should read "may be known only after". The case where S2-9 has already asked is not written.
- I recommend (b), not (a): the three other delays get a cap of their own, with its own dry run, run directory and `budget.json`. The Check then holds as written, per cap. If the lead takes (a), the Check needs both exceptions and a rule for attribution of the 10 ms records.

**5. L12, lines 1800-1802 and 1829-1830 (ruling 7). Wording alone, with two cases for the lead.**
- "Recorded with its date and reason" names no place, so the Check cannot be run. Proposed: "in a dated amendment of this text and in `set.json` `rules` of each of those sets".
- For the lead: whether "that body's other look sets" takes in its `-X` head's set; and a set already frozen when the other set's start fails (a decision after a closed-loop episode on it should carry the label "after a result").

**6. Q5, lines 1232 and 1244-1245. Wording alone.** No path for a cap's run directory or its committed dry run, and no task key. The Check needs them. Suggested, by the pattern of M3 and M6: `stage2/tb/runs/r2l-<set>-<body>/`, a `DRY_RUN.json` there (command, printed total, the set's freeze), task key `r2l`.

**7. W5 `range` (line 685) and W2's joint row (line 584): the draft is silent where the code is specific. Wording alone, under the lead's ruling that the reward stands as coded.**
- `range`: the code leaves out an end a joint rests on (`r5core.py` line 813, `r5lib.py` line 403). Add: "(an end a joint rests on by design, the jaw's closed end, is no limit: E5)".
- joint: on `-X` the reference carries the head command (`r5core.py` lines 737-742). Add: "(on `-X`, with the head command added: E5)". Read to the letter, a head that obeys a 0.45 rad gaze would lose most of the joint term (my arithmetic: exp(-(4.5)^2/16) = 0.28).

**8. Labels. Wording alone.**
- L11's marks at lines 1750-1751 and 1759-1760 lack "labelled as changed after the look work package had built look sets", though lines 2333-2335 say each wording carries it. Line 1759 names no ruling (6).
- The Check marks at lines 1053 and 1296 say "the fourth list" without a number (8; 1 and 2).
- L11's Check moved and reworded "L9 in none", unflagged; the meaning is the same.
- Line 108: `wt/wp-l` was at `c9b8819` from 15:34:43 UTC, before the commit that wrote "at `e2696b2`" (15:38:04). Its five `set.json` files still have `freezes` empty.
- Y10, line 1468: say that `budget_set_on` is a UTC date and time.
- Two labels are split over a line break (lines 22-23, 2150-2151).

**9. Code side, not the draft.** `r5core.py` line 32 still says W1 asks "every penalty 0". `REWARD_RULE`'s "charges nothing" is already with the lead (`ROUND_C_RULINGS_16NOV.md`, item 4).

## The integrator's other two points

- Point 2 (the candidate's caps are made again): I agree. My dry run on the frozen `F25F9cl` (`R2/42@10ms`, `@gated`) printed 320 episodes, 2.186 simulated hours, 0.344 core-hours, as the draft has it. It is a planning estimate at `COST.md`'s rate; no episode ran.
- Point 3 (K4, entry 8): keep. It says what `r5env.py` does.

## Checked and found right

- **History.** Only the draft differs from the working branch at `67cb6f5`. The worktree is clean. I changed no tracked file.
- **Nothing else changed.** 73 changed paragraphs, all in a named rule or the frame.
- **Fourth list.** Rulings 1 to 14 are in, with the reviewer's sentences word for word where the ruling asks. Facts of 13 and 14 agree with git and the file (`a0314fc` 03:56:06, `103bf35` 03:34:49, `984f81d` 04:01:56, heading 03:50 UTC).
- **Wording file.** The copy at `cd32e19` equals the working branch's. All 23 "New" texts are in the draft word for word once the marks are stripped. Labels A and B and the commits match. Old sentences are quoted.
- **Code, read at `67cb6f5`.** Weights; clearance (a) and (b); duty with its bounds and target speed; progress; heading with wrap and `yaw_home`; the turn in place's heading and the `heading` key; ranges ±0.45 and 0.3-0.45, event times, H1's seed; R6's overlay; T1's plans and holds; T3; K4's comparison, `RESUME_KEYS`, `LATER_FREEZE_KEYS`, `env_configs`, the crouch refusal. All as the draft says, but for findings 3 and 7.
- **Arithmetic** of Q5's table. 93 marks of 16 Nov, none split.
- **No conflict found** among M4/K4/K5 on task weights, T1/T3 on `stand`, E5/J4/W5/E2, Q5/Q6/U4, Y10/Y11, L8/L10/L11/L13.

## Not checked

- No test was run. The wording file's playback numbers are the env builders'; I did not reproduce them.
- `bar.py`'s ledger (H6 read in the plan only); the trainer and monitor (not merged); the look package's code and arrays, which are changing now.
- How `r5lib` counts T1's "2 (2.5) strides" on a K-stride look set.
- W1's playback sentence on a look set or on the 60 cm body: no Check tests it there.
- Join rule J1 against L10 for a start or a change (the integrator flagged it; no ruling names it).

Notes and scripts: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_trackb4/` (index: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_trackb4/NOTES.md`; word diff: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_trackb4/wd.porcelain`).
