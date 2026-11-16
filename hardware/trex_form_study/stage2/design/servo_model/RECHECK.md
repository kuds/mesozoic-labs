<!-- Working paper, 16 Nov 2026: the second independent check, of the revised design note. Nothing here is a rule. -->

## Verdict: ready after the listed fixes

The fixer handled all 25 findings in substance, and all five of its "corrected or not taken" details hold. The engineering sections are sound. What is not ready is the owner-facing text: section 11 and section 9 still say more than the note's own tables support, and one new number is wrong. No rework of the design is needed.

Checked file: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_fix/SERVO_MODEL_DESIGN.md` (sha256 `e84f9389…`, unchanged by me). Line numbers below are that file's. "[calc]" is my arithmetic; "[P]" is printed by a probe I ran today.

## Findings, most serious first

**1. Section 11 overstates what a pass covers (line 670); section 9 does the same (line 597).**
- Sentence: "…keeps the simulated robot walking at 0.30 m/s, at the delays you approved, with motors drawn at random from the table and with motors at its extremes of friction, gear looseness and trunk weight."
- Evidence, all from the note's section 4:
  - The drawn panel holds the rotor inertia and the bus resistance at nominal and the trunk mass at 1.
  - It tests 20 parameter sets, the same 20 in its four records.
  - The extremes move one factor at a time, and only staying up is judged there.
  - Condition 2 runs at 4 and 10 ms on the grippy floor only. Those numbers are proposed, not approved (`BUY_GATE_DRAFT.md:9`, `41-44`; `DECISION_LOG.md:55`).
- Replace with: "It would tell you that two separately trained walking controllers keep the simulated robot walking at 0.30 m/s in two kinds of test. First, with every motor value at the table's expected value, on both hard floors, at each delay from 0 to 10 ms (condition 1, which you approved). Second, on the grippy floor at 4 and 10 ms (condition 2, proposed here): with 20 sets of motor values drawn at random from the table, and with friction, gear looseness and trunk weight each put at its extreme in turn. At an extreme the robot only has to stay up. The 20 random sets keep the rotor weight and the wiring resistance at their expected values. A pass does not show that the controller stays up everywhere between these points, or with two extremes at once."
- Line 597, add: "…from the model's table, except the rotor inertia and the bus resistance, which stay nominal; the same 20 sets serve both delays and both policies."

**2. The note never says plainly that a policy can pass every corner and the panel and still fall inside the ranges.**
- It adds the panel for that reason (line 360) but states no residual risk in sections 4, 7 or 11.
- Add to the list after line 390 and to section 7: "Passing all 28 records does not show that the policy stays up everywhere inside the ranges. The corners move one factor at a time, `D` tests 20 parameter sets with the rotor inertia, the bus resistance and the trunk mass at nominal, and the combined corners `W1` and `W2` are only reported."

**3. The corner pass rule departs from a sentence the owner approved, and choice 22 cites only the other side (line 578).**
- Sentence: "This is the draft's own wording for the corners ("stays up in 20 of 20")."
- Evidence: `BUY_GATE_DRAFT.md:34-36` puts all five conditions under the bar's rule (20 of 20 up and at least 17 meet every limit). Line 109 and P40 (`DECISION_LOG.md:55`) say "the gate's conditions use the bar's own pass rule". The note uses that same P40 sentence against route (b) in choice 18.
- Replace with: "The draft is not of one mind. Its condition 2 says "stays up in 20 of 20" (`BUY_GATE_DRAFT.md:41`); its preamble (`:34-36`), its line 109 and P40 (DL:55) say the gate's conditions use the bar's own pass rule, which also asks that 17 of 20 meet every limit. Stay-up only at the corners therefore changes a sentence the owner approved. If it is taken, the draft's lines 34-36 and 109 need the exception, and the owner's question must say it."
- Add to option 1 in section 11: "At these extremes the robot must stay up; the other limits are recorded there, not judged."

**4. New number wrong (line 167): "the brisk walk's 4.6 rad/s (0.31 m/s, the stored walk nearest gate 1's 0.30 m/s)".**
- Evidence: `refs/brisk.npz` has a stride displacement of 0.311 m over 0.887 s, which is 0.351 m/s [P]. `STAGE1_PLAN.md:85` gives 0.351 m/s for this body. 0.311 is metres per stride.
- Replace with: "(0.35 m/s [S], `STAGE1_PLAN.md:85`; gate 1's 0.30 m/s lies between this stored walk and the nominal one at 0.24 m/s)". The torques in that sentence are right.

**5. "Only the maker" contradicts the table (lines 524, 590, 668).**
- Sentences: "Five things only the maker can say"; "Five open numbers can only come from the maker"; "Five facts only the maker can give".
- Evidence: section 3's last column sends the inertia to "the maker's answer; proposed N5", the X6 line to N2, the gain limits to "sheet A step 1 (BENCH:519-525)", and the encoder side to N6. Only "which X6 ships" has no bench route. Two of the five are not numbers.
- Replace line 668 with: "Five facts the maker can give for free, before any part is bought; a bench could find four of them later. We propose to ask before gate 1 is judged." Lines 524 and 590: "five open facts the maker can answer now".

**6. The owner's three options are not yet a fair set (lines 673-675).**
- Option 1 says "settings with a stated basis". The note itself says the high-friction set is "above my bound on purpose" (line 303), and that 1° is four times the maker's limit, from one outside test of another model (line 268).
- Option 2 says "Every setting may stop it". Section 8's option (b) is the eight corners; the gain limits, the other X6, the combined corners and the slippery floor stay reported under every option. Whether (b) keeps the drawn panel is not said.
- No option gives its size, and option 3 does not state its cost.
- Replace with:
  1. "*Recommended.* Six extremes and the random panel can stop it: friction low and high, gear looseness none and 1°, trunk weight 10 % down and up, and 20 runs with randomly drawn motors. 560 runs must all stay up. The high-friction extreme is above the study's own estimate, and 1° is four times the maker's limit. Two harsher settings are run and shown but cannot stop a purchase yet: [as now]."
  2. "The same, plus the heavy rotor and the weak battery: 640 runs. Stricter; the purchase could then stop on a misread catalogue number or on a wiring resistance nobody has measured, and the weak-battery setting as defined is below the motors' own minimum voltage."
  3. "Only the expected values and the random panel can stop it: 80 runs in this condition. Lightest: a controller that falls at an extreme would still pass."
- At a true stay-up rate of 99.9 % the three pass 57 %, 53 % and 92 % of the time [calc]. With the panel kept in option 2 it is 720 runs and 49 % [calc].

**7. The tail's new nominal inertia has knock-ons the note does not carry (lines 224, 278, 286, 330).**
- By the plant's rule the tail's kv becomes 3.60 instead of 2.72 [calc; the fixer's `calc_fix.py` prints the same]. The note never says so.
- `kv_scale` tail still reads "0.62 N·m·s/rad, 0.23 of the model's". Against 3.60 that is 0.17, and the range's low end of 0.2 no longer reaches 0.62 [calc].
- The tail's range is "0.0002 to 0.028" with the nominal at its top. Training draws then lie at or below the judged value (median 0.0024 [calc]), and `J+` equals `N` for the tail.
- Line 330 holds `armature` at nominal in `D` "until the maker answers question 1". That question is about the leg drives only.
- Replace: line 224 add "For the tail the nominal kv becomes 3.60 (today 2.72) [E]." Line 286: "[E from M]: 0.62 N·m·s/rad; 0.23 of today's kv and 0.17 of the nominal kv under the new inertia", range 0.15 to 1.2. Line 278 add "The nominal is the top of the range; `J+` equals `N` for the tail." Line 330: "(leg drives: until the maker answers question 1; neck and tail: until N5)".

**8. `D` is not fully defined (line 330).**
- Sentence: "Drawn: every drawn parameter of section 3, except `armature` … and `bus_r`".
- Evidence: `latency_steps` is a drawn row of section 3 (line 292), but a `D` record is at 4 or 10 ms.
- Replace: "…except `latency_steps` (the delay is the record's), `armature` … and `bus_r`".
- Also unsaid: in `D`, `sense_steps` adds 0 to 4 ms of feedback age and `cmd_lag_ms` up to 4 ms of lag on top of the judged delay. Add one sentence to choice 19 and to section 9's "Two things to know".

**9. Section 11's picture of today's model is not true to section 1 (line 657).**
- "full strength at speed" and "joint sensors that read exactly". Today's model already drops torque above the rated speed (line 70), and its sensors are rounded, noisy in rate and 2 ms old (line 60).
- Replace: "…full strength up to their rated speed, a battery that never dips, and joint sensors with no zero error."

**10. The 86 % sentences lack the confidence level and the unit (lines 601, 670).**
- Replace line 670's clause: "yet 20 clean runs of one controller at one setting and one delay show, with 95 % confidence, only that it stays up more than 86 % of the time there." Same in line 601.
- The paragraph's heading is "a pass at gate 1", but 560 runs is condition 2 alone. Say "condition 2 alone needs 560 runs".

**11. Small inconsistencies.**
- Line 43 and choice 15 cite `BUY_GATE_DRAFT.md:59-60`; section 9 cites `59-61`. The sentence is 59-61.
- T14 is in no work package (S1 says "T4 to T12", line 494).
- Line 196's rule ("no command is served by a kv other than the one it was folded with") contradicts the row above it for `kv_scale`. Limit it to `armature` and `gain_cap`. Section 5.1 still says the env enters the layer "after the delay"; for that rule the gain change must come before the queue is filled (`r5env.py:513-514`).
- Lines 37, 442 and 634: `wt/wpc-ce` is at `cdc4762` now, not `4523ed6`. Its `r5env.py` is unchanged since `4523ed6` and its tree is clean.
- Line 145 and T6: "no shift of their reading" for the neck and tail should say "no play shift"; `enc_offset` still applies to all (line 288).
- Line 159: the 7.24 rad/s example holds with `ts_corner` at today's 8.69. Say so.
- Line 164: MAT:205-206 gives 194 W and 8.9 A. The 214 W is REC:298 plus REC:301.
- Line 5: "a second STS3215" is `BUDGET_75.md:318`, not 312-314.
- `D`'s seed `[s, 2]` is the training env's own form. It would equal a training episode's servo seed if a training stream's reset seed were 0 (`designlab/stage1/env.py:117-123`). A different tag, such as `[s, 3]`, removes the case.

## Choices for the lead (not made here)

1. **Corner pass rule (finding 3).** Stay-up only at corners, with the draft amended and the owner told (my recommendation), or the bar's whole rule at every judged setting.
2. **Tail kv (finding 7).** Recompute by the plant's rule and re-base `kv_scale` (recommended: one rule for every drive), or keep the box servos' kv at today's.
3. **Tail inertia range.** Leave the nominal at the top and say so (recommended: nothing sources a higher value), or widen it.
4. **Delay-like draws in `D` (finding 8).** Keep them and say so (recommended), or hold `sense_steps` at 0 in `D`.
5. **Does option (b) keep `D`?** Recommended: yes, so the options differ only in the two harsh corners.
6. **Replication.** `D`'s 20 sets are known after a first attempt. A passing `D` record could also be run on the bar's second block, seeds 5042 to 5061 (`bar.py:87-88`): 80 more runs [calc]. Recommended: consider it.

## Checked and found right

- **The fixer's five details.**
  - `r5env.py:475-476` is the right line pair.
  - The worktree edits are committed and unmerged.
  - Reader 3's line 46 has no "worn".
  - A zero nominal for a signed error is consistent.
  - "Anywhere in the ranges" was rightly dropped.
- **Every code, record and sheet line the revision adds or changes**, opened by me; the list is in my notes. All say what the note says, apart from the two citation slips in finding 11.
- **Arithmetic, redone.**
  - Pass table and panel bound: all ten cells; 0.861 (`panels.py:286-288`); 87 %, 57 %, 98 %, 94 %.
  - X6: 8.30, 4.16, 55 V, 8.01, 2.15, 15.8, 11.5, 20.4, 20.3.
  - X4 at 21 V: 7.61 against 7.24.
  - Bus: 0.09, 0.11, 0.08 to 0.10 Ω; 19.3 to 19.7 V; 735, 2,464, 888 W.
  - Gains: 19.47, 33 N·m, 11.3; 0.023 and 29 %.
  - Friction and peaks: 19.1, 38.3, 17.25, 35.4; 31.5; 0.667 × 30.
  - Play and inertia: 5.2, 9.9, 2.6 N·m; 8.7 mm; 0.039, 12-fold, 40-fold.
  - Table count: 38 = 1 + 4 + 6 + 27.
  - Records: 28 and 560; 56 and 1,120.
- **Probes rerun [P].** The tail and knee torques on the stored references match the note. Friction rows are 14, 18, 14.
- **Third-party files.** The MX-106, MX-64 and STS3215 fits and the cloned code's lines 72-73 at commit `0daab16` match.
- **Section 9 against the tables.** Sources, settings, values and counts agree, apart from findings 1, 8 and 10.
- **Section 11.** It makes no promise about hardware, and keeps makers' figures, the study's guesses and bench values apart, apart from finding 5.
- **Units** are consistent in the changed passages.
- **The fixer's side note.** `bar.py:2201-2205` does refuse a non-R0 row on an unregistered set. I checked no further.

## Not checked

- Any maker page, the maker's protocol document, the third-party paper.
- The two source hashes (`c0027992…`, `45719d03…`), the timing probe and the reviewer's toy probe.
- Passages unchanged since the first review, beyond reading them. PLANT values, most LAT lines, and the `ledger.py`, `policy_rows.py` and `estimator.py` lines.
- Whether tendon friction acts as assumed in a physics step; none was run.
- The wall time of the runs, the 60 cm body, the README beside the note.

No episode, fit or training was run and no web page was opened. Four small probes ran one at a time through `tools/lowprio.sh`. `git status` is clean.

Files are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_recheck/`:
- NOTES.md
- probe_ref_speed.py
