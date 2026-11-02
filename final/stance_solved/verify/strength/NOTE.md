# Verification of the "strength of evidence across seeds" note

**Bottom line**
- Every number I could check is right: TDs, p values, plug-in chances, C8 ranges, onset steps and slopes. Ratings (a) "likely, not established" and (b) "more likely than not, not established" are fair.
- Three corrections are needed:
  - The s60 "does not carry over" argument rests on a flag that cannot see a park.
  - s60's own rail on r9 is missing.
  - The decision paragraph leaves out main's rule: walking needs a certified two-seed stance first, and the recipe is not on main.
- Strengths played down:
  - PHASE cleared the usual onset window (1.2-4.5M) with the tail command at 0.031 or below.
  - The deviation was fixed before any TD was seen.
  - Under line 4's own definition, OFF went 0 of 3.
- Limits left out: the pairing by seed (a paired p cannot go below 1/8), and PHASE s60's falls clustering just after step 200.
- "Roughly 3 in 4" is more precise than 3 runs allow.

OUT = `/home/user/work/pc_study/final/stance_solved/verify/strength`. It holds read-only scripts `c8_check.py` and `ledger_check.py` with their `.out` files. No training, replay, Drive, network or git write was done.

## 1. Confirmed (each checked against its source)

- **PHASE 3 of 3:** CLEAR, GATE PASS, habit FREE [measured: SCORE 321, 325, 329, 357].
  - Per evaluation: CMD_L ≥ 0.60 in 0 evaluations, 0 TAIL flags, ANY ALARMs 0 [measured: SCORE 322, 326, 330]. The note cites 321/325/329, the header lines, instead.
  - My own check over every commanded actuator, legs included: no PHASE evaluation mean reaches 0.17, and saturated_actuator_fraction is 0.000 in all 910 rows of each run [measured: OUT/ledger_check.out].
- **TDs:** 0.00790 / 0.01733 / 0.01224 against 0.38823 / 0.63534 / 0.15434 [measured: RUNLOG 968]. Ratios 12.6 / 36.7 / 49.1 [inferred: my arithmetic].
- **Seeds and timing:**
  - Pinned at 03:07, launched 03:26-03:28 [measured: RUNLOG 880, 885-887].
  - Seeds "chosen with their earlier outcomes known" [measured: CRIT 507-508].
- **OFF = STANCE6, bit for bit** [measured: RUNLOG 970-971].
  - OFF's TDs are printed in CRIT [measured: CRIT 229, 309].
  - TD's form was chosen with STANCE6 in view [measured: CRIT 305-307].
- **The statistics:**
  - NOT SEPARATED (p = 20/20) [measured: SCORE 362], "whatever the six TDs" [measured: AMENDMENT2 158-161].
  - LOWER (p = 1/20) [measured: DEV 393], flagged at DEV 373. Alpha 0.05 [measured: CRIT 294].
  - 0.296 [measured: SCORE 359]; Fisher 0.2 [measured: SCORE 363].
  - 0.037 [measured: REPORT 337-338], post hoc, and absent from SCORE and DEV [measured: grep].
- **The note's arithmetic, recomputed:** 0.368; [0.292, 1]; 0.800; 0.875; 0.512; 0.233; 0.156; [0.094, 0.992] → 0.743 to 5.93e-07 [inferred: my arithmetic].
- **Late onsets:**
  - OFF s60: 6,422,528 / 6,946,816 / 8,126,464 [measured: SCORE 335].
  - OFF s80 tail_3: 9,830,400 / 11,796,480 [measured: SCORE 343].
  - OFF neck_pitch: 11,010,048 / 11,141,120 [measured: OUT/ledger_check.out].
- **PHASE drift after 8M:** head_pitch +0.0120 / +0.0126 / +0.0212 per 1M (largest 0.162); s80 tail_2 +0.0113 per 1M [measured: the note's late_window.out; SCORE 332; ledger_check agrees on 0.162].
- **Budget:** `timesteps = 12000000` [measured: MAIN stance.toml 532]. The ledgers end at 11,927,552.
- **C8:** 18 of 18 IN FORCE; the phase moves the critic by 0.296-0.757 and the actor by 0.0 [measured: OUT/c8_check.out]. OFF's C8 stops at "a stock critic" [measured: f2/measure/stance7_r10LTSBoff_s60/c8_final.out].
- **The screens:**
  - BS2: p 1/70, loop gain NOT SUPPORTED, spread chance about 0.03 [measured: RUNLOG 794, 796].
  - PS: p 1/70, SEC-MECH SUPPORTED, tail_2 guard p 2/70, spread chance 0.036 [measured: RUNLOG 857, 862]. No sham arm [measured: PS REPORT 107].
- **Mechanism record:** the why-probe reading [measured: RUNLOG 804]; the onset is unresolved [measured: RUNLOG 771]; the critic readouts were not run [measured: RUNLOG 967].
- **Other:** PHASE s60 fell 9 times in 910 episodes [measured: SCORE 324]. K1-K4 were not run [measured: SCORE 361].

## 2. Corrections

1. **The s60 "outcome only partly carries over" example fails** [inferred].
   - The r9 evidence is "no saturation flag ... up to 6,815,744" [measured: RUNLOG 303]. A park raises no such flag: OFF s60 has TAIL-flagged 0 and ANY ALARMs 0 [measured: SCORE 335].
   - s60's r10 tail reached 0.60 only at 6,946,816, after that window [measured: SCORE 335].
   - r9 parks were never classified [measured: the note's rates.out B].
   - CRIT points the other way: "outcomes that persist by seed (r9's s42 and r10's s42 both first over the bar at 3,670,016)" [measured: CRIT 505].
2. **The hard draw is harder than stated.** s60 railed on r9 too:
   - stance3 Run 2 = `stance3_armLT_s60` [measured: MAIN BRP D-D37].
   - It "railed in training (alarm 2,359,296; 44 of 91 evaluations flagged ... parked at 0.841)" [measured: RUNLOG 613].
   - So all three seeds have tail-habit history on r9 [inferred].
3. **The stance6 3.01M row tells half the story.** The same line records "s42 tail_3_pitch command mean up from ~0.00 at 1.31M to +0.394 at 3.01M" [measured: RUNLOG 681]. The flags were clean; the drive was not.
4. **The bottom line's "the pre-registered test failed"** should read "failed on procedure". On the data, the separation is complete [measured: DEV 393; AMENDMENT2 158-161].
5. **"Decided after the slip"** leaves out that the deviation was decided "before any F2 file and before any tail-command, TD or ledger figure of any run was seen" [measured: AMENDMENT2 162-165].
6. **"Roughly 3 in 4" is false precision.**
   - "Three runs per arm measure no rate" [measured: CRIT 492].
   - The uniform and Jeffreys priors assume exchangeable seeds, but these were a known-hard draw [inferred].
   - Keep "more likely than not, not established".
7. **The stance5 analogy is weak.**
   - "1 of 3 → 2 of 5" is a rise, judged on K1-K4; s60 failed on K3 joint-noise falls [measured: RUNLOG 349, 491].
   - No earlier recipe reached 3 of 3 habit-free [inferred].
8. **"For the decision" leaves out main's rule.**
   - GQ-17: "the trex now needs a stance retrain (two seeds at today's `certification_seeds`) before any trex walk" [measured: MAIN GQP 700].
   - Main's stance.toml has no LTSB or critic-phase keys [measured: `git grep` origin/main]. BR's stance.toml is headed "SCREEN, NEVER MERGED" [measured: BR stance.toml line 1].
   - "Screen runs certify nothing" [measured: MAIN BRP D-D37].
   - So the two-seed bar needs a recipe decision and a code landing first [measured: NEXT_STEPS Option E]. It then adds new-seed evidence, not a control [inferred].
9. **Length:** the note runs 1,526 words, against about 1,400.

## 3. Missing

- **A strength, onset timing.** Earlier habits mostly began at 1.2-4.5M:
  - stance4 s42 at 1.8M [SUMMARY E2];
  - stance3 Run 2 at 2.36M [RUNLOG 613];
  - stance5 s42 at 3.54M [SUMMARY E3];
  - stance5r s80 at 4.33M [RUNLOG 455];
  - stance6 s42 at 2.88M [SCORE 339];
  - OFF s80 neck at 3.01M [ledger_check].

  PHASE's largest tail command was 0.021-0.031 [measured: SCORE 322, 326, 330].
- **A strength, post hoc: line 4's own count.** By line 4's definition, OFF succeeded 0 of 3 against PHASE's 3 of 3: one-sided Fisher p = 1/C(6,3) = 0.05 [inferred: arithmetic]. It carries the same caveats (known controls, no sham).
- **A limit, pairing.** The design is paired by seed: the same starting networks [measured: CRIT 112-125, A5]. A paired test cannot go below p = 1/8 with 3 pairs [measured: CRIT 316].
- **A limit, OFF s80's late head habit.** DRIVE ONLY: CMD ≥ 0.60 at 10,485,760 and at the final, first at 9,437,184 [measured: SCORE 346; ledger_check]. PHASE's head drift has the opposite sign (+0.16 against −0.80) [measured: ledger_check].
- **A limit, PHASE s60's falls.** 7 of the 9 end at steps 211-378, just after the step-200 switch-on [measured: W/short_eps.out]. They may be a side effect, and they bear on (b) [inferred].
- **A weak check.** C8's observation-6 check counts nonzero weights (512); the report-only effect on the critic's value is 0.0095-0.026 [measured: OUT/c8_check.out].
- **A screen left out.** BS (from 1.05M) was NOT INFORMATIVE [measured: RUNLOG 779]. The mechanism evidence rests on one start state [measured: PS REPORT 103].
- **The gate label.** The bottom line's "passed the final gate" should say "a screen measurement, not certification", with M4 owed [measured: SCORE 357, 361].
