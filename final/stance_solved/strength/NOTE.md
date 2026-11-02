# Strength of evidence across seeds: is the habit-free stance solid?

**Bottom line**
- True: seeds 60, 42 and 80 each trained a stance with the phase critic that had no tail rail or park and no saturating habit elsewhere, and that passed the final gate: 3 of 3. The habit-free part held over all 91 evaluations; the gate reads the final only [measured: SCORE lines 321, 325, 329, 357].
- Not true as framed: this is 3 seeds, and the reading of record says "NOT CONFIRMATORY". The pre-registered test failed. The control arm is the old stance6 runs again. No sham arm was run, and the mechanism was not measured in these runs.
- (a) The phase critic removes the habit: **likely, not established.** (b) A new seed would be habit-free and pass the gate: **more likely than not (roughly 3 in 4), not established.** The one-sided 95% lower bound on the success rate is 0.368.
- Caution from history: earlier recipes looked fine on some seeds, or early, then failed on other seeds or after 6M. The 12M PHASE runs do cover that window.

OUT = `/home/user/work/pc_study/final/stance_solved/strength` (my read-only scripts and outputs: `late_window`, `rates`, `c8_range`).

## 1. How many seeds, and when were they fixed

- **Three independent PHASE seeds.** PHASE TDs were 0.00790 / 0.01733 / 0.01224, against OFF 0.38823 / 0.63534 / 0.15434 [measured: RUNLOG l.968]. No evaluation-mean command on tail_1/2/3, neck or head reached 0.30 in any of the 91 PHASE evaluations [measured: OUT/late_window.out].
- **Fixed before the PHASE outcomes, but not blind to seed history.**
  - Pinned at 03:07 UTC on 2026-11-01 [measured: REPORT "Design, in brief"; CRIT line 145], before the launches at 03:26-03:28 [measured: RUNLOG l.885-887].
  - "chosen with their earlier outcomes known" [measured: CRIT lines 507-508]: stance6's D2 seeds, the same seeds "LTSB ran ... on r9" [measured: RUNLOG l.613-614].
- **A hard draw, not a random one.**
  - s42 railed in stance4, stance5 and stance6; s80 railed on r9 and had a neck habit on r10; s60 parked on r10 [measured: SUMMARY E1, E3, E5, E12].
  - Outcomes only partly carry over by seed (s60: no tail flag on r9 LTSB to 6,815,744 [RUNLOG l.303], parked on r10): suggestive, not a test [inferred].
- **PHASE determinism is unmeasured** [measured: NEXT_STEPS §2 C]. If it matches OFF, reruns of these seeds add nothing; more evidence needs new seeds [inferred].

## 2. History: solved on some seeds, then failed

| Earlier case | What happened | Source |
|---|---|---|
| stance5 (r9 LTSB) | 1 of 3 held, then "replicate on 2 more seeds": s70 held, s80 railed. 2 of 5, not adopted | RUNLOG l.349, 351, 492-493 |
| stance5r s80 | 10/10 clean at 3.93-4.19M; alarm at 4,456,448; "recovered by 10.5M" (38 of 40 clean at the final) | RUNLOG l.455, 493 |
| stance6 at 3.01M | "No ANY- or TAIL-flagged evaluation and no alarm in any run" | RUNLOG l.681 |
| stance6 at 7.6M and final | s42 rail, s80 neck alarms, s60 tail_3 at +0.76; at the final: a rail, a park, a neck habit | RUNLOG l.689, 711 |

- **Late onsets are real.**
  - OFF s60 (= stance6): tail_3 reached ≥0.30 at 6,422,528 and ≥0.60 at 6,946,816; first over-bar at 8,126,464 [measured: SCORE line 335].
  - OFF s80: tail_3 reached ≥0.30 at 9,830,400 and ≥0.60 at 11,796,480. OFF neck_pitch reached 0.30 at 11,010,048 (s42) and 11,141,120 (s60) [measured: OUT/late_window.out].
- **Coverage.**
  - The PHASE ledgers run to 11,927,552, and 12M is main's stance budget (`timesteps = 12000000`) [measured: MAIN stance.toml line 532]. So the runs cover every onset seen so far, and the budget a parent is trained to.
  - Not covered: after 12M, or between evaluations [measured: CRIT line 515]. In the OFF runs, drifts were still starting at 9.8-11.1M, so nothing shows the risk window had closed by 12M [inferred].
- **A small PHASE drift (report only).** After 8M the head_pitch command rises in all three PHASE runs (+0.0120, +0.0126, +0.0212 per 1M; largest |command| 0.162), and s80's tail_2 rises +0.0113 per 1M [measured: OUT/late_window.out; SCORE line 332]. Far below 0.60, but same direction 3 of 3: a watch item [inferred].

## 3. What OFF = stance6 means for the control

- The OFF runs are bitwise copies of stance6 [measured: RUNLOG l.970-971]. So there are **three control seeds in total**, and their values were known at design: TDs printed in CRIT (lines 229, 309), and TD's form chosen "with STANCE6's values in view" (lines 305-307) [measured].
- One gain: the 17:44 look could not have changed the OFF outcomes [inferred: RUNLOG l.970].
- The limit: training is deterministic but sensitive, so any change, even a sham, sends a run elsewhere. Each pair shows that a perturbed run lost its habit, not that the phase information removed it [inferred: REPORT "What it means"; CRIT line 513].

## 4. The numbers, checked

| Figure | Source | Shows | Does not show |
|---|---|---|---|
| NOT SEPARATED (p = 20/20) | SCORE line 362 | The pre-registered test failed | Anything about the TDs: with OFF at −infinity, p is 20/20 "whatever the six TDs" [AMENDMENT2 l.158-161] |
| LOWER (p = 1/20) | DEV line 393, flagged as a deviation at line 373 | Every PHASE TD is below every OFF TD | It was decided after the slip. 1/20 is the floor of a 3 + 3 test, exactly at alpha. Controls known at design; no sham |
| 0.296 | SCORE line 359 | At a rail rate of 1/3, "no rail in 3" happens about 30% of the time by luck | Parks and other habits; the rate comes from 3 runs |
| Fisher p 0.2 | SCORE line 363 | Any-alarm 0 of 3 against 2 of 3 is not significant (3/15 = 0.200) | 3 vs 3 reaches 0.05 only at 0 of 3 against 3 of 3 [OUT/rates.out C] |
| 0.037 | REPORT line 338 [inferred, post hoc]; not in SCORE or DEV | At a rail-or-park rate of 2/3, 0 of 3 would be rare | That 2/3 has a 95% interval of [0.094, 0.992], so the chance runs from 0.743 down to 5.93e-07 [OUT/rates.out B] |

Pooled rails (r9's 4 of 10 [RUNLOG l.613] plus r10's 1 of 3) give 0.233; rail-or-park at 6 of 13 gives 0.156 [OUT/rates.out B]. So "0 of 3 by luck" is about 0.04-0.3: suggestive, not decisive [inferred].

## 5. Mechanism: measured or inferred?

- **C8 (measured):** at all 18 PHASE checkpoints, perturbing the phase moves the critic by 0.296-0.757 and the actor by exactly 0.0, and the critic still reads observation 6 [measured: OUT/c8_range.out; complete_post.txt line 68 per PHASE run]. This shows the input was delivered as designed. It does not show whether the critic uses the tail angle as a clock. An OFF comparator for the obs-6 reading: not found.
- **BS2 (one start, s42 at 2,097,152):** blinding the critic to the tail angle reduced the climb, p 1/70; loop-gain secondary NOT SUPPORTED; spread caveat about 0.03 [measured: RUNLOG l.794, 796].
- **PS (the same single start):** REDUCED, p 1/70; critic tail slope (SEC-MECH) SUPPORTED; tail_2 guard rose, 2/70; spread caveat 0.036 [measured: RUNLOG l.857, 862]. With no sham, SEC-MECH is not separated from "a tail that simply climbed less" [measured: PS REPORT line 107].
- **Why-probe:** "(A) PHASE PROXY favoured strongly [measured inputs, inferred reading]" [RUNLOG l.804]. The cause of the onset is unresolved [RUNLOG l.771].
- **In PC:** the critic readouts (tail slope, V−G, loop gain) were not run [measured: RUNLOG l.967; CRIT lines 329-337].
- **Verdict: inferred.** The inputs were measured and two single-start screens agree, but the mechanism was not measured in these from-scratch runs.

## 6. How confident to be

**(a) The phase critic removes the habit (RB recipe, r10, these settings): likely, not established** [inferred].
- For: 3 of 3 clean, while every OFF seed had a habit [SCORE lines 334, 338, 342]. The gap is 12.6-49.1 times by TD [SUMMARY E25]. PS and BS2 agree.
- Against: no sham, three controls known in advance, the failed pinned test, and a luck probability of about 0.04-0.3.

**(b) A new seed is habit-free and passes the gate: more likely than not, roughly 3 in 4, not established** [inferred].
- From 3 of 3: the rule of succession gives 0.800 and a Jeffreys prior 0.875. The one-sided 95% lower bound is 0.368. Even at a true rate of 0.80, three new seeds would all pass only 0.512 of the time [OUT/rates.out A].
- stance5 went from 1 of 3 to 2 of 5 when seeds were added [RUNLOG l.351, 490-493].
- The gate is final-only and "not certification" [SCORE line 357]. K1-K4 were not run [SCORE line 361]. PHASE s60 fell 9 times in 910 evaluation episodes, against 0-1 in the other five runs [SCORE line 324].

**For the decision** [inferred]: strength alone does not block moving on if stance is carried as "working, provisional". Any further stance training should use new, pre-registered seeds; main's two-seed r10 certification bar [MAIN stages.toml line 34] would add exactly that.
