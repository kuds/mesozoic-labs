# Tail_3 bound screen: verified result (2026-10-24)

The score stands: STATUS FINAL, OUTCOME C. BOUND90 is NOT ADOPTED (RAIL) and BOUND60 is NOT ADOPTED (PARKED). The criteria's name for this is "the lower bound cleared the rail but not the posture". None of the three checks found a blocking error. REPORT.md was not written: the Write call was refused (subagent report-file rule). The report text follows.

SCORE AS READ [measured]: score.json shows status final, provisional_reasons [], incomplete [], lost {}. State readings: s42 BOUND90 RAIL, s42 BOUND60 PARKED, s80 BOUND90 RAIL, s80 BOUND60 PARKED. OTHER is 0 vs 0 at every state.

CHECKS [measured]: all three checks agree with the score.
- Re-score: rebuilt the rules from criteria.txt alone, working from the per-step traces of all 720 R1/R2 episodes. It matches all 18 classes, all 4 state readings with their lines_applying, the onset note, both verdicts and outcome C.
- Provenance: all hashes and pins hold. This covers the start and final checkpoints, all 162 panel rows, criteria.txt (9930ee72), APPROVED, screen_common, the 7 tools and coef.json. The clone is at 4c0c8da7 and clean; gate_pass.json is in place; the wave order is correct. The durable branch 1106cd3f has scores byte-equal to the local ones. The judge clones are clean.
- Re-measure: 4 final panels came out identical field for field, with flags.csv byte-identical.
- I also recomputed every class, R1/R2 mean and flag count from panels.csv myself. All equal score.txt.

KEY NUMBERS [measured]. Classes and R1/R2 mean CMD are listed r101/r202/r303, then the arm mean. Tail-up counts are out of 40 episodes per replicate.

s42:
- CONTROL: RAIL / CURED / RAIL. CMD 0.986 / 0.265 / 0.902, arm mean 0.718. Tail-up 36 / 0 / 38.
- BOUND90: RAIL x3. CMD 0.886 / 0.924 / 0.951, arm mean 0.920. Tail-up 1 / 14 / 34.
- BOUND60: PARKED x3. CMD 0.6070 / 0.6002 / 0.6128, arm mean 0.607. Tail-up 0 / 0 / 0.

s80:
- CONTROL: RAIL x3. CMD 0.937 / 0.911 / 0.844, arm mean 0.897. Tail-up 23 / 24 / 13.
- BOUND90: RAIL x3. CMD 0.900 / 0.921 / 0.971, arm mean 0.931. Tail-up 4 / 18 / 32.
- BOUND60: PARKED x3. CMD 0.6318 / 0.6280 / 0.6152, arm mean 0.625. Tail-up 0 / 0 / 0.

Other results:
- OTHER flags are 0 everywhere. There are no falls at R1/R2.
- NOT SEPARATED does not apply: only 1 s42 CONTROL replicate is CURED, and the line needs 2.
- Onset note: INFORMATIVE. All 3 s80 CONTROL replicates had tail-up flags, and each has 9 of 9 checkpoints measured.
- Dose ordering (report-only) is False at both states, because BOUND90's arm mean is above CONTROL's.

MANIPULATION CHECK (report-only telemetry, not a criterion) [measured]:
- CONTROL's applied loss (coef x L_B) is exactly 0 in all 128 updates of all 6 runs.
- Both bound arms log coef 2.4 and a non-zero loss in 128 of 128 updates.
- In every same-seed pair, the share of training rows with tail_3's mean above b is lower under the bound. At b 0.90: s42 0.611 / 0.444 / 0.503 vs 0.249 / 0.330 / 0.237; s80 0.391 / 0.316 / 0.332 vs 0.194 / 0.242 / 0.243. At b 0.60: s42 0.931 / 0.673 / 0.861 vs 0.589 / 0.567 / 0.549; s80 0.839 / 0.796 / 0.754 vs 0.559 / 0.575 / 0.523.
- CONTROL's logged tail3_bound_penalty is not 0. That is by design: criteria lines 95-96 make it the unweighted L_B, computed for logging only.

MINOR FINDINGS (none blocking):
1. All six BOUND60 means are within 0.032 of 0.60 [measured]. s42_b60_r202 clears it by 0.00017 and r101 by 0.0070. If both fell below, s42 BOUND60 would read PASS [inferred]. The verdict would still be PARKED through s80, whose three replicates clear the line by +0.015 to +0.032 (paired z 2.3 to 7.2) [measured]. KNOWN LIMITS line 421 already flags this.
2. s42_b90_r101's RAIL rests on one R2 episode with a tail-up share of 0.111 against the 0.10 cutoff [measured]. The state reading does not depend on it, because r202 has 14 flags and r303 has 34. That replicate also had flags at 7 of its 9 checkpoints (report-only) [measured].
3. Bookkeeping only [measured]:
   - APPROVED.runlog_line 551 is the maintainer's re-pin decision; the hash 9930ee72 first appears on RUNLOG line 552.
   - PINS.txt is a stale pre-approval snapshot, but all its hashes match the current files.
   - score.txt prints an onset value on every row; only the s80 CONTROL values are used, and they are correct.

WHAT IT MEANS:
- BOUND90 at coef 2.4 cut the training rows above 0.90, but it did not keep the rail away in deterministic play [measured].
- BOUND60 cleared the rail: there are no tail-up flags, and its highest raw tail_3 value is 0.858 [measured]. Its command still stayed at 0.600 to 0.632, so it reads PARKED, which ruling (3) counts as NOT ADOPTED [measured].
- The screen does not rank the two bounds [inferred]. BOUND60's closeness to the line does not make PARKED a near-PASS [inferred].
- Dose ordering and the manipulation numbers stay report-only.
- A NOT ADOPTED at coef 2.4 does not reject a stronger dose (criteria line 162).

ROUTING (criteria, outcome C):
- BOUND90 (RAIL) goes to the maintainer's choice of next candidate: the deterministic-state variant (BRIEF 2A), a stronger dose (DEC1 D3(c)), or the posture decision (B / C).
- BOUND60 (PARKED) goes to the posture decision (ruling (3)).
- Nothing trains on this outcome without the maintainer's approval.
