# Critic-blind screen from s42's 2.10M checkpoint (BS2): result

2026-10-30. A scratch CPU training screen, report only; nothing merged. Every figure below was reproduced by two
independent checks (statistic and process; 0 blocking findings). [inferred] marks reasoning, not measurement.

## The reading: line 2, REDUCED (status final)

> REDUCED: BLINDING THE CRITIC TO THE TAIL_3 ANGLE (observation 6) REDUCED THE ON-POLICY TAIL_3 COMMAND OVER THE
> LAST 8 UPDATES BEFORE E, FROM s42's 2,097,152 STATE (exact one-sided permutation test, p <= 0.05)

- **Design (pinned before launch):** 8 continuations of r10 stance6 s42 from 2,097,152 to 3,538,944 (1.44M steps):
  - 4 control;
  - 4 with the critic blind to the tail_3 angle. The angle is zeroed at the critic's input only; the actor still sees
    it.
- **Primary:** the policy's tail_3 command on its own states over the last 8 updates.
- **Gate:** the control median had to rise at least 0.20, about a third of s42's own rise of +0.58 over this window.
- **Test:** an exact permutation test.
- **All 8 runs** were COUNTED and the manipulation held on every rollout.

| Seed | Control rise | Critic-blind rise | Tail command at E (control / blind) | Margin to rail b* at E (control / blind) |
|---|---|---|---|---|
| 2501 | +0.543 | +0.016 | 0.88 / 0.18 | -0.03 / +0.98 |
| 2511 | +0.666 | -0.002 | 0.96 / 0.13 | -0.06 / +1.11 |
| 2521 | +0.384 | +0.012 | 0.60 / 0.15 | +0.40 / +1.30 |
| 2531 | +0.293 | -0.061 | 0.48 / 0.07 | +0.35 / +1.19 |

- **Informative:** the control median rose +0.464, against the 0.20 gate.
- **Separation:** every control ended above every blinded run (lowest control 0.419, highest blind 0.143). p = 1/70,
  the smallest attainable with 4 per arm.
- **Secondaries (pre-registered; they never change the line):**
  - **The ledger tail command at E: SUPPORTED** (p 1/70; survives Bonferroni).
  - **The margin to the rail at E: SUPPORTED** (p 1/70; survives Bonferroni).
  - **The end loop gain (last night's after-the-fact pattern): NOT SUPPORTED.** The blinded runs' gain was lower than
    the controls' in every pair. Controls that climb work against this secondary, so it does not refute last night's
    pattern at 1.05-2.49M.

## What else was seen (report only, untested)

- **Two of the four controls reached the rail; no blinded run did.**
  - control_2501 had 1 of 10 evaluation episodes over the bar at 3,407,872.
  - control_2511 had 2 of 10 there and 8 of 10 at E. That is a tail-flagged evaluation; an alarm needs two in a row,
    so none can show at the last evaluation.
  - Every over-bar episode was on tail_3. s42 itself first went over the bar at 3,670,016 (4 of 10).
  - No blinded run had any over-bar episode or flag.
- **The controls followed s42's climb, then lagged it.** The control median stayed within 0.03 of s42's command
  through 3,145,728, then ended 0.115 below it (+0.590 against +0.705). That is a median rise of 80% of s42's.
- **The loop stayed far from the rail in the blinded runs.** At E, the blinded runs' tail offset c0 was 0.10-0.25
  (0.08-0.25 at every saved checkpoint), against 0.44-0.78 in control. Their margin to the rail was 0.98-1.30
  (at least 0.95 throughout).
- **The critic's slope stayed positive in control.** Its slope on the tail angle was positive at every saved
  checkpoint (+0.008 to +0.062 per degree).
- **No sign of a return cost from blinding.** The blinded runs' window return was higher in all 4 pairs (+6 to +34).
  Each arm had one fall, both in episode 0 of the same seed-2521 evaluation [inferred: a shared start state, not the
  arm].
- **The spread caveat.** The arms' spreads differ (SD 0.166 against 0.036). If blinding had only narrowed the spread,
  with no shift in the mean, a complete separation like this would occur about 3% of the time at this spread ratio
  (2.5-4% for ratios 3-7; normal model), rather than 1.4%. The pre-registered p decides the line, and it stands.
  Post hoc: Welch t 5.67, one-sided p 0.004.

## What it means

- [measured] From s42's 2.10M state, removing the critic's view of the tail_3 angle stopped the climb of the policy's
  tail command toward the rail. With the view, all four controls climbed and two reached the rail within 1.44M steps.
- [inferred] This fits the verified probe: the critic's bootstrapped value pays for a raised tail, the policy gradient
  follows it, and the loop closes. It is the most direct causal evidence so far for the critic as the driver of the
  climb.
- **Not shown:**
  - that this works from other runs or states (only s42 at 2.10M was tested);
  - that a full training run with the critic blinded from the start avoids the habit and keeps a good stance;
  - what the onset at about 1.2M does;
  - whether the critic would in time recover the angle from its other inputs;
  - whether the habit moves to another joint. s80's habit was on the neck.

## Process notes

- The only repository change during the screen was the maintainer's merge of #654 (13:36 UTC), which is unrelated.
- The run log holds APPROVED's content as compact JSON rather than its exact bytes; the exact bytes can be rebuilt
  (sha256 184f9ab9). The 0.20 answer came at 12:53:09 UTC, 41 s before the pin.

## Files

`S2/r10round/bias_screen2/`: criteria.txt, APPROVED, score/score.txt, check/stat, check/proc, review/, NOTES.txt.
Run log: `SP/foot/stance4/live/run_log.txt` (BS2 from line 784).
