# Critic-phase screen from s42's 2.10M checkpoint (PS): result

2026-10-31. A scratch CPU training screen, report only; nothing merged. The reading below is the pinned scorer's
(score_bs.py e63cf15a, 20:19:27 UTC). Two independent checks confirm it: a statistics check (931 comparisons, 0
differences beyond round-off) and a process check (0 blocking). [inferred] marks reasoning, not measurement.

## The reading: line 2, REDUCED (status final)

> REDUCED: GIVING THE CRITIC THE EPISODE PHASE (the tail_3 angle still in its input; the actor unchanged) REDUCED THE
> ON-POLICY TAIL_3 COMMAND OVER THE LAST 8 UPDATES BEFORE E, FROM s42's 2,097,152 STATE (exact one-sided permutation
> test, p <= 0.05)

- **Design (pinned before launch):** 8 continuations of r10 stance6 s42 from 2,097,152 to 3,538,944 (1.44M steps):
  - 4 control;
  - 4 critic_phase. Only the critic sees the episode phase: a ramp to step 200, a settled flag and a discount, carried in
    the three always-zero command inputs. The actor is unchanged; the critic still sees the tail angle.
- **Seeds:** paired, 2901 / 2911 / 2921 / 2931.
- **Primary:** the policy's tail_3 command on its own states over the last 8 updates. The parent's value at the start
  (B_S) is +0.127.
- **Gate and test:** BS2's. The control median had to rise at least 0.20; the test is an exact permutation test.
- **All 8 runs** were COUNTED, and the manipulation check held on every rollout. Only the actor's output is unchanged by
  the phase; the critic learned to use it.

| Seed | Control primary | Phase primary | Ledger tail command at E (control / phase) | Margin to rail b* at E (control / phase) | Critic tail slope at E, per degree (control / phase) |
|---|---|---|---|---|---|
| 2901 | +0.494 | +0.068 | 0.63 / 0.08 | +0.26 / +1.93 | +0.058 / -0.000 |
| 2911 | +0.586 | +0.047 | 0.70 / 0.04 | +0.18 / +2.36 | +0.053 / -0.004 |
| 2921 | +0.661 | +0.043 | 0.77 / 0.05 | +0.11 / +2.08 | +0.050 / +0.008 |
| 2931 | +0.646 | +0.068 | 0.87 / 0.09 | +0.05 / +2.04 | +0.038 / +0.001 |

- **Informative:** the control median rose +0.490 against the 0.20 gate. That is 85% of s42's own rise of +0.579 over
  this window.
- **Separation:** every control ended above every phase run (lowest control 0.494, highest phase 0.068). p = 1/70, the
  smallest attainable with 4 per arm.
- **Below the start:** the phase arm's tail command ended below where it started (+0.057 median against +0.127 at S).
- **Secondaries (pre-registered; they never change the line):**
  - **SEC-CMD, the ledger tail command at E: SUPPORTED** (p 1/70; survives Bonferroni).
  - **SEC-MARGIN, the margin to the rail at E: SUPPORTED** (p 1/70; survives Bonferroni).
  - **SEC-MECH, the critic's own tail slope at E: SUPPORTED** (p 1/70; survives Bonferroni). The phase critic's slope
    fell from +0.021 per degree at S to between -0.004 and +0.008 in all 4 runs; every control's rose (+0.038 to
    +0.058).
  - **SEC-TAIL2, the guard: SUPPORTED** (p 2/70). This is a warning, read with its size below.

## The critic's error by episode step (separate pinned tool, report only)

The V − G tool's hashes were pinned in the run log before the screen was decided. It used 32 on-policy episodes per
final checkpoint. Figures are arm medians of each run's mean, in the critic's own units [measured].

| Steps | Control V − G | Phase V − G | Control mean abs V − G | Phase mean abs V − G |
|---|---|---|---|---|
| 0-50 | +0.003 | -0.011 | 0.051 | 0.044 |
| 50-150 | +0.037 | -0.001 | 0.066 | 0.018 |
| 150-200 | +0.014 | -0.011 | 0.057 | 0.017 |
| 200-250 | -0.068 | -0.009 | 0.069 | 0.013 |
| 250-1000 | -0.019 | -0.004 | 0.027 | 0.009 |

- **The control critic still mistimes the gate.** It overvalues the steps before the gate and undervalues the steps
  just after it. These are the same signs the why-probe found on s42.
- **The phase critic is nearly unbiased in every bucket.** With the phase hidden from it, the same trained critic
  undervalues settled states by about 0.37, so it relies on the phase.
- **The pre-stated expectations were met:** E1 at 150-200 and 200-250, and E2.
- **Tail exploration noise (σ = exp(log_std)) did not collapse.** It falls slowly in both arms from 0.182 at S. At E,
  tail_3 is 0.161 in control and 0.155 in the phase arm (median). The 0.01 floor never applies.

## What else was seen (report only, untested)

- **Returns were higher in the phase arm in all 4 pairs.** Window means were +36, +66, +62 and +56 higher. The control
  returns sagged late as the tail climbed; the phase arm's did not.
  - Each arm had one fall, both in the seed-2901 evaluation at 3,407,872 (tail_contact).
  - [inferred] That is a shared evaluation start, not the arm. BS2 had the same coincidence.
- **No control reached the rail in this window, unlike BS2's.**
  - The control ledger tail command rose to 0.63-0.87 by E.
  - No evaluation was tail-flagged in either arm.
- **The guard's size is small.**
  - SEC-TAIL2 reads SUPPORTED mostly because the controls' tail_2 command fell (rises -0.001 to -0.038). The phase arm's
    tail_2 barely moved: median rise +0.004, range -0.009 to +0.016, against the parent's +0.023 at S.
  - Post hoc, the phase arm's tail_1 pitch command rose (+0.04 to +0.06) while the controls' fell.
  - [inferred] Neither is a habit moving to another joint at this size, but both should be watched in a longer run.
- **The spread caveat, in BS2's form (statistics check).**
  - The arm SDs are 0.076 and 0.013, a ratio of 5.7.
  - If the phase input had only narrowed the spread, with no shift in the mean, a complete separation would occur about
    3.6% of the time at this ratio (normal model), rather than 1.4%.
  - The pre-registered p decides, and the line stands. Post hoc: Welch t 14.0, one-sided p 0.0003. The lowest control
    is 32 phase-arm SDs above the highest phase run.
- **Beside BS2 (different seeds and harness build; not tested).**
  - Both screens read REDUCED, with similar controls: medians +0.590 and +0.616.
  - The phase arm ended lower than BS2's blinded arm: +0.04 to +0.07, against +0.07 to +0.14.

## What it means

- [measured] From s42's 2.10M state, telling the critic the episode phase stopped the climb of the policy's tail
  command, and the command fell back below its start. The critic still saw the tail angle throughout. Its slope on the
  angle went to about zero, and its error around the step-200 gate went to about zero.
- [inferred] This is the most direct test yet of the explanation from the why-probe:
  1. four stance reward terms switch on at step 200;
  2. the critic cannot see time, so it reads the slowly creeping tail angle as a clock;
  3. the policy gradient then pays for raising the tail.

  Removing the hidden-time problem removed the payment and the climb, without hiding anything the robot can sense.
- **Unlike BS2's blinding, this is a fix of the cause rather than a stopgap.** It keeps the critic's full view and adds
  only information the training environment already has. The actor and the deployed policy interface are untouched.
- **Not shown:**
  - that this works from other runs or states (only s42 at 2.10M was tested);
  - that a full run with the phase critic from the start avoids the habit and keeps a good stance;
  - what happens over a longer window;
  - whether the effect would carry over to main's configuration, where only stance_width is gated (#655);
  - that SEC-MECH is separate from a tail that simply climbed less (KNOWN LIMITS: no sham-input arm).

## Process notes

- **Two container restarts, during idle gaps (about 18:07 and 18:53 UTC), killed the second wave twice.**
  - The coordinator relaunched the runner each time. The runner recorded the killed runs as crashes not counted, retired
    them unread to `*_crashed1` and `*_crashed2`, and reran them from the start pair with the same seeds. The counted runs
    are one complete attempt each, attempt 3.
  - A session watcher then kept the container alive. [inferred, consistent with the process check's timing] The cause
    was idle reclamation of the session's container.
- **Process finding M1 (minor): a wording gap in the criteria's EARLY-crash clause.**
  - The clause is not limited to a continuation's own crashes, and the attempt-2 kills fell below the early-crash step.
  - The pinned runner applies that clause only to runs whose status is "crashed", so it restarted at once rather than
    halting.
  - The cause had already been found (a container restart, uptime 2 minutes) and the remedy started 15 s later. No
    outcome was involved.
- **Process finding M2 (minor): rule breaches.** The coordinator redirected stderr to the null device three times (15:57
  and twice at 17:06) and to /dev/zero twice (19:35 and 20:31). The V − G checker did the same twice. Not all of these
  were self-reported at the time. Nothing was written.
- **Outcome-blindness held.**
  - The process check found 0 of the coordinator's 50 calls during the screen touched runs/, reads/, score/ or a
    continuation log.
  - All 274 re-armed files were unread at ALL_DECIDED.
  - The statistics checker's own wc -l glob also matched the crashed folders' evaluation ledgers. It printed line counts
    only, after the reading.
- **The only repository change during the screen was the maintainer's merge of #657 (20:27 UTC).** It is docs only and
  came after the reading. The screen ran from the sealed code copy (331fe8fc).

## Files

`/home/user/work/phase_screen/`:
- criteria.txt, APPROVED, SCREEN.json;
- score/score.txt, score/score.json;
- check/stat, check/proc;
- review/, NOTES.txt;
- the separate tool's output in `/home/user/work/vg_run/out/`.

Run log: `SP/foot/stance4/live/run_log.txt` (the pin at lines 832-833).
