# Overnight report: the tail habit (2026-10-29 to 2026-10-30)

Written for the maintainer after the night's runs. Everything below is verified by an independent check unless marked
otherwise; [inferred] marks reasoning, not measurement. Nothing was merged; one scratch training screen ran under your
00:45 UTC authorisation.

## In one paragraph

The verified probe narrows what starts the tail rise:
- Only the policy-gradient part of training moves the tail command.
- s42 already has an upward push at its onset (1.31M).
- From 2.1M, the critic's bootstrapped value is supported as the source of that push.

The one training screen you authorised blinded the critic to the tail angle from s42's 1.05M checkpoint. Its
pre-registered reading is **NOT INFORMATIVE**: the control arm itself did not reliably repeat s42's rise. Only 1 of 4
controls rose like the parent, a second rose partway, and the control median fell just short of the 0.10 bar (+0.092).
So the screen says nothing either way about the critic. Report only, and not evidence:
- no critic-blinded run rose by more than 0.07;
- the blinded arm's mean was lower (+0.013 against +0.093; p 0.19).

The most useful new fact [measured] is that, from the 1.05M state, the rise is a matter of chance under fresh rollout
seeds. A follow-up screen needs a later start or more seeds. On early warning, no signal tested tonight separates the
runs that will rail from those that won't, and the r9 held-out test failed. Nothing was merged. #654 is waiting for your
review.

## 1. What starts the tail rise (the early-bias probe, verified with corrections)

- **Only the policy-gradient part of training moves the tail command.** The entropy and value terms have exactly zero
  effect on the tail command's mean; the mirror-symmetry term is small and changes sign.
- **The first rise is s42's own, and it is a policy-gradient push.** At 1.31M steps, about 0.1M after s42's rise begins
  (1.204M), s42 already has an upward push on tail_3 carried by tail_3's own exploration noise, and the critic's slope
  on the tail angle is positive. s80 has neither at that step; it gets the same kind of push about 0.8M steps later.
  (The verifier fetched the 1.31M checkpoints read-only from Drive to establish this; the first probe had only the
  1.05M-spaced local checkpoints and missed it.)
- **From 2.1M the critic is the source.** On the policy's own states, the push is larger than what the actually
  collected rewards justify: the critic's bootstrapped estimate of future value pays for a higher tail while the
  short-horizon rewards push it down. The push grows tenfold in s42 by 3.15M and disappears in s80.
- **Not settled:** whether the critic also drives the very first push (at 1.31M the data cannot separate the critic
  from the collected rewards), and why the critic comes to value a raised tail. One reading is that the tail angle
  becomes a stand-in for "how far into the episode" [inferred, a hypothesis the verifier did not settle].

## 2. Can the habit be seen coming? (early-warning curve and two follow-ups, verified)

- **The loop flips just before the rail, but s80 drifts there too.** The tail loop's gain turns positive one saved
  checkpoint before the first saturated tail episode in s42 and s60. Late s80 reaches the same loop state without ever
  saturating, so no loop-only rule separates them with any margin.
- **Signals training already records are as early or earlier** (the tail command on the fixed reference states; the
  ledger's tail command), but:
- **They fail on the r9 runs.** Tested out of sample with the rules locked first, the reference-state signal fired in
  all 5 r9 runs by 2.1M, including the cleanest no-habit run. Evidence against it as a cross-run warning.
- **How close to the rail (margin test):** the smallest constant push that parks the command at +1:
  s42 1.09 (2.1M) and 0.28 (3.15M); s60 0.70 (6.3M) and 0.20 (7.3M); s80 0.52, 0.33, 0.30 (10.5, 11.5, 12.0M). Late s80
  is as close as s42 was one checkpoint before it first saturated, but approaches 4-12 times more slowly. A margin that
  accounts for the loop's gain ranks all seven checkpoints correctly; none rails on its own at these checkpoints.
- **Correction to the record:** the early loop-gain values in yesterday's posture results came from a sweep wide enough
  to hit the normaliser's clip; unclipped they are 2-3 times more negative (signs unchanged).

## 3. The training screen: does blinding the critic to the tail angle stop the early rise?

- **Design (locked before launch):** 8 CPU continuations from s42's 1.05M checkpoint (before the rise) to 2.49M:
  4 control, 4 with the critic blinded to the tail_3 angle (the actor unchanged). Primary outcome: the tail command on
  the policy's own states over the last 8 updates; exact one-sided permutation test, alpha 0.05 (smallest possible p
  1/70); informative only if the control arm's median rises by at least 0.10. The first draft of the criteria and
  scorer was reviewed independently (9 findings, none blocking). The fixer applied and tested the fixes; the fixed
  version was not re-reviewed. It was pinned (APPROVED sha256 7bf25afd...) 14 s before launch.
- **Reading: line 1, NOT INFORMATIVE** (status final; both independent checks agree, 0 blocking findings). The control
  median's rise was +0.092, against the pre-registered 0.10, so no reading about the critic is made.

  | Seed | Control (rise from 1.05M) | Critic blinded (rise) |
  |---|---|---|
  | 2101 | +0.001 | +0.060 |
  | 2111 | +0.057 | +0.003 |
  | 2121 | +0.126 | +0.068 |
  | 2131 | +0.314 | +0.047 |
  | s42 itself, same window | +0.231 | |

- **Only control_2131 rose like the parent** on the primary and the logged tail command (0.32 against the parent's
  0.25). control_2121 rose partway on the primary. On the report-only resting-pose tail offset, control_2121 tracked the
  parent (+0.21 against +0.22) and control_2131 did not (+0.12).
- **What this does tell us [measured]:** from the same 1.05M networks, the early rise happens in some seeds and not in
  others. It is not locked in by 1.05M. Even so, every one of the 8 runs was still far from the rail at 2.49M: no tail
  alarm, and no logged tail command above 0.32.
- **The manipulation held.** In every blinded rollout (700 of 700), the critic's output did not change at all when the
  tail angle was perturbed. Returns and falls look alike in both arms (evaluation means 4,400-4,475; 0-2 falls in 110
  episodes per run).
- **Report only, not tested, spotted after the fact:**
  - The blinded runs' tail offsets stayed small (at most 0.09 at the 7 saved checkpoints). Even so, one blinded run's
    loop gain turned positive (+0.32), the only run of the 8 to do so.
  - Across the 4 controls, the critic's slope on the tail angle at 2.49M ranks the same way as their rise: +0.060,
    +0.012, +0.004, -0.019 for 2131, 2121, 2111, 2101. This is consistent with the critic idea, but 4 runs ranked
    after the fact is not evidence. (These slopes come from the scorer's report-only lines; the checks did not test the
    pattern.)
- **Within the authorisation [measured]:**
  - 8 continuations of 1,441,792 steps each, with one intervention;
  - outcome-blind (no continuation was read before all 8 were decided);
  - nothing pushed or merged during the screen.

## 4. Repository

- **#653 merged** (00:15 UTC): the CoM absolute-target tests, the calibration ratchet and the reduction's move into
  the gait package. CI 10/10.
- **#654 open, CI green**: the records PR with the tail-habit note, its ledger and evidence, run log lines 721-759, the
  decisions D-D52 (amended) to D-D55, and #652/#653's landings. It lists a few defaults for you to accept or change.

## 5. One incident

At about 01:05 UTC an agent ran a command that replaced the system `/dev/null` with a regular file; it restored it
within a minute (character device 1:3, mode 666), and I confirmed it is correct. The early-bias probe's logs show no
error and no file written in that window. Agents were told afterwards never to redirect outside their own folders.

## 6. What I suggest next

These are proposals. Each needs your word, and options A and B need a reviewed and pinned pre-registration first.

- **A. (Recommended) Re-run the critic-blind screen from s42's 2.10M checkpoint.** This is the point where the critic
  is verified as the source and the push grows tenfold by 3.15M.
  - Design: 4+4 continuations to about 3.6M, within the 1.5M-step size you allowed. That window covers s42's climb:
    its tail command went from +0.15 at 2.10M to +0.52 at 3.15M (deterministic traces) and +0.82 at 3.41M (training
    ledger). It ends just before
    s42's first over-bar episode, at 3.67M.
  - Question: does hiding the tail angle from the critic stop the progression toward the rail?
  - Why: this is the practical question behind a fix, asked where the control is more likely to move. That is
    [inferred]; the gate should still be pre-registered.
  - Cost: about 3 h on CPU, the same as tonight. It also needs a small harness change for the start checkpoint
    (the 2.10M checkpoint is already held locally).
- **B. A larger onset screen from 1.05M:** 8 per arm, about 6 h on CPU. Use it only if you still want the onset
  question answered. Tonight's control spread suggests 4 per arm cannot be powered for it.
- **C. If A shows blinding stops the progression:** a full-length training run with the critic blind to the tail
  angles, as a candidate fix. This is a model-interface change and needs your approval.
- **Housekeeping (no approval needed unless you object):**
  - for future screens, key the reader's cache on the reader's own hash (process check P3);
  - remove the old scratch worktrees (about 2.2 GB of disk is free);
  - review #654.
- **Still parked:** the CoM-P1 report-only reading goes into the next round's pre-registration.

## Files

- Early-bias probe: `S2/com_revisit/notes/early_bias/`, verifier `verify_early_bias/`
- Early-warning curve: `early_warning/`, `verify_early_warning/`; follow-ups `warning_r9/`, `warning_margin/`,
  `verify_warning_followups/`
- Screen: `S2/r10round/bias_screen/` (criteria.txt, APPROVED, score/, independent checks in check/stat and check/proc)
- Run log: `SP/foot/stance4/live/run_log.txt`

(S2 = /tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad)
