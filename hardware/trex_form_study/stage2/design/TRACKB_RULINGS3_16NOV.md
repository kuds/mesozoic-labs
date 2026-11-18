# Track B rules draft: the lead's third list of rulings (16 Nov 2026)

Working paper. It settles the ten conflicts the integrator left open in the Track B rules draft
(`checkpoint/reviews/trackb_rules__integrate_trackb-rulings.md`; the draft is `STAGE2_TRACKB_PLAN.md` on the
branch `wt/prereg-b`) and two open points on the policy files. **No Track B run, fit or training run has been made,
and no look set is frozen**, so every ruling here is made before any Track B result. The 75 cm panels and their
add-on have run; where a ruling could lean on them it says so. Rule numbers are the draft's.

An integrator writes these into the draft with the label "settled 16 Nov 2026, before any Track B run", and an
independent reviewer checks the result. Where a ruling says "the integrator checks", a mismatch comes back to the
lead and is not settled by the integrator.

## The ten conflicts

1. **M5 against M3 and M6 (which runs are judged).** M5's sentence becomes: "Every run of the Colab queue that
   passed G8a is judged on the panels at its last kept checkpoint, whatever ended it." S2-4 (M3) and the local
   60 cm run (M6) are not runs of the queue. Their checkpoints are never judged on the panels, as M3 and M6 say.

2. **L13 against L10 (the turn in place's flat feet).** L13 no longer cites L10. It says: "The turn in place has
   flat feet as its own law: no toe-off and no rolling entry in any stride, the start and the stop included." L10
   is about starts and changes of a walk and does not govern this gait. The integrator checks that the look work
   package's code builds it so (`wt/wp-l`).

3. **L10 against `STAGE2_75_PLAN.md` A5 and A7 (frames before a move's instant).** L10's sentence "the frames
   before the instant equal the look walk's, foot pitch included" is wrong as written: starts and changes are built
   with foreknowledge (A5), so the body's motion before the instant is the move's own. It becomes: "Before the
   instant, the foot pitch of a change follows the law of the walk it leaves, toe-off and rolling entry included.
   The rest of the frame is the move's own, built with foreknowledge (A5), and A7's gates read it. A start has no
   frame before its instant but the stand." The same correction applies where L10's account of the committed
   wording repeats the sentence.

4. **L12 against A8 (a start that still fails).** A8 stands on a look set: a failed `start` or
   `nominal_to_v0.4` stops that body's set, which is then not frozen, and goes to the lead. "Not available" is for
   the other moves: a ladder change (A8 step 3), the stop (its conditions are then "not run", H3), a turn (A11) and
   the turn in place (L13). L12's last sentence on "not available" names these and excepts the two.

5. **U4 and the stop item (the bar's record against its companion; the candidate set).** Two parts.
   - On the look sets `V6` and `V6noF5`, reading 1 lives in the companion record and is the verdict of a Track B
     row. The bar's own record is printed beside it as "the bar's reading (C7)". Wherever the two differ on a stop
     item, the table prints both and names the term `on_command_at_stop`.
   - On any other set, `F25F9cl` included, reading 1 does not count (U4 says so already). The row
     `R2L-F25F9cl/42` is therefore judged there by the bar's own readings, exactly as `R2/42` and `R2/43` beside
     it. The term `on_command_at_stop` is recorded in its companion and decides nothing on that set. Q6 says this
     in one sentence.

6. **Q6 (no budget for the R2L rows' panel records and the re-runs beside them).** Each gets a cap, written into
   Q5 beside the fits' shares: 1.5 times the bar runner's dry-run total at the committed rates
   (`stage2/runs/2026-11-15-75cm/COST.md`), per set and body, counted in the bar's own `budget.json`. The
   integrator runs the dry run for each (the 16 gated conditions and `walk/nominal` at 10 ms; the same at 0, 4 and
   6 ms where the plan asks for them) and writes the commands and the numbers in core-hours into the rule. Order
   of cuts when a cap is reached: the 6, 4 and 0 ms records first, in that order; the 10 ms records are never cut.
   If the draft's "R2 re-runs" turn out to be something other than `R2/42` and `R2/43` run beside the R2L row on
   the same set and code state, the integrator hands the point back.

7. **M1 against K5 (a check run "again, once", and a resume).** An attempt at a check is logged in `run.json`
   `monitor[]` before the monitor starts, with the checkpoint's file hashes, and closed with the monitor's exit
   code when it ends. M1's "once" counts closed attempts that left no record: after two of them the check counts
   as without a gain and as a failed G8b reading. An attempt that was never closed (the session died while the
   check ran) is an interruption: K5's resume runs that check again and it does not count toward the two.

8. **Y10 against Y11 (a smoke that cannot pass), and one number.** Y10 becomes: "The smoke passes when Y1, Y3, Y6
   and Y9 pass and Y4 passes or has been answered by the owner's budget (`QUEUE.json` `budget_by: owner`, with
   the date)." Y11's floor is written as the inequality alone, `r x 72,000 < 15,000,000`, with no rounded rate
   beside it (the draft's two rounded values, 208.33 and 208.34, both go).

9. **Q5 ("a started cell finishes" against "the only overrun").** A reported cell starts only when the fit's share
   still holds its estimate: the decisive cell's mean seconds per episode, times 25, times 1.25. Otherwise it reads
   "not run: budget". A started cell finishes. If it passes the share all the same, the excess is recorded
   (`overrun_reported`) and no further cell starts. The words "the only one that may pass the share" become "the
   only one that may pass the share without limit".

10. **X1 against the wording of `STAGE2_75_PLAN.md` R4.** The lead adds a dated closing amendment to
    `STAGE2_75_PLAN.md` that points to X1's reading of R4 and records that task 3's and task 4's units ran on
    15 Nov 2026 and that the lead lifted R4 and X1 for the frozen 75 cm sets on 16 Nov 2026. X1's first sentence
    gains: "(lifted for the frozen 75 cm sets on 16 Nov 2026, after the panels)". The look sets' part of X1 is
    unchanged.

## Two open points on the policy files (work package CP)

11. **Whether a judging load reads the `.pkl`** (review of CP, finding 1). Confirmed as built: by K9 a judging
    load does not read it. Nothing changes.

12. **`actions_at_bound` in the fingerprint's record** (review of CP, finding 8). Accepted: it joins the record's
    keys, with its assertion, before `s2load.py` is frozen at state B. It goes to a builder as a small change to
    CP's own files.

## The reward's wording (work package CE)

13. `stage2/design/CE_RULE_WORDING.md` is integrated into the draft's W, E, J, K and T rules after the env's
    small fix round has updated it. The lead's ruling on W1 is in `checkpoint/STATUS.md` (16 Nov, 03:02 UTC): rule
    E4 stays in walking turns, and W1 says that a playback's heading penalty is not zero there.
