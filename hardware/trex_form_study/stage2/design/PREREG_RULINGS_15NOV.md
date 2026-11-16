# Lead's rulings on STAGE2_75_PLAN.md's "Open for the lead" (15 Nov 2026, 00:45 UTC; before any set is frozen)

Each item is settled as follows. "Proposed" means the reviewers' proposed wording quoted in that section.

1. Episodes before task 3's first unit (rule R4): ACCEPTED with one addition. "Between this commit and the first unit
   of task 3 the only closed-loop episodes run on a Stage 2 set are the build's checks: single episodes on set V0 under
   R0 rows. The R2 rows and the servo-delay wrapper are smoke-tested only on Stage 1's own run directory (the read-only
   S1 view, whose conditions and results are already published in Stage 1's records). No episode of F25F9 or F25F9cl
   and no R2 row on any of the three sets runs before task 3's first unit. Every such check episode is listed in
   stage2/runs/2026-11-15-75cm/dev_episodes.json and is no record." COST.md's rates are measured on Stage 1's own
   committed conditions (item 2).
2. What the budget cuts (H6): the proposed text is ACCEPTED word for word.
3. Running again after a result (R2, A13, R4): the three proposed texts are ACCEPTED word for word.
4. The omega control's records (B5, H6, R1): ACCEPTED: records under stage2/runs/2026-11-15-75cm/controls/omega_s1/,
   a `control` field reading "omega at Stage 1's 0.03 H crouch", and `omega` and `crouch_frac` enter the key of R1.
5. A turn frozen with `a1_over` (A11, C2, H3): the proposed text is ACCEPTED.
6. Verdict words away from 10 ms (H2): the proposed text is ACCEPTED.
7. What stands beside a candidate's verdict (H4): the proposed text is ACCEPTED. A pass of an R0 row is NOT
   replicated: R0 rows give no verdict (H2).
8. A turn that fails a gate other than the stride rule's (A11): the proposed text is ACCEPTED ("it goes to the owner.
   A turn has no other fallback ...").
9. Two reported conditions (B6, C4): both proposals ACCEPTED (stop/one_step read from Stage 1's command, labelled;
   walk/nominal = start, then nominal to the end, judged over its last 10 s, with peak ground reaction, cadence and
   stride judged as Stage 1 judges them).
10. The budgets and the plan's estimate (H6): the 16 + 12 core-hours come ON TOP of STAGE2_PLAN.md section 6's
    estimate for this CPU (about 80 core-hours), which was written before the 75 cm track existed; with the sets and
    screens (about 4 core-hours) the track adds about 32 core-hours, an estimate.
11. Which rules are this track's alone (S2-16): ACCEPTED, as: "(with the judged controller's two seeds in place of
    R2/42 and R2/43; rules A1 to A3, B5's crouch values, C2's headline for the three sets, H4, H6 and rule R4 are the
    75 cm round's own)".
12. STAGE2_PLAN.md section 9 (the lead's own wording is changed as follows):
    - "What had been run": add the PR's closed-loop runs as the 75 cm plan's fourth bullet words them.
    - Opening sentence: "..., records an owner decision, or adds a rule this round needs".
    - S2-9: "the ladders'" becomes "the speed ladder's"; the bar's reported delays become 0, 2, 4, 6, 8, 14, 16 and
      20 ms ("the plan's 5 and 15 ms are replaced by their neighbours, and 2 and 8 ms are added because the owner's
      decision of 14 Nov, P30, reports 0 to 8 ms beside the judged delay"); the last bullet becomes "the 75 cm track
      reads its walk at 0, 2, 4, 6, 8 and 10 ms and the rest of its bar at 0, 4, 6 and 10 ms (STAGE2_75_PLAN.md)";
      add: "A bench value that is not a whole number of steps is judged at the next whole step above it, and the step
      below is run and reported beside it."
    - S2-10: the proposed text is ACCEPTED (succeeding with the look, reading 1; mu 0.9; seeds committed before the
      run; one record in the run directory).
    - S2-11: (a) and (c) each write one G9 record per body, whose path is fixed by the R5 build before G9 runs; in (c)
      "with margin" means: played at the commanded clock (5 % fast) the reference passes its own item's readings, and
      the quick turn also reaches 80 degrees at the slowest commanded clock (STAGE2_75_PLAN.md A10); (b) asserts only
      that every 10 ms bar record of 75-M-H-10 holds both readings for every episode.
    - S2-15: the candidate is "priced with the Stage 0 screen, run through the official panels and shown to the owner
      beside the look ladder with clips (P30)".
    - S2-17: the proposed sentence is ACCEPTED ("R5, R6 and R2L edit no module that enters a bar or panels record key:
      they subclass or wrap such modules in designlab/stage2/; B.4's extensions of train.py, which is in no such key,
      stand.").
