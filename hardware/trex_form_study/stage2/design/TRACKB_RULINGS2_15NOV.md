# Lead's rulings on STAGE2_TRACKB_PLAN.md's 25 open points (15 Nov 2026, 03:50 UTC)

Fixed before any closed-loop episode on a look set, any fit and any training run of Track B. "Proposed" is the
reviewers' proposed wording quoted under the point in the committed text (103bf35).

1. G9-ref on V6noF5: on `V6noF5` part (b) makes no hash and no toe-off comparison with the ladder and records the
   toe-off as "not read"; bob, head and tail are compared with the same body's V6 set's own playback; part (c)'s
   flat-feet playback applies to V6 only. X1: R2L's set must pass G9-ref in this form before its first episode.
2. X1 and rule R4: `STAGE2_75_PLAN.md` rule R4's "a Stage 2 set" means the 75 cm round's three sets (V0, F25F9,
   F25F9cl); rule R4 is the 75 cm round's own by S2-16. X1 is tagged "differs: rule R4's wording" and listed in the
   change list.
3. W4: the interference penalty is ACCEPTED (weight as drafted) and the termination is ACCEPTED, tagged "differs: plan
   B.1's reward list" (LESSONS_LEARNED.md section 2.8 is the reason). A fourth term beyond the plan's list.
4. L12: the proposed text is ACCEPTED word for word, with the tags the other reviewer proposed ("differs: A8, A11, for
   a join gate failed on a K-stride window"; cut 3 differs from B6 and T1). A set rebuilt under a cut before any
   closed-loop episode on that set is not "a change after a result" (a join gate is a kinematic build reading); after
   one, it is. A nudge's onset "2 + u strides into the walk" is read on a K-stride window as (2 + u) x the walk's
   stride period `period_s` after the walk segment's first frame.
5. L13: the proposed text is ACCEPTED, and the numbers are confirmed: 4 hold strides more than Stage 1's start writes;
   the reference is the second of them; cut 4 is in force from the start for the turn in place and is no further
   fallback.
6. L10 and L11, starts and changes: the instant is the start of the move's ramp (the entry's `move.command_phase`,
   Stage 1's convention). Every stance that ends between that instant and the end of the move's last ramp is flat; a
   stance that ends after the last ramp has ended follows the walk's own law, so that the move's last frame equals the
   walk's first, foot pitch included; the frames before the instant equal the look walk's, foot pitch included. The
   same per-stance rule holds for a causal move with its command instant in place of the ramp's start. (This refines
   the lead's choice "flat feet inside every move": flat while the gait is changing.)
7. R2L on the candidate set (Q6): `R2L-F25F9cl/42`'s bar records are reported with A3's label and NO verdict word; it
   is not replicated. It runs the 75 cm plan's `walk/v0.4` and `walk/v0.4_mu0.54` at 10, 0, 4 and 6 ms and the other
   fourteen gated conditions at 10 ms, on seeds 3042-3061, beside `R2/42` and `R2/43`, which are run again from the
   same commit where a key module differs (S2-17).
8. Q5 and Q4: both proposed texts ACCEPTED word for word.
9. M1: the first proposal is ACCEPTED word for word (an error of a policy row counts as not succeeding; R0L's baseline
   is repaired and run again whole; a check without a record is run again once).
10. M3: the FIRST option: N* is the largest N, and S* the largest S, over R0L's 18 episodes and every earlier check
    of the run, each taken separately; for the first check they are R0L's.
11. M3b: the FIRST option: the route belongs to S2-4's reading alone; elsewhere its count is printed and decides
    nothing.
12. M3: the proposed text and Check are ACCEPTED word for word.
13. M5: the proposed text is ACCEPTED word for word.
14. M6: the proposed text is ACCEPTED word for word.
15. The stop item: the FIRST option (the term `on_command_at_stop` for Track B rows; an episode that fails it succeeds
    in no reading; tagged NEW, differs: C7).
16. U4: on a turn frozen with `a1_over` reading 1 reads "not read: a1_over"; its headline on a look set is then
    reading 3 with the four bands kept, printed "shown but for A1, with the look" (or "provisional ..."), a verdict for
    that case only. Reading 1's two words are "shown with the look" and "provisional with the look".
17. O3: the group is the body: the probe is run once per body and head as proposed; every run of a body (S2-4, M6,
    Colab, fine-tunes' parents) uses the smallest passing rate over the heads of THAT body probed when the run starts;
    a body's `-X` head is probed before any Colab job of that body is queued, and if its rate is lower, later runs of
    the body use the lower one and the record of every earlier run says which heads its rate rested on. The two
    bodies may differ, and the report says so. (This changes the lead's ruling 10 of 02:45 UTC, "one rate for every
    body and head", before any run: comparisons are made within a body, and the 75 cm runs then need not wait for the
    60 cm sets.)
18. O4 and O2: the proposed texts ACCEPTED word for word.
19. Z2: the proposed text ACCEPTED word for word.
20. Z2 (b) and U3: the proposed changes ACCEPTED; U3's threshold stays 0.02 rad.
21. K8: the proposed text ACCEPTED word for word.
22. Y11, Y4, Y10: the proposed texts ACCEPTED word for word.
23. X1: the SECOND proposal is ACCEPTED (the training start with MUJOCO_GL assigned and R6's start run on the view
    S1; from S2-4's start every closed-loop episode on a look set belongs to a named run, check, gate, probe or panel,
    any other being listed in `stage2/tb/dev_episodes.json` before it runs). In addition R2L's fits (section Q) are
    added to X1's list of what may run on a look set once G9-ref has passed there; the order of R2L and S2-4 is free
    (a labelled change to plan section 7's order).
24. "Not run": the first wording ("outside this text: Track D's levers; a fine-tune runs under plan 4 D and section
    6, by rule K8; third seeds only from freed hours").
25. "What had been run": CONFIRMED by the lead as written, with this completion: the Track B builds' own test episodes
    ran since 02:59 UTC on the view S1 and on stub environments only; the look work package's kinematic builds of V6
    windows and moves began at 02:54 UTC, are prescribed-motion builds and no closed-loop episode, and their join
    numbers are what L12's trigger reads. The corrections the author applied after the review are confirmed. Builders
    of the remaining packages are given this file's name and its citation convention by the lead.
