# Lead's rulings on the round-C design and PREREG_TRACKB_draft.md (15 Nov 2026, 02:45 UTC)

Fixed before any Track B episode, fit or training run. "Draft" is design/PREREG_TRACKB_draft.md; item numbers are its
"Open for the lead" list. Every [RULING NEEDED] tag is resolved here; the committed text carries the tag
"[lead, 15 Nov, before any Track B run]" with the departure from the plan named.

1. M3b (a second route to a gain when the survival route is closed): RULED IN, as drafted, labelled as differing from
   S2-10. A "no gain" goes to the owner with the per-term failure table of both rows, whatever term it rests on.
2. E3's heading input: ACCEPTED (labelled: one input beyond plan B.1's list; it is the estimator's own yaw against
   the commanded heading, a quantity the robot has). W5's three terms: `progress` ACCEPTED, `heading` ACCEPTED,
   `range` ACCEPTED, each labelled as beyond plan B.1's list, with the drafted weights.
3. T2: ACCEPTED. "Walking turns at 0.35-1.0 rad/s" are the finite 90 degree turns; no periodic walking-turn gait is
   built or trained. "No loss" for R5 and R6 is run as the drafted look-set counterparts, with the finite 0.35 rad/s
   turn in place of the periodic walking turn, at 0 ms, Stage 1's committed counts printed beside them; each
   departure is labelled in the records and the report.
4. M3's credited survival: ACCEPTED (labelled as differing from section 6's "mean survival"). M1, U5, U6 (an unread
   or too-short episode of a policy row is "not succeeding", never run again; C8 stands for rows without a policy):
   ACCEPTED.
5. O2, sigma: the plan's initial sigma 0.1 of each action's range is KEPT. The owner is told that early training
   episodes are short. The learning-rate probe must pass at sigma 0.1 and at 0.03, as the final design has it.
6. O1: n_steps 1024 with 8 envs is ACCEPTED. E6: CONFIRMED: no extension of designlab/stage1/train.py (the trainer is
   designlab/stage2/train2.py: a labelled change to plan B.4's wording) and no hook in bar.py (the look reading is
   lookread.py with the lookbar.py row wrapper). Round C edits no existing file.
7. E2: R6's expected outcome (standing or shuffling) may be stated in advance, as drafted. E5: gaze script H1's
   numbers ACCEPTED as drafted.
8. T1: the task weights ACCEPTED as drafted. A set without the turn in place may be trained on; the report then says
   plan B.1's list is not met for that body.
9. M6 (the local 60 cm run: seed 41, 5,000,000 steps, read by G8b's rule, 5.2 core-hours): ACCEPTED, to be run
   before any 60 cm Colab job, not before the day-one smoke; its cost comes on top of the plan's estimate.
10. O3: one learning rate for every body and head (the smallest that passes the probe on any of them): ACCEPTED.
    O4's 0.9: ACCEPTED.
11. Q5: R2L's budget is 12 core-hours: two fits on V6noF5 (75-M-H-10 first, then 60-N-T-10; 8 core-hours) and one
    on the candidate F25F9cl on 75-M-H-10 (4 core-hours), in that order; a fit that does not fit the budget is cut
    and reported so.
12. Y10 and K7: ACCEPTED as drafted.

## The look inside moves (work package L): rules fixed now

The round-A design's section 3.9 (rules L1-L10, J1-J3) stands, with these choices made now, before any look set is
built, each a labelled look choice that the owner is told about:
- F5 (toe-off with rolling contact) lives in the periodic walks only. Inside EVERY move (start, change, stop, turn)
  the feet are flat by rule L10, stated per stance so that commanded moves stay causal. (Cuts 1 and 2 of 3.9, taken
  from the start.) V6noF5 is V6 with F5 off in the walks as well.
- J1 (a move leaves a K-stride window at the warp's own time of its command phase, with the window resampled on the
  move's grid) is built as designed. Its pre-registered fallback, used only if a V6 move fails a join gate
  (join_command or join_b, A7's limits) after the builder's implementation is reviewed, is cut 3 ("a move leaves and
  enters a K-stride window only at its boundary"), then cut 4; a fallback taken is recorded per set with the failing
  gate's numbers.
- The turn in place (0.5 rad/s) is a periodic look gait cut from the steady part of a long plan, with its start and
  stop, gated for periodicity as 3.9 says; if it fails that gate on a body it is recorded "not available" there
  (ruling 8).
- Sets, in build order: V6 on 75-M-H-10; V6noF5 on 75-M-H-10; V6 and V6noF5 on 60-N-T-10; V6 on 75-M-H-10-X and
  60-N-T-10-X (neck yaw and jaw at their home angles in the references). Speeds as the 75 cm plan's A2; a rung over
  a leg drive's Stage 0 limit is dropped with its numbers (A12; the 75 cm animal-pace rung is already out, L1 freeze).
- Every rule of STAGE2_75_PLAN.md sections A (A4-A13) applies to these sets (S2-16): the stride rule, the appendage
  rule, the causal stop and turn, the gates, what blocks a freeze.

### ADDENDUM for work package L, 15 Nov 03:50 UTC (the lead's later rulings; they override the lines above)

The independent review of the rule text raised three points on the look rules; the lead ruled (full text:
design/TRACKB_RULINGS2_15NOV.md, points 4, 5 and 6; the committed rule text is STAGE2_TRACKB_PLAN.md, section L):
- FLAT FEET WHILE THE GAIT IS CHANGING (refines "flat feet inside every move"). For a start or a change the instant is
  the start of the move's ramp (the entry's `move.command_phase`); for a causal move it is the command instant. Every
  stance that ENDS between that instant and the END of the move's last ramp is flat (L10). A stance that ends after
  the last ramp has ended follows the walk's own law, so that the move's last frame equals the walk's first frame,
  foot pitch included; the frames before the instant equal the look walk's, foot pitch included.
- THE JOIN FALLBACK (L12). It is taken only when a move of a look set, built after an independent review of the join
  code (the review's record named by path and sha256 in `set.json` `rules`), has `join_command` or `join_b` outside
  A7's limits. A change or a start that fails is first retried by A8's steps under join rule J1; cut 3 is taken for a
  causal move (the stop, a turn) or when A8 is exhausted. A cut is taken for a BODY, not for a move: every look set
  of that body and of its -X head (V6, V6noF5) is rebuilt under it. A move that still fails is "not available" with
  its numbers. Under cut 3 the "2 (2.5) strides of v0.4" before a command read "one window of v0.4 (and half a stride
  for a right-foot command)". No limit is changed for a failing move.
- THE TURN IN PLACE (L13). The plan is the start from the stand, then 4 hold strides more than Stage 1's start
  writes. The reference is the SECOND of those hold strides: one stride, with no F14 variation and flat feet; the
  periodicity gate compares it with the hold stride before it and the hold stride after it. No other plan length and
  no other stride is tried: a gait that fails is "not available". Cut 4 (no F14 inside periodic turns) is in force
  from the start for the turn in place and is no further fallback.
