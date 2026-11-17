# Servo model for gate 1: the lead's rulings on the open choices (16 Nov 2026)

Working paper. The design note was reviewed, revised and checked a second time (`REVIEW.md`, `FIX_REPORT.md`,
`RECHECK.md`). These rulings settle the choices the fixer and the second checker left to the lead. They are the
lead's proposals for a design that is not built; nothing here is a result. What is the owner's stays the owner's
and is marked so.

**From the second check (`RECHECK.md`, "Choices for the lead").**

1. *The pass rule at a corner.* At an extreme the robot must stay up in 20 of 20; the other limits are recorded,
   not judged. The drawn panel is judged by the bar's whole rule. This is an exception to the gate's sentence that
   its conditions use the bar's own pass rule, so it is said in the owner's question and written into
   `BUY_GATE_DRAFT.md` only if the owner approves it.
2. *The tail's kv.* Recomputed by the plant's rule under the new inertia, and `kv_scale` re-based on it: one rule
   for every drive.
3. *The tail's inertia range.* The nominal stays at the top of the range, and the note says so.
4. *Delay-like draws in the drawn panel.* Kept, and said.
5. *The stricter option keeps the drawn panel*, so that the owner's options differ only in the two harsh corners.
6. *Replication.* A drawn-panel record that passes is also run on the bar's second block (seeds 5042 to 5061).

**From the fixer (`FIX_REPORT.md`, "Open choices for the lead").**

1. *Five questions to the makers.* Asked before gate 1 is judged. Sending them is the owner's: the note carries
   the five questions as a text ready to send.
2. *Which settings can stop a purchase.* The owner's question (the note's section for the owner).
3. *"Passes" at a corner.* As ruling 1 above.
4. *Route of gate 1's records.* A runner outside `bar.py`, so that the bar's hash and every registered record stay
   as they are.
5. *The judged delay under the model.* The command lag and the 100 Hz hold stay in the nominal model and the note
   says so. A measured feedback age goes to `sense_steps`.
6. *The X6's low end.* Its zero-torque speed is also drawn if the maker has not answered.
7. *The tail (condition 4).* Read at the worse of nominal and high friction, with the tail's nominal inertia. The
   neck's same asymmetry is kept and flagged. Condition 4 stays a proposal until put to the owner.
8. *A torque-scale draw.* None now. The note says the 0.70 of assumption A2 is used up.
9. *The stand-in rows' ankle cap under drawn inertia.* Accepted.
10. *Steps below 20 V.* Reported in every record; judged runs are not failed on it yet.
11. *Mass corners.* Policy rows need a nominal-model estimator before the two mass corners can be judged. It joins
    the build's work packages.
12. *The bench sheet.* The six added measurements (N1 to N6) are proposed additions to
    `stage2/jetson75/BENCH_SHEET.md`; the gate draft's sentences on what the bench measures are reworded with them
    once the owner has seen them.

## Second list (16 Nov 2026, 15:55 UTC), after the note's last revision

The first list named 18 points; the note's section of open choices holds 27. The rest are ruled here.

- *Choices 2, 4, 5, 6, 8, 12, 13, 14 and 17* (the gain limits until the maker answers; the run-time dead zone for
  gear play on the leg drives; friction's load term on for all classes; the feedforward keeps today's friction
  values; a nominal bus of 22.2 V with 0.05 ohm; no servo draw in the critic's inputs now; the encoder offset
  patched after each step; the 60 cm body gets its own table before its own gate 1; the box servos' nominal
  stiffness stays the model's until the bench): the note's recommendation stands in each.
- *Choice 7* (kv under a drawn inertia): one rule for every drive, as ruled for the tail.
- *The X6's drawn corner and zero-torque speed:* the corner is drawn first, then the zero between the larger of
  8.01 rad/s and corner + 1.20 rad/s, and 9.215 rad/s.
- *The names of the served rows:* settled in the build's brief, not here.
- *A failed repeat of the drawn panel on the second block of seeds:* whether it stops a purchase is the owner's,
  and is asked with condition 2. The lead recommends that the repeat must pass too.

## Third list (17 Nov 2026, ~10:55 UTC), on S1's fix round (`S1_fix_17nov.md`)

- *The gain-copy split (fix report, point 1).* Kept. Section 2d's gain-copies table and the note to ruling 21 are
  amended to say what the code does. A served R2 row whose tracker is built inside the episode takes the run's kv
  in its ankle cap at J+ and W1, and both are reported only.
- *The five cells with no kind (point 6):* the armature of the neck, kp_scale of the tail, kv_scale of the neck and
  the tail, and latency_steps. Option (c): section 3 names their kind as "the study's assumption [A]", which is what
  the code already writes, so `TABLE_SHA256` does not move.
- *Trap 8 (point 7).* The stricter refusal stays: the layer refuses to enter over a step hook it did not install.
  Trap 8 is reworded to say so.
- *`train2.SOURCE_FILES` (point 8).* S2 adds `servo` and `g8s`.
- *The builder's (b):* the first box-servo target is taken at step 0, as built and tested.
- *The builder's (d):* X6L's slope stays 6.23, as built, while X6L is reported only.
