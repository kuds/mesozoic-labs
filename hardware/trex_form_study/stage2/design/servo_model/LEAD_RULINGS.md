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

## Fourth list (17 Nov 2026, ~12:15 UTC), on S2 (`S2_{build,check}_17nov.md`)

S2 is merged as it stands.

- *The builder's (a), (b) and (e)* are accepted as built:
  - the option values are "draw", a named setting, or "D/<bar seed>";
  - the env refuses joint factors other than (1, 1) rather than forcing them;
  - the reset info names the setting only when the option is on.
- *The builder's (c), the mid-start forward pass.* Accepted as built. Once per episode, the first reading's forward
  pass runs on today's model arrays with the oldest queued command folded with the run's copies, and no command is
  served in that pass. `SERVO_RULE`'s words "today's model's" are read as the model arrays only. S3's rule text says
  this in a sentence; the existing string is not edited.
- *The builder's (d).* G8s is run again against S3's `servo.Served` when S3 lands. The named settings use seed
  3042 and D uses seeds 3042 to 3044, as built; the design names no seeds, and this list now does.
- *The builder's (f) and the check's point 6*, for S3 and S4:
  - `s2rows`' equivalence check gives an episode seed when the run's config has `servo="draw"`;
  - G8a is run again after S2, as trap 13 says, in S4's measuring step.
- *The check's finding 1.* Section 5.1's "Training env" row is amended to match 2d's split: copies at entry, model
  gains at the first physics step. The line references to `r5env.py` are refreshed. Both are done in S3.
- *The check's finding 2.* The early "draw" refusal's test matches `SERVO_RULE`. Done in S3.
- *The check's points 4 and 5* stand as built (trap 2; T14's form at the env as at the plant).
- *The names of the served rows (section 8, choice 18; the second list left them to the build's brief).* The
  design's examples: `SV0/<setting>` (the served R0 stand-in), `SV2/<seed>/<setting>` (the served R2 stand-in), and
  `SV5/<policy>/<setting>` for a served policy row once `s2rows.py` serves one. A served stand-in row gives no
  verdict and no served row is attributed (5.1). Every reader of their records carries its own lists.
- *Choice 27* (a second-block record that fails) was settled by the owner (`DECISION_LOG.md` P41): the repeat must
  pass too.

## Fifth list (17 Nov 2026, ~12:45 UTC), on S3 (`S3_{build,check}_17nov.md`)

S3 goes to one fix round, with a re-check, before it is merged.

- *Check, finding 1 (a repeat not run or not judged counts as a failure).* Fixed. "replicated" counts as passing,
  "not replicated" as failing, and anything else as pending (None). A test covers each case. This keeps C8: an
  episode that cannot be read is run again and never counts.
- *Check, finding 2 (the runner applies none of the bar's placement rules).*
  - The runner refuses SV2 rows (R2 inside) on a set that is not registered, as amendment 5 says of R2 rows.
  - Every gate1 run on V0, F25F9 or F25F9cl is a development run until gate 1's servo rules are registered. It is
    listed with its command, set, rows, settings, seeds and count in the runner's own listing, beside
    `stage2/tb/dev_episodes.json`.
  - The runner takes an episode cap (`--max-episodes`) and refuses a plan over it. S4 sets the rates; a budget
    ledger comes with the rules.
- *Check, finding 3.* A `V-` or `W1` run that leaves the supply window is not counted as stayed up, as the design's
  section 4 says. It is "left the supply window" in the counts and in the summary. Both settings are reported only,
  so no verdict moves.
- *Check, finding 4.* `--cut-s` is for tests only: its help says so, and a cut walk episode is never judged.
- *Check, finding 5.* Accepted as built: `describe()`'s `mid_start_rule` key when the option is on; `train2.sources()`
  moving; the refresh note in 5.1's row. The design's Files paragraph (line 28) is refreshed at the next text edit.
- *Check, finding 6.* `GateWorker` also checks that `servo.py` and `gate1.py` have not changed during a session.
- *The builder's (a).* Until SV5 exists, the equivalence check gives an unserved row's env `options.servo="today"`
  (the T3 identity, like with like). It compares against SV5 at the env's setting once SV5 exists.
- *The builder's (c).* The runner keeps refusing look sets. Served rows join the look sets' rule (`lookbar`) when
  SV5 is built: SV5 in the rule's families, after G9-ref like every row. SV0 and SV2 on a look set are development
  runs under X1's list.
- *The builder's (d).* The bar's form: a row that gives no verdict (SV0, SV2) is never sent to the second block. Its
  D record is reported.
- *The builder's (f).* Accepted: N and the reported settings are read by no rule here.
