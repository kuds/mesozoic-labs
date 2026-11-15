# Lead's text for STAGE2_PLAN.md section 9: "Amendments before any run (14 Nov 2026)"

Every item below is a rule fixed on 14 Nov 2026, before any Stage 2 closed-loop episode, fit or training run. Each
settles an open rule of LESSONS_LEARNED.md section 5.3 or records an owner decision (DECISION_LOG.md P30). Where an
item changes the plan's own wording it says so. Numbering continues the plan's S2-n.

## What had been run when these rules were fixed (state this in the section's opening paragraph)

- No Stage 2 closed-loop episode on any new reference, no fit, no training run.
- Kinematic and Stage 0 scratch prototypes of references on 75-M-H-10 and 60-N-T-10 (thigh-twist screens of steady
  turns, a prototype causal stop and quick turn of today's 0.40 m/s walk, their Stage 0 drive ratios). Constants
  marked "(prototype)" in STAGE2_75_PLAN.md were set on these. They are prescribed-motion results, not records.
- Reproductions of committed Stage 1 and 1b episodes from unchanged code, to check that this container reproduces
  the first session's simulations: 160 of Stage 1b's 10 ms turn episodes of 60-N-T-10 (R2s/42@10ms and R2@10ms,
  four turn conditions, 20 seeds each), each trace byte-identical to the sha256 in its committed episode record,
  and four of Stage 1's 8 Nov episodes of 75-M-H-10, also byte for byte. These add no result.

## The rules

**S2-9. Delays come in whole physics steps.** The plant's servo delay is k physics steps of 2 ms (the env's and
Stage 1b's delay queue). Every delay of this plan is therefore a multiple of 2 ms. Changes to the plan's wording:
- judged delay: 10 ms (k = 5), unchanged (S2-7; owner decision of 14 Nov, P30);
- "the bar also runs at 0, 5, 15 and 20 ms" becomes 0, 4, 6, 14, 16 and 20 ms: each odd value is replaced by its two
  neighbours, and nothing is read between delays;
- the ladders' "also at 0 and 5 ms" becomes 0, 4 and 6 ms;
- the G8b fallback's "bar at 0, 5 and 10 ms" becomes 0, 4, 6 and 10 ms;
- training's randomised delay "0-25 ms" becomes k uniform on the integers 0 to 12 (0-24 ms);
- R2L's mix keeps Stage 1b's k uniform on 0 to 5 (0-10 ms), with the fixed draws of S2-18;
- the 75 cm track reads 0, 2, 4, 6, 8 and 10 ms (STAGE2_75_PLAN.md).
No sub-step delay model is built.

**S2-10. S2-4's reading point.** S2-4 trains R5 for 2,000,000 env steps (rounded up to a whole number of PPO
rollouts; the count reached is recorded) on 75-M-H-10 with the current head, seed 42, with training's delay draw.
The plan's 2.5M-step monitor is never reached, so its reading is: section 6's monitor (the nominal and 0.40 m/s
walks, the 90 degree turn to the left, the stop, and the forward and left nudges; 3 episodes each at 10 ms; 18
episodes) run on the run's final checkpoint and on R0L, on the same 18 episodes (the monitor's seeds are fixed in
code before the run and recorded). A gain is section 6's: at least 2 more of the 18 episodes succeeding than R0L,
or a mean survival at least 1 s longer. The monitor at each earlier kept checkpoint is reported and decides
nothing. A gain lets Colab start; no gain goes to the owner first. This is a go or no-go reading, not a verdict
(LESSONS_LEARNED.md section 3, lesson 15): verdicts come from the 20-episode panels.

**S2-11. G9's toe-clearance evidence.** The plan's G9 reads the own-swing toe clearance on "Stage 1b's 10 ms turn
traces"; a fresh clone has none, and a stored 50 Hz trace lacks the 500 Hz contacts the reading needs. Amended:
(a) Stage 1b's 10 ms turn episodes of 60-N-T-10 are re-run with the stock rows; a re-run episode counts only if
its trace is byte-identical to the sha256 in its committed episode record (so it is Stage 1b's episode); the
own-swing reading is taken on the live recorder and reported beside the reference-clock reading; only the trace
identity is asserted. (b) For 75-M-H-10, where Stage 1b ran nothing, the evidence is the bar's own 10 ms records,
which hold both readings for every episode. (c) Every frozen reference is played kinematically through the same
readings and must pass its own bar item with margin (the plan's wording, unchanged).

**S2-12. How checkpoints trained on Colab are judged here.** The panels run here, in the study's venv
(tools/requirements-study.txt). A checkpoint trained under other library versions is judged under a flagged
cross-version rule:
- every kept checkpoint is saved with a policy fingerprint: 256 raw observation vectors, drawn once from a seeded
  generator and stored in the run, with the deterministic actions that the saved normalisation statistics and
  policy give them at save time (float32);
- here the checkpoint is loaded through the loader of S2-14 and must reproduce every fingerprint action within
  1e-5 (absolute); a checkpoint that does not is not judged, and the owner is told;
- every policy-row record then holds both version sets (trained under, judged under), the largest fingerprint
  difference and `cross_version: true`;
- torch's build label ("+cpu", "+cu...") is dropped on both sides before versions are compared; the version
  number is compared.
The tolerance is set now, before any Colab archive exists.

**S2-13. Resuming a run across library versions.** A resume under library versions other than the run's last
session's is refused unless `--allow-version-change` is given. With the flag, the session records both version
sets and `version_change: true`, every later record of the run carries the flag, and the checkpoint resumed from
must first reproduce its fingerprint (S2-12) under the new versions. The same holds for a changed source hash
(`--allow-source-change`, as Stage 1). One REPO_REF and one run path are kept for a run's whole life.

**S2-14. Archives that load across Python versions.** The learning-rate schedule (and the clip range) saved in a
checkpoint is an instance of a module-level class, not a closure, so that the archive holds no bytecode; every
load goes through one loader that passes the schedule objects explicitly. Stage 1's judged checkpoints, which hold
a closure, are not touched and stay loadable only in Stage 1's PPO venv. The day-one smoke loads a Colab-saved
archive here.

**S2-15. Gate L1 (owner, 14 Nov, P30).** The frozen look reference is V6 (every look feature) on both bodies. A
look feature may cost a speed rung: on 75-M-H-10 the V6 rung at the animal pace is over the knee's Stage 0 speed
limit (1.012) and is not in the ladder. The duty band is 0.55-0.72 below 0.30 m/s and 0.55-0.65 from 0.30 m/s up.
The owner's pick decided L1; the scores of 3-5 viewers new to the project are added later as a labelled late
reading (a change to A.5's order, taken by the owner before any Stage 2 run). The tuned walk with the
counter-phased tail and the duty and lift margin (set F25F9cl of STAGE2_75_PLAN.md) is a labelled candidate
reference for the 75 cm body, shown to the owner beside V6 once it has records; until the owner decides otherwise
the look reference for training is V6.

**S2-16. Commanded moves are causal, and the bar's times are simulated seconds since the command.** For a stop and
a turn the reference is the periodic walk, frame for frame, up to the command instant; nothing before it knows of
the command; the command is given at a lift-off. This is stricter than Stage 1's convention (decision 18: command
= the start of the reference's own speed ramp, before which the reference's centre of mass has already left the
walk by about half a second); decision 18's reading is kept in every stop record. "2.0 s after the command", "3 s
of walking after" and "the last 10 s" are simulated seconds; the track is played 5 % fast, so the reference is
read at 1.05 times that. The rules of STAGE2_75_PLAN.md sections A (references), B (episodes, rows, conditions),
C (clocks and readings), H (verdicts) and R (records) apply to every body, head and reference set of Stage 2, the
60 cm body and the look sets included, unless a later labelled amendment says otherwise. In particular: the quick
turn's stride rule and its tail and neck amplitude rule (a look choice: inside a turn's quicker steps the tail's
and the neck's swing amplitudes scale with the square of the step period), the two-touchdown stop, the own-swing
toe clearance, the walk's 4.0 m and 0.25 m readings, the turn's heading readings, and the three readings (shown
with the look, shown, shown but for A1) carried by every episode.

**S2-17. Code states.** State A is the commit at which the 75 cm sets are frozen: the legacy check (Stage 1's 36
transitions regenerated bit for bit by the new builder) and the closed-loop regression record both pass at it, and
the one Stage 1 edit of this round (an optional key read by `control.Track.add_periodic`, default path unchanged)
and the Jetson body's code are already in. From state A until the 75 cm panels are done no module that enters a
record key changes. R5, R6 and R2L are built in `designlab/stage2/` by subclassing or wrapping; if a module in a
key must change after state A, the affected baseline rows are run again into a new directory and both are kept.

**S2-18. Fixed delay draws in R2L's fit.** (LESSONS_LEARNED.md section 3, lesson 17.) In R2L's fit every episode
kind runs at each delay of its mix in a fixed rotation over the generations (not a draw from the seed alone), every
member of a generation runs the same episodes, and the selection (speed G5's kind) reads its validation episodes
per delay, with the 10 ms cell decisive and the others reported. The rotation table is written into the fit's
record before the first generation.
