# Round C (training tooling): the lead's rulings of 16 Nov 2026 on points handed up by builders, reviewers and fixers

Working paper for the builders' briefs. Nothing here is a result.

**The worker pool (CV; merged at `0c9dd1d`).** The fixer's readings stand: an env's exception in `reset` or `step`
ends the pool at every worker count, 0 included; after a failed `env_method` or `set_attr` the pool goes on. For the
trainer (CT): an exception out of `reset` or `step` ends the pool, so close it and build a new one; do not carry on
after a failed `env_method` that changes env state; no `os.fork` and no fork-start multiprocessing while a pool is
open; clamp the worker count to the number of envs.

**The bundle tool (CB; merged at `5e11171` and six cherry-picked commits).**
1. `stage2/tb/.gitignore` loses its line `checkpoints/`: the records `checkpoint_<n>.json` are committed, as the
   design's layout and the notebook's brief want. The binaries stay ignored by their own lines. Decision D22's
   "14 lines" is 13 from now.
2. The builder's merge commit `5aaf021` lacked the two attribution lines. It is kept out of the working branch: the
   package came in as its build commits and cherry-picked fix commits.
3. `verify --json` and `unpack --json` join the command line. The notebook (CN) reads `bundle_id` and the 64-hex
   `content_sha256` from `verify --json` and compares them with the quoted ones.
4. `build` refuses when its root is the study itself and git cannot tell whether the records are committed.
5. `manifest_problems` also cross-checks the manifest's own tables (bodies, models, sets) against its files.
6. The fixer's four readings stand.

**The policy files (CP; ruling 12 of `TRACKB_RULINGS3_16NOV.md`).** `actions_at_bound` joins the fingerprint
record's keys, with its assertion.

**The training env (CE).** The eight minor points are being fixed (`wt/wpc-ce`). What a resumed run compares: the
config's named freeze, the input files and the task names, not the env's whole description; the env configs store
the resolved task names.

**The trainer (CT), added to its brief `WPC_CT.md`.** The four sentences on the pool above. The resume comparison
above. The servo model of `stage2/design/servo_model/` is not part of this package: the trainer must only leave the
env's config open to further named options without a change to its own files.

**The look sets (L), added to `WP_L.md`.** Rule L10 as settled on 16 Nov (`TRACKB_RULINGS3_16NOV.md` ruling 3,
`TRACKB_RULINGS4_16NOV.md` ruling 6): a stance is flat when it ends after the move's instant and no later than the
end of the move's last ramp; before the instant a change's foot pitch follows the law of the walk it leaves. The
turn in place has flat feet as its own law (ruling 2), its start and stop included. `move.command_phase` must hold
the instant for `start` and `nominal_to_v0.4` too. A failed `start` or `nominal_to_v0.4` stops that set (ruling 4).
No look set is frozen by the package: freezing is the lead's step after the review.

## Added 16 Nov, 15:40 UTC, after the env's small fix round was merged (`7b4df13`, `67cb6f5`)

**The training env (CE).**
1. The builder's merge commit `cdc4762` lacked the two attribution lines. As for the bundle tool, it stays out of
   the working branch: the round came in as a merge of its last build commit and one cherry-picked commit.
2. What a resume compares keeps `tasks` among its keys, "as built": the resolved task table of rule K4. The task
   weights in force are run state (rule K5; rule M4's fix changes them) and are not part of that comparison.
3. `env_configs` names the tasks the records hold today and does not refuse when a later manifest's table differs
   (rule T2 allows an absent task for a new run, and `absent` lists it with its reason).
4. `REWARD_RULE` says in one place that a playback scores "progress below 1e-4" and in another that the speed
   ripple "charges nothing". The two are aligned with the next change to `r5core.py`, not before, because the
   string is part of that file's hash.
5. Owed by the lead: the design's section 2.2 gains `env_configs(..., root=)`, `from_plan(..., heading=)` and the
   new description keys at the next edit of `ROUND_C_DESIGN.md`.

**For the briefs.** `WPC_CT.md` and `WP_L.md` now point to this file at their top. The Track B draft's integration
of the reward wording must read `stage2/design/CE_RULE_WORDING.md` as merged here (the third round changed its
entries 3a, 3b and 8 and withdrew entry 9).

## Added 16 Nov, 16:28 UTC, after the follow-ups were reviewed and brought in

**The follow-ups on the bundle tool and the policy files** came in as four cherry-picked commits (the branch's
history still holds the builder's merge commit without the two lines). Their review: the code can go in as it
stands (`checkpoint/reviews/roundC__{build,review}_followups.md`).
1. The builder's three extras in the manifest's cross-check stay: a content key outside the six is a problem; a
   part with no file or with two owners is a problem; a set's files must fill its parts without a gap.
2. At the next change to `bundle.py`: the shapes of a file entry and a part entry are made exact; three test gaps
   the review named are closed (a set's file that is no `.npz`; only sound file entries reach the table check;
   one branch that cannot be reached is replaced by a plain assignment).
3. Owed by the lead in `ROUND_C_DESIGN.md`: the ignore lines lose `checkpoints/` (13 lines, decision D22); the
   fingerprint record's keys gain `actions_at_bound`; section 2.2 gains the env's new arguments and keys.

## Added 16 Nov, 18:21 UTC, after the look reading (CK) was reviewed, fixed and brought in

**The look reading (CK)** came in as nine cherry-picked commits (two merge commits of the earlier builder lack the
two lines and stay out). Review: merge as it stands, no blocker or major defect in the code; the fixer closed its
minor points (`checkpoint/reviews/roundC__{build,review,fix}_CK.md`). On the working branch its tests give
"60 passed, 1 skipped".
1. The playback's lead of 0.1 s before the window (`LEAD_S`) stays. No rule gives the number; any lead of one
   physics step or more gives the same events on clean support flags.
2. The rule texts stay as marked: the plan's rule first, then the module's own reading after its mark.
3. `on_command_at_stop` belongs to the rows package (CW); its brief says so.
4. **The tail's amplitude gets a band** (a rule decision, for the Track B draft's look rules, before code state B):
   the tail term fails when the robot's fitted amplitude is under 0.02 rad while the reference's is read. A tail
   that does not swing cannot pass a reading of how it swings. To be built in `lookread.py` with a test.
5. The bob has no "not read" threshold: outside the look sets the rows package records the numbers only.
6. Contact is read literally, as built; the rows package records the short air runs.
7. The gates package (CG) lists each item's reference toe-off and its side of 0.02 rad.
8. Owed by the lead in `ROUND_C_DESIGN.md`: section 3.7's toe-off sentence (the rule is 0.02 rad and 10 %) and
   section 3.5's "3 s of walking".
9. For CW and CG: `played` takes `lead_s` and returns `window` and `lead_s`; `window` reads the fall itself;
   `series` refuses a pose past the last load. A bar walk or turn item that cannot be played is a fault of the
   definition, never "no look reading".

## Added 16 Nov, 19:01 UTC, after the trainer (CT) was reviewed, fixed and merged (`e7f5386`)

**The trainer (CT).** Review: merge after two major fixes in the probes; both are fixed with the eight smaller
points (`checkpoint/reviews/roundC__{build,review,fix}_CT.md`). On the merged tree its tests give "68 passed,
3 skipped".
1. **A monitor that refuses to run is no attempt** (a rule decision, for M1): the monitor leaves with its own exit
   code when it refuses (moved sources, a checkpoint that is not intact, another run's checkpoint); the trainer
   halts with exit 5; the attempt does not count toward M1's two. Owed as a small change to the trainer's files and
   one sentence in M1.
2. The fixer's readings stand: rule O3's probe is refused without the filter record on any set but S1; a later
   probe call with other settings is refused, not merged; a policy row's table without its clock times is refused;
   a lost baseline is made again and compared, and the run halts if it differs; rule K8's probe is made again when
   a fine-tune is cut before its first rollout.
3. For the rows package (CW): every policy episode that ends read, fell or unread returns its clock times and
   `planned_s`.
4. For the notebook (CN): `progress.json` has three new keys (`checks_without_record`, `halt`,
   `last_monitor.counted_without_record`); exit codes 4 and 5 as before; copy `monitor/` with `run.json`.
5. For the gates (CG): a run that is not tiny needs `filter.json`'s choice or the config's `filter_hz`, as it
   needs a rate.
6. The trainer's processes set `GIT_OPTIONAL_LOCKS=0`, so a killed job leaves no `index.lock`. Whether
   `s2io._git` should do the same for every tool is decided when that file is next changed; it is not in this
   package.
7. Still open from the builder's list, to be ruled with the gates package's brief: `--expect-baseline` for rule
   M3, the S1 stand-in in the monitor, no timeout on the monitor child, "R0L exactly" to rounding.

## Added 16 Nov, 19:56 UTC, after two code follow-ups and the fifth list's integration were brought in

- **The monitor's refusal** (the trainer's files) and **the tail's amplitude band** (the look reading) are in, as
  three cherry-picked commits. Their review: bring in as they stand, no defect in the look reading, three minor
  points on the trainer for its next change (`checkpoint/reviews/roundC__{build,review}_followups2.md`). On the
  merged tree the trainer's, the monitor's and the look reading's tests give "141 passed, 4 skipped".
- At the next change to the trainer's files: the halt message after a refusal under `--retry-check` names that
  parameter; one test kills the trainer around a refusal; a checkpoint record of the wrong shape, a missing
  `run.json` and a fingerprint not reproduced at the row's load are refusals too (`TRACKB_RULINGS6_16NOV.md`,
  ruling 3).
- For the gates package (CG): a fifth degraded playback, the tail scaled to a tenth about its mean, which must
  fail the tail term and no other (`TRACKB_RULINGS6_16NOV.md`, ruling 2).
