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
