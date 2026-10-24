# T. rex stance evidence (2026-10-12 to 2026-10-17)

Evidence for [TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](../TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md),
frozen with it on 2026-10-17. These are copies of the files that decided or
document each verdict, taken from the session's scratch tree before it is lost.
The scripts are frozen copies as they ran: research tools, imported by nothing,
outside the library, its tests and CI (CI's ruff, mypy and pytest cover
`environments/` only). Checkpoints, rollouts, traces and per-episode judge files
are not here; where they are, and their sha256, is in the note's §14 and in
`r10_screen/fetch_record.json`.

Seven folders were added on 2026-10-18 with the evidence of the follow-ups of
2026-10-17, which the note predates: walking pilot 1's judging
(`pilot1_judging/`), pilot 2's pre-registration (`pilot2/`), the offline
six-heading certificates (`offline_certificates/`), the offline battery
(`battery/`), the foot and ankle audit (`audit/`), the literature report
(`literature/`) and the recommendation on what to log and sense next
(`next_signals/`). Every one of
them is offline and report only: the certificates were rolled on uncertified
scratch-recipe checkpoints that the judge would refuse on task, the pilot's
verdict is its fixed criteria's and no gate's, and nothing here certifies a
run, writes a result bundle or promotes anything. The judgements among them
are rows of the repository's ledger,
[`results/offline_judgements.csv`](../../../results/README.md#offline-judgements-offline_judgementscsv)
(28 rows, each pointing at its file here).

One more folder was added later on 2026-10-18: `stance3/`, the
pre-registration of the three full T. rex stance screen runs of decision
D-D37 (the note's §16): their fixed criteria, Colab steps, the recommendation
that designed them from the offline diagnosis of 2026-10-18, and frozen copies
of their judge and scorer. It holds no judgement: the runs were launched on
2026-10-18 and are not yet judged, and as screens they certify nothing.

*Added 2026-10-19:* `stance3/results/` holds the three runs' judgement by
their fixed criteria (all three fail; the decision table's line is 3b), the
report-only readouts and the overnight scratch experiment that followed, and
an index of its own, [stance3/results/README.md](stance3/results/README.md)
(the note's §17). As screens they certify nothing.

*Added 2026-10-19, later:* `stance3/night/` holds the overnight
experiment's harness-matched no-change comparison (the note's §17.8), and
`stance4/` the criteria of the next screen, decision D-D39 (the note's
§18), fixed before training. `stance4/` holds no judgement, and as a screen
it certifies nothing.

*Added 2026-10-20:* `stance4/results/` holds the stance4 screen's judgement
by its fixed criteria (all three runs fail; the decision table's line is 5,
final; the note's §19), with report-only readouts and an index of its own,
[stance4/results/README.md](stance4/results/README.md). `stance4/night/`
holds the replicate study of the stagger penalty and `stance4/night2/` the
CoP-band continuation test, both overnight scratch experiments of
2026-10-20 and report only, and `stance4/copband/` the design note of the
per-foot CoP band prototype, the basis of decision D-D40 (the note's §20).
As a screen and as scratch experiments they certify nothing.

*Added 2026-10-22:* `stance5/` holds the pre-registration of the stance5 screen
of decision D-D40 (the note's §20): its criteria, fixed before training, their
notes, the Colab steps, and frozen copies of the judge and scorer.
`stance5/results/` holds the screen's judgement by those criteria (Run 1, seed
52, holds; Runs 2 and 3 do not; the decision table's line is 3, final; the
note's §21), with report-only readouts and an index of its own,
[stance5/results/README.md](stance5/results/README.md). As a screen it certifies
nothing.

*Added 2026-10-22, later:* `stance4/satdiag/` holds the offline, report-only
diagnosis of stance4 s42's actuator saturation that decision D-D40 5a asked for
(the note's §21.5): the report and the five tables it cites. It changes no
verdict.

*Added 2026-10-22, later:* `pilot2/` gains walking pilot 2's criteria and Colab
steps as amended on 2026-10-21, beside the copies of 2026-10-18, which are left
as they were, and the summary of the rebuild for its new parent (decision D-D36
as amended on 2026-10-20; the note's §21.6). Pilot 2 is not yet trained, and
nothing here is a result.

*Added 2026-10-24:* `stance5/replication/` holds the pre-registration of the
stance5 replication of decision D-D41 (the note's §22): its criteria, fixed
before training, the Colab steps, the joint-noise sigma sidecar for the new
seeds, and frozen copies of the judge and scorer.
`stance5/replication/results/` holds the replication's judgement by those
criteria (Run 4, seed 70, holds; Run 5, seed 80, does not; the decision table's
line is 2, final; the note's §23), with report-only readouts and an index of its
own, [stance5/replication/results/README.md](stance5/replication/results/README.md).
As a screen it certifies nothing.

## What changed in the copies

[SOURCES.txt](SOURCES.txt) lists every file with the scratch file it came from,
that file's sha256, and what the copy changed. Three kinds of change, nothing
else:

- **prefix:** the absolute path of the session's scratch directory is replaced
  by `scratchpad` (and the session directory above it by `session`). Paths in
  these files such as `scratchpad/foot/...` or `foot/...` name that tree; only
  the files listed here survive. Scripts with a `SCRATCH` constant
  (`judge_r10.py`, `score_r10.py`) now resolve it relative to the working
  directory.
- **whitespace:** trailing whitespace removed and a final newline added, as the
  repository's pre-commit hooks require; JSON content is unchanged.
- **trailer:** `symmetry/frozen_params.json` held the scratch clone's two
  commit-trailer lines (`repo.commit_trailer`); they are replaced by a
  placeholder. Its sha256 is therefore not the `58b2c8fb…` that READY-v1
  recorded before the symmetry screen trained (the original's sha256 is in
  SOURCES.txt). `judge_r10.py` and `score_r10.py` assert that original value,
  and `verify_final.py` and `recount_final.py` the value READY-v1 passes them,
  so against this copy they stop with a sha256 mismatch; every threshold they
  read is unchanged (`decide_final.py` and `test_decide_final.py` run on it as
  it is).

*Added 2026-10-18:* the copies of 2026-10-18 add two more kinds of
change, nothing else:

- **links:** `literature/trex_foot_and_standing_balance.md` linked four times to
  files of the session's checkout (`/home/user/mesozoic-labs/...`); the links
  now lead to the same files relative to the repository root.
- **gzip:** the ten largest JSON files copied (pilot 1's nine probe outputs, about
  0.7 MB each, and the battery's `push_summary.json`, 0.65 MB) are kept as
  `<name>.gz` (Python's `gzip.compress`, level 9, mtime 0) after the other
  changes; `gunzip` gives the changed copy back.

No copy of 2026-10-18 needed a redaction: none held a model identifier, an
email address or a commit trailer. The folders added on 2026-10-18 rename the
scratch's `logs/` directories `run_logs/` or `judge_logs/`, because the
repository's `.gitignore` ignores every directory named `logs`. SOURCES.txt
ends with the per-episode certificate files that were not copied (about 1.8 MB
and 0.76 MB each), with their sha256 and size.

*Added 2026-10-18 (`stance3/`):* `stance3/criteria.txt` and
`stance3/colab_steps.txt` are byte-identical to the files the maintainer
received (the criteria's sha256 `1ce910dc…` is the one recorded before
training, and the file has no final newline, which is kept so that its hash
stays that one; the repository's pre-commit `end-of-file-fixer` would add one,
so leave it out of a manual pre-commit run). `stance3/recommendation.md` and
the judge and scorer carry the **prefix** change only; as in `judge_r10.py`,
their `SCRATCH` constant now resolves `scratchpad` relative to the working
directory. Nothing else differs, and none of the six held a model
identifier, an email address or a commit trailer.

*Added 2026-10-19 (`stance3/results/`):* the copies of 2026-10-19 add four
more kinds of change, nothing else: **excerpt** (one section of the overnight
design, verbatim), **trim** (the six-heading certificates' `run_summary.json`
cut to their verdict fields), **extract** (one key of the night analysis
workflow's output, written on its own) and **redact** (the session's
workflow run ids and the report document's id removed from the run log);
SOURCES.txt defines them, and
[stance3/results/README.md](stance3/results/README.md) says which file each
applies to. A source outside the scratch root is written below `session`. No
copy of 2026-10-19 held a model identifier, an email address or a commit
trailer.

*Added 2026-10-19, later (`stance3/night/`):* its two copies carry the
**prefix** change, and `nochange_compare.md` the **whitespace** change too
(one line of trailing spaces); nothing else. `stance3/night/interpretation.md`
has the same source as `stance3/results/night_scratch/interpretation.md` and
is byte-identical to it.

*Added 2026-10-19, later (`stance4/`):* `stance4/criteria.txt` is
byte-identical to the file fixed before training (sha256 `978a93da…`; it
ends with a newline). It holds no model identifier, email address or commit
trailer.

*Added 2026-10-20 (`stance4/results/`, `stance4/night/`, `stance4/night2/`,
`stance4/copband/`):* the copies carry the **prefix**, **whitespace** (here a
final newline added to three JSON files and a blank last line removed from
`stance4/results/split_form.md`), **trim** (the notebook's gate verdicts cut to
their verdict fields) and **redact** changes, and one more kind, nothing else:
**checkout**, the session's checkout directory (an absolute path below the home
directory) replaced by `checkout`, once each in `stance4/night/spec.md`,
`stance4/night2/spec.md` and `stance4/copband/design.md`. The redaction is in
`stance4/results/run_log.txt` only: 6 workflow run ids (3 distinct), one
scheduled-task id and the report document's artifact link, each now
`[id removed]`, and the assistant's name in a GitHub check's name, now
`[name removed]`. SOURCES.txt defines them and
[stance4/results/README.md](stance4/results/README.md) says which file each
applies to. No copy holds a model identifier, an email address or a commit
trailer. `stance4/copband/design.md` is kept as written: what later decisions
supersede in it is in its entry under Contents, not in the file.

*Added 2026-10-22 (`stance5/`, `stance5/results/`):* the copies carry the
**prefix**, **whitespace** (here a final newline added to nine JSON files),
**trim** (the notebook's gate verdicts cut as stance4's, the certificates'
summaries as stance3's), **redact** and **checkout** changes, and one more kind,
nothing else: **lines**, a line range of the source, the rest left out:
`stance5/results/run_log.txt` is lines 208 to 395 of the session's run log,
whose first 207 lines `stance4/results/run_log.txt` copies. The redaction is in
that log only: 30 workflow run ids (18 distinct), 2 scheduled-task ids and the
report document's artifact link, each now `[id removed]`; the assistant's name
in a GitHub check's name, now `[name removed]`; and, an item not redacted
before, 9 shortened Drive folder ids (seven or eight characters and `...`), each
now `[id removed]`. The **checkout** change is in that log and in
`stance5/judge/judge_stance5.py`. `stance5/criteria.txt` is byte-identical to
the file fixed before training (sha256 `3b28963f…`; it ends with a newline), and
so are `stance5/criteria_notes.md`, `stance5/colab_steps.txt` and
`stance5/judge/test_score_stance5.py`. The judge and scorer carry the **prefix**
change (the judge the **checkout** change too): as in `stance3/judge/`, their
`SCRATCH` constant now resolves `scratchpad` relative to the working directory,
and their sha256 is not the one the judge files and the Colab steps record.
SOURCES.txt defines the changes and
[stance5/results/README.md](stance5/results/README.md) says which file each
applies to. No copy holds a model identifier, an email address or a commit
trailer.

*Added 2026-10-22, later (`stance4/satdiag/`):* `report.md` carries the
**prefix** change (one line) and `rollout/tables/onset_within_episode.txt` the
**whitespace** change (a blank last line removed); the other four are
byte-identical. Nothing else differs, and no copy holds a model identifier, an
email address or a commit trailer.

*Added 2026-10-22, later (`pilot2/`):* `pilot2/criteria_2026-10-21.txt` and
`pilot2/rebuild_1021_summary.md` carry the **redact** change, one workflow run
id each, now `[id removed]`, so the criteria copy's sha256 is not its source's
`bdcdae4b…` (SOURCES.txt gives it); `pilot2/colab_steps_2026-10-21.txt` is
byte-identical to its source (`60845fdc…`). `pilot2/criteria.txt` and
`pilot2/colab_steps.txt` stay the texts of 2026-10-18 (`a240ff5d…` and
`bc94bda3…`); the amended criteria cite the first as their pre-amendment copy.
No copy holds a model identifier, an email address or a commit trailer.

*Added 2026-10-24 (`stance5/replication/`, `stance5/replication/results/`):* the
copies carry the **prefix**, **whitespace**, **gzip**, **trim**, **redact**,
**checkout** and **lines** changes, nothing else and no new kind:
`stance5/replication/results/run_log.txt` is lines 396 to 504 of the session's
run log, whose lines 1 to 207 and 208 to 395 are the stance4 and stance5 copies.
The redaction is in that log only: 49 workflow run ids (28 distinct), and, items
not redacted before, 15 background-job ids of the session's queue workers (13
distinct) and 6 shortened Drive folder ids, each now `[id removed]`.
`stance5/replication/results/scores/score_final.json.gz` is gzipped, as the
JSON is above the repository's 500 KB limit for an added file.
`stance5/replication/criteria.txt` is byte-identical to the file fixed before
training (sha256 `a38536de…`; it ends with a newline), and so are
`stance5/replication/colab_steps.txt` (`d646fc3c…`, the steps sent at 18:00 UTC
on 2026-10-22) and `stance5/replication/judge/test_score_stance5r.py`. The
judge, the scorer and the sidecar `sigma_s70_s80.json` carry the **prefix**
change (the judge the **checkout** change too), so their sha256 is not the
`16d39f7d…`, `55c37ba7…` and `89778b3f…` the criteria, the Colab steps and the
judge files record; SOURCES.txt gives those as the sources' sha256.
[stance5/replication/results/README.md](stance5/replication/results/README.md)
says which file each change applies to. No copy holds a model identifier, an
email address or a commit trailer.

[SHA256SUMS](SHA256SUMS) pins every file here but this README and itself
(`sha256sum -c SHA256SUMS` in this directory). *Added 2026-10-19:* nor
`stance3/results/README.md`, which, like this README, has no source.
*Added 2026-10-20:* nor `stance4/results/README.md`, likewise.
*Added 2026-10-22:* nor `stance5/results/README.md`, likewise.
*Added 2026-10-24:* nor `stance5/replication/results/README.md`, likewise.

## Contents

**`run_a/`: run A and the 2026-10-12 foot and CoP investigation (§3).**

- `report.md`: the corrected report (mechanism, reward, per-species anatomy,
  probes, recommendation, open questions, 11 corrections). It cites further
  scratch files under `foot/`; the ones that back the note are copied beside
  it.
- `stance_gate_report_20261011_165957.txt`: run A's `stance_quality/v2` report
  from Drive (1/40 clean).
- `ablate_table.txt` (held-pose ablation of run A's mean action),
  `thresholds.txt` (ramped single-ankle thresholds per model),
  `an_moment.txt` (the verifier's ankle-moment balance).
- `reward_pairs.txt` (per-term paired returns), `observability.txt` (CoP fits
  from observation channels).
- `noise_saturation.txt`, `noise_s48.txt`, `noise_s56.txt`: the statue and the
  trained policies under the policies' own noise.
- `probe0_analysis.txt`, `probe1_results.txt`, `probe2_results.txt`: the three
  1M probes on `main` `8565d299`. Read probe 2 with the report's Corrections 8
  (its episode-level p-values are not evidence).
- `anatomy_summary.txt`, `statue_per_species.txt`: statue panels per species.

**`noise_rank/`: the offline noise-ranking check (§4.1).** `summary.txt` is the
corrected verdict (it supersedes a first version, not copied);
`pass_corrected.txt` the corrected pass table; `verify_summary.txt` the
independent re-derivation that withdrew the "does not reward falling early"
claim.

**`cap_screens/`: the screens before symmetry and the roll diagnosis (§4).**

- `screen_1m_analysis.txt`, `screen_3m_noise_analysis.txt`,
  `screen_3m_cop_authority_analysis.txt`, `screen_6m_authority_analysis.txt`,
  `screen_hip_roll_analysis.txt`: each screen's analysis with its
  predeclared criteria. In `screen_3m_noise_analysis.txt` the matched-sigma
  noise comparison ("about 8x ... 14x") and the proposed 12M entry rule built
  on it are withdrawn; `control_s{42,52,60}_uniform_sigma016.json` are the
  control under the same uniform σ 0.16 (9 + 11 + 17 = 37 of 60 falls).
- `roll_diag_analysis.txt`: the lateral pad-roll diagnosis.
  `roll_verify_reward_terms.json` is the independent check's per-term returns;
  it corrects the analysis's "the reward does not price the lean" and its
  option (b) range (net spread ±32).

**`symmetry/`: the symmetry screen, arms L and C, verdict O9 (§5).**

- `frozen_params.json`: every frozen parameter, threshold, seed and checkpoint
  (redacted as above).
- `interim.json`, `verdict.json`: the 3M interim and the 6M verdict;
  `J1_recount.log`, `J2_recount.log`: the independent recount (120 and 537
  quantities, no disagreement); `rerun_ledger.json`: the two infrastructure
  re-runs.
- `verdict_scripts/`: `verify_final.py`, `recount_final.py`,
  `decide_final.py`, `arbiter_final.py` and `test_decide_final.py`,
  byte-identical to the versions whose sha256 `frozen_params.json` records.
- `mirror_ratio.json` (why the KL-form mirror loss, not augmentation),
  `lean_calibration.json` (the lateral lean term's weight and width),
  `ema_credit.json` (PPO's credit for an EMA reward term),
  `groundwork_verify_summary.json` (independent check of the mirror map,
  equivariance, π_sym and lean calibration, with its discrepancies list).

**`r10_screen/`: the r10 foot and arm L screen (§6).**

- `criteria.txt`: K1-K4, written before training, with the dated tie-break
  clarification. `colab_steps.txt`: how the three Colab runs were set up.
- `score.txt`, `score.json`: the scored verdict beside r9 arm L, the best
  stance by the fixed rule and the report-only block. Two reading notes: the
  best-stance listing prints joint-space falls only, though the rule sums
  joint-space and own-sigma falls (the pick is the same); and `score.json`'s
  r9 `foot_cop_fh` fields count unmeasured episodes as over the rail and are
  unused (r9's K2 uses its pad counts).
- `judge_r10.py` (judges one checkpoint, binding plant, task, seed and
  commit), `score_r10.py` (computes K1-K4), `test_score_r10.py` (33 boundary
  cases), `validate_r9.json` (the judge reproducing an r9 judge file before
  use).
- `verify_recompute.py`, `verify_recompute.json`: the independent
  recomputation of every cell from the per-episode rows.
- `report_only_summary.json`: the notebook's end-of-stage certificates and the
  6M-12M in-training statistics. `fetch_record.json`: Drive file ids, sizes,
  md5 and sha256 of the 36 files fetched.

**`diagnosis/`: why seeds 52 and 60 stood on their toes (§7).**
`findings_v3.txt` is the current version. `findings_v1_superseded.txt` is kept
because its figures reached the public #623 comment: its noise hypothesis (H1)
is withdrawn, its edge "1.06" and its slopes -192 / -196 are superseded (v3,
section 4 and its header).

**`pilot/`: the walking pilot's pre-registration (§9; launched on 2026-10-17,
not yet judged).**
*Added 2026-10-18: it was judged on 2026-10-17; see `pilot1_judging/` below.*

- `criteria.txt`: the fixed criteria, byte-identical (sha256 `1cdd495c…`).
  Its first line says "revised three times on 2026-10-17"; its Revisions
  list has four (the fourth, wording only), and the copy is left as frozen.
- `pilot_score.py` (applies the rule), `pilot_judge.py` (runs the probe and
  binds the run, including the parent's sha256), `pilot_parent.py` (the
  screen's parent rule), `gait_probe_pilot.py` (the adapter that runs the
  frozen [gait probe](../gait_2026_09/gait_probe.py) on today's API),
  `make_gait_cases.py` and `fixtures/gait_cases.json` (the probe-computed hop
  and walk cases the rule is tested against), `test_pilot_score.py`,
  `test_pilot_parent.py`.
- `duty_cut_summary.json`: the cost table behind the 0.80 duty guard.
  `pilot_parent_on_screen_score.log.txt`: `pilot_parent.py` on the screen's
  score, naming r10 arm L seed 42 6M, the rule's choice that the maintainer
  departed from.
- `pilot_video.py` and the Colab steps stay with the session; the branch
  `claude/pilot-walk-r9` (`eaeef50f`) holds the parent checkpoint and its
  `SOURCE.txt`.

**`telemetry/`.** `posture_drift_replay.json`: the `PostureDrift` default
replayed on the r10 runs' `gate_progress.npz`, on the 20:20 snapshot and on the
full series (two in a row against three).

### Added on 2026-10-18

**`pilot1_judging/`: walking pilot 1's judging (D-D36; judged 2026-10-17,
final).** Outcome **STILL on 3 of 3 seeds** at 4,194,304 steps: each seed
STILL on 40 of 40 panel episodes, no WALK episode, median speed 0.001-0.002 m/s,
no touchdowns, foot duty 1.00/1.00; at 2,097,152 seed 42 read STILL 39 and
FALL 1. 2,097,152 and 3,145,728 are the trajectory and decide nothing, and no
video review was needed (only WALKS needs one). The rule is the fixed criteria
in `pilot/criteria.txt`; the verdict is the criteria's, not a gate's.

- `run_log.txt`: the runs (Drive folder ids, launch times, what each
  `stage_config.json` was checked for), the 2M pipeline check and the final
  judging, with report-only figures: the in-training evaluation reward "flat at
  ~1,097-1,100 throughout" and the forward-reward term "0.5-3.2 at 2-4.2M (max
  7.5)", both of which hold at the 2M, 3M and 4.2M evaluations only (the
  note's §15.1 gives every evaluation's range, from the runs'
  `gate_progress.npz`, which stay on Drive), the policy's action std
  0.108-0.111 at 4.2M against the parent's 0.095, and the notebook's own
  locomotion gate FAIL on speed (recorded, not a pilot criterion). The log is
  kept as written.
- `OUT/score.json`: `pilot_score.py --seeds 42,52,60` on the nine panels (the
  original's sha256 `86d7350e…`, the value `run_log.txt` records; the copy adds
  a final newline).
- `OUT/s<seed>/<steps>/binding.json`: what `pilot_judge.py` bound (tree
  `eaeef50f`, the checkpoint's and VecNormalize sidecar's sha256, the parent
  `sha256:70f3a444…`, the budget, the plant). `probe.json.gz`: the frozen gait
  probe's 40-episode output for that checkpoint (gzipped; `pilot_score.py` reads
  it unzipped). The traces and plots stay with the session; the checkpoints are
  on Drive (folder ids in `run_log.txt`).
- `judge_logs/`: each judge run's log and the scorer's printout.
- `tools/`: `loadcheck.py` (loads every VecNormalize pickle under an allow-list
  of classes and checks each run's `stage_config.json`, `stage_summary.txt` and
  `gate_progress.npz`), `globals.py` (lists the globals a pickle names) and
  `stdcheck.py` (the policy's action std from a checkpoint's `log_std`, the
  source of the run log's std figures).

**`pilot2/`: walking pilot 2's pre-registration (decided 2026-10-17; not yet
trained, no result).** The maintainer chose "Second pilot, wider exploration":
pilot 1 again with one change, every actuator's `log_std` set to -1.5 (std
0.2231, the stance stage's own starting noise) at the warm start.
`criteria.txt` (written 2026-10-17 and amended 2026-10-18, both before any
pilot-2 training step) states the change, its checks and everything kept from
pilot 1; `colab_steps.txt` is how the runs are launched (run IDs
`pilot2_walk_s42/52/60`). The pilot trains from the branch
`claude/pilot-walk-r9-b` (`ba464855`: pilot 1's `eaeef50f` plus three commits),
pushed with no pull request and never merged.

**`offline_certificates/`: the stage's own six-heading certificate rolled
offline on the r9 arm L symmetry-screen checkpoints (2026-10-17, report only).**
`scripts/run_cert.py` calls `build_stance_heading_report`, the function the
notebook's end-of-stage certificate calls, on `main` `2e734e29` (physics r9)
with `stance.toml` as committed (`--options off`) or with arm L's three env
options added (`--options armL`). Eleven certificates:

| Checkpoint (run `sym_lean_settled_s<seed>`) | Panel | Options | Verdict | Seeds clean at all six headings | LCB (bar 0.80) |
|---|---|---|---|---|---|
| s42 6,291,456 | 3042-3081 | off; armL | FAIL; FAIL | 29/40; 29/40 | 0.5861 |
| s52 6,291,456 | 3042-3081 | off; armL | PASS; PASS | 38/40; 38/40 | 0.8508 |
| s60 6,291,456 (the pilots' parent) | 3042-3081 | off; armL | PASS; PASS | 40/40; 40/40 | 0.9278 |
| s42 robust_best (655,360) | 3042-3081 | off | PASS | 40/40 | 0.9278 |
| s52 robust_best (917,504) | 3042-3081 | off | PASS | 39/40 | 0.8868 |
| s60 robust_best (524,288) | 3042-3081 | off | PASS | 40/40 | 0.9278 |
| s52 6,291,456, audit | 7042-7081 | off | PASS | 38/40 | 0.8508 |
| s60 6,291,456, audit | 7042-7081 | off | PASS | 39/40 | 0.8868 |

Source: `offline_certificates.txt` and each folder's `run_summary.json`. **None
is gate evidence.** The runs were trained in the session's scratch harness
(MirrorKLPPO with scratch `experiment_*` keys, not on Drive) and recorded task
`a2af3c5c…`; these certificates roll `bde44225…` (off) or `cd963518…` (armL), so
the judge would refuse them on task, and an audit panel on 7042-7081 is refused
as gate evidence by design (`evidence_table.txt` §E). Both audits sit on the
`max_hop_or_fall_episodes` 1 rail: both checkpoints fall on seed 7047, where the
statue stands.

- `offline_certificates.txt`: every certificate per heading, its unclean seeds
  and the reasons. `evidence_table.txt`: the bar with its `stance.toml` lines,
  every earlier result and these. `cert_margins.txt`: each bar's worst and
  median value over the 240 policy and 240 statue episodes (s60 6M's worst pad
  CoP is 0.780 against the 0.80 bar). `cert_summary.json`: the cross-checks (the
  0° cells of the three 6M options-off certificates reproduce the scratch
  judge's v2 panels; options off and armL give identical floor metrics, so the
  options change only the reward). `rederive_from_csv.json`: each verdict
  re-derived from its panel CSV by the library's own gate (all 11 agree).
- `<certificate>/run_summary.json` (checkpoint and sidecar sha256, verdict,
  counts, LCB, per-heading means) and `stance_heading_report.txt` (the
  certificate as printed). The per-episode `stance_heading_report.json` and
  `stance_heading_panel_selected.csv` stay with the session (sha256 at the end
  of SOURCES.txt).
- `scripts/`: `run_cert.py` and the shell lines that ran it (`run_all.sh`,
  `run_audit.sh`, `run_robust.sh`), `rederive.py`, `analyze.py` (wrote
  `cert_summary.json`), `margins.py` and `tab_certs.py`. `run_logs/`: each
  run's log.

**`battery/`: the offline battery on the same checkpoints (2026-10-17, report
only).** `battery_report.md` is the report; its citation prefixes `B/`, `H/`,
`N/` and `V/` (printed as `scratchpad/foot/...` paths) are `battery/`,
`offline_certificates/`, `next_signals/` and `battery/verify_skeptic/` here.
Its verdict: **s60 6M is a real but thin pass; s52 6M is on the rail.**

- `fresh_seeds/`: eight more certificates (options off) on fresh panels
  5042-5081 and 9042-9081 for s52 and s60 at 6M and at robust_best, by the same
  `run_cert.py` (`scripts/one_cert.sh`). All eight PASS: s60 6M 40/40 and
  39/40, s52 6M 37/40 (the minimum passing count, LCB 0.8174) and 39/40, both
  robust_best checkpoints 40/40 on both. `fresh_seed_summary.txt` and `.json`
  pool them with the 3042 and 7042 panels (exact Clopper-Pearson intervals; a
  pooled rate is not a gate statistic); `offline_certificates.txt`,
  `cert_margins.txt`, `rederive_from_csv.json` (all eight agree) and each
  certificate's `run_summary.json` and `stance_heading_report.txt` as above.
- `shared_fall/`: seed 7047 replayed at six headings for s60 6M, s52 6M and the
  statue (`shared_fall_summary.json`, `shared_fall_report.txt`, the figure
  `shared_fall_7047.png`): the reset leaves the animal tipped forward, the
  statue survives by folding (its hips sag and its heels come down), and both
  policies hold the hips at or above home and topple forward.
- `push/`: `main`'s impulse probe as shipped and a 16-seed copy with a finer
  grid and fore-aft rows (`push_summary.json.gz`, `push_report.txt`,
  `repo_probe/*/stance_gate_probe_impulse.txt`). The weak push direction
  follows where each policy parks its pad CoP, and no policy beats the statue
  in every direction.
- `own_sigma/`: heading-0 v2 panels with actions sampled from each policy's own
  Gaussian beside deterministic re-runs (`own_sigma_summary.json`,
  `own_sigma_report.txt`): no policy falls, the 6M verdicts hold, and every
  robust_best checkpoint is clean only when run deterministically (0 of 40
  clean under its own noise).
- `report/`: `panel_pass_prob.py` and its output (an illustration under an
  i.i.d. assumption, not a gate statistic) and `disk_usage.txt`.
  `verify_skeptic/`: the independent verifier's scripts and outputs; the
  corrections they forced are applied in the report (its last table).

**`audit/`: the T. rex foot, ankle and posture audit (2026-10-17, main
`2e734e29`, physics r9; measurement only).** `foot_ankle_audit.md` is the
report; its evidence paths are relative to `audit/`. It finds nothing in the
anatomy that pushes the quiet stance forward (in the statue and both 6M
policies the COM projects 2.4-6.8 cm ahead of MTP III, inside the zone where
tyrannosaur tracks are deepest); the COM sits 4-18 cm too high relative to the
hip, the digit spread is 3-5× too narrow, the ankle extends 11° past the only
published T. rex range, and the gate's unsigned pad statistic would fail a
track-like stance over the MTP joints. Its seven candidate physics changes each
need the maintainer's approval.

- `geometry/` (`geometry_audit.json`, `geometry_summary.txt`, the scripts),
  `verify/` (the verifier's static, settle and posture measurements and its
  scripts), `posture/` (`posture_table.txt`, `posture_timecourse.json`, the
  tools), `literature/` (`out/`: the published values restated in the model's
  frame, the digitised Brochu 2003 figures and `literature_crosscheck.md`;
  `scripts/` with `published_inputs.json`, the hand-entered values with their
  locators). The downloaded papers the scripts parse (`literature/dl/`), the
  posture rolls and the per-episode posture files stay with the session.

**`literature/`: the literature report of 2026-10-17 and its research notes.**
`trex_foot_and_standing_balance.md` ("Longer toes cannot cure a forward lean")
and `research_notes/`, the four notes it draws on and cites (T. rex foot
anatomy and posture, bird standing balance, standing-balance mechanics and foot
control, RL for standing). Read its model figures with the audit's
"Corrections to the literature report" (`audit/foot_ankle_audit.md`), which
supersede them where they differ.

**`next_signals/`: what to log and sense next (2026-10-17, report only; nothing
trained).** `recommendation.md`: in behaviour a certified stance is close, in
certification it is not (0 of the 2 certifying seeds), and sixteen numbered
items, from report-only telemetry to policy-interface changes, each with its
approval. Its `NS/` citations are the files beside it here (`certify/`,
`diagnose/`, `questions/`, `sensing/`, `sensing_robotics/`), except
`NS/howclose/`, which is `offline_certificates/`. Its code citations are to
`main` `2e734e29`.

### Added on 2026-10-18 (decision D-D37)

**`stance3/`: three full T. rex stance screen runs, pre-registered
(decided 2026-10-18; launched that day on Colab L4 runtimes, not yet judged).**
Run 1 `stance3_armL_s60` (arm L as #624 ships it, seed 60; branch
`claude/screen-trex-r9-arml-12m` at `7e6b557d`), Runs 2 and 3
`stance3_armLT_s60` and `stance3_armLT_s52` (arm L plus
`leg_home_pose_weight` 1.0 and `leg_home_pose_tolerance` 0.05 rad; branch
`claude/screen-trex-r9-armlt` at `1af2d4d7`), each 12M steps on physics r9;
neither branch is ever merged or has a pull request.

- `criteria.txt`: K1-K4 and the decision table, written before any training
  step (sha256 `1ce910dc…`). Its title's "2026-10-19" is the date first
  planned; the runs started on 2026-10-18.
- `colab_steps.txt`: how the runs are launched and resumed, as sent. It names
  a CPU runtime as the maintainer's choice of 2026-10-18; the runs went on
  L4, which the maintainer kept for all three (the note's §16.2).
- `recommendation.md`: the offline diagnosis of 2026-10-18 ("What the offline
  work found"), the three runs, the criteria, an alternative layout, the
  concerns, the decisions it asked for and the build specification. Its `T/`
  paths (`scratchpad/foot/tonight/`) and `SP/` paths name the session's
  scratch, where the diagnosis' tables and scripts stay.
- `judge/judge_stance3.py` (judges one checkpoint against its branch clone,
  binding plant, task, symmetry record and commit), `judge/score_stance3.py`
  (scores K1-K4, the run verdicts and the decision-table line, and the
  report-only readouts) and `judge/test_score_stance3.py` (80 cases at every
  bar).

### Added on 2026-10-19 (decision D-D37's outcome)

**`stance3/results/`: the three stance3 screen runs judged (2026-10-18 to
2026-10-19, final), with report-only readouts and the overnight scratch
experiment.** Outcome: **all three runs fail** their fixed criteria, so none
holds and none holds fore-aft: Run 1 (arm L, seed 60) fails K2 on the K2c
range (33.9 mm against 30) and K3 (joint-space noise falls 7 and 11 of 20);
Run 2 (L+T, seed 60) fails K1 and K2 early (from 4,194,304 to 6,291,456);
Run 3 (L+T, seed 52) fails K2 (K2a 6 at 8,388,608, K2b 4 at the final, K2c
range 30.6 mm). The decision table's line is **3b**. The best final
checkpoint by the fixed rule is Run 2's (40/40 clean, no noise falls). Its
[README](stance3/results/README.md) indexes the folder:

- `run_log.txt` (the judging log, ids removed), `scores/` (the frozen
  scorer's printouts: `score_final2.txt` and `.json` are the verdict of
  record), `checkpoints.csv` (one row per judge file, 37, with the
  `checkpoints_csv.py` that recomputes it with the scorer's own counts) and
  `certificates/` (the stage's six-heading certificate rolled offline on the
  three finals, report only: all three PASS, none gate evidence).
- `readouts/`: report-only readouts of the Colab checkpoints (Run 2's tail
  actuator saturation and left-right foot split at 4,194,304, Run 1's
  ankle-carried drift and falls, Run 3's split at 8,388,608, and the
  whole-body drift in the eval ledgers).
- `night_scratch/`: the overnight scratch experiment of 2026-10-19 (three
  continuations with a candidate term each, from a scratch clone never
  merged; report only): its pre-registered readouts, the readouts of the arms
  and of the matched no-change controls, the interpretation as written, and
  the verifier's verdicts, which correct it.

### Added on 2026-10-19, later (the night continuations' no-change comparison)

**`stance3/night/`: the harness-matched no-change comparison of the overnight
scratch experiment (2026-10-19, report only).** The spec's local no-change
continuation of Run 3 (`night_R3_nochange`, from the Colab 8,388,608
checkpoint with no added term) was judged and rolled at 9,437,184 and
9,961,472 like the arms. It did not recover: at 9,961,472 it was clean on 25
of 40, with K2a 15 and a window stagger sd of 25.9 mm, against the stagger
arm's 40, 0 and 7.5 mm and the Colab no-change run's 40, 0 and 9.7 mm. The
ankle anchor (H_B) and the stagger term on Run 1 (H_C) stay refuted; the
stagger term on Run 3 (H_A) stays inconclusive, since the two no-change runs
differ by as much as the stagger arm and the local one do, with one run per
arm. None of it enters K1–K4 or certifies anything.

- `nochange_compare.md`: the comparison as written (R1 and R2 at both
  checkpoints, a paired seed bootstrap and exact McNemar tests, the eval
  ledgers and the reading).
- `interpretation.md`: the night interpretation it cites by name,
  byte-identical to `stance3/results/night_scratch/interpretation.md`; read
  it with that folder's `verify_result.json`.

Its judge files, rollouts and scripts stay in the session's scratch
(`scratchpad/foot/stance3/night/cont/`). Its paths: `S3` is
`scratchpad/foot/stance3` and `C` is `S3/night/cont`.

### Added on 2026-10-19, later (decision D-D39)

**`stance4/`: the next T. rex stance screen, pre-registered (decided
2026-10-19; not yet trained when recorded).** Three new runs of arm LTS,
the stance3 L+T recipe plus a stance stagger penalty (weight 0.30,
tolerance 0.045 m), from scratch on physics r9: `stance4_LTS_s52`,
`stance4_LTS_s60` and `stance4_LTS_s42`, from the branch
`claude/screen-trex-r9-armlt-stagger` (`1af2d4d7` plus `ff7114c0` and
`c1e146a8`), which is never merged and has no pull request. The control is
stance3 Runs 2 and 3 as already judged, not retrained. The maintainer's
decision named Colab CPU runtimes, as stance3's Colab steps had. Measured,
stance3's runs trained on Colab L4 GPU runtimes (in the `stage_config.json`
files fetched at 4,194,304 steps; that the control stayed on L4 throughout
is inferred). Before any of the three runs started, the criteria were
re-fixed for L4, the control's runtime type, for the maintainer to confirm
when starting the runs.

- `criteria.txt`: what is trained, how it is judged (K1–K4 exactly as
  stance3's), the control and its caveats, the credit rule, the decision
  table, when the decision is read and the pre-registered readouts, written
  before any training step (sha256 `978a93da…`). Its `SP/` paths name the
  session's scratch (`scratchpad/`), where the judge, the scorer, the build
  record and the statue measurement stay. It replaces a first file (sha256
  `8bda933e…`) that named a Colab CPU runtime, as the decision did. The
  control's L4 is measured in the `gpu` block of its `stage_config.json`
  files, as fetched at 4,194,304 steps; each records the session that wrote
  it, so that the control stayed on L4 throughout is inferred. The first
  file was re-fixed before any of the three runs started and differs only
  in the runtime (measured: line 7 changed and two lines added); it stays
  in the session's scratch (`scratchpad/foot/stance4/superseded/`).

### Added on 2026-10-20 (decision D-D39's outcome)

**`stance4/results/`: the stance4 screen judged (2026-10-20, final), with
report-only readouts.** Outcome: **all three runs fail** their fixed
criteria (H = 0, K2p = 0): Run 1 `stance4_LTS_s52` and Run 2
`stance4_LTS_s60` fail K1 and K2, and Run 3 `stance4_LTS_s42` fails K1–K4.
The decision table's line is **5**, final: the stagger penalty is not
adopted. The best stagger final by stance3's rule is Run 1's (39/40 clean, no
noise falls), one episode short of stance3 Run 2's final (40/40), which stays
the walking pilot 2 parent the criteria offer. Measured, all three runs
recorded a Colab L4 GPU runtime (the `gpu` block of each run's
`stage_config.json`), and the maintainer confirmed it. Its
[README](stance4/results/README.md) indexes the folder:

- `run_log.txt` (the stance4 log, ids removed, ending with the maintainer's
  decisions of 2026-10-20), `scores/` (`score_final.txt` and `.json`, the
  verdict of record), `checkpoints.csv` (one row per judge file, 51, with the
  `checkpoints_csv.py` that recomputes it with the frozen scorer's own
  counts), `extras.txt` and `extras.py` (the fresh, tail and common-noise
  panels on each final, report only) and `notebook_gates/` (the notebook's
  own six-heading gate on each run's `robust_best`, trimmed; report only).
- `panels_summary.csv` and `panels_summary.py`: the diagnostic panels
  (mirrored-leg spawns, the prey straight ahead, both) at every J checkpoint
  of the three runs and the two controls, report only.
- `split_form.md`: where the left-right split went, at 4,194,304 (the
  stance4 feet nearly in line there, the per-foot CoP split not; report
  only).

**`stance4/night/`: the replicate study of the stagger penalty (2026-10-20,
scratch only, report only).** Pre-registered continuations from stance3 Run
3's Colab 8,388,608 checkpoint to 9,961,472, on two new training seeds (152
and 252) with the stagger penalty and without, read beside the three runs
of the night before: stagger n = 3 against no change n = 4. Its reading is
line (2), partial, met through a flagged run (the seed-152 stagger run had 2
panel falls and a reward ratio of 0.945; the seed-252 no-change run was
flagged too). The window stagger sd separates completely (7.5, 7.7 and 4.2
mm against 25.9, 9.7, 14.8 and 29.3; one-sided p 1/35 = 0.029, optimistic
because three of the seven runs suggested the idea, and 0.086 given their
known order); K2a does not (0, 3 and 0 against 15, 0, 3 and 8; p 0.200).
None of it enters a stance4 verdict.

- `spec.md`: the pre-registration, frozen at 01:48:58 UTC (its header says
  01:50, written a minute before; `review_handling.md`), before the queue
  launched at 01:49:08 UTC and so before any training step (sha256
  `24af508a…` as frozen; this copy carries the prefix and checkout changes).
- `review_handling.md`: the review's findings and how each was handled
  before launch.
- `readout.md` and `readout.json`: the readouts R1 (the judge) and R2 (the
  split rollout) at 9,961,472, the tests and the reading, with the R2
  reproduction log.

**`stance4/night2/`: the CoP-band continuation test (2026-10-20, scratch
only, report only).** Pre-registered, paired continuations from stance4
s52's and s60's Colab 4,194,304 checkpoints to 5,767,168, with the band
prototype on (CB) and off (NC), one run per arm per parent, from the scratch
commit `f94452ad` (never pushed). Its reading is line (2), partial: CB is
lower than NC on both readouts for both parents (K2a 0 against 18 for s52
and 3 against 22 for s60; CoP split sd 0.29 against 1.20 and 0.11 against
1.27), but the s60 band run raised new safety flags (3 panel falls,
hop-or-fall 3, a reward ratio of 0.928). In that run the guards left 242 of
59,836 window foot-steps uncounted, all in those 3 episodes (a foot unloaded
or under the 35% load share; in 2 of them a foot also on its digits alone);
in the s52 band run, none. The guards cannot see the route the design calls
its main risk (each foot's CoP held in band by asymmetric ankle torque while
the feet stay staggered); in the s52 pair the band run's window stagger sd
was the higher (9.4 against 6.7 mm). A continuation is not a from-scratch
screen, and none of it enters a stance4 verdict.

- `spec.md`: the pre-registration, frozen at 06:29 UTC before any training
  step (sha256 `b1606f74…` as frozen; this copy carries the prefix and
  checkout changes).
- `review_handling.md`: the review's eight findings and how they were
  handled; no frozen file changed.
- `readout2.md` and `readout2.json`: R1 and R2 at 5,767,168, the guard
  readouts, a paired panel-seed bootstrap, the Colab no-change references
  and the diagnostic panels of the s52 pair (the s60 pair's were skipped at
  the deadline).

**`stance4/copband/design.md`: the per-foot CoP band prototype's design note
(2026-10-20, 00:35–01:30 UTC; a proposal, nothing trained), the basis of
decision D-D40. Decisions 3a–3c of 2026-10-20 supersede its sections on the
step guard, the gait import exception and the speed**; the file is kept as
written:

- 3a, the step guard: guard 1's 35% load-share step (§4.1, and the reward
  cliff of §7 item 4) becomes a ramp, the full price at a load share of 0.25
  or less, no guard price at 0.35 or more, linear between;
- 3b, the speed: the measured +0.62 ms per env step (one thread, 1.40 to
  2.02 ms; §6.3, §7 item 6 and §8 item 3) is to come down, with one fused
  per-step reduction byte-identical to the recorder's readings (no numeric
  target was decided);
- 3c, the gait import exception: the contact reduction moves out of
  `environments/shared/gait/` into a neutral shared module, so the gait
  layering rule (D-D23) holds with no exception (§1's file table, §6.1's
  layering-test exception, §7 item 5 and §8 item 2); the recorder
  bit-identity test, which pins the band's readings, moves with it (§8 items
  2 and 3).

Decision 1a takes the note's term (§2 and §4) as an opt-in `[env]` option,
off by default, with the fixes above, and the runs its §8 item 7 names
(three seeds from scratch, 12M steps, Colab L4); 3d takes its proposed
weight, 0.30 (§0 and §3.4). No recorded decision adopts or supersedes the
note's other proposals, such as §8's reward plumbing (item 4) and the
criteria's contents (item 6), except its recommended arm: §8 item 5
recommended L+T plus the band alone for a line-5 outcome; the arm decided
(2A) is L+T plus the stagger penalty plus the band, with the three stance4
runs as its single-difference control, because the overnight band test ran
on stagger parents and the stagger term prices the route the note calls its
main risk (§5's last row). The prototype is the scratch commit `f94452ad` on
`c1e146a8`, never pushed: an opt-in `[env]` option of the T. rex env, off by
default, that charges a foot whose whole-foot CoP lies outside [−0.6, +0.4]
span half-lengths. The band's code, criteria, judge and Colab steps are not
part of this change: they are being built separately and are not committed
anywhere public.

### Added on 2026-10-22 (decision D-D40's outcome)

**`stance5/`: the stance5 screen of the per-foot CoP band, pre-registered
(decided 2026-10-20 as D-D40; criteria fixed on 2026-10-21 before any training
step; launched that day on Colab L4 runtimes).** Three runs of arm LTSB,
stance4's arm LTS plus the per-foot centre-of-pressure band
(`stance_cop_band_weight` 0.30), from scratch on physics r9, 12M steps each:
`stance5_LTSB_s52`, `stance5_LTSB_s60` and `stance5_LTSB_s42`, from the branch
`claude/screen-trex-r9-armlts-copband` at `cf80ecb1` (stance4's `c1e146a8` plus
the five band commits `f94452ad`, `3a1d5de3`, `4819eb6c`, `a15494ed` and
`99e205bb` and the config commit `cf80ecb1`), pushed on 2026-10-21 and never
merged, with no pull request. The control is the three stance4 runs as already
judged, not retrained.

- `criteria.txt`: what is trained (the term, and what it costs and moves,
  measured on the statue), how it is judged (K1–K4 exactly as stance4's and
  stance3's), the control and its caveats, the credit rule, the drift-or-split
  rule, the decision table, when the decision is read and the pre-registered
  readouts, written before any training step (sha256 `3b28963f…`). It was
  re-fixed after its review on 2026-10-21, before any training step; the first
  text (`c2809201…`) stays in the session's scratch. Its `SP/` and `T/` paths
  name the session's scratch.
- `criteria_notes.md`: where each of the maintainer's decisions of 2026-10-20
  lives in the criteria, and the review's outcome.
- `colab_steps.txt`: how the runs are launched and resumed, as sent; it holds
  the criteria's full text, byte for byte.
- `judge/judge_stance5.py` (judges one checkpoint against its branch clone,
  binding plant, task, symmetry record and commit, and records the band
  readouts), `judge/score_stance5.py` (scores K1–K4 with `score_stance3.py`'s
  own code, loaded unchanged, then the credit rule, the drift-or-split class and
  the decision line, and the report-only readouts) and
  `judge/test_score_stance5.py` (101 test cases).

*Added 2026-10-22:* the band's criteria, judge and Colab steps are now here, and
its code is on the screen branch above, pushed at `cf80ecb1`, whose history
holds the prototype `f94452ad`. This supersedes "being built separately and are
not committed anywhere public" and "never pushed" in the entries of 2026-10-20
(`stance4/night2/` and `stance4/copband/design.md`), which are left as written.

**`stance5/results/`: the stance5 screen judged (2026-10-22, final), with
report-only readouts.** Outcome: Run 1 `stance5_LTSB_s52` **holds** (K1–K4); Run
2 `stance5_LTSB_s60` fails K3 only (3 of 20 joint-space noise falls at the
final, against a bar of 2); Run 3 `stance5_LTSB_s42` fails K1 and K3 (mostly the
actuator-saturation rail, and 13 of 20 joint-space noise falls at the final).
Fore-aft holds on s52 and s60. H = 1 and K2p = 3; the band arm passes K3 on 1 of
3 runs against the control's 2 of 3, so the credit rule's (b) fails too, and the
decision table's line is **3**, final: not adopted, "replicate on 2 more seeds
before any commit". Under the parent rule s52's final ties stance3 Run 2's final
(report only). Measured, all three runs recorded a Colab L4 GPU runtime (the
`gpu` block of each run's `stage_config.json`). Its
[README](stance5/results/README.md) indexes the folder:

- `run_log.txt` (the stance5 log, ids removed, from #633's opening to the
  maintainer's decisions of 2026-10-22 on the replication), `scores/`
  (`score_final.txt` and `.json`, the verdict of record), `checkpoints.csv` (one
  row per judge file, 36, with the `checkpoints_csv.py` that recomputes it with
  the frozen scorer's own counts), `extras.txt` and `extras.py` (the fresh, tail
  and common-noise panels on each final, report only, measured for this record),
  `notebook_gates/` (the notebook's own six-heading gate on each run's
  `robust_best`, trimmed: all three PASS; report only) and `certificates/` (the
  stage's six-heading certificate rolled offline on the s52 and s60 finals,
  trimmed: both PASS; report only, not gate evidence).
- `panels_summary.csv` and `panels_summary.py`: the diagnostic panels
  (mirrored-leg spawns, the prey straight ahead, both) at every J checkpoint of
  the three runs, with the band readouts and the ankle asymmetry, report only.
- `saturation/`: the saturation watch's files for s42, report only: the ledger's
  alarm at 3,670,016, and the actuator named by rollouts of the 4,194,304
  checkpoint, `tail_3_pitch_act` at +1 (tail raised), unchanged at 5,242,880 and
  6,291,456.

### Added on 2026-10-22, later (decision D-D40's 5a)

**`stance4/satdiag/`: the offline diagnosis of stance4 s42's actuator saturation
(2026-10-20 to 2026-10-21, report only).** Deterministic rollouts of stance4
s42's checkpoints, with stance4 s52's and s60's and stance3 Run 2's beside them,
through the judge's own loading path; the five runs' eval ledgers; a
counterfactual; and eight verifiers. Measured: the saturating actuator is
`tail_3_pitch_act`, commanded at +1 (tail tip up), the actuator and sign of
stance3 Run 2; the command is pinned, not the joint (0.216 rad below its stop)
or the motor (0.26 of its force limit), so the ledger's force columns cannot see
it; the onset lies between the 1,048,576 checkpoint (clean) and 2,097,152 (17 of
20 rolled episodes flagged), the ledger flags from 1,835,008 and the alarm comes
at 2,228,224; holding tail_3 at 0 gains +85.9 (at 2,097,152) and +73.1 (at the
final) per 800-step window, deterministically; why training keeps the habit is
not measured. It lists candidate fixes (T1, T2, P1, R1, R2, Y1, N0), none acted
on. None of it enters a verdict.

- `report.md`: the diagnosis as written on 2026-10-21, each line labelled
  measured or inferred, with its verification. Its `O` is
  `scratchpad/foot/stance4/satdiag`, and its Sources section names files that
  stay there (the rollouts, the counterfactual and the verifiers' folders and
  scripts; about 14 MB in all).
- `rollout/tables/summary_tables.md`, `checkpoint_table.txt`,
  `actuator_matrix.md` and `onset_within_episode.txt`: the rollout tables the
  report cites (per checkpoint, the flagged episodes and each actuator's share
  of window steps at 0.9 and 0.99; the window-mean command per actuator; the
  ramp within an episode).
- `ledger/timeline.csv`: the eval-ledger timeline, one row per evaluation of
  s42, s52 and s60 and stance3 Runs 2 and 3 (saturated, clean and full-horizon
  episodes, returns, the centre of mass and the joints, and the training-episode
  columns).

### Added on 2026-10-22, later (decision D-D36's rebuilt pilot 2)

**`pilot2/` (three files added): walking pilot 2 rebuilt for its new parent
(2026-10-21; not yet trained, no result).** Decision D-D36 as amended on
2026-10-20 (7a, option i): pilot 2 warm-starts from stance3 Run 2's final
(`stance3_armLT_s60` at 12,001,280 steps; `stage1_final.zip` `cc21604a…`, its
normaliser `894e86f4…`) instead of r9 arm L seed 60 at 6M, which stays pilot 1's
parent. The new parent is committed on `claude/pilot-walk-r9-b` in three
pilot-only commits on `ba464855` (`6a13a303`, `8fc76eec` and `8c5478a6`; only
`pilot_checkpoints/` changes), pushed on 2026-10-21 at 19:49 UTC with the
maintainer's OK, a fast-forward from `ba464855` to `8c5478a6`; nothing more is
pushed to the branch until all three seeds are judged.

- `criteria_2026-10-21.txt`: the criteria as amended on 2026-10-21, before any
  pilot-2 training step (pilot 2 was never launched): the parent, the branch
  head, the parent's numbers, the readings reworded (pilot 2 now differs from
  pilot 1 in the parent and the noise reset, so no outcome is attributed to
  either alone), the hold on the branch, and the pre-launch checks re-run on
  `8fc76eec` and on `8c5478a6`. No outcome definition, threshold, guard, label,
  seed, budget or tool changes. It cites the 2026-10-18 text, `criteria.txt`
  here (sha256 `a240ff5d…`), as its pre-amendment copy.
- `colab_steps_2026-10-21.txt`: the launch steps for the rebuilt pilot, with the
  push recorded.
- `rebuild_1021_summary.md`: the rebuild's summary, written at 19:16 UTC (its
  file time), before the push (so it lists the push as open): the commits, the
  parent's action std
  (mean 0.064, knees 0.009, max 0.125; pilot 1's parent 0.095, 0.018 and 0.164),
  the criteria amendment, and the pre-launch checks (the notebook smoke; the
  judge accepting the run and refusing pilot 1's parent; 29 warm-start tests;
  the full suites on `8fc76eec`, 5,937 passed).

### Added on 2026-10-24 (decision D-D41's outcome)

**`stance5/replication/`: the stance5 replication of the per-foot CoP band,
pre-registered (decided 2026-10-22 as D-D41; criteria fixed that day before any
training step; launched that day on Colab L4 runtimes).** Two runs of arm LTSB,
unchanged, from scratch on physics r9, 12M steps each: `stance5_LTSB_s70` (Run
4) and `stance5_LTSB_s80` (Run 5), from the branch
`claude/screen-trex-r9-armlts-copband` at `cf80ecb1`, the commit the three
stance5 runs trained, never merged and with no pull request. The three stance5
runs count as recorded and are not re-judged; the stance4 control enters no
decision.

- `criteria.txt`: why it is run, what is trained (as stance5's), the stance5
  runs and the control (not retrained), how it is judged (the judge, the new
  seeds' joint-noise sigma and the scorer), the run handling of 2026-10-22, K1–K4
  as stance5's, the decision table of three lines (adopt only if both new runs
  hold), when the decision is read, the pre-registered readouts, what is
  measured and inferred, and the known limits, written before any training step
  (sha256 `a38536de…`). It was made final at 17:46 UTC on 2026-10-22 (sha256
  `cbf98ba7…`) and re-fixed at 18:00 UTC for the run log's time correction,
  before any training step; the earlier texts are not copied. Its `SP/` paths
  name the session's scratch.
- `colab_steps.txt`: how the runs are launched and resumed, as sent at 18:00
  UTC; it holds the criteria's full text.
- `sigma_s70_s80.json`: the joint-space noise sigma for seeds 70 and 80, one
  vector, the element-wise mean of the frozen `control_sigma` of seeds 42, 52
  and 60 (mean 0.160713), with its computation.
- `judge/judge_stance5r.py` (stance5's judge with `--seed` also taking 70 and
  80, whose sigma it reads from the sidecar, the sigma file recorded, and the
  schema `stance5r-judge/v1`; `judge/judge_diff.txt` is the whole diff),
  `judge/score_stance5r.py` (loads `score_stance5.py` and through it
  `score_stance3.py` unchanged, checks the stance5 runs' recorded verdicts and
  the control's, and reads the replication's decision line) and
  `judge/test_score_stance5r.py`.

**`stance5/replication/results/`: the replication judged (2026-10-24, final),
with report-only readouts.** Outcome: Run 4 `stance5_LTSB_s70` **holds**
(K1–K4); Run 5 `stance5_LTSB_s80` fails K1 only (40, 4, 14, 26, 39 and 38 of 40
clean, the three below 33 on the actuator-saturation rail). H = 2 of 5 (s52 and
s70), so the decision table's line is **2**, final: "Exactly one of s70 and s80
HOLDS (H = 2 of 5): not adopted." The maintainer chooses the next round from
stance5 line 4's menu (the tail saturation habit, the dose, the leg
exploration-noise floor). Under the parent rule s52's final still ties stance3
Run 2's final (report only). Measured, both runs recorded a Colab L4 GPU
runtime (the `gpu` block of each run's `stage_config.json`). Its
[README](stance5/replication/results/README.md) indexes the folder:

- `run_log.txt` (the replication's log, ids removed, from the package made
  final at 17:46 UTC on 2026-10-22 to the entry timed 15:27 UTC on 2026-10-24),
  `scores/` (`score_final.txt` and `score_final.json.gz`, the verdict of
  record), `notebook_gates/` (the notebook's own six-heading gate on each run's
  `robust_best`, trimmed: both PASS; report only) and `certificates/` (the
  stage's six-heading certificate rolled offline on the s70 and s80 finals,
  trimmed: 40/40 and 38/40, both PASS; report only, not gate evidence).
- `saturation/`: the saturation watch's files for s80, report only: the alarm
  at 4,456,448 (first flagged evaluation 4,325,376), and the actuator named by
  rollouts of the 4,718,592, 5,242,880 and 6,291,456 checkpoints,
  `tail_3_pitch_act` at +1 (tail raised).

## How to re-check each verdict

Commands as run on 2026-10-17 from a copy of this directory (Python 3.13;
`PYTHONDONTWRITEBYTECODE=1`). None needs the repository except where stated.
The three r10 rows needed the session's scratch tree and are not reproducible
from the repository alone: the original `frozen_params.json` (sha256
`58b2c8fb…`, which `judge_r10.py` and `score_r10.py` assert) and the r9 judge
files are not kept, so `verify_recompute.json` and `score.json` are the
retained records.

*Added 2026-10-18:* the rows below the first "The files themselves" row were
added that day; those marked 2026-10-18 were run that day on the committed
copies.

| Check | Command | Result |
|---|---|---|
| The symmetry screen's outcome from its stored per-arm records | in `symmetry/verdict_scripts/`: `python3 -c "import json, decide_final as D; P = json.load(open('../frozen_params.json')); V = json.load(open('../verdict.json')); print(D.outcome(V['arms']['L'], V['arms']['C'], P))"` | `O9`, no PR arm, no 12M arms |
| The symmetry decision table | `python3 test_decide_final.py ../frozen_params.json` | `"ok": true`, every combination mapped |
| The r10 verdict from the per-episode rows (session scratch only) | `python3 r10_screen/verify_recompute.py <judge dir> out.json` on the nine judge JSONs (not copied; `judge_r10.py` regenerates them from the Drive checkpoints in `fetch_record.json` on a clone of `claude/screen-trex-r10-arm-l` at `026eb833`, but only with the original `frozen_params.json`) | identical to `verify_recompute.json`: K1-K4 all FAIL |
| The r10 score (session scratch only) | `python3 r10_screen/score_r10.py <judge dir> --json score.json`, with the r9 judge files and the original `frozen_params.json` under `scratchpad/foot/...` | identical to `score.json` apart from the judge directory's path |
| The r10 scorer at every bar (session scratch only) | `python3 r10_screen/test_score_r10.py` (reads the original `frozen_params.json` at `scratchpad/foot/symmetry/design/`) | 33/33 |
| The pilot rule | in `pilot/`: `python3 test_pilot_score.py` | 203/203 (probe recomputation skipped) |
| The pilot rule, with every fixture row recomputed by the frozen probe | `python3 test_pilot_score.py <tree>` with `<tree>` a checkout of `claude/pilot-walk-r9` at `eaeef50f` and numpy, MuJoCo and the repository's dependencies installed | 254/254 |
| The parent rule and the judge's bindings | `python3 test_pilot_parent.py` | 40/40 |
| The files themselves | `sha256sum -c SHA256SUMS` | all 70 files OK |
| Pilot 1's outcome from its stored panels (2026-10-18) | copy `pilot1_judging/OUT` to `<work>`, `gunzip <work>/s*/*/probe.json.gz`, then `python3 pilot/pilot_score.py <work> --seeds 42,52,60 --json score.json` | `PILOT OUTCOME: STILL (3 of 3 launched seeds)`; `score.json` has the original's sha256 `86d7350e…` (`OUT/score.json` is it plus a final newline), and the printout is `judge_logs/score.log.txt` apart from the directory it names |
| Pilot 1's panels themselves (Drive checkpoints) | `pilot/pilot_judge.py` on a clone of `claude/pilot-walk-r9` at `eaeef50f`, with the checkpoints from the Drive folders in `pilot1_judging/run_log.txt` (sha256 in each `binding.json`) | not re-run on 2026-10-18 |
| The ledger's 28 rows against their evidence, and each certificate's LCB against its counts (2026-10-18; needs the repository) | from the repository root, the block below | `28 rows agree with their evidence` |
| A certificate's verdict from its per-episode rows (session scratch only) | `python3 offline_certificates/scripts/rederive.py <dir>` on `main` `2e734e29`, `<dir>` holding the certificate folders with their `stance_heading_panel_selected.csv` (not copied; sha256 at the end of SOURCES.txt) | as `rederive_from_csv.json` here and in `battery/fresh_seeds/`: all 11 and all 8 agree with their reports |
| The certificates themselves (session scratch only) | `offline_certificates/scripts/run_cert.py` on the scratch-harness checkpoints (not on Drive; sha256 in each `run_summary.json`) on `main` `2e734e29` | not re-run on 2026-10-18 |
| The battery's panel-pass illustration (2026-10-18) | `python3 battery/report/panel_pass_prob.py battery/fresh_seeds/fresh_seed_summary.json` (writes `panel_pass_prob.json` in the working directory) | the source's bytes (sha256 `09d0be6b…`; the copy adds a final newline) |
| The push report from its summary (2026-10-18) | `gunzip` a copy of `battery/push/push_summary.json.gz` into `<dir>`, then `python3 battery/push/scripts/make_report.py <dir>` | `push_report.txt`, apart from the summary path it names and one trailing blank line |
| The shared-fall report from its summary (2026-10-18) | in a directory holding a copy of `battery/shared_fall/shared_fall_summary.json` at `scratchpad/foot/battery/shared_fall/`: `python3 <this directory>/battery/shared_fall/scripts/make_report.py` | identical to `shared_fall_report.txt` |
| The audit's geometry summary (2026-10-18) | `python3 audit/geometry/scripts/trex_geom_summary.py audit/geometry/geometry_audit.json out.txt` | the source's bytes (sha256 `1632ed48…`; the copy drops one trailing space) |
| The rest of `battery/` and `audit/` (session scratch only) | their scripts read per-episode panels, traces, MuJoCo rolls on `main` `2e734e29` and downloaded papers, none of which is kept here | not re-run on 2026-10-18 |
| The files themselves (2026-10-18) | `sha256sum -c SHA256SUMS` | all 285 files OK |
| The stance3 scorer at every bar (2026-10-18; session scratch only) | from a directory holding the session's scratch as `scratchpad/` (the scorer asserts the original `frozen_params.json`, sha256 `58b2c8fb…`, and the criteria's `1ce910dc…` there): `python3 -I <this directory>/stance3/judge/test_score_stance3.py` | 80/80 cases as expected |
| The stance3 criteria (2026-10-18) | `sha256sum stance3/criteria.txt` | `1ce910dcd6e2ff7b41a4f592b9f34b9a6c9cb398c523e88f0ebd19af754f18fa` |
| The files themselves (2026-10-18, with `stance3/`) | `sha256sum -c SHA256SUMS` | all 291 files OK |
| The stance3 verdicts, decision line and best final checkpoint (2026-10-19; session scratch only) | from a directory holding the session's scratch as `scratchpad/`: `python3 -I <this directory>/stance3/judge/score_stance3.py scratchpad/foot/stance3/judged --runs-root scratchpad/foot/stance3/runs --json score.json` | identical to `stance3/results/scores/score_final2.txt` and `.json` apart from the paths they name |
| The stance3 scorer at every bar (2026-10-19; session scratch only) | as on 2026-10-18 | 80/80 cases as expected |
| `stance3/results/checkpoints.csv` (2026-10-19; session scratch only) | `python3 -I stance3/results/checkpoints_csv.py <judged dir> out.csv --check <score_final2.json>` | 28 cells agree with the scorer; `out.csv` identical |
| The files themselves (2026-10-19, with `stance3/results/`) | `sha256sum -c SHA256SUMS` | all 319 files OK |
| The files themselves (2026-10-19, later, with `stance3/night/`) | `sha256sum -c SHA256SUMS` | all 321 files OK |
| The stance4 criteria (2026-10-19) | `sha256sum stance4/criteria.txt` | `978a93da8da469c1acde7b9f7261d9236e4799eb2111b3af0f0cbeaa153f0f38` |
| The files themselves (2026-10-19, later, with `stance4/`) | `sha256sum -c SHA256SUMS` | all 322 files OK |
| The stance4 verdicts, decision line and parent inputs (2026-10-20; session scratch only) | in the session's scratch: `python3 -I -B scratchpad/foot/stance4/judge/score_stance4.py --runs-ended --json score.json` | identical to `stance4/results/scores/score_final.txt` and `.json` apart from the JSON's path and s52's two selection cells, judged after the record |
| `stance4/results/checkpoints.csv` (2026-10-20; session scratch only) | `python3 -I -B stance4/results/checkpoints_csv.py <judged dir> out.csv --check <score_final.json>` | 40 cells agree with the scorer; `out.csv` identical |
| `stance4/results/panels_summary.csv` (2026-10-20; session scratch only) | `python3 -I -B stance4/results/panels_summary.py <panels dir> out.csv --judged <stance4 judged dir> --control-judged <stance3 judged dir>` | `out.csv` identical; the model sha256 and the standard K2a and K2b equal the judge files' |
| `stance4/results/extras.txt` (2026-10-20; session scratch only) | from a directory holding the session's scratch as `scratchpad/`: `python3 -I -B <this directory>/stance4/results/extras.py` | identical to `stance4/results/extras.txt` |
| The notebook gates' checkpoints (2026-10-20, `stance4/results/notebook_gates/`) | compare each `gate_verdict.json`'s `checkpoint_sha256` with the `robust_best` row of `stance4/results/checkpoints.csv` | equal for all three |
| The files themselves (2026-10-20, with `stance4/results/`, `night/`, `night2/` and `copband/`) | `sha256sum -c SHA256SUMS` | all 344 files OK |
| The stance5 verdicts, decision line and parent comparison (2026-10-22; session scratch only) | from a directory holding the session's scratch as `scratchpad/`: `python3 -I -B scratchpad/foot/stance5/judge/score_stance5.py --runs-ended --json score.json` | identical to the scratch's `scores/score_final.txt` and `.json` (the copies in `stance5/results/scores/` differ from those only by the prefix change) apart from the JSON's path and s60's two selection cells, judged after the record; the committed copy of the scorer differs as `stance5/results/README.md` says |
| The stance5 scorer at every bar (2026-10-22; session scratch only) | in `scratchpad/foot/stance5/judge/`: `python3 -m pytest -p no:cacheprovider test_score_stance5.py` | 101 passed |
| `stance5/results/checkpoints.csv` (2026-10-22; session scratch only) | `python3 -I -B stance5/results/checkpoints_csv.py <judged dir> out.csv --check <score_final.json>` | 25 cells agree with the scorer; `out.csv` identical |
| `stance5/results/panels_summary.csv` (2026-10-22; session scratch only) | `python3 -I -B stance5/results/panels_summary.py <panels dir> out.csv --judged <stance5 judged dir>` | `out.csv` identical; the model sha256 and the standard K2a and K2b equal the judge files' |
| `stance5/results/extras.txt` (2026-10-22; session scratch only) | from a directory holding the session's scratch as `scratchpad/`: `python3 -I -B <this directory>/stance5/results/extras.py` | identical to `stance5/results/extras.txt` |
| The notebook gates' and certificates' checkpoints (2026-10-22, `stance5/results/`) | compare each `notebook_gates/<run id>/gate_verdict.json`'s `checkpoint_sha256` with the `robust_best` row of `stance5/results/checkpoints.csv`, and each `certificates/<run id>/run_summary.json`'s `model_sha256` with the final row | equal for all three gates and both certificates |
| The stance5 criteria (2026-10-22) | `sha256sum stance5/criteria.txt` | `3b28963f644ffb765e666a8cf6ada5d31ab8a2769dd1a77ff53cbc1e0cf3cf16` |
| The files themselves (2026-10-22, with `stance5/`) | `sha256sum -c SHA256SUMS` | all 372 files OK |
| The files themselves (2026-10-22, later, with `stance4/satdiag/`) | `sha256sum -c SHA256SUMS` | all 378 files OK |
| The amended pilot-2 criteria against their source (2026-10-22; session scratch only) | `diff scratchpad/foot/pilot2/criteria.txt pilot2/criteria_2026-10-21.txt` | line 149 only (the redaction) |
| The files themselves (2026-10-22, later, with the `pilot2/` files) | `sha256sum -c SHA256SUMS` | all 381 files OK |
| The replication's verdicts, decision line and parent comparison (2026-10-24; session scratch only, restored from the handoff branch) | in the session's scratch: `python3 -I scratchpad/foot/stance5/replication/judge/score_stance5r.py --runs-ended --json score.json` | the same verdicts, cells, H and decision line (2) as `stance5/replication/results/scores/`; the differences the handoff's missing files cause are in `stance5/replication/results/README.md` |
| The replication scorer's tests (2026-10-24; session scratch only, restored) | in `scratchpad/foot/stance5/replication/judge/`: `python3 -m pytest -p no:cacheprovider test_score_stance5r.py` | 42 passed, 2 skipped, 1 failed, as `stance5/replication/results/README.md` explains |
| The replication certificates' checkpoints (2026-10-24; session scratch only) | compare each `stance5/replication/results/certificates/<run id>/run_summary.json`'s `model_sha256` with the scratch's `judged/<run id>/judge_final.json` | equal for both |
| The replication criteria (2026-10-24) | `sha256sum stance5/replication/criteria.txt` | `a38536de152e735947fd5322960f9e75b6520b59a5526a414afc0d0c4a0f3ef2` |
| The files themselves (2026-10-24, with `stance5/replication/`) | `sha256sum -c SHA256SUMS` | all 401 files OK |

`test_pilot_score.py` prints sample "Walking pilot score" blocks while it runs;
they score synthetic panels, not the pilot.

The ledger check (from the repository root, with its dependencies installed):

```python
import json
from environments.shared.curriculum.recovery_gate import binomial_lcb
from environments.shared.reporting import load_offline_judgements

rows = load_offline_judgements()
for r in rows:
    ev = json.load(open(r["evidence"]))
    if r["gate_kind"] == "stance_quality_heading/v1":
        assert abs(binomial_lcb(ev["n_clean_seeds"], ev["n_seeds"]) - ev["clean_seed_lcb"]) < 1e-12
        got = (ev["passed"], ev["n_clean_seeds"], ev["n_seeds"], round(ev["clean_seed_lcb"], 4),
               "sha256:" + ev["model_sha256"])
        want = (r["verdict"] == "PASS", r["passing_count"], r["panel_size"], r["lower_bound"],
                r["checkpoint_sha256"])
    else:  # walking pilot 1: the outcome in score.json, the checkpoint in binding.json
        seed, step = r["run_path"][-2:], str(r["checkpoint_step"])
        cell = ev["seeds"][seed][step]
        binding = json.load(open(r["evidence"].replace("score.json", f"s{seed}/{step}/binding.json")))
        got = (cell["outcome"], cell["counts"]["WALK"], sum(cell["counts"].values()),
               binding["checkpoint_sha256"])
        want = (r["verdict"], r["passing_count"], r["panel_size"], r["checkpoint_sha256"])
    assert got == want, (r["evidence"], got, want)
print(len(rows), "rows agree with their evidence")
```

It checks what the rows say against the files here; it does not re-run a
judgement.

## Lint

Under the repository's ruff configuration (ruff 0.16.9, the version CI and
pre-commit pin; 0.16.8 gives the same), `ruff check` passes on
14 of the 16 scripts here. It reports 11 findings in
`r10_screen/verify_recompute.py` (a verifier's quick script: a multi-module
import, an unused `os`, statements joined by `;` and `:`) and 2 in
`symmetry/verdict_scripts/recount_final.py` (an unused `math`, a variable named
`I`). `ruff format --check` would reformat all 16. They are left as they ran,
so that each stays the file that decided its verdict (and the verdict scripts
keep the sha256 the screen recorded).

The 43 scripts added on 2026-10-18 (ruff 0.16.8, the same configuration) are
left as they ran too, apart from the prefix and the two lines of trailing
whitespace SOURCES.txt records. `ruff check` passes on 12 of them and reports
193 findings in the other 31, all of them quick-script style: statements
joined by `;` (67) or `:` (19), unsorted imports (31), several modules on one
import line (23), variables named `l` or `O` (16), imports below
code (14), unused imports (14), f-strings without placeholders (4), stray
semicolons (2), and one each of a bare `except`, a lambda assigned to a name
and an unused variable. Most are in the verifiers' scripts
(`battery/verify_skeptic/`, 91). `ruff format --check` would reformat all 43.

The three `stance3/judge/` scripts added later on 2026-10-18 pass `ruff check`
under the same configuration (ruff 0.16.9); `ruff format --check` would
reformat all three. They are left as they ran.

Of the two scripts added on 2026-10-19 in `stance3/results/`,
`checkpoints_csv.py`, written for the record, passes `ruff check` and
`ruff format --check` under the same configuration (ruff 0.16.8);
`certificates/run_cert_clone.py` is left as it ran, with 8 findings (several
modules on one import line, unsorted imports, imports below code), and
`ruff format --check` would reformat it.

Of the three scripts added on 2026-10-20 in `stance4/results/`,
`checkpoints_csv.py` and `panels_summary.py`, written for the record, pass
`ruff check` and `ruff format --check` under the same configuration (ruff
0.16.8); `extras.py` is left as it ran, with 3 findings (several modules on
one import line, unsorted imports, two statements joined by `;`), and
`ruff format --check` would reformat it.

*Added 2026-10-22:* of the six scripts added in `stance5/`, the three in
`stance5/results/`, written for the record (`checkpoints_csv.py` and
`panels_summary.py` adapted from stance4's), pass `ruff check` and `ruff format
--check` under the same configuration (ruff 0.16.9). The three in
`stance5/judge/` are left as they ran: `ruff check` reports 4 findings in
`score_stance5.py` (an unused `math` and three variables named `l`) and none in
the other two, and `ruff format --check` would reformat all three. All six parse
on Python 3.11, 3.12 and 3.13.

*Added 2026-10-24:* the three scripts added in `stance5/replication/judge/` are
left as they ran: `ruff check` (ruff 0.16.9, the same configuration) reports 1
finding in `test_score_stance5r.py` (an f-string without placeholders) and none
in the other two, and `ruff format --check` would reformat all three. All three
parse on Python 3.11, 3.12 and 3.13.
